# irr/cli/optuna_tune.py
from __future__ import annotations

import argparse
import math
from functools import partial

import optuna
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
from irr.data.datamodule import IrrDataModule
from irr.models.mlp_classifier import IrrMLPClassifier, ModelConfig

class LightningOptunaPruner(Callback):
    def __init__(self, trial: optuna.Trial, monitor: str):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module) -> None:
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        # metric is a torch tensor; convert to float
        value = float(metric.detach().cpu().item())
        step = trainer.current_epoch
        self.trial.report(value, step)
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()    

# ---------------- device/precision ----------------

def pick_accel_and_precision() -> tuple[str, str]:
    if torch.cuda.is_available():
        return "gpu", "16-mixed"   # CUDA AMP
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", "32-true"    # MPS: stick to fp32
    return "cpu", "32-true"


# ---------------- data helpers ----------------

def make_datamodule(
    data_glob: str,
    batch_size: int,
    val_ratio: float,
    seed: int,
    group_col: str | None,
    include_states: list[str] | None,
) -> IrrDataModule:
    dm = IrrDataModule(
        data_glob=data_glob,
        batch_size=batch_size,
        val_ratio=val_ratio,
        seed=seed,
        group_col=group_col,
        include_states=include_states,
    )
    dm.setup(stage="fit")
    dm.assert_no_group_leakage()
    return dm


def compute_pos_weight_from_dm(dm: IrrDataModule) -> tuple[torch.Tensor | None, float]:
    y_train = dm.y_train
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    if pos == 0 or neg == 0:
        return None, 0.0
    ratio = neg / pos
    return torch.tensor([ratio], dtype=torch.float32), pos / (pos + neg)


# ---------------- model factory ----------------

def build_model(trial: optuna.Trial, in_dim: int, pos_weight: torch.Tensor | None, standardize: bool) -> IrrMLPClassifier:
    cfg = ModelConfig(
        in_dim=in_dim,
        # removed depth, hidden size, using defaults 2 and 256
        depth=2,
        hidden=256,
        dropout=trial.suggest_float("dropout", 0.01, 0.06),
        # removed silu and relu
        act="gelu",
        lr=trial.suggest_float("lr", 2e-4, 6e-4, log=True),
        weight_decay=trial.suggest_float("weight_decay", 3e-5, 2e-4, log=True),
        standardize=standardize,
    )
    model = IrrMLPClassifier(cfg, pos_weight=pos_weight)
    # keep standardizer no-op unless you flip --standardize
    with torch.no_grad():
        model.x_mean.zero_()
        model.x_std.fill_(1.0)
    return model


# ---------------- objective ----------------

def _monitor_and_mode(objective: str) -> tuple[str, str]:
    """Return (monitor_key, mode) given objective name."""
    obj = objective.lower()
    if obj == "auprc":
        return "val_auprc", "max"
    if obj == "bce":
        return "val_bce", "min"
    if obj == "bce_cal":
        return "val_bce_calibrated", "min"
    raise ValueError(f"Unknown objective: {objective}")

def objective(trial: optuna.Trial, args: argparse.Namespace, accel: str, prec: str) -> float:
    seed_everything(args.seed, workers=True)
    # using 1024 based on previous experiments
    batch_size = 1024
    #batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])

    dm = make_datamodule(
        data_glob=args.data_glob,
        batch_size=batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        group_col=(None if args.group_col and args.group_col.lower() == "none" else args.group_col),
        include_states=args.include_states,
    )

    pos_weight, pi = compute_pos_weight_from_dm(dm)
    model = build_model(trial, in_dim=dm.X_train.size(1), pos_weight=pos_weight, standardize=args.standardize)

    # prior-based bias init
    if 0.0 < pi < 1.0:
        with torch.no_grad():
            model.final_linear.bias.copy_(torch.tensor(math.log(pi / (1.0 - pi)), dtype=model.final_linear.bias.dtype))

    monitor_key, mode = _monitor_and_mode(args.objective)
    prune_cb = LightningOptunaPruner(trial, monitor=monitor_key)

    es = EarlyStopping(monitor=monitor_key, mode=mode, patience=args.patience, min_delta=1e-5, verbose=False)
    ckpt = ModelCheckpoint(monitor=monitor_key, mode=mode, save_top_k=1, filename="best", auto_insert_metric_name=False)
    logger = TensorBoardLogger(save_dir=args.log_dir, name=f"{args.study_name}/trial_{trial.number}")

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=accel,
        devices="auto",
        precision=prec,
        deterministic=True,
        logger=logger,
        callbacks=[es, ckpt, prune_cb],
        enable_progress_bar=False,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
    )

    trainer.fit(model, datamodule=dm)

    # save useful info on the trial
    trial.set_user_attr("best_ckpt", ckpt.best_model_path)
    trial.set_user_attr("log_dir", logger.log_dir)

    metric = trainer.callback_metrics.get(monitor_key)
    return float(metric.cpu().item()) if metric is not None else float("nan")


# ---------------- CLI ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna hyperparameter tuning for IrrMLPClassifier.")
    p.add_argument("--data-glob", required=True, help='e.g. "raw_data/*.csv"')
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=92)
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--study-name", type=str, default="mlp_optuna")
    p.add_argument("--storage", type=str, default=None, help='e.g. "sqlite:///outputs/optuna/optuna.db"')
    p.add_argument("--log-dir", type=str, default="outputs/optuna_tb")

    p.add_argument("--group-col", type=str, default="h3_r5",
                   help="Grouping for train/val split: 'county_fips', '.geo', or 'h3_r{res}'. Use 'none' for stratified.")
    p.add_argument("--include-states", nargs="*", default=None, help="Subset to these state codes.")
    p.add_argument("--standardize", action="store_true", help="Apply (x-mean)/std inside the model.")
    p.add_argument("--objective", type=str, choices=["auprc", "bce", "bce_cal"], default="auprc",
                   help="Tune for ranking (auprc) or calibration (bce / bce_cal).")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    accel, prec = pick_accel_and_precision()

    direction = "maximize" if args.objective == "auprc" else "minimize"
    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=10)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    study = optuna.create_study(
        direction=direction,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(partial(objective, args=args, accel=accel, prec=prec), n_trials=args.n_trials)

    best = study.best_trial
    print("\n=== BEST TRIAL ===")
    print("number:", best.number)
    print("value:", best.value)
    print("params:", best.params)
    print("best_ckpt:", best.user_attrs.get("best_ckpt"))
    print("log_dir:", best.user_attrs.get("log_dir"))


if __name__ == "__main__":
    main()
