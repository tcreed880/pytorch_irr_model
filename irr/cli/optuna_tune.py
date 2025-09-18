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

from irr.data.datamodule import IrrDataModule
from irr.models.mlp_classifier import IrrMLPClassifier, ModelConfig


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
    dm.assert_no_group_leakage()  # checks the configured group_col
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
        hidden=trial.suggest_categorical("hidden", [128, 256, 512]),
        depth=trial.suggest_int("depth", 1, 4),
        dropout=trial.suggest_float("dropout", 0.0, 0.5),
        act=trial.suggest_categorical("act", ["silu", "gelu", "relu"]),
        lr=trial.suggest_float("lr", 3e-5, 3e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        standardize=standardize,
    )
    model = IrrMLPClassifier(cfg, pos_weight=pos_weight)
    # If embeddings are unit-norm, keep standardizer a no-op:
    with torch.no_grad():
        model.x_mean.zero_()
        model.x_std.fill_(1.0)
    return model


# ---------------- objective ----------------

def objective(trial: optuna.Trial, args: argparse.Namespace, accel: str, prec: str) -> float:
    seed_everything(args.seed)

    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])

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

    # Bias init to prior improves early calibration
    if 0.0 < pi < 1.0:
        with torch.no_grad():
            model.final_linear.bias.fill_(math.log(pi / (1.0 - pi)))

    es = EarlyStopping(monitor="val_auprc", mode="max", patience=args.patience, min_delta=1e-5)
    ckpt = ModelCheckpoint(monitor="val_auprc", mode="max", save_top_k=1, filename="best")
    logger = TensorBoardLogger(save_dir=args.log_dir, name=f"{args.study_name}/trial_{trial.number}")

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=accel,
        devices="auto",
        precision=prec,
        gradient_clip_val=1.0,
        deterministic=True,
        logger=logger,
        callbacks=[es, ckpt],
        enable_progress_bar=False,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=dm)
    val_auprc = trainer.callback_metrics.get("val_auprc")
    return float(val_auprc.cpu().item()) if val_auprc is not None else float("nan")


# ---------------- CLI ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna hyperparameter tuning for IrrMLPClassifier.")
    p.add_argument("--data-glob", required=True, help='e.g. "raw_data/*cropland*.csv"')
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=88)
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--study-name", type=str, default="mlp_optuna")
    p.add_argument("--storage", type=str, default=None, help='e.g. "sqlite:///optuna.db" to persist trials')
    p.add_argument("--log-dir", type=str, default="outputs/optuna_tb")

    p.add_argument("--group-col", type=str, default="h3_r7",
                   help="Grouping for train/val split: 'county_fips', '.geo', or 'h3_r{res}'. Use 'none' for s


"""
Usage example:
poetry run python -m irr.cli.optuna_tune \
  --data-glob "raw_data/*training*.csv" \
  --val-ratio 0.2 \
  --seed 88 \
  --max-epochs 40 \
  --patience 7 \
  --n-trials 50 \
  --study-name mlp_h3r7_optuna \
  --storage "sqlite:///optuna.db" \
  --group-col h3_r7 \
  --include-states MT OR ID
"""