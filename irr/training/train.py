# irr/training/train.py
from __future__ import annotations

import dataclasses
import math
import warnings
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from irr.data.datamodule import IrrDataModule
from irr.configs import TrainConfig
from irr.models.mlp_classifier import IrrMLPClassifier, ModelConfig


def pick_accel_and_precision() -> tuple[str, str]:
    """Choose accelerator + precision based on available backends."""
    if torch.cuda.is_available():
        return "gpu", "16-mixed"   # CUDA AMP is mature
    # Apple Silicon (MPS) — mixed precision isn't stable; stick to fp32
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", "32-true"
    return "cpu", "32-true"


def run_train(cfg: TrainConfig, datamodule: IrrDataModule | None = None) -> dict:
    # ----- Data -----
    dm = datamodule or IrrDataModule(
        data_glob=cfg.data_glob,
        batch_size=cfg.batch_size,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        group_col=getattr(cfg, "group_col", "h3_r7"),
        include_states=getattr(cfg, "include_states", None),
    )
    dm.setup()
    dm.assert_no_group_leakage()  # checks cfg.group_col by default
    print(f"[Split] group_col={getattr(cfg, 'group_col', None)!r}")

    # ----- Class balance on TRAIN ONLY -----
    y_train = dm.y_train
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    if pos == 0:
        warnings.warn("No positive examples in the TRAIN split; pos_weight disabled.")
        pos_weight_tensor = None
        pi = 0.0
    else:
        ratio = neg / max(pos, 1)
        # BCEWithLogitsLoss expects a 1D tensor of size [1] for binary pos_weight
        pos_weight_tensor = torch.tensor([ratio], dtype=torch.float32)
        pi = pos / (pos + neg)

    # ----- Model config -----
    model_cfg = getattr(cfg, "model", None)
    if model_cfg is None:
        model_cfg = ModelConfig(
            in_dim=dm.X_train.size(1),
            hidden=getattr(cfg, "hidden", 256),
            depth=getattr(cfg, "depth", 2),
            dropout=getattr(cfg, "dropout", 0.10),
            act=getattr(cfg, "act", "silu"),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            standardize=getattr(cfg, "standardize", False),
        )
    else:
        # ensure in_dim matches data
        model_cfg.in_dim = dm.X_train.size(1)

    # ----- Model -----
    model = IrrMLPClassifier(model_cfg, pos_weight=pos_weight_tensor)

    # If your embeddings are already unit-length, keep standardizer as no-op:
    model.set_standardizer(torch.zeros(model_cfg.in_dim), torch.ones(model_cfg.in_dim))

    # Initialize final bias to match TRAIN positive prior (helps early calibration)
    if 0.0 < pi < 1.0:
        with torch.no_grad():
            model.final_linear.bias.fill_(math.log(pi / (1.0 - pi)))

    # ----- Logging -----
    csv_logger = CSVLogger(save_dir="outputs/logs", name="mlp_classifier")
    tb_logger = TensorBoardLogger(save_dir="outputs/logs", name="mlp_classifier_tb")

    # (Optional) log hyperparams into TB HParams tab
    try:
        hp = dataclasses.asdict(cfg)
    except Exception:
        # cfg may not be a dataclass; fall back to a minimal dict
        hp = {
            "batch_size": cfg.batch_size,
            "val_ratio": cfg.val_ratio,
            "seed": cfg.seed,
            "group_col": getattr(cfg, "group_col", None),
            "include_states": getattr(cfg, "include_states", None),
            "hidden": getattr(cfg, "hidden", None),
            "depth": getattr(cfg, "depth", None),
            "dropout": getattr(cfg, "dropout", None),
            "act": getattr(cfg, "act", None),
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
        }
    hp.update({
        "train_pos": pos,
        "train_neg": neg,
        "pos_weight": None if pos_weight_tensor is None else float(pos_weight_tensor[0]),
    })
    try:
        tb_logger.log_hyperparams(hp)
    except Exception:
        pass

    # ----- Callbacks -----
    monitor = getattr(cfg, "monitor", "val_auprc")
    patience = getattr(cfg, "patience", 10)
    min_delta = getattr(cfg, "min_delta", 1e-5)

    callbacks = [
        EarlyStopping(monitor=monitor, mode="max", patience=patience, min_delta=min_delta),
        ModelCheckpoint(monitor=monitor, mode="max", save_top_k=1, filename="best"),
    ]

    # ----- Trainer -----
    accelerator, precision = pick_accel_and_precision()
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=accelerator,
        devices="auto",
        precision=precision,
        logger=[tb_logger, csv_logger],
        callbacks=callbacks,
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(model, datamodule=dm)
    return {"log_dir": csv_logger.log_dir}
