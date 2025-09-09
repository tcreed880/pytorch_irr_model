# irr/training/cv.py
from __future__ import annotations
from dataclasses import replace, asdict
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from pathlib import Path
import json

from irr.data.datamodule import IrrDataModule
from irr.training.train import run_train
from irr.constants import FEATURES

def run_kfold(cfg, k: int = 5, shuffle: bool = True):
    """Runs K-fold CV. Returns a DataFrame of fold metrics and aggregate stats dict."""
    df = IrrDataModule.load_all_df(cfg.data_glob)
    X = df[FEATURES].values
    y = df["label"].values.astype(int)

    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=cfg.seed)
    fold_rows = []

    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n=== Fold {fold_id}/{k} ===")

        dm = IrrDataModule(
            cfg.data_glob, batch_size=cfg.batch_size,
            val_ratio=cfg.val_ratio, seed=cfg.seed,
            train_idx=tr_idx, val_idx=va_idx
        )

        # tweak cfg to tag the fold (for logging)
        fold_cfg = replace(cfg)
        setattr(fold_cfg, "fold", fold_id)

        result = run_train(fold_cfg, datamodule=dm)  # returns dict with log_dir
        log_dir = Path(result["log_dir"])

        # Read metrics.csv and take last epoch's val metrics
        metrics_csv = log_dir / "metrics.csv"
        if metrics_csv.exists():
            m = pd.read_csv(metrics_csv)
            if "epoch" not in m.columns and "step" in m.columns:
                m["epoch"] = m["step"]
            m = m.sort_values(["epoch"]).groupby("epoch").last()
            val_auprc = m["val_auprc"].iloc[-1] if "val_auprc" in m.columns else float("nan")
            val_auroc = m["val_auroc"].iloc[-1] if "val_auroc" in m.columns else float("nan")
            train_loss = m["train_loss"].iloc[-1] if "train_loss" in m.columns else float("nan")
            val_loss   = m["val_loss"].iloc[-1] if "val_loss" in m.columns else float("nan")
        else:
            val_auprc = val_auroc = train_loss = val_loss = float("nan")

        fold_rows.append({
            "fold": fold_id,
            "log_dir": str(log_dir),
            "val_auprc": val_auprc,
            "val_auroc": val_auroc,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

    folds_df = pd.DataFrame(fold_rows).set_index("fold")
    summary = {
        "val_auprc_mean": float(folds_df["val_auprc"].mean()),
        "val_auprc_std":  float(folds_df["val_auprc"].std(ddof=1)),
        "val_auroc_mean": float(folds_df["val_auroc"].mean()),
        "val_auroc_std":  float(folds_df["val_auroc"].std(ddof=1)),
    }
    return folds_df, summary
