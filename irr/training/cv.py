# irr/training/cv.py
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from irr.constants import FEATURES, LABEL_COL
from irr.configs import TrainConfig
from irr.data.datamodule import IrrDataModule
from irr.training.train import run_train


# helpers

def _normalize_states(states: Optional[Iterable[str]]) -> Optional[list[str]]:
    if states is None:
        return None
    return [str(s).strip().upper() for s in states]


def _compute_groups_with_datamodule(df: pd.DataFrame, group_col: Optional[str], seed: int) -> Optional[np.ndarray]:
    if not group_col or str(group_col).lower() == "none":
        return None
    dm_tmp = IrrDataModule(
        data_glob="",
        batch_size=1,
        val_ratio=0.2,
        seed=seed,
        group_col=group_col,
        include_states=None,
        debug=False,
    )
    return dm_tmp._make_groups(df)


def _iter_folds(y: np.ndarray, k: int, seed: int, groups: Optional[np.ndarray], shuffle: bool = True):
    n = len(y)
    X_dummy = np.arange(n)

    if groups is not None:
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=shuffle, random_state=seed)
        yield from sgkf.split(X_dummy, y, groups=groups)
        return

    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)
    yield from skf.split(X_dummy, y)


# main API 

def run_kfold(cfg: TrainConfig, k: int = 5, shuffle: bool = True):
    """
    Group-aware, label-stratified K-fold CV.

    - If cfg.group_col is set, folds are group-disjoint via StratifiedGroupKFold, otherwise StratifiedKFold.

    Returns
    -------
    folds_df : pd.DataFrame
        One row per fold with paths and key metrics.
    summary : dict
        mean/std for val_auprc and val_auroc across folds.
    """
    # Load once
    df = IrrDataModule.load_all_df(cfg.data_glob)

    # Optional state filter
    inc_states = _normalize_states(getattr(cfg, "include_states", None))
    if inc_states is not None:
        if "state" not in df.columns:
            raise ValueError("include_states provided but 'state' column not found.")
        df = df.copy()
        df["state"] = df["state"].astype(str).str.strip().str.upper()
        df = df[df["state"].isin(inc_states)].reset_index(drop=True)

    # Schema check
    missing = [c for c in FEATURES + [LABEL_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Targets nd groups
    y = df[LABEL_COL].astype(int).to_numpy()
    groups = _compute_groups_with_datamodule(df, getattr(cfg, "group_col", None), seed=cfg.seed)

    rows = []
    for fold_id, (tr_idx, va_idx) in enumerate(_iter_folds(y, k=k, seed=cfg.seed, groups=groups, shuffle=shuffle), start=1):
        print(f"\n=== Fold {fold_id}/{k} ===")

        # each split must contain both classes, sanity check
        y_tr, y_va = y[tr_idx], y[va_idx]
        if y_tr.min() == y_tr.max():
            raise RuntimeError(f"Fold {fold_id}: train split is single-class.")
        if y_va.min() == y_va.max():
            raise RuntimeError(f"Fold {fold_id}: val split is single-class.")

        # If grouped, assert no leakage
        if groups is not None:
            tr_g, va_g = set(groups[tr_idx]), set(groups[va_idx])
            overlap = tr_g & va_g
            print(f"[Fold {fold_id}] groups train={len(tr_g):,} val={len(va_g):,} overlap={len(overlap):,}")
            if overlap:
                raise AssertionError(f"Group leakage in fold {fold_id}: {len(overlap)} overlapping groups.")

        # Build per-fold DataModule using explicit indices
        dm = IrrDataModule(
            data_glob=cfg.data_glob,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            train_idx=tr_idx,
            val_idx=va_idx,
            group_col=None,
            include_states=inc_states,
        )

        fold_cfg = replace(cfg)
        result = run_train(fold_cfg, datamodule=dm)
        log_dir = Path(result["log_dir"])

        # Pull last-epoch metrics from CSV logger
        metrics_csv = log_dir / "metrics.csv"
        if metrics_csv.exists():
            m = pd.read_csv(metrics_csv)
            if "epoch" not in m.columns and "step" in m.columns:
                m["epoch"] = m["step"]
            m = m.sort_values("epoch").groupby("epoch").last()
            val_auprc  = float(m["val_auprc"].iloc[-1]) if "val_auprc" in m.columns else float("nan")
            val_auroc  = float(m["val_auroc"].iloc[-1]) if "val_auroc" in m.columns else float("nan")
            train_loss = float(m["train_loss"].iloc[-1]) if "train_loss" in m.columns else float("nan")
            val_loss   = float(m["val_loss"].iloc[-1])   if "val_loss"   in m.columns else float("nan")
        else:
            val_auprc = val_auroc = train_loss = val_loss = float("nan")

        rows.append({
            "fold": fold_id,
            "log_dir": str(log_dir),
            "val_auprc": val_auprc,
            "val_auroc": val_auroc,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

    folds_df = pd.DataFrame(rows).set_index("fold")
    summary = {
        "val_auprc_mean": float(folds_df["val_auprc"].mean()),
        "val_auprc_std":  float(folds_df["val_auprc"].std(ddof=1)),
        "val_auroc_mean": float(folds_df["val_auroc"].mean()),
        "val_auroc_std":  float(folds_df["val_auroc"].std(ddof=1)),
    }
    return folds_df, summary
