# irr/training/cv.py
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

import json
import numpy as np
import pandas as pd

from irr.constants import FEATURES, LABEL_COL
from irr.configs import TrainConfig
from irr.data.datamodule import IrrDataModule
from irr.training.train import run_train

# sklearn splitters
from sklearn.model_selection import StratifiedKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold  # sklearn >= 1.1
    _HAS_SGK = True
except Exception:
    _HAS_SGK = False
from sklearn.model_selection import GroupKFold

# optional h3 for spatial grouping
try:
    import h3
    _HAS_H3 = True
except Exception:
    _HAS_H3 = False


def _compute_groups(df: pd.DataFrame, group_col: Optional[str]) -> Optional[np.ndarray]:
    """Return group ids as ndarray[str] or None if no grouping."""
    if not group_col or str(group_col).lower() == "none":
        return None

    gc = str(group_col).lower()

    if gc.startswith("h3_r"):
        if ".geo" not in df.columns:
            raise ValueError("H3 grouping requires a '.geo' column.")
        if not _HAS_H3:
            raise ImportError("Install 'h3' to use H3 grouping.")
        res = int(gc.split("h3_r")[-1])

        def to_h3(s: str) -> str:
            c = json.loads(s)["coordinates"]
            lon, lat = float(c[0]), float(c[1])
            return h3.geo_to_h3(lat, lon, res)

        return df[".geo"].astype(str).apply(to_h3).astype(str).to_numpy()

    # direct column
    if gc in df.columns:
        return df[gc].astype(str).to_numpy()

    # allow literal ".geo"
    if gc == ".geo" and ".geo" in df.columns:
        return df[".geo"].astype(str).to_numpy()

    # unknown → no grouping
    return None


def _iter_folds(y: np.ndarray, k: int, seed: int, groups: Optional[np.ndarray]):
    """Yield (train_idx, val_idx) across K folds with group-aware fallback."""
    if groups is None:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for tr_idx, va_idx in skf.split(np.zeros_like(y), y):
            yield tr_idx, va_idx
        return

    # Prefer stratified + grouped if available; otherwise grouped only
    if _HAS_SGK:
        sgk = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        for tr_idx, va_idx in sgk.split(np.zeros_like(y), y, groups=groups):
            yield tr_idx, va_idx
    else:
        # GroupKFold has no shuffling and no stratification
        gkf = GroupKFold(n_splits=k)
        for tr_idx, va_idx in gkf.split(np.zeros_like(y), y, groups=groups):
            yield tr_idx, va_idx


def run_kfold(cfg: TrainConfig, k: int = 5, shuffle: bool = True):
    """
    Run K-fold CV using:
      - StratifiedKFold if group_col is None/'none'
      - StratifiedGroupKFold (if available) or GroupKFold when group_col is provided
    Respects cfg.include_states filter.
    Returns (folds_df, summary_dict).
    """
    # Load once
    df = IrrDataModule.load_all_df(cfg.data_glob)

    # Optional state filter
    if cfg.include_states:
        df = df[df["state"].isin(cfg.include_states)].reset_index(drop=True)

    # Sanity: required cols
    missing = [c for c in FEATURES + [LABEL_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURES].to_numpy(dtype=np.float32)  # only for length; run_train re-loads anyway
    y = df[LABEL_COL].astype(int).to_numpy()

    # Groups (e.g., 'h3_r7', 'county_fips', '.geo')
    groups = _compute_groups(df, cfg.group_col)

    fold_rows = []
    for fold_id, (tr_idx, va_idx) in enumerate(_iter_folds(y, k=k, seed=cfg.seed, groups=groups), start=1):
        print(f"\n=== Fold {fold_id}/{k} ===")
        # Defensive group leakage check (indices view)
        if groups is not None:
            tr_g, va_g = set(groups[tr_idx]), set(groups[va_idx])
            overlap = tr_g & va_g
            print(f"[Fold {fold_id}] groups train={len(tr_g):,} val={len(va_g):,} overlap={len(overlap):,}")
            if overlap:
                raise AssertionError(f"Group leakage in fold {fold_id}: {len(overlap)} overlapping groups.")

        # Build a per-fold DataModule with fixed indices
        dm = IrrDataModule(
            data_glob=cfg.data_glob,
            batch_size=cfg.batch_size,
            # val_ratio unused when indices provided
            seed=cfg.seed,
            train_idx=tr_idx,
            val_idx=va_idx,
            group_col=None,                 # indices are explicit; don't re-split
            include_states=cfg.include_states,
        )

        # Tag the fold in a shallow copy of cfg (for TB hparams, etc.)
        fold_cfg = replace(cfg)

        result = run_train(fold_cfg, datamodule=dm)  # returns {"log_dir": ...}
        log_dir = Path(result["log_dir"])

        # Pull last-epoch metrics from CSV logger
        metrics_csv = log_dir / "metrics.csv"
        if metrics_csv.exists():
            m = pd.read_csv(metrics_csv)
            if "epoch" not in m.columns and "step" in m.columns:
                m["epoch"] = m["step"]
            m = m.sort_values(["epoch"]).groupby("epoch").last()
            val_auprc = m["val_auprc"].iloc[-1] if "val_auprc" in m.columns else float("nan")
            val_auroc = m["val_auroc"].iloc[-1] if "val_auroc" in m.columns else float("nan")
            train_loss = m["train_loss"].iloc[-1] if "train_loss" in m.columns else float("nan")
            val_loss   = m["val_loss"].iloc[-1]   if "val_loss"   in m.columns else float("nan")
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
