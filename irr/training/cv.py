# irr/training/cv.py
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import pandas as pd

from irr.constants import FEATURES, LABEL_COL
from irr.configs import TrainConfig
from irr.data.datamodule import IrrDataModule
from irr.training.train import run_train

# sklearn splitters
try:
    from sklearn.model_selection import StratifiedGroupKFold  # sklearn >= 1.1
except Exception:
    StratifiedGroupKFold = None  # type: ignore

try:
    from sklearn.model_selection import GroupKFold
except Exception:
    GroupKFold = None  # type: ignore

try:
    from sklearn.model_selection import StratifiedKFold
except Exception:
    StratifiedKFold = None  # type: ignore


def _normalize_states(states: Optional[Iterable[str]]) -> Optional[list[str]]:
    if states is None:
        return None
    return [str(s).strip().upper() for s in states]


def _compute_groups_with_datamodule(df: pd.DataFrame, group_col: Optional[str], seed: int) -> Optional[np.ndarray]:
    """
    Reuse IrrDataModule's grouping logic (supports H3 like 'h3_r7', '.geo', or a df column).
    Does NOT call setup(), only uses _make_groups.
    """
    if not group_col or str(group_col).lower() == "none":
        return None

    # Minimal instance; batch_size/val_ratio unused here
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
    """
    Yield (train_idx, val_idx) for k folds using the best available splitter:
      - StratifiedGroupKFold (grouped + stratified)
      - GroupKFold (grouped only)
      - StratifiedKFold (stratified only)
    """
    n = len(y)
    X_dummy = np.arange(n)  # sklearn wants X but we don't need features to split

    if groups is not None:
        if StratifiedGroupKFold is not None:
            sgkf = StratifiedGroupKFold(n_splits=k, shuffle=shuffle, random_state=seed)
            for tr_idx, va_idx in sgkf.split(X_dummy, y, groups=groups):
                yield tr_idx, va_idx
            return
        if GroupKFold is not None:
            gkf = GroupKFold(n_splits=k)  # no shuffle in GroupKFold
            for tr_idx, va_idx in gkf.split(X_dummy, y, groups=groups):
                yield tr_idx, va_idx
            return
        # If we get here, sklearn is too old or missing; fall back to stratified without grouping
        if StratifiedKFold is None:
            raise ImportError("scikit-learn is required for cross-validation.")
        skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)
        for tr_idx, va_idx in skf.split(X_dummy, y):
            yield tr_idx, va_idx
        return

    # No groups → use stratified k-fold (preferred)
    if StratifiedKFold is None:
        raise ImportError("scikit-learn is required for cross-validation.")
    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)
    for tr_idx, va_idx in skf.split(X_dummy, y):
        yield tr_idx, va_idx


def run_kfold(cfg: TrainConfig, k: int = 5, shuffle: bool = True):
    """
    Run K-fold CV with group-aware, label-stratified splitting when possible.

    - If cfg.group_col is set (e.g., 'h3_r7' or 'county_fips'), we:
        * Prefer StratifiedGroupKFold (both grouped+stratified)
        * Else GroupKFold (grouped only)
      Otherwise, we use StratifiedKFold (stratified only).
    - Respects cfg.include_states if present.
    - For each fold, we pass explicit indices to IrrDataModule so it won’t re-split.
    - Returns (folds_df, summary_dict).
    """
    # Load data once
    df = IrrDataModule.load_all_df(cfg.data_glob)

    # Optional state filter
    inc_states = _normalize_states(getattr(cfg, "include_states", None))
    if inc_states is not None:
        if "state" not in df.columns:
            raise ValueError("include_states was provided but the dataframe has no 'state' column.")
        df = df.copy()
        df["state"] = df["state"].astype(str).str.strip().str.upper()
        df = df[df["state"].isin(inc_states)].reset_index(drop=True)

    # Schema checks
    missing = [c for c in FEATURES + [LABEL_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # y only (run_train loads features again via DataModule)
    y = df[LABEL_COL].astype(int).to_numpy()

    # Build groups using the same logic as DataModule
    groups = _compute_groups_with_datamodule(df, getattr(cfg, "group_col", None), seed=cfg.seed)

    fold_rows = []
    for fold_id, (tr_idx, va_idx) in enumerate(_iter_folds(y, k=k, seed=cfg.seed, groups=groups, shuffle=shuffle), start=1):
        print(f"\n=== Fold {fold_id}/{k} ===")

        # Defensive class sanity
        y_tr, y_va = y[tr_idx], y[va_idx]
        if y_tr.max() == y_tr.min():
            raise RuntimeError(f"Fold {fold_id}: train split is single-class (pos or neg only).")
        if y_va.max() == y_va.min():
            raise RuntimeError(f"Fold {fold_id}: val split is single-class (pos or neg only).")

        # Defensive group leakage check (when groups exist)
        if groups is not None:
            tr_g, va_g = set(groups[tr_idx]), set(groups[va_idx])
            overlap = tr_g & va_g
            print(f"[Fold {fold_id}] groups train={len(tr_g):,} val={len(va_g):,} overlap={len(overlap):,}")
            if overlap:
                raise AssertionError(f"Group leakage in fold {fold_id}: {len(overlap)} overlapping groups.")

        # Build a per-fold DataModule with fixed indices (no re-splitting)
        dm = IrrDataModule(
            data_glob=cfg.data_glob,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            train_idx=tr_idx,
            val_idx=va_idx,
            group_col=None,  # explicit indices => don't compute a split again
            include_states=inc_states,
        )

        # Tag the fold if you want (e.g., TB hparams); otherwise cfg is fine
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
