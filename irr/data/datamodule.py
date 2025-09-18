# irr/data/datamodule.py
from __future__ import annotations

import os
import json
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from irr.constants import FEATURES, LABEL_COL
from irr.data.io import load_csvs
from irr.data.splits import stratified_split_idx  # keep your existing helper

from h3 import latlng_to_cell 


class IrrDataModule(pl.LightningDataModule):
    """
    DataModule for CSV data (AlphaEarth features + label), with:
      - loading from a glob
      - optional state filtering (include_states)
      - group-aware splits by 'county_fips', '.geo', or H3 (e.g., 'h3_r7')
      - standardization stats on the training split
      - train/val DataLoaders

    Parameters
    ----------
    data_glob : str
        Glob pattern for CSVs.
    batch_size : int
    val_ratio : float, default 0.2
    seed : int, default 88
    train_idx, val_idx : np.ndarray | None
        Precomputed indices (for CV). If provided, bypass internal splitter.
    num_workers : int | None
        Defaults to min(8, cpu_count) if None.
    group_col : str | None
        Column or directive for grouping, e.g.:
          - "county_fips" (default) — requires column present
          - ".geo"                — groups by geometry string
          - "h3_r7"               — groups by H3 at res=7 (requires 'h3')
        If None or missing → stratified split by label.
    include_states : list[str] | None
        If provided, keep only rows where df['state'] is in this list.
    """

    def __init__(
        self,
        data_glob: str,
        batch_size: int,
        val_ratio: float = 0.2,
        seed: int = 88,
        train_idx: Optional[np.ndarray] = None,
        val_idx: Optional[np.ndarray] = None,
        num_workers: Optional[int] = None,
        group_col: Optional[str] = "h3_r7",
        include_states: Optional[List[str]] = None,
    ):
        super().__init__()
        self.data_glob = data_glob
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.seed = seed
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.group_col = group_col
        self.include_states = include_states

        if num_workers is None:
            num_workers = min(8, os.cpu_count() or 2)
        self.num_workers = num_workers

        # Set in setup()
        self.df: Optional[pd.DataFrame] = None
        self.X_train: torch.Tensor
        self.y_train: torch.Tensor
        self.X_val: torch.Tensor
        self.y_val: torch.Tensor
        self.x_mean: torch.Tensor
        self.x_std: torch.Tensor

    # ---------- helpers ----------

    @staticmethod
    def _geo_to_h3(series: pd.Series, res: int) -> pd.Series:
        def to_h3(s: str) -> str:
            c = json.loads(s)["coordinates"]  # GeoJSON order: [lon, lat]
            lon, lat = float(c[0]), float(c[1])
            return latlng_to_cell(lat, lon, res)  

        return series.astype(str).apply(to_h3)

    def _make_groups(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        if not self.group_col:
            return None

        gc = str(self.group_col).lower()

        if gc.startswith("h3_r"):
            res = int(gc.split("h3_r")[-1])
            df = df.copy()
            if ".geo" not in df.columns:
                raise ValueError("H3 grouping requires a '.geo' column in the dataframe.")
            df[gc] = self._geo_to_h3(df[".geo"], res)
            return df[gc].astype(str).to_numpy()

        if gc in df.columns:
            return df[gc].astype(str).to_numpy()

        if gc == ".geo" and ".geo" in df.columns:
            return df[".geo"].astype(str).to_numpy()

        # Unknown or missing group column → no grouping
        return None

    # ---------- PL hooks ----------

    def setup(self, stage: Optional[str] = None) -> None:
        df = load_csvs(self.data_glob)

        # Optional filter by states (e.g., train on MT/OR/ID only)
        if self.include_states:
            df = df[df["state"].isin(self.include_states)].reset_index(drop=True)

        # Ensure required columns exist
        missing = [c for c in FEATURES + [LABEL_COL] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Drop rows with NaNs in features/label
        df = df.dropna(subset=FEATURES + [LABEL_COL]).reset_index(drop=True)
        self.df = df  # keep around for diagnostics

        X_np = df[FEATURES].to_numpy(dtype=np.float32)
        y_np = df[LABEL_COL].astype(np.int64).to_numpy()

        # Use provided indices (e.g., K-fold) or compute a split
        if self.train_idx is not None and self.val_idx is not None:
            tr_idx, va_idx = self.train_idx, self.val_idx
            split_desc = "predefined indices"
        else:
            groups = self._make_groups(df)
            if (groups is not None) and _HAS_SK:
                gss = GroupShuffleSplit(n_splits=1, test_size=self.val_ratio, random_state=self.seed)
                idx = np.arange(len(df))
                tr_idx, va_idx = next(gss.split(idx, y_np, groups=groups))
                split_desc = f"grouped by '{self.group_col}'"
            else:
                # fallback: your helper does stratified splitting by label
                tr_idx, va_idx = stratified_split_idx(y_np, val_ratio=self.val_ratio, seed=self.seed)
                split_desc = "label-stratified"

        self.train_idx, self.val_idx = tr_idx, va_idx
        print(f"[Split] {split_desc}. train={len(tr_idx):,} val={len(va_idx):,}")

        # Tensors
        X = torch.from_numpy(X_np)
        y = torch.from_numpy(y_np)
        self.X_train, self.y_train = X[tr_idx], y[tr_idx]
        self.X_val, self.y_val = X[va_idx], y[va_idx]

        # Standardization statistics from training split
        self.x_mean = self.X_train.mean(dim=0)
        self.x_std = self.X_train.std(dim=0).clamp_min(1e-8)

    # ---------- loaders ----------

    def _make_loader(self, X: torch.Tensor, y: torch.Tensor, shuffle: bool) -> DataLoader:
        ds = TensorDataset(X, y)
        persistent = self.num_workers > 0
        pin = torch.cuda.is_available()  # only helps on CUDA
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=persistent,
            pin_memory=pin,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.X_train, self.y_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.X_val, self.y_val, shuffle=False)

    # ---------- diagnostics ----------

    def assert_no_group_leakage(self, group_col: Optional[str] = None) -> None:
        """Raise if any group appears in both train and val."""
        if self.df is None or self.train_idx is None or self.val_idx is None:
            raise RuntimeError("Call setup() first.")

        gcol = (group_col or self.group_col)
        if not gcol:
            print("[Leakage check] group_col=None → skipped.")
            return

        gcol = str(gcol)

        # If H3 was requested but not precomputed, compute now
        if gcol.startswith("h3_r"):
            res = int(gcol.split("h3_r")[-1])
            if gcol not in self.df.columns:
                if ".geo" not in self.df.columns:
                    print(f"[Leakage check] '.geo' missing; cannot compute {gcol}. Skipping.")
                    return
                self.df[gcol] = self._geo_to_h3(self.df[".geo"], res)
            col = gcol
        else:
            col = gcol

        if col not in self.df.columns:
            print(f"[Leakage check] column '{col}' not found → skipped.")
            return

        gvals = self.df[col].astype(str).to_numpy()
        train_groups = set(gvals[self.train_idx])
        val_groups = set(gvals[self.val_idx])
        overlap = train_groups & val_groups
        print(f"[Leakage check] train_groups={len(train_groups):,} "
              f"val_groups={len(val_groups):,} overlap={len(overlap):,}")
        if overlap:
            examples = list(overlap)[:3]
            raise AssertionError(f"Found {len(overlap)} overlapping groups for '{col}'. Examples: {examples}")

    # allows cv.py to get the whole dataframe if needed without running setup()
    @staticmethod
    def load_all_df(data_glob: str) -> pd.DataFrame:
        return load_csvs(data_glob)
