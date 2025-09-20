# irr/data/datamodule.py
from __future__ import annotations

import os
import json
from typing import Iterable, Optional, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

from irr.constants import FEATURES, LABEL_COL
from irr.data.io import load_csvs
from irr.data.splits import stratified_split_idx  # fallback split

# --- optional deps (guarded) ---
try:
    from sklearn.model_selection import GroupShuffleSplit  # type: ignore
except Exception:
    GroupShuffleSplit = None  # type: ignore

# H3 is optional; we support both v3 and v4 function names.
try:
    import h3  # type: ignore
    _HAS_H3 = True
except Exception:
    h3 = None  # type: ignore
    _HAS_H3 = False


def _normalize_states(states: Optional[Iterable[str]]) -> Optional[List[str]]:
    if states is None:
        return None
    return [str(s).strip().upper() for s in states]


def _latlon_to_h3(lat: float, lon: float, res: int) -> str:
    """
    Wrapper that supports both h3 v4 (latlng_to_cell) and v3 (geo_to_h3).
    """
    if not _HAS_H3:
        raise ImportError("To use H3 grouping, install 'h3' (e.g., pip install h3).")
    if hasattr(h3, "latlng_to_cell"):  # v4
        return h3.latlng_to_cell(lat, lon, res)  # type: ignore[attr-defined]
    if hasattr(h3, "geo_to_h3"):       # v3
        return h3.geo_to_h3(lat, lon, res)      # type: ignore[attr-defined]
    raise RuntimeError("Unsupported h3 version: no latlng_to_cell / geo_to_h3 found.")


class IrrDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for tabular CSV data.

    - Loads with irr.data.io.load_csvs(glob)
    - Optional state filtering (include_states=['OR','ID',...])
    - Optional group-aware train/val split (H3 like 'h3_r7', 'county_fips', or '.geo')
    - Falls back to label-stratified split when grouping not available
    - Exposes train/val DataLoaders
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
        group_col: Optional[str] = None,              # e.g., 'h3_r7', 'county_fips', '.geo', or None
        include_states: Optional[Iterable[str]] = None,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.data_glob = data_glob
        self.batch_size = batch_size
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.group_col = group_col
        self.include_states = _normalize_states(include_states)
        self.debug = bool(debug)

        # dataloader workers
        if num_workers is None:
            num_workers = min(8, os.cpu_count() or 2)
        self.num_workers = int(num_workers)

        # will be set in setup()
        self.df: Optional[pd.DataFrame] = None
        self.X_train: torch.Tensor
        self.y_train: torch.Tensor
        self.X_val: torch.Tensor
        self.y_val: torch.Tensor
        self.x_mean: torch.Tensor
        self.x_std: torch.Tensor

    # ---------- grouping helpers ----------
    @staticmethod
    def _geo_to_h3(series: pd.Series, res: int) -> pd.Series:
        """
        Convert GeoJSON point strings in '.geo' to H3 cell ids at resolution `res`.
        """
        def to_h3(s: str) -> str:
            c = json.loads(s)["coordinates"]
            lon, lat = float(c[0]), float(c[1])
            return _latlon_to_h3(lat, lon, res)
        return series.astype(str).apply(to_h3)

    def _make_groups(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Return group ids (np.ndarray[str]) given group_col:
          - 'h3_r{res}' uses H3 on '.geo'
          - '.geo' uses the raw geo string
          - any existing dataframe column name
          - None → no grouping
        """
        if not self.group_col or str(self.group_col).lower() == "none":
            return None

        gc = str(self.group_col).lower()

        if gc.startswith("h3_r"):
            if ".geo" not in df.columns:
                raise ValueError("H3 grouping requires a '.geo' column in the dataframe.")
            res = int(gc.split("h3_r")[-1])
            df_local = df.copy()
            df_local[gc] = self._geo_to_h3(df_local[".geo"], res)
            return df_local[gc].astype(str).to_numpy()

        if gc == ".geo" and ".geo" in df.columns:
            return df[".geo"].astype(str).to_numpy()

        if gc in df.columns:
            return df[gc].astype(str).to_numpy()

        # unknown → no grouping
        return None

    # ---------- diagnostics ----------
    @staticmethod
    def _print_group_stats(groups: Optional[np.ndarray], label: Optional[str] = None) -> None:
        """
        Light debug: print number of unique groups if provided.
        Accepts Optional to satisfy type checkers and avoid len(None).
        """
        if groups is None:
            if label:
                print(f"[Groups:{label}] no grouping")
            return
        n_unique = len(set(groups.tolist()))
        if label:
            print(f"[Groups:{label}] unique={n_unique:,}")
        else:
            print(f"[Groups] unique={n_unique:,}")

    def assert_no_group_leakage(self, group_col: Optional[str] = None) -> None:
        """
        Ensure no group overlaps between train and val (only if groups are available).
        """
        if self.df is None or self.train_idx is None or self.val_idx is None:
            raise RuntimeError("Call setup() before assert_no_group_leakage().")

        gc = group_col if group_col is not None else self.group_col
        groups = self._make_groups(self.df) if gc else None
        if groups is None:
            print("[Group check] no group_col set; skipping leakage check.")
            return

        tr_g = set(groups[self.train_idx])
        va_g = set(groups[self.val_idx])
        overlap = tr_g & va_g
        print(f"[Group check] train_groups={len(tr_g):,}  val_groups={len(va_g):,}  overlap={len(overlap):,}")
        if overlap:
            examples = list(overlap)[:3]
            raise AssertionError(f"Found {len(overlap)} overlapping groups. Examples: {examples}")

    # ---------- Lightning hooks ----------
    def setup(self, stage: Optional[str] = None) -> None:
        # Load data
        df = load_csvs(self.data_glob)

        # Optional filter by state
        if self.include_states is not None:
            if "state" not in df.columns:
                raise ValueError("include_states was provided but dataframe has no 'state' column.")
            df = df.copy()
            df["state"] = df["state"].astype(str).str.strip().str.upper()
            df = df[df["state"].isin(self.include_states)].reset_index(drop=True)

        # Ensure schema
        missing = [c for c in FEATURES + [LABEL_COL] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Drop NaNs
        df = df.dropna(subset=FEATURES + [LABEL_COL]).reset_index(drop=True)
        self.df = df  # keep for diagnostics

        # Numpy arrays
        X_np = df[FEATURES].to_numpy(dtype=np.float32)
        y_np = df[LABEL_COL].astype(np.int64).to_numpy()

        # Use provided indices or compute split
        if self.train_idx is not None and self.val_idx is not None:
            tr_idx, va_idx = self.train_idx, self.val_idx
            split_desc = "predefined indices"
        else:
            groups = self._make_groups(df)
            # Pylance-safe guard: only call GroupShuffleSplit if imported
            if (groups is not None) and (GroupShuffleSplit is not None):
                gss = GroupShuffleSplit(n_splits=1, test_size=self.val_ratio, random_state=self.seed)
                all_idx = np.arange(len(df))
                tr_idx, va_idx = next(gss.split(all_idx, y_np, groups=groups))
                split_desc = f"grouped by '{self.group_col}'"
            else:
                if (groups is not None) and (GroupShuffleSplit is None):
                    print("group_col provided but scikit-learn is not installed; using label-stratified split instead.")
                tr_idx, va_idx = stratified_split_idx(y_np, val_ratio=self.val_ratio, seed=self.seed)
                split_desc = "label-stratified"

        self.train_idx, self.val_idx = tr_idx, va_idx
        print(f"[Split] {split_desc}. train={len(tr_idx):,}  val={len(va_idx):,}")

        # Tensors
        X = torch.from_numpy(X_np)
        y = torch.from_numpy(y_np)
        self.X_train, self.y_train = X[tr_idx], y[tr_idx]
        self.X_val, self.y_val = X[va_idx], y[va_idx]

        # Standardization stats (kept even if your embeddings are unit norm)
        self.x_mean = self.X_train.mean(dim=0)
        self.x_std = self.X_train.std(dim=0).clamp_min(1e-8)

        # Optional: quick diagnostics
        if self.debug:
            self._print_group_stats(self._make_groups(df), label="full")
            print(f"[Targets] train pos={int((self.y_train==1).sum()):,} / {len(self.y_train):,} | "
                  f"val pos={int((self.y_val==1).sum()):,} / {len(self.y_val):,}")

    def _make_loader(self, X: torch.Tensor, y: torch.Tensor, shuffle: bool) -> DataLoader:
        ds = TensorDataset(X, y)
        persistent = self.num_workers > 0
        # pin_memory is only useful for CUDA; keep False for portability (OK for MPS/CPU)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=persistent,
            pin_memory=False,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.X_train, self.y_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.X_val, self.y_val, shuffle=False)

    # convenience for CV: load DF without running setup
    @staticmethod
    def load_all_df(data_glob: str) -> pd.DataFrame:
        return load_csvs(data_glob)
