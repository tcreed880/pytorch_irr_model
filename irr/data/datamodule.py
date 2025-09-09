# irr/data/datamodule.py
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

from irr.constants import FEATURES, LABEL_COL
from irr.data.splits import stratified_split_idx
from irr.data.io import load_csvs


class IrrDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for CSV data
    - Loads csv data with irr.data.io.load_csvs
    - Performs a stratified train/val split by default or can accept explicit indices for k-fold CV
    - Standardizes features based on training set stats
    - Provides train and val DataLoaders
    """
    def __init__(
        self,
        data_glob: str,
        batch_size: int,
        val_ratio: float = 0.2,
        seed: int = 88,
        train_idx: Optional[np.ndarray] = None,
        val_idx: Optional[np.ndarray]   = None,
        num_workers: Optional[int] = None,
    ):
        super().__init__()
        self.data_glob = data_glob
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.seed = seed
        self.train_idx = train_idx
        self.val_idx   = val_idx

        # dataloader workers
        if num_workers is None:
            num_workers = min(8, os.cpu_count() or 2)
        self.num_workers = num_workers

        # used after setup()
        self.X_train: torch.Tensor
        self.y_train: torch.Tensor
        self.X_val: torch.Tensor
        self.y_val: torch.Tensor
        self.x_mean: torch.Tensor
        self.x_std:  torch.Tensor

        # Keep the full dataframe around if needed for diagnostics
        self.df: Optional[pd.DataFrame] = None

    def setup(self, stage: Optional[str] = None):
        # load data with load_csvs from io.py
        df = load_csvs(self.data_glob)
        self.df = df  # keep full df if needed

        X_np = df[FEATURES].values
        y_np = df[LABEL_COL].values

        X = torch.tensor(X_np, dtype=torch.float32)
        y = torch.tensor(y_np, dtype=torch.int64)

        # accept premade splits or use stratified random split function
        if self.train_idx is None or self.val_idx is None:
            tr_idx, va_idx = stratified_split_idx(
                y.numpy(), val_ratio=self.val_ratio, seed=self.seed
            )
        else:
            tr_idx, va_idx = self.train_idx, self.val_idx

        # index tensors for train/val sets
        self.X_train, self.y_train = X[tr_idx], y[tr_idx]
        self.X_val, self.y_val = X[va_idx], y[va_idx]

        # standardization metrics with training data
        self.x_mean = self.X_train.mean(dim=0)
        self.x_std  = self.X_train.std(dim=0).clamp_min(1e-8)


    def _make_loader(self, X: torch.Tensor, y: torch.Tensor, shuffle: bool) -> DataLoader:
        ds = TensorDataset(X, y)
        # persistent_workers is False when num_workers == 0
        # workers will parallelize data loading on multi-core CPUs
        persistent = self.num_workers > 0
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=persistent,
            pin_memory=torch.cuda.is_available(),
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.X_train, self.y_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.X_val, self.y_val, shuffle=False)


    # allows cv.py to get the whole dataframe if needed without running setup()
    @staticmethod
    def load_all_df(data_glob: str) -> pd.DataFrame:
        return load_csvs(data_glob)
