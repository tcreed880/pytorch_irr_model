import numpy as np
import pandas as pd
import torch

from irr.constants import FEATURES, LABEL_COL
from irr.data.datamodule import IrrDataModule
from irr.data.splits import stratified_split_idx

def _make_df(n=300, seed=88):
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=0.0, scale=1.0, size=(n, len(FEATURES))).astype(np.float32)
    y = np.zeros(n, dtype=np.int64)
    y[: n // 2] = 1  # balanced
    rng.shuffle(y)
    df = pd.DataFrame(X, columns=FEATURES)
    df[LABEL_COL] = y
    return df

def test_standardization_uses_train_only_stats(tmp_path):
    # build synthetic CSV
    df = _make_df(n=300, seed=123)
    csv_path = tmp_path / "toy.csv"
    df.to_csv(csv_path, index=False)

    # create and set up the DataModule
    dm = IrrDataModule(str(csv_path), batch_size=64, val_ratio=0.2, seed=88)
    dm.setup()

    # recompute train indices exactly as the DM does
    X_full = torch.tensor(df[FEATURES].values, dtype=torch.float32)
    y_full = torch.tensor(df[LABEL_COL].values, dtype=torch.int64)
    tr_idx, _ = stratified_split_idx(y_full.numpy(), val_ratio=0.2, seed=88)

    # expected train-only stats
    train_mean = X_full[tr_idx].mean(dim=0)
    train_std = X_full[tr_idx].std(dim=0)

    # DM should store train-only stats
    assert torch.allclose(dm.x_mean, train_mean, atol=1e-6)
    assert torch.allclose(dm.x_std, train_std, atol=1e-6)

    # Standardizing TRAIN data should give ~zero mean and ~unit std per feature
    z_train = (dm.X_train - dm.x_mean) / dm.x_std
    mean_after = z_train.mean(dim=0)
    std_after = z_train.std(dim=0)
    assert torch.allclose(mean_after, torch.zeros_like(mean_after), atol=1e-5)
    # allow small numerical drift for std
    assert torch.allclose(std_after, torch.ones_like(std_after), atol=1e-3)
