# irr/data/splits.py
# module for creating training/validation splits for binary classification

import numpy as np

# imports only the split function when entire module is imported
__all__ = ["stratified_split_idx"]

# returns (train_idx, val_idx) with class-stratified split for binary labels y in {0,1}
def stratified_split_idx(y: np.ndarray, val_ratio: float, seed: int = 88):
    y = np.asarray(y).astype(int)
    assert set(np.unique(y)) <= {0, 1}, "y must be binary {0,1}"

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))

    i0 = idx[y == 0].copy()
    i1 = idx[y == 1].copy()
    rng.shuffle(i0)
    rng.shuffle(i1)

    n0_val = int(len(i0) * val_ratio)
    n1_val = int(len(i1) * val_ratio)

    val_idx = np.concatenate([i0[:n0_val], i1[:n1_val]])
    train_idx = np.concatenate([i0[n0_val:], i1[n1_val:]])

    rng.shuffle(train_idx); rng.shuffle(val_idx)
    return train_idx, val_idx
