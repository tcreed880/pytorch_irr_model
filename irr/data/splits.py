# irr/data/splits.py
# Utilities for creating train/val splits for binary classification.
# Return (train_idx, val_idx) for binary labels y in {0,1} with a class-stratified split.

from __future__ import annotations
import numpy as np

__all__ = ["stratified_split_idx"]


def stratified_split_idx(y: np.ndarray, val_ratio: float, seed: int = 88) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(np.int64)
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0,1); got {val_ratio}")

    n = y.shape[0]
    idx = np.arange(n, dtype=np.int64)
    i0 = idx[y == 0]
    i1 = idx[y == 1]

    rng = np.random.default_rng(seed)

    # If one class is missing, do a simple random split
    if len(i0) == 0 or len(i1) == 0:
        perm = rng.permutation(idx)
        n_val = int(round(n * val_ratio))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        return train_idx, val_idx

    # Shuffle within each class
    rng.shuffle(i0)
    rng.shuffle(i1)

    # Stratified counts
    n0_val = int(round(len(i0) * val_ratio))
    n1_val = int(round(len(i1) * val_ratio))

    # Try to keep at least one per class in both splits when possible
    if len(i0) > 1:
        n0_val = min(max(n0_val, 1), len(i0) - 1)
    else:
        n0_val = 0  # single sample stays in train

    if len(i1) > 1:
        n1_val = min(max(n1_val, 1), len(i1) - 1)
    else:
        n1_val = 0

    val_idx = np.concatenate([i0[:n0_val], i1[:n1_val]])
    train_idx = np.concatenate([i0[n0_val:], i1[n1_val:]])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx