import numpy as np
from irr.data.splits import stratified_split_idx

def test_stratified_split_balance_disjoint_deterministic():
    n = 1000
    rng = np.random.default_rng(0)
    # 30% positives, 70% negatives
    y = (rng.random(n) < 0.30).astype(int)
    val_ratio = 0.20

    train_idx, val_idx = stratified_split_idx(y, val_ratio=val_ratio, seed=88)

    # disjoint and complete
    train_set, val_set = set(train_idx), set(val_idx)
    assert train_set.isdisjoint(val_set)
    assert len(train_idx) + len(val_idx) == n

    # class balance in val matches floor(len(class) * val_ratio)
    y_val = y[val_idx]
    pos_total = int(y.sum())
    neg_total = n - pos_total
    exp_pos_val = int(pos_total * val_ratio)
    exp_neg_val = int(neg_total * val_ratio)
    assert y_val.sum() == exp_pos_val
    assert (len(y_val) - y_val.sum()) == exp_neg_val

    # deterministic with same seed
    t2, v2 = stratified_split_idx(y, val_ratio=val_ratio, seed=42)
    assert np.array_equal(train_idx, t2)
    assert np.array_equal(val_idx, v2)
