# irr/cli/predict.py
from __future__ import annotations
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from irr.models.tiny_head import TinyHead, TinyCfg
from irr.constants import FEATURES


def read_new_data(data_glob: str) -> pd.DataFrame:
    """Load one or more CSVs that contain FEATURES (no label required)."""
    paths = sorted(glob.glob(data_glob))
    if not paths:
        raise FileNotFoundError(f"No CSV files matched: {data_glob}")
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"{p} missing feature columns: {missing}")
        # keep only features + any passthrough columns you want to preserve
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    # Drop rows with missing feature values
    full = full.dropna(subset=FEATURES)
    return full


def predict_df(df: pd.DataFrame, ckpt_path: str, batch_size: int = 4096,
               device: str | None = None, threshold: float = 0.5) -> pd.DataFrame:
    """Run model inference on a DataFrame of new samples and return df with predictions."""
    # Load model from checkpoint
    # Note: cfg is saved in the checkpoint; passing TinyCfg() here is a placeholder.
    model = TinyHead.load_from_checkpoint(ckpt_path, cfg=TinyCfg())
    model.eval()

    # Choose device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Build DataLoader for efficiency
    X = torch.tensor(df[FEATURES].values, dtype=torch.float32)
    ds = TensorDataset(X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))

    # Inference loop
    logits_list = []
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            logits = model(xb)  # (B,)
            logits_list.append(logits.detach().cpu().numpy())

    logits = np.concatenate(logits_list, axis=0)
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    out = df.copy()
    out["prob_irrigated"] = probs
    out["pred_irrigated"] = (probs > threshold).astype(np.int64)
    return out


def main():
    p = argparse.ArgumentParser(description="Apply trained TinyHead model to new data CSVs.")
    p.add_argument("--ckpt", required=True, help="Path to Lightning checkpoint (best.ckpt).")
    p.add_argument("--data-glob", required=True, help='CSV glob, e.g. "new_data/*.csv".')
    p.add_argument("--out-csv", required=True, help="Output CSV path.")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for class=1.")
    p.add_argument("--device", type=str, default=None, help='Force device: "cpu" or "cuda".')
    args = p.parse_args()

    df = read_new_data(args.data_glob)
    preds = predict_df(df, args.ckpt, batch_size=args.batch_size, device=args.device, threshold=args.threshold)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_path, index=False)
    print(f"Wrote predictions to {out_path}")


if __name__ == "__main__":
    main()
