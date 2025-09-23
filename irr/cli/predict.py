# irr/cli/predict.py
# The prediction notebook can also be used instead of this CLI script.

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from irr.models.mlp_classifier import IrrMLPClassifier, ModelConfig
from irr.constants import FEATURES



def pick_device(cli_device: str | None) -> str:
    """Choose device: CLI override > CUDA > MPS > CPU."""
    if cli_device:
        return cli_device.lower()
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_new_data(data_glob: str) -> pd.DataFrame:
    """Load one or more CSVs that contain FEATURES (label not required)."""
    paths = sorted(glob.glob(data_glob))
    if not paths:
        raise FileNotFoundError(f"No CSV files matched: {data_glob}")
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"{p} is missing required feature columns: {missing}")
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    # Drop rows with missing feature values
    full = full.dropna(subset=FEATURES)
    return full


def predict_df(
    df: pd.DataFrame,
    ckpt_path: str,
    batch_size: int = 4096,
    device: str | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Run model inference on a DataFrame of new samples and return df with predictions."""
    # Load model from checkpoint (cfg is saved in the ckpt; ModelConfig() is a placeholder)
    model: IrrMLPClassifier = IrrMLPClassifier.load_from_checkpoint(ckpt_path, cfg=ModelConfig())
    model.eval()

    # Choose device
    dev = pick_device(device)
    model.to(dev)

    # Build dataLoader
    X = torch.tensor(df[FEATURES].values, dtype=torch.float32)
    ds = TensorDataset(X)
    pin = (dev == "cuda")  # pin_memory helps only on CUDA
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    # Inference loop
    logits_list = []
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(dev, non_blocking=pin)
            logits = model(xb)  # (B,)
            logits_list.append(logits.detach().cpu().numpy())

    logits = np.concatenate(logits_list, axis=0)
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    out = df.copy()
    out["logit_irrigated"] = logits
    out["prob_irrigated"] = probs
    out["pred_irrigated"] = (probs > threshold).astype(np.int64)
    return out


def main():
    p = argparse.ArgumentParser(description="Apply trained IrrMLPClassifier to new data CSVs.")
    p.add_argument("--ckpt", required=True, help="Path to Lightning checkpoint (e.g., best.ckpt).")
    p.add_argument("--data-glob", required=True, help='CSV glob, e.g. "new_data/*.csv".')
    p.add_argument("--out-csv", required=True, help="Output CSV path.")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for class=1.")
    p.add_argument("--device", type=str, default=None, help='Force device: "cpu", "cuda", or "mps".')
    args = p.parse_args()

    df = read_new_data(args.data_glob)
    preds = predict_df(
        df,
        args.ckpt,
        batch_size=args.batch_size,
        device=args.device,
        threshold=args.threshold,
    )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_path, index=False)
    print(f"Wrote predictions to {out_path}")
    print(f"Rows: {len(preds):,} | Threshold: {args.threshold} | Device: {pick_device(args.device)}")


if __name__ == "__main__":
    main()
