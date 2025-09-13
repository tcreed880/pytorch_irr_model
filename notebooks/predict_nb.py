# %% [markdown]
# # TinyHead: Batch Predict + Metrics & Confusion Matrix
# - Loads Lightning checkpoint (TinyHead)
# - Applies to `raw_data/*unbalanced*.csv`
# - If a label column is present, computes AUROC, AUPRC, and plots confusion matrix
# - Provides per-county counts and per-county confusion tallies

# %%
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    classification_report,
)

# Import your model
from irr.models.tiny_head import TinyHead, TinyCfg
from irr.constants import FEATURES

# ---------------- User-configurable ----------------
CKPT_PATH = "../outputs/logs/tiny_head_tb/version_21/checkpoints/best.ckpt"  # adjust if needed
DATA_GLOB = "../raw_data/*unbalanced_val*.csv"                           # your files
OUT_CSV   = "predictions/preds_unbalanced_all.csv"
LABEL_COL = "label"      # set to actual label column if different
COUNTY_COL = "county"    # set to None if not available
THRESHOLD = 0.96         # decision threshold for confusion matrix and report
BATCH_SIZE = 4096
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------

# %%
# 1) Load CSVs
paths = sorted(glob.glob(DATA_GLOB))
if not paths:
    raise FileNotFoundError(f"No CSV files matched: {DATA_GLOB}")

dfs = []
for p in paths:
    df = pd.read_csv(p)
    dfs.append(df)

full = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(paths)} files, combined shape: {full.shape}")
full.head()

# %%
# 2) Ensure required feature columns exist and drop rows with missing features
missing_any = [c for c in FEATURES if c not in full.columns]
if missing_any:
    raise ValueError(f"Missing feature columns: {missing_any}")

full = full.dropna(subset=FEATURES).reset_index(drop=True)
print("After dropping rows with missing FEATURES:", full.shape)

# %%
# 3) Load model from checkpoint
model = TinyHead.load_from_checkpoint(CKPT_PATH, cfg=TinyCfg(), strict=False)
model.eval().to(DEVICE)

# 4) Build DataLoader and run inference to get logits & probs
X = torch.tensor(full[FEATURES].values, dtype=torch.float32)
ds = TensorDataset(X)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=0, pin_memory=(DEVICE == "cuda"))

logits_list = []
with torch.no_grad():
    for (xb,) in dl:
        xb = xb.to(DEVICE)
        logits = model(xb)  # (B,)
        logits_list.append(logits.detach().cpu().numpy())

logits = np.concatenate(logits_list, axis=0)
probs = 1.0 / (1.0 + np.exp(-logits))

preds_df = full.copy()
preds_df["prob_irrigated"] = probs
preds_df["pred_irrigated"] = (probs > THRESHOLD).astype(int)

# Save predictions
out_path = Path(OUT_CSV)
out_path.parent.mkdir(parents=True, exist_ok=True)
preds_df.to_csv(out_path, index=False)
print(f"Wrote predictions to {out_path.resolve()}")

# %%
# 5) If labels exist, compute metrics & confusion matrix
has_labels = LABEL_COL in preds_df.columns
if has_labels:
    y_true = preds_df[LABEL_COL].astype(int).values
    y_prob = preds_df["prob_irrigated"].values
    y_pred = (y_prob > THRESHOLD).astype(int)

    # Metrics
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)  # [[TN, FP], [FN, TP]]

    print(f"AUROC: {auroc:.4f}  |  AUPRC: {auprc:.4f}")
    print("Confusion matrix (thr = {:.2f}):".format(THRESHOLD))
    print(cm)

    # Classification report
    print("\nClassification Report (thr={:.2f}):".format(THRESHOLD))
    print(classification_report(y_true, y_pred, digits=4))

# %%
# 6) Plot confusion matrix
if has_labels:
    fig, ax = plt.subplots(figsize=(3,3), dpi=120)
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion Matrix @ thr={THRESHOLD:.2f}")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred 0","Pred 1"])
    ax.set_yticklabels(["True 0","True 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout()
    plt.show()

# %%
# 7) Plot ROC and PR curves
if has_labels:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    # ROC
    plt.figure(figsize=(4,4), dpi=120)
    plt.plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Precision-Recall
    plt.figure(figsize=(4,4), dpi=120)
    plt.plot(rec, prec, label=f"AUPRC={auprc:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

# y_true: 0/1 labels on a proper validation set
# y_prob: model sigmoid probs on that same val set
prec, rec, thr = precision_recall_curve(y_true, y_prob)
f1 = (2*prec[:-1]*rec[:-1]) / np.maximum(prec[:-1]+rec[:-1], 1e-9)
best_idx = np.argmax(f1)
best_thr = float(thr[best_idx])
print(f"Best F1 at threshold ~ {best_thr:.3f}  (P={prec[best_idx]:.2f}, R={rec[best_idx]:.2f})")


# %%
# 8) Per-year and per-county breakdowns (counts of 0/1)
# Try to infer year from filename if a 'year' column doesn't exist
if "year" not in preds_df.columns:
    # attempt to parse years like 2018..2025 from any string column 'source' or fallback to DATA_GLOB file stems
    # If you already have a 'year' column, skip this block.
    import re
    if "source_path" in preds_df.columns:
        src = preds_df["source_path"].astype(str)
    else:
        # create a simple mapping by re-reading file names and assigning in order (approximate)
        # safer: re-load with an added column per file to tag origin
        preds_df = pd.concat(
            [pd.read_csv(p).assign(_source=p) for p in paths],
            ignore_index=True
        )
        # recompute predictions if we reloaded—otherwise skip re-run for speed in this sketch
        # For now, just try to parse year from _source into 'year' for grouping:
        src = preds_df["_source"].astype(str)

    years = src.apply(lambda s: int(re.search(r"20\d{2}", s).group()) if re.search(r"20\d{2}", s) else -1)
    preds_df["year"] = years

# Class counts per year
if has_labels:
    by_year = preds_df.groupby("year")[LABEL_COL].value_counts().unstack(fill_value=0).sort_index()
    print("\nLabel counts by year:")
    display(by_year.head(10))

# Per-county counts and per-county confusion
if COUNTY_COL and COUNTY_COL in preds_df.columns:
    # counts of each class per county
    if has_labels:
        by_county = preds_df.groupby(COUNTY_COL)[LABEL_COL].value_counts().unstack(fill_value=0)
        print("\nLabel counts by county (head):")
        display(by_county.head(10))

        # per-county confusion tallies
        def cm_counts(g):
            yt = g[LABEL_COL].astype(int).values
            yp = (g["prob_irrigated"].values > THRESHOLD).astype(int)
            tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
            return pd.Series(dict(TN=tn, FP=fp, FN=fn, TP=tp, support=len(g)))

        county_cm = preds_df.groupby(COUNTY_COL).apply(cm_counts).sort_values("support", ascending=False)
        print("\nPer-county confusion counts (head):")
        display(county_cm.head(10))
    else:
        print("\nNo labels found, showing raw counts per county based on predictions:")
        pred_counts = preds_df.groupby(COUNTY_COL)["pred_irrigated"].value_counts().unstack(fill_value=0)
        display(pred_counts.head(10))
else:
    print("\nNo COUNTY_COL available; skipping county breakdowns.")

