# irr/cli/kfold.py
from __future__ import annotations

import argparse

from irr.training.cv import run_kfold
from irr.configs import TrainConfig
from irr.models.mlp_classifier import ModelConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="K-fold CV for IrrMLPClassifier.")

    # Data and split controls
    p.add_argument("--data-glob", required=True, help='Glob for CSVs, e.g. "raw_data/*.csv"')
    p.add_argument("--k", type=int, default=5, help="Number of folds")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=88)
    p.add_argument(
        "--include-states", nargs="*", default=None,
        help="Filter to these state codes (e.g., --include-states MT OR ID)."
    )
    p.add_argument(
        "--group-col", type=str, default="h3_r7",
        help="Grouping for folds: 'county_fips', '.geo', or 'h3_r{res}'. Use 'none' for label-stratified."
    )

    # Training controls
    p.add_argument("--monitor", type=str, default="val_auprc")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max-epochs", type=int, default=40)

    # Model hyperparameters
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--act", type=str, choices=["relu", "silu", "gelu"], default="silu")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--standardize", action="store_true", help="Apply (x-mean)/std inside the model.")

    return p.parse_args()


def main() -> None:
    a = parse_args()
    group_col = None if a.group_col and a.group_col.lower() == "none" else a.group_col

    # Build nested model config
    model_cfg = ModelConfig(
        hidden=a.hidden,
        depth=a.depth,
        dropout=a.dropout,
        act=a.act,
        lr=a.lr,
        weight_decay=a.weight_decay,
        standardize=a.standardize,
    )

    # Train config used by run_kfold (cv runner passes explicit indices per fold)
    cfg = TrainConfig(
        data_glob=a.data_glob,
        batch_size=a.batch_size,
        # val_ratio is unused in CV (splits are passed explicitly)
        seed=a.seed,
        monitor=a.monitor,
        patience=a.patience,
        max_epochs=a.max_epochs,
        model=model_cfg,
        group_col=group_col,
        # optional state filter for CV
        include_states=a.include_states,
    )

    folds_df, summary = run_kfold(cfg, k=a.k)
    print("\n=== K-fold results ===")
    print(folds_df)
    print("\nSummary:", summary)


if __name__ == "__main__":
    main()


"""
poetry run python -m irr.cli.kfold \
  --data-glob "raw_data/*.csv" \
  --k 5 \
  --group-col h3_r7 \
  --include-states OR ID MT \
  --batch-size 512 \
  --max-epochs 40 \
  --hidden 256 --depth 2 --dropout 0.10 --act silu \
  --lr 1e-3 --weight-decay 1e-4
"""