# irr/cli/train.py
from __future__ import annotations

import argparse
from irr.training.train import run_train
from irr.configs import TrainConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MLP classifier on AE/IRR CSV data.")

    # Data / splitting
    p.add_argument("--data-glob", required=True, help='Glob for CSVs, e.g. "raw_data/*.csv"')
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=88)
    p.add_argument(
        "--include-states",
        nargs="*",
        default=None,
        help='Filter to these state codes (e.g., --include-states MT OR ID). Omit to use all.',
    )
    p.add_argument(
        "--group-col",
        type=str,
        default="h3_r7",
        help="Grouping for train/val split: 'county_fips', '.geo', or 'h3_r{res}'. "
             "Use 'none' for label-stratified split.",
    )

    # Training control
    p.add_argument("--monitor", type=str, default="val_auprc")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--max-epochs", type=int, default=40)

    # Model hyperparams (convenience fields; train.py will build ModelConfig)
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

    cfg = TrainConfig(
        data_glob=a.data_glob,
        batch_size=a.batch_size,
        val_ratio=a.val_ratio,
        seed=a.seed,
        include_states=a.include_states,
        group_col=group_col,
        monitor=a.monitor,
        patience=a.patience,
        min_delta=a.min_delta,
        max_epochs=a.max_epochs,
        # model fields
        hidden=a.hidden,
        depth=a.depth,
        dropout=a.dropout,
        act=a.act,
        lr=a.lr,
        weight_decay=a.weight_decay,
        standardize=a.standardize,
    )

    info = run_train(cfg)
    print(f"[INFO] Finished. Artifacts in: {info['log_dir']}")


if __name__ == "__main__":
    main()


"""
example usage:
poetry run python -m irr.cli.train \
  --data-glob "raw_data/*cropland*.csv" \
  --include-states MT OR ID \
  --group-col h3_r7 \
  --batch-size 1024 \
  --monitor val_auprc \
  --patience 12 \
  --max-epochs 80 \
  --hidden 512 --depth 2 --dropout 0.02 --act relu \
  --lr 3.2e-4 --weight-decay 3.7e-5
"""