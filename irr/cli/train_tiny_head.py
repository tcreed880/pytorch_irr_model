from irr.models.tiny_head import TinyCfg
from irr.training.train import RunCfg, run_train
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-glob", required=True)
    # datamodule args
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=88)
    p.add_argument("--monitor", type=str, default="val_auprc")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max-epochs", type=int, default=40)   

    # model args
    p.add_argument("--hidden", type=int, default=TinyCfg.hidden)
    p.add_argument("--depth", type=int, default=TinyCfg.depth)
    p.add_argument("--dropout", type=float, default=TinyCfg.dropout)
    p.add_argument("--act", type=str, choices=["relu","silu","gelu"], default=TinyCfg.act)
    p.add_argument("--lr", type=float, default=TinyCfg.lr)
    p.add_argument("--weight-decay", type=float, default=TinyCfg.weight_decay)
    a = p.parse_args()

    # Build TinyCfg from the subset, then RunCfg
    model_cfg = TinyCfg(
        hidden=a.hidden, depth=a.depth, dropout=a.dropout, act=a.act,
        lr=a.lr, weight_decay=a.weight_decay
    )
    run_cfg = RunCfg(
        data_glob=a.data_glob, batch_size=a.batch_size, val_ratio=a.val_ratio,
        seed=a.seed, monitor=a.monitor, patience=a.patience, max_epochs=a.max_epochs, model=model_cfg
    )

    logdir = run_train(run_cfg)
    print(f"[INFO] Finished. Artifacts in: {logdir}")

if __name__ == "__main__":
    main()
