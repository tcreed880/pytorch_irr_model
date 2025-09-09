# irr/training/train.py
from pathlib import Path
import dataclasses
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import torch 

from irr.data.datamodule import IrrDataModule
from irr.training.config import RunCfg
from irr.models.tiny_head import TinyHead, TinyCfg

def run_train(cfg: RunCfg, datamodule: IrrDataModule | None = None) -> dict:
    # Data
    dm = datamodule or IrrDataModule(cfg.data_glob, batch_size=cfg.batch_size,
                                     val_ratio=cfg.val_ratio, seed=cfg.seed)
    dm.setup()

    # pos_weight (binary)
    y_train = dm.y_train
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    pos_weight = None if pos == 0 else (neg / max(pos, 1))

    # Model cfg
    model_cfg = getattr(cfg, "model", None)
    if model_cfg is None:
        model_cfg = TinyCfg(
            in_dim=dm.X_train.size(1),
            hidden=getattr(cfg, "hidden", 256),
            depth=getattr(cfg, "depth", 2),
            dropout=getattr(cfg, "dropout", 0.10),
            act=getattr(cfg, "act", "silu"),
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
    else:
        model_cfg.in_dim = dm.X_train.size(1)

    # Model
    model = TinyHead(model_cfg, pos_weight=dm.X_train.new_tensor(pos_weight) if pos_weight else None)

    # Make standardizer a no-op for unit-length embeddings (AlphaEarth)
    model.set_standardizer(torch.zeros(model_cfg.in_dim), torch.ones(model_cfg.in_dim))

    # Loggers
    csv_logger = CSVLogger(save_dir="outputs/logs", name="tiny_head")
    tb_logger  = TensorBoardLogger(save_dir="outputs/logs", name="tiny_head_tb")  # separate run dir

    # (Optional) log hyperparams into TensorBoard’s HParams tab
    try:
        tb_logger.log_hyperparams(dataclasses.asdict(cfg))
    except Exception:
        pass

    callbacks = [
        EarlyStopping(monitor=getattr(cfg, "monitor", "val_auprc"), mode="max",
                      patience=getattr(cfg, "patience", 10), min_delta=getattr(cfg, "min_delta", 1e-4)),
        ModelCheckpoint(monitor=getattr(cfg, "monitor", "val_auprc"), mode="max",
                        save_top_k=1, filename="best")
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=[tb_logger, csv_logger],  # ← both loggers enabled
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=dm)

    return {"log_dir": csv_logger.log_dir}  # or tb_logger.log_dir
