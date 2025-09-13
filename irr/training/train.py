# irr/training/train.py
import dataclasses
import math
import warnings
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import torch

from irr.data.datamodule import IrrDataModule
from irr.training.config import RunCfg
from irr.models.tiny_head import TinyHead, TinyCfg

def run_train(cfg: RunCfg, datamodule: IrrDataModule | None = None) -> dict:
    # Data
    dm = datamodule or IrrDataModule(
        cfg.data_glob, batch_size=cfg.batch_size,
        val_ratio=cfg.val_ratio, seed=cfg.seed
    )
    dm.setup()

    # ---- class balance on TRAIN ONLY ----
    y_train = dm.y_train
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    if pos == 0:
        warnings.warn("No positive examples in the TRAIN split; pos_weight disabled.")
        pos_weight_tensor = None
        pi = 0.0
    else:
        ratio = neg / max(pos, 1)
        pos_weight_tensor = torch.tensor([ratio], dtype=torch.float32)  # shape [1]
        pi = pos / (pos + neg)

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
    model = TinyHead(model_cfg, pos_weight=pos_weight_tensor)

    # Standardizer (no-op for unit-length embeddings)
    model.set_standardizer(torch.zeros(model_cfg.in_dim), torch.ones(model_cfg.in_dim))

    # init final bias to TRAIN prior to help calibration
    # do this outside autograd because it’s initialization, not a learnable update this instant.
    if 0.0 < pi < 1.0:
        with torch.no_grad():
            model.final_linear.bias.fill_(math.log(pi / (1.0 - pi)))


    # Loggers
    csv_logger = CSVLogger(save_dir="outputs/logs", name="tiny_head")
    tb_logger  = TensorBoardLogger(save_dir="outputs/logs", name="tiny_head_tb")
    try:
        tb_logger.log_hyperparams({**dataclasses.asdict(cfg),
                                   "train_pos": pos, "train_neg": neg,
                                   "pos_weight": None if pos_weight_tensor is None else float(pos_weight_tensor[0])})
    except Exception:
        pass

    callbacks = [
        EarlyStopping(monitor=getattr(cfg, "monitor", "val_auprc"), mode="max",
                      patience=getattr(cfg, "patience", 10), min_delta=getattr(cfg, "min_delta", 1e-5)),
        ModelCheckpoint(monitor=getattr(cfg, "monitor", "val_auprc"), mode="max",
                        save_top_k=1, filename="best")
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=[tb_logger, csv_logger],
        callbacks=callbacks,
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(model, datamodule=dm)
    return {"log_dir": csv_logger.log_dir}
