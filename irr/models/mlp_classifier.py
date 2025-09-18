# irr/models/mlp_classifier.py
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryConfusionMatrix,
)
import matplotlib.pyplot as plt


# ---------- Blocks ----------

class ResidualMLPBlock(nn.Module):
    """Simple residual MLP block: Linear(d,d) -> LayerNorm -> act -> Dropout, with skip."""
    def __init__(self, d: int, p: float = 0.1, act: nn.Module | None = None):
        super().__init__()
        act = act if act is not None else nn.SiLU()
        self.seq = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            act,
            nn.Dropout(p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.seq(x)


class MLPHead(nn.Module):
    """[in_dim] -> Linear -> act -> (ResidualBlock x depth) -> Linear -> 1 logit."""
    def __init__(self, in_dim=64, hidden=256, depth=2, p=0.1, act_name="silu"):
        super().__init__()
        act = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}[act_name.lower()]()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), act]
        for _ in range(depth):
            layers.append(ResidualMLPBlock(hidden, p=p, act=act))
        layers.append(nn.Linear(hidden, 1))
        self.layers = nn.Sequential(*layers)
        # expose the final linear for calibration/bias init
        self.out: nn.Linear = self.layers[-1]  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, in_dim) -> (B,)
        return self.layers(x).squeeze(1)


# ---------- LightningModule ----------

@dataclass
class ModelConfig:
    in_dim: int = 64
    hidden: int = 256
    depth: int = 2
    dropout: float = 0.10
    act: str = "silu"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    standardize: bool = False  # set True only if your features need it


class IrrMLPClassifier(pl.LightningModule):
    """
    Binary classifier:
      - optional standardization: x <- (x - mean) / std
      - MLPHead backbone -> 1 logit
      - BCEWithLogitsLoss (supports pos_weight)
      - Logs AUROC/AUPRC (from raw scores) + a confusion matrix at threshold=0.5
    """
    def __init__(self, cfg: ModelConfig, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weight"])
        self.cfg = cfg

        self.net = MLPHead(
            in_dim=cfg.in_dim,
            hidden=cfg.hidden,
            depth=cfg.depth,
            p=cfg.dropout,
            act_name=cfg.act,
        )

        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # metrics (epoch-level)
        self.train_auroc = BinaryAUROC()
        self.train_auprc = BinaryAveragePrecision()
        self.val_auroc = BinaryAUROC()
        self.val_auprc = BinaryAveragePrecision()
        self.val_cm = BinaryConfusionMatrix()  # default threshold=0.5

        # standardizer (fit on train, set via set_standardizer)
        self.register_buffer("x_mean", torch.zeros(cfg.in_dim))
        self.register_buffer("x_std", torch.ones(cfg.in_dim))

    # ----- utilities -----
    def set_standardizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Copy mean/std into buffers; guards against zeros in std."""
        std = std.clone()
        std[std == 0] = 1.0
        self.x_mean.copy_(mean)
        self.x_std.copy_(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.standardize:
            x = (x - self.x_mean) / self.x_std
        return self.net(x)

    # ----- train/val -----
    def _step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y.float())
        self.log(
            f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True,
            batch_size=y.size(0)
        )
        return loss, logits, y

    def training_step(self, batch, _):
        loss, logits, y = self._step(batch, "train")
        # AUROC/AUPRC can consume raw scores (sigmoid monotone for AP)
        self.train_auroc.update(logits, y.int())
        self.train_auprc.update(logits, y.int())
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_auroc", self.train_auroc.compute(), prog_bar=True)
        self.log("train_auprc", self.train_auprc.compute(), prog_bar=True)
        self.train_auroc.reset()
        self.train_auprc.reset()

    def validation_step(self, batch, _):
        loss, logits, y = self._step(batch, "val")
        self.val_auroc.update(logits, y.int())
        self.val_auprc.update(logits, y.int())
        probs = torch.sigmoid(logits)
        self.val_cm.update(probs, y.int())  # thresh=0.5 inside metric
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_auroc", self.val_auroc.compute(), prog_bar=True)
        self.log("val_auprc", self.val_auprc.compute(), prog_bar=True)
        self.val_auroc.reset()
        self.val_auprc.reset()

        # confusion matrix figure
        cm = self.val_cm.compute().detach().cpu().numpy()
        self.val_cm.reset()

        fig, ax = plt.subplots(figsize=(3, 3), dpi=120)
        ax.imshow(cm, interpolation="nearest")
        ax.set_title("Val Confusion Matrix @ thr=0.5")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="w")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        fig.tight_layout()

        # Log to TensorBoard (works with single or multiple loggers)
        tb = None
        if isinstance(self.logger, TensorBoardLogger):
            tb = self.logger.experiment
        elif hasattr(self.trainer, "loggers"):
            for lg in self.trainer.loggers:
                if isinstance(lg, TensorBoardLogger):
                    tb = lg.experiment
                    break
        if tb is not None:
            tb.add_figure("val/confusion_matrix", fig, global_step=self.current_epoch)
        plt.close(fig)

    # ----- optimizers & schedulers -----
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        # Cosine over epochs (Lightning sets max_epochs on the trainer)
        t_max = self.trainer.max_epochs if self.trainer.max_epochs is not None else 100
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

    @property
    def final_linear(self) -> nn.Linear:
        """Access the final linear layer (for bias init / calibration)."""
        layer = getattr(self.net, "out", None)
        if layer is None:
            # fallback if internal name changes
            layer = self.net.layers[-1]
        assert isinstance(layer, nn.Linear)
        return layer


