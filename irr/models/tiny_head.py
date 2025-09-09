# irr/models/tiny_head.py
from dataclasses import dataclass
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryConfusionMatrix
import matplotlib.pyplot as plt

# First we standardize inputs: (x - mean) / std
# Map 64 input vlues to 256 features + SiLU activation
# Then N residual MLP blocks (256-dim) with LayerNorm, SiLU, Dropout(0.1), + shortcut(original x)
# Then linear layer 256 to 1 output logit, BCEWithLogitsLoss (sigmoid, binary cross-entropy)
# Loss, backprop, AdamW, cosine LR scheduler
# AUROC and AUPRC from raw logits

# ---------- Blocks ----------

class ResidualMLPBlock(nn.Module):
    def __init__(self, d: int, p: float = 0.1, act: nn.Module | None = None):
        super().__init__()
        act = act if act is not None else nn.SiLU()
        self.seq = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            act,
            nn.Dropout(p),
        )
    def forward(self, x):
        return x + self.seq(x)

class MLPHead(nn.Module):
    """64-dim input -> [Linear->act] -> N residual blocks -> Linear->1"""
    def __init__(self, in_dim=64, hidden=256, depth=2, p=0.1, act_name="silu"):
        super().__init__()
        act = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}[act_name.lower()]()
        layers = [nn.Linear(in_dim, hidden), act]
        for _ in range(depth):
            layers.append(ResidualMLPBlock(hidden, p=p, act=act))
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):  # (B, in_dim) -> (B,)
        return self.net(x).squeeze(1)

# ---------- LightningModule ----------
@dataclass
class TinyCfg:
    in_dim: int = 64
    hidden: int = 256
    depth: int = 2
    dropout: float = 0.10
    act: str = "silu"
    lr: float = 1e-3
    weight_decay: float = 1e-4


class TinyHead(pl.LightningModule):
    def __init__(self, cfg: TinyCfg, pos_weight: torch.Tensor | None = None):
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
        self.val_cm = BinaryConfusionMatrix()
        # standardizer (fit on train)
        self.register_buffer("x_mean", torch.zeros(cfg.in_dim))
        self.register_buffer("x_std", torch.ones(cfg.in_dim))

    # ----- utilities -----
    def set_standardizer(self, mean: torch.Tensor, std: torch.Tensor):
        std = std.clone()
        std[std == 0] = 1.0
        self.x_mean.copy_(mean)
        self.x_std.copy_(std)

    def forward(self, x):
        x = (x - self.x_mean) / self.x_std
        return self.net(x)

    # ----- train/val -----
    def _step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y.float())
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=y.size(0))
        return loss, logits, y

    def training_step(self, batch, _):
        loss, logits, y = self._step(batch, "train")
        self.train_auroc.update(logits, y.int())
        self.train_auprc.update(logits, y.int())
        return loss

    def on_train_epoch_end(self):
        self.log("train_auroc", self.train_auroc.compute(), prog_bar=True)
        self.log("train_auprc", self.train_auprc.compute(), prog_bar=True)
        self.train_auroc.reset()
        self.train_auprc.reset()

    def validation_step(self, batch, _):
        loss, logits, y = self._step(batch, "val")
        self.val_auroc.update(logits, y.int())
        self.val_auprc.update(logits, y.int())
        probs = torch.sigmoid(logits)
        self.val_cm.update(probs, y.int()) 
        return loss

    def on_validation_epoch_end(self):
        self.log("val_auroc", self.val_auroc.compute(), prog_bar=True)
        self.log("val_auprc", self.val_auprc.compute(), prog_bar=True)
        self.val_auroc.reset()
        self.val_auprc.reset()
        # Compute confusion matrix (2x2 tensor: [[TN, FP], [FN, TP]])
        cm = self.val_cm.compute().detach().cpu().numpy()
        self.val_cm.reset()

        # Make a small figure
        fig, ax = plt.subplots(figsize=(3,3), dpi=120)
        ax.imshow(cm, interpolation="nearest")
        ax.set_title("Val Confusion Matrix @ thr=0.5")
        ax.set_xticks([0,1])
        ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred 0","Pred 1"])
        ax.set_yticklabels(["True 0","True 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="w")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.tight_layout()

        # Log to TensorBoard (works since TB logger is first in your list)
        if isinstance(self.logger, TensorBoardLogger) or hasattr(self.logger, "experiment"):
            tb = getattr(self.logger, "experiment", None)
            if tb is None and hasattr(self.trainer, "loggers"):
                # Fallback if multiple loggers
                for lg in self.trainer.loggers:
                    if isinstance(lg, TensorBoardLogger):
                        tb = lg.experiment
                        break
            if tb is not None:
                tb.add_figure("val/confusion_matrix", fig, global_step=self.current_epoch)
        plt.close(fig)


    # ----- optimizers & schedulers -----
    def configure_optimizers(self):
        # AdamW + cosine decay (nice general recipe)
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        # Cosine over the full training; Lightning will set T_max to max_epochs
        # If you want exact steps: use CosineAnnealingLR with T_max=self.trainer.max_epochs
        t_max = self.trainer.max_epochs if self.trainer.max_epochs is not None else 100
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch",
            },
        }
