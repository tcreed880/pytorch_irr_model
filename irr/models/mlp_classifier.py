# irr/models/mlp_classifier.py
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torchmetrics.functional.classification import (
    binary_confusion_matrix,
    binary_precision_recall_curve,
)
import matplotlib.pyplot as plt


# Blocks 

class ResidualMLPBlock(nn.Module):
    """Linear(d,d) -> LayerNorm -> act -> Dropout, with residual skip."""
    def __init__(self, d: int, p: float = 0.1, act: Optional[nn.Module] = None):
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


# LightningModule 

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
    calibrate_on_val: bool = False  # turn ON in normal training; keep OFF during Optuna to save time


class IrrMLPClassifier(pl.LightningModule):
    def __init__(self, cfg: ModelConfig, pos_weight: Optional[torch.Tensor] = None):
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

        # metrics at epoch-level
        self.train_auroc = BinaryAUROC()
        self.train_auprc = BinaryAveragePrecision()
        self.val_auroc   = BinaryAUROC()
        self.val_auprc   = BinaryAveragePrecision()

        # standardizer (fit on train, set via set_standardizer)
        self.register_buffer("x_mean", torch.zeros(cfg.in_dim))
        self.register_buffer("x_std", torch.ones(cfg.in_dim))

        # calibration buffers for probability scaling at prediction
        self.register_buffer("calib_T", torch.tensor(1.0))                # T=1 → no temp scaling
        self.register_buffer("calib_b", torch.tensor(0.0))                # b=0 → no bias shift
        self.register_buffer("use_calibration", torch.tensor(0, dtype=torch.uint8))
        self.register_buffer("best_threshold", torch.tensor(0.50))

        # holders for per-epoch val aggregation
        self._val_logits: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []
        self._last_val_cm: Optional[torch.Tensor] = None

    # utilities
    def set_standardizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        std = std.clone()
        std[std == 0] = 1.0
        self.x_mean.copy_(mean)
        self.x_std.copy_(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.standardize:
            x = (x - self.x_mean) / self.x_std
        return self.net(x)

    def _apply_calibration(self, logits: torch.Tensor) -> torch.Tensor:
        if bool(self.use_calibration.item()):
            return logits / self.calib_T + self.calib_b
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self(x)
        z = self._apply_calibration(logits)
        return torch.sigmoid(z)

    def _get_tb(self):
        if isinstance(self.logger, TensorBoardLogger):
            return self.logger.experiment
        if hasattr(self.trainer, "loggers"):
            for lg in self.trainer.loggers:
                if isinstance(lg, TensorBoardLogger):
                    return lg.experiment
        return None

    def _plot_and_log_cm(self, cm: torch.Tensor, tag: str, step: int) -> None:
        fig, ax = plt.subplots(figsize=(3, 3), dpi=120)
        cm_cpu = cm.cpu().numpy()
        ax.imshow(cm_cpu, interpolation="nearest")
        ax.set_title(tag.split("/")[-1])
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(int(cm_cpu[i, j])), ha="center", va="center")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        fig.tight_layout()
        tb = self._get_tb()
        if tb is not None:
            tb.add_figure(tag, fig, global_step=step)
        plt.close(fig)

    # calibration fitter
    def _fit_temp_bias(self, logits: torch.Tensor, targets: torch.Tensor, max_iter: int = 50):
        """
        Fit temperature (T>0) and bias (b) by minimizing unweighted BCE on validation.
        Optimizes two scalars with LBFGS: T = softplus(t_raw)+eps and b.
        Runs on CPU to be backend-agnostic (MPS/AMP quirks).
        """
        z = logits.detach().float().cpu()
        y = targets.detach().float().cpu()

        t_raw = torch.zeros((), requires_grad=True)   # scalar
        b     = torch.zeros((), requires_grad=True)

        opt = torch.optim.LBFGS([t_raw, b], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(set_to_none=True)
            T = F.softplus(t_raw) + 1e-6
            z_cal = z / T + b
            loss = F.binary_cross_entropy_with_logits(z_cal, y)
            loss.backward()
            return loss

        opt.step(closure)
        T = (F.softplus(t_raw) + 1e-6).detach()
        b = b.detach()
        return T, b

    # train/val steps
    def _step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y.float())
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=y.size(0))
        return loss, logits, y

    def training_step(self, batch, _):
        loss, logits, y = self._step(batch, "train")
        # AUROC/AUPRC take raw scores (logits)
        self.train_auroc.update(logits, y.int())
        self.train_auprc.update(logits, y.int())
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_auroc", self.train_auroc.compute(), prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_auprc", self.train_auprc.compute(), prog_bar=True, on_step=False, on_epoch=True)
        self.train_auroc.reset(); self.train_auprc.reset()

    def on_validation_epoch_start(self) -> None:
        self._val_logits = []
        self._val_targets = []

    def validation_step(self, batch, _):
        loss, logits, y = self._step(batch, "val")
        # unweighted BCE for monitoring/calibration comparability
        val_bce_unweighted = F.binary_cross_entropy_with_logits(logits, y.float())
        self.log("val_bce", val_bce_unweighted, prog_bar=True, on_step=False, on_epoch=True)

        self.val_auroc.update(logits, y.int())
        self.val_auprc.update(logits, y.int())
        self._val_logits.append(logits.detach())
        self._val_targets.append(y.detach().int())
        return loss

    def on_validation_epoch_end(self) -> None:
        # epoch-level ranking metrics
        self.log("val_auroc", self.val_auroc.compute(), prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_auprc", self.val_auprc.compute(), prog_bar=True, on_step=False, on_epoch=True)
        self.val_auroc.reset(); self.val_auprc.reset()

        # aggregate logits/targets
        if len(self._val_logits) == 0:
            return
        logits  = torch.cat(self._val_logits,  dim=0)
        targets = torch.cat(self._val_targets, dim=0)

        # raw uncalibrated epoch BCE
        val_bce_epoch = F.binary_cross_entropy_with_logits(logits, targets.float())
        self.log("val_bce_epoch", val_bce_epoch, on_step=False, on_epoch=True)

        probs_for_cm = torch.sigmoid(logits)
        best_thr_to_use = 0.5

        # optionally fit temperature & bias and compute calibrated metrics
        if self.cfg.calibrate_on_val:
            T, b = self._fit_temp_bias(logits, targets)
            z_cal = logits / T + b
            probs_cal = torch.sigmoid(z_cal)
            val_bce_cal = F.binary_cross_entropy_with_logits(z_cal, targets.float())
            self.log("val_bce_calibrated", val_bce_cal, on_step=False, on_epoch=True)

            # store buffers for checkpoint / inference
            self.calib_T.copy_(T.cpu())
            self.calib_b.copy_(b.cpu())
            self.use_calibration.fill_(1)

            # choose F1-opt threshold on calibrated probs
            prec, rec, thr = binary_precision_recall_curve(probs_cal, targets)
            prec_t, rec_t = prec[:-1], rec[:-1]  # align with thr
            f1 = 2 * prec_t * rec_t / (prec_t + rec_t + 1e-12)
            best_idx = torch.argmax(f1)
            best_thr_to_use = float(thr[best_idx])
            self.best_threshold.fill_(best_thr_to_use)
            self.log("val/best_threshold_f1_cal", best_thr_to_use, on_step=False, on_epoch=True)

            probs_for_cm = probs_cal

        # confusion matrix at chosen threshold
        cm = binary_confusion_matrix(preds=probs_for_cm, target=targets, threshold=best_thr_to_use)
        self._last_val_cm = cm.detach().cpu()

    def on_fit_end(self) -> None:
        cm = getattr(self, "_last_val_cm", None)
        if cm is not None:
            tag = "final/val_confusion_matrix_cal" if bool(self.use_calibration.item()) else "final/val_confusion_matrix"
            self._plot_and_log_cm(cm, tag=tag, step=self.current_epoch)

    # optimizers & schedulers
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        # Cosine across epochs (Lightning sets max_epochs on the trainer)
        t_max = self.trainer.max_epochs if self.trainer and self.trainer.max_epochs is not None else 100
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

    @property
    def final_linear(self) -> nn.Linear:
        """Access the final linear layer (for bias init / calibration)."""
        layer = getattr(self.net, "out", None)
        if layer is None:
            layer = self.net.layers[-1]
        assert isinstance(layer, nn.Linear)
        return layer
