import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

class PredictionDumpCallback(pl.Callback):
    """Write val/test predictions to CSV: y_true, y_prob (binary) or prob_<i> (multiclass)."""
    def __init__(self, split="val", out_name="val_predictions.csv", include_logits=False):
        assert split in {"val","test"}
        self.split, self.out_name, self.include_logits = split, out_name, include_logits
        self._probs = []; self._targets = []; self._logits = []

    def on_validation_epoch_start(self, trainer, pl_module):
        if self.split=="val": self._reset()
    def on_test_epoch_start(self, trainer, pl_module):
        if self.split=="test": self._reset()

    @torch.no_grad()
    def _collect(self, pl_module, batch):
        x, y = batch
        logits = pl_module(x)
        if logits.ndim==2 and logits.shape[1]==1: logits = logits.squeeze(1)
        logits = logits.detach().cpu()
        y = y.detach().cpu()
        if logits.ndim==1:
            probs = torch.sigmoid(logits)
            self._probs.append(probs.numpy())
        else:
            self._probs.append(torch.softmax(logits, dim=1).numpy())
        if self.include_logits: self._logits.append(logits.numpy())
        self._targets.append(y.numpy())

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, *_):
        if self.split=="val": self._collect(pl_module, batch)
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, *_):
        if self.split=="test": self._collect(pl_module, batch)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.split=="val": self._write(trainer)
    def on_test_epoch_end(self, trainer, pl_module):
        if self.split=="test": self._write(trainer)

    def _reset(self): self._probs.clear(); self._targets.clear(); self._logits.clear()

    def _write(self, trainer):
        if not trainer.is_global_zero: return
        probs = np.concatenate(self._probs, axis=0) if self._probs else np.array([])
        targs = np.concatenate(self._targets, axis=0) if self._targets else np.array([])
        data = {"y_true": targs.astype(int)}
        if probs.ndim==1 or (probs.ndim==2 and probs.shape[1]==1):
            data["y_prob"] = probs.reshape(-1)
        else:
            for i in range(probs.shape[1]): data[f"prob_{i}"] = probs[:, i]
        if self.include_logits and self._logits:
            logits = np.concatenate(self._logits, axis=0)
            if logits.ndim==1: data["logit"] = logits.reshape(-1)
            else:
                for i in range(logits.shape[1]): data[f"logit_{i}"] = logits[:, i]
        # next to metrics.csv
        logger = trainer.logger
        save_dir = Path(getattr(logger, "log_dir", getattr(logger, "_fs", ".")))
        if not save_dir.exists() and hasattr(logger, "save_dir"):
            save_dir = Path(logger.save_dir) / str(getattr(logger, "name","")) / str(getattr(logger, "version",""))
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / self.out_name
        pd.DataFrame(data).to_csv(out, index=False)
        print(f"[PredictionDumpCallback] Wrote {len(targs):,} rows -> {out}")
