# irr/configs.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from irr.models.mlp_classifier import ModelConfig


@dataclass
class TrainConfig:
    # Data
    data_glob: str
    batch_size: int = 512
    val_ratio: float = 0.2
    seed: int = 88
    include_states: Optional[List[str]] = None  # e.g., ["MT","OR","ID"]
    group_col: Optional[str] = "h3_r7"          # e.g., "county_fips", ".geo", or "h3_r7"

    # Training control
    monitor: str = "val_auprc"
    patience: int = 10
    min_delta: float = 1e-5
    max_epochs: int = 300

    # Model: either provide a full ModelConfig here...
    model: Optional[ModelConfig] = None

    # ...or let train.py build one from these convenience fields:
    hidden: int = 256
    depth: int = 2
    dropout: float = 0.10
    act: str = "silu"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    standardize: bool = False
