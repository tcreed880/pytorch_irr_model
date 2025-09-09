from dataclasses import dataclass, field
from irr.models.tiny_head import TinyCfg

@dataclass
class RunCfg:
    data_glob: str
    batch_size: int = 512
    val_ratio: float = 0.2
    seed: int = 88
    monitor: str = "val_auprc"
    patience: int = 10
    max_epochs: int = 200
    # composed model config
    model: TinyCfg = field(default_factory=TinyCfg)
