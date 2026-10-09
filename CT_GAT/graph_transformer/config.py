"""One run's configuration, saved as config.json in every run directory."""
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal, Optional

CACHE = Path('C:/Users/chaoa/Downloads/graph_transformer_cache')
RUNS = Path('C:/Users/chaoa/Downloads/graph_transformer_runs')


@dataclass(frozen=True)
class RunConfig:
    protocol: Literal['bundle', 'kfold'] = 'bundle'
    fold: Optional[int] = None
    arch: Literal['x86', 'all'] = 'x86'
    seed: int = 0
    no_graph: bool = False

    d_model: int = 128
    heads: int = 4
    fn_layers: int = 2
    bin_layers: int = 2
    dropout: float = 0.1
    max_segment_blocks: int = 128
    max_segments: int = 4096
    degree_clip: int = 15

    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 4
    epochs: int = 40
    patience: int = 8
    threshold: float = 0.5
    min_df: int = 2
    val_fraction: float = 0.15
    grad_clip: float = 1.0
    num_workers: int = 2

    def __post_init__(self):
        if self.protocol not in ('bundle', 'kfold'):
            raise ValueError(f'protocol {self.protocol}')
        if self.arch not in ('x86', 'all'):
            raise ValueError(f'arch {self.arch}')
        if (self.protocol == 'kfold') != (self.fold is not None):
            raise ValueError('kfold needs --fold and bundle must not have one')
        if self.fold is not None and not 0 <= self.fold < 5:
            raise ValueError(f'fold {self.fold}')

    @property
    def name(self):
        split = 'bundle' if self.protocol == 'bundle' else f'kfold{self.fold}'
        return f"{split}_{self.arch}_{'nograph' if self.no_graph else 'graph'}_s{self.seed}"

    @property
    def group(self):
        """Runs that differ only by seed (and fold, for kfold) share a group."""
        return f"{self.protocol}_{self.arch}_{'nograph' if self.no_graph else 'graph'}"

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})
