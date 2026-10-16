"""One run's configuration, saved as config.json in every run directory."""
import json
import os
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal, Optional

DEFAULT_CACHE = Path('C:/Users/chaoa/Downloads/graph_transformer_cache_v2')
COMBINED_CACHE = Path('C:/Users/chaoa/Downloads/graph_transformer_cache_combined')


def cache_for(protocol, env=None):
    """GT_CACHE when set; otherwise the combined cache for kfold_v3 and the v2 cache for every other protocol."""
    env = os.environ if env is None else env
    if env.get('GT_CACHE'):
        return Path(env['GT_CACHE'])
    return COMBINED_CACHE if protocol == 'kfold_v3' else DEFAULT_CACHE


def protocol_in(argv):
    """The value of --protocol in a command line, or None."""
    for i, a in enumerate(argv):
        if a.startswith('--protocol='):
            return a.partition('=')[2]
        if a == '--protocol' and i + 1 < len(argv):
            return argv[i + 1]


CACHE = cache_for(protocol_in(sys.argv[1:]))
RUNS = Path('C:/Users/chaoa/Downloads/graph_transformer_runs')

DIRECTIONS = ('both', 'forward', 'backward', 'undirected')
DIRECTION_SUFFIX = {'forward': 'fwd', 'backward': 'bwd', 'undirected': 'undir'}
CONTROLS = {'': '', 'no_edges': 'noedges', 'chain': 'chain', 'shuffle': 'shuffle', 'permute_types': 'permtypes'}


@dataclass(frozen=True)
class RunConfig:
    protocol: Literal['bundle', 'kfold', 'kfold_v3'] = 'bundle'
    fold: Optional[int] = None
    arch: Literal['x86', 'all', 'matched'] = 'x86'
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

    model: Literal['transformer', 'gnn'] = 'transformer'
    encoder: Optional[str] = None
    readout: str = 'fixed'
    control: str = ''
    untyped: bool = False
    no_call_mp: bool = False
    stack_tfidf: bool = False
    mask_tokens: float = 0.0
    drop_functions: float = 0.0
    drop_library: float = 0.0
    mask_blocks: float = 0.0
    aux_recon: float = 0.0
    aux_edge: float = 0.0
    aux_weight: float = 0.3
    gnn_fn_layers: int = 3
    gnn_bin_layers: int = 2
    direction: str = 'both'
    drop_edges: float = 0.0
    flip_edges: float = 0.0
    train_fraction: float = 1.0
    arch_balance: bool = False
    batch_blocks: int = 300000
    fixed_epochs: int = 0
    tag: str = ''
    cache: str = ''

    def __post_init__(self):
        if self.protocol not in ('bundle', 'kfold', 'kfold_v3'):
            raise ValueError(f'protocol {self.protocol}')
        if self.arch not in ('x86', 'all', 'matched'):
            raise ValueError(f'arch {self.arch}')
        if (self.protocol != 'bundle') != (self.fold is not None):
            raise ValueError('kfold needs --fold and bundle must not have one')
        if self.fold is not None and not 0 <= self.fold < 5:
            raise ValueError(f'fold {self.fold}')
        if self.model not in ('transformer', 'gnn'):
            raise ValueError(f'model {self.model}')
        if self.encoder not in (None, 'mean', 'tfidf'):
            raise ValueError(f'encoder {self.encoder}')
        if self.readout not in ('fixed', 'gated'):
            raise ValueError(f'readout {self.readout}')
        if self.control not in CONTROLS:
            raise ValueError(f'control {self.control}')
        rates = (self.mask_tokens, self.drop_functions, self.drop_library, self.mask_blocks,
                 self.aux_recon, self.aux_edge)
        if any(not 0 <= r < 1 for r in rates):
            raise ValueError(f'rates must be in [0, 1): {rates}')
        if self.direction not in DIRECTIONS:
            raise ValueError(f'direction {self.direction}')
        if any(not 0 <= r < 1 for r in (self.drop_edges, self.flip_edges)):
            raise ValueError(f'edge rates must be in [0, 1): {self.drop_edges}, {self.flip_edges}')
        if not 0 < self.train_fraction <= 1:
            raise ValueError(f'train_fraction must be in (0, 1]: {self.train_fraction}')
        if self.model == 'gnn':
            if self.no_graph:
                raise ValueError('no_graph applies to the transformer only')
            if self.untyped and self.control in ('permute_types', 'no_edges'):
                raise ValueError(f'untyped has no meaning with control {self.control}')
        else:
            gnn_only = {'encoder': self.encoder is not None, 'readout': self.readout != 'fixed',
                        'control': bool(self.control), 'untyped': self.untyped, 'no_call_mp': self.no_call_mp,
                        'stack_tfidf': self.stack_tfidf, 'mask/drop/aux rate': any(r > 0 for r in rates),
                        'direction': self.direction != 'both',
                        'drop/flip edges': bool(self.drop_edges or self.flip_edges)}
            used = [k for k, v in gnn_only.items() if v]
            if used:
                raise ValueError(f"{', '.join(used)} need model='gnn'")

    @property
    def block_encoder(self):
        return self.encoder or ('mean' if self.model == 'transformer' else 'tfidf')

    @property
    def segment_cap(self):
        return self.max_segment_blocks if self.model == 'transformer' else None

    @property
    def variant(self):
        tf = [f'tf{round(100 * self.train_fraction)}'] if self.train_fraction < 1 else []
        ab = ['ab'] if self.arch_balance else []
        if self.model == 'transformer':
            return '-'.join(['nograph' if self.no_graph else 'graph'] + tf + ab)
        parts = ['gnn']
        if self.control:
            parts.append(CONTROLS[self.control])
        parts += [s for s, on in (('untyped', self.untyped), ('nocall', self.no_call_mp),
                                  ('stack', self.stack_tfidf)) if on]
        parts += [f'{s}{round(100 * p)}' for s, p in (('mtok', self.mask_tokens), ('dfn', self.drop_functions),
                                                      ('dlib', self.drop_library), ('mblk', self.mask_blocks)) if p]
        if self.aux_recon > 0 or self.aux_edge > 0:
            parts.append('aux')
        parts += [DIRECTION_SUFFIX[self.direction]] if self.direction != 'both' else []
        parts += [f'{s}{round(100 * p)}' for s, p in (('dedge', self.drop_edges), ('flip', self.flip_edges)) if p]
        parts += [f'{s}{n}' for s, n, default in (('L', self.gnn_fn_layers, 3), ('B', self.gnn_bin_layers, 2))
                  if n != default]
        parts += tf + ab
        if self.block_encoder == 'mean':
            parts.append('enc-mean')
        if self.readout == 'gated':
            parts.append('gated')
        return '-'.join(parts)

    @property
    def name(self):
        split = {'bundle': 'bundle', 'kfold': f'kfold{self.fold}', 'kfold_v3': f'v3kfold{self.fold}'}[self.protocol]
        return f"{split}_{self.arch}_{self.variant}_s{self.seed}" + (f'_{self.tag}' if self.tag else '')

    @property
    def group(self):
        """Runs that differ only by seed (and fold, for kfold) share a group."""
        return f"{self.protocol}_{self.arch}_{self.variant}" + (f'_{self.tag}' if self.tag else '')

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})
