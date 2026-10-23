"""One run's configuration, saved as config.json in every run directory."""
import json
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal, Optional

from common.paths import CACHE, COMBINED_CACHE, DEFAULT_CACHE, RUNS, cache_for, protocol_in  # noqa: F401

DIRECTIONS = ('both', 'forward', 'backward', 'undirected')
DIRECTION_SUFFIX = {'forward': 'fwd', 'backward': 'bwd', 'undirected': 'undir'}
CONTROLS = {'': '', 'no_edges': 'noedges', 'chain': 'chain', 'shuffle': 'shuffle', 'permute_types': 'permtypes'}
ENCODERS = ('mean', 'tfidf', 'raw', 'presence', 'rawlog')
SELECT_ON = ('overall', 'cell')
SUBGRAPH_POLICIES = ('cap',)
BLOCK_INPUTS = ('tokens', 'vec')
MIN_BUDGET = 100  # blocks; 0 means the binary cap
VOCABS = ('ours', 'yanping', 'hybrid', 'noapi')
IMPORTS = ('', 'header')
IMPORT_SUFFIX = {'header': 'imph'}
MODELS = ('gnn', 'tokenseq', 'fivechannel')
THRESHOLD_FROM = ('val', 'oof')      # where the decision threshold of the sequence models is chosen (s04_evaluation/val_threshold.py)
# options that were removed on 2026-10-20 (step 25); the run directories that used them stay on disk but are not read
REMOVED = {'model': ('transformer',), 'subgraph': ('lcc',), 'imports': ('unreferenced',)}
SEQ_LEN, MAX_WINDOWS = 512, 64      # Model 1 window length and training windows per binary (defaults: no name suffix)
TOKENSEQ_VAL_SEED = 0                # the block export carves its validation split with seed 0
MAX_INSTRUCTIONS = 128               # Model 1 over five-channel tuples (fivechannel): instructions per binary, as Janice wrote it
FIVECHANNEL_DATASETS = (('kfold', 'x86'), ('kfold', 'all'), ('kfold_v3', 'all'), ('kfold_v3', 'matched'))     # the (protocol, arch) pairs of a pipeline-B build


class RemovedOption(ValueError):
    pass


@dataclass(frozen=True)
class RunConfig:
    protocol: Literal['bundle', 'kfold', 'kfold_v3'] = 'bundle'
    fold: Optional[int] = None
    arch: Literal['x86', 'all', 'matched'] = 'x86'
    seed: int = 0
    val_seed: Optional[int] = None

    d_model: int = 128
    dropout: float = 0.1
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

    model: Literal['gnn', 'tokenseq', 'fivechannel'] = 'gnn'
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
    select_on: str = 'overall'
    subgraph: str = 'cap'
    budget: int = 0
    block_input: str = 'tokens'
    block_vec: str = ''
    vocab: str = 'ours'
    imports: str = ''
    seq_len: int = 512
    max_windows: int = 64
    max_instructions: int = 128
    per_block: bool = False
    threshold_from: str = 'val'
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
        if self.model not in MODELS:
            raise ValueError(f'model {self.model}')
        if self.val_seed is not None and self.val_seed < 0:
            raise ValueError(f'val_seed {self.val_seed}')
        if self.seq_len < 1 or self.max_windows < 1:
            raise ValueError(f'seq_len and max_windows are at least 1: {self.seq_len}, {self.max_windows}')
        if self.model != 'tokenseq' and (self.seq_len, self.max_windows) != (SEQ_LEN, MAX_WINDOWS):
            raise ValueError("seq_len and max_windows need model='tokenseq'")
        if self.max_instructions < 1:
            raise ValueError(f'max_instructions is at least 1: {self.max_instructions}')
        if self.model != 'fivechannel' and self.max_instructions != MAX_INSTRUCTIONS:
            raise ValueError("max_instructions needs model='fivechannel'")
        if self.per_block and self.model != 'fivechannel':
            raise ValueError("per_block needs model='fivechannel'")
        if self.threshold_from not in THRESHOLD_FROM:
            raise ValueError(f'threshold_from {self.threshold_from}: one of {THRESHOLD_FROM}')
        if self.threshold_from != 'val' and self.model not in ('tokenseq', 'fivechannel'):
            raise ValueError(f"threshold_from={self.threshold_from!r} needs model='tokenseq' or 'fivechannel'")
        if self.block_input not in BLOCK_INPUTS:
            raise ValueError(f'block_input {self.block_input}')
        if (self.block_input == 'vec') != bool(self.block_vec):
            raise ValueError("block_vec names the precomputed vectors and goes with block_input='vec' (and only then)")
        if self.block_vec and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', self.block_vec):
            raise ValueError(f'block_vec is a directory name (letters, digits, . _ -): {self.block_vec}')
        if self.vocab not in VOCABS:
            raise ValueError(f'vocab {self.vocab}')
        if self.imports not in IMPORTS:
            raise ValueError(f'imports {self.imports}')
        if self.vocab != 'ours' and self.block_input == 'vec':
            raise ValueError("vocab needs block_input='tokens' (precomputed block vectors have no vocabulary)")
        if self.encoder is not None and self.encoder not in ENCODERS:
            raise ValueError(f'encoder {self.encoder}')
        if self.select_on not in SELECT_ON:
            raise ValueError(f'select_on {self.select_on}')
        if self.subgraph not in SUBGRAPH_POLICIES:
            raise ValueError(f'subgraph {self.subgraph}')
        if self.budget != 0 and self.budget < MIN_BUDGET:
            raise ValueError(f'budget must be 0 (the binary cap) or at least {MIN_BUDGET} blocks: {self.budget}')
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
        if self.model == 'tokenseq':
            self.check_tokenseq()
        elif self.model == 'fivechannel':
            self.check_fivechannel()
        else:
            if self.untyped and self.control in ('permute_types', 'no_edges'):
                raise ValueError(f'untyped has no meaning with control {self.control}')
            if self.block_input == 'vec':
                if self.mask_tokens > 0:
                    raise ValueError("mask_tokens needs block_input='tokens' (a vector has no tokens to mask)")
                if self.encoder is not None:
                    raise ValueError("encoder (token weighting) needs block_input='tokens'")
                if self.protocol != 'kfold':
                    raise ValueError("block_input='vec' reads per-fold vectors and needs protocol 'kfold'")

    def check_tokenseq(self):
        """Model 1 reads the fold of a block export: its validation split is the export's carve (val_seed 0, forced here),
        and none of the GNN, graph or weighting options exist for it."""
        if self.protocol == 'bundle':
            raise ValueError("model 'tokenseq' reads a fold of a block export and needs a kfold protocol")
        if self.val_seed not in (None, TOKENSEQ_VAL_SEED):
            raise ValueError(f"model 'tokenseq' uses the export's validation carve (val_seed {TOKENSEQ_VAL_SEED}), "
                             f'not val_seed {self.val_seed}')
        object.__setattr__(self, 'val_seed', TOKENSEQ_VAL_SEED)
        used = self._graph_options_used()
        if used:
            raise ValueError(f"{', '.join(used)} do not apply to model='tokenseq'")

    def check_fivechannel(self):
        """Model 1 over the five-channel tuples of a pipeline-B build (Janice Park's model): one fold of a dataset of the team
        pipelines (`FIVECHANNEL_DATASETS`: kfold x86, the default, kfold all, kfold_v3 all and matched), the validation carve
        follows the run seed (as for the GNN arms and the teammates' arms), and none of the GNN or graph options exists for
        it. `arch_balance` is the exception: it weights the loss by (class, architecture) cell and adds `-ab` to the name, on a
        dataset with both architectures."""
        if (self.protocol, self.arch) not in FIVECHANNEL_DATASETS:
            raise ValueError("model 'fivechannel' reads a pipeline-B build: protocol 'kfold' with arch 'x86' or 'all', or protocol "
                             "'kfold_v3' with arch 'all' or 'matched'")
        if self.val_seed is not None:
            raise ValueError("model 'fivechannel' carves its validation split with the run seed (val_seed stays unset)")
        if self.arch_balance and self.arch == 'x86':
            raise ValueError("arch_balance weights the (class, architecture) cells: model 'fivechannel' needs arch 'all' or 'matched' for it")
        used = [u for u in self._graph_options_used() if u != 'arch_balance']
        if used:
            raise ValueError(f"{', '.join(used)} do not apply to model='fivechannel'")

    def _graph_options_used(self):
        """The GNN-only options set on this config, for the error message of a sequence model."""
        rates = (self.mask_tokens, self.drop_functions, self.drop_library, self.mask_blocks, self.aux_recon, self.aux_edge,
                 self.drop_edges, self.flip_edges)
        unsupported = {'encoder': self.encoder is not None, 'readout': self.readout != 'fixed',
                       'control': bool(self.control), 'untyped': self.untyped, 'no_call_mp': self.no_call_mp,
                       'stack_tfidf': self.stack_tfidf, 'mask/drop/aux/edge rate': any(r > 0 for r in rates),
                       'direction': self.direction != 'both', 'subgraph/budget': self.subgraph != 'cap' or self.budget != 0,
                       'block_input/block_vec': self.block_input != 'tokens' or bool(self.block_vec),
                       'vocab': self.vocab != 'ours', 'imports': bool(self.imports),
                       'gnn_fn_layers/gnn_bin_layers': (self.gnn_fn_layers, self.gnn_bin_layers) != (3, 2),
                       'train_fraction': self.train_fraction != 1.0, 'arch_balance': self.arch_balance,
                       'select_on': self.select_on != 'overall', 'fixed_epochs': self.fixed_epochs > 0}
        return [k for k, v in unsupported.items() if v]

    @property
    def block_encoder(self):
        return self.encoder or 'tfidf'

    @property
    def variant(self):
        tf = [f'tf{round(100 * self.train_fraction)}'] if self.train_fraction < 1 else []
        ab = ['ab'] if self.arch_balance else []
        sel = ['selcell'] if self.select_on == 'cell' else []
        vs = [f'vs{self.val_seed}'] if self.val_seed is not None and self.val_seed != self.seed else []
        thoof = ['thoof'] if self.threshold_from == 'oof' else []
        if self.model == 'tokenseq':
            return '-'.join(['tokenseq'] + [f'{s}{n}' for s, n, default in (('L', self.seq_len, SEQ_LEN),
                                                                             ('w', self.max_windows, MAX_WINDOWS))
                                            if n != default] + thoof)
        if self.model == 'fivechannel':
            return '-'.join(['fivechannel'] + ([f'L{self.max_instructions}'] if self.max_instructions != MAX_INSTRUCTIONS else [])
                            + (['pb'] if self.per_block else []) + ab + thoof)
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
        if self.block_encoder != 'tfidf':
            parts.append(f'enc-{self.block_encoder}')
        if self.block_input == 'vec':
            parts.append(f'bv-{self.block_vec}')
        if self.vocab != 'ours':
            parts.append(f'voc-{self.vocab}')
        if self.imports:
            parts.append(IMPORT_SUFFIX[self.imports])
        if self.readout == 'gated':
            parts.append('gated')
        parts += sel
        if self.subgraph != 'cap' or self.budget:
            # below 1000 blocks the suffix spells the budget out (cap500), so 128 and 500 cannot share a name
            parts.append(f'sg-{self.subgraph}' + (f'{self.budget // 1000}k' if self.budget >= 1000 or not self.budget
                                                  else str(self.budget)))
        return '-'.join(parts + vs)

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
        """The config saved in `path`; raises RemovedOption for a run that used an option removed in step 25."""
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
        raw.setdefault('model', 'transformer')                                  # files before the model field were transformer runs
        gone = [f'{k}={raw[k]}' for k, values in REMOVED.items() if raw.get(k) in values]
        if raw.get('no_graph'):
            gone.append('no_graph')
        if gone:
            raise RemovedOption(f"{path}: {', '.join(gone)} no longer exist")
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    @classmethod
    def try_load(cls, path):
        """`load`, or None for a run directory of a removed option (the directory stays on disk, nothing reads it)."""
        try:
            return cls.load(path)
        except RemovedOption:
            return None
