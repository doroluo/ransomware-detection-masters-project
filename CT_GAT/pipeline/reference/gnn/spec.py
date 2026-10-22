"""The GNN as a `common.registry.ModelSpec`: what `experiments/train.py` asks of it. The options it serves (controls, masking,
vocabulary, imports header, subgraph cap and budget, block-vector input, TF-IDF stacking) are documented with the training
CLI and in README.md of this stage."""
import importlib.util
import json
from pathlib import Path

import numpy as np

from common.config import CACHE, RUNS, RunConfig
from s02_preprocess.tokenizer.vocab import TokenVocab
from s03b_cfg_feature.gnn.batching import loader
from s03b_cfg_feature.gnn.model import HierGNN
from s03b_cfg_feature.gnn.s10_unused_import_features.imports_feat import ImportVocab

VECTORS_PACKAGE = 's03c_cfg_transformer.precompute_blockvec'


def build_model(cfg: RunConfig, vocab, d_vec=None, n_imports=0):
    """The network of `cfg`. With block_input='vec' there is no vocabulary (`vocab` None) and `d_vec` is the size of
    the precomputed block vectors; `n_imports` is the size of the import vocabulary of an `imports` run."""
    return HierGNN(cfg, vocab.total if vocab else 0, d_vec, n_imports)


def vectors_package_present():
    try:
        return importlib.util.find_spec(VECTORS_PACKAGE) is not None
    except ModuleNotFoundError:
        return False


def open_block_vecs(cfg: RunConfig):
    """The BlockVecSource of a block_input='vec' run, checked against the run: same cache, protocol and arch as the export
    it was computed from. The vectors are written by `s03c_cfg_transformer/precompute_blockvec.py`; without that package
    the option stops here with a message instead of a missing-file error."""
    if not vectors_package_present():
        raise ModuleNotFoundError(
            f"--block-input vec reads the block vectors written by {VECTORS_PACKAGE}, which this checkout does not have "
            f"(stage 3c, s03c_cfg_transformer, has not been added); run without --block-input vec or add that package")
    from s03b_cfg_feature.gnn.s05_block_counting.block_vecs import BlockVecSource
    vecs = BlockVecSource(CACHE, cfg.block_vec, cfg.fold)
    meta = vecs.store_meta()
    if (meta['protocol'], meta['arch']) != (cfg.protocol, cfg.arch) or Path(meta['cache']) != Path(CACHE):
        raise ValueError(f"{cfg.block_vec} is for {meta['protocol']} {meta['arch']} on {meta['cache']}, this run is "
                         f'{cfg.protocol} {cfg.arch} on {CACHE}')
    return vecs


def stack_logits(cfg: RunConfig, manifest):
    """{sample_id: standardised out-of-fold TF-IDF logit} for train, val and test rows of `manifest`. The logits are
    read from RUNS/baselines/<baseline dir>/stack_logits.csv; the file must hold every row under the same role it has
    here (train rows out of fold, val and test rows from the full fit), and the standardisation uses the training rows
    only (mean and standard deviation)."""
    from s04_evaluation.baselines import baseline_name, read_stack_logits
    raw = read_stack_logits(RUNS / 'baselines' / baseline_name(cfg) / 'stack_logits.csv', manifest)
    train = np.array([raw[r['sample_id']] for r in manifest.train])
    mean, std = float(train.mean()), float(max(train.std(), 1e-6))
    return {k: np.float32((v - mean) / std) for k, v in raw.items()}


class GnnSpec:
    name = 'gnn'
    requires = ('index.csv', 'vocab.json', 'samples')

    def __init__(self):
        self.vocab = self.vecs = self.stack = self.imports = None

    def check(self, cfg: RunConfig):
        if cfg.aux_recon > 0 or cfg.aux_edge > 0:
            raise NotImplementedError('--aux-recon / --aux-edge are deferred')

    def build(self, cfg: RunConfig, info):
        manifest, out = info.manifest, info.out
        self.vecs = open_block_vecs(cfg) if cfg.block_input == 'vec' else None
        self.vocab = None if self.vecs else TokenVocab(manifest.train, cfg)
        (out / 'vocab.json').write_text(
            json.dumps(self.vocab.summary() if self.vocab else {'block_vec': cfg.block_vec, 'd': self.vecs.d}, indent=2),
            encoding='utf-8')
        self.stack = stack_logits(cfg, manifest) if cfg.stack_tfidf else None
        self.imports = ImportVocab(manifest.train, cfg.min_df) if cfg.imports else None
        if self.imports is not None:
            (out / 'imports_vocab.json').write_text(json.dumps(self.imports.summary(), indent=2), encoding='utf-8')
        info.summary['vocab'] = self.vocab.summary() if self.vocab else cfg.block_vec
        return build_model(cfg, self.vocab, self.vecs.d if self.vecs else None,
                           len(self.imports) if self.imports is not None else 0)

    def loader(self, cfg: RunConfig, rows, train: bool):
        return loader(rows, self.vocab, cfg, train=train, vecs=self.vecs, stack=self.stack, imports=self.imports)

    def forward(self, model, batch):
        return model(batch)

    def variant_tokens(self, cfg: RunConfig):
        return tuple(cfg.variant.split('-'))
