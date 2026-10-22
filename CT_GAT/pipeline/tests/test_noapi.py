"""The `noapi` vocabulary (the api-token ablation) through the GNN: names, the vocabulary and block tokens without the api kind,
masking, and the network on a batch of it.

Ported from the 3-block-level-counting branch (tests/test_noapi.py); the baseline, queue and gate parts of that file belong to
s04_evaluation and the experiment queue.
"""
import sys

import numpy as np
import pytest
import torch

from common.config import VOCABS, RunConfig
from common.manifests import load_sample
from experiments.train import parse_config
from s02_preprocess.tokenizer.block_tokens import token_table
from s02_preprocess.tokenizer.vocab import VOCAB_INSN_KINDS, VOCAB_KINDS, TokenVocab, uses_reader
from reference.gnn.batching import encode_binary, loader
from reference.gnn.spec import build_model
from tiny_cache import tiny  # noqa: F401  (a fixture)


def gnn(**kw):
    return RunConfig(**{'model': 'gnn', 'num_workers': 0, **kw})


def fold(f=0, **kw):
    return gnn(protocol='kfold', fold=f, **kw)


def api_free(tokens, kind, tok_offsets, per_block, n_kinds):
    """(tokens, kind) of an `ours` table without its api tokens, and the offsets / sizes of the blocks that remain."""
    keep = kind != n_kinds
    block_of = np.repeat(np.arange(len(per_block)), per_block)
    sizes = per_block - np.bincount(block_of[~keep], minlength=len(per_block))
    return tokens[keep], kind[keep], np.cumsum(sizes) - sizes, sizes


def test_noapi_is_a_vocab_with_its_own_suffix():
    assert 'noapi' in VOCABS and VOCABS[:3] == ('ours', 'yanping', 'hybrid')
    assert fold(0, mask_blocks=0.15, vocab='noapi').name == 'kfold0_x86_gnn-mblk15-voc-noapi_s0'
    assert fold(2, vocab='noapi', seed=1).name == 'kfold2_x86_gnn-voc-noapi_s1'
    assert fold(0, mask_blocks=0.15, vocab='noapi').variant == 'gnn-mblk15-voc-noapi'
    assert fold(0, vocab='noapi', imports='header').variant == 'gnn-voc-noapi-imph'      # header imports stay a separate switch
    for bad in (dict(model='transformer', vocab='noapi'), dict(vocab='noapi', block_input='vec', block_vec='x')):
        with pytest.raises(ValueError):
            RunConfig(protocol='kfold', fold=0, **bad)
    cfg, _ = parse_config(['--protocol', 'kfold', '--fold', '0', '--model', 'gnn', '--vocab', 'noapi'])
    assert cfg.vocab == 'noapi'


def test_vocabulary_kinds_are_explicit_and_the_old_ones_unchanged():
    assert VOCAB_KINDS == {'ours': ('mn', 'sig', 'beh', 'api'), 'noapi': ('mn', 'sig', 'beh'),
                           'yanping': ('y0', 'y1', 'y2', 'api'), 'hybrid': ('mn', 'beh', 'y1', 'y2', 'api')}
    assert VOCAB_INSN_KINDS['noapi'] == VOCAB_INSN_KINDS['ours'] == ('mn', 'sig', 'beh')
    assert [uses_reader(v) for v in ('ours', 'noapi', 'yanping', 'hybrid')] == [False, False, True, True]


def train_rows(tiny, n=30):  # noqa: F811
    return tiny.rows[:n]


def test_noapi_vocabulary_is_ours_without_the_api_kind(tiny, monkeypatch):  # noqa: F811
    rows = train_rows(tiny)
    ours = TokenVocab(rows, fold(0))
    monkeypatch.setitem(sys.modules, 'experiments.side_arrays.import_reader_side', None)       # importing it would raise
    new = TokenVocab(rows, fold(0, vocab='noapi'))
    assert new.kinds == new.insn_kinds == ('mn', 'sig', 'beh') and new.has_api is False and ours.has_api is True
    assert ours.kinds == ('mn', 'sig', 'beh', 'api') and ours.insn_kinds == ('mn', 'sig', 'beh')
    for table in (new.lut, new.size, new.offset, new.mask_id):
        assert 'api' not in table
    for k in new.kinds:                                                     # the surviving kinds are cut and numbered alike
        assert (new.lut[k] == ours.lut[k]).all() and new.size[k] == ours.size[k] and new.offset[k] == ours.offset[k]
    assert new.total == ours.total - ours.size['api'] - 1 and len(new.idf) == new.total
    assert new.mask_id == {k: new.total - 3 + j for j, k in enumerate(new.kinds)}
    assert new.idf[-3:].tolist() == [1.0] * 3
    assert (new.idf[:ours.offset['api']] == ours.idf[:ours.offset['api']]).all()


def test_noapi_block_tokens_are_ours_without_the_api_tokens(tiny):  # noqa: F811
    rows = train_rows(tiny)
    ours, new = TokenVocab(rows, fold(0)), TokenVocab(rows, fold(0, vocab='noapi'))
    for sid in tiny.ids[30:34]:
        z = load_sample(sid)
        blocks = np.arange(0, len(z['blk_insn_ptr']) - 1)
        t_o, k_o, off_o, per_o, ins_o, api_o = token_table(z, ours, blocks)
        t_n, k_n, off_n, per_n, ins_n, api_n = token_table(z, new, blocks)
        want_t, want_k, want_off, want_per = api_free(t_o, k_o, off_o, per_o, 3)
        assert len(api_o) > 0 and len(api_n) == 0 and len(ins_n) == 3
        assert (t_n == want_t).all() and (k_n == want_k).all() and (off_n == want_off).all() and (per_n == want_per).all()
        assert k_n.max() < 3 and t_n.max() < new.total - 3                  # no api kind, no mask id


@pytest.mark.parametrize('mask', [dict(), dict(mask_tokens=0.5), dict(mask_blocks=0.5)])
def test_noapi_items_masking_and_network(tiny, mask):  # noqa: F811
    rows = train_rows(tiny)
    cfg = fold(0, vocab='noapi', batch_size=8, **mask)
    v = TokenVocab(rows, cfg)
    z = load_sample(tiny.ids[2])
    out = encode_binary(z, v, cfg, np.random.default_rng(3) if mask else None, 's')
    masked = out['tokens'] >= v.total - len(v.kinds)
    assert (out['tokens'] < v.total).all() and masked.any() == bool(mask)
    assert len(out['tok_weights']) == len(out['tokens'])
    batch = next(iter(loader(tiny.rows[10:18], v, cfg, train=False)))
    model = build_model(cfg, v)
    logits = model(batch)
    assert logits.shape == (8,) and torch.isfinite(logits).all()
    logits.sum().backward()
