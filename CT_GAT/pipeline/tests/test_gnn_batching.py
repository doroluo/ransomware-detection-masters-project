"""The GNN's data path: `batching.encode_binary`, `collate`, the block-budget sampler and the loaders (whole-function segments,
edge types, controls, masking, edge augmentation, the per-binary block cap, the count weightings).

Ported from the 3-block-level-counting branch (tests/test_gnn_data.py); the manifest, vocabulary and subgraph tests of that file are
in test_gnn_data.py. Tests marked `real_data` read the cache of the machine (datasets.json, `--cache <name>` or GT_CACHE) and skip
without it; the others run on arrays made in test_gnn_data.py or on the synthetic tiny cache (tests/tiny_cache.py). Parametrized
tests run on both sources of binaries. `test_tfidf_and_mean_weights_are_bit_identical_to_before` compared the weights with the
code of an older commit; test_gnn_equivalence.py now holds the recorded outputs of the branch.
"""
from collections import Counter

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfTransformer

from common import manifests
from common.config import COMBINED_CACHE
from common.manifests import load_index, load_sample, make_manifest
from s02_preprocess import subgraph
from s02_preprocess.tokenizer import vocab as vocab_module
from s02_preprocess.tokenizer.block_tokens import ranges
from s02_preprocess.tokenizer.vocab import TokenVocab
from s03b_cfg_feature.gnn.batching import BlockBudgetSampler, EpochSampler, collate, encode_binary, keep_functions, loader
from s03b_cfg_feature.gnn.s05_block_counting.weights import token_weights
from test_gnn_data import (gnn, index, needs_combined, real, real_data, real_ids, real_vocab, small_cap,  # noqa: F401
                           synth_binary, train_rows)
from test_gnn_data import synth as synth_of
from tiny_cache import tiny  # noqa: F401  (a fixture)


def synth(vocab=None, seed=7):
    return synth_of(seed)


@pytest.fixture(params=['synthetic', pytest.param('real', marks=real_data)])
def source(request):
    return request.param


@pytest.fixture
def binaries(source, request, tiny):  # noqa: F811
    """[(z, sample_id)] for four binaries."""
    if source == 'synthetic':
        rng = np.random.default_rng(7)
        return [(synth_binary(rng), f'synthetic{i}') for i in range(4)]
    request.getfixturevalue('real')
    out = [(load_sample(i), i) for i in request.getfixturevalue('real_ids')]
    if 'edge_kind' not in out[0][0]:
        pytest.skip('cache has no edge_kind/edge_back (v1); point the cache at the v2 cache')
    return out


@pytest.fixture
def vocab(request, tiny):  # noqa: F811
    """The vocabulary of the binaries of the test: the real one next to real binaries, else a tiny-cache one."""
    if 'source' in request.fixturenames and request.getfixturevalue('source') == 'real':
        request.getfixturevalue('real')
        return request.getfixturevalue('real_vocab')
    return TokenVocab(tiny.rows[:30], gnn())


def encode(z, vocab, cfg, sample_id='s', epoch=None, i=0):
    rng = None if epoch is None else np.random.default_rng([cfg.seed, epoch, i])
    return encode_binary(z, vocab, cfg, rng, sample_id)


def global_blocks(out):
    """Batch-global (here: per-binary) block index of each edge endpoint, per the contract."""
    start = np.cumsum(out['seg_len']) - out['seg_len']
    return start[out['edge_seg']] + out['edge_src'], start[out['edge_seg']] + out['edge_dst']


def per_segment_edges(out):
    return [Counter(zip(out['edge_src'][out['edge_seg'] == s].tolist(), out['edge_dst'][out['edge_seg'] == s].tolist()))
            for s in range(len(out['seg_len']))]


def with_control(z, vocab, sid, control, **kw):
    cfg = gnn(control=control)
    return encode(z, vocab, cfg, sid, **kw)


def mask_cfg(**kw):
    return gnn(**kw)


def edge_multiset(out):
    return Counter(zip(*(out[k].tolist() for k in EDGE_KEYS)))


def edge_epochs(n_edges):
    return range(1, 1 + min(60, -(-30000 // max(n_edges, 1))))


def fn_sizes(z):
    return np.diff(z['fn_blk_ptr'].astype(np.int64))


def random_bags(seed, n_blk=300):
    rng = np.random.default_rng(seed)
    per_block = rng.integers(0, 14, n_blk)
    return rng.integers(0, 40, per_block.sum()).astype(np.int64), per_block


def bags_of(out):
    return out['tokens'], np.diff(np.append(out['tok_offsets'], len(out['tokens'])))


EDGE_KEYS = ('edge_seg', 'edge_src', 'edge_dst', 'edge_type')


def test_whole_function_segments(binaries, vocab):
    cfg = gnn()
    for z, sid in binaries:
        out = encode(z, vocab, cfg, sid)
        sizes = np.diff(z['fn_blk_ptr'].astype(np.int64))
        nonempty = np.flatnonzero(sizes > 0)
        assert out['n_fn'] == len(out['seg_len']) == len(nonempty) == out['n_segments_total']
        assert np.array_equal(out['seg_fn'], np.arange(out['n_fn']))
        assert np.array_equal(out['seg_len'], sizes[nonempty]) and out['seg_len'].sum() == sizes.sum()
        assert np.all(out['blk_pos'] < out['seg_len'][out['blk_seg']])
        assert len(out['tok_offsets']) == len(out['blk_feat']) == len(out['blk_seg']) == sizes.sum()
        orig_blk = ranges(z['fn_blk_ptr'][nonempty], sizes[nonempty])
        src, dst = global_blocks(out)
        got = Counter(zip(orig_blk[src].tolist(), orig_blk[dst].tolist(), out['edge_type'].tolist()))
        e = z['cfg_edges'].astype(np.int64)
        want = Counter(zip(e[:, 0].tolist(), e[:, 1].tolist(),
                           (z['edge_kind'].astype(np.int64) * 2 + z['edge_back']).tolist()))
        assert got == want and out['edge_type'].max() < 12
        c = z['call_edges'].astype(np.int64)
        c = c[(c[:, 0] != c[:, 1]) & (sizes[c[:, 0]] > 0) & (sizes[c[:, 1]] > 0)]
        local = np.cumsum(sizes > 0) - 1
        want_calls = set(zip(local[c[:, 0]].tolist(), local[c[:, 1]].tolist()))
        got_calls = list(zip(out['call_src'].tolist(), out['call_dst'].tolist()))
        assert len(got_calls) == len(set(got_calls)) and set(got_calls) == want_calls
        assert np.allclose(out['bin_feat'], np.log1p([len(z['blk_feat']), len(sizes), len(z['mn']), len(e),
                                                      len(z['call_edges'])]))
        assert out['tokens'].max() < vocab.total and out['tok_weights'].dtype == np.float32


def test_collate_shifts(binaries, vocab):
    cfg = gnn()
    a, b = (encode(z, vocab, cfg, sid) for z, sid in binaries[:2])
    items = []
    for out, (z, sid) in zip((a, b), binaries[:2]):
        items.append({**out, 'label': 1, 'sample_id': sid})
    batch = collate(items)
    S_a, T_a, B_a = len(a['seg_len']), len(a['tokens']), len(a['blk_seg'])
    cat = lambda k, shift=0: np.concatenate([a[k], b[k] + shift])
    for key, shift in (('edge_seg', S_a), ('call_src', S_a), ('call_dst', S_a), ('blk_seg', S_a), ('seg_fn', S_a),
                       ('tok_offsets', T_a)):
        assert np.array_equal(batch[key].numpy(), cat(key, shift)), key
    assert np.array_equal(batch['seg_bin'].numpy(), np.repeat([0, 1], [S_a, len(b['seg_len'])]))
    assert batch['n_fn'] == S_a + b['n_fn']
    for key in ('edge_type', 'edge_src', 'edge_dst', 'tokens', 'blk_pos', 'seg_len'):
        assert np.array_equal(batch[key].numpy(), cat(key)), key
    assert np.array_equal(batch['tok_weights'].numpy(), cat('tok_weights')) and batch['tok_weights'].dtype.is_floating_point
    assert np.array_equal(batch['bin_feat'].numpy(), np.stack([a['bin_feat'], b['bin_feat']]))
    assert batch['bin_feat'].shape == (2, 5) and batch['edge_type'].dtype == batch['edge_seg'].dtype
    start = np.cumsum(batch['seg_len'].numpy()) - batch['seg_len'].numpy()
    ga, gb = global_blocks(a), global_blocks(b)
    assert np.array_equal(start[batch['edge_seg'].numpy()] + batch['edge_src'].numpy(), np.concatenate([ga[0], gb[0] + B_a]))
    assert batch['n_segments_total'] == [a['n_segments_total'], b['n_segments_total']]


def test_tfidf_weights_match_sklearn(binaries, vocab):
    z, sid = binaries[0]
    out = encode(z, vocab, gnn(), sid)
    n_blk = len(out['tok_offsets'])
    per_block = np.diff(np.append(out['tok_offsets'], len(out['tokens'])))
    blk_of = np.repeat(np.arange(n_blk), per_block)
    shape = (n_blk, vocab.total)
    counts = sp.csr_matrix((np.ones(len(blk_of)), (blk_of, out['tokens'])), shape=shape)
    tf = TfidfTransformer(sublinear_tf=True, norm='l2').fit(np.ones((2, vocab.total)))
    tf.idf_ = vocab.idf.astype(np.float64)
    want = tf.transform(counts)
    got = sp.csr_matrix((out['tok_weights'].astype(np.float64), (blk_of, out['tokens'])), shape=shape)
    assert abs(got - want).max() < 1e-5
    assert (per_block > 0).sum() > 100
    mean = encode(z, vocab, gnn(encoder='mean'), sid)
    sums = np.bincount(blk_of, weights=mean['tok_weights'], minlength=n_blk)
    assert np.allclose(sums[per_block > 0], 1, atol=1e-6) and np.all(sums[per_block == 0] == 0)
    assert np.array_equal(mean['tokens'], out['tokens'])


def test_controls(binaries, vocab):
    z, sid = max(binaries, key=lambda b: len(b[0]['cfg_edges']))
    assert len(z['cfg_edges']) > 50
    real = encode(z, vocab, gnn(), sid)
    n_seg = len(real['seg_len'])
    real_src, real_dst = real['edge_src'], real['edge_dst']
    assert np.array_equal(with_control(z, vocab, sid, '', epoch=None)['edge_dst'], real_dst)

    shuf = with_control(z, vocab, sid, 'shuffle')
    assert np.array_equal(shuf['edge_seg'], real['edge_seg']) and np.array_equal(shuf['edge_src'], real['edge_src'])
    assert not shuf['edge_type'].any() and len(shuf['edge_dst']) == len(real_dst)
    assert np.array_equal(np.bincount(real['edge_seg'] * 10**6 + real_src), np.bincount(shuf['edge_seg'] * 10**6 + shuf['edge_src']))
    for s in range(n_seg):
        m = real['edge_seg'] == s
        assert np.array_equal(np.sort(real_dst[m]), np.sort(shuf['edge_dst'][m])), s
    assert (shuf['edge_dst'] != real_dst).any()
    assert np.array_equal(shuf['call_src'], real['call_src'])

    chain = with_control(z, vocab, sid, 'chain')
    want = [Counter((i, i + 1) for i in range(int(L) - 1)) for L in real['seg_len']]
    assert per_segment_edges(chain) == want and not chain['edge_type'].any()

    perm = with_control(z, vocab, sid, 'permute_types')
    assert np.array_equal(perm['edge_src'], real_src) and np.array_equal(perm['edge_dst'], real_dst)
    for s in range(n_seg):
        m = real['edge_seg'] == s
        assert np.array_equal(np.sort(real['edge_type'][m]), np.sort(perm['edge_type'][m])), s
    assert (perm['edge_type'] != real['edge_type']).any()

    for control in ('shuffle', 'chain', 'permute_types'):
        a, b = with_control(z, vocab, sid, control), with_control(z, vocab, sid, control)
        assert all(np.array_equal(a[k], b[k]) for k in ('edge_seg', 'edge_src', 'edge_dst', 'edge_type')), control
        t1 = with_control(z, vocab, sid, control, epoch=1)
        t1_again = with_control(z, vocab, sid, control, epoch=1)
        t2 = with_control(z, vocab, sid, control, epoch=2)
        assert all(np.array_equal(t1[k], t1_again[k]) for k in ('edge_dst', 'edge_type'))
        differs = any(not np.array_equal(t1[k], t2[k]) for k in ('edge_dst', 'edge_type'))
        assert differs == (control != 'chain'), control
    other = with_control(z, vocab, 'another-sample', 'shuffle')
    assert (other['edge_dst'] != shuf['edge_dst']).any()


def test_mask_rates_realised(binaries, vocab):
    z, sid = max(binaries, key=lambda b: len(b[0]['blk_feat']))
    sizes = np.diff(z['fn_blk_ptr'].astype(np.int64))
    n_ins, n_api, n_blk = len(z['mn']), len(z['api']), len(z['blk_feat'])
    clean = encode(z, vocab, gnn(), sid)
    epochs = range(1, 13)

    cfg = mask_cfg(mask_tokens=0.15)
    mn_hit = sig_hit = beh_hit = api_hit = 0
    for epoch in epochs:
        out = encode(z, vocab, cfg, sid, epoch)
        mn_hit += (out['tokens'] == vocab.mask_id['mn']).sum()
        sig_hit += (out['tokens'] == vocab.mask_id['sig']).sum()
        beh_hit += (out['tokens'] == vocab.mask_id['beh']).sum()
        api_hit += (out['tokens'] == vocab.mask_id['api']).sum()
        assert np.array_equal(out['tok_offsets'], clean['tok_offsets']) and np.array_equal(out['edge_dst'], clean['edge_dst'])
        assert np.array_equal(out['blk_feat'], clean['blk_feat'])
    assert mn_hit == sig_hit == beh_hit
    assert abs(mn_hit / (n_ins * len(epochs)) - 0.15) < 0.02
    assert abs(api_hit / (n_api * len(epochs)) - 0.15) < 0.02

    cfg = mask_cfg(mask_blocks=0.15)
    hit = tot = zeroed = 0
    for epoch in epochs:
        out = encode(z, vocab, cfg, sid, epoch)
        per_block = np.diff(np.append(out['tok_offsets'], len(out['tokens'])))
        is_mask = np.isin(out['tokens'], list(vocab.mask_id.values()))
        masked = np.add.reduceat(is_mask.astype(np.int64), out['tok_offsets'][per_block > 0]) == per_block[per_block > 0]
        hit += masked.sum()
        tot += len(masked)
        zero_rows = ~out['blk_feat'].any(1)
        zeroed += (zero_rows[per_block > 0] & masked).sum()
        assert np.array_equal(out['edge_dst'], clean['edge_dst']) and np.array_equal(out['edge_type'], clean['edge_type'])
        assert np.array_equal(out['call_src'], clean['call_src'])
    assert abs(hit / tot - 0.15) < 0.02 and zeroed == hit

    cfg = mask_cfg(drop_functions=0.15)
    kept = 0
    for epoch in epochs:
        out = encode(z, vocab, cfg, sid, epoch)
        kept += out['n_fn']
        assert out['n_fn'] >= 1 and out['n_segments_total'] == (sizes > 0).sum()
        assert out['edge_seg'].max() < out['n_fn'] and out['call_src'].max(initial=0) < out['n_fn']
        assert len(out['blk_seg']) == out['seg_len'].sum()
    assert abs(1 - kept / ((sizes > 0).sum() * len(epochs)) - 0.15) < 0.02

    cfg = mask_cfg(drop_library=0.4, drop_functions=0.05)
    flags = z['fn_flags']
    library = (flags & 3) > 0
    assert library.sum() > 30
    kept_lib = kept_rest = draws = 0
    for epoch in range(1, 201):
        keep = keep_functions(flags, sizes, cfg, np.random.default_rng([0, epoch, 1]))
        kept_lib += (keep & library & (sizes > 0)).sum()
        kept_rest += (keep & ~library & (sizes > 0)).sum()
        draws += 1
    assert abs(1 - kept_lib / (draws * (library & (sizes > 0)).sum()) - 0.4) < 0.02
    assert abs(1 - kept_rest / (draws * (~library & (sizes > 0)).sum()) - 0.05) < 0.02
    kept = np.mean([encode(z, vocab, cfg, sid, e)['n_fn'] for e in range(1, 101)])
    expected = np.where(library, 0.6, 0.95)[sizes > 0].sum()
    assert abs(kept - expected) / expected < 0.02

    cfg = mask_cfg(drop_functions=0.15, drop_library=0.4, mask_tokens=0.15, mask_blocks=0.15)
    assert not any(np.isin(clean['tokens'], list(vocab.mask_id.values())))
    out = encode(z, vocab, cfg, sid, None)
    assert all(np.array_equal(out[k], clean[k]) for k in clean if k not in ('n_fn', 'n_segments_total'))


def test_at_least_one_function_remains(binaries, vocab):
    cfg = mask_cfg(drop_functions=0.99, drop_library=0.99)
    for z, sid in binaries:
        for epoch in range(1, 30):
            assert encode(z, vocab, cfg, sid, epoch)['n_fn'] >= 1
    z = binaries[0][0]
    few = {**z, 'fn_blk_ptr': np.array([0, 3, 3, 5, 9], dtype=np.int32), 'fn_flags': np.array([1, 1, 0, 3], dtype=np.int8)}
    few['blk_feat'], few['blk_insn_ptr'] = z['blk_feat'][:9], z['blk_insn_ptr'][:10]
    few['blk_api_ptr'] = z['blk_api_ptr'][:10]
    few['cfg_edges'], few['edge_kind'], few['edge_back'] = z['cfg_edges'][:0], z['edge_kind'][:0], z['edge_back'][:0]
    few['call_edges'] = np.array([[0, 2], [3, 0]], dtype=np.int32)
    for epoch in range(1, 100):
        out = encode(few, vocab, cfg, 'few', epoch)
        assert out['n_fn'] >= 1 and len(out['seg_len']) == out['n_fn'] and len(out['blk_seg']) == out['seg_len'].sum()


@real_data
def test_block_budget_batching(train_rows):
    blocks = [int(r['num_blocks']) for r in train_rows]
    for batch_size, budget in ((4, 300000), (4, 6000), (6, 1000)):
        assert max(blocks) > budget or budget == 300000
        sampler = EpochSampler(len(blocks), 0, True)
        batches = BlockBudgetSampler(sampler, blocks, batch_size, budget)
        orders = []
        for _ in range(2):
            got = list(batches)
            flat = [i for b in got for i, _ in b]
            assert sorted(flat) == list(range(len(blocks)))
            orders.append(flat)
            for b in got:
                total = sum(blocks[i] for i, _ in b)
                assert len(b) <= batch_size and (total <= budget or len(b) == 1)
            for prev, nxt in zip(got, got[1:]):
                assert len(prev) == batch_size or sum(blocks[i] for i, _ in prev) + blocks[nxt[0][0]] > budget
            assert [i for b in got for i, _ in b] == [i for b in got for i, _ in b]
        assert orders[0] != orders[1]
    over = [i for i, n in enumerate(blocks) if n > 6000]
    assert over
    sampler = EpochSampler(len(blocks), 0, False)
    for b in BlockBudgetSampler(sampler, blocks, 4, 6000):
        if any(i in over for i, _ in b):
            assert len(b) == 1
    ordered = [i for b in BlockBudgetSampler(sampler, blocks, 4, 300000) for i, _ in b]
    assert ordered == list(range(len(blocks)))


def test_loaders(vocab, tiny):  # noqa: F811
    g = loader(tiny.rows[:6], vocab, gnn(batch_blocks=1), train=False)
    assert g.batch_size is None and isinstance(g.batch_sampler, BlockBudgetSampler)
    assert [len(b) for b in g.batch_sampler] == [1] * 6


def test_gnn_loader_end_to_end(vocab, tiny):  # noqa: F811
    rows = tiny.rows[:6]
    cfg = gnn(batch_blocks=5000, mask_tokens=0.15, control='shuffle')
    batches = list(loader(rows, vocab, cfg, train=True))
    assert sorted(s for b in batches for s in b['sample_id']) == sorted(r['sample_id'] for r in rows)
    for b in batches:
        assert b['tok_weights'].shape == b['tokens'].shape and b['bin_feat'].shape == (len(b['sample_id']), 5)
        assert b['n_fn'] == len(b['seg_len'])


def test_drop_and_flip_rates(binaries, vocab):
    z, sid = max(binaries, key=lambda b: len(b[0]['cfg_edges']))
    clean = encode(z, vocab, gnn(), sid)
    n = len(clean['edge_seg'])
    assert n > 50
    epochs = edge_epochs(n)
    kept = flipped = loops = 0
    for epoch in epochs:
        out = encode(z, vocab, gnn(drop_edges=0.2), sid, epoch)
        assert not (edge_multiset(out) - edge_multiset(clean))
        kept += len(out['edge_seg'])
        assert all(np.array_equal(out[k], clean[k]) for k in ('tokens', 'seg_len', 'call_src', 'call_dst'))
        out = encode(z, vocab, gnn(flip_edges=0.2), sid, epoch)
        assert len(out['edge_seg']) == n and np.array_equal(out['edge_seg'], clean['edge_seg'])
        assert np.array_equal(out['edge_type'], clean['edge_type'])
        assert Counter(out['edge_type'].tolist()) == Counter(clean['edge_type'].tolist())
        move = out['edge_src'] != clean['edge_src']
        assert np.array_equal(out['edge_src'][move], clean['edge_dst'][move])
        assert np.array_equal(out['edge_dst'][move], clean['edge_src'][move])
        flipped += int(move.sum())
        loops += int((clean['edge_src'] == clean['edge_dst']).sum())
    assert abs(kept / (n * len(epochs)) - 0.8) < 0.02
    assert abs(flipped / (n * len(epochs) - loops) - 0.2) < 0.02


def test_edge_augmentation_is_train_only_and_seeded(binaries, vocab):
    z, sid = max(binaries, key=lambda b: len(b[0]['cfg_edges']))
    clean = encode(z, vocab, gnn(), sid)
    cfg = gnn(drop_edges=0.3, flip_edges=0.3)
    ev = encode(z, vocab, cfg, sid)
    assert all(np.array_equal(ev[k], clean[k]) for k in EDGE_KEYS)
    a, b = encode(z, vocab, cfg, sid, epoch=1), encode(z, vocab, cfg, sid, epoch=1)
    c = encode(z, vocab, cfg, sid, epoch=2)
    assert all(np.array_equal(a[k], b[k]) for k in EDGE_KEYS)
    assert any(not np.array_equal(a[k], c[k]) for k in EDGE_KEYS)
    zero = encode(z, vocab, gnn(drop_edges=0.0, flip_edges=0.0), sid, epoch=1)
    assert all(np.array_equal(zero[k], encode(z, vocab, gnn(), sid, epoch=1)[k]) for k in EDGE_KEYS)


def test_edge_augmentation_after_controls(binaries, vocab):
    z, sid = max(binaries, key=lambda b: len(b[0]['cfg_edges']))
    chain = with_control(z, vocab, sid, 'chain')
    ch_edges = edge_multiset(chain)
    dropped = encode(z, vocab, gnn(control='chain', drop_edges=0.4), sid, epoch=1)
    assert 0 < len(dropped['edge_seg']) < len(chain['edge_seg']) and not (edge_multiset(dropped) - ch_edges)
    flipped = encode(z, vocab, gnn(control='chain', flip_edges=0.5), sid, epoch=1)
    assert len(flipped['edge_seg']) == len(chain['edge_seg']) and not flipped['edge_type'].any()
    assert np.abs(flipped['edge_dst'] - flipped['edge_src']).max() == 1
    assert (flipped['edge_dst'] < flipped['edge_src']).any() and (flipped['edge_dst'] > flipped['edge_src']).any()
    both = encode(z, vocab, gnn(control='shuffle', flip_edges=0.5, drop_edges=0.2), sid, epoch=1)
    assert not both['edge_type'].any() and 0 < len(both['edge_seg']) < len(encode(z, vocab, gnn(), sid)['edge_seg'])


def test_function_dropping_happens_before_the_cap(vocab, small_cap):
    z = synth(vocab)
    cfg = gnn(drop_functions=0.5)
    for epoch in (1, 2, 3):
        rng = np.random.default_rng([cfg.seed, epoch, 0])
        survivors = keep_functions(z['fn_flags'], fn_sizes(z), cfg, rng)
        out = encode(z, vocab, cfg, 's', epoch)
        assert out['n_blocks_kept'] <= small_cap
        assert out['n_blocks_kept'] <= fn_sizes(z)[survivors].sum()
        assert out['n_blocks_total'] == fn_sizes(z).sum()
        assert not (Counter(out['seg_len'].tolist()) - Counter(fn_sizes(z)[survivors].tolist()))


def test_bin_feat_counts_the_full_binary(vocab, small_cap, monkeypatch):
    z = synth(vocab)
    capped = encode(z, vocab, gnn(), 's')
    want = np.log1p([len(z['blk_feat']), len(z['fn_flags']), len(z['mn']), len(z['cfg_edges']),
                     len(z['call_edges'])]).astype(np.float32)
    assert np.array_equal(capped['bin_feat'], want)
    monkeypatch.setattr(subgraph, 'MAX_BINARY_BLOCKS', 10 ** 9)
    assert np.array_equal(encode(z, vocab, gnn(), 's')['bin_feat'], want)


def test_collate_and_loader_carry_the_block_counts(vocab, small_cap):
    z = synth(vocab)
    items = [{**encode(z, vocab, gnn(), 's'), 'label': 1, 'sample_id': 'a'}]
    batch = collate(items)
    assert batch['n_blocks_kept'] == [items[0]['n_blocks_kept']] and batch['n_blocks_total'] == [fn_sizes(z).sum()]
    rows = [{'sample_id': 'x', 'label': 0, 'num_blocks': '8000000'}, {'sample_id': 'y', 'label': 0, 'num_blocks': '10'}]
    sampler = loader(rows, vocab, gnn(), train=False).batch_sampler
    assert sampler.blocks == [small_cap, 10]


@real_data
def test_v2_batches_are_unchanged_by_the_cap(index, real, real_vocab, monkeypatch):
    vocab = real_vocab
    rows = sorted((r for r in index if int(r['num_blocks']) <= 500_000), key=lambda r: -int(r['num_blocks']))[:4]
    assert len(rows) == 4 and all(int(r['num_blocks']) <= subgraph.MAX_BINARY_BLOCKS for r in rows)
    if 'edge_kind' not in load_sample(rows[0]['sample_id']):
        pytest.skip('cache has no edge_kind/edge_back (v1)')
    cfgs = [gnn(), gnn(control='shuffle', mask_tokens=0.15, drop_functions=0.1, drop_edges=0.2)]
    numeric = lambda out: {k: v for k, v in out.items() if isinstance(v, np.ndarray)}
    for cfg in cfgs:
        for epoch in (None, 1):
            for i, r in enumerate(rows):
                z = load_sample(r['sample_id'])
                monkeypatch.setattr(subgraph, 'MAX_BINARY_BLOCKS', 500_000)
                capped = encode(z, vocab, cfg, r['sample_id'], epoch, i)
                monkeypatch.setattr(subgraph, 'MAX_BINARY_BLOCKS', 10 ** 9)
                free = encode(z, vocab, cfg, r['sample_id'], epoch, i)
                assert (capped['n_blocks_kept'], capped['n_blocks_total']) == (free['n_blocks_kept'],
                                                                               free['n_blocks_total'])
                assert cfg.drop_functions or capped['n_blocks_kept'] == capped['n_blocks_total']
                assert capped.keys() == free.keys()
                for k, v in numeric(capped).items():
                    assert np.array_equal(v, free[k]) and v.dtype == free[k].dtype, k
    for train in (False, True):
        monkeypatch.setattr(subgraph, 'MAX_BINARY_BLOCKS', 500_000)
        a = list(loader(rows, vocab, gnn(), train=train))
        monkeypatch.setattr(subgraph, 'MAX_BINARY_BLOCKS', 10 ** 9)
        b = list(loader(rows, vocab, gnn(), train=train))
        assert len(a) == len(b) and [x['sample_id'] for x in a] == [x['sample_id'] for x in b]
        for x, y in zip(a, b):
            for k, v in x.items():
                if torch.is_tensor(v):
                    assert torch.equal(v, y[k]) and v.dtype == y[k].dtype, k
                else:
                    assert v == y[k], k


@real_data
@needs_combined
def test_the_8m_block_goodware_fits_the_cap(monkeypatch):
    monkeypatch.setattr(manifests, 'CACHE', COMBINED_CACHE)
    monkeypatch.setattr(vocab_module, 'CACHE', COMBINED_CACHE)
    index = load_index(COMBINED_CACHE)
    big = max(index, key=lambda r: int(r['num_blocks']))
    assert big['sample_id'].startswith('a1a05c00ab707554') and int(big['num_blocks']) == 7_991_293
    cfg = gnn(protocol='kfold_v3', fold=0, arch='all')
    vocab = TokenVocab([r for r in make_manifest(cfg, index).train][:40], cfg)
    z = load_sample(big['sample_id'])
    out = encode_binary(z, vocab, cfg, None, big['sample_id'])
    assert 0 < out['n_blocks_kept'] <= 500_000 and out['n_blocks_total'] == 7_991_293
    assert out['n_blocks_kept'] > 400_000 and len(out['blk_seg']) == out['n_blocks_kept']
    assert out['bin_feat'][0] == np.float32(np.log1p(7_991_293))
    assert out['call_src'].size == 0 or out['call_src'].max() < out['n_fn']


def test_encode_binary_uses_the_policy_and_budget(vocab):
    z = synth_binary(np.random.default_rng(5), n_fn=400)
    total = int(fn_sizes(z).sum())
    assert total > 1500
    cfg = gnn(budget=1000)
    for epoch in (None, 1):
        rng = lambda: None if epoch is None else np.random.default_rng([0, epoch, 0])
        out = encode_binary(z, vocab, cfg, rng(), 's')
        keep = keep_functions(z['fn_flags'], fn_sizes(z), cfg, rng())
        fn, blk = subgraph.select_subgraph(z, cfg.subgraph, cfg.budget, rng(), 's', keep=keep)
        assert out['n_blocks_kept'] == len(blk) == len(out['blk_seg']) <= 1000
        assert out['n_blocks_total'] == total
        assert np.array_equal(out['seg_len'], fn_sizes(z)[fn]) and out['n_fn'] == len(fn)


def test_encode_binary_with_a_budget_equals_the_global_cap(vocab, monkeypatch):
    z = synth_binary(np.random.default_rng(5), n_fn=400)
    via_budget = encode_binary(z, vocab, gnn(budget=1000), None, 's')
    monkeypatch.setattr(subgraph, 'MAX_BINARY_BLOCKS', 1000)
    via_cap = encode_binary(z, vocab, gnn(), None, 's')
    assert via_budget.keys() == via_cap.keys()
    for k, v in via_budget.items():
        assert np.array_equal(v, via_cap[k]) if isinstance(v, np.ndarray) else v == via_cap[k], k


def test_loader_sizes_binaries_by_the_budget(vocab):
    rows = [{'sample_id': 'x', 'label': 0, 'num_blocks': '8000000'}, {'sample_id': 'y', 'label': 0, 'num_blocks': '10'},
            {'sample_id': 'z', 'label': 0, 'num_blocks': '30000'}]
    assert loader(rows, vocab, gnn(), train=False).batch_sampler.blocks == [500_000, 10, 30000]
    assert loader(rows, vocab, gnn(budget=20000), train=False).batch_sampler.blocks == [20000, 10, 20000]


def test_count_weightings(vocab, binaries):
    cases = [random_bags(s) for s in range(3)] + [bags_of(encode(z, vocab, gnn(), sid)) for z, sid in binaries]
    for tokens, per_block in cases:
        n_blk, blk_of = len(per_block), np.repeat(np.arange(len(per_block)), per_block)
        per = lambda w: np.bincount(blk_of, weights=w.astype(np.float64), minlength=n_blk)
        raw, pres, rawlog = (token_weights(tokens, per_block, vocab, e) for e in ('raw', 'presence', 'rawlog'))
        assert raw.dtype == pres.dtype == rawlog.dtype == np.float32
        distinct = np.array([len(set(tokens[blk_of == b].tolist())) for b in range(n_blk)])
        assert np.array_equal(per(raw), per_block) and set(np.unique(raw)) <= {1.0}
        assert np.array_equal(per(pres), distinct) and set(np.unique(pres)) <= {0.0, 1.0}
        for b in np.flatnonzero(per_block)[:60]:
            t = tokens[blk_of == b]
            for v in set(t.tolist()):
                c = int((t == v).sum())
                assert pres[blk_of == b][t == v].tolist() == [1.0] + [0.0] * (c - 1)
                assert abs(float(rawlog[blk_of == b][t == v].sum()) - np.log1p(c)) < 1e-5


def test_count_weightings_on_empty_and_single_token_blocks(vocab):
    tokens, per_block = np.array([5, 5, 5, 9], dtype=np.int64), np.array([0, 3, 0, 1])
    assert token_weights(tokens, per_block, vocab, 'raw').tolist() == [1, 1, 1, 1]
    assert token_weights(tokens, per_block, vocab, 'presence').tolist() == [1, 0, 0, 1]
    assert np.allclose(token_weights(tokens, per_block, vocab, 'rawlog'), [np.log(4) / 3] * 3 + [np.log(2)])
    nothing = np.zeros(0, dtype=np.int64)
    for enc in ('raw', 'presence', 'rawlog', 'mean', 'tfidf'):
        assert len(token_weights(nothing, np.zeros(3, dtype=np.int64), vocab, enc)) == 0


def test_encoder_option_reaches_the_batch(binaries, vocab):
    z, sid = binaries[0]
    base = encode(z, vocab, gnn(), sid)
    for enc in ('raw', 'presence', 'rawlog'):
        out = encode(z, vocab, gnn(encoder=enc), sid)
        assert np.array_equal(out['tokens'], base['tokens'])
        assert not np.array_equal(out['tok_weights'], base['tok_weights'])
    assert (encode(z, vocab, gnn(encoder='raw'), sid)['tok_weights'] == 1).all()
