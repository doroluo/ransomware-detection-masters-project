"""python -m pytest test_gnn_data.py: the GNN data path (vocab idf and MASK, whole-function segments, edge types,
controls, masking, block-budget batching) and the unchanged transformer path.

The cache is config.CACHE (override with the GT_CACHE environment variable). Tests marked [real] need a cache with
edge_kind/edge_back (v2) and are skipped on v1; the [synthetic] twins run on any cache.
"""
import importlib.util
import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer

import data
from config import RunConfig
from data import (BlockBudgetSampler, EpochSampler, TokenVocab, collate, encode_binary, keep_functions, load_index,
                  load_sample, loader, make_manifest, ranges)

if os.environ.get('GT_CACHE'):
    data.CACHE = Path(os.environ['GT_CACHE'])

HERE = Path(__file__).parent
BASE_COMMIT = 'fe7f581'


def gnn(**kw):
    return RunConfig(model='gnn', num_workers=0, **kw)


def load_real(sample_id):
    return load_sample(sample_id)


@pytest.fixture(scope='session')
def index():
    return load_index()


@pytest.fixture(scope='session')
def train_rows(index):
    return make_manifest(RunConfig(), index).train


@pytest.fixture(scope='session')
def vocab(train_rows):
    return TokenVocab(train_rows[:100], gnn())


@pytest.fixture(scope='session')
def real_ids(index):
    rows = sorted((r for r in index if r['arch'] == 'x86' and r['split'] == 'train'
                   and 300 <= int(r['num_blocks']) <= 8000 and int(r['num_edges']) > 50),
                  key=lambda r: r['sample_id'])
    return [r['sample_id'] for r in rows[:4]]


def synth_binary(rng, vocab, n_fn=40):
    sizes = rng.integers(0, 12, n_fn)
    sizes[[3, 17]] = 0
    sizes[5] = 30
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    n_blk = int(ptr[-1])
    n_ins, n_api = rng.integers(0, 7, n_blk), rng.integers(0, 4, n_blk)
    n_ins[::9] = 0
    n_ins_total, n_api_total = int(n_ins.sum()), int(n_api.sum())
    edges, calls = [], []
    for f in range(n_fn):
        if sizes[f] >= 2:
            k = int(rng.integers(sizes[f], 3 * sizes[f]))
            edges.append(ptr[f] + rng.integers(0, sizes[f], (k, 2)))
    for _ in range(80):
        calls.append(rng.integers(0, n_fn, 2))
    calls += calls[:5]
    cfg_edges = np.concatenate(edges)
    return {
        'mn': rng.integers(0, len(vocab.lut['mn']), n_ins_total).astype(np.int32),
        'sig': rng.integers(0, len(vocab.lut['sig']), n_ins_total).astype(np.int32),
        'beh': rng.integers(0, len(vocab.lut['beh']), n_ins_total).astype(np.int8),
        'api': rng.integers(0, len(vocab.lut['api']), n_api_total).astype(np.int32),
        'blk_insn_ptr': np.concatenate([[0], np.cumsum(n_ins)]).astype(np.int32),
        'blk_api_ptr': np.concatenate([[0], np.cumsum(n_api)]).astype(np.int32),
        'blk_feat': (rng.random((n_blk, 6)) + 0.1).astype(np.float16),
        'fn_blk_ptr': ptr.astype(np.int32), 'fn_flags': rng.integers(0, 4, n_fn).astype(np.int8),
        'cfg_edges': cfg_edges.astype(np.int32), 'call_edges': np.array(calls, dtype=np.int32),
        'edge_kind': rng.integers(0, 6, len(cfg_edges)).astype(np.int8),
        'edge_back': rng.integers(0, 2, len(cfg_edges)).astype(np.int8),
    }


@pytest.fixture(params=['synthetic', 'real'])
def binaries(request, vocab, real_ids):
    """[(z, sample_id)] for four binaries."""
    if request.param == 'synthetic':
        rng = np.random.default_rng(7)
        return [(synth_binary(rng, vocab), f'synthetic{i}') for i in range(4)]
    out = [(load_real(i), i) for i in real_ids]
    if 'edge_kind' not in out[0][0]:
        pytest.skip('cache has no edge_kind/edge_back (v1); set config.CACHE to the v2 cache')
    return out


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


# criterion 1: transformer path unchanged

@pytest.fixture(scope='module')
def old_data(tmp_path_factory):
    src = subprocess.run(['git', 'show', f'{BASE_COMMIT}:CT_GAT/graph_transformer/data.py'], cwd=HERE, check=True,
                         capture_output=True).stdout
    path = tmp_path_factory.mktemp('old') / 'data_old.py'
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location('data_old', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.CACHE = data.CACHE
    return mod


@pytest.mark.parametrize('kw', [dict(), dict(max_segment_blocks=16, max_segments=24)], ids=['default', 'capped'])
def test_transformer_path_unchanged(old_data, train_rows, real_ids, kw):
    cfg = RunConfig(num_workers=0, **kw)
    old_vocab, new_vocab = old_data.TokenVocab(train_rows[:100], cfg.min_df), TokenVocab(train_rows[:100], cfg)
    assert new_vocab.total == old_vocab.total
    assert new_vocab.size == old_vocab.size and new_vocab.offset == old_vocab.offset
    assert new_vocab.idf is None and new_vocab.mask_id is None
    for sid in real_ids:
        z = load_sample(sid)
        for epoch in (None, 3):
            rng = lambda: None if epoch is None else np.random.default_rng([cfg.seed, epoch, 5])
            old, new = old_data.encode_binary(z, old_vocab, cfg, rng()), encode_binary(z, new_vocab, cfg, rng(), sid)
            assert old.keys() == new.keys()
            for k in old:
                a, b = np.asarray(old[k]), np.asarray(new[k])
                assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), k
    assert ('edge_type' not in new) and ('tok_weights' not in new)


# criterion 2: vocab

def test_vocab_idf_and_mask(train_rows, vocab):
    rows = train_rows[:100]
    cfg = RunConfig()
    plain = TokenVocab(rows, cfg)
    assert vocab.total == plain.total + 4 and len(vocab.idf) == vocab.total and vocab.idf.dtype == np.float32
    assert vocab.size == plain.size and vocab.offset == plain.offset
    assert vocab.mask_id == {k: plain.total + j for j, k in enumerate(('mn', 'sig', 'beh', 'api'))}
    df = np.zeros(vocab.total, dtype=np.int64)
    for r in rows:
        z = load_sample(r['sample_id'])
        ids = [np.unique(vocab.lut[k][z[k]] + vocab.offset[k]) for k in ('mn', 'sig', 'beh', 'api')]
        df[np.concatenate(ids)] += 1
    want = np.log((1 + len(rows)) / (1 + df)) + 1
    for k in ('mn', 'sig', 'api'):
        want[vocab.offset[k]] = 1
    want[list(vocab.mask_id.values())] = 1
    assert np.allclose(vocab.idf, want, atol=1e-6)
    assert all(vocab.idf[vocab.offset[k]] == 1 for k in ('mn', 'sig', 'api'))
    assert all(vocab.idf[m] == 1 for m in vocab.mask_id.values())
    beh = slice(vocab.offset['beh'], vocab.offset['beh'] + vocab.size['beh'])
    assert df[beh].max() > 0 and np.allclose(vocab.idf[beh], want[beh], atol=1e-6)


def test_load_sample_keys(real_ids):
    z = load_real(real_ids[0])
    assert 'blk_rva' not in z
    if 'edge_kind' in z:
        assert len(z['edge_kind']) == len(z['edge_back']) == len(z['cfg_edges'])
    else:
        pytest.skip('cache has no edge_kind/edge_back (v1); the key check ran on v2 only')


# criteria 4-5

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


# criterion 4/6: collate

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


# criterion 7

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


# criterion 8

def with_control(z, vocab, sid, control, **kw):
    cfg = gnn(control=control)
    return encode(z, vocab, cfg, sid, **kw)


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


# criterion 9

def mask_cfg(**kw):
    return gnn(**kw)


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


# criterion 10

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


def test_loaders(vocab, train_rows):
    plain = loader(train_rows[:6], vocab, RunConfig(num_workers=0), train=True)
    assert plain.batch_size == 4 and isinstance(plain.sampler, EpochSampler)
    g = loader(train_rows[:6], vocab, gnn(batch_blocks=1), train=False)
    assert g.batch_size is None and isinstance(g.batch_sampler, BlockBudgetSampler)
    assert [len(b) for b in g.batch_sampler] == [1] * 6


def test_gnn_loader_end_to_end(vocab, train_rows):
    rows = train_rows[:6]
    if 'edge_kind' not in load_real(rows[0]['sample_id']):
        pytest.skip('cache has no edge_kind/edge_back (v1); set config.CACHE to the v2 cache')
    cfg = gnn(batch_blocks=5000, mask_tokens=0.15, control='shuffle')
    batches = list(loader(rows, vocab, cfg, train=True))
    assert sorted(s for b in batches for s in b['sample_id']) == sorted(r['sample_id'] for r in rows)
    for b in batches:
        assert b['tok_weights'].shape == b['tokens'].shape and b['bin_feat'].shape == (len(b['sample_id']), 5)
        assert b['n_fn'] == len(b['seg_len'])
