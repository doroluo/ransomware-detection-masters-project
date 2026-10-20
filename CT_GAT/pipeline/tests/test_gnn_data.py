"""Manifests, the token vocabulary, subgraph selection and the chain measurement.

Ported from the 3-block-level-counting branch (tests/test_gnn_data.py). The tests that run a binary through
`batching.encode_binary` / `collate` / the loaders move with the model (S5-3). Tests marked `real_data` read the cache of the
machine (config.CACHE: datasets.json, `--cache <name>` or GT_CACHE) and skip without it; the others run on the synthetic
tiny cache (tests/tiny_cache.py) or on arrays made here.
"""
from pathlib import Path

import numpy as np
import pytest

from common import config, manifests
from common.config import COMBINED_CACHE, RunConfig
from common.manifests import load_index, load_sample, make_manifest
from s02_preprocess import subgraph
from s02_preprocess.tokenizer import vocab as vocab_module
from s02_preprocess.tokenizer.block_tokens import ranges, token_table
from s02_preprocess.tokenizer.vocab import OURS_INSN_KINDS, TOKEN_KINDS, VOCAB_KINDS, TokenVocab
from tiny_cache import N_API, N_BEH, N_MN, N_SIG, tiny  # noqa: F401  (tiny is a fixture)

real_data = pytest.mark.real_data
needs_combined = pytest.mark.skipif(not (Path(COMBINED_CACHE) / 'index.csv').exists(), reason='needs the combined cache')


def gnn(**kw):
    return RunConfig(model='gnn', num_workers=0, **kw)


REAL_CACHE = config.CACHE        # the cache of this machine; the `tiny` fixture points the module-level CACHEs elsewhere


@pytest.fixture
def real(monkeypatch):
    """The machine's cache in place of the tiny one, for the duration of a test."""
    monkeypatch.setattr(manifests, 'CACHE', REAL_CACHE)
    monkeypatch.setattr(vocab_module, 'CACHE', REAL_CACHE)
    return REAL_CACHE


@pytest.fixture(scope='session')
def index():
    return load_index(REAL_CACHE)


@pytest.fixture(scope='session')
def train_rows(index):
    return make_manifest(RunConfig(), index).train


@pytest.fixture(scope='session')
def real_vocab(train_rows):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(vocab_module, 'CACHE', REAL_CACHE)
        return TokenVocab(train_rows[:100], gnn())


@pytest.fixture(scope='session')
def real_ids(index):
    rows = sorted((r for r in index if r['arch'] == 'x86' and r['split'] == 'train'
                   and 300 <= int(r['num_blocks']) <= 8000 and int(r['num_edges']) > 50),
                  key=lambda r: r['sample_id'])
    return [r['sample_id'] for r in rows[:4]]


def synth_binary(rng, n_fn=40, n_mn=N_MN, n_sig=N_SIG, n_beh=N_BEH, n_api=N_API):
    sizes = rng.integers(0, 12, n_fn)
    sizes[[3, 17]] = 0
    sizes[5] = 30
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    n_blk = int(ptr[-1])
    n_ins, n_apis = rng.integers(0, 7, n_blk), rng.integers(0, 4, n_blk)
    n_ins[::9] = 0
    n_ins_total, n_api_total = int(n_ins.sum()), int(n_apis.sum())
    edges, calls = [], []
    for f in range(n_fn):
        if sizes[f] >= 2:
            k = int(rng.integers(sizes[f], 3 * sizes[f]))
            edges.append(ptr[f] + rng.integers(0, sizes[f], (k, 2)))
    for _ in range(80):
        calls.append(rng.integers(0, n_fn, 2))
    cfg_edges = np.concatenate(edges)
    return {
        'mn': rng.integers(0, n_mn, n_ins_total).astype(np.int32),
        'sig': rng.integers(0, n_sig, n_ins_total).astype(np.int32),
        'beh': rng.integers(0, n_beh, n_ins_total).astype(np.int8),
        'api': rng.integers(0, n_api, n_api_total).astype(np.int32),
        'blk_insn_ptr': np.concatenate([[0], np.cumsum(n_ins)]).astype(np.int32),
        'blk_api_ptr': np.concatenate([[0], np.cumsum(n_apis)]).astype(np.int32),
        'blk_feat': (rng.random((n_blk, 6)) + 0.1).astype(np.float16),
        'fn_blk_ptr': ptr.astype(np.int32), 'fn_flags': rng.integers(0, 4, n_fn).astype(np.int8),
        'cfg_edges': cfg_edges.astype(np.int32), 'call_edges': np.array(calls, dtype=np.int32),
        'edge_kind': rng.integers(0, 6, len(cfg_edges)).astype(np.int8),
        'edge_back': rng.integers(0, 2, len(cfg_edges)).astype(np.int8),
    }


def synth(seed=7, **kw):
    return synth_binary(np.random.default_rng(seed), **kw)


@pytest.fixture(params=['synthetic', pytest.param('real', marks=real_data)])
def binaries(request):
    """[(z, sample_id)] for four binaries."""
    if request.param == 'synthetic':
        rng = np.random.default_rng(7)
        return [(synth_binary(rng), f'synthetic{i}') for i in range(4)]
    request.getfixturevalue('real')
    out = [(load_sample(i), i) for i in request.getfixturevalue('real_ids')]
    if 'edge_kind' not in out[0][0]:
        pytest.skip('cache has no edge_kind/edge_back (v1); point the cache at the v2 cache')
    return out


# the vocabulary

def check_vocab_idf_and_mask(vocab, rows):
    base = sum(vocab.size.values())
    assert vocab.total == base + 4 and len(vocab.idf) == vocab.total and vocab.idf.dtype == np.float32
    assert vocab.mask_id == {k: base + j for j, k in enumerate(TOKEN_KINDS)}
    df = np.zeros(vocab.total, dtype=np.int64)
    for r in rows:
        z = load_sample(r['sample_id'])
        ids = [np.unique(vocab.lut[k][z[k]] + vocab.offset[k]) for k in TOKEN_KINDS]
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


@real_data
def test_vocab_idf_and_mask(train_rows, real_vocab, real):
    check_vocab_idf_and_mask(real_vocab, train_rows[:100])


def test_vocab_idf_and_mask_on_the_tiny_cache(tiny):  # noqa: F811
    rows = make_manifest(RunConfig(), tiny.rows).train
    check_vocab_idf_and_mask(TokenVocab(rows, gnn()), rows)


def test_vocab_cuts_tokens_seen_in_fewer_than_min_df_training_binaries(tiny):  # noqa: F811
    rows = tiny.rows
    cut = TokenVocab(rows, gnn(min_df=2))
    assert cut.lut['mn'][N_MN - 1] == 0 and cut.lut['sig'][N_SIG - 1] == 0         # in binary 0 only: df 1
    assert (cut.lut['mn'][:N_MN - 1] > 0).all()                                    # everything else is kept, ids 1..
    assert sorted(set(cut.lut['mn'][:N_MN - 1].tolist())) == list(range(1, N_MN))
    keep_all = TokenVocab(rows, gnn(min_df=1))
    assert keep_all.size['mn'] == cut.size['mn'] + 1 and keep_all.lut['mn'][N_MN - 1] > 0
    assert cut.lut['beh'].tolist() == list(range(N_BEH)) and cut.size['beh'] == N_BEH       # beh keeps every id
    # fitted on a subset of the binaries the vocabulary only shrinks
    assert TokenVocab(rows[:3], gnn(min_df=2)).size['mn'] <= cut.size['mn'] + N_MN
    assert cut.summary() == {'sizes_with_unk': cut.size, 'total': cut.total}


def test_vocab_kinds_per_name_and_noapi_has_no_api_tokens(tiny):  # noqa: F811
    assert VOCAB_KINDS['ours'] == ('mn', 'sig', 'beh', 'api') and VOCAB_KINDS['noapi'] == ('mn', 'sig', 'beh')
    ours, noapi = TokenVocab(tiny.rows, gnn()), TokenVocab(tiny.rows, gnn(vocab='noapi'))
    assert ours.kinds == VOCAB_KINDS['ours'] and noapi.has_api is False and noapi.insn_kinds == OURS_INSN_KINDS
    assert 'api' not in noapi.kinds
    z = load_sample(tiny.ids[0])
    blk = np.arange(len(z['blk_feat']))
    for vocab in (ours, noapi):
        tokens, kind, offsets, per_block, ins_pos, api_pos = token_table(z, vocab, blk)
        n_ins = np.diff(z['blk_insn_ptr']).astype(np.int64)
        n_api = np.diff(z['blk_api_ptr']).astype(np.int64) if vocab.has_api else 0 * n_ins
        assert len(tokens) == len(kind) == int((3 * n_ins + n_api).sum()) == int(per_block.sum())
        assert (kind[api_pos] == 3).all() and len(api_pos) == int(n_api.sum())
        assert (tokens >= 0).all() and (tokens < vocab.total).all()
        # the first instruction's mnemonic of the first non-empty block, in this vocabulary's id space
        b = int(np.flatnonzero(n_ins)[0])
        first = tokens[offsets[b]]
        assert first == vocab.lut['mn'][z['mn'][z['blk_insn_ptr'][b]]] + vocab.offset['mn']
    assert noapi.mask_id.keys() == {'mn', 'sig', 'beh'}


@real_data
def test_load_sample_keys(real_ids, real):
    z = load_sample(real_ids[0])
    assert 'blk_rva' not in z
    if 'edge_kind' in z:
        assert len(z['edge_kind']) == len(z['edge_back']) == len(z['cfg_edges'])
    else:
        pytest.skip('cache has no edge_kind/edge_back (v1); the key check ran on v2 only')


def test_ranges():
    assert ranges([5, 0, 9], [3, 0, 2]).tolist() == [5, 6, 7, 9, 10]
    assert ranges([], []).tolist() == [] and ranges([4], [0]).tolist() == []
    assert ranges(np.array([2]), np.array([1])).dtype == np.int64


# manifests

def test_kfold_v3_manifest_uses_v3_folds_and_drops_unassigned():
    rows = [{'sample_id': f'{c}{i}', 'label': i % 2, 'arch': 'x86', 'family': f'fam{i % 6}', 'identity_group': f'g{i}',
             'split': 'train' if c == 'm' else '', 'fold': i % 5 if c == 'm' else None,
             'fold_v3': None if i == 7 else (i + 1) % 5, 'corpus': c}
            for c in ('m', 'n') for i in range(40)]
    m = make_manifest(RunConfig(protocol='kfold_v3', fold=2, arch='all', val_fraction=0.0), rows)
    used = m.train + m.val + m.test
    assert {r['sample_id'] for r in used} == {r['sample_id'] for r in rows if r['fold_v3'] is not None}
    assert {r['fold'] for r in m.test} == {2} and all(r['fold'] != 2 for r in m.train + m.val)
    assert all(r['fold'] == r['fold_v3'] for r in used) and rows[0]['fold'] == 0
    m1 = make_manifest(RunConfig(protocol='kfold', fold=2, arch='all', val_fraction=0.0), rows)
    assert all(r['corpus'] == 'm' for r in m1.train + m1.val + m1.test)
    with pytest.raises(ValueError):
        make_manifest(RunConfig(protocol='kfold_v3', fold=0), [{**r, 'fold_v3': None} for r in rows])


def test_manifest_keeps_families_and_identity_groups_whole(tiny):  # noqa: F811
    for fold in range(5):
        m = make_manifest(RunConfig(protocol='kfold', fold=fold, arch='all'), tiny.rows)
        pool = m.train + m.val
        assert {r['fold'] for r in m.test} == {fold} and all(r['fold'] != fold for r in pool)
        fam = lambda rows: {r['family'] for r in rows if r['label']}
        grp = lambda rows: {r['identity_group'] for r in rows if not r['label']}
        assert not fam(m.train) & fam(m.val) and not grp(m.train) & grp(m.val)
        assert not fam(pool) & fam(m.test) and set(m.val_families) == fam(m.val)
        assert len(m.train) + len(m.val) + len(m.test) == 40
    a = make_manifest(RunConfig(protocol='kfold', fold=1, arch='all', seed=3), tiny.rows)
    assert a == make_manifest(RunConfig(protocol='kfold', fold=1, arch='all', seed=3), tiny.rows)
    assert a.val != make_manifest(RunConfig(protocol='kfold', fold=1, arch='all', seed=4), tiny.rows).val
    # a fixed val_seed fixes the split across seeds
    fixed = [make_manifest(RunConfig(protocol='kfold', fold=1, arch='all', seed=s, val_seed=9), tiny.rows) for s in (0, 1)]
    assert fixed[0] == fixed[1]


@real_data
@pytest.mark.parametrize('protocol,fold', [('kfold', f) for f in range(5)] + [('bundle', None)])
def test_train_fraction(index, protocol, fold):
    kw = dict(protocol=protocol, fold=fold, seed=0)
    full = make_manifest(RunConfig(**kw), index)
    key = {1: 'family', 0: 'identity_group'}
    previous = None
    for p in (0.25, 0.5):
        cfg = RunConfig(train_fraction=p, **kw)
        sub = make_manifest(cfg, index)
        ids = {r['sample_id'] for r in sub.train}
        assert ids < {r['sample_id'] for r in full.train}
        assert previous is None or previous < ids
        previous = ids
        assert sub.val == full.val and sub.test == full.test and sub.val_families == full.val_families
        for label in (0, 1):
            members = [r for r in full.train if r['label'] == label]
            kept = [r for r in members if r['sample_id'] in ids]
            assert abs(len(kept) / len(members) - p) < 0.05, (label, p, len(kept), len(members))
            groups = {r[key[label]] for r in members}
            for g in groups:
                inside = [r['sample_id'] in ids for r in members if r[key[label]] == g]
                assert all(inside) or not any(inside), g
        assert make_manifest(cfg, index).train == sub.train
        assert make_manifest(RunConfig(train_fraction=p, **{**kw, 'seed': 1}), index).train != sub.train
    assert make_manifest(RunConfig(train_fraction=1.0, **kw), index).train == full.train
    assert make_manifest(RunConfig(model='gnn', train_fraction=0.5, **kw), index).train == \
        make_manifest(RunConfig(train_fraction=0.5, **kw), index).train


def matched_rows():
    return [{'sample_id': f'{i:03d}', 'label': i % 2, 'arch': ('x86', 'x64')[i % 3 == 0], 'family': f'fam{i % 6}',
             'identity_group': f'g{i}', 'split': '', 'fold': None, 'fold_v3': i % 5, 'corpus': 'n',
             'arch_matched': '1' if i % 4 else ''} for i in range(80)]


def test_arch_matched_selects_flagged_rows_of_any_arch():
    rows = matched_rows()
    m = make_manifest(RunConfig(protocol='kfold_v3', fold=1, arch='matched', val_fraction=0.0), rows)
    used = m.train + m.val + m.test
    assert {r['sample_id'] for r in used} == {r['sample_id'] for r in rows if r['arch_matched'] == '1'}
    assert {r['arch'] for r in used} == {'x86', 'x64'}
    every = make_manifest(RunConfig(protocol='kfold_v3', fold=1, arch='all', val_fraction=0.0), rows)
    assert len(every.train + every.val + every.test) == 80
    x86 = make_manifest(RunConfig(protocol='kfold_v3', fold=1, arch='x86', val_fraction=0.0), rows)
    assert {r['arch'] for r in x86.train + x86.val + x86.test} == {'x86'}
    ints = [{**r, 'arch_matched': 1 if r['arch_matched'] else 0} for r in rows]
    again = make_manifest(RunConfig(protocol='kfold_v3', fold=1, arch='matched', val_fraction=0.0), ints)
    assert [r['sample_id'] for r in again.test] == [r['sample_id'] for r in m.test]


def test_arch_matched_without_the_column_raises():
    rows = [{k: v for k, v in r.items() if k != 'arch_matched'} for r in matched_rows()]
    for protocol, fold in (('kfold_v3', 0), ('kfold', 0), ('bundle', None)):
        cfg = RunConfig(protocol=protocol, fold=fold, arch='matched')
        with pytest.raises(ValueError, match='arch_matched'):
            make_manifest(cfg, rows)
    with pytest.raises(ValueError, match='arch_matched'):
        make_manifest(RunConfig(protocol='kfold_v3', fold=0, arch='matched'), [])


V3_TEST_SIZES = {'all': [1039, 1116, 1126, 1136, 1235], 'matched': [884, 924, 974, 1007, 1061]}


@real_data
@needs_combined
@pytest.mark.parametrize('arch', ['all', 'matched'])
def test_v3_manifests_on_the_combined_cache(arch):
    index = load_index(COMBINED_CACHE)
    assert index[0]['arch_matched'] in ('0', '1', '')
    for fold, n_test in enumerate(V3_TEST_SIZES[arch]):
        m = make_manifest(RunConfig(protocol='kfold_v3', fold=fold, arch=arch), index)
        assert len(m.test) == n_test and {r['fold'] for r in m.test} == {fold}
        if arch == 'matched':
            assert all(r['arch_matched'] == '1' for r in m.train + m.val + m.test)
        assert not {r['family'] for r in m.test if r['label']} & {r['family'] for r in m.train + m.val if r['label']}
        assert not ({r['identity_group'] for r in m.test if not r['label']}
                    & {r['identity_group'] for r in m.train + m.val if not r['label']})


def test_load_index_reads_built_rows_and_defaults_the_corpus(tiny):  # noqa: F811
    rows = load_index(tiny.root)
    assert len(rows) == 40 and {r['corpus'] for r in rows} == {'mendeley'}
    assert all(isinstance(r['label'], int) and r['fold_v3'] is None for r in rows)
    assert sorted({r['fold'] for r in rows}) == [0, 1, 2, 3, 4]


# the per-binary block cap

def fn_sizes(z):
    return np.diff(z['fn_blk_ptr'].astype(np.int64))


def expected_eval_kept(z, cap):
    """Independent reading of the eval rule: non-library first, API references desc, size desc, index asc; the longest
    prefix whose blocks fit."""
    sizes = fn_sizes(z)
    cand = np.flatnonzero(sizes > 0)
    api = z['blk_api_ptr'].astype(np.int64)
    ptr = z['fn_blk_ptr'].astype(np.int64)
    n_api = api[ptr[cand + 1]] - api[ptr[cand]]
    library = (z['fn_flags'][cand] & 1).astype(int)
    order = sorted(range(len(cand)), key=lambda k: (library[k], -n_api[k], -sizes[cand[k]], k))
    kept, total = [], 0
    for k in order:
        if total + sizes[cand[k]] > cap:
            break
        kept.append(int(cand[k]))
        total += int(sizes[cand[k]])
    return sorted(kept)


@pytest.fixture
def small_cap(monkeypatch):
    monkeypatch.setattr(subgraph, 'MAX_BINARY_BLOCKS', 150)
    return 150


def test_cap_constant_and_untouched_binaries():
    assert subgraph.MAX_BINARY_BLOCKS == 500_000
    z = synth()
    sizes = fn_sizes(z)
    fn, blk = subgraph.select_subgraph(z, 'cap', 0, None)
    assert np.array_equal(fn, np.flatnonzero(sizes > 0)) and len(blk) == int(sizes.sum())


def test_cap_respected_whole_functions_in_eval_and_train(small_cap):
    z = synth()
    sizes = fn_sizes(z)
    assert sizes.sum() > small_cap
    for seed in (None, 1, 2, 3):
        rng = None if seed is None else np.random.default_rng([0, seed, 0])
        fn, blk = subgraph.select_subgraph(z, 'cap', 0, rng)
        assert 0 < len(blk) <= small_cap and len(blk) == sizes[fn].sum() and len(fn) >= 1
        assert np.array_equal(blk, ranges(z['fn_blk_ptr'][fn].astype(np.int64), sizes[fn]))
        assert (sizes[fn] > 0).all()


def test_cap_eval_selection_is_deterministic_and_ordered(small_cap):
    z = synth()
    a = subgraph.select_subgraph(z, 'cap', 0, None)
    assert all(np.array_equal(x, y) for x, y in zip(a, subgraph.select_subgraph(z, 'cap', 0, None)))
    keep = subgraph.cap_functions(z['fn_blk_ptr'], z['fn_flags'], z['blk_api_ptr'], fn_sizes(z) > 0, None, small_cap)
    assert np.flatnonzero(keep).tolist() == expected_eval_kept(z, small_cap)
    assert a[0].tolist() == expected_eval_kept(z, small_cap)


def test_cap_train_selection_varies_across_epochs_and_is_reproducible(small_cap):
    z = synth()
    kept = []
    for epoch in range(1, 7):
        fn, blk = subgraph.select_subgraph(z, 'cap', 0, np.random.default_rng([0, epoch, 0]))
        assert len(blk) <= small_cap
        kept.append(tuple(fn.tolist()))
        assert kept[-1] == tuple(subgraph.select_subgraph(z, 'cap', 0, np.random.default_rng([0, epoch, 0]))[0].tolist())
    assert len(set(kept)) >= 4
    eval_kept = tuple(subgraph.select_subgraph(z, 'cap', 0, None)[0].tolist())
    assert any(k != eval_kept for k in kept)


def test_cap_edge_cases():
    ptr = np.array([0, 40, 50, 60, 70])
    flags = np.zeros(4, dtype=np.int8)
    api = np.arange(71)
    keep = np.ones(4, dtype=bool)
    assert subgraph.cap_functions(ptr, flags, api, keep, None, 50).tolist() == [True, True, False, False]
    first_too_big = subgraph.cap_functions(ptr, flags, api, keep, None, 30)
    assert first_too_big.tolist() == [False, True, False, False]
    huge = subgraph.cap_functions(np.array([0, 90]), flags[:1], np.arange(91), np.ones(1, dtype=bool), None, 30)
    assert huge.tolist() == [True]
    sub = np.array([True, False, True, True])
    for rng in (None, np.random.default_rng(0)):
        got = subgraph.cap_functions(ptr, flags, api, sub, rng, 25)
        assert not (got & ~sub).any() and got.any() and (np.diff(ptr)[got]).sum() <= 25
    assert subgraph.cap_functions(ptr, flags, api, keep, None, 10 ** 9) is keep


def test_select_subgraph_cap_is_cap_functions(binaries, small_cap):
    for z, sid in binaries + [(synth(s), 's') for s in range(4)]:
        sizes = fn_sizes(z)
        for keep in (sizes > 0, (sizes > 0) & (np.arange(len(sizes)) % 3 != 0)):
            for budget in (0, 40, 90, 10 ** 6):
                for rng_seed in (None, 3):
                    rng = None if rng_seed is None else np.random.default_rng(rng_seed)
                    want = subgraph.cap_functions(z['fn_blk_ptr'], z['fn_flags'], z['blk_api_ptr'], keep,
                                                  None if rng_seed is None else np.random.default_rng(rng_seed),
                                                  budget or None)
                    fn, blk = subgraph.select_subgraph(z, 'cap', budget, rng, sid, keep=keep)
                    assert np.array_equal(fn, np.flatnonzero(want))
                    assert np.array_equal(blk, ranges(z['fn_blk_ptr'][fn].astype(np.int64), sizes[fn]))
                    assert (np.diff(fn) > 0).all() and (np.diff(blk) > 0).all() and keep[fn].all()


def test_select_subgraph_default_keep_is_the_nonempty_functions():
    z = synth()
    fn, blk = subgraph.select_subgraph(z, 'cap', 10 ** 6, None)
    assert np.array_equal(fn, np.flatnonzero(fn_sizes(z) > 0)) and len(blk) == fn_sizes(z).sum()
    for removed in ('lcc', 'bfs'):
        with pytest.raises(ValueError):
            subgraph.select_subgraph(z, removed, 10, None)


def test_cap_policy_with_a_budget_equals_the_global_cap(monkeypatch):
    z = synth_binary(np.random.default_rng(5), n_fn=400)
    via_budget = subgraph.select_subgraph(z, 'cap', 1000, None)
    monkeypatch.setattr(subgraph, 'MAX_BINARY_BLOCKS', 1000)
    via_cap = subgraph.select_subgraph(z, 'cap', 0, None)
    assert all(np.array_equal(a, b) for a, b in zip(via_budget, via_cap)) and len(via_budget[1]) <= 1000


# the chain measurement

def chain_binary():
    """Two functions (blocks 0-3 and 4-7) with CFG edges (src, dst, kind); the mergeable ones are marked."""
    edges = [(0, 1, 0),   # mergeable
             (1, 2, 0),   # out-degree of 1 is 2
             (1, 3, 1),   # not a fallthrough
             (2, 3, 0),   # in-degree of 3 is 3 (with the edge from block 1 and the edge from 2 itself)
             (3, 4, 0),   # crosses into the next function
             (4, 5, 0),   # mergeable
             (5, 6, 2),   # not a fallthrough
             (7, 7, 0)]   # a self loop
    return {'blk_feat': np.zeros((8, 6), dtype=np.float16), 'fn_blk_ptr': np.array([0, 4, 8], dtype=np.int32),
            'cfg_edges': np.array([e[:2] for e in edges], dtype=np.int32),
            'edge_kind': np.array([e[2] for e in edges], dtype=np.int8)}


def test_chain_edges_apply_each_condition():
    assert subgraph.chain_edges(chain_binary()).tolist() == [True, False, False, False, False, True, False, False]
    z = chain_binary()
    z['cfg_edges'] = np.append(z['cfg_edges'], [[0, 1]], axis=0)
    z['edge_kind'] = np.append(z['edge_kind'], 0).astype(np.int8)
    assert not subgraph.chain_edges(z)[0] and not subgraph.chain_edges(z)[-1]


def test_chain_reduction_counts_and_fraction(tmp_path):
    (tmp_path / 'samples').mkdir()
    z = chain_binary()
    np.savez(tmp_path / 'samples' / 'a.npz', **z)
    plain = {**z, 'cfg_edges': np.array([[0, 1], [2, 3]], dtype=np.int32), 'edge_kind': np.array([1, 2], dtype=np.int8)}
    np.savez(tmp_path / 'samples' / 'b.npz', **plain)
    out = subgraph.chain_reduction([{'sample_id': 'a'}, {'sample_id': 'b'}], tmp_path)
    assert (out['n_binaries'], out['n_blocks'], out['n_chain_blocks']) == (2, 16, 2)
    assert out['reduction'] == 2 / 16 and out['max_binary_reduction'] == 2 / 8 and out['mean_binary_reduction'] == 1 / 8
    np.savez(tmp_path / 'samples' / 'c.npz', **{k: v for k, v in z.items() if k != 'edge_kind'})
    with pytest.raises(KeyError, match='edge_kind'):
        subgraph.chain_reduction([{'sample_id': 'c'}], tmp_path)


@real_data
def test_chain_reduction_on_real_binaries(index, real):
    rows = [r for r in index if r['arch'] == 'x86' and int(r['num_blocks']) < 3000][:6]
    out = subgraph.chain_reduction(rows, manifests.CACHE)
    assert out['n_binaries'] == len(rows) and 0 <= out['n_chain_blocks'] < out['n_blocks'] and 0 <= out['reduction'] < 1
