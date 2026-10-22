"""Manifests: the kfold, kfold_v3 and bundle splits, the validation carve, `--train-fraction`, `--arch matched` and the index.

Tests marked `real_data` read the cache of the machine (config.CACHE: datasets.json, `--cache <name>` or GT_CACHE) and skip
without it; the others run on the synthetic tiny cache (tests/tiny_cache.py) or on rows made here.
"""
from pathlib import Path

import pytest

from common import config
from common.config import COMBINED_CACHE, RunConfig
from common.manifests import load_index, make_manifest
from tiny_cache import tiny  # noqa: F401  (a fixture)

real_data = pytest.mark.real_data
needs_combined = pytest.mark.skipif(not (Path(COMBINED_CACHE) / 'index.csv').exists(), reason='needs the combined cache')
REAL_CACHE = config.CACHE        # the cache of this machine; the `tiny` fixture points the module-level CACHEs elsewhere
V3_TEST_SIZES = {'all': [1039, 1116, 1126, 1136, 1235], 'matched': [884, 924, 974, 1007, 1061]}


@pytest.fixture(scope='session')
def index():
    return load_index(REAL_CACHE)


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
