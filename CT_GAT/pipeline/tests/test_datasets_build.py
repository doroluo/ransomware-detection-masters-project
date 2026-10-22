"""python -m pytest tests/test_datasets_build.py: the team pipelines on any dataset (plan v6, steps A1 to A3).

What is checked, on the second synthetic dataset `tiny_v3` (tests/tiny_cache.py: 48 Mendeley rows with a Mendeley fold, 70 rows with a
v3 fold, 50 of them matched) and on the synthetic collections (tests/s02_collection.py):
  * the dataset itself: its sizes under the four (protocol, arch) pairs, matched inside all, and the leakage checks of the folds
    (families and identity groups never split, every row without a fold in no split), which the `real_data` tests repeat on the
    published caches;
  * `common.py`: the zero-argument calls are the kfold x86 ones, the id sets, the run names of every (protocol, arch, arm,
    arch_balance), `paths_for` by corpus;
  * `build_b --protocol --arch --max-instructions --extend`, on a stand-in `process_sample` (the pattern of test_seeded_rows.py);
  * `real_data`: a 3-id build equals the published build, 3 ids rebuilt `--frozen-from` equal the stored tensors, and 5 x64 ids
    extend a copy of a published build.
"""
import filecmp
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from common.config import COMBINED_CACHE, DEFAULT_CACHE, RUNS, RunConfig
from common.manifests import arch_rows, load_index, make_manifest
from experiments.team_pipelines import build_b, common as tp, compare_builds, match_rates, train_b
from s02_collection import collections
from test_team_pipelines import fresh_team_caches, make_bundle, zero_config  # noqa: F401  (fixtures and helpers)
import tiny_cache
from tiny_cache import tiny_v3  # noqa: F401  (a fixture)

HERE = Path(__file__).resolve().parent
FOLDS = range(5)
DATASETS = [('kfold', 'x86'), ('kfold', 'all'), ('kfold_v3', 'all'), ('kfold_v3', 'matched')]
SIZES = {('kfold', 'x86'): 40, ('kfold', 'all'): 48, ('kfold_v3', 'all'): 70}
real_data = pytest.mark.real_data
needs_v2 = pytest.mark.skipif(not (Path(DEFAULT_CACHE) / 'index.csv').exists(), reason='needs the mendeley_x86 cache')
needs_combined = pytest.mark.skipif(not (Path(COMBINED_CACHE) / 'index.csv').exists(), reason='needs the combined cache')


def key(sample):
    return sample['sample_id']


def groups_of(rows):
    """The groups a split must not share: ransomware families and goodware identity groups (the keys of `carve_validation`)."""
    return {r['family'] if r['label'] else r['identity_group'] for r in rows}


def check_folds(index, protocol, arch, seeds=(0,)):
    """The leakage checks of plan section 5 on the folds of a dataset: the test sets partition the rows that have a fold, train,
    validation and test of a fold hold no id and no family or identity group twice, a row without a fold is in no split."""
    fold_column = 'fold_v3' if protocol == 'kfold_v3' else 'fold'
    with_fold = {r['sample_id'] for r in arch_rows(arch, index) if r[fold_column] is not None}
    tests = []
    for fold in FOLDS:
        for seed in seeds:
            m = make_manifest(RunConfig(protocol=protocol, fold=fold, arch=arch, seed=seed, model='gnn', mask_blocks=0.15), index)
            splits = {'train': m.train, 'val': m.val, 'test': m.test}
            ids = [key(r) for rows in splits.values() for r in rows]
            assert len(ids) == len(set(ids)) == len(with_fold), (protocol, arch, fold)
            assert set(ids) == with_fold
            names = list(splits)
            for a in range(3):
                for b in range(a + 1, 3):
                    assert not groups_of(splits[names[a]]) & groups_of(splits[names[b]]), (protocol, arch, fold, names[a], names[b])
            assert {r['fold'] for r in m.test} == {fold} and all(r['fold'] != fold for r in m.train + m.val)
        tests.append({key(r) for r in m.test})
    assert sum(map(len, tests)) == len(with_fold) and set().union(*tests) == with_fold
    return with_fold


def check_matched_inside_all(index, protocol):
    for fold in FOLDS:
        both = {arch: make_manifest(RunConfig(protocol=protocol, fold=fold, arch=arch), index) for arch in ('all', 'matched')}
        every = lambda m: {key(r) for r in m.train + m.val + m.test}      # noqa: E731
        assert every(both['matched']) <= every(both['all'])
        assert {key(r) for r in both['matched'].test} <= {key(r) for r in both['all'].test}
        assert all(str(r['arch_matched']) == '1' for r in both['matched'].train + both['matched'].val + both['matched'].test)


# ---- A1: the second dataset --------------------------------------------------------------------------------------------------

def test_the_second_dataset_leaves_the_first_forty_binaries_of_tiny_unchanged(tiny_v3, tmp_path):
    plain = tiny_cache.build(tmp_path / 'plain')
    assert plain['ids'] == tiny_v3.ids[:40]
    for a, b in zip(plain['rows'], tiny_v3.rows[:40]):
        assert {k: b[k] for k in a} == a and b['corpus'] == 'mendeley' and b['arch'] == 'x86'
    for sample_id in plain['ids']:
        with np.load(tmp_path / 'plain' / 'samples' / f'{sample_id}.npz') as a, np.load(tiny_v3.root / 'samples' / f'{sample_id}.npz') as b:
            assert a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)
    header = (tiny_v3.root / 'index.csv').read_text(encoding='utf-8').splitlines()[0]
    assert header.startswith((tmp_path / 'plain' / 'index.csv').read_text(encoding='utf-8').splitlines()[0])


def test_the_second_dataset_has_the_sizes_of_the_plan(tiny_v3):
    index = load_index(tiny_v3.root)
    assert len(index) == 72 and {r['corpus'] for r in index} == {'mendeley', 'ransomware_balanced', 'balanced_goodware'}
    assert sum(r['corpus'] == 'mendeley' for r in index) == 48 and sum(r['arch'] == 'x64' and r['corpus'] == 'mendeley' for r in index) == 8
    assert all(r['fold'] is None for r in index if r['corpus'] != 'mendeley')
    assert all(r['fold'] is not None and r['fold_v3'] is not None for r in index if r['corpus'] == 'mendeley')
    assert sorted(r['sample_id'] for r in index if r['fold_v3'] is None) == sorted([index[53]['sample_id'], index[71]['sample_id']])
    cells = {(r['label'], r['arch']) for r in index if r['fold_v3'] is not None}
    assert cells == {(1, 'x86'), (1, 'x64'), (0, 'x86'), (0, 'x64')}
    for protocol, arch, n in (('kfold', 'x86', 40), ('kfold', 'all', 48), ('kfold_v3', 'all', 70)):
        m = [make_manifest(RunConfig(protocol=protocol, fold=f, arch=arch), index) for f in FOLDS]
        assert sum(len(x.test) for x in m) == n
        assert all(len(x.train) and len(x.val) and len(x.test) for x in m)
    matched = [make_manifest(RunConfig(protocol='kfold_v3', fold=f, arch='matched'), index) for f in FOLDS]
    n_matched = sum(len(x.test) for x in matched)
    assert 0 < n_matched < 70 and n_matched == sum(1 for r in index if str(r['arch_matched']) == '1')
    assert all(len(x.train) and len(x.val) and len(x.test) for x in matched)


@pytest.mark.parametrize('protocol, arch', DATASETS)
def test_no_family_or_identity_group_is_split_and_no_row_without_a_fold_is_used(tiny_v3, protocol, arch):
    check_folds(load_index(tiny_v3.root), protocol, arch, seeds=(0, 1, 2))


def test_matched_is_a_subset_of_all_in_every_fold(tiny_v3):
    check_matched_inside_all(load_index(tiny_v3.root), 'kfold_v3')


@real_data
@needs_v2
@pytest.mark.parametrize('arch', ['x86', 'all'])
def test_real_mendeley_folds_keep_families_and_identity_groups_whole(arch):
    index = load_index(DEFAULT_CACHE)
    with_fold = check_folds(index, 'kfold', arch)
    assert len(with_fold) == {'x86': 1810, 'all': 2411}[arch]


@real_data
@needs_combined
@pytest.mark.parametrize('arch', ['all', 'matched'])
def test_real_combined_folds_keep_families_and_identity_groups_whole(arch):
    index = load_index(COMBINED_CACHE)
    with_fold = check_folds(index, 'kfold_v3', arch)
    assert len(with_fold) == {'all': 5652, 'matched': 4850}[arch]
    if arch == 'matched':
        check_matched_inside_all(index, 'kfold_v3')


# ---- A2: the adapter on any dataset ------------------------------------------------------------------------------------------

def test_the_zero_argument_calls_are_the_kfold_x86_ones(tiny_v3, tmp_path):
    plain = tiny_cache.build(tmp_path / 'plain')
    index = load_index(tmp_path / 'plain')
    assert set(tp.sample_rows()) == set(plain['ids']) == set(tp.sample_rows('kfold', 'x86')) and len(tp.sample_rows()) == 40
    assert tp.all_ids() == tp.all_ids('kfold', 'x86') == sorted(plain['ids'])
    assert tp.fold_config(3, 2) == RunConfig(protocol='kfold', fold=3, arch='x86', seed=2, model='gnn', mask_blocks=0.15)
    for fold in FOLDS:
        want, got = make_manifest(tp.fold_config(fold, 1), index), tp.fold_manifest(fold, 1)
        for name in ('train', 'val', 'test'):
            assert [key(r) for r in getattr(got, name)] == [key(r) for r in getattr(want, name)]
        explicit = tp.fold_manifest(fold, 1, 'kfold', 'x86')
        assert [key(r) for r in explicit.test] == [key(r) for r in got.test] and [key(r) for r in explicit.val] == [key(r) for r in got.val]
    cfg = tp.team_config(2, 1, 'sig500v3-emb')
    assert cfg.name == 'kfold2_x86_gnn_s1_team-sig500v3-emb' and cfg.group == 'kfold_x86_gnn_team-sig500v3-emb'
    assert cfg == tp.team_config(2, 1, 'sig500v3-emb', 'kfold', 'x86', False)


def test_the_id_sets_of_the_four_datasets(tiny_v3):
    sets = {pair: tp.all_ids(*pair) for pair in DATASETS}
    rows = {pair: tp.sample_rows(*pair) for pair in DATASETS}
    for pair, ids in sets.items():
        assert ids == sorted(ids) and set(ids) == set(rows[pair])
    assert [len(sets[p]) for p in DATASETS[:3]] == [40, 48, 70] and 0 < len(sets[DATASETS[3]]) < 70
    assert set(sets[('kfold', 'x86')]) < set(sets[('kfold', 'all')]) < set(sets[('kfold_v3', 'all')])
    assert set(sets[('kfold_v3', 'matched')]) < set(sets[('kfold_v3', 'all')])
    for fold in FOLDS:
        assert all(str(r['arch_matched']) == '1' for r in tp.fold_manifest(fold, 0, 'kfold_v3', 'matched').test)
    union = [r['sample_id'] for f in FOLDS for r in tp.fold_manifest(f, 0, 'kfold_v3', 'all').test]
    assert sorted(union) == sets[('kfold_v3', 'all')] and len(union) == len(set(union))
    index = load_index(tiny_v3.root)
    by_v3_fold = {r['sample_id']: r['fold_v3'] for r in index}
    assert all(r['fold'] == by_v3_fold[i] for i, r in rows[('kfold_v3', 'all')].items())      # the rows carry the v3 fold
    assert {i: r['fold'] for i, r in rows[('kfold', 'all')].items()} == {r['sample_id']: r['fold'] for r in index if r['corpus'] == 'mendeley'}
    assert {r['corpus'] for r in rows[('kfold', 'all')].values()} == {'mendeley'}
    assert {r['arch'] for r in rows[('kfold', 'all')].values()} == {'x86', 'x64'}
    no_fold = [r['sample_id'] for r in index if r['fold_v3'] is None]
    assert len(no_fold) == 2 and not set(no_fold) & set(sets[('kfold_v3', 'all')])


def test_an_unknown_protocol_or_arch_is_refused(tiny_v3):
    with pytest.raises(ValueError, match='bundle'):
        tp.sample_rows('bundle', 'x86')
    with pytest.raises(ValueError, match='arch'):
        tp.fold_config(0, 0, 'kfold', 'x128')
    with pytest.raises(ValueError, match='protocol'):
        tp.team_config(0, 0, 'sig500v3-feat', 'bundle', 'x86')


def test_the_dataset_table_names_what_the_catalog_knows():
    from common import catalog
    assert set(tp.DATASET_OF.values()) <= set(catalog.DATASETS) and set(tp.DATASET_OF) == set(tp.DEFAULT_ARCH) == set(tp.BUILD_ARCHES)
    assert tp.DEFAULT_ARCH == {'kfold': 'x86', 'kfold_v3': 'all'}
    assert tp.BUILD_ARCHES == {'kfold': ('x86', 'all'), 'kfold_v3': ('all',)} and 'matched' not in sum(tp.BUILD_ARCHES.values(), ())
    for protocol, name in tp.DATASET_OF.items():
        assert protocol in catalog.DATASETS[name].protocols


def all_run_names():
    """{run name: (protocol, arch, arm, arch_balance)} of every team arm over folds 0-4 and seeds 0-2 on the four datasets;
    `arch_balance` is not offered on kfold x86."""
    out = {}
    for protocol, arch in DATASETS:
        for ab in (False, True):
            if (protocol, arch) == ('kfold', 'x86') and ab:
                continue
            for arm in train_b.ARMS:
                for fold in FOLDS:
                    for seed in range(3):
                        name = tp.team_config(fold, seed, arm, protocol, arch, ab).name
                        assert name not in out, name
                        out[name] = (protocol, arch, arm, ab)
    return out


def test_the_run_names_of_every_dataset_are_distinct_and_those_of_kfold_x86_are_unchanged():
    names = all_run_names()
    assert len(names) == 7 * len(train_b.ARMS) * 15
    for name, (protocol, arch, arm, ab) in names.items():
        split = {'kfold': r'kfold\d', 'kfold_v3': r'v3kfold\d'}[protocol]
        assert re.fullmatch(rf'{split}_{arch}_gnn{"-ab" if ab else ""}_s\d_team-{re.escape(arm)}', name), name
        if (protocol, arch) == ('kfold', 'x86'):
            assert re.fullmatch(rf'kfold\d_x86_gnn_s\d_team-{re.escape(arm)}', name)
    groups = {tp.team_config(0, 0, 'sig500v3-feat', p, a, ab).group for p, a in DATASETS for ab in (False, True)
              if (p, a) != ('kfold', 'x86') or not ab}
    assert groups == {'kfold_x86_gnn_team-sig500v3-feat', 'kfold_all_gnn_team-sig500v3-feat', 'kfold_all_gnn-ab_team-sig500v3-feat',
                      'kfold_v3_all_gnn_team-sig500v3-feat', 'kfold_v3_all_gnn-ab_team-sig500v3-feat',
                      'kfold_v3_matched_gnn_team-sig500v3-feat', 'kfold_v3_matched_gnn-ab_team-sig500v3-feat'}


def test_the_new_run_names_collide_with_nothing_known():
    names = set(all_run_names())
    new = {n for n, (protocol, arch, _, _) in all_run_names().items() if (protocol, arch) != ('kfold', 'x86')}
    assert len(new) == 6 * len(train_b.ARMS) * 15 and not new & {n for n in names if n not in new}
    groups = {tp.team_config(0, 0, arm, p, a, ab).group for arm in train_b.ARMS for p, a in DATASETS[1:] for ab in (False, True)}
    assert len(groups) == 6 * len(train_b.ARMS)
    for fixture in (HERE / 'fixtures').rglob('*'):
        if fixture.is_file() and fixture.suffix in ('.txt', '.json', '.csv', '.md'):
            tokens = set(re.findall(r'[\w-]*kfold[\w-]*', fixture.read_text(encoding='utf-8', errors='replace')))
            assert not tokens & (new | groups), fixture
    if Path(RUNS).is_dir():
        assert not new & {p.name for p in Path(RUNS).iterdir()}


@pytest.fixture
def collection_folders(tiny_v3, tmp_path, monkeypatch):
    """The four collection folders of the tiny ids as the `collections` source, no bundle, no listing directory."""
    folder = collections(tmp_path / 'data', tiny_v3.rows)
    zero_config(monkeypatch, tmp_path, folder)
    return folder


def test_paths_for_takes_a_balanced_binary_from_its_collection_and_a_mendeley_one_as_before(tiny_v3, collection_folders):
    by_corpus = {}
    for r in tiny_v3.rows:
        by_corpus.setdefault(r['corpus'], r)
    for corpus, folder in (('ransomware_balanced', 'Ransomware_Balanced'), ('balanced_goodware', 'Balanced_Goodware')):
        sid = by_corpus[corpus]['sample_id']
        assert tp.paths_for(sid, corpus) == (collection_folders / folder / 'samples' / f'{sid}.jsonl.gz',
                                             collection_folders / folder / 'samples' / f'{sid}.asm.gz')
        assert tp.team_row(sid, corpus)['cfg_file'].parent.parent.name == folder
        with pytest.raises(KeyError, match='no row of the mendeley corpus'):
            tp.paths_for(sid)                                    # the default corpus is mendeley: the id is not one of its rows
        with pytest.raises(KeyError, match='no row'):
            tp.paths_for(tiny_v3.ids[0], corpus)
    sid = by_corpus['mendeley']['sample_id']
    folder = 'Mendeley_Ransomware' if by_corpus['mendeley']['label'] else 'Mendeley_Goodware'
    assert tp.paths_for(sid) == tp.paths_for(sid, 'mendeley')
    assert tp.paths_for(sid)[0] == collection_folders / folder / 'samples' / f'{sid}.jsonl.gz'


def test_a_balanced_binary_ignores_the_bundle_and_the_listing_directory_which_still_rule_the_mendeley_ones(tiny_v3, collection_folders,
                                                                                                           tmp_path, monkeypatch):
    mendeley = next(r['sample_id'] for r in tiny_v3.rows if r['corpus'] == 'mendeley')
    balanced = next(r['sample_id'] for r in tiny_v3.rows if r['corpus'] == 'ransomware_balanced')
    bundle, asm = make_bundle(tmp_path / 'b', [mendeley])
    monkeypatch.setenv('GT_BUNDLE', str(bundle))
    with pytest.raises(tp.TeamSourceNotConfigured, match='GT_TEAM_ASM_DIR'):                   # half configured: the Mendeley id asks
        tp.paths_for(mendeley)
    assert tp.paths_for(balanced, 'ransomware_balanced')[0].is_file()                         # the balanced one does not
    monkeypatch.setenv('GT_TEAM_ASM_DIR', str(asm))
    assert tp.paths_for(mendeley) == (bundle / 'cfg' / f'{mendeley}.jsonl.gz', asm / f'{mendeley}.asm.gz')
    assert tp.paths_for(balanced, 'ransomware_balanced')[1].parent.parent.name == 'Ransomware_Balanced'


def test_a_missing_collection_names_the_folder_for_a_balanced_binary(tiny_v3, tmp_path, monkeypatch):
    zero_config(monkeypatch, tmp_path, tmp_path / 'nothing')
    sid = next(r['sample_id'] for r in tiny_v3.rows if r['corpus'] == 'balanced_goodware')
    with pytest.raises(tp.TeamSourceNotConfigured, match='Balanced_Goodware') as e:
        tp.paths_for(sid, 'balanced_goodware')
    assert 'GT_BUNDLE' not in str(e.value)


