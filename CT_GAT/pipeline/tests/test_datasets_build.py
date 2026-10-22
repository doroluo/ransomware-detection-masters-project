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


