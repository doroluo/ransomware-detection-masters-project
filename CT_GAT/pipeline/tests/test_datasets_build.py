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


M1FROZEN_RECORDINGS = {'team_m1frozen', 'team_v3_m1frozen'}


def test_the_new_run_names_collide_with_nothing_known():
    names = set(all_run_names())
    new = {n for n, (protocol, arch, _, _) in all_run_names().items() if (protocol, arch) != ('kfold', 'x86')}
    assert len(new) == 6 * len(train_b.ARMS) * 15 and not new & {n for n in names if n not in new}
    groups = {tp.team_config(0, 0, arm, p, a, ab).group for arm in train_b.ARMS for p, a in DATASETS[1:] for ab in (False, True)}
    assert len(groups) == 6 * len(train_b.ARMS)
    recordings = {'team_mendeley_all', 'team_v3'}               # the recorded plans on the new datasets name exactly these runs
    for fixture in (HERE / 'fixtures').rglob('*'):
        if fixture.is_file() and fixture.suffix in ('.txt', '.json', '.csv', '.md'):
            tokens = set(re.findall(r'[\w-]*kfold[\w-]*', fixture.read_text(encoding='utf-8', errors='replace')))
            if fixture.parent.name == 'plans_dry' and fixture.stem in M1FROZEN_RECORDINGS:
                continue       # these name runs of the x86 dataset next to runs of the new ones: test_team_m1frozen.py compares their full run names
            if fixture.parent.name == 'plans_dry' and fixture.stem in recordings:
                assert not [t for t in tokens if re.match(r'kfold\d_x86_', t)], fixture                      # none is a run of the x86 dataset
                assert {t for t in tokens if '_team-' in t} <= new and tokens & new, fixture             # every team run is one of the new names
                continue
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


# ---- A3: build_b on a dataset ------------------------------------------------------------------------------------------------

class Result(dict):
    """What the stand-in `process_sample` returns: the dict `save_sample` reads, plus `.sample`."""


@pytest.fixture
def stand_in(tiny_v3, monkeypatch):
    """(real, order, rec): `build_b` run on a stand-in `process_sample` that grows the vocabulary by two names per sample (named
    by the position of the id among the 70 ids of kfold_v3) and saves block embeddings pooled from the new rows. `rec` collects
    the `dataset_metadata` of each built id (`rec.meta`), the calls of `paths_for` (`rec.paths`) and the ids to fail on
    (`rec.fail`)."""
    real = tp.import_team('preprocess_full', source='repo')
    order = tp.all_ids('kfold_v3', 'all')
    rec = type('Rec', (), {'meta': {}, 'paths': [], 'fail': set()})

    class Shim:
        def __getattr__(self, name):
            return getattr(real, name)

        @staticmethod
        def process_sample(asm_path, cfg_path, label, seed, api_vocab, channel_embedder, file_id, dataset_metadata=None, **kwargs):
            if file_id in rec.fail:
                raise RuntimeError('stopped on purpose')
            rec.meta[file_id] = dataset_metadata
            k = order.index(file_id)
            for j in (2 * k, 2 * k + 1):
                api_vocab.id_for(f'lib.dll!Fn{j}')
            channel_embedder.expand_api(real.api_table_size(api_vocab))
            api = torch.arange(real.tok.FIRST_API_ID, real.api_table_size(api_vocab))
            ids = torch.stack([torch.ones_like(api)] * 3 + [api, torch.zeros_like(api)], 1)
            pooled = real.pool_block_embeddings(ids, torch.zeros(len(api), dtype=torch.long), 1, channel_embedder)
            sample = type('S', (), {'original_blocks': 1, 'sampled_blocks': 1})
            return Result(sample=sample, manifest={'sample_id': file_id, 'label': label}, models={'block_embeddings': pooled})

        @staticmethod
        def save_sample(result, out):
            torch.save({'file_id': result['manifest']['sample_id'], 'block_embeddings': result['models']['block_embeddings']},
                       out / f"{result['manifest']['sample_id']}.pt")

    def spy(sample_id, *corpus):
        rec.paths.append((sample_id, *corpus))
        return None, None

    monkeypatch.setattr(build_b, 'import_team', lambda *args, **kwargs: Shim())
    monkeypatch.setattr(build_b, 'paths_for', spy)
    monkeypatch.setattr(build_b, 'SAVE_EVERY', 2)
    return real, order, rec


def run(out, *flags, source='repo'):
    return build_b.main(['--out', str(out), '--source', source, *flags])


def ids_of(pairs=None, **where):
    """Ids of tiny_v3 rows from `tp.sample_rows('kfold_v3', 'all')` (sorted) whose fields equal `where`."""
    rows = tp.sample_rows('kfold_v3', 'all')
    return [i for i in sorted(rows) if all(rows[i][k] == v for k, v in where.items())]


def json_of(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def test_a_kfold_x86_build_writes_the_settings_it_always_wrote(stand_in, tmp_path):
    _, order, _ = stand_in
    x86 = ids_of(arch='x86', corpus='mendeley')[:4]
    for flags in ((), ('--protocol', 'kfold'), ('--protocol', 'kfold', '--arch', 'x86')):
        out = tmp_path / '-'.join(flags or ('none',)).replace('-', '')
        run(out, '--ids', *x86, *flags)
        settings = json_of(out / 'settings.json')
        assert set(settings) == {'sample', 'num_blocks', 'seed', 'source', 'pins', 'excluded'}
        assert json_of(out / 'checkpoint.json')['settings'] == {k: settings[k] for k in ('sample', 'num_blocks', 'seed', 'source', 'pins')}
        assert 'protocol' not in json_of(out / 'build_summary.json')['settings']
        run(out, '--ids', *x86, *flags)                                                      # the resume agrees with itself


def test_a_kfold_v3_build_records_its_dataset_and_a_resume_for_another_dataset_stops(stand_in, tmp_path):
    _, _, _ = stand_in
    some = ids_of(corpus='ransomware_balanced')[:2] + ids_of(corpus='balanced_goodware')[:2] + ids_of(corpus='mendeley')[:2]
    out = tmp_path / 'v3'
    summary = run(out, '--protocol', 'kfold_v3', '--ids', *some)
    assert summary['processed'] == summary['total'] == 6
    settings = json_of(out / 'settings.json')
    assert (settings['protocol'], settings['arch']) == ('kfold_v3', 'all')
    assert json_of(out / 'checkpoint.json')['settings']['protocol'] == 'kfold_v3' and json_of(out / 'checkpoint.json')['settings']['arch'] == 'all'
    assert run(out, '--protocol', 'kfold_v3', '--arch', 'all', '--ids', *some)['built_this_run'] == 0
    with pytest.raises(SystemExit, match='was built for kfold_v3 all, this run asks for kfold x86'):
        run(out, '--ids', *ids_of(arch='x86', corpus='mendeley')[:2])
    with pytest.raises(SystemExit, match='was built for kfold_v3 all, this run asks for kfold all'):
        run(out, '--protocol', 'kfold', '--arch', 'all', '--ids', *ids_of(corpus='mendeley')[:2])
    old = tmp_path / 'old'                                                                   # a build without the keys is a kfold x86 one
    run(old, '--ids', *ids_of(arch='x86', corpus='mendeley')[:2])
    with pytest.raises(SystemExit, match='was built for kfold x86, this run asks for kfold_v3 all'):
        run(old, '--protocol', 'kfold_v3', '--ids', *some)
    with pytest.raises(SystemExit, match='was built for kfold x86, this run asks for kfold all'):
        run(old, '--protocol', 'kfold', '--arch', 'all', '--ids', *some[-2:])


def test_the_dataset_flags_are_checked_before_anything_is_built(stand_in, tmp_path):
    some = ids_of(corpus='mendeley')[:2]
    with pytest.raises(SystemExit, match='matched is a training-time filter of the all build'):
        run(tmp_path / 'm', '--protocol', 'kfold_v3', '--arch', 'matched', '--ids', *some)
    with pytest.raises(SystemExit, match='kfold_v3 build is made for arch all'):
        run(tmp_path / 'x', '--protocol', 'kfold_v3', '--arch', 'x86', '--ids', *some)
    with pytest.raises(SystemExit, match='--out is required for a kfold_v3 all build'):
        build_b.main(['--protocol', 'kfold_v3', '--ids', *some])
    with pytest.raises(SystemExit, match='--out is required for a kfold all build'):
        build_b.main(['--protocol', 'kfold', '--arch', 'all', '--ids', *some])
    x64 = ids_of(arch='x64', corpus='mendeley')[:1]
    with pytest.raises(SystemExit, match=r'not kfold x86 ids: \[.*\]'):
        run(tmp_path / 'a', '--ids', *x64)
    no_fold = next(r['sample_id'] for r in tiny_rows() if r['fold_v3'] is None)
    with pytest.raises(SystemExit, match='not kfold_v3 all ids'):
        run(tmp_path / 'b', '--protocol', 'kfold_v3', '--ids', no_fold)
    with pytest.raises(SystemExit, match='not kfold all ids'):
        run(tmp_path / 'c', '--protocol', 'kfold', '--arch', 'all', '--ids', ids_of(corpus='ransomware_balanced')[0])
    assert not any((tmp_path / n).exists() for n in 'mxabc')


def tiny_rows():
    return load_index(tp.manifests.CACHE)


def test_a_missing_cache_names_the_build_command_of_the_dataset(tmp_path, monkeypatch):
    from common import manifests
    monkeypatch.setattr(manifests, 'CACHE', tmp_path / 'no-cache')
    for flags, name in ((['--protocol', 'kfold_v3'], 'combined_v3'), (['--protocol', 'kfold', '--arch', 'all'], 'mendeley_x86'), ([], 'mendeley_x86')):
        with pytest.raises(SystemExit, match=rf'index.csv is missing.*build_cache --cache {name}'):
            build_b.main(['--out', str(tmp_path / 'b'), *flags])
    assert not (tmp_path / 'b').exists()


def test_the_dataset_decides_the_ids_and_the_fold_recorded_for_each_binary(stand_in, tmp_path):
    _, _, rec = stand_in
    run(tmp_path / 'all', '--protocol', 'kfold_v3', '--ids', *tp.all_ids('kfold_v3', 'all'))
    index = {r['sample_id']: r for r in tiny_rows()}
    assert set(rec.meta) == set(tp.all_ids('kfold_v3', 'all')) and len(rec.meta) == 70
    for sample_id, meta in rec.meta.items():
        row = index[sample_id]
        assert meta == {'sample_id': sample_id, 'label': row['label'], 'arch': row['arch'], 'family': row['family'], 'fold': row['fold_v3']}
    rec.meta.clear()
    run(tmp_path / 'mendeley', '--protocol', 'kfold', '--arch', 'all')
    assert set(rec.meta) == set(tp.all_ids('kfold', 'all')) and len(rec.meta) == 48
    assert all(meta['fold'] == index[i]['fold'] for i, meta in rec.meta.items())
    assert {meta['arch'] for meta in rec.meta.values()} == {'x86', 'x64'}
    rec.meta.clear()
    run(tmp_path / 'x86')
    assert len(rec.meta) == 40 and all(meta['fold'] == index[i]['fold'] and meta['arch'] == 'x86' for i, meta in rec.meta.items())


def test_the_files_come_from_the_collection_of_the_corpus_of_the_row(stand_in, tmp_path):
    _, _, rec = stand_in
    ids = ids_of(corpus='ransomware_balanced')[:2] + ids_of(corpus='balanced_goodware')[:2] + ids_of(corpus='mendeley')[:2]
    run(tmp_path / 'out', '--protocol', 'kfold_v3', '--ids', *ids)
    rows = tp.sample_rows('kfold_v3', 'all')
    for sample_id in ids:
        call = [c for c in rec.paths if c[0] == sample_id]
        assert call == [(sample_id,) if rows[sample_id]['corpus'] == 'mendeley' else (sample_id, rows[sample_id]['corpus'])]


def test_max_instructions_excludes_the_larger_binaries_with_the_automatic_reason(stand_in, tmp_path):
    _, _, rec = stand_in
    rows = tp.sample_rows('kfold_v3', 'all')
    counts = sorted(int(r['num_instructions']) for r in rows.values())
    cap = counts[len(counts) // 2]
    big = {i for i, r in rows.items() if int(r['num_instructions']) > cap}
    assert 0 < len(big) < len(rows)
    out = tmp_path / 'capped'
    summary = run(out, '--protocol', 'kfold_v3', '--max-instructions', str(cap))
    reason = f'num_instructions > {cap}: load_cfg needs about 3 KB per instruction (31 GB RAM)'
    assert summary['excluded'] == {i: reason for i in sorted(big)} and json_of(out / 'settings.json')['excluded'] == summary['excluded']
    assert summary['processed'] + len(summary['excluded']) == summary['total'] == len(rows)
    assert set(rec.meta) == set(rows) - big and not any((out / f'{i}.pt').exists() for i in big)
    assert set(json_of(out / 'checkpoint.json')['done']) == set(rows) - big
    assert 'excluded' not in json_of(out / 'checkpoint.json')['settings']
    assert run(out, '--protocol', 'kfold_v3', '--max-instructions', str(cap))['built_this_run'] == 0           # the same cap on the resume
    assert train_b.read_excluded(out) == summary['excluded']
    for protocol, arch in (('kfold_v3', 'all'), ('kfold_v3', 'matched')):                      # the trainers' rule: in no split of any fold
        for fold in FOLDS:
            m = tp.fold_manifest(fold, 0, protocol, arch)
            left = {r['sample_id'] for r in m.train + m.val + m.test if r['sample_id'] not in train_b.read_excluded(out)}
            assert not left & big


def test_max_instructions_is_recorded_only_when_given_and_a_resume_must_pass_it_again(stand_in, tmp_path, capsys):
    rows = tp.sample_rows('kfold_v3', 'all')
    sizes = sorted(int(r['num_instructions']) for r in rows.values())
    cap, lower = sizes[len(sizes) // 2], sizes[len(sizes) // 4]
    out = tmp_path / 'capped'
    flags = ('--protocol', 'kfold_v3')
    run(out, *flags, '--max-instructions', str(cap))
    assert json_of(out / 'settings.json')['max_instructions'] == cap
    assert 'max_instructions' not in json_of(out / 'checkpoint.json')['settings']
    assert 'max_instructions' not in json_of(out / 'build_summary.json')['settings']
    assert 'WARNING' not in capsys.readouterr().err
    run(out, *flags, '--max-instructions', str(cap))                                         # the same cap: silent
    assert 'WARNING' not in capsys.readouterr().err
    summary = run(out, *flags, '--max-instructions', str(lower))                             # lower: warns, goes on, records the new cap
    assert f'--max-instructions {cap}' in capsys.readouterr().err and json_of(out / 'settings.json')['max_instructions'] == lower
    assert summary['processed'] + len(summary['excluded']) == summary['total']
    before = (out / 'settings.json').read_bytes()
    with pytest.raises(SystemExit) as refused:                                               # omitted: refused, nothing is rewritten
        run(out, *flags)
    assert f'records --max-instructions {lower}' in str(refused.value) and f'pass --max-instructions {lower}' in str(refused.value)
    assert '--max-instructions 0' in str(refused.value) and (out / 'settings.json').read_bytes() == before
    assert run(out, *flags, '--max-instructions', str(lower))['built_this_run'] == 0           # the recorded value: resumes, silent
    assert 'WARNING' not in capsys.readouterr().err and json_of(out / 'settings.json')['max_instructions'] == lower
    uncapped = run(out, *flags, '--max-instructions', '0')                                   # 0: the cap is dropped, the excluded ids are built
    assert f'--max-instructions {lower}, this run passes 0' in capsys.readouterr().err
    assert 'max_instructions' not in json_of(out / 'settings.json') and uncapped['excluded'] == {}
    assert uncapped['processed'] == uncapped['total'] == len(rows)
    run(out, *flags)                                                                         # no record any more: omitting it is as before
    assert 'WARNING' not in capsys.readouterr().err
    plain = tmp_path / 'plain'                                                               # a build without a cap neither records nor warns
    run(plain, *flags, '--ids', *list(rows)[:2])
    run(plain, *flags, '--ids', *list(rows)[:2], '--max-instructions', str(cap))
    assert 'WARNING' not in capsys.readouterr().err and json_of(plain / 'settings.json')['max_instructions'] == cap


def test_max_instructions_of_an_extension_is_recorded_and_checked_like_any_other(stand_in, tmp_path, capsys):
    base = base_build(tmp_path)
    extend(base, '--max-instructions', '10000000')
    assert json_of(base / 'settings.json')['max_instructions'] == 10000000 and 'WARNING' not in capsys.readouterr().err
    with pytest.raises(SystemExit, match='records --max-instructions 10000000'):
        extend(base)
    assert json_of(base / 'settings.json')['max_instructions'] == 10000000
    extend(base, '--max-instructions', '10000000')
    assert 'WARNING' not in capsys.readouterr().err
    extend(base, '--max-instructions', '0')
    assert '--max-instructions 10000000, this run passes 0' in capsys.readouterr().err
    assert 'max_instructions' not in json_of(base / 'settings.json')


def test_max_instructions_adds_to_exclude_and_leaves_built_binaries_alone(stand_in, tmp_path):
    rows = tp.sample_rows('kfold_v3', 'all')
    by_size = sorted(rows, key=lambda i: int(rows[i]['num_instructions']))
    small, large = by_size[:3], by_size[-3:]
    out = tmp_path / 'mixed'
    hand = by_size[0]
    summary = run(out, '--protocol', 'kfold_v3', '--ids', *small, *large, '--exclude', hand, '--exclude-reason', 'by hand',
                  '--max-instructions', str(int(rows[small[-1]]['num_instructions'])))
    assert summary['excluded'][hand] == 'by hand'
    assert {i for i, why in summary['excluded'].items() if why != 'by hand'} == {i for i in large if int(rows[i]['num_instructions']) >
                                                                                 int(rows[small[-1]]['num_instructions'])}
    # a lower cap on a resume leaves what the checkpoint holds built, and lists only what is still not built
    lower = run(out, '--protocol', 'kfold_v3', '--ids', *small, *large, '--exclude', hand, '--exclude-reason', 'by hand',
                '--max-instructions', str(int(rows[small[0]]['num_instructions'])))
    assert set(lower['excluded']) >= set(summary['excluded']) and set(small[1:]) <= set(json_of(out / 'checkpoint.json')['done'])
    assert not set(lower['excluded']) & set(json_of(out / 'checkpoint.json')['done'])
    assert lower['processed'] + len(lower['excluded']) == lower['total']


# ---- A3: --extend ------------------------------------------------------------------------------------------------------------

def x86_ids():
    return ids_of(arch='x86', corpus='mendeley')[:6]


def x64_ids():
    return ids_of(arch='x64', corpus='mendeley')[:4]


def base_build(tmp_path, *flags):
    base = tmp_path / 'base'
    run(base, '--ids', *x86_ids(), *flags)
    return base


def extend(out, *flags, ids=None):
    return run(out, '--protocol', 'kfold', '--arch', 'all', '--extend', '--ids', *(x86_ids() + x64_ids() if ids is None else ids), *flags)


def relabel(build, source, keep_pins=False):
    """Make a copy of a `repo` build look like a build of another source label, as the published builds do (a `main-fix` build's
    checkpoint settings carry no pins)."""
    for name in ('settings.json', 'checkpoint.json'):
        doc = json_of(build / name)
        target = doc['settings'] if name == 'checkpoint.json' else doc
        target['source'] = source
        if not keep_pins:
            if name == 'checkpoint.json':
                target.pop('pins', None)
        (build / name).write_text(json.dumps(doc), encoding='utf-8')


def weights(build):
    return torch.load(build / 'channel_embedder.pt')['api_embedding.weight']


def test_extend_adds_only_the_x64_ids_and_leaves_every_x86_payload_byte_identical(stand_in, tmp_path):
    real, _, rec = stand_in
    base = base_build(tmp_path)
    ext = tmp_path / 'ext'
    shutil.copytree(base, ext)
    rec.meta.clear()
    summary = extend(ext)
    assert summary['built_this_run'] == 4 and summary['processed'] == summary['total'] == 10 and set(rec.meta) == set(x64_ids())
    for sample_id in x86_ids():
        assert filecmp.cmp(base / f'{sample_id}.pt', ext / f'{sample_id}.pt', shallow=False), sample_id
    assert all((ext / f'{i}.pt').is_file() for i in x64_ids())
    assert set(json_of(ext / 'checkpoint.json')['done']) == set(x86_ids() + x64_ids())
    assert filecmp.cmp(base / 'settings.json', base / 'settings.json')                      # the source build is not touched
    assert json_of(base / 'checkpoint.json')['settings'].get('arch') is None and len(json_of(base / 'checkpoint.json')['done']) == 6
    settings = json_of(ext / 'settings.json')
    assert (settings['protocol'], settings['arch']) == ('kfold', 'all') and settings['source'] == 'repo'
    assert settings['pins'] == json_of(base / 'settings.json')['pins']
    (entry,) = settings['extensions']
    assert {k: v for k, v in entry.items() if k != 'date'} == {'arch': 'all', 'ids_added': 4, 'done_before': 6, 'source': 'repo', 'pins': dict(tp.source_pins('repo')),
                                                               'embedder_rows': 'seeded-per-id'}
    assert re.fullmatch(r'\d{4}-\d\d-\d\d', entry['date'])
    assert json_of(ext / 'checkpoint.json')['settings'] == {**json_of(base / 'checkpoint.json')['settings'], 'protocol': 'kfold', 'arch': 'all'}
    assert 'embedder_rows' not in settings                                                  # the original rows are the global generator's
    assert extend(ext)['built_this_run'] == 0                                               # a second run changes nothing
    assert len(json_of(ext / 'settings.json')['extensions']) == 1


def test_extend_draws_the_new_rows_per_api_id_and_keeps_the_old_ones(stand_in, tmp_path):
    real, _, _ = stand_in
    base = base_build(tmp_path)
    ext = tmp_path / 'ext'
    shutil.copytree(base, ext)
    extend(ext)
    old, new = weights(base), weights(ext)
    assert torch.equal(new[:len(old)], old) and len(new) == real.tok.FIRST_API_ID + 2 * (len(x86_ids()) + len(x64_ids())) > len(old)
    for row in range(len(old), len(new)):
        generator = torch.Generator().manual_seed(0 * 1_000_003 + row)
        assert torch.equal(new[row], torch.randn(new.shape[1], generator=generator)), row


def test_a_stopped_and_resumed_extension_equals_a_straight_one(stand_in, tmp_path):
    _, _, rec = stand_in
    base = base_build(tmp_path)
    straight, resumed = tmp_path / 'straight', tmp_path / 'resumed'
    shutil.copytree(base, straight)
    shutil.copytree(base, resumed)
    extend(straight)
    rec.fail = {x64_ids()[2]}
    with pytest.raises(SystemExit):
        extend(resumed)
    assert set(json_of(resumed / 'checkpoint.json')['done']) == set(x86_ids() + x64_ids()[:2])       # the checkpoint names the dataset
    assert json_of(resumed / 'checkpoint.json')['settings']['arch'] == 'all' and len(json_of(resumed / 'settings.json')['extensions']) == 1
    with pytest.raises(SystemExit, match='was built for kfold all, this run asks for kfold x86'):
        run(resumed, '--ids', *x86_ids())
    with pytest.raises(SystemExit, match='pass --extend to resume it'):
        run(resumed, '--protocol', 'kfold', '--arch', 'all', '--ids', *(x86_ids() + x64_ids()))
    rec.fail = set()
    assert extend(resumed)['built_this_run'] == 2
    assert len(json_of(resumed / 'settings.json')['extensions']) == 1
    a, b = torch.load(straight / 'channel_embedder.pt'), torch.load(resumed / 'channel_embedder.pt')
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    for sample_id in x86_ids() + x64_ids():
        assert torch.equal(torch.load(straight / f'{sample_id}.pt')['block_embeddings'], torch.load(resumed / f'{sample_id}.pt')['block_embeddings'])
    assert (straight / 'api_vocab.json').read_bytes() == (resumed / 'api_vocab.json').read_bytes()


def test_ids_added_of_a_resumed_extension_counts_the_ids_of_all_its_runs(stand_in, tmp_path):
    base = base_build(tmp_path)
    straight, resumed = tmp_path / 'straight', tmp_path / 'resumed'
    shutil.copytree(base, straight)
    shutil.copytree(base, resumed)
    extend(straight, ids=x86_ids() + x64_ids()[:3])
    assert extend(resumed, ids=x86_ids() + x64_ids()[:1])['built_this_run'] == 1
    (entry,) = json_of(resumed / 'settings.json')['extensions']
    assert entry['ids_added'] == 1 and entry['done_before'] == 6
    assert extend(resumed, ids=x86_ids() + x64_ids()[:3])['built_this_run'] == 2
    for build in (straight, resumed):
        assert json_of(build / 'settings.json')['extensions'][0]['ids_added'] == 3
    drop_date = lambda doc: {**doc, 'extensions': [{k: v for k, v in e.items() if k != 'date'} for e in doc['extensions']]}      # noqa: E731
    assert drop_date(json_of(straight / 'settings.json')) == drop_date(json_of(resumed / 'settings.json'))


def test_ids_added_follows_the_checkpoint_when_an_extension_stops(stand_in, tmp_path):
    _, _, rec = stand_in
    base = base_build(tmp_path)
    rec.fail = {x64_ids()[3]}
    with pytest.raises(SystemExit):
        extend(base)
    held = len(json_of(base / 'checkpoint.json')['done']) - 6
    assert json_of(base / 'settings.json')['extensions'][0]['ids_added'] == held < 3 + 1
    rec.fail = set()
    extend(base)
    assert json_of(base / 'settings.json')['extensions'][0]['ids_added'] == 4


def test_a_resumed_extension_must_run_the_files_it_began_with(stand_in, tmp_path, monkeypatch):
    _, _, rec = stand_in
    base = base_build(tmp_path)
    rec.fail = {x64_ids()[2]}
    with pytest.raises(SystemExit):
        extend(base)
    rec.fail = set()
    pins = dict(tp.source_pins('repo'))
    first = sorted(pins)[0]
    monkeypatch.setattr(build_b, 'source_pins', lambda source: {**pins, first: '0' * 64})
    with pytest.raises(SystemExit, match=r'extension began with other files.*' + re.escape(first.split('/')[-1])):
        extend(base)


@pytest.mark.parametrize('label', ['main-fix', 'repo'])
def test_extend_continues_a_build_made_with_the_files_of_this_repository(stand_in, tmp_path, label):
    base = base_build(tmp_path)
    original = json_of(base / 'settings.json')
    relabel(base, label)
    summary = extend(base)
    assert summary['built_this_run'] == 4
    settings = json_of(base / 'settings.json')
    assert settings['source'] == label and settings['pins'] == original['pins']
    assert settings['extensions'][0]['source'] == 'repo' and settings['extensions'][0]['pins'] == dict(tp.source_pins('repo'))
    assert json_of(base / 'checkpoint.json')['settings']['source'] == label
    assert train_b.read_source(base) == label and train_b.build_pins(base, label) == original['pins']      # still read as what it was


@pytest.mark.parametrize('label', ['gnn-331ce0b', 'main-c7899f0'])
def test_extend_refuses_a_build_made_with_a_teammate_source(stand_in, tmp_path, label):
    base = base_build(tmp_path)
    relabel(base, label)
    with pytest.raises(SystemExit, match=rf'source repo or main-fix.*the build has the source {label}'):
        extend(base)
    assert 'extensions' not in json_of(base / 'settings.json') and len(json_of(base / 'checkpoint.json')['done']) == 6
    clean = base_build(tmp_path / 'clean')
    with pytest.raises(SystemExit, match=rf'source repo or main-fix.*this run has the source {label}'):
        build_b.main(['--out', str(clean), '--source', label, '--protocol', 'kfold', '--arch', 'all', '--extend', '--ids', *x86_ids()])


def test_extend_refuses_everything_but_a_finished_kfold_x86_build_and_kfold_all(stand_in, tmp_path):
    base = base_build(tmp_path)
    with pytest.raises(SystemExit, match='no checkpoint.json'):
        extend(tmp_path / 'empty')
    with pytest.raises(SystemExit, match='matched is a training-time filter'):
        run(base, '--protocol', 'kfold_v3', '--arch', 'matched', '--extend', '--ids', *x86_ids())
    with pytest.raises(SystemExit, match='pass --protocol kfold --arch all'):
        run(base, '--extend', '--ids', *x86_ids())                                             # the x86 dataset has nothing to add
    with pytest.raises(SystemExit, match='pass --protocol kfold --arch all'):
        run(base, '--protocol', 'kfold_v3', '--extend', '--ids', *x86_ids())
    with pytest.raises(SystemExit, match='do not pass --seeded-rows'):
        extend(base, '--seeded-rows')
    with pytest.raises(SystemExit, match='was built with'):                                    # another sampler: the settings must agree
        extend(base, '--num-blocks', '250')
    v3 = tmp_path / 'v3'
    run(v3, '--protocol', 'kfold_v3', '--ids', *x86_ids())
    with pytest.raises(SystemExit, match='was built for kfold_v3 all: --extend grows a kfold x86 build'):
        extend(v3)
    assert 'extensions' not in json_of(base / 'settings.json') and len(json_of(base / 'checkpoint.json')['done']) == 6


def test_extend_needs_an_embedder_that_takes_a_seed(stand_in, tmp_path, monkeypatch):
    base = base_build(tmp_path)

    class Old:
        FiveChannelEmbedder = type('E', (), {'__init__': lambda self, size: None})
    monkeypatch.setattr(build_b, 'import_team', lambda *args, **kwargs: Old())
    with pytest.raises(SystemExit, match='--extend needs an embedder that takes a seed'):
        extend(base)


def test_extend_of_a_seeded_build_keeps_its_embedder_rows_key(stand_in, tmp_path):
    base = base_build(tmp_path, '--seeded-rows')
    summary = extend(base)
    assert summary['built_this_run'] == 4
    settings = json_of(base / 'settings.json')
    assert settings['embedder_rows'] == 'seeded-per-id' and settings['extensions'][0]['embedder_rows'] == 'seeded-per-id'
    assert json_of(base / 'checkpoint.json')['settings']['embedder_rows'] == 'seeded-per-id'


# ---- A3: the published builds (real_data: GT_DATASETS or GT_CACHE for the caches, GT_RUNS for the builds, GT_COLLECTIONS) ---------

B_SIG500_MAIN, B_SIG500_V3 = RUNS / 'team' / 'b_sig500_main', RUNS / 'team' / 'b_sig500_v3'
needs_builds = pytest.mark.skipif(not (B_SIG500_MAIN / 'checkpoint.json').is_file() or not (B_SIG500_V3 / 'checkpoint.json').is_file(),
                                  reason='needs the published builds b_sig500_main and b_sig500_v3 under GT_RUNS/team')


def _collections_missing():
    from common.paths import MissingSource, source_for
    try:
        return not all((source_for('collections') / name / 'samples.csv').is_file() for name in ('Mendeley_Ransomware', 'Mendeley_Goodware'))
    except MissingSource:
        return True


needs_mendeley_collections = pytest.mark.skipif(_collections_missing(), reason='needs the Mendeley collections (GT_COLLECTIONS)')


def real(fn):
    """A test on the published caches and builds of this machine; it skips without them."""
    for mark in (real_data, needs_v2, needs_builds, needs_mendeley_collections):
        fn = mark(fn)
    return pytest.mark.usefixtures('real_cache')(fn)


@pytest.fixture
def real_cache(monkeypatch):
    from common import manifests
    monkeypatch.setattr(manifests, 'CACHE', Path(DEFAULT_CACHE))


@real
def test_a_limit_3_build_has_the_x86_settings_and_the_tensors_of_the_published_build(tmp_path):
    """`build_b --limit 3 --source repo` (signal 500, seed 0) writes the settings of an x86 build, no `protocol` or `arch` key, and
    tensors that are `torch.equal` to those of the same three ids in `b_sig500_main`, a `repo` build of the same sampler."""
    out = tmp_path / 'limit3'
    build_b.main(['--limit', '3', '--source', 'repo', '--out', str(out)])
    settings = json_of(out / 'settings.json')
    assert set(settings) == {'sample', 'num_blocks', 'seed', 'source', 'pins', 'excluded'} and settings['excluded'] == {}
    assert json_of(out / 'checkpoint.json')['settings'] == {k: settings[k] for k in ('sample', 'num_blocks', 'seed', 'source', 'pins')}
    ids = json_of(out / 'checkpoint.json')['done']
    assert len(ids) == 3
    result = compare_builds.compare(B_SIG500_MAIN, out, ids, state=True)
    assert result['identical'] == 3 and result['max_block_embedding_diff'] == 0.0, result['differing_arrays']
    for sample_id in ids:
        a, b = compare_builds.load(B_SIG500_MAIN, sample_id), compare_builds.load(out, sample_id)
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a if isinstance(a[k], torch.Tensor)), sample_id


@real
def test_three_x86_ids_rebuilt_frozen_from_a_published_build_equal_its_tensors(tmp_path):
    """The evidence that the files of this repository may extend a `main-fix` build: `--frozen-from` the published `b_sig500_v3`,
    three of its ids come out `torch.equal` (the vocabulary and the embedder are loaded, not grown)."""
    ids = sorted(json_of(B_SIG500_V3 / 'checkpoint.json')['done'])[::600][:3]
    out = tmp_path / 'frozen'
    build_b.main(['--source', 'repo', '--frozen-from', str(B_SIG500_V3), '--ids', *ids, '--out', str(out)])
    result = compare_builds.compare(B_SIG500_V3, out, ids, state=True)
    assert result['identical'] == 3 and result['max_block_embedding_diff'] == 0.0, result['differing_arrays']


@real
def test_five_x64_ids_extend_a_copy_of_a_published_build(tmp_path, capsys):
    """`--extend` on a scratch copy of `b_sig500_v3` (its checkpoint, vocabulary and embedder, three x86 payloads): five Mendeley x64
    ids are built, the x86 payloads and rows are untouched, `check_repool` holds on one x86 and one x64 payload, and the match
    rates of both architectures are printed."""
    x64 = sorted((r for r in tp.sample_rows('kfold', 'all').values() if r['arch'] == 'x64'), key=lambda r: int(r['num_instructions']))
    ids = [x64[k]['sample_id'] for k in (120, 240, 360, 480, 590)]
    x86 = sorted(json_of(B_SIG500_V3 / 'checkpoint.json')['done'])[::600][:3]
    copy = tmp_path / 'copy'
    copy.mkdir()
    for name in ('settings.json', 'checkpoint.json', 'api_vocab.json', 'channel_embedder.pt', 'build_log.csv'):
        shutil.copy2(B_SIG500_V3 / name, copy / name)
    for sample_id in x86:
        for suffix in ('.pt', '.manifest.json'):
            shutil.copy2(B_SIG500_V3 / f'{sample_id}{suffix}', copy / f'{sample_id}{suffix}')
    rows_before = torch.load(copy / 'channel_embedder.pt')['api_embedding.weight']
    summary = build_b.main(['--protocol', 'kfold', '--arch', 'all', '--extend', '--source', 'repo', '--ids', *ids, '--out', str(copy)])
    assert summary['built_this_run'] == 5
    for sample_id in x86:
        assert filecmp.cmp(B_SIG500_V3 / f'{sample_id}.pt', copy / f'{sample_id}.pt', shallow=False)
    after = torch.load(copy / 'channel_embedder.pt')['api_embedding.weight']
    assert torch.equal(after[:len(rows_before)], rows_before) and len(after) >= len(rows_before)
    settings = json_of(copy / 'settings.json')
    assert settings['source'] == 'main-fix' and settings['extensions'][0]['ids_added'] == 5 and (settings['protocol'], settings['arch']) == ('kfold', 'all')
    pp, embedder, _ = train_b.build_state(copy)
    for sample_id in (x86[0], ids[0]):
        payload = torch.load(copy / f'{sample_id}.pt', map_location='cpu', weights_only=False)
        assert train_b.check_repool(pp, embedder, payload) <= 1e-5
    rates = {arch: [match_rates.sample_rates(copy, [i])[i][1] for i in group] for arch, group in (('x86', x86), ('x64', ids))}
    with capsys.disabled():
        print('\nmatch rate per arch (mean, n):', {a: (round(sum(v) / len(v), 4), len(v)) for a, v in rates.items()}, rates)
