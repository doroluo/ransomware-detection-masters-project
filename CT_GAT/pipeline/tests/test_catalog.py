"""common/catalog.py: the table of collections, corpora and datasets, the merged registry of common/paths.py, the source
defaults, and `python -m common.catalog`. Synthetic folders in tmp_path only (a `samples/` folder under the repository would
be gitignored)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from common import catalog, paths
from common.paths import MissingSource, cache_for, registry, source_for

ROOT = Path(__file__).resolve().parents[1]


def file_env(tmp_path, doc):
    path = tmp_path / 'datasets.json'
    path.write_text(json.dumps(doc), encoding='utf-8')
    return {'GT_DATASETS': str(path)}


# ---- the table ---------------------------------------------------------------------------------------------------------

def test_every_corpus_collection_is_a_known_collection():
    assert all(set(c.collections) <= set(catalog.COLLECTIONS) for c in catalog.CORPORA.values())
    assert set(catalog.COLLECTIONS) == {f for c in catalog.CORPORA.values() for f in c.collections}     # and no folder is unused
    assert {c.rule for c in catalog.CORPORA.values()} == {'split', 'eligible'}


def test_every_dataset_names_known_corpora_a_known_base_and_existing_fold_files():
    for name, ds in catalog.DATASETS.items():
        assert set(ds.corpora) <= set(catalog.CORPORA), name
        assert ds.base is None or ds.base in catalog.DATASETS, name
        assert set(ds.folds) <= set(paths.SOURCES), name
        for key, rel in ds.folds.items():
            path = catalog.FOLDS_ROOT / rel
            assert path.exists(), (name, key, path)
            if key == 'folds_v3':
                assert all((path / f).is_file() for f in catalog.V3_FILES)


def test_the_roots_are_computed_and_nested():
    assert catalog.PIPELINE_ROOT == ROOT and catalog.CT_GAT == ROOT.parent
    assert catalog.DATA_ROOT == ROOT.parent / 'data' and catalog.CACHE_ROOT == ROOT.parent / 'cache'
    assert catalog.FOLDS_ROOT == ROOT / 's02_preprocess' / 'folds' and catalog.PROCESSED == ROOT.parent / 'processed'
    assert paths.PIPELINE_ROOT == catalog.PIPELINE_ROOT


def test_there_is_one_collections_source():
    assert paths.SOURCES['collections'][0] == 'GT_COLLECTIONS' and 'GT_MENDELEY' not in ' '.join(v for v, _ in paths.SOURCES.values())


# ---- the merged registry -----------------------------------------------------------------------------------------------

def test_without_a_file_the_registry_is_the_catalog():
    env = {'GT_DATASETS': 'nowhere.json'}
    merged = registry(env)
    assert list(merged) == list(catalog.DATASETS)
    for name, ds in catalog.DATASETS.items():
        assert merged[name]['cache'] == catalog.CACHE_ROOT / name and merged[name]['protocols'] == ds.protocols
        assert merged[name]['sources']['folds'] == catalog.FOLDS_ROOT / 'folds_mendeley.csv'
        assert merged[name]['sources']['folds_v3'] == catalog.FOLDS_ROOT / 'v3'
        assert merged[name]['sources']['collections'] == catalog.DATA_ROOT
        assert merged[name]['sources']['team_b_build'] == catalog.PROCESSED / 'cfg_subgraphs'
        assert not {'bundle', 'reader_root', 'runs'} & set(merged[name]['sources'])
    assert cache_for('bundle', env) == cache_for('kfold', env) == catalog.CACHE_ROOT / 'mendeley_x86'
    assert cache_for('kfold_v3', env) == catalog.CACHE_ROOT / 'combined_v3'
    assert cache_for('kfold_v3', env, name='mendeley_x86') == catalog.CACHE_ROOT / 'mendeley_x86'
    assert cache_for('nobody', env) == paths.UNSET


def test_file_entries_come_first_in_file_order_then_the_catalog_fills_the_rest(tmp_path):
    env = file_env(tmp_path, {'zeta': {'cache': 'z', 'protocols': ['kfold']}, 'alpha': {'cache': 'a', 'protocols': ['kfold_v3']}})
    merged = registry(env)
    assert list(merged) == ['zeta', 'alpha', 'mendeley_x86', 'combined_v3']
    assert cache_for('kfold', env) == tmp_path / 'z'                                    # the file's dataset wins the protocol
    assert cache_for('kfold_v3', env) == tmp_path / 'a'
    assert cache_for('bundle', env) == catalog.CACHE_ROOT / 'mendeley_x86'              # nobody in the file lists it
    assert cache_for('kfold', env, name='combined_v3') == catalog.CACHE_ROOT / 'combined_v3'


def test_a_file_entry_with_a_catalog_name_replaces_the_catalog_one(tmp_path):
    env = file_env(tmp_path, {'combined_v3': {'cache': 'mine', 'protocols': ['kfold_v3', 'kfold'],
                                              'sources': {'collections': 'coll'}}})
    merged = registry(env)
    assert list(merged) == ['combined_v3', 'mendeley_x86']
    assert merged['combined_v3']['cache'] == tmp_path / 'mine' and merged['combined_v3']['protocols'] == ('kfold_v3', 'kfold')
    assert merged['combined_v3']['sources'] == {'collections': tmp_path / 'coll'}       # replaced, not merged
    assert source_for('collections', env) == tmp_path / 'coll'
    assert source_for('folds', env) == catalog.FOLDS_ROOT / 'folds_mendeley.csv'        # the other catalog dataset still defaults it


# ---- source defaults ---------------------------------------------------------------------------------------------------

def test_defaults_apply_only_when_neither_the_environment_nor_the_file_sets_the_key(tmp_path):
    silent = {'GT_DATASETS': str(tmp_path / 'none.json')}
    assert source_for('folds', silent) == catalog.FOLDS_ROOT / 'folds_mendeley.csv'
    assert source_for('folds_v3', silent) == catalog.FOLDS_ROOT / 'v3'
    assert source_for('collections', silent) == catalog.DATA_ROOT
    assert source_for('team_b_build', silent) == catalog.PROCESSED / 'cfg_subgraphs'
    assert source_for('runs', silent) == paths.PIPELINE_ROOT / 'runs'
    for key in ('bundle', 'reader_root'):
        with pytest.raises(MissingSource, match=paths.SOURCES[key][0]):
            source_for(key, silent)
    file_env_ = file_env(tmp_path, {'v2': {'cache': 'c', 'sources': {'folds': 'f.csv', 'collections': 'coll'}}})
    assert source_for('folds', file_env_) == tmp_path / 'f.csv'                         # the file beats the default
    assert source_for('collections', file_env_) == tmp_path / 'coll'
    assert source_for('folds_v3', file_env_) == catalog.FOLDS_ROOT / 'v3'               # keys the file leaves silent default
    assert source_for('folds', {**file_env_, 'GT_FOLDS': 'env.csv'}) == Path('env.csv')  # the environment beats both
    assert source_for('collections', {**silent, 'GT_COLLECTIONS': 'envcoll'}) == Path('envcoll')


# ---- status and the command --------------------------------------------------------------------------------------------

def write_collection(root, name, snapshot=None):
    (root / name).mkdir(parents=True)
    (root / name / 'samples.csv').write_text('sample_id\n', encoding='utf-8')
    if snapshot:
        (root / name / 'summary.json').write_text(json.dumps({'schema': 'collection-cfg-asm/1', 'snapshot_utc': snapshot}), encoding='utf-8')


def test_status_on_a_data_root_with_one_of_two_folders(tmp_path):
    data = tmp_path / 'data'
    write_collection(data, 'Mendeley_Ransomware', '20261008T184757Z')
    (data / 'Mendeley_Goodware').mkdir()                                                # a folder without samples.csv is missing
    caches = tmp_path / 'cache'
    (caches / 'mendeley_x86').mkdir(parents=True)
    (caches / 'mendeley_x86' / 'index.csv').write_text('sample_id\n', encoding='utf-8')
    report = catalog.status(data, caches, folds_root=tmp_path / 'nofolds')
    mendeley = report['mendeley_x86']
    assert mendeley['collections'] == {'Mendeley_Ransomware': {'present': True, 'snapshot': '20261008T184757Z'},
                                       'Mendeley_Goodware': {'present': False, 'snapshot': None}}
    assert mendeley['bundle'] == ('absent', None)
    assert mendeley['cache'] == caches / 'mendeley_x86' and mendeley['built'] is True
    assert mendeley['folds']['folds'] == (tmp_path / 'nofolds' / 'folds_mendeley.csv', False)
    combined = report['combined_v3']
    assert list(combined['collections']) == ['Mendeley_Ransomware', 'Mendeley_Goodware', 'Ransomware_Balanced', 'Balanced_Goodware']
    assert [c['present'] for c in combined['collections'].values()] == [True, False, False, False]
    assert combined['built'] is False
    assert all(ok for _, ok in catalog.status()['mendeley_x86']['folds'].values())      # the committed ones are found
    text = catalog.render(report, data)
    assert 'Mendeley_Ransomware: present (snapshot 20261008T184757Z)' in text and 'Ransomware_Balanced: missing' in text
    assert 'mendeley_x86: protocols bundle, kfold' in text and 'cache CT_GAT/cache/mendeley_x86' not in text   # the test cache is under tmp
    assert text.count(': built') == 1 and text.count('not built') == 1


def test_status_bundle_kinds(tmp_path):
    data = tmp_path / 'data'
    bundle = data / 'Mendeley_GNN_v1'
    bundle.mkdir(parents=True)
    (bundle / 'summary.json').write_text(json.dumps({'schema': 'something-else/1'}), encoding='utf-8')
    assert catalog.status(data)['mendeley_x86']['bundle'] == ('absent', None)
    (bundle / 'summary.json').write_text(json.dumps({'schema': catalog.BUNDLE_SCHEMA}), encoding='utf-8')
    assert catalog.status(data)['mendeley_x86']['bundle'] == ('discovered', bundle)
    assert catalog.status(data, bundle=tmp_path / 'mine')['mendeley_x86']['bundle'] == ('configured', tmp_path / 'mine')


def test_the_command_prints_every_dataset_and_exits_zero(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GT_')}
    env.update(GT_DATASETS=str(tmp_path / 'none.json'), GT_COLLECTIONS=str(tmp_path / 'empty'), PYTHONDONTWRITEBYTECODE='1')
    r = subprocess.run([sys.executable, '-m', 'common.catalog'], cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    for name in catalog.DATASETS:
        assert f'{name}:' in r.stdout
    assert 'Mendeley_Ransomware: missing' in r.stdout and 'bundle: absent' in r.stdout and 's02_preprocess/folds/folds_mendeley.csv: found' in r.stdout
    assert 'cache CT_GAT/cache/mendeley_x86: ' in r.stdout
