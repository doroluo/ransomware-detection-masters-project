"""build_cache and folds driven by common/catalog.py: `--cache <dataset>`, the corpora and base defaults, the guard, the missing
collection message, `exclusion` by the corpus's rule, the bundle discovery order. Synthetic folders in tmp_path only (a
`samples/` folder under the repository would be gitignored); nothing builds a cache."""
import csv
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from common import catalog, paths
from common.paths import MissingSource
from s02_preprocess.cache import build_cache as bc
from s02_preprocess.cache import folds

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['sample_id', 'label', 'family', 'representation', 'recommended_split', 'architecture', 'eligible_for_gnn', 'cfg_path']


@pytest.fixture
def env(tmp_path, monkeypatch):
    """No GT_* variable and no datasets.json; the data root is tmp_path/data."""
    for var in [v for v in os.environ if v.startswith('GT_')]:
        monkeypatch.delenv(var)
    monkeypatch.setenv('GT_DATASETS', str(tmp_path / 'none.json'))
    monkeypatch.setenv('GT_COLLECTIONS', str(tmp_path / 'data'))
    return tmp_path


def collection(data, name, rows=()):
    folder = data / name
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'samples.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return folder


def run_module(tmp_path, *args, **extra):
    """`python -m s02_preprocess.cache.build_cache` in a subprocess with the GT_* of `extra` only."""
    base = {k: v for k, v in os.environ.items() if not k.startswith('GT_')}
    base.update(GT_DATASETS=str(tmp_path / 'none.json'), GT_COLLECTIONS=str(tmp_path / 'data'), PYTHONDONTWRITEBYTECODE='1',
                PYTHONPATH=str(ROOT))
    base.update(extra)
    return subprocess.run([sys.executable, '-m', 's02_preprocess.cache.build_cache', *args], cwd=tmp_path, capture_output=True,
                          text=True, env=base)


def dry(tmp_path, *args, **extra):
    r = run_module(tmp_path, '--dry', *args, **extra)
    assert r.returncode == 0, r.stderr
    return dict(line.split(': ', 1) for line in r.stdout.splitlines())


# ---- --cache selection, corpora and base defaults ------------------------------------------------------------------------

def test_without_a_flag_the_dataset_is_the_one_the_cache_resolved_to(tmp_path):
    out = dry(tmp_path)
    assert out['dataset'] == 'mendeley_x86' and out['corpora'] == 'mendeley' and out['base cache'] == '-'
    assert out['cache'].endswith('CT_GAT/cache/mendeley_x86')
    out = dry(tmp_path, GT_CACHE=str(tmp_path / 'elsewhere'))              # a directory of no dataset
    assert out['dataset'] == '-' and out['corpora'] == 'mendeley' and out['cache'] == (tmp_path / 'elsewhere').as_posix()
    out = dry(tmp_path, GT_CACHE=str(catalog.CACHE_ROOT / 'combined_v3'))  # a directory that is a dataset's cache
    assert out['dataset'] == 'combined_v3' and out['corpora'] == 'mendeley ransomware_balanced balanced_goodware'


def test_cache_selects_a_catalog_dataset_with_its_corpora_and_base(tmp_path):
    out = dry(tmp_path, '--cache', 'combined_v3')
    assert out['dataset'] == 'combined_v3' and out['cache'].endswith('CT_GAT/cache/combined_v3')
    assert out['corpora'] == 'mendeley ransomware_balanced balanced_goodware'
    assert out['base cache'].endswith('CT_GAT/cache/mendeley_x86')
    assert out['collections'] == (tmp_path / 'data').as_posix()
    assert out['folds'].endswith('s02_preprocess/folds/folds_mendeley.csv') and out['folds_v3'].endswith('s02_preprocess/folds/v3')


def test_explicit_corpora_and_base_cache_win(tmp_path):
    out = dry(tmp_path, '--cache', 'combined_v3', '--corpora', 'balanced_goodware', 'mendeley', '--base-cache', str(tmp_path / 'b'))
    assert out['corpora'] == 'balanced_goodware mendeley' and out['base cache'] == (tmp_path / 'b').as_posix()


def test_an_unknown_dataset_is_an_argparse_error_naming_the_choices(tmp_path):
    r = run_module(tmp_path, '--cache', 'nope')
    assert r.returncode == 2 and 'invalid choice' in r.stderr and 'mendeley_x86' in r.stderr and 'combined_v3' in r.stderr


def test_help_lists_the_catalog_and_datasets_json_names(tmp_path):
    (tmp_path / 'datasets.json').write_text(json.dumps({'v2': {'cache': 'v2'}}), encoding='utf-8')
    r = run_module(tmp_path, '--help', GT_DATASETS=str(tmp_path / 'datasets.json'))
    text = ' '.join(r.stdout.split())
    assert r.returncode == 0 and 'mendeley_x86, combined_v3' in text and 'datasets.json (v2)' in text


def test_a_datasets_json_entry_declares_no_corpora_so_it_builds_mendeley_without_a_base(tmp_path):
    doc = {'v2': {'cache': 'cv2', 'protocols': ['bundle', 'kfold']}, 'combined': {'cache': 'cc', 'protocols': ['kfold_v3']}}
    file = tmp_path / 'datasets.json'
    file.write_text(json.dumps(doc), encoding='utf-8')
    for args, name in (((), 'v2'), (('--cache', 'combined'), 'combined'), (('--cache', 'v2'), 'v2')):
        out = dry(tmp_path, *args, GT_DATASETS=str(file))
        assert out['dataset'] == name and out['corpora'] == 'mendeley' and out['base cache'] == '-'
    assert dry(tmp_path, '--cache', 'combined', GT_DATASETS=str(file))['cache'] == (tmp_path / 'cc').as_posix()
    assert dry(tmp_path, '--cache', 'combined_v3', GT_DATASETS=str(file))['cache'].endswith('CT_GAT/cache/combined_v3')   # the catalog's, after the file's


def test_a_file_entry_named_like_a_catalog_dataset_replaces_its_corpora_and_cache(tmp_path):
    file = tmp_path / 'datasets.json'
    file.write_text(json.dumps({'combined_v3': {'cache': 'mine'}}), encoding='utf-8')
    out = dry(tmp_path, '--cache', 'combined_v3', GT_DATASETS=str(file))
    assert out['cache'] == (tmp_path / 'mine').as_posix() and out['corpora'] == 'mendeley' and out['base cache'] == '-'


# ---- the guard -----------------------------------------------------------------------------------------------------------

def plan(env_, name=None, corpora=None, base=None, cache=None):
    reg = paths.registry()
    cache = cache or (reg[name]['cache'] if name else reg['mendeley_x86']['cache'])
    return bc.resolve_build(name, corpora, base, cache, reg)


def test_a_dataset_never_gets_corpora_beyond_its_own(env):
    assert plan(env, 'mendeley_x86')['corpora'] == ['mendeley']
    assert plan(env, 'mendeley_x86', ['mendeley'])['corpora'] == ['mendeley']
    with pytest.raises(SystemExit, match='mendeley_x86 holds mendeley: corpora ransomware_balanced'):
        plan(env, 'mendeley_x86', ['mendeley', 'ransomware_balanced'])
    with pytest.raises(SystemExit, match='corpora balanced_goodware do not go into'):
        plan(env, None, ['balanced_goodware'])                                     # no --cache: the default dataset
    assert plan(env, 'combined_v3', ['ransomware_balanced'])['corpora'] == ['ransomware_balanced']      # a subset is allowed
    assert plan(env, 'combined_v3')['corpora'] == list(catalog.DATASETS['combined_v3'].corpora)


def test_a_cache_of_no_dataset_takes_any_corpora(env):
    elsewhere = env / 'elsewhere'
    assert plan(env, None, ['mendeley', 'balanced_goodware'], cache=elsewhere)['corpora'] == ['mendeley', 'balanced_goodware']
    assert plan(env, None, cache=elsewhere)['corpora'] == ['mendeley'] and plan(env, None, cache=elsewhere)['dataset'] is None


def test_the_default_cache_of_a_configured_machine_stays_mendeley_only_and_its_other_caches_do_not(tmp_path, monkeypatch):
    file = tmp_path / 'datasets.json'
    file.write_text(json.dumps({'v2': {'cache': 'cv2', 'protocols': ['bundle', 'kfold']},
                                'combined': {'cache': 'cc', 'protocols': ['kfold_v3']}}), encoding='utf-8')
    monkeypatch.setenv('GT_DATASETS', str(file))
    reg = paths.registry()
    both = ['mendeley', 'ransomware_balanced']
    default = paths.cache_for('bundle', env={}, registry=reg)
    assert default == tmp_path / 'cv2'
    monkeypatch.setattr(bc, 'DEFAULT_CACHE', default)
    with pytest.raises(SystemExit, match='default cache stays Mendeley-only'):
        bc.resolve_build('v2', both, None, default, reg)
    assert bc.resolve_build('combined', both, None, tmp_path / 'cc', reg)['corpora'] == both   # as before the catalog
    assert bc.resolve_build(None, both, None, tmp_path / 'new', reg)['corpora'] == both


# ---- missing folders and the base cache ----------------------------------------------------------------------------------

def test_missing_collections_are_all_named_with_their_resolved_path_and_zip(env):
    data = env / 'data'
    collection(data, 'Mendeley_Ransomware')
    collection(data, 'Mendeley_Goodware')
    bc.require_collections(['mendeley'])                                             # nothing missing
    with pytest.raises(MissingSource) as e:
        bc.require_collections(['mendeley', 'ransomware_balanced', 'balanced_goodware'])
    text = str(e.value)
    assert f'{data.as_posix()}/Ransomware_Balanced is missing: unzip Collections/Ransomware_Balanced.zip' in text
    assert f'{data.as_posix()}/Balanced_Goodware is missing: unzip Collections/Balanced_Goodware.zip' in text
    collection(data, 'Ransomware_Balanced')
    with pytest.raises(MissingSource) as e:
        bc.require_collections(['ransomware_balanced', 'balanced_goodware'])
    assert 'Ransomware_Balanced' not in str(e.value) and 'Balanced_Goodware.zip' in str(e.value)


def test_a_nested_or_still_zipped_collection_says_so_and_a_plain_missing_one_keeps_the_old_text(env):
    data = env / 'data'
    collection(data / 'Ransomware_Balanced', 'Ransomware_Balanced')                  # one folder too deep
    (data / 'Balanced_Goodware.zip').write_bytes(b'PK')                               # not unzipped
    with pytest.raises(MissingSource) as e:
        bc.require_collections(['ransomware_balanced', 'balanced_goodware'])
    text = str(e.value)
    assert f'{data.as_posix()}/Ransomware_Balanced/Ransomware_Balanced/samples.csv is one folder too deep' in text
    assert f'move the inner Ransomware_Balanced folder up so that {data.as_posix()}/Ransomware_Balanced/samples.csv exists' in text
    assert f'{data.as_posix()}/Balanced_Goodware.zip is not unzipped: unzip it into {data.as_posix()}' in text
    assert 'unzip Collections/' not in text


def test_a_missing_mendeley_corpus_names_a_nested_folder_and_a_zip_left_unextracted(env):
    data = env / 'data'
    collection(data / 'Mendeley_Ransomware', 'Mendeley_Ransomware')
    (data / 'Mendeley_Goodware.zip').write_bytes(b'PK')
    with pytest.raises(MissingSource) as e:
        bc.require_collections(['mendeley'])
    text = str(e.value)
    assert 'GT_BUNDLE' in text and 'Mendeley_Ransomware/Mendeley_Ransomware/samples.csv is one folder too deep' in text
    assert 'Mendeley_Goodware.zip is not unzipped' in text


def test_a_missing_mendeley_corpus_keeps_the_bundle_or_collections_message(env):
    with pytest.raises(MissingSource, match='GT_BUNDLE.*Mendeley_Ransomware and Mendeley_Goodware'):
        bc.require_collections(['mendeley'])


def test_a_combined_build_without_the_balanced_folders_stops_before_the_cache_exists(tmp_path):
    data = tmp_path / 'data'
    collection(data, 'Mendeley_Ransomware')
    collection(data, 'Mendeley_Goodware')
    r = run_module(tmp_path, '--cache', 'combined_v3', '--limit', '4', GT_CACHE=str(tmp_path / 'cache'))
    assert r.returncode == 1 and 'Traceback' not in r.stderr and 'MissingSource' not in r.stderr
    assert f'{data.as_posix()}/Ransomware_Balanced is missing: unzip Collections/Ransomware_Balanced.zip' in r.stderr
    assert 'Balanced_Goodware.zip' in r.stderr
    assert not (tmp_path / 'cache').exists()


def test_a_combined_build_whose_base_is_not_built_names_the_command_that_builds_it(tmp_path):
    data = tmp_path / 'data'
    for name in catalog.COLLECTIONS:
        collection(data, name)
    file = tmp_path / 'datasets.json'                                              # mendeley_x86 in tmp_path, not built
    file.write_text(json.dumps({'mendeley_x86': {'cache': 'base'}}), encoding='utf-8')
    r = run_module(tmp_path, '--cache', 'combined_v3', GT_DATASETS=str(file), GT_CACHE=str(tmp_path / 'cache'))
    assert r.returncode != 0 and 'has no index.csv' in r.stderr
    assert 'python -m s02_preprocess.cache.build_cache --cache mendeley_x86' in r.stderr and (tmp_path / 'base').as_posix() in r.stderr
    assert not (tmp_path / 'cache').exists() and not (tmp_path / 'base').exists()
    r = run_module(tmp_path, '--cache', 'combined_v3', '--base-cache', str(tmp_path / 'else'), GT_DATASETS=str(file),
                   GT_CACHE=str(tmp_path / 'cache'))                                  # an explicit base is checked the same way
    assert r.returncode != 0 and (tmp_path / 'else').as_posix() in r.stderr and 'python -m s02_preprocess.cache.build_cache' in r.stderr
    (tmp_path / 'base').mkdir()
    (tmp_path / 'base' / 'index.csv').write_text('sample_id\n', encoding='utf-8')
    r = run_module(tmp_path, '--cache', 'combined_v3', GT_DATASETS=str(file), GT_CACHE=str(tmp_path / 'cache'))
    assert 'has no index.csv' not in r.stderr                                          # past the base check (stops later, on the empty rows)


# ---- exclusion by rule ---------------------------------------------------------------------------------------------------

def old_exclusion(r, alias):
    """`exclusion` as it was before the catalog: keyed by `corpus == 'mendeley'`."""
    if r['corpus'] == 'mendeley':
        if not r['recommended_split']:
            return 'no recommended split'
    elif r['eligible_for_gnn'] != '1' or r['architecture'] not in ('x86', 'x64'):
        return 'ineligible'
    if r['corpus'] == 'mendeley' and r['representation'] == 'upx_packed':
        return 'upx stub'
    if r['corpus'] != 'mendeley' and (r['family'] == '_upx_packed' or alias.get('current_representation') == 'upx_packed'):
        return 'upx stub'
    return ''


def test_the_rule_of_the_corpus_selects_the_branch_not_its_name():
    assert {c: s.rule for c, s in catalog.CORPORA.items()} == {'mendeley': 'split', 'ransomware_balanced': 'eligible',
                                                               'balanced_goodware': 'eligible'}
    row = dict(sample_id='x', label='1', family='conti', representation='normal', recommended_split='', architecture='x86',
               eligible_for_gnn='1')
    assert bc.exclusion({**row, 'corpus': 'mendeley'}, {}) == 'no recommended split'       # `split` ignores eligible_for_gnn
    assert bc.exclusion({**row, 'corpus': 'ransomware_balanced'}, {}) == ''                # `eligible` ignores recommended_split
    assert bc.exclusion({**row, 'corpus': 'mendeley', 'recommended_split': 'train', 'eligible_for_gnn': '0', 'architecture': 'arm'}, {}) == ''
    assert bc.exclusion({**row, 'corpus': 'balanced_goodware', 'eligible_for_gnn': '0'}, {}) == 'ineligible'
    assert bc.exclusion({**row, 'corpus': 'balanced_goodware', 'recommended_split': 'x'}, {'current_representation': 'upx_packed'}) == 'upx stub'
    assert bc.exclusion({**row, 'corpus': 'mendeley', 'recommended_split': 'x'}, {'current_representation': 'upx_packed'}) == ''


def test_exclusion_is_unchanged_for_every_combination_of_the_fields_it_reads():
    grid = itertools.product(catalog.CORPORA, ('', 'train', 'test'), ('normal', 'upx_packed'), ('x86', 'x64', 'arm', ''),
                             ('1', '0', ''), ('conti', '_upx_packed'), ({}, {'current_representation': 'normal'},
                                                                         {'current_representation': 'upx_packed'}))
    n = 0
    for corpus, split, rep, arch, eligible, family, alias in grid:
        row = dict(sample_id='x', corpus=corpus, recommended_split=split, representation=rep, architecture=arch,
                   eligible_for_gnn=eligible, family=family)
        assert bc.exclusion(row, alias) == old_exclusion(row, alias), row
        n += 1
    assert n == 3 * 3 * 2 * 4 * 3 * 2 * 3


# ---- fold files by dataset ----------------------------------------------------------------------------------------------

def write_csv(path, header, *rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join([header, *rows]) + '\n', encoding='utf-8')


def fold_files(csv_path, v3_dir, fold, fold_v3, sample='aa'):
    write_csv(csv_path, 'sha256,fold', f'{sample},{fold}')
    write_csv(v3_dir / 'folds_all.csv', 'sha256,label,arch,fold,family,group', f'{sample},1,x86,{fold_v3},f,g')
    write_csv(v3_dir / 'arch_matched.csv', 'sha256,keep', f'{sample},1')


def folded_world(env, monkeypatch):
    """One Mendeley sample `aa` (ransomware, x86) and a catalog dataset `synth` whose own fold files give it fold 3 and fold_v3 4;
    the default fold files of the catalog do not know the sample."""
    data = env / 'data'
    row = dict(sample_id='aa', label='1', family='f', representation='normal', recommended_split='train', architecture='x86',
               eligible_for_gnn='1', cfg_path='aa.json')
    for name in ('Mendeley_Ransomware', 'Mendeley_Goodware'):
        collection(data, name, [row] if name == 'Mendeley_Ransomware' else [])
        (data / name / 'provenance').mkdir()
        (data / name / 'provenance' / 'identity_aliases.json').write_text('[]', encoding='utf-8')
    monkeypatch.setattr(paths, 'FOLDS_ROOT', env / 'folds')
    monkeypatch.setitem(catalog.DATASETS, 'synth',
                        catalog.Dataset(('kfold',), ('mendeley',), {'folds': 'synth.csv', 'folds_v3': 'synth_v3'}))
    fold_files(env / 'folds' / 'synth.csv', env / 'folds' / 'synth_v3', 3, 4)
    fold_files(env / 'folds' / 'folds_mendeley.csv', env / 'folds' / 'v3', 1, 1, sample='zz')       # the catalog's default files


def folds_of(dataset):
    (row,), _ = bc.eligible_rows(['mendeley'], dataset)
    return row['fold'], row['fold_v3']


def test_a_catalog_dataset_is_built_with_its_own_fold_files(env, monkeypatch):
    folded_world(env, monkeypatch)
    assert folds_of('synth') == ('3', '4')
    with pytest.raises(SystemExit, match='aa joins neither folds_mendeley.csv'):
        folds_of('mendeley_x86')                                                  # its entry is the default file, which lacks `aa`
    with pytest.raises(SystemExit, match='aa joins neither folds_mendeley.csv'):
        folds_of(None)                                                            # no dataset: the first catalog default
    plan = {'dataset': 'synth', 'cache': env / 'c', 'corpora': ['mendeley'], 'base_cache': None}
    lines = dict(line.split(': ', 1) for line in bc.resolution_lines(plan))
    assert lines['folds'] == (env / 'folds' / 'synth.csv').as_posix() and lines['folds_v3'] == (env / 'folds' / 'synth_v3').as_posix()


def test_the_environment_beats_the_own_folds_of_a_dataset(env, monkeypatch):
    folded_world(env, monkeypatch)
    fold_files(env / 'envfolds.csv', env / 'envv3', 5, 6)
    monkeypatch.setenv('GT_FOLDS', str(env / 'envfolds.csv'))
    assert folds_of('synth') == ('5', '4')                                        # GT_FOLDS replaces the CSV only
    monkeypatch.setenv('GT_FOLDS_V3', str(env / 'envv3'))
    assert folds_of('synth') == ('5', '6')


def test_datasets_json_beats_the_own_folds_of_a_dataset_and_the_environment_beats_the_file(env, monkeypatch):
    folded_world(env, monkeypatch)
    fold_files(env / 'filefolds.csv', env / 'filev3', 7, 8)
    file = env / 'datasets.json'
    file.write_text(json.dumps({'v2': {'cache': 'c', 'sources': {'folds': 'filefolds.csv', 'folds_v3': 'filev3'}}}), encoding='utf-8')
    monkeypatch.setenv('GT_DATASETS', str(file))
    assert folds_of('synth') == ('7', '8')
    fold_files(env / 'envfolds.csv', env / 'envv3', 5, 6)
    monkeypatch.setenv('GT_FOLDS', str(env / 'envfolds.csv'))
    assert folds_of('synth') == ('5', '8')


def test_a_file_entry_of_the_dataset_itself_comes_before_the_other_entries_of_the_file(env, monkeypatch):
    doc = {'first': {'cache': 'a', 'sources': {'folds': 'first.csv'}}, 'synth': {'cache': 'b', 'sources': {'folds': 'own.csv'}}}
    (env / 'datasets.json').write_text(json.dumps(doc), encoding='utf-8')
    monkeypatch.setenv('GT_DATASETS', str(env / 'datasets.json'))
    monkeypatch.setitem(catalog.DATASETS, 'synth', catalog.Dataset(('kfold',), ('mendeley',), {'folds': 'synth.csv'}))
    assert paths.source_for('folds', dataset='synth') == env / 'own.csv'
    assert paths.source_for('folds', dataset='mendeley_x86') == env / 'first.csv'      # the catalog row comes after the file's entries
    assert paths.source_for('folds') == env / 'first.csv'


# ---- the corpus functions: bundle discovery order ------------------------------------------------------------------------

def test_the_mendeley_roots_are_the_configured_bundle_then_the_bundle_folder_then_the_collections(env, monkeypatch):
    data = env / 'data'
    mr, mg = collection(data, 'Mendeley_Ransomware'), collection(data, 'Mendeley_Goodware')
    assert bc.corpus_roots('mendeley') == [mr, mg]
    found = data / 'Mendeley_GNN_v1'
    found.mkdir()
    (found / 'summary.json').write_text(json.dumps({'schema': catalog.BUNDLE_SCHEMA}), encoding='utf-8')
    assert bc.corpus_roots('mendeley') == [found]
    configured = env / 'configured'
    monkeypatch.setenv('GT_BUNDLE', str(configured))
    assert bc.corpus_roots('mendeley') == [configured]
    assert dry(env, GT_BUNDLE=str(configured), GT_COLLECTIONS=str(data))['bundle'] == configured.as_posix()
    assert dry(env, GT_COLLECTIONS=str(data))['bundle'] == found.as_posix()
    other = collection(data, 'Ransomware_Balanced')
    assert bc.corpus_roots('ransomware_balanced') == [other]                         # no bundle for the other corpora


# ---- folds.py ------------------------------------------------------------------------------------------------------------

def test_folds_out_defaults_to_the_cache_folder_of_the_dataset_never_the_committed_folds(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(catalog, 'CACHE_ROOT', tmp_path / 'cache')
    rows = [{'sha256': f'r{i}', 'label': 1, 'family': f'f{i % 3}', 'group': f'f{i % 3}', 'arch': 'x86', 'orig_set': '', 'corpus': 'm'}
            for i in range(6)] + [{'sha256': f'g{i}', 'label': 0, 'family': 'goodware', 'group': f'g{i}', 'arch': 'x86',
                                   'orig_set': '', 'corpus': 'm'} for i in range(4)]
    monkeypatch.setattr(folds, 'pool_rows', lambda corpus, groups=None: rows)
    before = sorted((p.name, p.stat().st_mtime_ns) for p in catalog.FOLDS_ROOT.rglob('*') if p.is_file())
    folds.main([])
    assert (tmp_path / 'cache' / 'mendeley_x86' / 'folds_mendeley.csv').is_file()
    folds.main(['--corpus', 'ransomware_balanced'])
    assert (tmp_path / 'cache' / 'combined_v3' / 'folds_ransomware_balanced.csv').is_file()    # the first dataset that holds it
    folds.main(['--corpus', 'mendeley', '--dataset', 'combined_v3'])
    assert (tmp_path / 'cache' / 'combined_v3' / 'folds_mendeley.csv').is_file()
    folds.main(['--out', str(tmp_path / 'here')])
    assert (tmp_path / 'here' / 'folds_mendeley.csv').is_file()
    assert before == sorted((p.name, p.stat().st_mtime_ns) for p in catalog.FOLDS_ROOT.rglob('*') if p.is_file())
    with pytest.raises(SystemExit, match='does not hold ransomware_balanced'):
        folds.main(['--corpus', 'ransomware_balanced', '--dataset', 'mendeley_x86'])


def test_folds_help_says_the_generated_folds_are_not_the_published_ones():
    r = subprocess.run([sys.executable, '-m', 's02_preprocess.cache.folds', '--help'], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    text = ' '.join(r.stdout.split())
    assert r.returncode == 0 and 'CT_GAT/cache/<dataset>/' in text and 'not the published' in text
    assert [c for c in catalog.CORPORA if c not in text] == []                       # the corpus choices come from the catalog


# ---- main(): the resolved corpora and base reach the build --------------------------------------------------------------

class InlinePool:
    """multiprocessing.Pool without processes, so that main() can be run with its stages replaced."""
    def __init__(self, workers):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap_unordered(self, fn, items, chunksize=1):
        return map(fn, items)


@pytest.mark.parametrize('flags, corpora, base', [
    ([], ['mendeley'], None),
    (['--cache', 'combined_v3'], ['mendeley', 'ransomware_balanced', 'balanced_goodware'], 'base'),
    (['--cache', 'combined_v3', '--corpora', 'balanced_goodware'], ['balanced_goodware'], 'base'),
    (['--cache', 'combined_v3', '--base-cache', 'other'], ['mendeley', 'ransomware_balanced', 'balanced_goodware'], 'other')])
def test_main_builds_the_corpora_of_the_dataset_on_top_of_its_base_cache(env, monkeypatch, flags, corpora, base):
    for name in catalog.COLLECTIONS:
        collection(env / 'data', name)
    for folder in ('base', 'other'):
        (env / folder).mkdir()
        (env / folder / 'index.csv').write_text('sample_id,status\n', encoding='utf-8')
    file = env / 'datasets.json'
    file.write_text(json.dumps({'mendeley_x86': {'cache': 'base'}}), encoding='utf-8')     # the base of combined_v3, built
    monkeypatch.setenv('GT_DATASETS', str(file))
    out = env / 'out'
    seen = {}
    rows = [{'sample_id': 'a', 'label': '1', 'num_instructions': '5', 'corpus': 'mendeley', 'fold_v3': '1'}]
    monkeypatch.setattr(bc, 'CACHE', out)
    monkeypatch.setattr(bc, 'Pool', InlinePool)
    monkeypatch.setattr(bc, 'eligible_rows', lambda c, d: seen.update(corpora=list(c)) or (rows, {}))
    monkeypatch.setattr(bc, 'seed_from_base', lambda b, r: seen.update(base=Path(b)) or 0)
    monkeypatch.setattr(bc, 'build_one', lambda r: (r['sample_id'], 'ok', '', 0.0))
    monkeypatch.setattr(bc, 'finalize', lambda r: ({'mn': 0}, 'sha', 0))
    monkeypatch.setattr(bc, 'write_index', lambda r, results: [])
    monkeypatch.setattr(bc, 'fold_table', lambda index: '')
    monkeypatch.setattr(bc, 'edge_stats', lambda r: {})
    monkeypatch.setattr(sys, 'argv', ['build_cache', *[str(env / f) if f in ('other',) else f for f in flags]])
    bc.main()
    summary = json.loads((out / 'build_summary.json').read_text(encoding='utf-8'))
    assert seen['corpora'] == summary['corpora'] == corpora
    assert seen.get('base') == (env / base if base else None) and summary['base_cache'] == (str(env / base) if base else '')
    assert set(summary['eligible']) == set(corpora)
