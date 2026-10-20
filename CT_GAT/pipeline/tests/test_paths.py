"""common/paths.py: the dataset registry, `--cache <name>`, the environment overrides, and an unconfigured machine."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from common import paths
from common.paths import UNSET, MissingSource, cache_for, cache_in, export_for, load_registry, protocol_in, runs_for, source_for

ROOT = Path(__file__).resolve().parents[1]
V2, COMBINED = Path('data/v2'), Path('data/combined')


def entry(cache, protocols, **sources):
    return {'cache': cache, 'protocols': tuple(protocols), 'sources': {k: Path(v) for k, v in sources.items()}}


REGISTRY = {'v2': entry(V2, ['bundle', 'kfold'], bundle='raw/bundle', folds='raw/folds.csv'),
            'combined': entry(COMBINED, ['kfold_v3'], collections='raw/coll')}


def test_cache_for_resolution():
    assert cache_for('kfold_v3', {}, REGISTRY) == COMBINED
    for protocol in ('bundle', 'kfold', None):
        assert cache_for(protocol, {}, REGISTRY) == V2
    for protocol in ('bundle', 'kfold', 'kfold_v3'):
        assert cache_for(protocol, {'GT_CACHE': 'X:/mine'}, REGISTRY) == Path('X:/mine')
    assert cache_for('kfold_v3', {'GT_CACHE': ''}, REGISTRY) == COMBINED
    assert cache_for('kfold_v3', {}, REGISTRY, name='v2') == V2                 # a named dataset beats the protocol
    assert cache_for('bundle', {'GT_CACHE': 'X:/mine'}, REGISTRY, name='v2') == Path('X:/mine')    # GT_CACHE beats the name
    with pytest.raises(ValueError, match='unknown dataset'):
        cache_for('bundle', {}, REGISTRY, name='v3')


def test_an_unconfigured_machine_resolves_to_a_path_that_does_not_exist():
    assert cache_for('bundle', {}, {}) == UNSET == cache_for(None, {}, {})
    assert not UNSET.exists() and not UNSET.is_absolute()
    with pytest.raises(ValueError, match='datasets.json.example'):
        cache_for('bundle', {}, {}, name='v2')
    assert cache_for('kfold_v3', {}, {'only': entry(V2, ['bundle'])}) == UNSET     # no dataset reads this protocol


def test_protocol_and_cache_in_a_command_line():
    assert protocol_in(['--fold', '1', '--protocol', 'kfold_v3']) == 'kfold_v3'
    assert protocol_in(['--protocol=kfold']) == 'kfold' and protocol_in(['--protocol']) is None and protocol_in([]) is None
    assert cache_in(['--cache', 'combined', '--fold', '1'], REGISTRY) == 'combined'
    assert cache_in(['--cache=v2'], REGISTRY) == 'v2'
    assert cache_in(['--cache', 'some/dir'], REGISTRY) is None and cache_in(['--cache'], REGISTRY) is None
    assert cache_in([], REGISTRY) is None


def write_registry(path, doc):
    path.write_text(json.dumps(doc), encoding='utf-8')
    return {'GT_DATASETS': str(path)}


def test_registry_file_relative_paths_comments_and_validation(tmp_path):
    (tmp_path / 'sub').mkdir()
    env = write_registry(tmp_path / 'sub' / 'datasets.json', {
        '_comment': 'ignored',
        'v2': {'cache': 'caches/v2', 'protocols': ['bundle', 'kfold'], 'sources': {'bundle': 'raw/b', 'runs': str(tmp_path / 'r')}},
        'combined': {'cache': str(tmp_path / 'abs'), 'protocols': ['kfold_v3']}})
    registry = load_registry(env)
    assert list(registry) == ['v2', 'combined']
    assert registry['v2']['cache'] == tmp_path / 'sub' / 'caches' / 'v2'               # relative to the file
    assert registry['v2']['sources']['bundle'] == tmp_path / 'sub' / 'raw' / 'b'
    assert registry['combined']['cache'] == tmp_path / 'abs' and registry['combined']['protocols'] == ('kfold_v3',)
    assert cache_for('kfold_v3', env) == tmp_path / 'abs'
    for bad, match in (({'v2': {'protocols': []}}, 'needs "cache"'),
                       ({'v2': {'cache': 'c', 'surprise': 1}}, 'may only have'),
                       ({'v2': {'cache': 'c', 'sources': {'nowhere': 'x'}}}, 'unknown sources')):
        with pytest.raises(ValueError, match=match):
            load_registry(write_registry(tmp_path / 'bad.json', bad))
    assert load_registry({'GT_DATASETS': str(tmp_path / 'absent.json')}) == {}


def test_sources_come_from_the_environment_then_the_registry():
    env = {'GT_BUNDLE': 'env/bundle'}
    assert source_for('bundle', env, REGISTRY) == Path('env/bundle')
    assert source_for('bundle', {}, REGISTRY) == Path('raw/bundle')
    assert source_for('collections', {}, REGISTRY) == Path('raw/coll')                 # defined by a later dataset
    with pytest.raises(MissingSource, match='GT_FOLDS_V3.*datasets.json.example'):
        source_for('folds_v3', {}, REGISTRY)
    with pytest.raises(MissingSource, match='GT_TOKEN_MAPPING_DIR'):
        source_for('token_mapping', {}, {})


def test_runs_and_exports():
    assert runs_for({}, REGISTRY) == paths.PIPELINE_ROOT / 'runs'
    assert runs_for({'GT_RUNS': 'R'}, REGISTRY) == Path('R')
    assert runs_for({}, {'a': entry(V2, [], runs='my/runs')}) == Path('my/runs')
    assert export_for({'GT_RUNS': 'R'}, REGISTRY) == Path('R') / 'export'              # GT_RUNS alone moves both
    assert export_for({'GT_RUNS': 'R', 'GT_EXPORT': 'E'}, REGISTRY) == Path('E')


def test_the_example_file_is_a_valid_registry():
    example = ROOT / 'datasets.json.example'
    registry = load_registry({'GT_DATASETS': str(example)})
    assert {'v2', 'combined'} <= set(registry)
    assert cache_for('kfold_v3', {'GT_DATASETS': str(example)}) == registry['combined']['cache']
    assert set(paths.SOURCES) >= {k for e in registry.values() for k in e['sources']}


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GT_')}
    env.update(extra, PYTHONDONTWRITEBYTECODE='1')
    return env


def probe(code, **env):
    return subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, env=clean_env(**env))


def test_importing_config_needs_no_datasets_file(tmp_path):
    r = probe('from common.config import CACHE, RUNS, EXPORT, DEFAULT_CACHE, COMBINED_CACHE, RunConfig\n'
              'print(CACHE, DEFAULT_CACHE, COMBINED_CACHE, RUNS.name, EXPORT.name)', GT_DATASETS=str(tmp_path / 'none.json'))
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == [str(UNSET)] * 3 + ['runs', 'export']


def test_the_module_level_cache_follows_the_environment_the_name_and_the_protocol(tmp_path):
    reg = tmp_path / 'datasets.json'
    reg.write_text(json.dumps({'v2': {'cache': 'v2', 'protocols': ['bundle', 'kfold']},
                               'combined': {'cache': 'combined', 'protocols': ['kfold_v3']}}), encoding='utf-8')
    code = 'import sys\nfrom common.config import CACHE, DEFAULT_CACHE, COMBINED_CACHE\nprint(CACHE.name, DEFAULT_CACHE.name, COMBINED_CACHE.name)\n'
    def run(*argv, **env):
        r = subprocess.run([sys.executable, '-c', 'import sys; sys.argv = ["x"] + sys.argv[1:]\n' + code, *argv], cwd=ROOT,
                           capture_output=True, text=True, env=clean_env(GT_DATASETS=str(reg), **env))
        assert r.returncode == 0, r.stderr
        return r.stdout.split()
    assert run() == ['v2', 'v2', 'combined']
    assert run('--protocol', 'kfold_v3', '--fold', '0') == ['combined', 'v2', 'combined']
    assert run('--cache', 'combined') == ['combined', 'v2', 'combined']
    assert run('--cache', 'somewhere/else') == ['v2', 'v2', 'combined']              # a directory is not a dataset name
    assert run('--protocol', 'kfold_v3', GT_CACHE=str(tmp_path / 'mine')) == ['mine', 'v2', 'combined']
