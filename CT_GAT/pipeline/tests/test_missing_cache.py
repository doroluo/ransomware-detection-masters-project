"""A cache that is not built: `load_index` raises FileNotFoundError with the build command, and the commands that read the cache
print that message and exit 1 instead of ending in a traceback."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from common import manifests
from common.manifests import MissingCache, load_index, missing_cache_message
from s04_evaluation import report

ROOT = Path(__file__).resolve().parents[1]
BUILD = 'python -m s02_preprocess.cache.build_cache --cache '
COMMANDS = {
    'gates': (['s04_evaluation.gates', '--stage', 'v3'], 'combined_v3'),
    'check_cache': (['s02_preprocess.cache.check_cache', '--cache', 'combined_v3'], 'combined_v3'),
    'train': (['experiments.train', '--protocol', 'kfold', '--fold', '0'], 'mendeley_x86'),
    'baselines': (['s04_evaluation.baselines', '--protocol', 'kfold_v3', '--fold', '0', '--arch', 'all'], 'combined_v3'),
    'subgraph_chain': (['experiments.subgraph_chain'], 'mendeley_x86'),
    'block_export': (['s03c_cfg_transformer.block_export', '--protocol', 'kfold_v3', '--arch', 'all', '--out', '{runs}/export/x'], 'combined_v3'),
}


def run(args, tmp_path, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GT_')}
    env.update({'GT_RUNS': str(tmp_path / 'runs'), 'PYTHONIOENCODING': 'utf-8', **extra})
    return subprocess.run([sys.executable, '-m', *args], cwd=ROOT, capture_output=True, text=True, env=env)


def assert_message(r, cache, dataset):
    assert r.returncode == 1, r.stderr
    assert 'Traceback' not in r.stderr and not r.stdout.strip()
    text = r.stderr.strip()
    assert '\n' not in text and f'{cache.as_posix()}/index.csv' in text and BUILD + dataset in text


def test_load_index_names_the_cache_and_the_build_command(tmp_path):
    with pytest.raises(FileNotFoundError, match='index.csv: the cache is not built') as e:
        load_index(tmp_path, 'kfold_v3')
    assert isinstance(e.value, MissingCache) and BUILD + 'combined_v3' in str(e.value)
    assert 'Ransomware_Balanced and Balanced_Goodware collections under CT_GAT/data' in str(e.value)
    assert 'mendeley_x86 cache built first' in str(e.value)
    assert BUILD + 'mendeley_x86' in missing_cache_message(tmp_path, 'kfold')
    assert '--cache <name>' in missing_cache_message(tmp_path)               # no protocol, no dataset: the general form


def test_the_report_still_skips_a_cache_that_is_not_built(tmp_path, monkeypatch):
    monkeypatch.setattr(manifests, 'CACHE', tmp_path)
    monkeypatch.setattr(report, 'cache_for', lambda protocol: tmp_path)
    assert report.all_labels() == {}


@pytest.mark.parametrize('name', COMMANDS)
def test_a_command_without_the_cache_prints_the_message_and_exits_1(name, tmp_path):
    args, dataset = COMMANDS[name]
    cache = tmp_path / 'cache'
    cache.mkdir()
    args = [a.format(runs=tmp_path / 'runs') for a in args]
    assert_message(run(args, tmp_path, GT_CACHE=str(cache)), cache, dataset)


@pytest.mark.parametrize('name', ['gates', 'check_cache'])
def test_a_registry_entry_pointing_at_an_empty_folder_gets_the_same_message(name, tmp_path):
    args, dataset = COMMANDS[name]
    cache = tmp_path / 'cache'
    cache.mkdir()
    (tmp_path / 'datasets.json').write_text(json.dumps({'combined_v3': {'cache': 'cache', 'protocols': ['kfold_v3']}}),
                                            encoding='utf-8')
    r = run(args, tmp_path, GT_DATASETS=str(tmp_path / 'datasets.json'))
    assert_message(r, cache, dataset)
    assert 'GT_CACHE is set' not in r.stderr
