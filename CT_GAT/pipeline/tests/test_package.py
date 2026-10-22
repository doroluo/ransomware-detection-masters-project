"""The package's boundaries: no machine path in a file, the teammates' scripts still import, the modules import clean."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SKIP_DIRS = {'__pycache__', '.pytest_cache', '.ruff_cache'}
MACHINE_PATH = ('C:' + '/Users', 'C:' + '\\Users')                      # written so that this file does not match itself


def files_under(root):
    for path in sorted(root.rglob('*')):
        if path.is_file() and not SKIP_DIRS & set(path.parts) and path.suffix not in ('.pyc', '.npz', '.npy'):
            yield path


def test_no_file_names_a_machine_path_except_the_example_registry():
    """datasets.json.example documents the format with placeholder paths; datasets.json (gitignored) is per machine."""
    offenders = []
    for path in files_under(ROOT):
        if path.name in ('datasets.json.example', 'datasets.json'):
            continue
        text = path.read_text(encoding='utf-8', errors='replace')
        if any(p in text for p in MACHINE_PATH):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, offenders


def run(code, cwd):
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
    return subprocess.run([sys.executable, '-c', code], cwd=cwd, capture_output=True, text=True, env=env)


@pytest.mark.parametrize('cwd,folder', [(ROOT, 's02_preprocess'), (REPO, 'CT_GAT/pipeline/s02_preprocess')])
def test_the_teammates_preprocessing_scripts_still_import(cwd, folder):
    """Yanping's flat scripts put their own folder on sys.path; the package files beside them must not shadow anything."""
    r = run(f"import sys; sys.path.insert(0, {folder!r}); import tokenize_asm_gz, preprocess_full; print('ok')", cwd)
    assert r.returncode == 0 and r.stdout.strip() == 'ok', r.stderr


def test_the_transformer_model_placeholder_is_gone():
    """PR #8 removed Yanping's placeholder `s03a_transformer/transformer_model.py`; nothing may import it any more."""
    assert not (ROOT / 's03a_transformer' / 'transformer_model.py').exists()
    r = run("import sys; sys.path.insert(0, 's03a_transformer'); import transformer_model", ROOT)
    assert r.returncode != 0 and 'ModuleNotFoundError' in r.stderr, r.stderr


def test_every_module_of_the_package_imports_without_a_datasets_file(tmp_path):
    code = ('import importlib\n'
            "for m in ('common.config', 'common.manifests', 'common.paths', 's02_preprocess',\n"
            "          's02_preprocess.cache.build_cache', 's02_preprocess.cache.check_cache', 's02_preprocess.cache.folds',\n"
            "          's03a_transformer', 's03b_cfg_feature', 's03c_cfg_transformer'):\n"
            '    importlib.import_module(m)\n'
            "print('ok')")
    env = {k: v for k, v in os.environ.items() if not k.startswith('GT_')}
    env.update(GT_DATASETS=str(tmp_path / 'none.json'), PYTHONDONTWRITEBYTECODE='1')
    r = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 0 and r.stdout.strip() == 'ok', r.stderr


def test_build_cache_without_data_stops_before_touching_the_disk(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GT_')}
    env.update(GT_DATASETS=str(tmp_path / 'none.json'), GT_COLLECTIONS=str(tmp_path / 'empty'), GT_CACHE=str(tmp_path / 'cache'),
               PYTHONDONTWRITEBYTECODE='1')
    r = subprocess.run([sys.executable, '-m', 's02_preprocess.cache.build_cache', '--limit', '2'], cwd=tmp_path,
                       capture_output=True, text=True, env={**env, 'PYTHONPATH': str(ROOT)})
    assert r.returncode != 0 and 'no mendeley data' in r.stderr and 'Mendeley_Goodware' in r.stderr
    assert not (tmp_path / 'cache').exists()                               # the cache directory was not created


def test_build_cache_stops_on_an_unset_cache(monkeypatch):
    from common.paths import UNSET
    from s02_preprocess.cache import build_cache
    monkeypatch.setattr(build_cache, 'CACHE', UNSET)
    monkeypatch.setattr(sys, 'argv', ['build_cache', '--limit', '2'])
    with pytest.raises(SystemExit, match='no cache configured'):
        build_cache.main()


def test_check_cache_and_build_cache_are_runnable_modules():
    for module in ('build_cache', 'check_cache', 'folds'):
        r = subprocess.run([sys.executable, '-m', f's02_preprocess.cache.{module}', '--help'], cwd=ROOT,
                           capture_output=True, text=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        assert r.returncode == 0 and 'usage' in r.stdout.lower(), (module, r.stderr)
