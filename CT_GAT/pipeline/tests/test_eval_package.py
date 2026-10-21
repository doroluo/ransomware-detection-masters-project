"""python -m pytest tests/test_eval_package.py: the modules of stage 4 import clean on a machine with no datasets.json, the
commands run as modules from the package root, and nothing in them reads a module of a later step at import time."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

import experiments.stages as stages_package

ROOT = Path(__file__).resolve().parents[1]
MODULES = ('common.run_io', 's04_evaluation', 's04_evaluation.metrics', 's04_evaluation.val_threshold', 's04_evaluation.stats',
           's04_evaluation.report', 's04_evaluation.baselines', 's04_evaluation.stage', 's04_evaluation.gate_tables',
           's04_evaluation.gate_layouts', 's04_evaluation.gates', 'experiments', 'experiments.stages',
           'experiments.stages.common', *(f'experiments.stages.{name}' for name in stages_package.STAGES))


def clean_env(tmp_path, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GT_')}
    env.update(GT_DATASETS=str(tmp_path / 'none.json'), PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8', **extra)
    return env


def run(args, tmp_path, **kw):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, env=clean_env(tmp_path), **kw)


def test_every_module_of_stage_4_imports_without_a_datasets_file(tmp_path):
    code = 'import importlib\nfor m in %r:\n    importlib.import_module(m)\nprint("ok")' % (MODULES,)
    r = run(['-c', code], tmp_path)
    assert r.returncode == 0 and r.stdout.strip() == 'ok', r.stderr


def test_importing_the_baselines_reads_no_module_of_a_later_step(tmp_path):
    code = ('import sys\nimport s04_evaluation.baselines, s04_evaluation.gates\n'
            'late = sorted(m for m in sys.modules if m.startswith(("s03b_cfg_feature", "s03c_cfg_transformer", "experiments.side_arrays",'
            ' "experiments.train", "torch")))\nprint(late)')
    r = run(['-c', code], tmp_path)
    assert r.returncode == 0 and r.stdout.strip() == '[]', (r.stdout, r.stderr)


@pytest.mark.parametrize('module', ['baselines', 'stats', 'report', 'gates'])
def test_the_commands_run_as_modules_from_the_package_root(module, tmp_path):
    r = run(['-m', f's04_evaluation.{module}', '--help'], tmp_path)
    assert r.returncode == 0 and 'usage' in r.stdout.lower(), r.stderr


def test_val_threshold_needs_the_run_folder_only_to_run(tmp_path):
    missing = tmp_path / 'no_runs'
    env = clean_env(tmp_path, GT_RUNS=str(missing))
    r = subprocess.run([sys.executable, '-m', 's04_evaluation.val_threshold', '--help'], cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 0 and 'usage' in r.stdout.lower(), r.stderr
    r = subprocess.run([sys.executable, '-m', 's04_evaluation.val_threshold'], cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode != 0 and 'Traceback' not in r.stderr, r.stderr
    assert str(missing) in r.stderr and 'training or baseline command' in r.stderr
    missing.mkdir()
    r = subprocess.run([sys.executable, '-m', 's04_evaluation.val_threshold'], cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 0 and r.stdout.startswith('| config |'), r.stderr


def test_the_stage_modules_do_not_import_the_engine():
    """Stage files are data: they may import stage.py and common.py only, so that the engine can run as __main__."""
    for name in stages_package.STAGES:
        text = (Path(stages_package.__file__).parent / f'{name}.py').read_text(encoding='utf-8')
        assert 's04_evaluation.gates' not in text and 'gate_layouts' not in text and 'gate_tables' not in text, name


def test_report_has_no_path_of_its_own(tmp_path):
    text = (ROOT / 's04_evaluation' / 'report.py').read_text(encoding='utf-8')
    assert "parents[1] / 'docs'" not in text and 'RUNS / \'RESULTS.md\'' in text
    r = run(['-c', 'import s04_evaluation.report as r; print(r.RUNS)'], tmp_path)
    assert r.returncode == 0 and r.stdout.strip().endswith('runs'), r.stdout        # RUNS falls back to runs/ in the package root
