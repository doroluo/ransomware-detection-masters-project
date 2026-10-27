"""python -m pytest tests/test_ci.py: the GitHub Actions workflow and the requirements it installs, and the rule that no file of
the pipeline names a machine (the real paths live in the gitignored datasets.json; datasets.json.example documents the format)."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT.parents[1] / '.github' / 'workflows' / 'pipeline-tests.yml'
SKIP_DIRS = {'.pytest_cache', '.ruff_cache', '__pycache__', 'runs', '.git'}
TEXT = {'.py', '.md', '.txt', '.json', '.yml', '.yaml', '.ini', '.csv', '.sh'}
PRIVATE = re.compile(r'[A-Za-z]:[/\\]+Users[/\\]|/c/Users/|/home/[a-z]|/Users/[a-z]', re.I)


def test_the_workflow_runs_the_cpu_tests_on_ubuntu_with_python_312():
    yaml = pytest.importorskip('yaml')
    doc = yaml.safe_load(WORKFLOW.read_text(encoding='utf-8'))
    job = doc['jobs']['pytest']
    assert job['runs-on'] == 'ubuntu-latest' and job['defaults']['run']['working-directory'] == 'CT_GAT/pipeline'
    assert job['env']['CUDA_VISIBLE_DEVICES'] == '-1' and job['env']['GT_DATASETS'].startswith('/nonexistent')
    steps = job['steps']
    setup = next(s for s in steps if str(s.get('uses', '')).startswith('actions/setup-python'))
    assert setup['with']['python-version'] == '3.12'
    commands = [s['run'] for s in steps if 'run' in s]
    assert any('pip install -r requirements.txt' in c for c in commands)
    assert commands[-1] == 'python -m pytest -m "not real_data and not gpu" -q -p no:cacheprovider'
    assert 'not installed on' not in WORKFLOW.read_text(encoding='utf-8')                  # torch_geometric comes from requirements.txt
    triggers = doc.get(True) or doc.get('on')
    assert 'CT_GAT/pipeline/**' in triggers['pull_request']['paths']


def test_the_requirements_are_pinned_and_use_the_cpu_torch_index():
    lines = [l.strip() for l in (ROOT / 'requirements.txt').read_text(encoding='utf-8').splitlines()]
    active = [l for l in lines if l and not l.startswith('#')]
    assert active[0] == '--extra-index-url https://download.pytorch.org/whl/cpu'
    pins = dict(l.split('==') for l in active[1:])
    assert set(pins) == {'torch', 'numpy', 'scipy', 'scikit-learn', 'pytest', 'PyYAML', 'torch_geometric'}
    assert all(re.fullmatch(r'\d+(\.\d+)+', v) for v in pins.values())
    assert pins['torch_geometric'] == '2.8.1'                                # what the published runs used; CI installs it


def test_no_file_of_the_pipeline_names_a_machine():
    offenders = []
    for path in ROOT.rglob('*'):
        parts = set(path.relative_to(ROOT).parts)
        if not path.is_file() or parts & SKIP_DIRS or path.suffix not in TEXT:
            continue
        if path.name == 'datasets.json':                                      # the machine's own, gitignored
            continue
        if path.name == 'datasets.json.example' or path == Path(__file__).resolve():     # the example documents the format; this file holds the pattern
            continue
        if PRIVATE.search(path.read_text(encoding='utf-8', errors='replace')):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, offenders
    workflow = WORKFLOW.read_text(encoding='utf-8')
    assert not PRIVATE.search(workflow)
