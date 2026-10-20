"""The boundaries of stage 3b and the training command: the modules import clean on a machine with no datasets.json, the commands
run as modules from the package root, the stage's imports stay inside the package, and the placeholder is gone."""
import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 's03b_cfg_feature'
MODULES = ('common.registry', 'experiments.train', 's03b_cfg_feature', 's03b_cfg_feature.gnn', 's03b_cfg_feature.gnn.spec',
           's03b_cfg_feature.gnn.model', 's03b_cfg_feature.gnn.batching',
           's03b_cfg_feature.gnn.s05_block_counting.block_vecs', 's03b_cfg_feature.gnn.s05_block_counting.encoder',
           's03b_cfg_feature.gnn.s05_block_counting.weights', 's03b_cfg_feature.gnn.s06_gnn_block_cfg.edges',
           's03b_cfg_feature.gnn.s06_gnn_block_cfg.layers', 's03b_cfg_feature.gnn.s07_pool_blocks_to_functions.readout',
           's03b_cfg_feature.gnn.s08_gnn_call_graph.calls', 's03b_cfg_feature.gnn.s09_pool_functions_to_binary',
           's03b_cfg_feature.gnn.s10_unused_import_features.import_side',
           's03b_cfg_feature.gnn.s10_unused_import_features.imports_feat', 's03b_cfg_feature.gnn.s11_final_classifier.head')


def clean_env(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GT_')}
    env.update(GT_DATASETS=str(tmp_path / 'none.json'), PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1')
    return env


def run(args, tmp_path):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, env=clean_env(tmp_path))


def test_every_module_imports_without_a_datasets_file(tmp_path):
    code = 'import importlib\nfor m in %r:\n    importlib.import_module(m)\nprint("ok")' % (MODULES,)
    r = run(['-c', code], tmp_path)
    assert r.returncode == 0 and r.stdout.strip() == 'ok', r.stderr


def test_the_placeholder_is_gone_and_the_stage_is_a_package():
    assert not (STAGE / 'placeholder.py').exists()
    assert (STAGE / '__init__.py').is_file() and (STAGE / 'README.md').is_file()
    steps = sorted(p.name for p in (STAGE / 'gnn').iterdir() if p.is_dir() and p.name[0] == 's')
    assert steps == ['s05_block_counting', 's06_gnn_block_cfg', 's07_pool_blocks_to_functions', 's08_gnn_call_graph',
                     's09_pool_functions_to_binary', 's10_unused_import_features', 's11_final_classifier']


@pytest.mark.parametrize('module', ['experiments.train', 's03b_cfg_feature.gnn.s10_unused_import_features.import_side'])
def test_the_commands_run_as_modules_from_the_package_root(module, tmp_path):
    r = run(['-m', module, '--help'], tmp_path)
    assert r.returncode == 0 and 'usage' in r.stdout.lower(), r.stderr


def test_the_stage_imports_only_the_package_and_the_libraries():
    """No import of the branch's flat layout (pipeline.*, training.*, evaluation.*, transformer_paths.*, token_sequence.*)."""
    old = {'pipeline', 'training', 'evaluation', 'transformer_paths', 'token_sequence'}
    offenders = []
    for path in list((STAGE).rglob('*.py')) + [ROOT / 'experiments' / 'train.py', ROOT / 'common' / 'registry.py']:
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(
                node, ast.ImportFrom) and node.module else []
            offenders += [f'{path.name}: {n}' for n in names if n.split('.')[0] in old]
    assert not offenders, offenders


def test_the_modules_of_earlier_steps_read_nothing_of_this_one_at_import_time(tmp_path):
    code = ('import sys\nimport common.config, common.manifests, common.run_io, s04_evaluation.baselines, s04_evaluation.gates\n'
            'print(sorted(m for m in sys.modules if m.startswith(("s03b_cfg_feature", "experiments.train", "torch"))))')
    r = run(['-c', code], tmp_path)
    assert r.returncode == 0 and r.stdout.strip() == '[]', (r.stdout, r.stderr)


def test_the_lazy_imports_of_the_baselines_resolve_now():
    r = subprocess.run([sys.executable, '-c', 'from s04_evaluation.baselines import import_matrix\n'
                        'from s03b_cfg_feature.gnn.s10_unused_import_features.import_side import load_imports\n'
                        'from s03b_cfg_feature.gnn.s10_unused_import_features.imports_feat import ImportVocab\nprint("ok")'],
                       cwd=ROOT, capture_output=True, text=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    assert r.returncode == 0 and r.stdout.strip() == 'ok', r.stderr
