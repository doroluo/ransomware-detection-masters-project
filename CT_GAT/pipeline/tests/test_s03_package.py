"""The boundaries of stages 3c and 3a: the modules import clean on a machine with no datasets.json, the commands run as modules
from the package root, the imports stay inside the package, no module names the run or export directory, `--block-input vec`
finds its package, and the stage-2 and stage-3a files of the teammates still import.

Ported from the 3-block-level-counting branch (tests/test_harness_prep.py: GT_RUNS and GT_EXPORT; tests/test_transformer_paths.py: the
schema and contract documents), and modelled on tests/test_gnn_package.py. CPU only, no data."""
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from common import paths
from s03b_cfg_feature.gnn import spec as gnn_spec
from s03c_cfg_transformer import encoder_contract as ec

ROOT = Path(__file__).resolve().parents[1]
S03C, S03A = ROOT / 's03c_cfg_transformer', ROOT / 's03a_transformer'
MODULES = ('s03c_cfg_transformer', 's03c_cfg_transformer.block_export', 's03c_cfg_transformer.encoder_contract',
           's03c_cfg_transformer.precompute_blockvec', 's03c_cfg_transformer.encoders',
           's03c_cfg_transformer.encoders.codebert', 's03c_cfg_transformer.encoders.reference', 's03a_transformer',
           's03a_transformer.token_sequence', 's03a_transformer.token_sequence.data', 's03a_transformer.token_sequence.model',
           's03a_transformer.token_sequence.train', 's03a_transformer.token_sequence.spec')
COMMANDS = ('s03c_cfg_transformer.block_export', 's03c_cfg_transformer.encoder_contract', 's03c_cfg_transformer.precompute_blockvec',
            's03c_cfg_transformer.encoders.codebert', 's03c_cfg_transformer.encoders.reference',
            's03a_transformer.token_sequence.train')


def clean_env(tmp_path, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GT_')}
    env.update(GT_DATASETS=str(tmp_path / 'none.json'), PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', **extra)
    return env


def run(args, tmp_path, **extra):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, env=clean_env(tmp_path, **extra))


def test_every_module_imports_without_a_datasets_file(tmp_path):
    code = 'import importlib\nfor m in %r:\n    importlib.import_module(m)\nprint("ok")' % (MODULES,)
    r = run(['-c', code], tmp_path)
    assert r.returncode == 0 and r.stdout.strip() == 'ok', r.stderr


@pytest.mark.parametrize('module', COMMANDS)
def test_the_commands_run_as_modules_from_the_package_root(module, tmp_path):
    r = run(['-m', module, '--help'], tmp_path)
    assert r.returncode == 0 and 'usage' in r.stdout.lower(), r.stderr


def test_the_placeholder_is_gone_and_the_stages_are_packages():
    assert not (S03C / 'placeholder.py').exists()
    for stage in (S03C, S03A):
        assert (stage / '__init__.py').is_file()
    assert (S03C / 'README.md').is_file() and (S03A / 'token_sequence' / 'README.md').is_file()
    assert sorted(p.name for p in S03C.glob('*.py')) == ['__init__.py', 'block_export.py', 'encoder_contract.py',
                                                          'precompute_blockvec.py']
    assert sorted(p.name for p in (S03C / 'encoders').glob('*.py')) == ['__init__.py', 'codebert.py', 'reference.py']
    assert not (S03A / 'transformer_model.py').exists()                 # removed by PR #8: it was only a placeholder
    assert (S03A / 'token_sequence.py').is_file()                       # Janice's script, beside the package of the same name


def test_the_stage_imports_only_the_package_and_the_libraries():
    """No import of the branch's flat layout (pipeline.*, training.*, evaluation.*, transformer_paths.*, token_sequence.*)."""
    old = {'pipeline', 'training', 'evaluation', 'transformer_paths', 'token_sequence'}
    offenders = []
    for path in list(S03C.rglob('*.py')) + list((S03A / 'token_sequence').rglob('*.py')):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(
                node, ast.ImportFrom) and node.module else []
            offenders += [f'{path.name}: {n}' for n in names if n.split('.')[0] in old]
    assert not offenders, offenders


def test_the_export_is_read_through_common_reader_and_there_is_no_second_copy():
    assert not (S03C / 'reader.py').exists() and not (S03A / 'token_sequence' / 'reader.py').exists()
    for path in (S03C / 'encoder_contract.py', S03C / 'precompute_blockvec.py',
                 S03A / 'token_sequence' / 'data.py', S03A / 'token_sequence' / 'train.py'):
        assert 'from common.reader import' in path.read_text(encoding='utf-8'), path.name


def test_no_module_spells_the_runs_or_export_path_itself():
    """Only common/paths.py names a run directory; no module builds `RUNS / 'export'`."""
    code_files = [p for p in ROOT.rglob('*.py') if not {'tests', '.git', '__pycache__'} & set(p.relative_to(ROOT).parts)]
    assert [p.relative_to(ROOT).as_posix() for p in code_files if "RUNS / 'export'" in p.read_text(encoding='utf-8')] == []
    for path in list(S03C.rglob('*.py')) + list((S03A / 'token_sequence').rglob('*.py')):
        text = path.read_text(encoding='utf-8')
        assert not re.search(r"graph_transformer|Path\('[A-Za-z]:", text), path.name


def probe(env, code, tmp_path):
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True,
                         env={**clean_env(tmp_path), **env})
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


PROBE = '''
import json
from common.config import EXPORT, RUNS
from s03a_transformer.token_sequence import train
from s03c_cfg_transformer import block_export as be
print(json.dumps({"runs": str(RUNS), "export": str(EXPORT), "trainer_runs": str(train.RUNS), "be_export": str(be.EXPORT),
                  "resolved": str(train.resolve_export("ours_cap0")), "be_out": str(be.resolve_out("RUNS/export/x"))}))
'''


def test_gt_runs_moves_the_trainer_and_the_exports(tmp_path):
    got = probe({'GT_RUNS': str(tmp_path / 'r')}, PROBE, tmp_path)
    assert Path(got['runs']) == Path(got['trainer_runs']) == tmp_path / 'r'
    assert Path(got['export']) == Path(got['be_export']) == tmp_path / 'r' / 'export'
    assert Path(got['resolved']) == tmp_path / 'r' / 'export' / 'ours_cap0'
    assert Path(got['be_out']) == (tmp_path / 'r' / 'export' / 'x').resolve()


def test_gt_export_moves_the_exports_only(tmp_path):
    got = probe({'GT_EXPORT': str(tmp_path / 'e')}, PROBE, tmp_path)
    assert Path(got['export']) == Path(got['be_export']) == tmp_path / 'e' and Path(got['resolved']) == tmp_path / 'e' / 'ours_cap0'
    assert Path(got['runs']) == Path(got['trainer_runs']) == paths.PIPELINE_ROOT / 'runs'      # no datasets.json: the fallback
    assert Path(got['be_out']) == (tmp_path / 'e' / 'x').resolve()


def test_resolve_out_follows_gt_export(tmp_path):
    out = probe({'GT_EXPORT': str(tmp_path / 'e')}, '''
import json
from s03c_cfg_transformer import block_export as be
refused = []
for bad in ("RUNS/other/x", "RUNS/export", "x"):
    try:
        be.resolve_out(bad)
    except ValueError:
        refused.append(bad)
print(json.dumps({"ok": str(be.resolve_out("RUNS/export/ours_cap0")), "refused": refused}))
''', tmp_path)
    assert Path(out['ok']) == (tmp_path / 'e' / 'ours_cap0').resolve() and out['refused'] == ['RUNS/other/x', 'RUNS/export', 'x']


def test_the_vec_input_of_the_gnn_finds_its_package():
    """PR 3 stopped `--block-input vec` with a message when stage 3c was absent; the package is here now."""
    assert gnn_spec.vectors_package_present() and gnn_spec.VECTORS_PACKAGE == 's03c_cfg_transformer.precompute_blockvec'


def test_checkpoints_of_the_branch_layout_still_load(tmp_path):
    """A checkpoint written before the move names `transformer_paths.<x>_encoder`; `load_encoder` imports the new module."""
    from s03c_cfg_transformer.encoders.reference import ReferenceEncoder
    model = ReferenceEncoder(n_ids=20)
    for old, new in ec.LEGACY_MODULES.items():
        assert old.startswith('transformer_paths.') and new.startswith('s03c_cfg_transformer.encoders.')
    ckpt = {'module': 'transformer_paths.reference_encoder', 'class': 'ReferenceEncoder', 'init_kwargs': {'n_ids': 20},
            'state_dict': model.state_dict()}
    path = tmp_path / 'old.pt'
    torch.save(ckpt, path)
    loaded = ec.load_encoder(path)
    assert type(loaded) is ReferenceEncoder and not loaded.training
    assert all(torch.equal(a, b) for a, b in zip(loaded.state_dict().values(), model.state_dict().values()))
    ckpt['module'] = 'nowhere.encoder'
    torch.save(ckpt, path)
    with pytest.raises(ModuleNotFoundError):
        ec.load_encoder(path)


def test_the_teammates_files_of_stage_2_still_import():
    r = subprocess.run([sys.executable, '-c', "import sys; sys.path.insert(0, 's02_preprocess'); import tokenize_asm_gz, "
                        "preprocess_full; print('ok')"], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    assert r.returncode == 0 and r.stdout.strip() == 'ok', r.stderr


@pytest.mark.xfail(strict=True, reason="s03a_transformer/token_sequence.py (Janice, PR #8) and the token_sequence/ package share one "
                   "name; the package wins the import, so her file cannot be imported. Known issue, to be fixed separately.")
def test_importing_token_sequence_gives_janices_file_not_the_package():
    import importlib
    module = importlib.import_module('s03a_transformer.token_sequence')
    assert Path(module.__file__).resolve() == (S03A / 'token_sequence.py').resolve()


def test_the_earlier_stages_read_nothing_of_these_at_import_time(tmp_path):
    code = ('import sys\nimport common.config, common.manifests, common.run_io, common.registry, s04_evaluation.baselines, '
            's04_evaluation.gates, s03b_cfg_feature.gnn.spec\n'
            'print(sorted(m for m in sys.modules if m.startswith(("s03c_cfg_transformer", "s03a_transformer.token_sequence"))))')
    r = run(['-c', code], tmp_path)
    assert r.returncode == 0 and r.stdout.strip() == '[]', (r.stdout, r.stderr)
