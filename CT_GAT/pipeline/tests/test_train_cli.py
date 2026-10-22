"""python -m pytest tests/test_train_cli.py: the command line of `python -m experiments.train` and what it hands to the run.

Ported from the 3-block-level-counting branch (tests/test_gnn_config.py, the CLI and cache parts). `--model` is required and names
a model of the registry; the earlier jsonl-only models are on branch `jsonl-reference-models` and the command refuses them with a
message that says so. The cache of a run is the one `common/paths.py` resolves.
"""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from common.config import COMBINED_CACHE, DEFAULT_CACHE, RunConfig
from common.manifests import make_manifest
from common.registry import MODELS
from experiments.train import parse_args, parse_config

ROOT = Path(__file__).resolve().parents[1]
real_data = pytest.mark.real_data
needs_combined = pytest.mark.skipif(not (Path(COMBINED_CACHE) / 'index.csv').exists(), reason='needs the combined cache')


def five(**kw):
    return RunConfig(model='fivechannel', protocol='kfold', fold=0, **kw)


def test_cli_fivechannel():
    cfg, out = parse_config(['--model', 'fivechannel', '--protocol', 'kfold', '--fold', '0', '--max-instructions', '512'])
    assert cfg == five(max_instructions=512) and out is None
    assert cfg.name == 'kfold0_x86_fivechannel-L512_s0' and isinstance(cfg.max_instructions, int)


def test_cli_fields_and_old_flags():
    cfg, out = parse_config(['--model', 'fivechannel', '--protocol', 'kfold', '--fold', '2', '--seed', '1', '--d-model', '64',
                             '--lr', '1e-3', '--epochs', '3', '--threshold-from', 'oof', '--tag', 'x', '--out', 'custom'])
    assert out == 'custom'
    assert cfg == RunConfig(model='fivechannel', protocol='kfold', fold=2, seed=1, d_model=64, lr=1e-3, epochs=3,
                            threshold_from='oof', tag='x')
    assert cfg.name == 'kfold2_x86_fivechannel-thoof_s1_x'


def test_the_model_is_required():
    with pytest.raises(SystemExit):
        parse_config([])
    with pytest.raises(SystemExit):
        parse_config(['--protocol', 'kfold', '--fold', '0'])


@pytest.mark.parametrize('name', ['gnn', 'tokenseq'])
def test_the_removed_models_stop_with_the_branch_that_has_them(name, capsys):
    with pytest.raises(SystemExit) as e:
        parse_config(['--model', name, '--protocol', 'kfold', '--fold', '0'])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert f"model '{name}' is not on this branch" in err and 'jsonl-reference-models' in err and 'fivechannel' in err


def test_the_command_refuses_a_removed_model_from_the_shell():
    r = subprocess.run([sys.executable, '-m', 'experiments.train', '--model', 'gnn', '--protocol', 'kfold', '--fold', '0'],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and 'jsonl-reference-models' in r.stderr and 'Traceback' not in r.stderr


@pytest.mark.parametrize('argv', [['--model', 'cnn'], ['--model', 'transformer'], ['--model', 'fivechannel', '--encoder', 'bert'],
                                  ['--model', 'fivechannel', '--control', 'rotate'], ['--model', 'fivechannel', '--no-graph'],
                                  ['--model', 'fivechannel', '--imports', 'unreferenced'],
                                  ['--model', 'fivechannel', '--subgraph', 'lcc'], ['--model', 'fivechannel', '--direction', 'sideways'],
                                  ['--model', 'fivechannel', '--train-fraction', '0'], ['--model', 'fivechannel', '--select-on', 'neither'],
                                  ['--model', 'fivechannel', '--export', 'x'], ['--model', 'fivechannel', '--window-budget', '4']])
def test_cli_rejects(argv):
    with pytest.raises(SystemExit):
        parse_config(argv)


def test_the_options_a_model_does_not_have_are_refused_by_its_check():
    for flag in (['--control', 'no_edges'], ['--mask-blocks', '0.15'], ['--arch-balance'], ['--vocab', 'noapi']):
        with pytest.raises(SystemExit):
            parse_config(['--model', 'fivechannel', '--protocol', 'kfold', '--fold', '0', *flag])


def test_help_exits_zero():
    r = subprocess.run([sys.executable, '-m', 'experiments.train', '--help'], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for flag in ('--model', '--tag', '--cache', '--out', '--build', '--device', '--smoke', '--max-instructions', '--threshold-from'):
        assert flag in r.stdout
    assert '--export' not in r.stdout and '--window-budget' not in r.stdout


def test_cli_protocols_and_folds():
    with pytest.raises(SystemExit):
        parse_config(['--model', 'fivechannel', '--protocol', 'kfold'])                       # a kfold run needs --fold
    with pytest.raises(SystemExit):
        parse_config(['--model', 'fivechannel', '--protocol', 'kfold_v4', '--fold', '1'])


# a model with the registry's loop, in the place of fivechannel, to reach the checks that loop makes -------------------------

STUB = '''
import sys, types
class Stub:
    name = 'fivechannel'
    requires = ('index.csv',)
    def check(self, cfg): pass
    def build(self, cfg, info): raise NotImplementedError
    loader = forward = build
    def variant_tokens(self, cfg): return ('fivechannel',)
stub = types.ModuleType('stub_model')
stub.Stub = Stub
sys.modules['stub_model'] = stub
from common import registry
registry.MODELS['fivechannel'] = 'stub_model:Stub'
'''


def test_train_refuses_a_config_for_another_cache(monkeypatch):
    from experiments import train

    class Stub:
        name = 'fivechannel'
        requires = ()

        def check(self, cfg): pass
        def build(self, cfg, info): raise NotImplementedError
        loader = forward = build
        def variant_tokens(self, cfg): return ('fivechannel',)

    monkeypatch.setitem(sys.modules, 'stub_model', SimpleNamespace(Stub=Stub))
    monkeypatch.setitem(MODELS, 'fivechannel', 'stub_model:Stub')
    with pytest.raises(ValueError, match='cache'):
        train.run(five(cache=str(ROOT / 'not' / 'this' / 'cache')))


def test_train_accepts_a_dataset_name_as_the_cache(tmp_path):
    """`--cache <name>` names a dataset of datasets.json: common.paths resolves it, and the run's check must not read the name
    as a directory."""
    cache = tmp_path / 'cache'
    cache.mkdir()
    registry = tmp_path / 'datasets.json'
    registry.write_text('{"mine": {"cache": "cache", "protocols": ["bundle"]}}', encoding='utf-8')
    code = (STUB + 'from experiments import train\ncfg, _ = train.parse_config()\n'
            'from common.config import CACHE\nprint(CACHE, cfg.cache)\n'
            'try:\n    train.run(cfg)\nexcept FileNotFoundError as e:\n    print("missing", e)')
    env = {k: v for k, v in os.environ.items() if not k.startswith('GT_')}
    env.update(GT_DATASETS=str(registry), CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1')
    r = subprocess.run([sys.executable, '-c', code, '--model', 'fivechannel', '--protocol', 'kfold', '--fold', '0', '--cache', 'mine'],
                       cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    first, second = r.stdout.splitlines()
    assert Path(first.split()[0]) == cache and first.split()[1] == 'mine'
    assert second.startswith('missing') and 'index.csv' in second and 'GT_CACHE' in second       # got past the cache check


PROBE = """
import sys
from common import config, manifests
from common.config import RunConfig
a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
cfg = RunConfig(protocol=a['--protocol'], fold=int(a['--fold']), arch=a.get('--arch', 'x86'))
manifest = manifests.make_manifest(cfg, manifests.load_index())
ids = [r['sample_id'] for r in manifest.train + manifest.val + manifest.test]
print(config.CACHE, manifests.CACHE, cfg.protocol, cfg.arch, len(ids), len(set(r['corpus'] for r in manifest.test)))
"""


def probe(argv, cache=None):
    env = {k: v for k, v in os.environ.items() if k != 'GT_CACHE'}
    env.update({'GT_CACHE': cache} if cache else {}, CUDA_VISIBLE_DEVICES='-1')
    return subprocess.run([sys.executable, '-c', PROBE, *argv], cwd=ROOT, capture_output=True, text=True, env=env)


def test_kfold_v3_on_an_index_without_the_fold_v3_column_names_the_fix():
    index = [{'sample_id': f's{i}', 'label': i % 2, 'arch': 'x86', 'split': 'train', 'family': f'f{i}', 'fold': i % 2,
              'fold_v3': None, 'corpus': 'mendeley'} for i in range(8)]
    with pytest.raises(ValueError, match='no fold_v3 column; set GT_CACHE to the combined cache'):
        make_manifest(RunConfig(protocol='kfold_v3', fold=0), index)


@real_data
@needs_combined
def test_kfold_v3_without_env_reads_the_combined_cache():
    r = probe(['--protocol', 'kfold_v3', '--fold', '0', '--arch', 'all'])
    assert r.returncode == 0, r.stderr
    config_cache, data_cache, protocol, arch, n, _ = r.stdout.split()
    assert Path(config_cache) == Path(data_cache) == Path(COMBINED_CACHE) and (protocol, arch) == ('kfold_v3', 'all')
    assert int(n) == 5652
    matched = probe(['--protocol', 'kfold_v3', '--fold', '0', '--arch', 'matched'])
    assert matched.returncode == 0, matched.stderr
    assert int(matched.stdout.split()[4]) == 4850


@real_data
@needs_combined
def test_gt_cache_wins_and_other_protocols_keep_v2():
    r = probe(['--protocol', 'kfold_v3', '--fold', '0'], cache=str(COMBINED_CACHE))
    assert r.returncode == 0 and Path(r.stdout.split()[0]) == Path(COMBINED_CACHE), r.stderr
    r = probe(['--protocol', 'kfold_v3', '--fold', '0'], cache=str(DEFAULT_CACHE))   # GT_CACHE wins over the routing to combined_v3
    assert r.returncode == 0, r.stderr
    assert Path(r.stdout.split()[0]) == Path(DEFAULT_CACHE) and Path(r.stdout.split()[0]) != Path(COMBINED_CACHE)
    assert int(r.stdout.split()[-1]) == 1                  # the v2 cache holds Mendeley rows only, so the test fold has one corpus
    r = probe(['--protocol', 'kfold', '--fold', '0'])
    assert r.returncode == 0 and Path(r.stdout.split()[0]) == Path(DEFAULT_CACHE), r.stderr


def test_cli_tag_names_the_fivechannel_run_of_the_uncapped_build():
    """`--tag` is a RunConfig field, so the flag exists for every model: Janice's five-channel Model 1 on the uncapped build
    (Models 2 and 3 are compared with it) is `--model fivechannel --max-instructions 512 --tag whole`."""
    cfg, out, options = parse_args(['--model', 'fivechannel', '--protocol', 'kfold', '--fold', '0', '--seed', '0', '--max-instructions', '512',
                                    '--tag', 'whole', '--smoke', '8', '--device', 'cpu', '--build', 'RUNS/team/b_sigwhole_v3'])
    assert cfg.name == 'kfold0_x86_fivechannel-L512_s0_whole' and cfg.tag == 'whole' and out is None
    assert options == {'device': 'cpu', 'smoke': 8, 'build': 'RUNS/team/b_sigwhole_v3'}
    assert parse_args(['--model', 'fivechannel', '--protocol', 'kfold', '--fold', '0', '--max-instructions', '512'])[0].name == (
        'kfold0_x86_fivechannel-L512_s0')                                                       # without the tag: the earlier runs' names
