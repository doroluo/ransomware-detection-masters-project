"""python -m pytest tests/test_train_cli.py: the command line of `python -m experiments.train` and what it hands to the run.

Ported from the 3-block-level-counting branch (tests/test_gnn_config.py, the CLI and cache parts): every flag of `training/train.py`
exists with the same name, type and default, and the cache of a run is the one `common/paths.py` resolves.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from common.config import COMBINED_CACHE, DEFAULT_CACHE, RunConfig
from common.manifests import make_manifest
from experiments.train import parse_args, parse_config

ROOT = Path(__file__).resolve().parents[1]
real_data = pytest.mark.real_data
needs_combined = pytest.mark.skipif(not (Path(COMBINED_CACHE) / 'index.csv').exists(), reason='needs the combined cache')


def gnn(**kw):
    return RunConfig(model='gnn', **kw)


def test_cli_gnn():
    cfg, out = parse_config(['--model', 'gnn', '--control', 'shuffle', '--mask-tokens', '0.15'])
    assert cfg == gnn(control='shuffle', mask_tokens=0.15) and out is None
    assert isinstance(cfg.mask_tokens, float)


def test_cli_all_fields():
    argv = ['--model', 'gnn', '--encoder', 'mean', '--readout', 'gated', '--control', 'chain', '--untyped',
            '--no-call-mp', '--stack-tfidf', '--mask-tokens', '0.1', '--drop-functions', '0.2',
            '--drop-library', '0.3', '--mask-blocks', '0.4', '--aux-recon', '0.5', '--aux-edge', '0.6',
            '--aux-weight', '0.7', '--gnn-fn-layers', '5', '--gnn-bin-layers', '4', '--batch-blocks', '1000',
            '--fixed-epochs', '7', '--tag', 'x', '--out', 'custom']
    cfg, out = parse_config(argv)
    assert out == 'custom'
    assert cfg == RunConfig(model='gnn', encoder='mean', readout='gated', control='chain', untyped=True,
                            no_call_mp=True, stack_tfidf=True, mask_tokens=0.1, drop_functions=0.2,
                            drop_library=0.3, mask_blocks=0.4, aux_recon=0.5, aux_edge=0.6, aux_weight=0.7,
                            gnn_fn_layers=5, gnn_bin_layers=4, batch_blocks=1000, fixed_epochs=7, tag='x')
    assert (cfg.gnn_fn_layers, cfg.batch_blocks, cfg.fixed_epochs) == (5, 1000, 7)


def test_cli_new_flags():
    cfg, _ = parse_config(['--model', 'gnn', '--direction', 'backward', '--drop-edges', '0.2', '--flip-edges', '0.1',
                           '--train-fraction', '0.5', '--protocol', 'kfold', '--fold', '3'])
    assert cfg == gnn(direction='backward', drop_edges=0.2, flip_edges=0.1, train_fraction=0.5, protocol='kfold',
                      fold=3)
    assert cfg.name == 'kfold3_x86_gnn-bwd-dedge20-flip10-tf50_s0'
    assert parse_config(['--train-fraction', '0.25'])[0] == RunConfig(train_fraction=0.25)
    assert parse_config([])[0].direction == 'both'


def test_cli_defaults_and_old_flags():
    cfg, out = parse_config([])
    assert cfg == RunConfig() and out is None
    cfg, _ = parse_config(['--protocol', 'kfold', '--fold', '2', '--seed', '1', '--d-model', '64',
                           '--lr', '1e-3', '--epochs', '3'])
    assert cfg == RunConfig(protocol='kfold', fold=2, seed=1, d_model=64, lr=1e-3, epochs=3)
    assert cfg.name == 'kfold2_x86_gnn_s1'


def test_cli_has_the_flags_of_the_options_this_step_serves():
    cfg, _ = parse_config(['--model', 'gnn', '--mask-blocks', '0.15', '--vocab', 'noapi', '--imports', 'header',
                           '--subgraph', 'cap', '--budget', '20000', '--arch-balance', '--select-on', 'cell',
                           '--control', 'no_edges', '--protocol', 'kfold', '--fold', '0', '--val-seed', '1'])
    assert cfg == gnn(mask_blocks=0.15, vocab='noapi', imports='header', budget=20000, arch_balance=True,
                      select_on='cell', control='no_edges', protocol='kfold', fold=0, val_seed=1)
    vec, _ = parse_config(['--protocol', 'kfold', '--fold', '0', '--block-input', 'vec', '--block-vec',
                           'ours_cap0_codebert'])
    assert (vec.block_input, vec.block_vec) == ('vec', 'ours_cap0_codebert')


@pytest.mark.parametrize('argv', [['--model', 'cnn'], ['--encoder', 'bert'], ['--control', 'rotate'],
                                  ['--model', 'transformer'], ['--no-graph'], ['--imports', 'unreferenced'],
                                  ['--subgraph', 'lcc'], ['--direction', 'sideways'],
                                  ['--model', 'gnn', '--flip-edges', '1'], ['--train-fraction', '0'],
                                  ['--model', 'tokenseq']])
def test_cli_rejects(argv):
    with pytest.raises(SystemExit):
        parse_config(argv)


def test_help_exits_zero():
    r = subprocess.run([sys.executable, '-m', 'experiments.train', '--help'], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for flag in ('--model', '--encoder', '--control', '--untyped', '--mask-tokens', '--tag', '--direction',
                 '--drop-edges', '--flip-edges', '--train-fraction', '--vocab', '--imports', '--cache', '--out',
                 '--block-input', '--select-on', '--arch-balance'):
        assert flag in r.stdout


def test_cli_kfold_v3_and_matched():
    cfg, _ = parse_config(['--protocol', 'kfold_v3', '--fold', '4', '--arch', 'matched', '--model', 'gnn',
                           '--control', 'shuffle'])
    assert cfg == gnn(protocol='kfold_v3', fold=4, arch='matched', control='shuffle')
    assert cfg.name == 'v3kfold4_matched_gnn-shuffle_s0'
    with pytest.raises(SystemExit):
        parse_config(['--protocol', 'kfold_v3'])
    with pytest.raises(SystemExit):
        parse_config(['--protocol', 'kfold_v4', '--fold', '1'])


def test_arch_balance_cli_and_roundtrip(tmp_path):
    cfg, _ = parse_config(['--model', 'gnn', '--arch-balance', '--protocol', 'kfold_v3', '--fold', '1', '--arch',
                           'all'])
    assert cfg == gnn(arch_balance=True, protocol='kfold_v3', fold=1, arch='all')
    assert cfg.name == 'v3kfold1_all_gnn-ab_s0'
    assert parse_config(['--arch-balance'])[0].name == 'bundle_x86_gnn-ab_s0'
    assert parse_config([])[0].arch_balance is False
    cfg.save(tmp_path / 'config.json')
    assert RunConfig.load(tmp_path / 'config.json') == cfg
    old = tmp_path / 'old.json'
    old.write_text('{"model": "gnn"}')
    assert RunConfig.load(old).arch_balance is False


def test_step11_cli_and_roundtrip(tmp_path):
    argv = ['--model', 'gnn', '--select-on', 'cell', '--subgraph', 'cap', '--budget', '20000', '--encoder', 'rawlog',
            '--mask-blocks', '0.15']
    cfg, _ = parse_config(argv)
    assert cfg == gnn(select_on='cell', subgraph='cap', budget=20000, encoder='rawlog', mask_blocks=0.15)
    assert cfg.variant == 'gnn-mblk15-enc-rawlog-selcell-sg-cap20k'
    for enc in ('raw', 'presence'):
        assert parse_config(['--model', 'gnn', '--encoder', enc])[0].encoder == enc
    d = parse_config(['--model', 'gnn'])[0]
    assert (d.select_on, d.subgraph, d.budget) == ('overall', 'cap', 0)
    cfg.save(tmp_path / 'config.json')
    assert RunConfig.load(tmp_path / 'config.json') == cfg
    old = tmp_path / 'old.json'
    old.write_text('{"model": "gnn", "mask_blocks": 0.15}')
    assert RunConfig.load(old).name == gnn(mask_blocks=0.15).name
    with pytest.raises(SystemExit):
        parse_config(['--model', 'gnn', '--select-on', 'neither'])
    with pytest.raises(SystemExit):
        parse_config(['--subgraph', 'lcc'])


def test_train_refuses_a_config_for_another_cache():
    from experiments import train
    with pytest.raises(ValueError, match='cache'):
        train.run(RunConfig(cache=str(ROOT / 'not' / 'this' / 'cache')))


def test_train_accepts_a_dataset_name_as_the_cache(tmp_path):
    """`--cache <name>` names a dataset of datasets.json: common.paths resolves it, and the run's check must not read the name
    as a directory."""
    cache = tmp_path / 'cache'
    cache.mkdir()
    registry = tmp_path / 'datasets.json'
    registry.write_text('{"mine": {"cache": "cache", "protocols": ["bundle"]}}', encoding='utf-8')
    code = ('from experiments import train\ncfg, _ = train.parse_config()\n'
            'from common.config import CACHE\nprint(CACHE, cfg.cache)\n'
            'try:\n    train.run(cfg)\nexcept FileNotFoundError as e:\n    print("missing", e)')
    env = {k: v for k, v in os.environ.items() if not k.startswith('GT_')}
    env.update(GT_DATASETS=str(registry), CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1')
    r = subprocess.run([sys.executable, '-c', code, '--cache', 'mine'], cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    first, second = r.stdout.splitlines()
    assert Path(first.split()[0]) == cache and first.split()[1] == 'mine'
    assert second.startswith('missing') and 'index.csv' in second and 'GT_CACHE' in second       # got past the cache check


PROBE = """
from common import config, manifests
from experiments import train
cfg, _ = train.parse_config()
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
    matched = probe(['--protocol=kfold_v3', '--fold', '0', '--arch', 'matched'])
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
