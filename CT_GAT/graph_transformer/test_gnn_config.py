"""python -m pytest test_gnn_config.py: RunConfig fields, naming, validation and the CLI."""
import subprocess
import sys
from pathlib import Path

import pytest

from config import RUNS, RunConfig
from train import parse_config

HERE = Path(__file__).parent


def gnn(**kw):
    return RunConfig(model='gnn', **kw)


def test_defaults():
    cfg = RunConfig()
    assert (cfg.model, cfg.encoder, cfg.readout, cfg.control, cfg.tag) == ('transformer', None, 'fixed', '', '')
    assert (cfg.untyped, cfg.no_call_mp, cfg.stack_tfidf) == (False, False, False)
    assert (cfg.mask_tokens, cfg.drop_functions, cfg.drop_library, cfg.mask_blocks) == (0.0,) * 4
    assert (cfg.aux_recon, cfg.aux_edge, cfg.aux_weight) == (0.0, 0.0, 0.3)
    assert (cfg.gnn_fn_layers, cfg.gnn_bin_layers, cfg.batch_blocks, cfg.fixed_epochs) == (3, 2, 300000, 0)
    assert (cfg.direction, cfg.drop_edges, cfg.flip_edges, cfg.train_fraction) == ('both', 0.0, 0.0, 1.0)


def test_block_encoder_and_segment_cap():
    assert RunConfig().block_encoder == 'mean'
    assert RunConfig(max_segment_blocks=64).segment_cap == 64
    assert gnn().block_encoder == 'tfidf'
    assert gnn(encoder='mean').block_encoder == 'mean'
    assert gnn().segment_cap is None


@pytest.mark.parametrize('kw', [
    dict(model='gnn', no_graph=True),
    dict(encoder='mean'), dict(readout='gated'), dict(control='chain'), dict(untyped=True),
    dict(no_call_mp=True), dict(stack_tfidf=True), dict(mask_tokens=0.1), dict(drop_functions=0.1),
    dict(drop_library=0.1), dict(mask_blocks=0.1), dict(aux_recon=0.1), dict(aux_edge=0.1),
    dict(model='gnn', encoder='bert'), dict(model='gnn', readout='mean'), dict(model='gnn', control='rotate'),
    dict(model='gnn', untyped=True, control='permute_types'), dict(model='gnn', untyped=True, control='no_edges'),
    dict(model='gnn', mask_tokens=1.0), dict(model='gnn', drop_library=-0.1), dict(model='gnn', aux_edge=1.5),
    dict(model='cnn'),
    dict(direction='forward'), dict(drop_edges=0.1), dict(flip_edges=0.1),
    dict(model='gnn', direction='sideways'), dict(model='gnn', drop_edges=1.0), dict(model='gnn', flip_edges=-0.1),
    dict(train_fraction=0.0), dict(train_fraction=1.5), dict(model='gnn', train_fraction=0.0),
])
def test_rejected(kw):
    with pytest.raises(ValueError):
        RunConfig(**kw)


def test_existing_checks_stay():
    for kw in (dict(protocol='kfold'), dict(fold=1), dict(protocol='kfold', fold=5), dict(arch='arm'),
               dict(protocol='holdout')):
        with pytest.raises(ValueError):
            RunConfig(**kw)


def test_accepted():
    gnn(untyped=True, control='chain', mask_tokens=0.15, aux_weight=0.5, readout='gated', encoder='tfidf')
    RunConfig(aux_weight=0.5, gnn_fn_layers=4, batch_blocks=1000, fixed_epochs=3, tag='x')


def test_variant_suffix_order():
    cfg = gnn(control='shuffle', untyped=True, no_call_mp=True, stack_tfidf=True, mask_tokens=0.15,
              drop_functions=0.05, drop_library=0.4, mask_blocks=0.15, aux_recon=0.3, encoder='mean',
              readout='gated')
    assert cfg.variant == 'gnn-shuffle-untyped-nocall-stack-mtok15-dfn5-dlib40-mblk15-aux-enc-mean-gated'
    assert gnn().variant == 'gnn'
    assert gnn(aux_edge=0.1).variant == 'gnn-aux'
    for control, suffix in (('no_edges', 'noedges'), ('chain', 'chain'), ('shuffle', 'shuffle'),
                            ('permute_types', 'permtypes')):
        assert gnn(control=control).variant == f'gnn-{suffix}'
    assert RunConfig().variant == 'graph'
    assert RunConfig(no_graph=True).variant == 'nograph'


def test_new_fields_accepted():
    gnn(direction='undirected', drop_edges=0.2, flip_edges=0.2, train_fraction=0.25)
    RunConfig(train_fraction=0.5)
    RunConfig(train_fraction=1.0)


def test_variant_new_suffixes():
    assert gnn(direction='forward').variant == 'gnn-fwd'
    assert gnn(direction='backward').variant == 'gnn-bwd'
    assert gnn(direction='undirected').variant == 'gnn-undir'
    assert gnn(drop_edges=0.2).variant == 'gnn-dedge20' and gnn(flip_edges=0.2).variant == 'gnn-flip20'
    assert gnn(gnn_fn_layers=8).variant == 'gnn-L8' and gnn(gnn_bin_layers=4).variant == 'gnn-B4'
    assert gnn(gnn_fn_layers=3, gnn_bin_layers=2).variant == 'gnn'
    assert gnn(train_fraction=0.5).variant == 'gnn-tf50'
    assert RunConfig(train_fraction=0.5).variant == 'graph-tf50'
    assert RunConfig(train_fraction=0.25, no_graph=True).variant == 'nograph-tf25'
    full = gnn(control='shuffle', untyped=True, no_call_mp=True, stack_tfidf=True, mask_tokens=0.15,
               drop_functions=0.05, drop_library=0.4, mask_blocks=0.15, aux_recon=0.3, direction='forward',
               drop_edges=0.2, flip_edges=0.1, gnn_fn_layers=8, gnn_bin_layers=1, train_fraction=0.25,
               encoder='mean', readout='gated')
    assert full.variant == ('gnn-shuffle-untyped-nocall-stack-mtok15-dfn5-dlib40-mblk15-aux-fwd-dedge20-flip10-L8-B1'
                            '-tf25-enc-mean-gated')


def test_new_names():
    assert gnn(direction='forward').group == 'bundle_x86_gnn-fwd'
    assert gnn(gnn_fn_layers=8).group == 'bundle_x86_gnn-L8'
    assert gnn(train_fraction=0.25, protocol='kfold', fold=1).name == 'kfold1_x86_gnn-tf25_s0'
    assert RunConfig(train_fraction=0.5, protocol='kfold', fold=0, seed=1).name == 'kfold0_x86_graph-tf50_s1'
    assert gnn(drop_edges=0.2, tag='t').name == 'bundle_x86_gnn-dedge20_s0_t'


def test_names_and_groups():
    assert RunConfig().name == 'bundle_x86_graph_s0'
    assert RunConfig(protocol='kfold', fold=2, no_graph=True, seed=1).name == 'kfold2_x86_nograph_s1'
    assert gnn(protocol='kfold', fold=0, control='shuffle').name == 'kfold0_x86_gnn-shuffle_s0'
    assert gnn(encoder='mean', control='no_edges').name == 'bundle_x86_gnn-noedges-enc-mean_s0'
    assert gnn(mask_tokens=0.15).group == 'bundle_x86_gnn-mtok15'
    assert RunConfig(tag='t').name == 'bundle_x86_graph_s0_t'
    assert RunConfig(tag='t').group == 'bundle_x86_graph_t'
    assert gnn(tag='t', seed=2).name == 'bundle_x86_gnn_s2_t'


def test_existing_run_names_unchanged():
    dirs = [d for d in RUNS.iterdir() if (d / 'config.json').exists() and d.name != 'baselines'
            and not d.name.startswith(('smoke_', 'cap512_'))]
    assert dirs
    bad = [d.name for d in dirs if RunConfig.load(d / 'config.json').name != d.name]
    assert not bad, bad


def test_save_load_roundtrip(tmp_path):
    cfg = gnn(control='chain', mask_tokens=0.15, tag='t')
    cfg.save(tmp_path / 'config.json')
    assert RunConfig.load(tmp_path / 'config.json') == cfg


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
    cfg, _ = parse_config(['--protocol', 'kfold', '--fold', '2', '--no-graph', '--seed', '1', '--d-model', '64',
                           '--lr', '1e-3', '--epochs', '3'])
    assert cfg == RunConfig(protocol='kfold', fold=2, no_graph=True, seed=1, d_model=64, lr=1e-3, epochs=3)
    assert cfg.name == 'kfold2_x86_nograph_s1'


@pytest.mark.parametrize('argv', [['--model', 'cnn'], ['--encoder', 'bert'], ['--control', 'rotate'],
                                  ['--model', 'gnn', '--no-graph'], ['--mask-tokens', '0.1'],
                                  ['--direction', 'sideways'], ['--direction', 'forward'], ['--drop-edges', '0.1'],
                                  ['--model', 'gnn', '--flip-edges', '1'], ['--train-fraction', '0']])
def test_cli_rejects(argv):
    with pytest.raises(SystemExit):
        parse_config(argv)


def test_help_exits_zero():
    r = subprocess.run([sys.executable, 'train.py', '--help'], cwd=HERE, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for flag in ('--model', '--encoder', '--control', '--untyped', '--mask-tokens', '--tag', '--direction',
                 '--drop-edges', '--flip-edges', '--train-fraction'):
        assert flag in r.stdout
