"""RunConfig fields, naming, validation and the saved config.

Ported from the 3-block-level-counting branch (tests/test_gnn_config.py). The tests of the command line (`parse_config`,
`--help`, the `train.run` cache check) move with the trainer (S5-3), `test_cache_for_resolution` is test_paths.py, and
`test_existing_run_names_unchanged`, which read the run directories of one machine, is test_run_names.py (recorded names).
"""
import pytest

from common.config import RunConfig


def gnn(**kw):
    return RunConfig(model='gnn', **kw)


def test_defaults():
    cfg = RunConfig()
    assert (cfg.model, cfg.encoder, cfg.readout, cfg.control, cfg.tag) == ('gnn', None, 'fixed', '', '')
    assert (cfg.untyped, cfg.no_call_mp, cfg.stack_tfidf) == (False, False, False)
    assert (cfg.mask_tokens, cfg.drop_functions, cfg.drop_library, cfg.mask_blocks) == (0.0,) * 4
    assert (cfg.aux_recon, cfg.aux_edge, cfg.aux_weight) == (0.0, 0.0, 0.3)
    assert (cfg.gnn_fn_layers, cfg.gnn_bin_layers, cfg.batch_blocks, cfg.fixed_epochs) == (3, 2, 300000, 0)
    assert (cfg.direction, cfg.drop_edges, cfg.flip_edges, cfg.train_fraction) == ('both', 0.0, 0.0, 1.0)


def test_block_encoder():
    assert RunConfig().block_encoder == 'tfidf'
    assert gnn(encoder='mean').block_encoder == 'mean'


@pytest.mark.parametrize('kw', [
    dict(model='transformer'), dict(subgraph='lcc'), dict(imports='unreferenced'),
    dict(model='gnn', encoder='bert'), dict(model='gnn', readout='mean'), dict(model='gnn', control='rotate'),
    dict(model='gnn', untyped=True, control='permute_types'), dict(model='gnn', untyped=True, control='no_edges'),
    dict(model='gnn', mask_tokens=1.0), dict(model='gnn', drop_library=-0.1), dict(model='gnn', aux_edge=1.5),
    dict(model='cnn'),
    dict(model='gnn', direction='sideways'), dict(model='gnn', drop_edges=1.0), dict(model='gnn', flip_edges=-0.1),
    dict(train_fraction=0.0), dict(train_fraction=1.5), dict(model='gnn', train_fraction=0.0),
    dict(model='tokenseq', control='no_edges'),
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
    assert RunConfig(train_fraction=0.5, protocol='kfold', fold=0, seed=1).name == 'kfold0_x86_gnn-tf50_s1'
    assert gnn(drop_edges=0.2, tag='t').name == 'bundle_x86_gnn-dedge20_s0_t'


def test_names_and_groups():
    assert RunConfig().name == 'bundle_x86_gnn_s0'
    assert gnn(protocol='kfold', fold=0, control='shuffle').name == 'kfold0_x86_gnn-shuffle_s0'
    assert gnn(encoder='mean', control='no_edges').name == 'bundle_x86_gnn-noedges-enc-mean_s0'
    assert gnn(mask_tokens=0.15).group == 'bundle_x86_gnn-mtok15'
    assert RunConfig(tag='t').name == 'bundle_x86_gnn_s0_t'
    assert RunConfig(tag='t').group == 'bundle_x86_gnn_t'
    assert gnn(tag='t', seed=2).name == 'bundle_x86_gnn_s2_t'
    assert gnn(protocol='kfold_v3', fold=3, arch='all').name == 'v3kfold3_all_gnn_s0'
    assert gnn(protocol='kfold_v3', fold=3, arch='all').group == 'kfold_v3_all_gnn'
    with pytest.raises(ValueError):
        gnn(protocol='kfold_v3')


def test_removed_options_are_rejected_on_load(tmp_path):
    from common.config import RemovedOption
    for raw in ('{"model": "transformer"}', '{"model": "gnn", "subgraph": "lcc", "budget": 20000}',
                '{"model": "gnn", "imports": "unreferenced"}', '{"no_graph": true}',
                '{"protocol": "bundle", "arch": "x86"}'):                       # no model field: a transformer run
        path = tmp_path / 'config.json'
        path.write_text(raw)
        with pytest.raises(RemovedOption):
            RunConfig.load(path)
        assert RunConfig.try_load(path) is None
    path.write_text('{"model": "gnn", "heads": 4, "fn_layers": 2, "max_segments": 4096}')
    assert RunConfig.load(path) == RunConfig()                                # transformer-only fields are ignored


def test_save_load_roundtrip(tmp_path):
    cfg = gnn(control='chain', mask_tokens=0.15, tag='t')
    cfg.save(tmp_path / 'config.json')
    assert RunConfig.load(tmp_path / 'config.json') == cfg


def test_matched_arch_names_and_groups():
    for protocol, fold in (('bundle', None), ('kfold', 1), ('kfold_v3', 2)):
        assert RunConfig(protocol=protocol, fold=fold, arch='matched').arch == 'matched'
    assert gnn(protocol='kfold_v3', fold=0, arch='matched').name == 'v3kfold0_matched_gnn_s0'
    assert gnn(protocol='kfold_v3', fold=2, arch='all', control='no_edges').name == 'v3kfold2_all_gnn-noedges_s0'
    assert gnn(protocol='kfold_v3', fold=4, arch='matched', untyped=True).group == 'kfold_v3_matched_gnn-untyped'
    assert RunConfig(protocol='kfold_v3', fold=1, arch='matched').name == 'v3kfold1_matched_gnn_s0'
    with pytest.raises(ValueError):
        RunConfig(arch='matching')


def test_cache_field_is_not_part_of_names_or_groups(tmp_path):
    base = gnn(protocol='kfold_v3', fold=3, arch='all')
    assert base.cache == ''
    tagged = RunConfig(model='gnn', protocol='kfold_v3', fold=3, arch='all', cache='C:/elsewhere')
    assert (tagged.name, tagged.group) == (base.name, base.group)
    tagged.save(tmp_path / 'config.json')
    assert RunConfig.load(tmp_path / 'config.json').cache == 'C:/elsewhere'
    (tmp_path / 'old.json').write_text('{"protocol": "bundle", "arch": "x86", "model": "gnn"}')
    assert RunConfig.load(tmp_path / 'old.json') == RunConfig()


def test_arch_balance_field_and_names():
    assert RunConfig().arch_balance is False
    assert gnn(arch_balance=True).variant == 'gnn-ab'
    assert gnn(control='no_edges', arch_balance=True).variant == 'gnn-noedges-ab'
    assert gnn(arch_balance=True, encoder='mean').variant == 'gnn-ab-enc-mean'
    assert gnn(arch_balance=True, readout='gated').variant == 'gnn-ab-gated'
    assert gnn(arch_balance=True, encoder='mean', readout='gated', train_fraction=0.5).variant == (
        'gnn-tf50-ab-enc-mean-gated')
    assert gnn(protocol='kfold_v3', fold=2, arch='all', arch_balance=True).name == 'v3kfold2_all_gnn-ab_s0'
    assert gnn(protocol='kfold_v3', fold=2, arch='all', arch_balance=True).group == 'kfold_v3_all_gnn-ab'
    assert gnn(arch_balance=True).group != gnn().group
    for cfg in (RunConfig(), gnn(control='shuffle'), gnn(train_fraction=0.5)):
        assert 'ab' not in cfg.variant.split('-')


def test_step11_defaults_leave_every_name_unchanged():
    cfg = RunConfig()
    assert (cfg.select_on, cfg.subgraph, cfg.budget) == ('overall', 'cap', 0)
    for c in (RunConfig(), gnn(control='shuffle', mask_blocks=0.15, arch_balance=True),
              gnn(encoder='mean', readout='gated', train_fraction=0.5)):
        assert not {'selcell', 'sg'} & set(c.variant.split('-'))
    assert gnn(mask_blocks=0.15).name == 'bundle_x86_gnn-mblk15_s0'
    assert gnn(subgraph='cap').variant == gnn(budget=0).variant == 'gnn'


def test_select_on_suffix_and_names():
    assert gnn(select_on='cell').variant == 'gnn-selcell'
    cfg = gnn(protocol='kfold_v3', fold=2, arch='all', control='no_edges', mask_blocks=0.15, arch_balance=True,
              select_on='cell')
    assert cfg.name == 'v3kfold2_all_gnn-noedges-mblk15-ab-selcell_s0'
    assert cfg.group == 'kfold_v3_all_gnn-noedges-mblk15-ab-selcell'
    assert gnn(select_on='cell').group != gnn().group


def test_subgraph_suffix_and_budget():
    assert gnn(budget=20000).variant == 'gnn-sg-cap20k'
    assert gnn(subgraph='cap', budget=5000).variant == 'gnn-sg-cap5k'
    assert gnn(subgraph='cap', budget=20000, mask_blocks=0.15).variant == 'gnn-mblk15-sg-cap20k'
    assert gnn(budget=5000, select_on='cell', encoder='raw').variant == 'gnn-enc-raw-selcell-sg-cap5k'
    assert gnn(protocol='kfold', fold=3, mask_blocks=0.15, budget=5000).name == 'kfold3_x86_gnn-mblk15-sg-cap5k_s0'


def test_budget_floor_and_small_budget_names():
    assert gnn(budget=100).budget == 100                                      # the floor is 100 blocks
    with pytest.raises(ValueError):
        gnn(budget=99)
    # below 1000 the suffix spells the budget out, so the two matched-arm budgets cannot share a name
    assert gnn(budget=128, mask_blocks=0.15).variant == 'gnn-mblk15-sg-cap128'
    assert gnn(budget=500, mask_blocks=0.15).variant == 'gnn-mblk15-sg-cap500'
    assert gnn(budget=999).variant == 'gnn-sg-cap999' and gnn(budget=1000).variant == 'gnn-sg-cap1k'
    cfg = gnn(protocol='kfold', fold=2, seed=1, mask_blocks=0.15, subgraph='cap', budget=500, tag='b500')
    assert cfg.name == 'kfold2_x86_gnn-mblk15-sg-cap500_s1_b500'
    assert cfg.group == 'kfold_x86_gnn-mblk15-sg-cap500_b500'


def test_encoder_suffixes():
    for enc in ('mean', 'raw', 'presence', 'rawlog'):
        assert gnn(encoder=enc).variant == f'gnn-enc-{enc}' and gnn(encoder=enc).block_encoder == enc
    assert gnn(encoder='tfidf').variant == 'gnn' and gnn().variant == 'gnn'
    assert gnn(control='no_edges', mask_blocks=0.15, encoder='presence').variant == 'gnn-noedges-mblk15-enc-presence'
    assert gnn(encoder='raw', readout='gated').variant == 'gnn-enc-raw-gated'


@pytest.mark.parametrize('kw', [
    dict(model='gnn', select_on='both'), dict(select_on='bogus'), dict(model='gnn', subgraph='bfs'),
    dict(model='gnn', budget=-1), dict(model='gnn', budget=99), dict(subgraph='lcc'),
    dict(model='gnn', encoder='sum'),
])
def test_step11_rejected(kw):
    with pytest.raises(ValueError):
        RunConfig(**kw)
