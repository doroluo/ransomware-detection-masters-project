"""python -m pytest tests/test_tokenseq_config.py: `RunConfig` with model 'tokenseq' (the names, the forced validation carve, the
options it refuses) and what the registry does with it.

Ported from the 3-block-level-counting branch (tests/test_harness_prep.py, the tokenseq half). CPU only, no data. The rest of
that file is in tests/test_run_io.py, tests/test_paths.py and tests/test_run_names.py (the recorded names of every option)."""
import dataclasses
import json

import pytest

from common.config import MAX_WINDOWS, SEQ_LEN, RunConfig
from common.registry import spec_for
from experiments.train import parse_config
from s03a_transformer.token_sequence import train as ts_train


def tok(**kw):
    return RunConfig(**{'protocol': 'kfold', 'fold': 0, 'arch': 'x86', 'model': 'tokenseq', **kw})


def test_tokenseq_names():
    assert tok().name == 'kfold0_x86_tokenseq_s0' and tok().group == 'kfold_x86_tokenseq'
    assert [tok(fold=f, seed=s).name for f, s in ((0, 0), (4, 1), (2, 2))] == [
        'kfold0_x86_tokenseq_s0', 'kfold4_x86_tokenseq_s1', 'kfold2_x86_tokenseq_s2']
    assert tok(seq_len=256).name == 'kfold0_x86_tokenseq-L256_s0'
    assert tok(max_windows=32).name == 'kfold0_x86_tokenseq-w32_s0'
    assert tok(seq_len=256, max_windows=32, seed=2).name == 'kfold0_x86_tokenseq-L256-w32_s2'
    assert tok(seq_len=SEQ_LEN, max_windows=MAX_WINDOWS).name == tok().name == 'kfold0_x86_tokenseq_s0'       # defaults: no suffix
    assert (SEQ_LEN, MAX_WINDOWS) == (512, 64)
    assert tok(tag='t').name == 'kfold0_x86_tokenseq_s0_t'


def test_tokenseq_val_seed_is_forced_to_the_export_carve_and_never_printed():
    for seed in (0, 1, 2):
        cfg = tok(seed=seed)
        assert cfg.val_seed == 0 and 'vs' not in cfg.name and 'vs' not in cfg.group      # seeds 1, 2 would print -vs0 for a gnn
        assert tok(seed=seed, val_seed=0).name == cfg.name
    with pytest.raises(ValueError, match='validation carve'):
        tok(seed=1, val_seed=1)
    with pytest.raises(ValueError, match='validation carve'):
        tok(val_seed=2)
    assert RunConfig(protocol='kfold', fold=0, model='gnn', seed=1, val_seed=0).name == 'kfold0_x86_gnn-vs0_s1'    # others unchanged


def test_tokenseq_needs_a_fold_of_an_export():
    with pytest.raises(ValueError):
        RunConfig(protocol='bundle', model='tokenseq')
    with pytest.raises(ValueError):
        tok(fold=None)


GNN_ONLY = [dict(control='no_edges'), dict(control='shuffle'), dict(mask_blocks=0.15), dict(mask_tokens=0.1),
            dict(drop_functions=0.1), dict(drop_library=0.1), dict(aux_recon=0.1), dict(drop_edges=0.1), dict(flip_edges=0.1),
            dict(block_input='vec', block_vec='x'), dict(encoder='mean'), dict(readout='gated'), dict(untyped=True),
            dict(no_call_mp=True), dict(stack_tfidf=True), dict(direction='forward'), dict(subgraph='cap', budget=5000),
            dict(vocab='noapi'), dict(imports='header'), dict(gnn_fn_layers=4), dict(gnn_bin_layers=1),
            dict(train_fraction=0.5), dict(arch_balance=True), dict(select_on='cell'), dict(fixed_epochs=3)]


@pytest.mark.parametrize('option', GNN_ONLY, ids=lambda o: next(iter(o)))
def test_tokenseq_rejects_the_gnn_only_options(option):
    with pytest.raises(ValueError, match='do not apply'):
        tok(**option)


def test_check_supported_rejects_tokenseq_with_gnn_options_however_it_was_built():
    ok = tok()
    ts_train.check_supported(ok)
    spec_for(ok).check(ok)                                               # the registry entry makes the same check
    bad = tok()
    object.__setattr__(bad, 'control', 'no_edges')                       # bypasses __post_init__, as an edited config.json would not
    with pytest.raises(ValueError, match='control'):
        ts_train.check_supported(bad)
    with pytest.raises(ValueError, match='control'):
        spec_for(bad).check(bad)
    with pytest.raises(ValueError, match='tokenseq'):
        ts_train.check_supported(RunConfig(protocol='kfold', fold=0, model='gnn'))


def test_seq_len_and_max_windows_belong_to_tokenseq_only():
    with pytest.raises(ValueError, match="model='tokenseq'"):
        RunConfig(protocol='kfold', fold=0, model='gnn', seq_len=256)
    with pytest.raises(ValueError, match="model='tokenseq'"):
        RunConfig(protocol='kfold', fold=0, model='gnn', max_windows=8)
    for bad in (dict(seq_len=0), dict(max_windows=0)):
        with pytest.raises(ValueError):
            tok(**bad)


def test_tokenseq_config_round_trips_and_gnn_flags_exist(tmp_path):
    cfg = tok(seed=1, seq_len=256, max_windows=16, epochs=3, patience=2)
    cfg.save(tmp_path / 'config.json')
    again = RunConfig.load(tmp_path / 'config.json')
    assert again == cfg and again.name == cfg.name and again.val_seed == 0
    assert json.loads((tmp_path / 'config.json').read_text(encoding='utf-8'))['val_seed'] == 0
    old = {k: v for k, v in dataclasses.asdict(RunConfig(protocol='kfold', fold=0, model='gnn')).items()
           if k not in ('seq_len', 'max_windows')}
    (tmp_path / 'old.json').write_text(json.dumps(old), encoding='utf-8')                  # a config.json from before Model 1
    assert RunConfig.load(tmp_path / 'old.json') == RunConfig(protocol='kfold', fold=0, model='gnn')
    parsed, _ = parse_config(['--protocol', 'kfold', '--fold', '0', '--model', 'gnn'])
    assert (parsed.seq_len, parsed.max_windows) == (512, 64)
