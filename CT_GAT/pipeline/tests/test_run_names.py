"""RunConfig.name and .group are the names the runs on the 3-block-level-counting branch were stored under.

Two recordings, both generated on that branch (3764cc7) by its RunConfig, each row the fields that differ from the defaults
and the name and group it gave:
  fixtures/run_names.json       every run directory of the author's machine whose options still exist (57 directories of
                                the options removed in step 25 are not read);
  fixtures/run_names_grid.json  every naming option alone and with others, over the three protocols (the run directories
                                hold no readout, aux, tokenseq, block-vector or val-seed run), plus 400 seeded random
                                combinations.
The class here must give the same names and groups.
"""
import json
from pathlib import Path

import pytest

from common.config import RunConfig

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
RECORDED = json.loads((FIXTURES / 'run_names.json').read_text(encoding='utf-8'))
GRID = json.loads((FIXTURES / 'run_names_grid.json').read_text(encoding='utf-8'))


def test_the_recording_covers_the_options_of_the_branch():
    assert len(RECORDED) == 390 and len({r['name'] for r in RECORDED}) == 384
    used = {k for r in RECORDED for k in r['config']}
    assert {'protocol', 'fold', 'control', 'mask_blocks', 'budget', 'select_on', 'arch_balance', 'train_fraction',
            'encoder', 'direction', 'vocab', 'gnn_fn_layers'} <= used
    assert {r['config'].get('protocol', 'bundle') for r in RECORDED} == {'kfold', 'kfold_v3'}      # the bundle runs were transformer runs


@pytest.mark.parametrize('recorded', RECORDED, ids=[f"{r['name']}#{i}" for i, r in enumerate(RECORDED)])
def test_existing_run_names_unchanged(recorded):
    cfg = RunConfig(**recorded['config'])
    assert (cfg.name, cfg.group) == (recorded['name'], recorded['group'])


def test_the_grid_covers_the_options_the_run_directories_lack():
    used = {k for r in GRID for k in r['config']}
    assert {'readout', 'aux_recon', 'aux_edge', 'val_seed', 'seq_len', 'max_windows', 'block_vec', 'imports', 'vocab', 'tag',
            'select_on', 'untyped', 'no_call_mp', 'drop_library'} <= used
    assert {r['config'].get('model', 'gnn') for r in GRID} == {'gnn', 'tokenseq'}
    assert {r['config'].get('protocol', 'bundle') for r in GRID} == {'bundle', 'kfold', 'kfold_v3'}
    assert len(GRID) == 581 and len({r['name'] for r in GRID}) == 514


@pytest.mark.parametrize('recorded', GRID, ids=[f"{r['name']}#{i}" for i, r in enumerate(GRID)])
def test_every_naming_option_keeps_its_name(recorded):
    cfg = RunConfig(**recorded['config'])
    assert (cfg.name, cfg.group) == (recorded['name'], recorded['group'])
