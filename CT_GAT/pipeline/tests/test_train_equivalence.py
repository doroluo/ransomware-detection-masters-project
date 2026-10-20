"""Two epochs of R7 on fold 0 of the real cache, trained by `experiments.train`, equal the 3-block-level-counting branch's.

fixtures/eqcheck_r7_fold0.json records `python -m training.train --protocol kfold --fold 0 --model gnn --mask-blocks 0.15
--seed 0 --tag eqcheck --epochs 2` on the branch (3764cc7) with deterministic CUDA algorithms. This test runs the same
configuration here under the same switches and compares every epoch's loss and validation scores and the test scores to 1e-6.
The numbers are those of the GPU and torch build that recorded them (named in the fixture); on another GPU the test skips.
Without deterministic algorithms the branch's own runs differ from each other in the first epoch loss by 0.16, so the switch is
part of the check, not of the trainer.
"""
import csv
import json
from pathlib import Path

import pytest
import torch

from common.config import RunConfig

RECORDED = json.loads((Path(__file__).resolve().parent / 'fixtures' / 'eqcheck_r7_fold0.json').read_text(encoding='utf-8'))
TOLERANCE = 1e-6


@pytest.fixture
def deterministic():
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(before)


def test_the_fixture_is_the_r7_recipe():
    assert RECORDED['run'] == RunConfig(model='gnn', protocol='kfold', fold=0, seed=0, mask_blocks=0.15, tag='eqcheck').name
    assert [h['epoch'] for h in RECORDED['history']] == [1, 2]


@pytest.mark.real_data
@pytest.mark.gpu
def test_two_epochs_of_r7_on_fold_0_equal_the_branchs(tmp_path, monkeypatch, deterministic):
    if torch.cuda.get_device_name(0) != RECORDED['gpu']:
        pytest.skip(f"the numbers were recorded on {RECORDED['gpu']}")
    from experiments import train
    monkeypatch.setattr(train, 'RUNS', tmp_path)
    cfg = RunConfig(model='gnn', protocol='kfold', fold=0, seed=0, mask_blocks=0.15, tag='eqcheck', epochs=2)
    metrics = train.run(cfg)
    assert cfg.name == RECORDED['run']
    with (tmp_path / cfg.name / 'history.csv').open(encoding='utf-8') as f:
        got = list(csv.DictReader(f))
    assert [int(r['epoch']) for r in got] == [1, 2]
    for row, want in zip(got, RECORDED['history']):
        for key in ('loss', 'val_macro_f1', 'val_accuracy', 'val_auc', 'val_recall'):
            assert abs(float(row[key]) - want[key]) <= TOLERANCE, (row['epoch'], key, row[key], want[key])
    assert metrics['best_epoch'] == RECORDED['best_epoch'] and abs(metrics['val_macro_f1'] - RECORDED['val_macro_f1']) <= TOLERANCE
    for key, value in RECORDED['test'].items():
        assert abs(metrics['test'][key] - value) <= TOLERANCE, key
    auc = metrics['selection']['auc']
    want = RECORDED['auc_checkpoint']
    assert auc['best_epoch'] == want['best_epoch'] and auc['threshold_val'] == want['threshold_val']
    assert abs(auc['val_auc'] - want['val_auc']) <= TOLERANCE
    assert abs(auc['test']['auc'] - want['test_auc']) <= TOLERANCE and abs(auc['test']['macro_f1'] - want['test_macro_f1']) <= TOLERANCE
