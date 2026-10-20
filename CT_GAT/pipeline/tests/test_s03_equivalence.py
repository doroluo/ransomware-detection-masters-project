"""python -m pytest tests/test_s03_equivalence.py -m "real_data and gpu": stages 3c and 3a on the real data equal the
3-block-level-counting branch's.

(a) `precompute_blockvec` with the reference encoder on fold 0 gives the keys and vectors that the branch wrote to
    `<cache>/side/blockvec/ours_cap0_ref/fold0` (compared byte for byte; the checkpoint on disk records the branch's module name,
    which `encoder_contract.LEGACY_MODULES` maps to the new one). Skipped without that output or the export.
(b) two epochs of path 2 (typed GNN on the CodeBERT vectors, 15% block masking) on fold 0, trained by `experiments.train`, equal
    the branch's `training.train`; (c) two epochs of Model 1 (`--model tokenseq`) on fold 0 equal the branch's `token_sequence.train`.
    fixtures/eqcheck_s03_fold0.json records the branch's history (3764cc7, deterministic algorithms, CUBLAS_WORKSPACE_CONFIG=:4096:8);
    the numbers are those of the GPU and torch build named in it, and on another GPU (b) and (c) skip.
"""
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from common.config import CACHE, EXPORT, RUNS, RunConfig

RECORDED = json.loads((Path(__file__).resolve().parent / 'fixtures' / 'eqcheck_s03_fold0.json').read_text(encoding='utf-8'))
TOLERANCE = 1e-6
KEYS = ('loss', 'val_macro_f1', 'val_accuracy', 'val_auc', 'val_recall')
pytestmark = pytest.mark.real_data


@pytest.fixture
def deterministic():
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(before)


def test_the_fixture_names_the_runs_of_the_two_arms():
    path2 = RunConfig(model='gnn', protocol='kfold', fold=0, seed=0, mask_blocks=0.15, block_input='vec',
                      block_vec='ours_cap0_codebert', tag='eqcheck')
    tokenseq = RunConfig(model='tokenseq', protocol='kfold', fold=0, arch='x86', seed=0)
    assert RECORDED['path2']['run'] == path2.name == 'kfold0_x86_gnn-mblk15-bv-ours_cap0_codebert_s0_eqcheck'
    assert RECORDED['tokenseq']['run'] == tokenseq.name == 'kfold0_x86_tokenseq_s0'
    for arm in ('path2', 'tokenseq'):
        assert [h['epoch'] for h in RECORDED[arm]['history']] == [1, 2]


@pytest.mark.gpu
def test_the_reference_encoder_vectors_of_fold_0_equal_the_branchs(tmp_path):
    from s03c_cfg_transformer import precompute_blockvec as pb
    theirs = CACHE / 'side' / 'blockvec' / 'ours_cap0_ref' / 'fold0'
    if not (theirs / 'vectors.npy').is_file() or not (EXPORT / 'ours_cap0' / 'store' / 'store.json').is_file() \
            or not (RUNS / 'encoders' / 'ref' / 'fold0' / 'best.pt').is_file():
        pytest.skip('needs the branch\'s ours_cap0_ref output, the export ours_cap0 and the reference checkpoint')
    pb.precompute(EXPORT / 'ours_cap0', 'ref', [0], cache=tmp_path, device='cuda', log=lambda *_: None)
    mine = tmp_path / 'side' / 'blockvec' / 'ours_cap0_ref' / 'fold0'
    assert np.array_equal(np.load(mine / 'keys.npy'), np.load(theirs / 'keys.npy'))
    assert np.array_equal(np.load(mine / 'vectors.npy'), np.load(theirs / 'vectors.npy'))
    ours, branch = (json.loads((d / 'meta.json').read_text(encoding='utf-8')) for d in (mine, theirs))
    for key in ('name', 'export_sha', 'checkpoint_sha', 'manifest_sha', 'n_distinct', 'n_blocks', 'd', 'dtype', 'autocast'):
        assert ours[key] == branch[key], key


def check_against_recorded(want, metrics, out):
    with (out / 'history.csv').open(encoding='utf-8') as f:
        got = list(csv.DictReader(f))
    assert [int(r['epoch']) for r in got] == [1, 2]
    for row, rec in zip(got, want['history']):
        for key in KEYS:
            assert abs(float(row[key]) - rec[key]) <= TOLERANCE, (row['epoch'], key, row[key], rec[key])
    assert metrics['best_epoch'] == want['best_epoch'] and abs(metrics['val_macro_f1'] - want['val_macro_f1']) <= TOLERANCE
    for key, value in want['test'].items():
        assert abs(metrics['test'][key] - value) <= TOLERANCE, key
    auc, rec = metrics['selection']['auc'], want['auc_checkpoint']
    assert auc['best_epoch'] == rec['best_epoch'] and auc['threshold_val'] == rec['threshold_val']
    assert abs(auc['val_auc'] - rec['val_auc']) <= TOLERANCE
    assert abs(auc['test']['auc'] - rec['test_auc']) <= TOLERANCE and abs(auc['test']['macro_f1'] - rec['test_macro_f1']) <= TOLERANCE


def skip_on_another_gpu():
    if torch.cuda.get_device_name(0) != RECORDED['gpu']:
        pytest.skip(f"the numbers were recorded on {RECORDED['gpu']}")


@pytest.mark.gpu
def test_two_epochs_of_path_2_on_fold_0_equal_the_branchs(tmp_path, monkeypatch, deterministic):
    skip_on_another_gpu()
    if not (CACHE / 'side' / 'blockvec' / 'ours_cap0_codebert' / 'fold0' / 'vectors.npy').is_file():
        pytest.skip('needs the CodeBERT block vectors ours_cap0_codebert')
    from experiments import train
    monkeypatch.setattr(train, 'RUNS', tmp_path)
    cfg = RunConfig(model='gnn', protocol='kfold', fold=0, seed=0, mask_blocks=0.15, block_input='vec',
                    block_vec='ours_cap0_codebert', tag='eqcheck', epochs=2)
    metrics = train.run(cfg)
    assert cfg.name == RECORDED['path2']['run']
    check_against_recorded(RECORDED['path2'], metrics, tmp_path / cfg.name)


@pytest.mark.gpu
def test_two_epochs_of_model_1_on_fold_0_equal_the_branchs(tmp_path, monkeypatch, deterministic):
    skip_on_another_gpu()
    if not (EXPORT / 'ours_cap0' / 'store' / 'store.json').is_file():
        pytest.skip('needs the export ours_cap0')
    from experiments import train
    from s03a_transformer.token_sequence import spec as spec_module
    from s03a_transformer.token_sequence import train as ts
    for module in (train, spec_module, ts):
        monkeypatch.setattr(module, 'RUNS', tmp_path)
    cfg = RunConfig(model='tokenseq', protocol='kfold', fold=0, arch='x86', seed=0, epochs=2)
    metrics = train.run(cfg)
    assert cfg.name == RECORDED['tokenseq']['run']
    check_against_recorded(RECORDED['tokenseq'], metrics, tmp_path / cfg.name)
    shutil.rmtree(tmp_path / cfg.name)
