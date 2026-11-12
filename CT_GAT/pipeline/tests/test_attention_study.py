import json

import pytest
import torch
from torch import nn
from torch_geometric.nn import GATConv

from experiments.attention_study.models import candidates, make_model, use_attention_factory
from experiments.attention_study.report import effective_attention, result_record
from experiments.attention_study.watch import remaining_local
from experiments.final_paper import trial as paper_trial
from experiments.grid_study import grids
from experiments.model_grid.data import digest
from experiments.model_grid.models import BlockData, collate, forward


def test_attention_grid_keeps_width_fixed_and_pooling_paired():
    for model, expected in ((1, 12), (2, 12), (3, 96)):
        base = dict(grids.base_spec(model), dropout=0.3)
        specs = candidates(model, base)
        assert len(specs) == expected
        assert {spec['attention_dropout'] for spec in specs} == {0, 0.1, 0.2}
        assert all(spec['dropout'] == 0.3 for spec in specs)
        if model in (1, 3):
            assert all(spec['width'] == 96 and 96 % spec['transformer_heads'] == 0 for spec in specs)
        if model == 3:
            groups = {}
            for spec in specs:
                key = digest({k: v for k, v in spec.items() if k != 'pooling'})
                groups.setdefault(key, set()).add(spec['pooling'])
            assert all(pooling == {'mean', 'cls'} for pooling in groups.values())
    with pytest.raises(ValueError, match='divisible'):
        candidates(3, grids.base_spec(3), heads=(8, 12), width=64)


@pytest.mark.parametrize('model', (1, 2, 3))
def test_attention_dropout_is_independent_and_single_layer_heads_are_effective(model):
    spec = candidates(model, dict(grids.base_spec(model), dropout=0.3,
                                 gat_layers=1, gat_hidden=8, transformer_layers=1),
                      heads=(12,), dropouts=(0.2,))[0]
    network = make_model(model, 128, spec, 0)
    assert {layer.p for layer in network.modules() if isinstance(layer, nn.Dropout)} == {0.3}
    for layer in network.modules():
        if isinstance(layer, nn.MultiheadAttention):
            assert layer.num_heads == 12 and layer.dropout == 0.2
        if isinstance(layer, GATConv):
            assert layer.heads == 12 and layer.dropout == 0.2
    actual = effective_attention(model, spec)
    assert actual['effective_gat_heads'] == ('[12]' if model != 1 else None)


def test_model3_attention_adapter_forward_and_backward():
    spec = candidates(3, dict(grids.base_spec(3), gat_layers=1, gat_hidden=8,
                             transformer_layers=1, block_chunk=2),
                      heads=(12,), dropouts=(0.2,))[1]
    network = make_model(3, 128, spec, 0)
    sample = BlockData(
        insn_ids=torch.tensor([[1, 2, 2, 0, 1], [21, 2, 2, 76, 3], [1, 2, 2, 0, 0]]),
        insn_block=torch.tensor([0, 0, 1]), edge_index=torch.tensor([[0], [1]]),
        edge_attr=torch.zeros(1, 8), y=torch.tensor(1), num_nodes=2,
    )
    batch, label = collate(3, [sample], torch.device('cpu'))
    output = forward(3, network, batch)
    loss = nn.functional.cross_entropy(output, label)
    loss.backward()
    assert output.shape == (1, 2) and torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in network.encoder.parameters())
    assert any(parameter.grad is not None for parameter in network.gat.parameters())


def test_factory_restores_after_failure():
    original = paper_trial.make_model
    with pytest.raises(RuntimeError, match='fixture'):
        with use_attention_factory():
            assert paper_trial.make_model is make_model
            raise RuntimeError('fixture')
    assert paper_trial.make_model is original


def test_original_one_layer_head_labels_are_reported_accurately():
    spec = dict(grids.base_spec(3), gat_layers=1, gat_heads=8, dropout=0.1)
    assert effective_attention(3, spec)['effective_gat_heads'] == '[1]'
    assert effective_attention(3, spec)['dropout_control'] == 'shared'


def test_followup_waits_for_local_jobs_and_ignores_remote_reservations():
    config = dict(remote_jobs={'radeon': 'RX 9070 XT'})
    state = dict(jobs={'radeon': {'state': 'pending'}, 'local': {'state': 'running'},
                       'done': {'state': 'complete'}})
    assert remaining_local(config, state) == {'local': 'running'}
    state['jobs']['local']['state'] = 'pending'
    assert remaining_local(config, state) == {'local': 'pending'}
    state['jobs']['local']['state'] = 'complete'
    assert remaining_local(config, state) == {}


def test_export_retains_parameters_and_rejects_mismatched_contract(tmp_path):
    spec = candidates(3, dict(grids.base_spec(3), gat_layers=1, gat_heads=8),
                      heads=(12,), dropouts=(0.2,))[0]
    contract = dict(model=3, spec=spec, seed=0, epochs=12, patience=8,
                    evaluate_test=False, source_hashes={}, torch_version='fixture',
                    dataset_fingerprint='dataset')
    score = dict(accuracy=0.9, auc=0.95, macro_f1=0.89, tp=8, tn=10, fp=2, fn=0, n=20)
    result = dict(fingerprint=digest(contract), validation_fixed=score,
                  epochs_run=4, best_epoch=2, parameters=123, epoch_seconds_mean=1.0,
                  peak_gpu_mb=0)
    (tmp_path / 'config.json').write_text(json.dumps(contract))
    metrics = tmp_path / 'metrics.json'
    metrics.write_text(json.dumps(result))
    row = result_record(metrics, tmp_path, 'attention_followup', 0, 'screen', 'signal500', 'none')
    assert row['gat_heads'] == 12 and row['effective_gat_heads'] == '[12]'
    assert row['attention_dropout'] == row['gat_attention_dropout'] == 0.2
    assert row['precision'] == 0.8 and row['epochs_requested'] == 12
    result['fingerprint'] = 'wrong'
    metrics.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='contract mismatch'):
        result_record(metrics, tmp_path, 'attention_followup', 0, 'screen', 'signal500', 'none')
