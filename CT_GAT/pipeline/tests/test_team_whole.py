"""python -m pytest tests/test_team_whole.py: the sigwhole arms of `train_b` (the uncapped build), the per-batch node budget of `fit.py`,
and the whole rows of `gates --stage team`, including main's five-channel Model 1 on the same build (`--model fivechannel
--max-instructions 512 --tag whole`, a plain RunConfig run here).

The arm table and the refusals need nothing; the node budget needs torch_geometric (the tests skip with a reason without it). The
runs are synthetic, on the tiny cache of `tiny_cache.py`."""
import json

import numpy as np
import pytest
import torch

from common.config import RunConfig
from experiments.stages import team as team_stage
from experiments.stages.team import team_group
from experiments.team_pipelines import common as tp, fit, train_b
from s04_evaluation import gate_tables, gates, stats
from test_team_pipelines import fake_group, fresh_team_caches, needs_pyg, pinned_team_modules  # noqa: F401
from test_team_v2 import fake_build, fake_payload
from test_team_v3 import settings, synthetic_build
from tiny_cache import tiny  # noqa: F401  (a fixture)

WHOLE_ARMS = (('sigwhole', 'all', 'sig500v3'), ('sigwhole-feat', 'feat', 'sig500v3-feat'), ('sigwhole-emb', 'emb', 'sig500v3-emb'))
FIVE = dict(protocol='kfold', arch='x86', model='fivechannel', max_instructions=512, tag='whole')


def test_whole_arms_names_build_source_features_and_budget():
    assert train_b.ARMS[-3:] == ('sigwhole', 'sigwhole-feat', 'sigwhole-emb') and len(train_b.ARMS) == 8
    for arm, features, v3 in WHOLE_ARMS:
        row = train_b.ARM_TABLE[arm]
        assert train_b.ARM_DIR[arm].name == 'b_sigwhole_v3' and row.sources == ('main-fix', 'repo')
        assert (row.sampler, row.num_blocks, row.features) == ('signal', None, features) == (
            'signal', None, train_b.ARM_FEATURES[v3])
        assert train_b.ARM_DIR[v3].name == 'b_sig500_v3'
        cfg = tp.team_config(3, 2, arm)
        assert cfg.name == f'kfold3_x86_gnn_s2_team-{arm}' and cfg.group == team_group(arm)
        assert train_b.ARM_HP[arm] == {**train_b.HP, 'node_budget': train_b.NODE_BUDGET, 'gpu_cap_gb': train_b.GPU_CAP_GB}
        assert train_b.GPU_CAP_GB <= 12
        assert train_b.ARM_HP[v3] is train_b.HP and 'node_budget' not in train_b.HP                  # v3 and older: no budget
    assert all('node_budget' not in train_b.ARM_HP[a] for a in train_b.ARMS[:5])


def test_whole_arms_refuse_a_capped_build_and_a_contradicting_flag(tmp_path):
    quiet = lambda *_: None                                                                   # noqa: E731
    with pytest.raises(SystemExit, match='trains on --features feat, not all'):
        train_b.run('sigwhole-feat', 0, 0, smoke=2, device='cpu', features='all', log=quiet)
    with pytest.raises(SystemExit, match='trains on --features all, not emb'):
        train_b.run('sigwhole', 0, 0, smoke=2, device='cpu', features='emb', log=quiet)
    with pytest.raises(SystemExit, match='not --source main-c7899f0'):
        train_b.run('sigwhole', 0, 0, smoke=2, device='cpu', source='main-c7899f0', log=quiet)
    capped = settings(tmp_path / 'capped', source='main-fix', sample='signal', num_blocks=500)
    no_cap = settings(tmp_path / 'nocap', source='main-fix', sample='signal')
    for arm in ('sigwhole', 'sigwhole-feat', 'sigwhole-emb'):
        for b in (capped, no_cap):
            with pytest.raises(SystemExit, match='uncapped'):
                train_b.run(arm, 0, 0, smoke=2, device='cpu', b_dir=b, log=quiet)
    old = settings(tmp_path / 'old', source='gnn-331ce0b', sample='signal', num_blocks=1000000)
    with pytest.raises(SystemExit, match='was built with gnn-331ce0b'):
        train_b.run('sigwhole', 0, 0, smoke=2, device='cpu', b_dir=old, log=quiet)
    with pytest.raises(SystemExit):
        train_b.main(['--fold', '0', '--arm', 'sigwhole-emb', '--features', 'feat', '--smoke', '2', '--device', 'cpu'])


@needs_pyg
def test_a_whole_arm_trains_on_a_synthetic_uncapped_build_in_either_layout(tmp_path, monkeypatch, tiny):          # noqa: F811
    rows = tp.sample_rows()
    monkeypatch.setattr(fit, 'RUNS', tmp_path / 'runs')
    monkeypatch.setitem(train_b.ARM_HP, 'sigwhole-emb', {**train_b.HP_WHOLE, 'epochs': 2, 'node_budget': 12, 'gpu_cap_gb': None})
    for layout in ('build', 'zip'):
        folder = synthetic_build(tmp_path / layout, sorted(rows), rows, layout=layout, num_blocks=1000000)
        out = train_b.run('sigwhole-emb', 2, 1, device='cpu', b_dir=folder, log=lambda *_: None)
        metrics = json.loads((out['dir'] / 'metrics.json').read_text(encoding='utf-8'))
        assert metrics['in_dim'] == 64 and metrics['features'] == 'emb' and metrics['build_source'] == 'main-fix'
        assert any(h['split_batches'] > 0 for h in out['history'])                                  # the budget of 12 nodes cut batches
        import shutil
        shutil.rmtree(out['dir'])


# the node budget ----------------------------------------------------------------------------------------------------------


def graphs_of(sizes, labels=None, dim=6):
    from torch_geometric.data import Data
    out = []
    for k, n in enumerate(sizes):
        g = torch.Generator().manual_seed(k)
        e = torch.randint(0, n, (2, 2 * n), generator=g)
        out.append(Data(x=torch.randn(n, dim, generator=g), edge_index=e, edge_attr=torch.randn(2 * n, 3, generator=g),
                        y=torch.tensor(k % 2 if labels is None else labels[k])))
    return out


def tiny_net():
    """Per-graph logits that do not depend on the other graphs of the batch (no batch norm, no dropout)."""
    import torch.nn as nn
    from torch_geometric.nn import global_mean_pool

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(6, 8), nn.Linear(8, 2)

        def forward(self, data):
            return self.b(global_mean_pool(torch.relu(self.a(data.x)), data.batch))
    return TinyNet()


@needs_pyg
def test_pack_cuts_in_order_and_is_the_identity_within_the_budget():
    gs = graphs_of([5, 7, 3, 20, 4, 4, 9])
    assert fit.pack(gs, None) == [gs] and fit.pack([], None) == [] and fit.pack([], 10) == []
    assert fit.pack(gs, 10 ** 6) == [gs]                                                      # within the budget: one part, the same list
    parts = fit.pack(gs, 12)
    assert [[int(g.num_nodes) for g in p] for p in parts] == [[5, 7], [3], [20], [4, 4], [9]]  # a graph above the budget is alone
    assert [g for p in parts for g in p] == gs
    assert all(sum(int(g.num_nodes) for g in p) <= 12 or len(p) == 1 for p in parts)
    slices = [gs[i:i + 3] for i in range(0, len(gs), 3)]
    assert fit.eval_chunks(gs, 3, 10 ** 6) == slices and fit.eval_chunks(gs, 3, None) == slices   # same groups as train_cfg's slices
    assert [g for c in fit.eval_chunks(gs, 3, 12) for g in c] == gs and len(fit.eval_chunks(gs, 3, 12)) > len(slices)


def stepped(batch, budget, seed=0):
    import torch.nn as nn
    torch.manual_seed(seed)
    model = tiny_net()
    w = torch.tensor([1.0, 2.5])
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    loss, parts = fit.train_batch(model, batch, nn.CrossEntropyLoss(weight=w), nn.CrossEntropyLoss(weight=w, reduction='sum'), opt,
                                  torch.device('cpu'), budget)
    return loss, parts, [p.detach().clone() for p in model.parameters()]


@needs_pyg
def test_a_cut_batch_takes_the_step_of_the_whole_batch():
    from torch_geometric.data import Batch
    gs = graphs_of([30, 50, 20, 40, 10, 60, 25, 35], labels=[1, 0, 1, 1, 0, 0, 1, 0])
    batch = Batch.from_data_list(gs)
    loss, parts, whole = stepped(batch, None)
    assert parts == 1
    assert stepped(batch, 10 ** 6)[1] == 1 and all(torch.equal(a, b) for a, b in zip(whole, stepped(batch, 10 ** 6)[2]))   # not hit
    for budget in (100, 60, 1):
        loss_cut, parts_cut, cut = stepped(batch, budget)
        assert parts_cut > 1
        assert abs(loss_cut - loss) < 1e-5
        for a, b in zip(whole, cut):
            assert torch.allclose(a, b, atol=1e-5), budget


@needs_pyg
def test_fit_with_a_budget_records_the_cuts_and_keeps_the_loss_close(tiny):                                       # noqa: F811
    manifest = tp.fold_manifest(2, 1)
    ids = [r['sample_id'] for r in manifest.train][:24]
    val = [r['sample_id'] for r in manifest.val][:6]
    labels = {i: tp.sample_rows()[i]['label'] for i in ids + val}
    gs = dict(zip(ids + val, graphs_of([20 + 5 * k for k in range(len(ids + val))], [labels[i] for i in ids + val])))
    hp = {'lr': 1e-2, 'weight_decay': 0.0, 'batch_size': 4, 'epochs': 2, 'patience': 8}

    def go(extra):
        torch.manual_seed(0)
        return fit.fit(tiny_net(), ids, val, gs, hp={**hp, **extra}, seed=0, device=torch.device('cpu'), log=lambda *_: None)['history']
    base, cut = go({}), go({'node_budget': 100})
    assert all('split_batches' not in h for h in base)
    assert all(h['split_batches'] > 0 for h in cut) and abs(cut[0]['loss'] - base[0]['loss']) < 1e-3


@needs_pyg
def test_predict_with_a_budget_matches_the_unbudgeted_order_and_values():
    from torch_geometric.data import Batch

    class TC:
        @staticmethod
        @torch.no_grad()
        def predict_proba(model, graphs, device, batch_size):
            model.eval()
            probs, labels = [], []
            for start in range(0, len(graphs), batch_size):
                b = Batch.from_data_list(graphs[start:start + batch_size])
                probs.append(model(b).softmax(dim=1)[:, 1])
                labels.append(b.y.view(-1))
            return (torch.cat(probs), torch.cat(labels)) if probs else (torch.empty(0), torch.empty(0, dtype=torch.long))
    gs = graphs_of([30, 50, 20, 40, 10, 60, 25, 35, 15])
    model = tiny_net()
    p0, y0 = fit.predict(TC, model, gs, torch.device('cpu'), 4, None)
    p1, y1 = fit.predict(TC, model, gs, torch.device('cpu'), 4, 70)
    assert torch.allclose(p0, p1, atol=1e-6) and torch.equal(y0, y1)
    assert fit.predict(TC, model, [], torch.device('cpu'), 4, 70)[0].numel() == 0


# gates -----------------------------------------------------------------------------------------------------------------------


def test_main_five_channel_whole_runs_are_plain_run_configs_in_the_gates_group():
    cfg = RunConfig(**FIVE, fold=2, seed=1)
    assert cfg.name == 'kfold2_x86_fivechannel-L512_s1_whole' and cfg.group == team_stage.FIVECHANNEL_WHOLE == 'kfold_x86_fivechannel-L512_whole'
    assert RunConfig(**{**FIVE, 'tag': ''}, fold=0).group == 'kfold_x86_fivechannel-L512'


def test_team_gates_from_synthetic_whole_runs_including_a_fivechannel_group(tiny, tmp_path, monkeypatch):     # noqa: F811
    runs = tmp_path / 'runs'
    runs.mkdir()
    base = dict(protocol='kfold', arch='x86', model='gnn')
    fake_group(runs, lambda f, s: RunConfig(**base, fold=f, seed=s, mask_blocks=0.15), 0.5)                          # R7
    for arm, signal, seeds in (('sig500v3', 1.0, (0, 1, 2)), ('sig500v3-feat', 0.8, (0, 1, 2)), ('sig500v3-emb', 1.0, (0, 1, 2)),
                               ('sigwhole', 1.4, (0, 1, 2)), ('sigwhole-feat', 0.9, (0, 1, 2)), ('sigwhole-emb', 1.4, (0, 1))):
        fake_group(runs, lambda f, s, arm=arm: tp.team_config(f, s, arm), signal, seeds=seeds)
    fake_group(runs, lambda f, s: RunConfig(**FIVE, fold=f, seed=s), 1.2)
    monkeypatch.setattr(stats, 'RUNS', runs)
    gate_tables.pooled.cache_clear()
    text = '\n'.join(gates.run_stage('team', 40))
    gate_tables.pooled.cache_clear()
    pair = {(l.split('|')[1].strip(), l.split('|')[2].strip()): l.split('|')[-2].strip()
            for l in text.splitlines() if l.startswith('| ') and l.count('|') == 6}
    for v3, whole in zip(team_stage.V3, team_stage.WHOLE):
        assert pair[(v3, whole)] != 'missing'
    assert pair[('R7 typed, whole graph', 'fivechannel-L512 whole')] == 'beats'
    assert pair[('R7 typed, whole graph', 'team sigwhole')] == 'beats'
    assert 'seeds [0, 1] only' in pair[(team_stage.WHOLE[1], team_stage.WHOLE[2])]
    assert '| fivechannel-L512 whole | kfold_x86_fivechannel-L512_whole | ' in text and '| fivechannel-L512 whole | missing' not in text
    assert 'Match rates of `b_sigwhole_v3`: missing (no build_summary.json' in text


def test_team_gates_whole_match_rate_line(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    gate_tables.pooled.cache_clear()
    (tmp_path / 'team').mkdir()
    build = tmp_path / 'team' / 'b_sigwhole_v3'
    fake_build(build, {'r1': fake_payload([0] * 10, 1), 'g1': fake_payload([1] * 8 + [0] * 2, 0)})
    (build / 'build_summary.json').write_text(json.dumps({'processed': 1810, 'total': 1810}), encoding='utf-8')
    text = '\n'.join(gates.run_stage('team', 30))
    gate_tables.pooled.cache_clear()
    line = [l for l in text.splitlines() if l.startswith('Match rates of `b_sigwhole_v3`')]
    assert len(line) == 1 and 'ransomware 1 binaries, mean 0.000' in line[0] and 'goodware 1 binaries, mean 0.800' in line[0]
    assert [l for l in text.splitlines() if l.startswith('Match rates of `b_sig500_v3`')][0].startswith(
        'Match rates of `b_sig500_v3`: missing (no build_summary.json')                   # the v3 line is its own, unchanged
