"""Step 36: the sigwhole arms of train_b (the uncapped build), the per-batch node budget of fit.py, the GPU lane
scripts/team_whole_runs.sh (60 runs, three of the five-channel model from main's CLI), the whole rows and pairs of
`gates.py --stage team`, and the reading of main's fivechannel runs by evaluation/stats.py."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

from common.config import RUNS, ForeignRun, RunConfig, foreign_identity
from evaluation import gates, stats
from experiments.team_pipelines import common as tp, fit, train_b
from test_team_pipelines import BASH, ROOT, fake_group, needs_bash, needs_real_data, needs_worktrees, pinned_team_modules  # noqa: F401
from test_team_v2 import fake_build, fake_payload

WHOLE_ARMS = (('sigwhole', 'all', 'sig500v3'), ('sigwhole-feat', 'feat', 'sig500v3-feat'), ('sigwhole-emb', 'emb', 'sig500v3-emb'))
FIVE = {'protocol': 'kfold', 'arch': 'x86', 'model': 'fivechannel', 'max_instructions': 512, 'threshold_from': 'val', 'tag': 'whole'}


def test_whole_arms_names_build_source_features_and_budget():
    assert train_b.ARMS[-3:] == ('sigwhole', 'sigwhole-feat', 'sigwhole-emb') and len(train_b.ARMS) == 8
    for arm, features, v3 in WHOLE_ARMS:
        assert train_b.ARM_DIR[arm].name == 'b_sigwhole_v3' and train_b.ARM_SOURCE[arm] == 'main-fix'
        assert train_b.ARM_FEATURES[arm] == train_b.ARM_FEATURES[v3] == features
        assert train_b.ARM_DIR[v3].name == 'b_sig500_v3'
        cfg = tp.team_config(3, 2, arm)
        assert cfg.name == f'kfold3_x86_gnn_s2_team-{arm}' and cfg.group == gates.team_group(arm)
        assert train_b.ARM_HP[arm] == {**train_b.HP, 'node_budget': train_b.NODE_BUDGET}
        assert train_b.ARM_HP[v3] is train_b.HP and 'node_budget' not in train_b.HP                  # v3 and older: no budget
    assert all('node_budget' not in train_b.ARM_HP[a] for a in train_b.ARMS[:5])


def settings(path, **kw):
    path.mkdir(parents=True, exist_ok=True)
    (path / 'settings.json').write_text(json.dumps(kw), encoding='utf-8')
    return path


def test_whole_arms_refuse_a_contradicting_features_source_or_cap(tmp_path):
    quiet = lambda *_: None                                                                   # noqa: E731
    with pytest.raises(SystemExit, match='trains on --features feat, not all'):
        train_b.run('sigwhole-feat', 0, 0, smoke=2, device='cpu', features='all', log=quiet)
    with pytest.raises(SystemExit, match='trains on --features all, not emb'):
        train_b.run('sigwhole', 0, 0, smoke=2, device='cpu', features='emb', log=quiet)
    with pytest.raises(SystemExit, match='not --source main-c7899f0'):
        train_b.run('sigwhole', 0, 0, smoke=2, device='cpu', source='main-c7899f0', log=quiet)
    capped = settings(tmp_path / 'capped', source='main-fix', num_blocks=500)
    whole = settings(tmp_path / 'whole', source='main-fix', num_blocks=1000000)
    no_cap = settings(tmp_path / 'nocap', source='main-fix')
    for arm in ('sigwhole', 'sigwhole-feat', 'sigwhole-emb'):
        for b in (capped, no_cap):
            with pytest.raises(SystemExit, match='reads an uncapped build'):
                train_b.run(arm, 0, 0, smoke=2, device='cpu', b_dir=b, log=quiet)
    for arm in ('sig500v3', 'sig500v3-feat', 'sig500v3-emb'):
        with pytest.raises(SystemExit, match='reads a capped build'):
            train_b.run(arm, 0, 0, smoke=2, device='cpu', b_dir=whole, log=quiet)
    old = settings(tmp_path / 'old', source='gnn-331ce0b', num_blocks=1000000)
    with pytest.raises(SystemExit, match='was built with gnn-331ce0b'):
        train_b.run('sigwhole', 0, 0, smoke=2, device='cpu', b_dir=old, log=quiet)
    with pytest.raises(SystemExit):
        train_b.main(['--fold', '0', '--arm', 'sigwhole-emb', '--features', 'feat', '--smoke', '2', '--device', 'cpu'])


# the node budget ----------------------------------------------------------------------------------------------------------


def graphs_of(sizes, labels=None, dim=6):
    out = []
    for k, n in enumerate(sizes):
        g = torch.Generator().manual_seed(k)
        e = torch.randint(0, n, (2, 2 * n), generator=g)
        out.append(Data(x=torch.randn(n, dim, generator=g), edge_index=e, edge_attr=torch.randn(2 * n, 3, generator=g),
                        y=torch.tensor(k % 2 if labels is None else labels[k])))
    return out


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


class TinyNet(nn.Module):
    """Per-graph logits that do not depend on the other graphs of the batch (no batch norm, no dropout)."""
    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(6, 8), nn.Linear(8, 2)

    def forward(self, data):
        return self.b(global_mean_pool(torch.relu(self.a(data.x)), data.batch))


def stepped(batch, budget, seed=0):
    torch.manual_seed(seed)
    model = TinyNet()
    w = torch.tensor([1.0, 2.5])
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    loss, parts = fit.train_batch(model, batch, nn.CrossEntropyLoss(weight=w), nn.CrossEntropyLoss(weight=w, reduction='sum'), opt,
                                  torch.device('cpu'), budget)
    return loss, parts, [p.detach().clone() for p in model.parameters()]


def test_a_cut_batch_takes_the_step_of_the_whole_batch():
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


@needs_worktrees
@needs_real_data
def test_fit_without_a_budget_is_unchanged_and_with_one_records_the_cuts(pinned_team_modules):      # noqa: F811
    ids = [r['sample_id'] for r in tp.fold_manifest(0, 0).train][:24]
    val = [r['sample_id'] for r in tp.fold_manifest(0, 0).val][:6]
    labels = {i: tp.sample_rows()[i]['label'] for i in ids + val}
    gs = {i: g for i, g in zip(ids + val, graphs_of([20 + 5 * k for k in range(30)], [labels[i] for i in ids + val]))}
    hp = {'lr': 1e-2, 'weight_decay': 0.0, 'batch_size': 4, 'epochs': 2, 'patience': 8}

    def go(extra):
        torch.manual_seed(0)
        out = fit.fit(TinyNet(), ids, val, gs, hp={**hp, **extra}, seed=0, device=torch.device('cpu'), log=lambda *_: None)
        return out['history'], out['auc_state']
    base, state = go({})
    same, state_same = go({'node_budget': 10 ** 9})
    assert all('split_batches' not in h for h in base) and all(h['split_batches'] == 0 for h in same)
    assert [h['loss'] for h in base] == [h['loss'] for h in same] and [h['val_auc'] for h in base] == [h['val_auc'] for h in same]
    assert all(torch.equal(state[k], state_same[k]) for k in state)                           # a budget never hit changes nothing
    cut, _ = go({'node_budget': 100})
    assert all(h['split_batches'] > 0 for h in cut) and abs(cut[0]['loss'] - base[0]['loss']) < 1e-3


def test_predict_with_a_budget_matches_the_unbudgeted_order_and_values():
    tc = type('TC', (), {})()

    @torch.no_grad()
    def predict_proba(model, graphs, device, batch_size):
        model.eval()
        probs, labels = [], []
        for start in range(0, len(graphs), batch_size):
            b = Batch.from_data_list(graphs[start:start + batch_size])
            probs.append(model(b).softmax(dim=1)[:, 1])
            labels.append(b.y.view(-1))
        return (torch.cat(probs), torch.cat(labels)) if probs else (torch.empty(0), torch.empty(0, dtype=torch.long))
    tc.predict_proba = predict_proba
    gs = graphs_of([30, 50, 20, 40, 10, 60, 25, 35, 15])
    model = TinyNet()
    p0, y0 = fit.predict(tc, model, gs, torch.device('cpu'), 4, None)
    p1, y1 = fit.predict(tc, model, gs, torch.device('cpu'), 4, 70)
    assert torch.allclose(p0, p1, atol=1e-6) and torch.equal(y0, y1)
    assert fit.predict(tc, model, [], torch.device('cpu'), 4, 70)[0].numel() == 0


# the lane -----------------------------------------------------------------------------------------------------------------


def lane(*args, env=None):
    r = subprocess.run([BASH, 'scripts/team_whole_runs.sh', *args], cwd=ROOT, capture_output=True, text=True, env=env or os.environ)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


@needs_bash
@needs_real_data
def test_whole_lane_dry_lists_60_runs_named_like_run_config():
    lines = lane('--dry')
    runs = [l.split() for l in lines if l.startswith('run ')]
    assert len(runs) == 60 and lines[0].startswith('team_whole_runs start') and lines[-1].startswith('team_whole_runs done')
    waits = [l for l in lines if l.startswith('wait ')]
    assert len(waits) == 1 and waits[0].endswith('/team/b_sigwhole_v3/build_summary.json reports processed == 1810/1810, none excluded')
    assert 'cpu gates.py --stage team > RUNS/gates_team.md' in lines
    seen, five = [], []
    for _, name, module, *argv in runs:
        opt = dict(zip(argv[::2], argv[1::2]))
        fold, seed = int(opt['--fold']), int(opt['--seed'])
        assert opt['--device'] == 'cuda'
        if module == 'experiments.team_pipelines.train_b':
            arm = opt['--arm']
            assert opt['--source'] == 'main-fix' and opt['--features'] == train_b.ARM_FEATURES[arm] and arm.startswith('sigwhole')
            assert name == tp.team_config(fold, seed, arm).name == RunConfig(
                protocol='kfold', fold=fold, arch='x86', seed=seed, model='gnn', tag=f'team-{arm}').name
            seen.append((seed, arm, fold))
        else:
            assert module == 'experiments.train' and opt['--model'] == 'fivechannel' and opt['--max-instructions'] == '512'
            assert opt['--tag'] == 'whole' and opt['--protocol'] == 'kfold' and opt['--arch'] == 'x86'
            assert opt['--build'].endswith('graph_transformer_runs/team/b_sigwhole_v3')
            assert name == foreign_identity_of(fold, seed)[0] == f'kfold{fold}_x86_fivechannel-L512_s{seed}_whole'
            five.append((seed, fold))
    assert len(set(n for _, n, *_ in runs)) == 60
    assert seen == [(s, a, f) for s in (0, 1, 2) for a, _, _ in WHOLE_ARMS for f in range(5)]
    assert five == [(s, f) for s in (0, 1, 2) for f in range(5)]
    assert any(l.startswith('workdir of the experiments.train runs:') and l.endswith('rdmp-main-split/CT_GAT/pipeline') for l in lines)


def foreign_identity_of(fold, seed, tmp=None):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'config.json'
        path.write_text(json.dumps({**FIVE, 'fold': fold, 'seed': seed}), encoding='utf-8')
        return foreign_identity(path)


@needs_bash
def test_whole_lane_waits_for_the_complete_build_without_exclusions(tmp_path):
    runs = tmp_path / 'runs'
    (runs / 'team' / 'b_sigwhole_v3').mkdir(parents=True)
    env = {**os.environ, 'GT_RUNS': runs.as_posix(), 'GT_PY': Path(sys.executable).as_posix()}
    summary = runs / 'team' / 'b_sigwhole_v3' / 'build_summary.json'
    assert lane('--check-build', env=env) == ['build summary absent']
    for body, want in (({'processed': 1810, 'total': 1810, 'excluded': {}}, 'build complete'),
                       ({'processed': 1810, 'total': 1810}, 'build complete'),
                       ({'processed': 1809, 'total': 1810, 'excluded': {'a' * 64: 'why'}}, 'build incomplete'),
                       ({'processed': 1810, 'total': 1810, 'excluded': {'a' * 64: 'why'}}, 'build incomplete')):
        summary.write_text(json.dumps(body), encoding='utf-8')
        assert lane('--check-build', env=env) == [want], body
    r = subprocess.run([BASH, 'scripts/team_whole_runs.sh'], cwd=ROOT, capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 1 and 'not running' in r.stdout and not any(l.startswith('start ') for l in r.stdout.splitlines())
    assert not list(runs.glob('kfold*'))


@needs_bash
def test_whole_lane_stops_on_the_stop_file_while_waiting(tmp_path):
    runs = tmp_path / 'runs'
    (runs / 'team' / 'b_sigwhole_v3').mkdir(parents=True)
    (runs / 'STOP').write_text('', encoding='utf-8')
    env = {**os.environ, 'GT_RUNS': runs.as_posix(), 'GT_PY': Path(sys.executable).as_posix()}
    r = subprocess.run([BASH, 'scripts/team_whole_runs.sh'], cwd=ROOT, capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0 and 'stopped while waiting for the build' in r.stdout.splitlines()
    assert not any(l.startswith('start ') for l in r.stdout.splitlines()) and not list(runs.glob('kfold*'))


@needs_bash
def test_whole_lane_runs_main_from_its_directory_skips_finished_runs_and_retries_once(tmp_path):
    """The experiments.train runs execute inside GT_MAIN (stubbed here), the others inside the package root; a finished run is
    skipped, a failing one is retried once and its first log kept."""
    runs, main = tmp_path / 'runs', tmp_path / 'main'
    build = runs / 'team' / 'b_sigwhole_v3'
    build.mkdir(parents=True)
    (build / 'build_summary.json').write_text(json.dumps({'processed': 1810, 'total': 1810, 'excluded': {}}), encoding='utf-8')
    (main / 'experiments').mkdir(parents=True)
    (main / 'experiments' / '__init__.py').write_text('', encoding='utf-8')
    (main / 'experiments' / 'train.py').write_text(
        "import os, sys\nprint('main cwd', os.getcwd(), sys.argv[1:])\nsys.exit(3 if '--fold' in sys.argv and "
        "sys.argv[sys.argv.index('--fold') + 1] == '1' else 0)\n", encoding='utf-8')
    stub = tmp_path / 'stub_python.sh'
    stub.write_text('#!/bin/bash\n[ "$2" = experiments.train ] || [ "$3" = experiments.train ] && exec "' + Path(sys.executable).as_posix()
                    + '" "$@"\necho "package $PWD $@"\n', encoding='utf-8')
    done = runs / 'kfold0_x86_gnn_s0_team-sigwhole'
    done.mkdir()
    (done / 'metrics.json').write_text('{}', encoding='utf-8')
    env = {**os.environ, 'GT_RUNS': runs.as_posix(), 'GT_PY': stub.as_posix(), 'GT_MAIN': main.as_posix()}
    r = subprocess.run([BASH, 'scripts/team_whole_runs.sh'], cwd=ROOT, capture_output=True, text=True, env=env, timeout=600)
    out = r.stdout.splitlines()
    assert 'skip kfold0_x86_gnn_s0_team-sigwhole' in out
    assert 'retry kfold1_x86_fivechannel-L512_s0_whole' in out and (runs / 'logs' / 'kfold1_x86_fivechannel-L512_s0_whole.log.try1').exists()
    log = (runs / 'logs' / 'kfold0_x86_fivechannel-L512_s0_whole.log').read_text(encoding='utf-8')
    assert log.startswith('main cwd') and Path(log.split()[2]).resolve() == main.resolve() and '--tag' in log and "'whole'" in log
    package_log = (runs / 'logs' / 'kfold1_x86_gnn_s0_team-sigwhole.log').read_text(encoding='utf-8')
    assert package_log.startswith('package') and ROOT.name in package_log
    assert r.returncode == 1 and 'of 60 runs without metrics.json' in r.stdout


# gates and stats ------------------------------------------------------------------------------------------------------------


def test_team_gates_whole_rows_and_pairs():
    arms = dict(gates.TEAM_ARMS)
    for (arm, _, v3), label in zip(WHOLE_ARMS, gates.TEAM_WHOLE):
        assert arms[label] == gates.team_group(arm) == f'kfold_x86_gnn_team-{arm}'
    assert arms['fivechannel-L512 whole'] == 'kfold_x86_fivechannel-L512_whole'
    pairs = gates.TEAM_PAIRS
    for v3, whole in zip(gates.TEAM_V3, gates.TEAM_WHOLE):
        assert (v3, whole) in pairs
    assert [b for a, b in pairs if a == 'R7 typed, whole graph' and b in gates.TEAM_WHOLE_ROWS] == list(gates.TEAM_WHOLE_ROWS)
    assert (gates.TEAM_WHOLE[1], gates.TEAM_WHOLE[2]) in pairs and len(pairs) == len(set(pairs)) == 31
    assert not set(gates.TEAM_WHOLE) & set(gates.TEAM_OF)
    assert not any(b in gates.TEAM_WHOLE_ROWS for a, b in pairs if a in ('TF-IDF', 'ours b500 typed'))
    assert gates.TEAM_WHOLE_BUILD == 'b_sigwhole_v3'


def test_fivechannel_configs_are_foreign_to_run_config_but_named_as_main_names_them(tmp_path):
    path = tmp_path / 'config.json'
    for fold, seed, extra, name, group in (
            (2, 1, {}, 'kfold2_x86_fivechannel-L512_s1_whole', 'kfold_x86_fivechannel-L512_whole'),
            (0, 0, {'tag': ''}, 'kfold0_x86_fivechannel-L512_s0', 'kfold_x86_fivechannel-L512'),
            (4, 2, {'max_instructions': 128, 'tag': ''}, 'kfold4_x86_fivechannel_s2', 'kfold_x86_fivechannel'),
            (1, 0, {'threshold_from': 'oof'}, 'kfold1_x86_fivechannel-L512-thoof_s0_whole', 'kfold_x86_fivechannel-L512-thoof_whole')):
        path.write_text(json.dumps({**FIVE, 'fold': fold, 'seed': seed, **extra}), encoding='utf-8')
        assert foreign_identity(path) == (name, group, 'kfold', fold, seed)
        assert RunConfig.try_load(path) is None
        with pytest.raises(ForeignRun):
            RunConfig.load(path)
    path.write_text(json.dumps({'protocol': 'kfold', 'arch': 'x86', 'model': 'gnn', 'fold': 0}), encoding='utf-8')
    assert foreign_identity(path) is None and RunConfig.try_load(path).model == 'gnn'


def test_existing_fivechannel_run_directories_do_not_break_the_run_readers():
    dirs = sorted(RUNS.glob('kfold*_x86_fivechannel*'))
    if not dirs or not all((d / 'config.json').exists() for d in dirs):
        pytest.skip('no fivechannel run on disk')
    for d in dirs:
        name, group, protocol, fold, seed = foreign_identity(d / 'config.json')
        assert name == d.name and RunConfig.try_load(d / 'config.json') is None
        assert d in stats.folds_by_seed(group)[seed].values()


@needs_real_data
def test_team_gates_from_synthetic_whole_runs_including_a_fivechannel_group(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    runs.mkdir()
    base = dict(protocol='kfold', arch='x86', model='gnn')
    fake_group(runs, lambda f, s: RunConfig(**base, fold=f, seed=s, mask_blocks=0.15), 0.5)                          # R7
    for arm, signal, seeds in (('sig500v3', 1.0, (0, 1, 2)), ('sig500v3-feat', 0.8, (0, 1, 2)), ('sig500v3-emb', 1.0, (0, 1, 2)),
                               ('sigwhole', 1.4, (0, 1, 2)), ('sigwhole-feat', 0.9, (0, 1, 2)), ('sigwhole-emb', 1.4, (0, 1))):
        fake_group(runs, lambda f, s, arm=arm: tp.team_config(f, s, arm), signal, seeds=seeds)
    # fivechannel: runs written through a gnn config, then given main's config.json (name and group follow it)
    fake_five = lambda f, s: RunConfig(**base, fold=f, seed=s, tag='five-scratch')                                  # noqa: E731
    fake_group(runs, fake_five, 1.2)
    for seed in (0, 1, 2):
        for fold in range(5):
            scratch = runs / fake_five(fold, seed).name
            target = runs / f'kfold{fold}_x86_fivechannel-L512_s{seed}_whole'
            scratch.rename(target)
            (target / 'config.json').write_text(json.dumps({**FIVE, 'fold': fold, 'seed': seed}), encoding='utf-8')
    monkeypatch.setattr(stats, 'RUNS', runs)
    gates.pooled.cache_clear()
    text = '\n'.join(gates.team(40))
    gates.pooled.cache_clear()
    pair = {(l.split('|')[1].strip(), l.split('|')[2].strip()): l.split('|')[-2].strip()
            for l in text.splitlines() if l.startswith('| ') and l.count('|') == 6}
    for v3, whole in zip(gates.TEAM_V3, gates.TEAM_WHOLE):
        assert pair[(v3, whole)] != 'missing'
    assert pair[('R7 typed, whole graph', 'fivechannel-L512 whole')] == 'beats'
    assert pair[('R7 typed, whole graph', 'team sigwhole')] == 'beats'
    assert 'seeds [0, 1] only' in pair[(gates.TEAM_WHOLE[1], gates.TEAM_WHOLE[2])]
    assert '| fivechannel-L512 whole | kfold_x86_fivechannel-L512_whole | ' in text and '| fivechannel-L512 whole | missing' not in text
    assert 'Match rates of `b_sigwhole_v3`: missing (no build_summary.json' in text


def test_team_gates_whole_match_rate_line(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    gates.pooled.cache_clear()
    (tmp_path / 'team').mkdir()
    build = tmp_path / 'team' / 'b_sigwhole_v3'
    fake_build(build, {'r1': fake_payload([0] * 10, 1), 'g1': fake_payload([1] * 8 + [0] * 2, 0)})
    (build / 'build_summary.json').write_text(json.dumps({'processed': 1810, 'total': 1810}), encoding='utf-8')
    text = '\n'.join(gates.team(30))
    gates.pooled.cache_clear()
    line = [l for l in text.splitlines() if l.startswith('Match rates of `b_sigwhole_v3`')]
    assert len(line) == 1 and 'ransomware 1 binaries, mean 0.000' in line[0] and 'goodware 1 binaries, mean 0.800' in line[0]
    assert [l for l in text.splitlines() if l.startswith('Match rates of `b_sig500_v3`')][0].startswith(
        'Match rates of `b_sig500_v3`: missing (no build_summary.json')                   # the v3 line is its own, unchanged
