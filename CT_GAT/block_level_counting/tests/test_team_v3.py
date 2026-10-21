"""Step 35: the sig500v3 arms of train_b (--features all|feat|emb, --source), the GPU lane scripts/team_v3_runs.sh and the v3 rows,
pairs and match-rate line of `gates.py --stage team`."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from common.config import RunConfig
from evaluation import gates, stats
from experiments.team_pipelines import build_b, common as tp, train_b
from test_team_pipelines import BASH, ROOT, fake_group, needs_bash, needs_real_data, needs_worktrees, pinned_team_modules  # noqa: F401
from test_team_v2 import fake_build, fake_payload

V3_ARMS = (('sig500v3', 'all'), ('sig500v3-feat', 'feat'), ('sig500v3-emb', 'emb'))


def test_v3_arms_names_sources_and_feature_modes():
    assert train_b.ARMS == ('sig500', 'sig500v2', 'sig500v3', 'sig500v3-feat', 'sig500v3-emb')
    for arm, features in V3_ARMS:
        assert train_b.ARM_DIR[arm].name == 'b_sig500_v3' and train_b.ARM_SOURCE[arm] == 'main-fix'
        assert train_b.ARM_FEATURES[arm] == features
        cfg = tp.team_config(2, 1, arm)
        assert cfg.name == f'kfold2_x86_gnn_s1_team-{arm}' and cfg.group == gates.team_group(arm)
    assert [train_b.ARM_FEATURES[a] for a in ('sig500', 'sig500v2')] == ['all', 'all']       # the earlier arms are unchanged
    assert train_b.ARM_DIR['sig500'].name == 'b_sig500' and train_b.ARM_SOURCE['sig500v2'] == 'main-c7899f0'
    assert train_b.FEATURE_MODE == {'all': 'both', 'feat': 'features', 'emb': 'embeddings'}


def test_train_b_refuses_a_contradicting_features_value_or_source_or_build(tmp_path):
    quiet = lambda *_: None                                                                   # noqa: E731
    with pytest.raises(SystemExit, match='trains on --features feat, not all'):
        train_b.run('sig500v3-feat', 0, 0, smoke=2, device='cpu', features='all', log=quiet)
    with pytest.raises(SystemExit, match='trains on --features all, not emb'):
        train_b.run('sig500v3', 0, 0, smoke=2, device='cpu', features='emb', log=quiet)
    with pytest.raises(SystemExit, match='trains on --features all, not feat'):
        train_b.run('sig500', 0, 0, smoke=2, device='cpu', features='feat', log=quiet)
    with pytest.raises(ValueError):
        train_b.run('sig500v3', 0, 0, smoke=2, device='cpu', features='both', log=quiet)
    with pytest.raises(SystemExit, match='not --source gnn-331ce0b'):
        train_b.run('sig500v3', 0, 0, smoke=2, device='cpu', source='gnn-331ce0b', log=quiet)
    for built_with in ('gnn-331ce0b', 'main-c7899f0'):
        d = tmp_path / built_with
        d.mkdir()
        (d / 'settings.json').write_text(json.dumps({'source': built_with} if built_with != 'gnn-331ce0b' else {}), encoding='utf-8')
        for arm, _ in V3_ARMS:
            with pytest.raises(SystemExit, match=f'was built with {built_with}'):
                train_b.run(arm, 0, 0, smoke=2, device='cpu', b_dir=d, source='main-fix', log=quiet)
    with pytest.raises(SystemExit, match='built with main-fix'):                              # and the older arms refuse a v3 build
        v3 = tmp_path / 'v3'
        v3.mkdir()
        (v3 / 'settings.json').write_text(json.dumps({'source': 'main-fix'}), encoding='utf-8')
        train_b.run('sig500v2', 0, 0, smoke=2, device='cpu', b_dir=v3, log=quiet)
    with pytest.raises(SystemExit):
        train_b.main(['--fold', '0', '--arm', 'sig500v3', '--features', 'feat', '--smoke', '2', '--device', 'cpu'])


@pytest.fixture(scope='module')
def mini_v3(tmp_path_factory):
    """A 10-binary main-fix build of the smallest fold-0 training binaries (5 ransomware, 5 goodware)."""
    rows = sorted(tp.fold_manifest(0, 0).train, key=lambda r: int(tp.sample_rows()[r['sample_id']]['num_blocks']))
    ids = [r['sample_id'] for r in rows if r['label'] == 1][:5] + [r['sample_id'] for r in rows if r['label'] == 0][:5]
    out = tmp_path_factory.mktemp('mini') / 'v3'
    summary = build_b.main(['--source', 'main-fix', '--out', str(out), '--ids', *ids])
    assert summary['processed'] == summary['total'] == 10
    return out, ids


@needs_worktrees
@needs_real_data
def test_feature_modes_slice_the_141_d_input_and_keep_the_edges(mini_v3, pinned_team_modules):
    out, ids = mini_v3
    assert train_b.read_source(out) == 'main-fix'
    pp, embedder, _ = train_b.build_state(out)
    gl = train_b.import_team('graph_full_loader')
    for sample_id in ids[:3] + ids[-2:]:
        payload = train_b.load_slim(out, sample_id)
        api = payload['instruction_ids'][:, 3]
        real = torch.unique(api[api >= pp.tok.FIRST_API_ID])
        for seen in (None, real[: len(real) // 2], torch.zeros(0, dtype=torch.long)):         # stored ids, half the ids, none
            g = {m: train_b.make_graph(pp, gl, embedder, payload, seen, m) for m in train_b.FEATURES}
            assert (g['all'].x.shape[1], g['feat'].x.shape[1], g['emb'].x.shape[1]) == (141, 77, 64)
            assert torch.equal(g['feat'].x, g['all'].x[:, 64:]) and torch.equal(g['emb'].x, g['all'].x[:, :64])
            for m in ('feat', 'emb'):
                assert torch.equal(g[m].edge_index, g['all'].edge_index) and torch.equal(g[m].edge_attr, g['all'].edge_attr)
                assert torch.equal(g[m].edge_type, g['all'].edge_type) and g[m].y == g['all'].y and g[m].edge_attr.shape[1] == 8
    assert train_b.check_repool(pp, embedder, payload) <= 1e-5                               # the re-pool check is in force


@needs_worktrees
@needs_real_data
@pytest.mark.parametrize('arm,features,in_dim', [('sig500v3', 'all', 141), ('sig500v3-feat', 'feat', 77), ('sig500v3-emb', 'emb', 64)])
def test_v3_arm_smoke_on_a_mini_build_writes_no_run_dir(mini_v3, tmp_path, monkeypatch, arm, features, in_dim):
    from experiments.team_pipelines import fit
    monkeypatch.setattr(fit, 'RUNS', tmp_path)
    monkeypatch.setattr(fit, 'write_run', lambda *a, **k: pytest.fail('a smoke run must not call write_run'))
    lines = []
    out = train_b.run(arm, 0, 0, smoke=4, device='cpu', b_dir=mini_v3[0], source='main-fix', features=features, log=lines.append)
    assert any(f'features {features}, in_dim {in_dim}, edge_dim 8' in l for l in lines)
    assert any(l.startswith('re-pool check') for l in lines) and len(out['history']) == 2
    assert not any(tmp_path.iterdir())
    import shutil
    shutil.rmtree(out['dir'])


# the lane ---------------------------------------------------------------------------------------------------------------


def lane(*args, env=None):
    r = subprocess.run([BASH, 'scripts/team_v3_runs.sh', *args], cwd=ROOT, capture_output=True, text=True, env=env or os.environ)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


@needs_bash
@needs_real_data
def test_v3_lane_dry_lists_45_runs_named_like_run_config():
    lines = lane('--dry')
    runs = [l.split() for l in lines if l.startswith('run ')]
    assert len(runs) == 45 and lines[0].startswith('team_v3_runs start') and lines[-1].startswith('team_v3_runs done')
    assert [l for l in lines if l.startswith('wait ')] == [
        l for l in lines if l.endswith('/team/b_sig500_v3/build_summary.json reports processed == 1810/1810, none excluded')]
    assert len([l for l in lines if l.startswith('wait ')]) == 1 and 'cpu gates.py --stage team > RUNS/gates_team.md' in lines
    seen = []
    for _, name, module, *argv in runs:
        opt = dict(zip(argv[::2], argv[1::2]))
        fold, seed, arm = int(opt['--fold']), int(opt['--seed']), opt['--arm']
        assert module == 'experiments.team_pipelines.train_b' and opt['--device'] == 'cuda' and opt['--source'] == 'main-fix'
        assert opt['--features'] == train_b.ARM_FEATURES[arm]
        assert name == tp.team_config(fold, seed, arm).name == RunConfig(
            protocol='kfold', fold=fold, arch='x86', seed=seed, model='gnn', tag=f'team-{arm}').name
        seen.append((seed, arm, fold))
    assert len(set(n for _, n, *_ in runs)) == 45
    assert seen == [(s, a, f) for s in (0, 1, 2) for a, _ in V3_ARMS for f in range(5)]


@needs_bash
def test_v3_lane_waits_for_the_complete_build_without_exclusions(tmp_path):
    runs = tmp_path / 'runs'
    (runs / 'team' / 'b_sig500_v3').mkdir(parents=True)
    env = {**os.environ, 'GT_RUNS': runs.as_posix(), 'GT_PY': Path(sys.executable).as_posix()}
    summary = runs / 'team' / 'b_sig500_v3' / 'build_summary.json'
    assert lane('--check-build', env=env) == ['build summary absent']
    for body, want in (({'processed': 1810, 'total': 1810, 'excluded': {}}, 'build complete'),
                       ({'processed': 1810, 'total': 1810}, 'build complete'),
                       ({'processed': 1809, 'total': 1810, 'excluded': {'a' * 64: 'why'}}, 'build incomplete'),
                       ({'processed': 1808, 'total': 1810, 'excluded': {}}, 'build incomplete'),
                       ({'processed': 1810, 'total': 1810, 'excluded': {'a' * 64: 'why'}}, 'build incomplete')):
        summary.write_text(json.dumps(body), encoding='utf-8')
        assert lane('--check-build', env=env) == [want], body
    r = subprocess.run([BASH, 'scripts/team_v3_runs.sh'], cwd=ROOT, capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 1 and 'not running' in r.stdout and not any(l.startswith('start ') for l in r.stdout.splitlines())
    assert not list(runs.glob('kfold*'))                                                      # an incomplete build runs nothing


@needs_bash
def test_v3_lane_stops_on_the_stop_file_while_waiting(tmp_path):
    runs = tmp_path / 'runs'
    (runs / 'team' / 'b_sig500_v3').mkdir(parents=True)
    (runs / 'STOP').write_text('', encoding='utf-8')
    env = {**os.environ, 'GT_RUNS': runs.as_posix(), 'GT_PY': Path(sys.executable).as_posix()}
    r = subprocess.run([BASH, 'scripts/team_v3_runs.sh'], cwd=ROOT, capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0 and 'stopped while waiting for the build' in r.stdout.splitlines()
    assert not any(l.startswith('start ') for l in r.stdout.splitlines()) and not list(runs.glob('kfold*'))


# gates ---------------------------------------------------------------------------------------------------------------------


def test_team_gates_v3_rows_and_pairs():
    arms = dict(gates.TEAM_ARMS)
    assert arms['team sig500v3'] == 'kfold_x86_gnn_team-sig500v3'
    assert arms['team sig500v3 feat (Model 2)'] == 'kfold_x86_gnn_team-sig500v3-feat'
    assert arms['team sig500v3 emb (Model 3)'] == 'kfold_x86_gnn_team-sig500v3-emb'
    v3, feat, emb = gates.TEAM_V3
    pairs = gates.TEAM_PAIRS
    for ref in ('R7 typed, whole graph', 'ours b500 no-edges'):
        assert [b for a, b in pairs if a == ref and b in gates.TEAM_V3] == [v3, feat, emb]
    assert [a for a, b in pairs if b == v3] == ['team sig500v2', 'team sig500', 'R7 typed, whole graph', 'ours b500 no-edges']
    assert (feat, emb) in pairs and not any(b in gates.TEAM_V3 for a, b in pairs if a in ('TF-IDF', 'ours b500 typed'))
    assert not set(gates.TEAM_V3) & set(gates.TEAM_OF) and len(pairs) == len(set(pairs))


def test_team_gates_v3_match_rate_line(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    gates.pooled.cache_clear()
    absent = '\n'.join(gates.team(30))
    assert 'Match rates of `b_sig500_v3`: missing (no build_summary.json' in absent
    (tmp_path / 'team').mkdir()
    build = tmp_path / 'team' / 'b_sig500_v3'
    fake_build(build, {'r1': fake_payload([0] * 10, 1), 'r2': fake_payload([1] * 10, 1), 'g1': fake_payload([1] * 8 + [0] * 2, 0)})
    (build / 'build_summary.json').write_text(json.dumps({'processed': 1809, 'total': 1810}), encoding='utf-8')
    assert 'missing (build incomplete: 1809 of 1810' in '\n'.join(gates.team(30))
    (build / 'build_summary.json').write_text(json.dumps({'processed': 1810, 'total': 1810}), encoding='utf-8')
    text = '\n'.join(gates.team(30))
    gates.pooled.cache_clear()
    line = [l for l in text.splitlines() if l.startswith('Match rates of `b_sig500_v3`')]
    assert len(line) == 1
    assert 'ransomware 2 binaries, mean 0.500, below 50% 1 (50.0%), below 90% 1 (50.0%), exactly 0 50.0%' in line[0]
    assert 'goodware 1 binaries, mean 0.800, below 50% 0 (0.0%), below 90% 1 (100.0%), exactly 0 0.0%' in line[0]
    for name in gates.TEAM_V3:
        assert f'| {name} | kfold_x86_gnn_team-' in text and f'| {name} | missing' in text      # missing arms print as missing
    assert f'| {gates.TEAM_V3[1]} | {gates.TEAM_V3[2]} | missing | missing | missing |' in text


def test_team_gates_from_synthetic_v3_runs(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    runs.mkdir()
    base = dict(protocol='kfold', arch='x86', model='gnn')
    cap = dict(mask_blocks=0.15, subgraph='cap', budget=500, tag='b500')
    fake_group(runs, lambda f, s: RunConfig(**base, fold=f, seed=s, mask_blocks=0.15), 0.5)                          # R7
    fake_group(runs, lambda f, s: RunConfig(**base, fold=f, seed=s, control='no_edges', **cap), 0.4)                 # b500 no-edges
    fake_group(runs, lambda f, s: tp.team_config(f, s, 'sig500'), 0.7, seeds=(0, 1))
    fake_group(runs, lambda f, s: tp.team_config(f, s, 'sig500v2'), 0.9)
    fake_group(runs, lambda f, s: tp.team_config(f, s, 'sig500v3'), 1.2)
    fake_group(runs, lambda f, s: tp.team_config(f, s, 'sig500v3-feat'), 0.8)
    fake_group(runs, lambda f, s: tp.team_config(f, s, 'sig500v3-emb'), 1.0, seeds=(0, 1))
    monkeypatch.setattr(stats, 'RUNS', runs)
    gates.pooled.cache_clear()
    text = '\n'.join(gates.team(40))
    gates.pooled.cache_clear()
    pair = {(l.split('|')[1].strip(), l.split('|')[2].strip()): l.split('|')[-2].strip()
            for l in text.splitlines() if l.startswith('| ') and l.count('|') == 6}
    v3, feat, emb = gates.TEAM_V3
    for a, b in ((x, y) for x, y in gates.TEAM_PAIRS if y in gates.TEAM_V3 or x in gates.TEAM_V3):
        if a in ('TF-IDF',):
            continue
        assert pair[(a, b)] != 'missing', (a, b)
    assert 'seeds [0, 1] only' in pair[('team sig500', v3)] and pair[('team sig500v2', v3)] in ('beats', 'does not beat')
    assert 'seeds [0, 1] only' in pair[(feat, emb)] and 'seeds [0, 1] only' in pair[('R7 typed, whole graph', emb)]
    assert pair[('R7 typed, whole graph', v3)] == 'beats'
