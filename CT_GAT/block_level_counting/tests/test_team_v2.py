"""Step 21: the second pinned teammate source (main c7899f0, Yanping's streaming preprocess), `build_b --source`, the
`sig500v2` arm of train_b / team_runs.sh / gates, and the small tools compare_builds, match_rates and measure_rss."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from common.config import RunConfig
from evaluation import gates
from experiments.team_pipelines import build_b, common as tp, compare_builds, match_rates, measure_rss, train_b
from test_team_pipelines import BASH, ROOT, needs_bash, needs_real_data, needs_worktrees, pinned_team_modules  # noqa: F401


def smallest_ids(n=1):
    rows = sorted(tp.sample_rows().values(), key=lambda r: int(r['num_blocks']))
    return [r['sample_id'] for r in rows[:n]]


@needs_worktrees
@needs_real_data
def test_build_b_main_source_records_source_pins_and_stops_on_another_source(tmp_path):
    small = smallest_ids()[0]
    out = tmp_path / 'v2'
    args = ['--source', 'main-c7899f0', '--out', str(out), '--ids', small]
    summary = build_b.main(args)
    assert summary['processed'] == summary['total'] == 1 and summary['settings']['source'] == 'main-c7899f0'
    settings = json.loads((out / 'settings.json').read_text(encoding='utf-8'))
    assert settings['source'] == 'main-c7899f0' and settings['pins'] == tp.source_pins('main-c7899f0')
    assert set(settings['pins']) == {tp.STAGE + 'preprocess_full.py', tp.STAGE + 'tokenize_asm_gz.py'}
    ckpt = json.loads((out / 'checkpoint.json').read_text(encoding='utf-8'))
    assert ckpt['settings'] == {'sample': 'signal', 'num_blocks': 500, 'seed': 0, 'source': 'main-c7899f0'}
    assert train_b.read_source(out) == 'main-c7899f0'
    assert build_b.main(args)['built_this_run'] == 0                            # resumes
    with pytest.raises(SystemExit, match='was built with'):                     # the default source on a main build
        build_b.main(['--out', str(out), '--ids', small])
    payload = torch.load(out / f'{small}.pt', weights_only=False)
    assert payload['file_id'] == small and payload['instruction_matched'].dtype in (torch.bool, torch.float32, torch.int64)
    v1 = tmp_path / 'v1'
    build_b.main(['--out', str(v1), '--ids', small])
    s1 = json.loads((v1 / 'settings.json').read_text(encoding='utf-8'))
    assert s1['source'] == 'gnn-331ce0b' and s1['pins'] == tp.source_pins('gnn-331ce0b') and train_b.read_source(v1) == s1['source']
    # a build without a source in its checkpoint (the first build) is read as the default source and resumes
    ckpt1 = json.loads((v1 / 'checkpoint.json').read_text(encoding='utf-8'))
    del ckpt1['settings']['source']
    (v1 / 'checkpoint.json').write_text(json.dumps(ckpt1), encoding='utf-8')
    (v1 / 'settings.json').write_text(json.dumps({k: v for k, v in s1.items() if k not in ('source', 'pins')}), encoding='utf-8')
    assert train_b.read_source(v1) == tp.DEFAULT_SOURCE
    assert build_b.main(['--out', str(v1), '--ids', small])['built_this_run'] == 0


@needs_worktrees
@needs_real_data
def test_build_b_frozen_from_keeps_the_vocabulary_and_embedder(tmp_path):
    small = smallest_ids()[0]
    base = tmp_path / 'base'
    build_b.main(['--out', str(base), '--ids', small])
    out = tmp_path / 'frozen'
    summary = build_b.main(['--source', 'main-c7899f0', '--frozen-from', str(base), '--out', str(out), '--ids', small])
    assert summary['api_names'] == json.loads((base / 'checkpoint.json').read_text(encoding='utf-8'))['api_names']
    settings = json.loads((out / 'settings.json').read_text(encoding='utf-8'))
    assert settings['frozen_from'] == str(base) and settings['source'] == 'main-c7899f0'
    assert torch.equal(torch.load(out / 'channel_embedder.pt')['api_embedding.weight'],
                       torch.load(base / 'channel_embedder.pt')['api_embedding.weight'])
    result = compare_builds.compare(base, out, [small])
    assert result['compared'] == 1 and set(result['samples'][small]['arrays']) == {
        'block_keys', 'instruction_matched', 'ids_opcode_operands_behaviour', 'ids_api', 'block_features', 'edge_index',
        'edge_attr'}
    arrays = result['samples'][small]['arrays']
    assert arrays['block_keys']['equal'] and arrays['edge_index']['equal'] and arrays['edge_attr']['equal']   # same sampler


@needs_worktrees
@needs_real_data
def test_train_b_arms_read_the_build_of_their_source(tmp_path):
    assert train_b.ARMS[:2] == ('sig500', 'sig500v2')
    assert train_b.ARM_DIR['sig500'].name == 'b_sig500' and train_b.ARM_DIR['sig500v2'].name == 'b_sig500_v2'
    assert tp.team_config(2, 1, 'sig500v2').name == 'kfold2_x86_gnn_s1_team-sig500v2'
    assert tp.team_config(2, 1, 'sig500v2').group == gates.team_group('sig500v2') == 'kfold_x86_gnn_team-sig500v2'
    assert RunConfig(protocol='kfold', fold=0, arch='x86', model='gnn', tag='team-sig500').name == 'kfold0_x86_gnn_s0_team-sig500'
    v1, v2 = tmp_path / 'v1', tmp_path / 'v2'
    v1.mkdir()
    v2.mkdir()
    (v2 / 'settings.json').write_text(json.dumps({'source': 'main-c7899f0'}), encoding='utf-8')
    assert train_b.read_source(v1) == 'gnn-331ce0b' and train_b.read_source(v2) == 'main-c7899f0'
    with pytest.raises(SystemExit, match='built with main-c7899f0'):
        train_b.run('sig500', 0, 0, smoke=2, device='cpu', b_dir=v2, log=lambda *_: None)
    with pytest.raises(SystemExit, match='built with gnn-331ce0b'):
        train_b.run('sig500v2', 0, 0, smoke=2, device='cpu', b_dir=v1, log=lambda *_: None)
    (v2 / 'settings.json').write_text(json.dumps({'source': 'elsewhere'}), encoding='utf-8')
    with pytest.raises(SystemExit, match='unknown source'):
        train_b.read_source(v2)
    with pytest.raises(ValueError):
        train_b.run('sig500v4', 0, 0)


def fake_payload(matched, label, api=None, n_blocks=2):
    n = len(matched)
    ids = torch.zeros(n, 5, dtype=torch.long)
    ids[:, 0] = torch.arange(n) + 1
    if api is not None:
        ids[:, 3] = torch.tensor(api)
    return {'label': label, 'instruction_matched': torch.tensor(matched, dtype=torch.bool), 'instruction_ids': ids,
            'block_keys': [(0, i) for i in range(n_blocks)], 'block_features': torch.zeros(n_blocks, 3),
            'edge_index': torch.tensor([[0], [1]]), 'edge_attr': torch.zeros(1, 2)}


def fake_build(path, payloads):
    path.mkdir()
    for i, p in payloads.items():
        torch.save(p, path / f'{i}.pt')
    (path / 'checkpoint.json').write_text(json.dumps({'done': sorted(payloads)}), encoding='utf-8')


def test_compare_builds_reports_which_arrays_differ(tmp_path):
    same = fake_payload([1, 1, 0, 1], 1)
    lost = fake_payload([1, 1, 0, 0], 1)
    a = {'x': same, 'y': same, 'z': fake_payload([1, 1, 1, 1], 0, api=[0, 0, 80, 0])}
    z2 = fake_payload([1, 1, 1, 1], 0, api=[0, 0, 81, 0])
    fake_build(tmp_path / 'a', a)
    fake_build(tmp_path / 'b', {'x': same, 'y': lost, 'z': z2})
    out = compare_builds.compare(tmp_path / 'a', tmp_path / 'b', ['x', 'y', 'z'])
    assert (out['compared'], out['identical'], out['differing']) == (3, 1, 2)
    assert out['differing_arrays'] == {'instruction_matched': 1, 'ids_opcode_operands_behaviour': 0 or 1, 'ids_api': 1} or \
        out['differing_arrays'] == {'instruction_matched': 1, 'ids_api': 1}
    y = out['samples']['y']
    assert y['arrays']['instruction_matched'] == {'equal': False, 'rows_differing': 1, 'rows': 4, 'max_abs': 1.0}
    assert y['match_a'] == 0.75 and y['match_b'] == 0.5 and not y['identical']
    assert out['samples']['x']['identical'] and out['samples']['z']['arrays']['ids_api']['rows_differing'] == 1
    assert compare_builds.done_ids(tmp_path / 'a') == ['x', 'y', 'z']


def test_compare_builds_pick_takes_the_lowest_matches_and_balances_labels(tmp_path):
    payloads = {f'r{i:02d}': fake_payload([1] * (i + 1) + [0] * (20 - i), 1) for i in range(20)}
    payloads.update({f'g{i:02d}': fake_payload([1] * (i + 1) + [0] * (20 - i), 0) for i in range(20)})
    fake_build(tmp_path / 'b', payloads)
    ids, info = compare_builds.pick(tmp_path / 'b', n=12, low=4, seed=0)
    assert len(ids) == 12 and ids == sorted(ids) and info['ransomware'] == info['goodware'] == 6
    low = {r['sample_id'] for r in info['lowest']}
    assert len(low) == 4 and low <= set(ids) and all(payloads[i]['instruction_matched'].float().mean() <= 0.2 for i in low)
    assert compare_builds.pick(tmp_path / 'b', n=12, low=4, seed=0)[0] == ids       # deterministic


def test_match_rates_per_label_shares(tmp_path):
    payloads = {'r1': fake_payload([0] * 10, 1), 'r2': fake_payload([1] * 6 + [0] * 4, 1), 'r3': fake_payload([1] * 10, 1),
                'r4': fake_payload([1] * 19 + [0], 1), 'g1': fake_payload([1] * 10, 0), 'g2': fake_payload([1] * 8 + [0] * 2, 0)}
    fake_build(tmp_path / 'b', payloads)
    summary = match_rates.summarize(match_rates.sample_rates(tmp_path / 'b'))
    r, g = summary['ransomware'], summary['goodware']
    assert (r['n'], r['n_below_50'], r['n_below_90'], r['share_zero']) == (4, 1, 2, 0.25)
    assert r['mean_match'] == pytest.approx((0 + 0.6 + 1 + 0.95) / 4)
    assert (g['n'], g['n_below_50'], g['n_below_90'], g['share_below_90']) == (2, 0, 1, 0.5)
    text = match_rates.format_table(summary, 'b')
    assert '| ransomware | 4 | 0.638 | 1 (25.0%) | 2 (50.0%) | 25.0% |' in text and '| goodware | 2 | 0.900 |' in text
    fake_build(tmp_path / 'partial', {'r1': payloads['r1'], 'r4': payloads['r4'], 'g1': payloads['g1']})
    both = match_rates.main(['--build', str(tmp_path / 'b'), '--among', str(tmp_path / 'partial'), '--json', str(tmp_path / 's.json')])
    saved = json.loads((tmp_path / 's.json').read_text(encoding='utf-8'))
    assert both == 0 and (saved['ransomware']['n'], saved['goodware']['n'], saved['ransomware']['n_below_50']) == (2, 1, 1)


def test_measure_rss_reports_peak_and_returncode(tmp_path):
    out = tmp_path / 'm.json'
    code = 'import time; x = bytearray(300 * 2**20); x[::4096] = b"1" * len(x[::4096]); time.sleep(1.5)'
    assert measure_rss.main(['--out', str(out), '--interval', '0.1', '--', sys.executable, '-c', code]) == 0
    result = json.loads(out.read_text(encoding='utf-8'))
    assert result['returncode'] == 0 and result['wall_seconds'] >= 1.4 and result['peak_rss_bytes'] > 250 * 2 ** 20
    assert result['peak_rss_gb'] == round(result['peak_rss_bytes'] / 2 ** 30, 3) and result['polls'] > 3
    assert measure_rss.main(['--out', str(out), '--', sys.executable, '-c', 'raise SystemExit(3)']) == 3


def script(*args, env=None):
    r = subprocess.run([BASH, 'scripts/team_runs.sh', *args], cwd=ROOT, capture_output=True, text=True, env=env or os.environ)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


@needs_bash
@needs_real_data
def test_team_runs_v2_lane_dry_names_equal_run_config_names():
    lines = script('--v2', '--dry')
    runs = [l.split() for l in lines if l.startswith('run ')]
    assert len(runs) == 15 and lines[0].startswith('team_runs start') and lines[-1].startswith('team_runs done')
    waits = [l for l in lines if l.startswith('wait ')]
    assert len(waits) == 1 and waits[0].endswith('/team/b_sig500_v2/build_summary.json reports processed == 1810/1810, none excluded')
    seen = []
    for _, name, module, *argv in runs:
        opt = dict(zip(argv[::2], argv[1::2]))
        fold, seed = int(opt['--fold']), int(opt['--seed'])
        assert opt['--arm'] == 'sig500v2' and opt['--device'] == 'cpu' and module == 'experiments.team_pipelines.train_b'
        assert name == tp.team_config(fold, seed, 'sig500v2').name
        seen.append((seed, fold))
    assert seen == [(s, f) for s in (0, 1, 2) for f in range(5)]
    assert not any('gnn500' in l or 'cb500' in l for l in lines)
    plain = script('--dry')                                                      # without the flag nothing changed
    assert len([l for l in plain if l.startswith('run ')]) == 45 and not any('sig500v2' in l for l in plain)


@needs_bash
def test_v2_lane_waits_for_a_build_without_exclusions(tmp_path):
    runs = tmp_path / 'runs'
    (runs / 'team' / 'b_sig500_v2').mkdir(parents=True)
    env = {**os.environ, 'GT_RUNS': runs.as_posix(), 'GT_PY': Path(sys.executable).as_posix()}
    summary = runs / 'team' / 'b_sig500_v2' / 'build_summary.json'
    assert script('--v2', '--check-build', env=env) == ['build summary absent']
    for body, want in (({'processed': 1810, 'total': 1810, 'excluded': {}}, 'build complete'),
                       ({'processed': 1809, 'total': 1810, 'excluded': {'a' * 64: 'why'}}, 'build incomplete'),   # v2 excludes nothing
                       ({'processed': 1808, 'total': 1810}, 'build incomplete')):
        summary.write_text(json.dumps(body), encoding='utf-8')
        assert script('--v2', '--check-build', env=env) == [want], body
    assert script('--check-build', env=env) == ['build summary absent']          # the v1 lane looks at b_sig500


def test_team_gates_pair_sig500v2_with_sig500_r7_and_tfidf():
    assert 'team sig500v2' in dict(gates.TEAM_ARMS) and dict(gates.TEAM_ARMS)['team sig500v2'] == 'kfold_x86_gnn_team-sig500v2'
    v2 = [(a, b) for a, b in gates.TEAM_PAIRS if b == 'team sig500v2']
    assert sorted(v2) == sorted([('team sig500', 'team sig500v2'), ('R7 typed, whole graph', 'team sig500v2'),
                                 ('TF-IDF', 'team sig500v2')])
    assert 'team sig500v2' not in gates.TEAM_OF
