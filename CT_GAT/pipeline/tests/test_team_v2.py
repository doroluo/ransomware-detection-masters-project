"""Step 21: the second pinned teammate source (main c7899f0, Yanping's streaming preprocess), `build_b --source`, the
`sig500v2` arm of train_b, and the small tools compare_builds, match_rates and measure_rss. The tools run on synthetic builds
everywhere (measure_rss needs psutil and skips without it); what builds from the teammates' code needs GT_TEAM_GNN_ROOT,
GT_TEAM_MAIN_ROOT, the bundle and the IDA listings, and skips with a reason."""
import json
import sys

import pytest
import torch

from common.config import RunConfig
from experiments.stages.team import team_group
from experiments.team_pipelines import build_b, common as tp, compare_builds, match_rates, measure_rss, train_b
from test_team_pipelines import (fresh_team_caches, needs_pyg, needs_real_data, needs_worktrees, pinned_team_modules,  # noqa: F401
                                 real_cache)


def smallest_ids(n=1):
    rows = sorted(tp.sample_rows().values(), key=lambda r: int(r['num_blocks']))
    return [r['sample_id'] for r in rows[:n]]


@needs_pyg
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


@needs_pyg
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


@needs_pyg
@needs_worktrees
@needs_real_data
def test_train_b_arms_read_the_build_of_their_source(tmp_path):
    assert train_b.ARMS[:2] == ('sig500', 'sig500v2')
    assert train_b.ARM_DIR['sig500'].name == 'b_sig500' and train_b.ARM_DIR['sig500v2'].name == 'b_sig500_v2'
    assert tp.team_config(2, 1, 'sig500v2').name == 'kfold2_x86_gnn_s1_team-sig500v2'
    assert tp.team_config(2, 1, 'sig500v2').group == team_group('sig500v2') == 'kfold_x86_gnn_team-sig500v2'
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
        train_b.run('sig500v9', 0, 0)                                       # not an arm


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
    pytest.importorskip('psutil', exc_type=ImportError)
    out = tmp_path / 'm.json'
    code = 'import time; x = bytearray(300 * 2**20); x[::4096] = b"1" * len(x[::4096]); time.sleep(1.5)'
    assert measure_rss.main(['--out', str(out), '--interval', '0.1', '--', sys.executable, '-c', code]) == 0
    result = json.loads(out.read_text(encoding='utf-8'))
    assert result['returncode'] == 0 and result['wall_seconds'] >= 1.4 and result['peak_rss_bytes'] > 250 * 2 ** 20
    assert result['peak_rss_gb'] == round(result['peak_rss_bytes'] / 2 ** 30, 3) and result['polls'] > 3
    assert measure_rss.main(['--out', str(out), '--', sys.executable, '-c', 'raise SystemExit(3)']) == 3


def state_build(path, payloads, names, embedder):
    fake_build(path, payloads)
    (path / 'api_vocab.json').write_text(json.dumps(names), encoding='utf-8')
    torch.save(embedder, path / 'channel_embedder.pt')


def test_compare_builds_state_compares_embeddings_vocabulary_and_embedder_exactly(tmp_path, capsys):
    payload = fake_payload([1, 1, 0, 1], 1)
    payload['block_embeddings'] = torch.tensor([[0.25, 0.5], [1.0, -1.0]])
    near = dict(payload, block_embeddings=payload['block_embeddings'] + torch.tensor([[0.0, 0.0], [0.0, 1e-6]]))
    weights = {'api_embedding.weight': torch.arange(6.0).reshape(3, 2), 'opcode_embedding.weight': torch.ones(4, 2)}
    names = ['SetEvent', 'CloseHandle', 'ExitProcess']
    state_build(tmp_path / 'a', {'x': payload}, names, weights)
    state_build(tmp_path / 'same', {'x': dict(payload)}, list(names), {k: v.clone() for k, v in weights.items()})
    same = compare_builds.compare(tmp_path / 'a', tmp_path / 'same', ['x'], state=True)
    assert same['identical'] == 1 and same['max_block_embedding_diff'] == 0.0 and same['state']['identical']
    assert same['samples']['x']['arrays']['block_embeddings'] == {'equal': True, 'max_abs': 0.0}
    assert set(compare_builds.compare(tmp_path / 'a', tmp_path / 'same', ['x'])['samples']['x']['arrays']) == {
        'block_keys', 'instruction_matched', 'ids_opcode_operands_behaviour', 'ids_api', 'block_features', 'edge_index',
        'edge_attr'}                                                                            # without --state: the arrays of before
    other = {'api_embedding.weight': weights['api_embedding.weight'] + torch.tensor([0, 0, 0, 0, 0, 0.5]).reshape(3, 2),
             'opcode_embedding.weight': weights['opcode_embedding.weight'].clone()}
    state_build(tmp_path / 'b', {'x': near}, ['SetEvent', 'ExitProcess', 'CloseHandle'], other)
    out = compare_builds.compare(tmp_path / 'a', tmp_path / 'b', ['x'], state=True)
    assert out['identical'] == 0 and out['differing_arrays'] == {'block_embeddings': 1}
    assert out['max_block_embedding_diff'] == pytest.approx(1e-6, rel=0.1) and not out['samples']['x']['arrays']['block_embeddings']['equal']
    vocab, tensors = out['state']['api_vocab'], out['state']['channel_embedder']
    assert vocab['equal'] is False and vocab['first_difference'] == 1 and vocab['same_names_other_order'] is True
    assert tensors['api_embedding.weight'] == {'equal': False, 'max_abs': 0.5} and tensors['opcode_embedding.weight']['equal']
    assert not out['state']['identical']
    json_path = tmp_path / 'c.json'
    assert compare_builds.main(['compare', '--a', str(tmp_path / 'a'), '--b', str(tmp_path / 'b'), '--state', '--json', str(json_path)]) == 0
    printed = capsys.readouterr().out
    assert 'state: block_embeddings largest |difference|' in printed and 'api_vocab DIFFERS (3 / 3 names)' in printed
    assert 'channel_embedder 1 of 2 tensors equal' in printed
    assert json.loads(json_path.read_text(encoding='utf-8'))['state']['api_vocab']['first_difference'] == 1
    state_build(tmp_path / 'longer', {'x': payload}, names + ['Extra'], {'api_embedding.weight': torch.zeros(4, 2)})
    longer = compare_builds.compare_state(tmp_path / 'a', tmp_path / 'longer')
    assert longer['api_vocab']['first_difference'] == 3 and longer['channel_embedder']['api_embedding.weight']['equal'] is False
    assert longer['channel_embedder']['opcode_embedding.weight'] == {'equal': False, 'missing': 'b'}
    assert compare_builds.compare_state(tmp_path / 'a', tmp_path / 'nothing')['channel_embedder'] == {
        'channel_embedder.pt': {'equal': False, 'missing': 'b'}}
