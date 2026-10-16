"""Static fixtures only: no malware execution, IDA license or GPU required."""
import csv
import gzip
import json

import numpy as np
import pytest

from experiments.side_arrays import build_side
from evaluation import report
from evaluation import stats
from experiments.side_arrays import wl_baseline as wl
from common.config import RunConfig
from experiments.side_arrays.side_common import sample_rng, shuffle_edges, tag_control
from experiments.side_arrays.side_defuse import defuse_edges, semantics
from experiments.side_arrays.side_tags import TAG_NAMES, api_tag, function_tags


def ins(m, *registers, call=False):
    return dict(mnemonic=m, operands=[dict(category='register', register=r) if r else
                                     dict(category='immediate', immediate='0x1') for r in registers], is_call=call)


def blocks(*instructions):
    return [dict(instructions=list(seq)) for seq in instructions]


def pairs(bs, edges):
    return set(map(tuple, defuse_edges(bs, edges)[0].tolist()))


def test_defuse_diamond_and_kill():
    bs = blocks([ins('mov', 'eax', '')], [ins('mov', 'eax', '')], [ins('nop')], [ins('test', 'eax', 'eax')])
    assert pairs(bs, [(0, 1), (0, 2), (1, 3), (2, 3)]) == {(0, 3), (1, 3)}
    assert pairs(bs, [(0, 1), (1, 3)]) == {(1, 3)}


def test_defuse_loop_fixed_point_and_block_local_uses():
    bs = blocks([ins('mov', 'eax', '')], [ins('test', 'eax', 'eax')], [ins('add', 'eax', '')])
    assert pairs(bs, [(0, 1), (1, 2), (2, 1)]) == {(0, 1), (0, 2), (2, 1)}
    bs[1]['instructions'] = [ins('mov', 'eax', ''), ins('test', 'eax', 'eax')]
    assert pairs(bs, [(0, 1), (1, 2)]) == {(1, 2)}


def test_defuse_alias_weak_write_zero_idiom_and_barriers():
    bs = blocks([ins('mov', 'rax', '')], [ins('mov', 'al', '')], [ins('test', 'eax', 'eax')])
    assert pairs(bs, [(0, 1), (1, 2)]) == {(0, 2), (1, 2)}
    bs[1]['instructions'] = [ins('xor', 'eax', 'eax')]
    assert pairs(bs, [(0, 1), (1, 2)]) == {(1, 2)}
    for barrier in (ins('call', call=True), ins('mystery'), ins('add', 'xmm0', 'xmm1')):
        bs[1]['instructions'] = [barrier]
        assert pairs(bs, [(0, 1), (1, 2)]) == set()
    # A use before a later barrier in the same block still has its reaching edge.
    bs[1]['instructions'] = [ins('test', 'eax', 'eax'), ins('call', call=True)]
    assert pairs(bs, [(0, 1), (1, 2)]) == {(0, 1)}


def test_defuse_budget_refuses_incomplete_result():
    with pytest.raises(RuntimeError, match='budget'):
        defuse_edges(blocks([ins('nop')]), [], max_steps=0)
    with pytest.raises(RuntimeError, match='budget'):
        defuse_edges(blocks([ins('mov', 'eax', '')], [ins('test', 'eax', 'eax')]), [(0, 1)], max_edges=0)


def test_shift_weak_update_and_same_family_different_register_not_zero():
    bs = blocks([ins('mov', 'eax', '')], [ins('shl', 'eax', 'cl')], [ins('test', 'eax', 'eax')])
    assert pairs(bs, [(0, 1), (1, 2)]) == {(0, 1), (0, 2), (1, 2)}
    assert semantics(ins('xor', 'al', 'ah'))[0] == {0}


def test_no_cross_function_and_kind_degree_preservation():
    edges = np.array([[0, 1], [0, 2], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3], [5, 4]], dtype=np.int32)
    kind = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    result = shuffle_edges(edges, [0, 3, 3, 6], sample_rng('x', 7), kind)
    np.testing.assert_array_equal(result, shuffle_edges(edges, [0, 3, 3, 6], sample_rng('x', 7), kind))
    for k in (0, 1):
        for axis in (0, 1):
            np.testing.assert_array_equal(np.bincount(edges[kind == k, axis], minlength=6),
                                          np.bincount(result[kind == k, axis], minlength=6))
    assert np.all((result[:, 0] < 3) == (result[:, 1] < 3))
    with pytest.raises(ValueError, match='inter-function'):
        shuffle_edges([[0, 4]], [0, 3, 6], sample_rng('x'))


def test_tags_api_normalization_and_controls():
    assert api_tag('KERNEL32.dll!__imp__ReadFile@20') == TAG_NAMES.index('file_read')
    assert api_tag('kernel32.dll!CreateFileW') == TAG_NAMES.index('file_open')
    assert api_tag('custom!NotReadFile') is None
    z = dict(fn_flags=np.zeros(3), fn_blk_ptr=np.array([0, 1, 1, 2]),
             blk_api_ptr=np.array([0, 1, 2]), api=np.array([0, 1]), call_edges=np.array([[0, 2]]))
    tags, callee = function_tags(z, ['kernel32!ReadFile', 'bcrypt!BCryptEncrypt'])
    assert tags.shape == (3, len(TAG_NAMES)) and not tags[1].any()
    assert tags[0, TAG_NAMES.index('file_read')] == 1
    assert callee[0, TAG_NAMES.index('crypto_transform')] == 1
    assert not tag_control(tags, 'off').any()
    np.testing.assert_array_equal(tags.sum(0), tag_control(tags, 'permute', sample_rng('x')).sum(0))


def fixture(tmp_path):
    cache = tmp_path / 'cache'
    (cache / 'samples').mkdir(parents=True)
    sid = 'a' * 64
    vocab = {'mn': ['mov', 'test'], 'sig': ['mov r,i', 'test r,r'], 'api': ['kernel32!ReadFile']}
    (cache / 'vocab.json').write_text(json.dumps(vocab))
    z = dict(cache_version=2, ids_kind='global', fn_flags=np.array([0, 2], dtype=np.int8),
             fn_blk_ptr=np.array([0, 2, 2]), blk_rva=np.array([[10, 20], [20, 30]]),
             blk_insn_ptr=np.array([0, 1, 2]), blk_api_ptr=np.array([0, 1, 1]),
             mn=np.array([0, 1]), sig=np.array([0, 1]), api=np.array([0]),
             cfg_edges=np.array([[0, 1]], dtype=np.int32), call_edges=np.empty((0, 2), dtype=np.int32))
    np.savez(cache / 'samples' / f'{sid}.npz', **z)
    source = tmp_path / 'export.jsonl.gz'
    # Deliberately reverse export block list: cache sorts by start_rva, not block id.
    records = [dict(record='header', sha256=sid, schema='ida-cfg-research/2.1', analysis_options={'processor': 'metapc'}),
               dict(record='function', library=False, thunk=False, blocks=[
                   dict(id=9, start_rva=20, end_rva=30, instructions=[ins('test', 'eax', 'eax')]),
                   dict(id=7, start_rva=10, end_rva=20, instructions=[ins('mov', 'eax', '')])], edges=[[7, 9]]),
               dict(record='function', library=False, thunk=True, blocks=[], edges=[]),
               dict(record='footer', complete=True)]
    with gzip.open(source, 'wt') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    return cache, sid, source, vocab, z


def test_side_alignment_atomic_provenance_and_read_only_cache(tmp_path):
    cache, sid, source, vocab, z = fixture(tmp_path)
    path = cache / 'samples' / f'{sid}.npz'
    before = build_side.sha256(path)
    result = build_side.build_one(cache, sid, source, vocab)
    assert result['status'] == 'ok' and build_side.sha256(path) == before
    with np.load(cache / 'side' / f'{sid}.npz', allow_pickle=False) as out:
        np.testing.assert_array_equal(out['df_edges'], [[0, 1]])
        assert out['fn_tags'].shape == (2, len(TAG_NAMES))
        assert not out['fn_tags'][1].any()
        assert not any(out[k].dtype.hasobject for k in out.files)
        assert 'rax' not in str(out['provenance_json'])
    assert build_side.build_one(cache, sid, source, vocab)['status'] == 'skipped'
    with pytest.raises(ValueError, match='stale'):
        build_side.build_one(cache, sid, source, vocab, seed=9)


@pytest.mark.parametrize('field,value', [('blk_rva', np.array([[11, 20], [20, 30]])),
                                       ('fn_flags', np.array([1, 2])), ('mn', np.array([1, 0])),
                                       ('cfg_edges', np.array([[1, 0]]))])
def test_side_rejects_alignment_errors(tmp_path, field, value):
    cache, sid, source, vocab, z = fixture(tmp_path)
    z[field] = value
    with pytest.raises(ValueError, match='mismatch'):
        build_side.encode_side(z, source, sid, vocab)


def test_wl_isomorphism_direction_and_h0():
    labels = ['a', 'b', 'c']
    edges = np.array([[0, 1], [1, 2]])
    permutation = np.array([2, 0, 1])
    moved = np.empty(3, dtype=object)
    moved[permutation] = labels
    assert wl.wl_counts(labels, edges) == wl.wl_counts(moved, permutation[edges])
    assert wl.wl_counts(labels, edges) != wl.wl_counts(labels, edges[:, ::-1])
    assert wl.wl_counts(labels, edges, 0) == wl.wl_counts(labels, edges[:, ::-1], 0)
    assert sum(wl.wl_counts(labels, edges).values()) == 9


def test_wl_cache_vocab_ids_and_rvas_not_features(tmp_path):
    _, _, _, vocab, z = fixture(tmp_path)
    before = wl.block_labels(z, vocab)
    z['blk_rva'] += 100000
    z['fn_flags'][:] = 3
    z['sig'] = 1 - z['sig']
    vocab['sig'].reverse()
    assert wl.block_labels(z, vocab) == before


def test_wl_train_only_min_df():
    import scipy.sparse as sp
    xtr = sp.csr_matrix([[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0]])
    xva = sp.csr_matrix([[0, 1, 999]])
    xte = sp.csr_matrix([[1, 0, 100]])
    pva, pte, count = wl.fit_predict(xtr, xva, xte, np.array([0, 0, 1, 1]), RunConfig(), ['x86'] * 4)
    assert count == 2
    xva[0, 2] = 1
    other = wl.fit_predict(xtr, xva, xte, np.array([0, 0, 1, 1]), RunConfig(), ['x86'] * 4)
    np.testing.assert_array_equal(pva, other[0])


def test_wl_streamed_hashes_match_whole_graph(tmp_path):
    from sklearn.feature_extraction import FeatureHasher
    _, sid, _, vocab, z = fixture(tmp_path)
    # Two non-empty functions and an empty middle function.
    z['fn_blk_ptr'] = np.array([0, 1, 1, 2])
    z['cfg_edges'] = np.array([[0, 0], [1, 1]])
    got, diagnostics = wl.sample_features(z, vocab, 0, sid)
    labels = wl.block_labels(z, vocab)
    hasher = FeatureHasher(n_features=wl.N_FEATURES, input_type='dict', alternate_sign=False)
    for method, height in zip(wl.METHODS, (0, 2, 2)):
        want = hasher.transform([wl.wl_counts(labels, z['cfg_edges'], height)])
        assert (got[method] != want).nnz == 0
    assert diagnostics['blocks'] == 2


def test_wl_five_fold_harness_report_and_stats(tmp_path, monkeypatch):
    cache, _, _, vocab, z = fixture(tmp_path)
    rows = []
    for i in range(40):
        label = i % 2
        sid = f'{i:064x}'
        item = dict(z)
        item['sig'] = np.array([label, label])
        np.savez(cache / 'samples' / f'{sid}.npz', **item)
        rows.append(dict(sample_id=sid, label=label, arch='x86', family=f'family{i // 2}' if label else '',
                         identity_group=sid, fold=i // 2 % 5, fold_v3=i // 2 % 5, arch_matched=1,
                         status='ok', split='train'))
    with (cache / 'index.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    runs = tmp_path / 'runs'
    monkeypatch.setattr(report, 'RUNS', runs)
    monkeypatch.setattr(stats, 'RUNS', runs)
    for fold in range(5):
        out = wl.run(RunConfig(protocol='kfold', fold=fold), cache, runs / 'baselines_wl')
        m = json.loads((out / 'metrics.json').read_text())
        assert set(m['methods']) == set(wl.METHODS)
        assert all('threshold_val' in method for method in m['methods'].values())
        with (out / 'predictions.csv').open() as f:
            assert {r['split'] for r in csv.DictReader(f)} == {'val', 'test'}
    groups, left = report.baseline_runs(auc=True)
    assert len(groups) == 3 and not left
    result, info = stats.compare('wl0', 'wl2-shuffle', selection='auc', resamples=20)
    assert info['n'] == 40 and info['seeds'] == [0]
    assert set(result) == {'auc', 'pr_auc', 'macro_f1'}
    assert stats.baseline_method('wl2-shuffle-tf50-ab') == ('wl_h2_shuffle_lr', '_tf50_ab')
    assert stats.baseline_scopes('wl2', 'kfold_v3_all_gnn')['wl2'] == ('kfold_v3', 'all')
