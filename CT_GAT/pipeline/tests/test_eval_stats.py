"""python -m pytest tests/test_eval_stats.py: the paired cluster bootstrap of `s04_evaluation.stats` (weighted metrics against
sklearn, self-comparison, swap, cell-balanced weights), the group names of runs and baselines, the pooled predictions read from
synthetic run directories, and the "excluded from both arms" line."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from common.config import RunConfig
from common.manifests import load_index, make_manifest
from common.run_io import write_run
from eval_synth import (arch_preds, build_fake, fake_fold_rows, preds, resignal, val_threshold_of, write_baseline,  # noqa: F401
                        write_v3_baseline, write_v3_run)
from eval_synth import evaltiny  # noqa: F401  (a fixture)
from s04_evaluation import report, stats
from s04_evaluation.metrics import cell_balanced
from tiny_cache import tiny  # noqa: F401  (a fixture)

ROOT = Path(__file__).resolve().parents[1]


def test_weighted_metrics_match_sklearn_on_repeats():
    p = preds()
    w = np.random.default_rng(1).integers(0, 4, len(p.y)).astype(float)
    order, grp = stats.ranking(p)
    got = stats.metrics_weighted(p, order, grp, w)
    rep = np.repeat(np.arange(len(p.y)), w.astype(int))
    want = (roc_auc_score(p.y[rep], p.prob[rep]), average_precision_score(p.y[rep], p.prob[rep]),
            f1_score(p.y[rep], p.pred[rep], average='macro'))
    assert got == pytest.approx(want, abs=1e-12)
    ones = stats.metrics_weighted(p, order, grp, np.ones(len(p.y)))
    assert ones == pytest.approx((roc_auc_score(p.y, p.prob), average_precision_score(p.y, p.prob),
                                 f1_score(p.y, p.pred, average='macro')), abs=1e-12)


def test_bootstrap_self_comparison_is_zero_and_swap_flips_sign():
    a = {0: preds(seed=0), 1: preds(seed=1)}
    b = {s: resignal(p, 1.5, s) for s, p in a.items()}
    same, _ = stats.paired_bootstrap(a, a, resamples=200)
    for key, _ in stats.METRICS:
        assert same[key]['diff'] == 0 and same[key]['lo'] == 0 and same[key]['hi'] == 0
    ab, _ = stats.paired_bootstrap(a, b, resamples=300)
    ba, _ = stats.paired_bootstrap(b, a, resamples=300)
    for key, _ in stats.METRICS:
        assert ab[key]['diff'] == pytest.approx(-ba[key]['diff'])
        assert ab[key]['lo'] == pytest.approx(-ba[key]['hi']) and ab[key]['hi'] == pytest.approx(-ba[key]['lo'])
    assert ab['auc']['diff'] > 0 and ab['auc']['lo'] < ab['auc']['hi']


def test_baseline_group_in_stats(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    for fold in range(5):
        write_baseline(tmp_path, fold, 1.2)
    f1, _ = stats.load_group('tfidf', 'f1')
    auc, _ = stats.load_group('tfidf', 'auc')
    assert len(f1[0].ids) == len(auc[0].ids) == 600 and 'v' not in ''.join(f1[0].ids[:3])
    assert not np.array_equal(f1[0].pred, auc[0].pred)
    assert np.array_equal(auc[0].pred, (auc[0].prob >= 0.4).astype(int))
    build_fake(tmp_path, {(('control', 'no_edges'),): (1.2, range(5))})
    result, _ = stats.compare('tfidf', 'kfold_x86_gnn-noedges', 'auc', 50)
    assert set(result) == {'auc', 'pr_auc', 'macro_f1'}


def test_train_fraction_baseline_groups(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    monkeypatch.setattr(report, 'RUNS', tmp_path)
    for fold in range(5):
        write_baseline(tmp_path, fold, 1.2)
        write_baseline(tmp_path, fold, 0.9, tf=25)
    assert stats.baseline_method('tfidf') == ('tfidf_sig_api_lr', '')
    assert stats.baseline_method('structural-tf50') == ('structural_lr', '_tf50')
    assert stats.baseline_method('kfold_x86_gnn-tf50') is None and stats.baseline_method('tfidf-tfx') is None
    full, _ = stats.load_group('tfidf', 'auc')
    quarter, _ = stats.load_group('tfidf-tf25', 'auc')
    assert list(full) == list(quarter) == [0] and len(full[0].ids) == len(quarter[0].ids) == 600
    assert not np.array_equal(full[0].prob, quarter[0].prob)
    assert stats.load_group('tfidf-tf50', 'auc') == ({}, {})
    assert {k[2] for k in report.baseline_runs()[0]} == {'tfidf_sig_api_lr', 'tfidf_sig_api_lr tf25'}


def test_v3_groups_and_baseline_scopes(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    assert stats.group_scope('kfold_v3_matched_gnn-noedges') == ('kfold_v3', 'matched')
    assert stats.group_scope('kfold_x86_gnn') == ('kfold', 'x86') and stats.group_scope('tfidf') is None
    assert stats.baseline_scopes('tfidf', 'kfold_v3_all_gnn') == {'tfidf': ('kfold_v3', 'all')}
    assert stats.baseline_scopes('kfold_v3_matched_gnn', 'structural') == {'structural': ('kfold_v3', 'matched')}
    assert stats.baseline_scopes('tfidf', 'kfold_x86_gnn') == {'tfidf': ('kfold', 'x86')}
    assert stats.baseline_scopes('tfidf', 'structural') == {'tfidf': ('kfold', 'x86'), 'structural': ('kfold', 'x86')}
    assert stats.baseline_scopes('tfidf', 'kfold_x86_gnn', 'kfold_v3', 'all') == {'tfidf': ('kfold_v3', 'all')}
    for arch, signal in (('all', 1.3), ('matched', 0.6)):
        for fold in range(5):
            write_v3_baseline(tmp_path, arch, fold, signal)
    for fold in range(5):
        write_baseline(tmp_path, fold, 1.0)
    dirs = {a: stats.folds_by_seed('tfidf', ('kfold_v3', a))[0] for a in ('all', 'matched')}
    assert dirs['all'][2].name == 'v3kfold2_all_s0' and dirs['matched'][2].name == 'v3kfold2_matched_s0'
    assert stats.folds_by_seed('tfidf')[0][2].name == 'kfold2_x86_s0'
    assert stats.folds_by_seed('tfidf', ('kfold', 'all')) == {}
    all_, _ = stats.load_group('tfidf', 'auc', ('kfold_v3', 'all'))
    matched, _ = stats.load_group('tfidf', 'auc', ('kfold_v3', 'matched'))
    assert len(all_[0].ids) == len(matched[0].ids) == 600 and not np.array_equal(all_[0].prob, matched[0].prob)
    assert set(all_[0].arch) == {'x86', 'x64'}
    for fold in range(5):
        write_v3_run(tmp_path, 'all', fold, 1.0)
    auto, _ = stats.compare('tfidf', 'kfold_v3_all_gnn', 'auc', 50)
    forced, _ = stats.compare('tfidf', 'kfold_v3_all_gnn', 'auc', 50, protocol='kfold_v3', arch='all')
    assert auto == forced
    other, _ = stats.compare('tfidf', 'kfold_v3_all_gnn', 'auc', 50, protocol='kfold_v3', arch='matched')
    assert other['auc']['a'] != auto['auc']['a']


def test_stats_cli_has_protocol_and_arch():
    r = subprocess.run([sys.executable, '-m', 's04_evaluation.stats', '--help'], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and '--protocol' in r.stdout and '--arch' in r.stdout and 'matched' in r.stdout


def test_stats_pool_reads_truncation_by_column_name():
    rows = [{'sample_id': 'a', 'label': '1', 'prob': '0.9', 'pred': '1', 'family': 'f', 'arch': 'x86'},
            {'sample_id': 'b', 'label': '0', 'prob': '0.1', 'pred': '0', 'family': 'g', 'arch': 'x64',
             'n_blocks_kept': '5', 'n_blocks_total': '9'}]
    assert stats.pool({0: rows}).truncated.tolist() == [-1, 1]
    old = stats.pool({0: rows[:1]})
    assert old.truncated.tolist() == [-1] and old.ids.tolist() == ['a']


def test_balanced_bootstrap_uses_cell_weights_and_self_comparison_is_zero():
    a = {0: arch_preds(seed=0), 1: arch_preds(seed=1)}

    def nudge(p):
        prob = np.round(np.clip(p.prob + 0.1 * (2 * p.y - 1), 0, 1), 2)
        return stats.Preds(p.ids, p.y, prob, (prob >= 0.5).astype(int), p.cluster, p.arch)
    b = {s: nudge(p) for s, p in a.items()}
    same, _ = stats.paired_bootstrap(a, a, resamples=100, balanced=True)
    for key, _ in stats.METRICS:
        assert same[key]['diff'] == 0 and same[key]['lo'] == 0 and same[key]['hi'] == 0
    plain, _ = stats.paired_bootstrap(a, b, resamples=50)
    bal, _ = stats.paired_bootstrap(a, b, resamples=50, balanced=True)
    p = a[0]
    want = cell_balanced(p.y, p.prob, p.pred, p.arch)
    single, _ = stats.paired_bootstrap({0: p}, {0: p}, resamples=5, balanced=True)
    for key in ('auc', 'macro_f1', 'pr_auc'):
        assert single[key]['a'] == pytest.approx(want[key])
    assert bal['auc']['a'] != pytest.approx(plain['auc']['a'])
    ba, _ = stats.paired_bootstrap(a, b, resamples=60, balanced=True)
    ab, _ = stats.paired_bootstrap(b, a, resamples=60, balanced=True)
    for key, _ in stats.METRICS:
        assert ba[key]['diff'] == pytest.approx(-ab[key]['diff']) and ba[key]['lo'] == pytest.approx(-ab[key]['hi'])


def test_stats_balanced_from_disk_and_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    for fold in range(5):
        write_v3_run(tmp_path, 'all', fold, 1.0)
        write_v3_run(tmp_path, 'all', fold, 1.6, arch_balance=True)
        write_v3_baseline(tmp_path, 'all', fold, 1.0)
        write_v3_baseline(tmp_path, 'all', fold, 1.3, suffix='_ab')
    same, info = stats.compare('kfold_v3_all_gnn', 'kfold_v3_all_gnn', 'auc', 50, balanced=True)
    assert info['balanced'] and all(same[k]['diff'] == 0 and same[k]['lo'] == 0 == same[k]['hi'] for k in same)
    assert 'Cell-balanced' in stats.fmt_block('a', 'b', 'auc', 50, same, info)
    plain, plain_info = stats.compare('kfold_v3_all_gnn', 'kfold_v3_all_gnn-ab', 'auc', 50)
    bal, _ = stats.compare('kfold_v3_all_gnn', 'kfold_v3_all_gnn-ab', 'auc', 50, balanced=True)
    assert not plain_info['balanced'] and bal['auc']['diff'] > 0 and bal['auc']['a'] != plain['auc']['a']
    t, _ = stats.compare('tfidf', 'tfidf-ab', 'auc', 50, protocol='kfold_v3', arch='all', balanced=True)
    assert t['auc']['diff'] != 0
    mixed, _ = stats.compare('tfidf-ab', 'kfold_v3_all_gnn-ab', 'auc', 50, balanced=True)
    assert set(mixed) == {'auc', 'pr_auc', 'macro_f1'}
    r = subprocess.run([sys.executable, '-m', 's04_evaluation.stats', '--help'], cwd=ROOT, capture_output=True, text=True)
    assert '--balanced' in r.stdout


def test_baseline_group_names_for_arch_balance(tmp_path, monkeypatch):
    assert stats.baseline_method('tfidf') == ('tfidf_sig_api_lr', '')
    assert stats.baseline_method('structural-tf25') == ('structural_lr', '_tf25')
    assert stats.baseline_method('tfidf-ab') == ('tfidf_sig_api_lr', '_ab')
    assert stats.baseline_method('structural-tf50-ab') == ('structural_lr', '_tf50_ab')
    for bad in ('tfidf-', 'tfidf-tf', 'tfidf-abc', 'kfold_v3_all_gnn-ab', 'gnn-ab'):
        assert stats.baseline_method(bad) is None
    assert stats.baseline_scopes('tfidf-ab', 'kfold_v3_all_gnn-ab') == {'tfidf-ab': ('kfold_v3', 'all')}
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    for fold in range(5):
        write_v3_baseline(tmp_path, 'all', fold, 1.0)
    write_v3_baseline(tmp_path, 'all', 0, 1.3, suffix='_ab')
    scope = ('kfold_v3', 'all')
    assert sorted(stats.folds_by_seed('tfidf', scope)[0]) == [0, 1, 2, 3, 4]
    assert sorted(stats.folds_by_seed('tfidf-ab', scope)[0]) == [0]
    assert stats.folds_by_seed('tfidf-ab', scope)[0][0].name == 'v3kfold0_all_s0_ab'
    complete, partial = stats.load_group('tfidf-ab', 'auc', scope)
    assert not complete and partial == {0: [0]}


WHY = 'process_sample raises MemoryError reading the listing'


def tiny_fold_rows(manifest, rng, signal, skip=()):
    out = {}
    for split in ('val', 'test'):
        rows = [r for r in getattr(manifest, split) if r['sample_id'] not in skip]
        probs = np.clip(0.5 + signal * (np.array([r['label'] for r in rows]) - 0.5) + rng.normal(0, 0.2, len(rows)), 0.01, 0.99)
        out[split] = ([r['sample_id'] for r in rows], probs, [(10, 500, 700)] * len(rows))
    return out


def write_tiny_group(runs, signal, skip=(), **kw):
    """Five kfold runs on the tiny cache through run_io.write_run; `skip` ids are declared excluded."""
    for fold in range(5):
        cfg = RunConfig(protocol='kfold', fold=fold, seed=0, arch='x86', model='gnn', **kw)
        manifest = make_manifest(cfg, load_index())
        rng = np.random.default_rng(10 * fold + int(10 * signal))
        f1, auc = (tiny_fold_rows(manifest, rng, signal, skip) for _ in range(2))
        extra = {'best_epoch': 2, 'epochs_run': 7, 'auc_best_epoch': 3, 'val_auc': 0.9, 'val_macro_f1': 0.8}
        gone = sorted(set(skip) & {r['sample_id'] for r in manifest.val + manifest.test})
        write_run(runs / cfg.name, cfg, f1, auc, val_threshold_of(manifest.val, *f1['val'][:2]),
                  val_threshold_of(manifest.val, *auc['val'][:2]), extra, excluded=gone, excluded_reason=WHY if gone else '')


def test_stats_say_what_was_excluded_from_both_arms(evaltiny, tmp_path, monkeypatch):  # noqa: F811
    runs = tmp_path / 'runs'
    runs.mkdir()
    victim = next(r['sample_id'] for r in make_manifest(RunConfig(protocol='kfold', fold=3, seed=0), load_index()).test)
    write_tiny_group(runs, 0.5)                                                      # the whole fold set
    write_tiny_group(runs, 0.6, mask_blocks=0.15)                                    # a second arm, also whole
    write_tiny_group(runs, 0.8, skip={victim}, tag='cut')                            # an arm that declares the victim excluded
    monkeypatch.setattr(stats, 'RUNS', runs)
    plain, cut, other = 'kfold_x86_gnn', 'kfold_x86_gnn_cut', 'kfold_x86_gnn-mblk15'
    assert stats.group_exclusions(cut) == {victim: WHY} and stats.group_exclusions(plain) == {}
    result, info = stats.compare(plain, cut, 'auc', 30, seeds=[0])
    assert info['n'] == 39 and info['excluded'] == [victim] and info['excluded_reason'] == WHY
    block = stats.fmt_block(plain, cut, 'auc', 30, result, info)
    assert f'1 excluded from both arms: {victim} ({WHY})' in block and '39 test binaries' in block
    result, whole = stats.compare(plain, other, 'auc', 30, seeds=[0])                # two arms without exclusions: nothing printed
    assert whole['n'] == 40 and whole['excluded'] == []
    assert 'excluded from both arms' not in stats.fmt_block(plain, other, 'auc', 30, result, whole)
