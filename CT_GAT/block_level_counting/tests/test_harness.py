"""python -m pytest tests/test_harness.py: PR-AUC, stopping rules, the paired bootstrap, gates, report tables, baselines with
validation predictions, and the day-1 queue (`--dry` names against RunConfig, budget guard).

Run directories are synthetic (tmp_path) except the baselines test, which reads the cache.
"""
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from common.config import RunConfig
from evaluation import gates
from evaluation import report
from evaluation import stats
from evaluation.metrics import ARCHES, cell_balanced, cell_weights, floors, scores
from evaluation.val_threshold import GRID
from training.train import check_supported, loss_weight_fn, parse_config, should_stop, val_threshold_of

ROOT = Path(__file__).resolve().parents[1]
BASH = next((p for p in ('C:/Program Files/Git/bin/bash.exe', shutil.which('bash')) if p and Path(p).exists()), None)


def gnn(**kw):
    return RunConfig(model='gnn', **kw)


def test_pr_auc_in_scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    prob = np.clip(y * 0.3 + rng.random(200) * 0.7, 0, 1)
    s = scores(y, prob, (prob >= 0.5).astype(int), ['x86'] * 200)
    assert s['pr_auc'] == pytest.approx(average_precision_score(y, prob))
    one = scores([1, 1, 1], [0.2, 0.9, 0.5], [0, 1, 1], ['x86'] * 3)
    assert np.isnan(one['pr_auc']) and np.isnan(one['auc'])


def test_should_stop_rules():
    tf, g = RunConfig(patience=3), gnn(patience=3)
    assert not should_stop(tf, 4, 2, 4) and should_stop(tf, 5, 2, 4)
    assert not should_stop(g, 6, 2, 4) and should_stop(g, 7, 2, 4)
    assert not should_stop(gnn(patience=3, fixed_epochs=2), 99, 1, 1)
    assert not should_stop(RunConfig(patience=3, fixed_epochs=2), 99, 1, 1)


def test_unsupported_flags_raise():
    check_supported(gnn())
    for kw in (dict(stack_tfidf=True), dict(aux_recon=0.1), dict(aux_edge=0.1)):
        with pytest.raises(NotImplementedError):
            check_supported(gnn(**kw))
    cfg, _ = parse_config(['--model', 'gnn', '--stack-tfidf'])
    with pytest.raises(NotImplementedError):
        check_supported(cfg)


def test_val_threshold_of_uses_grid():
    rows = [{'sample_id': str(i), 'label': int(i >= 6)} for i in range(10)]
    probs = np.array([0.1] * 6 + [0.9] * 4)
    t = val_threshold_of(rows, [r['sample_id'] for r in rows], probs)
    assert t in GRID and 0.1 < t <= 0.9


def preds(n=300, seed=0, signal=1.0, ties=True):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.6).astype(int)
    prob = 1 / (1 + np.exp(-(signal * (2 * y - 1) + rng.normal(size=n))))
    prob = np.round(prob, 2) if ties else prob
    cluster = np.where(y == 1, [f'fam:{i % 7}' for i in range(n)], [f'id:{i}' for i in range(n)])
    return stats.Preds(np.array([f'{i:05d}' for i in range(n)]), y, prob, (prob >= 0.5).astype(int), cluster)


def resignal(p, signal, seed):
    """The same samples and labels with new probabilities."""
    rng = np.random.default_rng(seed + 50)
    prob = np.round(1 / (1 + np.exp(-(signal * (2 * p.y - 1) + rng.normal(size=len(p.y))))), 2)
    return stats.Preds(p.ids, p.y, prob, (prob >= 0.5).astype(int), p.cluster)


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


def write_run(root, cfg, seed_probs, selection=True, thresholds=0.4):
    """A fake kfold fold or bundle run: config, predictions.csv, predictions_auc.csv, metrics.json."""
    d = root / cfg.name
    d.mkdir(parents=True)
    cfg.save(d / 'config.json')
    head = ['sample_id', 'split', 'label', 'prob', 'pred', 'arch', 'family', 'fold', 'n_segments_total']
    for fname, thr in (('predictions.csv', 0.5), ('predictions_auc.csv', thresholds)):
        if fname == 'predictions_auc.csv' and not selection:
            continue
        with (d / fname).open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(head)
            for sid, y, p, fam in seed_probs:
                w.writerow([sid, 'test', y, f'{p:.6f}', int(p >= thr), 'x86', fam, cfg.fold, 5])
    return d


def fake_fold_rows(fold, seed, signal):
    p = preds(n=120, seed=100 * seed + fold, signal=signal)
    return [(f'f{fold}_{i}', int(y), float(pr), f'fam{i % 5}' if y else f'g{fold}_{i}')
            for i, (y, pr) in enumerate(zip(p.y, p.prob))]


def build_fake(root, groups):
    """groups: {variant kwargs tuple: (signal, folds)}; every group gets seed 0."""
    for kw, (signal, folds) in groups.items():
        for fold in folds:
            write_run(root, gnn(protocol='kfold', fold=fold, seed=0, **dict(kw)), fake_fold_rows(fold, 0, signal))


def test_stats_and_gates_from_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    gates.pooled.cache_clear()
    build_fake(tmp_path, {(('control', 'no_edges'),): (1.2, range(5)), (): (1.6, range(5)),
                          (('control', 'shuffle'),): (0.8, [0, 1, 2])})
    result, info = stats.compare('kfold_x86_gnn-noedges', 'kfold_x86_gnn-noedges', 'f1', 100)
    assert info['seeds'] == [0] and result['auc']['diff'] == 0 and result['macro_f1']['hi'] == 0
    result, info = stats.compare('kfold_x86_gnn-noedges', 'kfold_x86_gnn', 'auc', 100)
    assert result['auc']['diff'] > 0 and result['auc']['lo'] < result['auc']['hi']
    swapped, _ = stats.compare('kfold_x86_gnn', 'kfold_x86_gnn-noedges', 'auc', 100)
    assert swapped['auc']['diff'] == pytest.approx(-result['auc']['diff'])
    with pytest.raises(stats.MissingGroup):
        stats.compare('kfold_x86_gnn-noedges', 'kfold_x86_gnn-shuffle', 'f1', 50)
    text = '\n'.join(gates.day1(50))
    assert gates.LABEL in text
    assert 'kfold_x86_gnn-shuffle' in text and 'missing (seed 0 folds present [0, 1, 2] of 0-4)' in text
    assert '**R2 tfidf vs R7 typed**' in text and '**R2 tfidf vs R4 shuffle** (selection f1): missing' in text
    gates.pooled.cache_clear()


def write_baseline(root, fold, signal, threshold=0.4, tf=None):
    d = root / 'baselines' / (f'kfold{fold}_x86_s0' + (f'_tf{tf}' if tf else ''))
    d.mkdir(parents=True)
    methods = {'tfidf_sig_api_lr': {'threshold_val': threshold}}
    (d / 'metrics.json').write_text(json.dumps({'protocol': 'kfold', 'fold': fold, 'arch': 'x86', 'seed': 0,
                                                'methods': methods,
                                                'train_fraction': tf / 100 if tf else 1.0}))
    with (d / 'predictions.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'split', 'method', 'label', 'prob', 'pred', 'arch', 'family', 'fold'])
        for sid, y, p, fam in fake_fold_rows(fold, 0, signal):
            w.writerow([sid, 'test', 'tfidf_sig_api_lr', y, f'{p:.6f}', int(p >= 0.5), 'x86', fam, fold])
            w.writerow([sid + 'v', 'val', 'tfidf_sig_api_lr', y, f'{p:.6f}', int(p >= 0.5), 'x86', fam, fold])


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


def test_report_labels():
    assert report.model_label(RunConfig()) == 'graph transformer'
    assert report.model_label(RunConfig(no_graph=True)) == 'no-graph transformer'
    assert report.model_label(gnn()) == 'edge-typed GNN'
    assert report.model_label(gnn(control='no_edges')) == 'GNN no-edges'
    assert report.model_label(gnn(control='shuffle')) == 'GNN shuffle'
    assert report.model_label(gnn(untyped=True)) == 'GNN untyped'
    assert report.model_label(gnn(control='no_edges', encoder='mean')) == 'GNN no-edges enc-mean'
    assert report.model_label(gnn(mask_tokens=0.15)) == 'edge-typed GNN mask-tok 15%'
    labelled = set(report.LABELLED)
    for cfg in (gnn(), gnn(control='shuffle'), gnn(encoder='mean')):
        overrides = {k for k, v in report.asdict(cfg).items() if k not in labelled and v != report.DEFAULTS[k]}
        assert not overrides, overrides


def test_report_auc_table_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'RUNS', tmp_path)
    for name, selection in (('gnn', True), ('old', False)):
        cfg = gnn(seed=0) if selection else RunConfig(seed=0)
        d = tmp_path / cfg.name
        d.mkdir()
        cfg.save(d / 'config.json')
        test = scores([0, 1, 0, 1], [0.1, 0.9, 0.4, 0.6], [0, 1, 0, 1], ['x86'] * 4)
        m = {'test': test, 'family_recall': {}, 'floors': {}}
        if selection:
            m['selection'] = {'auc': {'test': {**test, 'macro_f1': 0.123}, 'family_recall': {}}}
        (d / 'metrics.json').write_text(json.dumps(m))
        (d / 'predictions.csv').write_text('sample_id,split,label,prob,pred,arch,family,fold,n_segments_total\n'
                                           'a,test,0,0.1,0,x86,g,0,1\nb,test,1,0.9,1,x86,f,0,1\n')
    main, main_notes = report.collect({})
    auc, auc_notes = report.collect({}, auc=True)
    assert {k[2] for k, _ in main} == {'edge-typed GNN', 'graph transformer'}
    assert [k[2] for k, _ in auc] == ['edge-typed GNN']
    assert auc[0][1][0][0]['macro_f1'] == 0.123
    assert any('graph transformer' in n and 'no AUC-selected checkpoint' in n for n in auc_notes)
    assert not main_notes


def test_baselines_write_val_rows_and_thresholds(tmp_path):
    from evaluation import baselines
    sizes = {k: len(v) for k, v in json.loads((baselines.CACHE / 'vocab.json').read_text(encoding='utf-8')).items()}
    baselines.run(RunConfig(protocol='kfold', fold=0, seed=0), sizes, tmp_path)
    d = tmp_path / 'kfold0_x86_s0'
    m = json.loads((d / 'metrics.json').read_text())
    for v in m['methods'].values():
        assert v['threshold_val'] in [float(g) for g in GRID]
        assert v['test_at_val_threshold']['n'] == v['test']['n'] and 'pr_auc' in v['test']
    with (d / 'predictions.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert {r['split'] for r in rows} == {'val', 'test'}
    assert len(report.read_predictions(d / 'predictions.csv', method='structural_lr')) == m['n_test']
    assert list(rows[0])[:3] == ['sample_id', 'split', 'method']


def queue(*args, **env):
    r = subprocess.run([BASH, 'queue_gnn.sh', *args], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, **env})
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


@pytest.mark.skipif(BASH is None, reason='needs bash')
def test_queue_dry_names_match_run_configs():
    lines = queue('--stage', 'day1', '--dry')
    runs = [l.split()[1:] for l in lines if l.startswith('run ')]
    assert len(runs) == 28
    for name, *argv in runs:
        cfg, _ = parse_config(argv)
        assert cfg.name == name, (name, cfg.name)
    names = [n for n, *_ in runs]
    assert names[0] == 'kfold0_x86_gnn-noedges_s0' and names[5] == 'kfold0_x86_gnn-noedges-enc-mean_s0'
    assert [n.split('_')[2] for n in names[:25:5]] == ['gnn-noedges', 'gnn-noedges-enc-mean', 'gnn', 'gnn-untyped',
                                                      'gnn-shuffle']
    assert names[25:] == ['kfold3_x86_nograph_s2', 'kfold4_x86_graph_s2', 'kfold4_x86_nograph_s2']
    assert lines[0].startswith('cpu baselines') and lines[-3:] == [
        'cpu report.py', 'cpu gates.py --stage day1 > RUNS/gates_day1.md', 'cpu val_threshold.py > RUNS/val_threshold.md']
    assert not any(l.startswith('skipped') for l in lines)


@pytest.mark.skipif(BASH is None, reason='needs bash')
def test_queue_budget_guard():
    lines = queue('--stage', 'day1', '--dry', GPU_BUDGET_MIN='290', DRY_RUN_MIN='100')
    assert sum(l.startswith('run ') for l in lines) == 3
    assert sum(l.startswith('skipped (budget) ') for l in lines) == 25
    assert lines[-1] == 'cpu val_threshold.py > RUNS/val_threshold.md'
    assert not any(l.startswith('skipped') for l in queue('--stage', 'day1', '--dry', GPU_BUDGET_MIN='290'))
    assert queue('--stage', 'day1', '--dry', GPU_BUDGET_MIN='0')[1].startswith('skipped (budget) kfold0_x86_gnn-noedges')


# step 5b: day1b gates, train-fraction baselines, labels, the day1b queue and the stop script

def test_day1b_gates_from_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    gates.pooled.cache_clear()
    build_fake(tmp_path, {(('control', 'no_edges'),): (1.2, range(5)), (): (1.6, range(5)),
                          (('direction', 'forward'),): (1.5, range(5)), (('train_fraction', 0.5),): (1.3, range(5)),
                          (('drop_edges', 0.2),): (1.4, [0, 1, 4])})
    for fold in range(5):
        write_baseline(tmp_path, fold, 1.0)
        write_baseline(tmp_path, fold, 0.8, tf=50)
    assert gates.gnn_groups() == ['kfold_x86_gnn', 'kfold_x86_gnn-dedge20', 'kfold_x86_gnn-fwd',
                                  'kfold_x86_gnn-noedges', 'kfold_x86_gnn-tf50']
    text = '\n'.join(gates.day1b(50))
    assert 'stage day1b' in text and gates.LABEL in text
    table = text.split('## Paired differences')[0]
    assert '| kfold_x86_gnn-dedge20 | missing (seed 0 folds present [0, 1, 4] of 0-4)' in table
    assert '| kfold_x86_gnn-fwd | 0.' in table and 'AUC checkpoint' in table
    assert '**kfold_x86_gnn-fwd vs R7 typed**' in text and '**kfold_x86_gnn-fwd vs R2 tfidf**' in text
    assert 'selection auc' in text and 'kfold_x86_gnn-fwd minus kfold_x86_gnn ' in text
    assert '**kfold_x86_gnn-dedge20 vs R7 typed** (selection auc): missing: kfold_x86_gnn-dedge20' in text
    assert '**kfold_x86_gnn-bwd vs R7 typed** (selection auc): missing' in text
    curve = text.split('## Learning curve')[1]
    rows = [l for l in curve.splitlines() if l.startswith('| ') and 'train fraction' not in l]
    assert [l.split(' | ')[:3] for l in rows] == [
        ['| 0.25', 'R7 typed', 'kfold_x86_gnn-tf25'], ['| 0.25', 'TF-IDF', 'tfidf-tf25'],
        ['| 0.5', 'R7 typed', 'kfold_x86_gnn-tf50'], ['| 0.5', 'TF-IDF', 'tfidf-tf50'],
        ['| 1.0', 'R7 typed', 'kfold_x86_gnn'], ['| 1.0', 'TF-IDF', 'tfidf']]
    assert 'missing' in rows[0] and 'missing' in rows[1] and 'missing' not in ''.join(rows[2:])
    gates.pooled.cache_clear()


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


def test_report_labels_new_arms():
    labels = {report.model_label(gnn(**kw)) for kw in (
        {}, dict(direction='forward'), dict(direction='backward'), dict(direction='undirected'),
        dict(drop_edges=0.2), dict(flip_edges=0.2), dict(gnn_fn_layers=8), dict(gnn_bin_layers=4),
        dict(train_fraction=0.5), dict(train_fraction=0.25))}
    assert len(labels) == 10
    assert report.model_label(gnn(direction='forward')) == 'edge-typed GNN forward'
    assert report.model_label(gnn(drop_edges=0.2, gnn_fn_layers=8)) == 'edge-typed GNN drop-edge 20% L8'
    assert report.model_label(RunConfig(train_fraction=0.5)) == 'graph transformer train 50%'
    labelled = set(report.LABELLED)
    cfg = gnn(direction='forward', drop_edges=0.2, flip_edges=0.1, gnn_fn_layers=8, train_fraction=0.5)
    assert not {k for k, v in report.asdict(cfg).items() if k not in labelled and v != report.DEFAULTS[k]}


def test_baselines_train_fraction(tmp_path):
    from evaluation import baselines
    from common.manifests import load_index, make_manifest
    sizes = {k: len(v) for k, v in json.loads((baselines.CACHE / 'vocab.json').read_text(encoding='utf-8')).items()}
    baselines.run(RunConfig(protocol='kfold', fold=1, seed=0, train_fraction=0.25), sizes, tmp_path)
    d = tmp_path / 'kfold1_x86_s0_tf25'
    m = json.loads((d / 'metrics.json').read_text())
    full = make_manifest(RunConfig(protocol='kfold', fold=1, seed=0), load_index())
    assert m['train_fraction'] == 0.25 and m['n_val'] == len(full.val) and m['n_test'] == len(full.test)
    assert 0.2 * len(full.train) < m['n_train'] < 0.3 * len(full.train)
    for v in m['methods'].values():
        assert v['threshold_val'] in [float(g) for g in GRID] and 'pr_auc' in v['test']
    with (d / 'predictions.csv').open() as f:
        assert {r['split'] for r in csv.DictReader(f)} == {'val', 'test'}
    assert [p.name for p in tmp_path.iterdir()] == ['kfold1_x86_s0_tf25']


DAY1B_ARMS = ['gnn-mtok15', 'gnn-dfn5-dlib40', 'gnn-undir', 'gnn-fwd', 'gnn-dfn15', 'gnn-tf50', 'gnn-tf25',
              'gnn-mblk15', 'gnn-dedge20', 'gnn-flip20', 'gnn-mtok50', 'gnn-bwd', 'gnn-L8']


@pytest.mark.skipif(BASH is None, reason='needs bash')
def test_queue_day1b_dry_names_match_run_configs():
    lines = queue('--stage', 'day1b', '--dry')
    runs = [l.split()[1:] for l in lines if l.startswith('run ')]
    assert len(runs) == 65
    for name, *argv in runs:
        cfg, _ = parse_config(argv)
        assert cfg.name == name, (name, cfg.name)
        assert cfg.protocol == 'kfold' and cfg.arch == 'x86' and cfg.seed == 0 and cfg.model == 'gnn'
    names = [n for n, *_ in runs]
    assert [n.split('_')[2] for n in names[::5]] == DAY1B_ARMS
    assert [n.split('_')[0] for n in names[:5]] == [f'kfold{f}' for f in range(5)]
    assert lines[:2] == ['cpu baselines --protocol kfold --fold 0 1 2 3 4 --arch x86 --seed 0 --train-fraction 0.5',
                         'cpu baselines --protocol kfold --fold 0 1 2 3 4 --arch x86 --seed 0 --train-fraction 0.25']
    assert lines[-3:] == ['cpu report.py', 'cpu gates.py --stage day1b > RUNS/gates_day1b.md',
                          'cpu val_threshold.py > RUNS/val_threshold.md']
    assert not any(l.startswith(('skipped', 'stopped')) for l in lines)
    by_arm = {n.split('_')[2]: parse_config(a)[0] for (n, *a) in runs[::5]}
    assert by_arm['gnn-undir'].direction == 'undirected' and by_arm['gnn-fwd'].direction == 'forward'
    assert by_arm['gnn-bwd'].direction == 'backward' and by_arm['gnn-L8'].gnn_fn_layers == 8
    assert (by_arm['gnn-dedge20'].drop_edges, by_arm['gnn-flip20'].flip_edges) == (0.2, 0.2)
    assert (by_arm['gnn-tf50'].train_fraction, by_arm['gnn-tf25'].train_fraction) == (0.5, 0.25)
    assert (by_arm['gnn-mtok15'].mask_tokens, by_arm['gnn-mtok50'].mask_tokens) == (0.15, 0.5)
    assert (by_arm['gnn-dfn5-dlib40'].drop_functions, by_arm['gnn-dfn5-dlib40'].drop_library) == (0.05, 0.4)
    assert (by_arm['gnn-dfn15'].drop_functions, by_arm['gnn-mblk15'].mask_blocks) == (0.15, 0.15)


@pytest.mark.skipif(BASH is None, reason='needs bash')
def test_queue_day1b_budget_guard():
    lines = queue('--stage', 'day1b', '--dry', DRY_RUN_MIN='100')
    assert sum(l.startswith('run ') for l in lines) == 6
    assert sum(l.startswith('skipped (budget) ') for l in lines) == 59
    assert sum(l.startswith('run ') for l in queue('--stage', 'day1b', '--dry', DRY_RUN_MIN='100',
                                                   GPU_BUDGET_MIN='290')) == 3
    assert queue('--stage', 'day1b', '--dry', GPU_BUDGET_MIN='0')[2].startswith('skipped (budget) kfold0_x86_gnn-mtok15')


STUB = """#!/bin/bash
echo "$@" >> "$GT_RUNS/calls.txt"
case $1 in
  train.py) n=$(grep -c '^train.py' "$GT_RUNS/calls.txt"); [ "$n" = "$STOP_AFTER" ] && touch "$GT_RUNS/STOP";;
  baselines.py) mkdir -p "$GT_RUNS/baselines";;
esac
exit 0
"""


@pytest.mark.skipif(BASH is None, reason='needs bash')
def test_queue_stop_file_with_stub_interpreter(tmp_path):
    """A scratch RUNS and a stub interpreter that creates RUNS/STOP during its 3rd train.py call, as stop_queue.sh
    would. The 3rd run finishes, every later run is logged as stopped, the report steps still run; a stale STOP from
    before the start is removed; runs with metrics.json (arms 2 and 3, 'run by the coordinator') are skipped."""
    runs = tmp_path / 'runs'
    (runs / 'logs').mkdir(parents=True)
    stub = tmp_path / 'stub.sh'
    stub.write_text(STUB, newline='\n')
    stub.chmod(0o755)
    (runs / 'STOP').write_text('')
    done = {f'kfold{f}_x86_{arm}_s0' for f in range(5) for arm in ('gnn-mtok15', 'gnn-dfn5-dlib40')}
    for name in done:
        (runs / name).mkdir()
        (runs / name / 'metrics.json').write_text('{}')
    env = dict(GT_PY=stub.as_posix(), GT_RUNS=runs.as_posix(), GT_TRAIN_PATTERN='gtqueue_never_matches_train[.]py',
               STOP_AFTER='3')
    r = subprocess.run([BASH, 'queue_gnn.sh', '--stage', 'day1b'], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, **env})
    assert r.returncode == 0, r.stderr + r.stdout
    out = r.stdout.splitlines()
    assert {l for l in out if l.startswith('skip ')} == {f'skip {n}' for n in done}
    started = [l.split()[1] for l in out if l.startswith('start kfold')]
    assert started == [f'kfold{f}_x86_gnn-undir_s0' for f in range(3)]
    assert [l for l in out if l.startswith('finished kfold')][-1].split()[1] == 'kfold2_x86_gnn-undir_s0'
    stopped = [l.split(' ', 2)[2] for l in out if l.startswith('stopped before ')]
    assert len(stopped) == 65 - len(done) - 3 and stopped[0] == 'kfold3_x86_gnn-undir_s0'
    assert stopped[-1] == 'kfold4_x86_gnn-L8_s0' and not any(l.startswith('skipped') for l in out)
    calls = (runs / 'calls.txt').read_text().splitlines()
    assert sum(c.startswith('train.py') for c in calls) == 3
    assert [c for c in calls if c.startswith(('report.py', 'gates.py', 'val_threshold.py'))] == [
        'report.py', 'gates.py --stage day1b', 'val_threshold.py']
    assert sum(c.startswith('baselines.py') for c in calls) == 2
    assert (runs / 'gates_day1b.md').exists() and out[-1].startswith('queue done')
    assert (runs / 'STOP').exists()

    r = subprocess.run([BASH, 'queue_gnn.sh', '--stage', 'day1b'], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, **env, 'STOP_AFTER': '0'})
    assert r.returncode == 0, r.stderr
    assert not (runs / 'STOP').exists()
    assert sum(l.startswith('start kfold') for l in r.stdout.splitlines()) == 65 - len(done)


def spawn_marked(marker):
    return subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)', marker])


@pytest.mark.skipif(BASH is None, reason='needs bash')
def test_stop_queue_script(tmp_path):
    """Without --now only RUNS/STOP appears; with --now the process whose command line matches the pattern dies and
    an unrelated python process survives. The pattern is a test marker, never the real train.py (a live training
    run on this machine must not be touched)."""
    env = {**os.environ, 'GT_RUNS': (tmp_path / 'runs').as_posix(), 'GT_TRAIN_PATTERN': 'gtstop_marker_train[.]py'}
    victim, bystander = spawn_marked('gtstop_marker_train.py'), spawn_marked('gtstop_bystander_train.py')
    try:
        r = subprocess.run([BASH, 'stop_queue.sh'], cwd=ROOT, capture_output=True, text=True, env=env)
        assert r.returncode == 0 and (tmp_path / 'runs' / 'STOP').exists() and 'killed' not in r.stdout
        assert victim.poll() is None
        (tmp_path / 'runs' / 'STOP').unlink()
        r = subprocess.run([BASH, 'stop_queue.sh', '--now'], cwd=ROOT, capture_output=True, text=True, env=env)
        assert r.returncode == 0 and (tmp_path / 'runs' / 'STOP').exists()
        assert 'created' in r.stdout and 'killed train.py process(es):' in r.stdout
        assert victim.wait(timeout=20) is not None and bystander.poll() is None
        r = subprocess.run([BASH, 'stop_queue.sh', '--now'], cwd=ROOT, capture_output=True, text=True, env=env)
        assert 'no train.py process was running' in r.stdout
        assert subprocess.run([BASH, 'stop_queue.sh', '--bogus'], cwd=ROOT, env=env, capture_output=True).returncode == 2
    finally:
        for p in (victim, bystander):
            p.kill()
            p.wait()


# step 6: kfold_v3 in report, stats, gates, val_threshold and the v3 queue

COMBINED = Path('C:/Users/chaoa/Downloads/graph_transformer_cache_combined')
needs_combined = pytest.mark.skipif(not (COMBINED / 'index.csv').exists(), reason='needs the combined cache')
V3_X86_RULE = {'all': [0.607, 0.686, 0.685, 0.733, 0.715], 'matched': [0.535, 0.598, 0.588, 0.660, 0.628]}
V3_X86_RULE_POOLED = {'all': 0.690, 'matched': 0.606}


def v3_rows(fold, seed, signal):
    """fake_fold_rows with an architecture per row (x86 for even ids)."""
    return [(sid, y, p, fam, 'x86' if i % 2 == 0 else 'x64')
            for i, (sid, y, p, fam) in enumerate(fake_fold_rows(fold, seed, signal))]


def write_v3_run(root, arch, fold, signal, **kw):
    cfg = gnn(protocol='kfold_v3', fold=fold, arch=arch, **kw)
    d = root / cfg.name
    d.mkdir(parents=True)
    cfg.save(d / 'config.json')
    head = ['sample_id', 'split', 'label', 'prob', 'pred', 'arch', 'family', 'fold', 'n_segments_total']
    rows = v3_rows(fold, 0, signal)
    for fname, thr in (('predictions.csv', 0.5), ('predictions_auc.csv', 0.4)):
        with (d / fname).open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(head)
            for sid, y, p, fam, a in rows:
                w.writerow([sid, 'test', y, f'{p:.6f}', int(p >= thr), a, fam, fold, 5])
    (d / 'manifest.json').write_text(json.dumps({'train': [f'tr{fold}_{i}' for i in range(10)]}))
    (d / 'metrics.json').write_text('{}')
    return d


def write_v3_baseline(root, arch, fold, signal, threshold=0.4, suffix=''):
    d = root / 'baselines' / f'v3kfold{fold}_{arch}_s0{suffix}'
    d.mkdir(parents=True)
    methods = {m: {'threshold_val': threshold, 'family_recall': {}, 'test': {}, 'test_at_val_threshold': {}}
               for m in ('tfidf_sig_api_lr', 'structural_lr')}
    (d / 'metrics.json').write_text(json.dumps({'protocol': 'kfold_v3', 'fold': fold, 'arch': arch, 'seed': 0,
                                                'train_fraction': 1.0, 'arch_balance': bool(suffix), 'floors': {},
                                                'methods': methods}))
    with (d / 'predictions.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'split', 'method', 'label', 'prob', 'pred', 'arch', 'family', 'fold'])
        for sid, y, p, fam, a in v3_rows(fold, 0, signal):
            for method in methods:
                w.writerow([sid, 'test', method, y, f'{p:.6f}', int(p >= 0.5), a, fam, fold])
                w.writerow([sid + 'v', 'val', method, y, f'{p:.6f}', int(p >= 0.5), a, fam, fold])


FAKE_FLOORS = {'majority': {'auc': 0.5, 'pr_auc': 0.6, 'macro_f1': 0.4, 'recall_R_x86': 1.0, 'recall_R_x64': 1.0,
                            'recall_G_x86': 0.0, 'recall_G_x64': 0.0,
                            'cell_balanced': {'auc': 0.5, 'macro_f1': 0.33}},
               'x86_rule': {'auc': 0.7, 'pr_auc': 0.8, 'macro_f1': 0.69, 'recall_R_x86': 1.0, 'recall_R_x64': 0.0,
                            'recall_G_x86': 0.0, 'recall_G_x64': 1.0,
                            'cell_balanced': {'auc': 0.5, 'macro_f1': 0.5}}}
V3_MASKED = dict(mask_blocks=0.15, select_on='cell')
V3_KW = {'gnn': {}, 'gnn-noedges': dict(control='no_edges'), 'gnn-shuffle': dict(control='shuffle')}
V3 = {'gnn': 'gnn-mblk15-selcell', 'gnn-noedges': 'gnn-noedges-mblk15-selcell',
      'gnn-shuffle': 'gnn-shuffle-mblk15-selcell'}


def build_v3_fake(tmp_path, monkeypatch):
    """arch all: R7 stronger than R2 = R4 (R4 has folds 0-2 only); arch matched: every arm equal. Every GNN arm is
    block-masked with cell-balanced selection, as in the v3 queue."""
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    monkeypatch.setattr(report, 'RUNS', tmp_path)
    monkeypatch.setattr(report, 'v3_floors', lambda arch, seed=0: (FAKE_FLOORS, [FAKE_FLOORS] * 5))
    gates.pooled.cache_clear()
    gates.v3_cell.cache_clear()
    plan = {'all': {'gnn-noedges': (1.2, 5), 'gnn': (1.7, 5), 'gnn-shuffle': (1.2, 3)},
            'matched': {v: (1.2, 5) for v in V3_KW}}
    for arch, arms in plan.items():
        for variant, (signal, n_folds) in arms.items():
            for fold in range(n_folds):
                write_v3_run(tmp_path, arch, fold, signal, **V3_KW[variant], **V3_MASKED)
        for fold in range(5):
            write_v3_baseline(tmp_path, arch, fold, 1.3 if arch == 'all' else 0.6)


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
    r = subprocess.run([sys.executable, 'stats.py', '--help'], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and '--protocol' in r.stdout and '--arch' in r.stdout and 'matched' in r.stdout


def test_v3_gates_from_disk(tmp_path, monkeypatch):
    build_v3_fake(tmp_path, monkeypatch)
    text = '\n'.join(gates.v3(50))
    assert 'stage v3' in text and gates.LABEL in text
    all_part, matched_part = text.split('## Arch matched: pooled table')
    head = all_part.split('### Arch all')[0]
    row = {l.split(' | ')[0].lstrip('| '): l for l in head.splitlines() if l.startswith('| ')}
    assert 'R x86 recall' in head and 'AUC ckpt, val thr AUC' in head and 'F1 ckpt, 0.5 cell-bal AUC' in head
    assert 'AUC ckpt, val thr cell-bal macro-F1' in head
    assert row['R0 TF-IDF'].split(' | ')[1] == 'tfidf' and 'missing' not in row['R0 TF-IDF']
    assert 'missing (seed 0 folds present [0, 1, 2] of 0-4)' in row['R4 shuffle']
    assert 'R5 untyped' not in head and 'selcell' in row['R7 typed'].split(' | ')[1] and 'mblk15' in row['R7 typed']
    cells = row['R7 typed'].split(' | ')
    assert len(cells) == 16 and 'missing' not in row['R7 typed']
    assert all(0 <= float(c.strip(' |')) <= 1 for c in cells[2:])
    assert '| floor: majority | - | 0.500 | 0.600 | 0.400 | 0.500 | 0.600 | 0.400 | 1.000 | 1.000 | 0.000 | 0.000 '            '| 0.500 | 0.330 | 0.500 | 0.330 |' in head
    assert '| floor: x86_rule | - | 0.700 | 0.800 | 0.690 | 0.700 | 0.800 | 0.690 | 1.000 | 0.000 | 0.000 | 1.000 '            '| 0.500 | 0.500 | 0.500 | 0.500 |' in head
    assert 'missing' in row['R7 typed ab'] and 'kfold_v3_all_gnn-mblk15-ab-selcell' in row['R7 typed ab']
    assert 'block-masked' in head and 'validation cell-balanced AUC' in head
    assert 'floor: x86_rule' in head and 'x86-rule macro-F1 per fold: 0.690, 0.690, 0.690, 0.690, 0.690' in head
    for pair in ('R2 no-edges vs R7 typed', 'R4 shuffle vs R7 typed', 'R0 TF-IDF vs R2 no-edges'):
        assert f'**{pair}**' in all_part and f'**{pair}**' in matched_part
    assert '(selection auc): missing: kfold_v3_all_gnn-shuffle-mblk15-selcell' in all_part
    assert 'kfold_v3_all_gnn-noedges-mblk15-selcell minus tfidf' in all_part
    assert 'kfold_v3_matched_gnn-noedges-mblk15-selcell minus tfidf' in matched_part
    assert 'kfold_v3_all_gnn-mblk15-selcell minus kfold_v3_all_gnn-noedges-mblk15-selcell' in all_part
    assert all_part.count('Cell-balanced: every test sample weighted') >= 1
    lines = [l for l in text.split('## Verdict')[1].splitlines() if l.startswith('- Arch')]
    assert len(lines) == 8 and all(gates.LABEL in l for l in lines)
    assert lines[0].startswith('- Arch all: cannot assess, missing: R4 shuffle')
    assert lines[4].startswith('- Arch matched: R7 does not beat both R2 and R4')
    assert lines[5].startswith('- Arch matched: cell-balanced: R7 does not beat both R2 and R4')
    assert lines[6].startswith('- Arch matched: cannot assess, missing: R2 no-edges ab')
    assert lines[7].startswith('- Arch matched: cell-balanced: cannot assess, missing: R2 no-edges ab')


def test_v3_verdict_rule():
    def res(auc, f1, lo=0.001):
        return {'auc': {'diff': auc, 'lo': lo, 'hi': auc + 0.1}, 'macro_f1': {'diff': f1, 'lo': lo, 'hi': f1 + 0.1}}
    assert gates.beats(res(0.01, 0.02)) and not gates.beats(res(0.009, 0.05)) and not gates.beats(res(0.05, 0.019))
    assert not gates.beats(res(0.05, 0.05, lo=-0.001)) and not gates.beats(res(0.05, 0.05, lo=0.0))
    both = {('R2 no-edges', 'R7 typed'): res(0.05, 0.05), ('R4 shuffle', 'R7 typed'): res(0.05, 0.05)}
    assert 'R7 beats both R2 and R4: beats R2 no-edges' in gates.v3_verdict('all', both)
    one = {**both, ('R4 shuffle', 'R7 typed'): res(0.0, 0.0, lo=-0.1)}
    line = gates.v3_verdict('matched', one)
    assert 'does not beat both' in line and 'does not beat R4 shuffle' in line and 'seed 0 only' in line


def test_v3_gates_beats_both_on_strong_signal(tmp_path, monkeypatch):
    build_v3_fake(tmp_path, monkeypatch)
    for fold in (3, 4):
        write_v3_run(tmp_path, 'all', fold, 1.2, control='shuffle', **V3_MASKED)
    gates.v3_cell.cache_clear()
    text = '\n'.join(gates.v3(100))
    line = next(l for l in text.split('## Verdict')[1].splitlines() if l.startswith('- Arch all'))
    assert line.startswith('- Arch all: R7 beats both R2 and R4: beats R2 no-edges') and 'beats R4 shuffle' in line


def test_v3_report_rows_and_floors(tmp_path, monkeypatch):
    build_v3_fake(tmp_path, monkeypatch)
    labels = {sid: y for fold in range(5) for sid, y, *_ in fake_fold_rows(fold, 0, 1.0)}
    for fold in range(5):
        labels.update({f'tr{fold}_{i}': i % 2 for i in range(10)})
    results, notes = report.collect(labels)
    keys = {k for k, _ in results}
    assert ('kfold_v3', 'all', 'GNN no-edges mask-blk 15% cell-select') in keys
    assert ('kfold_v3', 'matched', 'edge-typed GNN mask-blk 15% cell-select') in keys
    assert ('kfold_v3', 'all', 'tfidf_sig_api_lr') in keys and ('kfold_v3', 'matched', 'structural_lr') in keys
    assert any('GNN shuffle mask-blk 15% cell-select, kfold_v3, all, seed 0: only folds [0, 1, 2] present' in n for n in notes)
    s, _, fl = dict(results)[('kfold_v3', 'all', 'edge-typed GNN mask-blk 15% cell-select')][0]
    assert s['n'] == 600 and set(fl) == {'majority', 'x86_rule'} and s['recall_R_x64'] >= 0
    auc_results, _ = report.collect(labels, auc=True)
    assert ('kfold_v3', 'matched', 'tfidf_sig_api_lr') in {k for k, _ in auc_results}
    assert not any(k[0] == 'kfold' for k in keys)


def test_v3_val_threshold_pools_like_kfold(tmp_path, monkeypatch, capsys):
    from evaluation import val_threshold
    monkeypatch.setattr(val_threshold, 'RUNS', tmp_path)
    for fold in range(5):
        d = write_v3_run(tmp_path, 'all', fold, 1.5)
        with (d / 'predictions.csv').open('a', newline='') as f:
            w = csv.writer(f)
            for sid, y, p, fam, a in v3_rows(fold, 3, 1.5):
                w.writerow([sid + 'v', 'val', y, f'{p:.6f}', int(p >= 0.5), a, fam, fold, 5])
    val_threshold.main()
    out = capsys.readouterr().out
    assert out.count('| kfold_v3_all_gnn_s0 | folds | 5 |') == 1 and '| kfold_v3_all_gnn_s0 | pooled | 600 |' in out
    assert '| kfold_v3_all_gnn | 1 |' in out


@needs_combined
@pytest.mark.parametrize('arch', ['all', 'matched'])
def test_v3_floors_reproduce_the_plan_table(arch):
    report.v3_floors.cache_clear()
    pooled_floors, per_fold = report.v3_floors(arch)
    assert [round(f['x86_rule']['macro_f1'], 3) for f in per_fold] == V3_X86_RULE[arch]
    assert round(pooled_floors['x86_rule']['macro_f1'], 3) == V3_X86_RULE_POOLED[arch]
    assert pooled_floors['x86_rule']['n'] == (5652 if arch == 'all' else 4850)
    assert set(pooled_floors) == {'majority', 'x86_rule'}


@needs_combined
def test_v3_floor_rows_in_the_report_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'RUNS', tmp_path)
    monkeypatch.setattr(report, 'HERE', tmp_path)
    report.main()
    floors_part = (tmp_path / 'RESULTS.md').read_text(encoding='utf-8').split(
        '## Floors on the same test manifests')[1].split('## AUC-selected')[0]
    for arch in ('all', 'matched'):
        for name in ('x86_rule', 'majority'):
            assert f'| floor: {name} | kfold_v3 | {arch} | 1 |' in floors_part


V3_QUEUE_ARMS = ['gnn-noedges-mblk15', 'gnn-mblk15', 'gnn-shuffle-mblk15']
V3_QUEUE_ORDER = ([('all', f'{v}-ab-selcell') for v in V3_QUEUE_ARMS] + [('all', f'{v}-selcell') for v in V3_QUEUE_ARMS]
                  + [('matched', f'{v}-selcell') for v in V3_QUEUE_ARMS])


@pytest.mark.skipif(BASH is None, reason='needs bash')
def test_queue_v3_dry_names_match_run_configs():
    lines = queue('--stage', 'v3', '--dry')
    runs = [l.split()[1:] for l in lines if l.startswith('run ')]
    assert len(runs) == 45
    for name, *argv in runs:
        cfg, _ = parse_config(argv)
        assert cfg.name == name, (name, cfg.name)
        assert cfg.protocol == 'kfold_v3' and cfg.seed == 0 and cfg.model == 'gnn'
        assert cfg.mask_blocks == 0.15 and cfg.select_on == 'cell' and cfg.encoder is None and not cfg.untyped
        assert cfg.arch_balance == ('-ab-' in name.split('_')[2]) == ('--arch-balance' in argv)
        assert argv.count('--mask-blocks') == 1 and argv.count('--select-on') == 1
    names = [n for n, *_ in runs]
    assert [n.split('_')[0] for n in names[:5]] == [f'v3kfold{f}' for f in range(5)]
    assert [tuple(n.split('_')[1:3]) for n in names[::5]] == V3_QUEUE_ORDER
    arms = {tuple(n.split('_')[1:3]): parse_config(a)[0] for (n, *a) in runs[::5]}
    assert [arms['all', v].control for v in ('gnn-noedges-mblk15-ab-selcell', 'gnn-mblk15-ab-selcell',
                                             'gnn-shuffle-mblk15-ab-selcell')] == ['no_edges', '', 'shuffle']
    assert arms['all', 'gnn-mblk15-selcell'].group == 'kfold_v3_all_gnn-mblk15-selcell'
    assert arms['all', 'gnn-mblk15-ab-selcell'].group == 'kfold_v3_all_gnn-mblk15-ab-selcell'
    assert arms['all', 'gnn-mblk15-ab-selcell'].arch_balance and not arms['all', 'gnn-mblk15-selcell'].arch_balance
    assert arms['matched', 'gnn-shuffle-mblk15-selcell'].control == 'shuffle'
    assert arms['matched', 'gnn-mblk15-selcell'].arch == 'matched' and arms['all', 'gnn-mblk15-selcell'].arch == 'all'
    assert not any(a.arch_balance for (arch, v), a in arms.items() if arch == 'matched')
    head = 'cpu baselines --protocol kfold_v3 --fold 0 1 2 3 4'
    assert lines[:3] == [f'{head} --arch all --seed 0', f'{head} --arch matched --seed 0',
                         f'{head} --arch all --seed 0 --arch-balance']
    assert lines[-3:] == ['cpu report.py', 'cpu gates.py --stage v3 > RUNS/gates_v3.md',
                          'cpu val_threshold.py > RUNS/val_threshold.md']
    assert not any(l.startswith(('skipped', 'stopped')) for l in lines)


@pytest.mark.skipif(BASH is None, reason='needs bash')
def test_queue_v3_budget_guard_and_usage():
    lines = queue('--stage', 'v3', '--dry', GPU_BUDGET_MIN='290', DRY_RUN_MIN='100')
    assert sum(l.startswith('run ') for l in lines) == 3
    assert sum(l.startswith('skipped (budget) ') for l in lines) == 42
    assert lines[-1] == 'cpu val_threshold.py > RUNS/val_threshold.md'
    assert not any(l.startswith('skipped') for l in queue('--stage', 'v3', '--dry'))
    r = subprocess.run([BASH, 'queue_gnn.sh', '--stage', 'v4', '--dry'], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and 'day1|day1b|v3|subgraph' in r.stdout


SG_QUEUE_VARIANTS = ['gnn-mblk15-sg-cap20k', 'gnn-mblk15-sg-cap5k', 'gnn-mblk15-sg-lcc20k', 'gnn-mblk15-sg-lcc5k',
                     'gnn-noedges-mblk15-enc-raw', 'gnn-noedges-mblk15-enc-presence', 'gnn-noedges-mblk15-enc-rawlog']


@pytest.mark.skipif(BASH is None, reason='needs bash')
def test_queue_subgraph_dry_names_match_run_configs():
    lines = queue('--stage', 'subgraph', '--dry')
    runs = [l.split()[1:] for l in lines if l.startswith('run ')]
    assert len(runs) == 35
    for name, *argv in runs:
        cfg, _ = parse_config(argv)
        assert cfg.name == name, (name, cfg.name)
        assert cfg.protocol == 'kfold' and cfg.arch == 'x86' and cfg.seed == 0 and cfg.model == 'gnn'
        assert cfg.mask_blocks == 0.15 and cfg.select_on == 'overall' and not cfg.arch_balance
    names = [n for n, *_ in runs]
    assert [n.split('_')[0] for n in names[:5]] == [f'kfold{f}' for f in range(5)]
    assert [n.split('_')[2] for n in names[::5]] == SG_QUEUE_VARIANTS
    arms = {n.split('_')[2]: parse_config(a)[0] for (n, *a) in runs[::5]}
    assert [(arms[v].subgraph, arms[v].budget, arms[v].control) for v in SG_QUEUE_VARIANTS[:4]] == [
        ('cap', 20000, ''), ('cap', 5000, ''), ('lcc', 20000, ''), ('lcc', 5000, '')]
    assert [(arms[v].encoder, arms[v].control, arms[v].subgraph, arms[v].budget) for v in SG_QUEUE_VARIANTS[4:]] == [
        (e, 'no_edges', 'cap', 0) for e in ('raw', 'presence', 'rawlog')]
    assert lines[0] == 'cpu subgraph_chain.py'
    assert lines[-3:] == ['cpu report.py', 'cpu gates.py --stage subgraph > RUNS/gates_subgraph.md',
                          'cpu val_threshold.py > RUNS/val_threshold.md']
    assert not any(l.startswith(('skipped', 'stopped')) for l in lines)


@pytest.mark.skipif(BASH is None, reason='needs bash')
def test_queue_subgraph_runs_the_chain_measurement_once(tmp_path):
    runs, out = run_stub_queue(tmp_path, 'subgraph')
    calls = (runs / 'calls.txt').read_text().splitlines()
    assert [c.split()[0] for c in calls].count('subgraph_chain.py') == 1
    assert sum(c.startswith('train.py') for c in calls) == 35 and any(l.startswith('start subgraph_chain') for l in out)
    (runs / 'subgraph_chain.json').write_text('{}')
    again = subprocess.run([BASH, 'queue_gnn.sh', '--stage', 'subgraph'], cwd=ROOT, capture_output=True, text=True,
                           env={**os.environ, 'GT_PY': (tmp_path / 'stub.sh').as_posix(), 'GT_RUNS': runs.as_posix(),
                                'GT_TRAIN_PATTERN': 'gtqueue_never_matches_train[.]py', 'FAIL_ONCE_FOLD': 'none',
                                'FAIL_ALWAYS_FOLD': 'none'})
    assert 'skip subgraph_chain' in again.stdout.splitlines() and 'start subgraph_chain' not in again.stdout


CACHE_STUB = """#!/bin/bash
echo "$1 GT_CACHE=$GT_CACHE" >> "$GT_RUNS/calls.txt"
case $1 in
  train.py) n=$(grep -c '^train.py' "$GT_RUNS/calls.txt"); [ "$n" = "$STOP_AFTER" ] && touch "$GT_RUNS/STOP";;
esac
exit 0
"""


@pytest.mark.skipif(BASH is None, reason='needs bash')
def test_queue_v3_exports_the_combined_cache_and_keeps_stop_and_skip(tmp_path):
    """A stub interpreter records GT_CACHE for every call, whatever the caller exported. Its 2nd train.py call creates
    RUNS/STOP, so one run finishes after it; baselines with all five thresholds and finished runs are skipped."""
    runs = tmp_path / 'runs'
    (runs / 'logs').mkdir(parents=True)
    stub = tmp_path / 'stub.sh'
    stub.write_text(CACHE_STUB, newline='\n')
    stub.chmod(0o755)
    for fold in range(5):
        d = runs / 'baselines' / f'v3kfold{fold}_all_s0'
        d.mkdir(parents=True)
        (d / 'metrics.json').write_text('{"methods": {"m": {"threshold_val": 0.4}}}')
    for fold in range(2):
        d = runs / f'v3kfold{fold}_all_gnn-noedges-mblk15-ab-selcell_s0'
        d.mkdir()
        (d / 'metrics.json').write_text('{}')
    env = dict(GT_PY=stub.as_posix(), GT_RUNS=runs.as_posix(), GT_TRAIN_PATTERN='gtqueue_never_matches_train[.]py',
               STOP_AFTER='2', GT_CACHE='C:/somewhere/else')
    r = subprocess.run([BASH, 'queue_gnn.sh', '--stage', 'v3'], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, **env})
    assert r.returncode == 0, r.stderr + r.stdout
    out = r.stdout.splitlines()
    assert 'skip baselines v3 all' in out and any(l.startswith('start baselines v3 matched') for l in out)
    assert [l.split()[1] for l in out if l.startswith('start v3kfold')] == [
        f'v3kfold{f}_all_gnn-noedges-mblk15-ab-selcell_s0' for f in (2, 3)]
    assert [l for l in out if l.startswith('stopped before ')][0] == 'stopped before v3kfold4_all_gnn-noedges-mblk15-ab-selcell_s0'
    calls = (runs / 'calls.txt').read_text().splitlines()
    combined = 'GT_CACHE=C:/Users/chaoa/Downloads/graph_transformer_cache_combined'
    assert {c.split(' ', 1)[0] for c in calls} == {'-c', 'baselines.py', 'train.py', 'report.py', 'gates.py',
                                                   'val_threshold.py'}
    assert all(c.endswith(combined) for c in calls), calls
    assert (runs / 'gates_v3.md').exists() and out[-1].startswith('queue done')


# step 6b: truncation columns, the v3 truncation line, queue env and retry

def test_write_predictions_adds_trailing_block_columns_and_readers_still_work(tmp_path):
    from training import train
    rows = [{'sample_id': s, 'label': y, 'arch': a, 'family': 'f', 'fold': 1}
            for s, y, a in (('a', 1, 'x86'), ('b', 0, 'x64'), ('c', 1, 'x64'))]
    by_split = {'val': (['a'], np.array([0.9]), [(5, 10, 10)]),
                'test': (['b', 'c'], np.array([0.2, 0.7]), [(7, 400, 900), (3, None, None)])}
    path = tmp_path / 'predictions.csv'
    train.write_predictions(path, rows, by_split, 0.5)
    with path.open(newline='') as f:
        read = list(csv.DictReader(f))
    assert list(read[0]) == ['sample_id', 'split', 'label', 'prob', 'pred', 'arch', 'family', 'fold',
                             'n_segments_total', 'n_blocks_kept', 'n_blocks_total']
    assert [(r['n_segments_total'], r['n_blocks_kept'], r['n_blocks_total']) for r in read] == [
        ('5', '10', '10'), ('7', '400', '900'), ('3', '', '')]
    assert train.truncated_count(by_split['test'][2]) == 1 and train.truncated_count([(1, 5, 5)]) == 0
    assert [stats.truncation(r) for r in read] == [0, 1, -1]
    assert len(report.read_predictions(path)) == 2
    from evaluation import val_threshold
    assert val_threshold.load(tmp_path)[2].tolist() == [0, 1]


def test_stats_pool_reads_truncation_by_column_name():
    rows = [{'sample_id': 'a', 'label': '1', 'prob': '0.9', 'pred': '1', 'family': 'f', 'arch': 'x86'},
            {'sample_id': 'b', 'label': '0', 'prob': '0.1', 'pred': '0', 'family': 'g', 'arch': 'x64',
             'n_blocks_kept': '5', 'n_blocks_total': '9'}]
    assert stats.pool({0: rows}).truncated.tolist() == [-1, 1]
    old = stats.pool({0: rows[:1]})
    assert old.truncated.tolist() == [-1] and old.ids.tolist() == ['a']


def write_truncated_v3_run(root, arch, truncated_ids):
    for fold in range(5):
        d = write_v3_run(root, arch, fold, 1.5, **V3_MASKED)
        for fname in ('predictions.csv', 'predictions_auc.csv'):
            with (d / fname).open(newline='') as f:
                rows = list(csv.DictReader(f))
            with (d / fname).open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=[*rows[0], 'n_blocks_kept', 'n_blocks_total'])
                w.writeheader()
                for r in rows:
                    cut = r['sample_id'] in truncated_ids
                    w.writerow({**r, 'n_blocks_kept': 100 if cut else 50, 'n_blocks_total': 900 if cut else 50})


def test_v3_gates_truncation_line(tmp_path, monkeypatch):
    build_v3_fake(tmp_path, monkeypatch)
    for d in tmp_path.glob('v3kfold*_matched_gnn-mblk15-selcell_s0'):
        shutil.rmtree(d)
    cut = {f'f{fold}_{i}' for fold in range(5) for i in range(0, 120, 10)}
    write_truncated_v3_run(tmp_path, 'matched', cut)
    gates.v3_cell.cache_clear()
    text = '\n'.join(gates.v3(30))
    lines = {l.split(':')[0]: l for l in text.splitlines() if 'truncat' in l and l.startswith('- Arch')}
    assert 'truncation not assessed, R7 predictions lack n_blocks_kept' in lines['- Arch all']
    line = lines['- Arch matched']
    assert f'R7 truncated {5 * 12} of 600 test binaries' in line and 'cap 500000' in line
    assert 'truncated: R recall' in line and '; untruncated: R recall' in line
    p = stats.load_group('kfold_v3_matched_gnn-mblk15-selcell', 'auc', ('kfold_v3', 'matched'))[0][0]
    m = p.truncated == 1
    assert m.sum() == 60 and f"G recall {np.mean(p.pred[m & (p.y == 0)] == 0):.3f} (n={int((m & (p.y == 0)).sum())})" in line


STAGE_STUB = """#!/bin/bash
echo "$* ALLOC=$PYTORCH_CUDA_ALLOC_CONF" >> "$GT_RUNS/calls.txt"
if [ "$1" = train.py ]; then
  case " $* " in *" --fold $FAIL_ALWAYS_FOLD "*) exit 1;; esac
  case " $* " in *" --fold $FAIL_ONCE_FOLD "*) [ -f "$GT_RUNS/failed_once" ] || { touch "$GT_RUNS/failed_once"; exit 1; };; esac
fi
exit 0
"""


def run_stub_queue(tmp_path, stage, **env):
    runs = tmp_path / 'runs'
    (runs / 'logs').mkdir(parents=True)
    stub = tmp_path / 'stub.sh'
    stub.write_text(STAGE_STUB.replace('grep -c "^train.py.*fold $FAIL_ONCE_FOLD "', 'grep -c "^train.py.*--fold $FAIL_ONCE_FOLD "'),
                    newline='\n')
    stub.chmod(0o755)
    e = {**os.environ, 'GT_PY': stub.as_posix(), 'GT_RUNS': runs.as_posix(),
         'GT_TRAIN_PATTERN': 'gtqueue_never_matches_train[.]py', 'FAIL_ONCE_FOLD': 'none', 'FAIL_ALWAYS_FOLD': 'none',
         **env}
    r = subprocess.run([BASH, 'queue_gnn.sh', '--stage', stage], cwd=ROOT, capture_output=True, text=True, env=e)
    assert r.returncode == 0, r.stderr + r.stdout
    return runs, r.stdout.splitlines()


@pytest.mark.skipif(BASH is None, reason='needs bash')
@pytest.mark.parametrize('stage', ['day1', 'day1b', 'v3', 'subgraph'])
def test_queue_sets_expandable_segments_for_every_python_call(tmp_path, stage):
    runs, out = run_stub_queue(tmp_path, stage, PYTORCH_CUDA_ALLOC_CONF='something_else')
    calls = (runs / 'calls.txt').read_text().splitlines()
    assert calls and all(c.endswith('ALLOC=expandable_segments:True') for c in calls), calls
    assert any(c.startswith('train.py') for c in calls) and not any(l.startswith('retry') for l in out)


@pytest.mark.skipif(BASH is None, reason='needs bash')
def test_queue_retries_a_failed_run_once(tmp_path):
    runs, out = run_stub_queue(tmp_path, 'v3', FAIL_ONCE_FOLD='2')
    calls = (runs / 'calls.txt').read_text().splitlines()
    name = 'v3kfold2_all_gnn-noedges-mblk15-ab-selcell_s0'
    assert [l for l in out if l.startswith('retry')][0] == f'retry {name}' and out.count(f'retry {name}') == 1
    i = out.index(f'retry {name}')
    assert out[i - 1].startswith(f'start {name}') and out[i + 1].startswith(f'finished {name} rc=0')
    assert sum(c.startswith('train.py') and '--fold 2 --arch all' in c and '--control no_edges' in c
               and '--arch-balance' in c for c in calls) == 2
    assert (runs / 'logs' / f'{name}.log.try1').exists() and (runs / 'logs' / f'{name}.log').exists()
    assert out[-1].startswith('queue done')


@pytest.mark.skipif(BASH is None, reason='needs bash')
def test_queue_retry_is_a_single_attempt_and_the_queue_goes_on(tmp_path):
    runs, out = run_stub_queue(tmp_path, 'day1b', FAIL_ALWAYS_FOLD='1')
    retries = [l for l in out if l.startswith('retry')]
    assert len(retries) == 13 and all(l.split()[1].startswith('kfold1_x86_') for l in retries)
    bad = [l for l in out if l.startswith('finished kfold1_')]
    assert len(bad) == 13 and all(' rc=1 ' in l for l in bad)
    calls = (runs / 'calls.txt').read_text().splitlines()
    assert sum(c.startswith('train.py') for c in calls) == 65 + 13
    assert sum(l.startswith('finished kfold') and ' rc=0 ' in l for l in out) == 52 and out[-1].startswith('queue done')


@pytest.mark.skipif(BASH is None, reason='needs bash')
def test_queue_does_not_retry_after_stop(tmp_path):
    stub = STAGE_STUB.replace('exit 0\n', 'exit 0\n')
    runs = tmp_path / 'runs'
    (runs / 'logs').mkdir(parents=True)
    killer = tmp_path / 'killer.sh'
    killer.write_text('#!/bin/bash\necho "$1" >> "$GT_RUNS/calls.txt"\n'
                      '[ "$1" = train.py ] && { touch "$GT_RUNS/STOP"; exit 1; }\nexit 0\n', newline='\n')
    killer.chmod(0o755)
    r = subprocess.run([BASH, 'queue_gnn.sh', '--stage', 'day1b'], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, 'GT_PY': killer.as_posix(), 'GT_RUNS': runs.as_posix(),
                            'GT_TRAIN_PATTERN': 'gtqueue_never_matches_train[.]py'})
    assert r.returncode == 0, r.stderr
    assert 'retry' not in r.stdout and (runs / 'calls.txt').read_text().count('train.py') == 1


# step 7: arch-balanced training weights and cell-balanced metrics

CELLS = [(1, 'x86'), (1, 'x64'), (0, 'x86'), (0, 'x64')]


def cell_sample(counts):
    """y, arch of a synthetic set with counts[i] samples in CELLS[i]."""
    y, arch = [], []
    for (label, a), n in zip(CELLS, counts):
        y += [label] * n
        arch += [a] * n
    return np.array(y), np.array(arch)


def test_cell_weights_on_a_synthetic_manifest():
    y, arch = cell_sample([60, 10, 20, 30])
    w = cell_weights(y, arch)
    assert len(w) == 120 and w.mean() == pytest.approx(1.0)
    assert w[0] == pytest.approx(120 / (4 * 60)) and w[60] == pytest.approx(120 / (4 * 10))
    assert w[70] == pytest.approx(120 / (4 * 20)) and w[90] == pytest.approx(120 / (4 * 30))
    for cell in CELLS:
        mask = (y == cell[0]) & (arch == cell[1])
        assert w[mask].sum() == pytest.approx(120 / 4)
    y, arch = cell_sample([60, 0, 20, 30])
    w = cell_weights(y, arch)
    assert w.mean() == pytest.approx(1.0) and w[0] == pytest.approx(110 / (3 * 60))
    y, arch = cell_sample([5, 0, 3, 0])
    assert cell_weights(y, arch) == pytest.approx([8 / (2 * 5)] * 5 + [8 / (2 * 3)] * 3)
    assert len(cell_weights([], [])) == 0


def test_loss_weights_arch_balance_and_default():
    import torch
    cpu = torch.device('cpu')
    y, arch = cell_sample([60, 10, 20, 30])
    rows = [{'sample_id': f's{i}', 'label': int(l), 'arch': a} for i, (l, a) in enumerate(zip(y, arch))]
    batch = {'sample_id': [r['sample_id'] for r in rows], 'label': torch.tensor(y, dtype=torch.float32)}
    ab = loss_weight_fn(gnn(arch_balance=True), rows, cpu)(batch).numpy()
    assert ab == pytest.approx(cell_weights(y, arch)) and ab.mean() == pytest.approx(1.0)
    plain = loss_weight_fn(gnn(), rows, cpu)(batch).numpy()
    n_pos = int(y.sum())
    assert plain == pytest.approx(np.where(y == 1, 120 / (2 * n_pos), 120 / (2 * (120 - n_pos))), abs=0)
    assert plain.mean() == pytest.approx(1.0) and not np.allclose(plain, ab)
    flipped = {'sample_id': batch['sample_id'][::-1], 'label': batch['label'].flip(0)}
    assert loss_weight_fn(gnn(arch_balance=True), rows, cpu)(flipped).numpy() == pytest.approx(ab[::-1])
    x86 = [r for r in rows if r['arch'] == 'x86']
    sub = {'sample_id': [r['sample_id'] for r in x86], 'label': torch.tensor([r['label'] for r in x86]).float()}
    assert (loss_weight_fn(gnn(arch_balance=True), x86, cpu)(sub).numpy()
            == pytest.approx(loss_weight_fn(gnn(), x86, cpu)(sub).numpy()))


@needs_combined
def test_cell_weights_on_a_real_v3_manifest():
    from common.manifests import load_index, make_manifest
    m = make_manifest(RunConfig(protocol='kfold_v3', fold=0, arch='all'), load_index(COMBINED))
    y, arch = np.array([r['label'] for r in m.train]), np.array([r['arch'] for r in m.train])
    w = cell_weights(y, arch)
    assert w.mean() == pytest.approx(1.0)
    for label, a in CELLS:
        assert w[(y == label) & (arch == a)].sum() == pytest.approx(len(y) / 4)


def test_cell_balanced_x86_rule_is_exactly_half():
    y, arch = cell_sample([60, 10, 20, 30])
    pred = (arch == 'x86').astype(int)
    cb = cell_balanced(y, pred.astype(float), pred, arch)
    assert cb['macro_f1'] == pytest.approx(0.5) and cb['accuracy'] == pytest.approx(0.5)
    assert cb['auc'] == pytest.approx(0.5) and cb['n_cells'] == 4
    assert [cb[f'n_{t}_{a}'] for t in 'RG' for a in ARCHES] == [60, 10, 20, 30]
    ordinary = scores(y, pred.astype(float), pred, arch)
    assert ordinary['accuracy'] != pytest.approx(0.5, abs=0.05) and ordinary['cell_balanced'] == cb
    fl = floors([0, 1], y, arch)
    assert fl['x86_rule']['cell_balanced']['macro_f1'] == pytest.approx(0.5)
    assert fl['x86_rule']['cell_balanced']['accuracy'] == pytest.approx(0.5)
    assert fl['majority']['cell_balanced']['accuracy'] == pytest.approx(0.5)
    assert fl['majority']['cell_balanced']['auc'] == pytest.approx(0.5)


def test_cell_balanced_hand_computed_one_inverted_cell():
    """Perfect on R x86, R x64 and G x86, inverted on G x64; every cell weighs 1/4 whatever its size. Positives: weight
    0.5 at prob 0.9; negatives: 0.25 at 0.1 and 0.25 at 0.9. Class 1: tp 0.5, fp 0.25, fn 0 -> F1 0.8. Class 0: tp 0.25,
    fp 0, fn 0.25 -> F1 2/3. AUC (0.5 * 0.25 + 0.5 * 0.25 / 2) / (0.5 * 0.5) = 0.75. AP: one step at recall 1 with
    precision 0.5 / 0.75."""
    y, arch = cell_sample([30, 50, 70, 20])
    pred = y.copy()
    pred[(y == 0) & (arch == 'x64')] = 1
    prob = np.where(pred == 1, 0.9, 0.1)
    cb = cell_balanced(y, prob, pred, arch)
    assert cb['accuracy'] == pytest.approx(0.75)
    assert cb['macro_f1'] == pytest.approx((0.8 + 2 / 3) / 2)
    assert cb['auc'] == pytest.approx(0.75)
    assert cb['pr_auc'] == pytest.approx(2 / 3)
    plain = scores(y, prob, pred, arch)
    assert plain['accuracy'] == pytest.approx(1 - 20 / 170)


def test_cell_balanced_with_an_empty_cell_and_one_class():
    y, arch = cell_sample([40, 0, 10, 0])
    pred = np.concatenate([np.ones(40, int), np.zeros(5, int), np.ones(5, int)])
    cb = cell_balanced(y, np.where(pred == 1, 0.8, 0.2), pred, arch)
    assert cb['n_cells'] == 2 and cb['n_R_x64'] == 0 and cb['n_G_x64'] == 0 and cb['n_R_x86'] == 40
    assert cb['n_G_x86'] == 10 and cb['accuracy'] == pytest.approx(0.75)
    y1, arch1 = cell_sample([5, 5, 0, 0])
    one = cell_balanced(y1, np.full(10, 0.5), np.ones(10, int), arch1)
    assert np.isnan(one['auc']) and np.isnan(one['pr_auc']) and one['n_cells'] == 2 and one['accuracy'] == 1.0
    s = scores([1, 1, 1], [0.2, 0.9, 0.5], [0, 1, 1], ['x86'] * 3)
    assert s['cell_balanced']['n_cells'] == 1 and np.isnan(s['cell_balanced']['auc'])
    json.dumps(s)


def arch_preds(n=400, seed=0, signal=1.0):
    p = preds(n=n, seed=seed, signal=signal)
    return stats.Preds(p.ids, p.y, p.prob, p.pred, p.cluster, np.where(np.arange(n) % 3 == 0, 'x64', 'x86'))


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
    r = subprocess.run([sys.executable, 'stats.py', '--help'], cwd=ROOT, capture_output=True, text=True)
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


def test_report_labels_for_arch_balance():
    assert report.model_label(gnn(arch_balance=True)) == 'edge-typed GNN arch-bal'
    assert report.model_label(gnn(control='shuffle', arch_balance=True)) == 'GNN shuffle arch-bal'
    assert report.model_label(RunConfig(arch_balance=True)) == 'graph transformer arch-bal'
    assert report.model_label(gnn(arch_balance=True, encoder='mean')) == 'edge-typed GNN arch-bal enc-mean'
    assert 'arch_balance' in report.LABELLED
    overrides = {k for k, v in report.asdict(gnn(arch_balance=True)).items()
                 if k not in set(report.LABELLED) and v != report.DEFAULTS[k]}
    assert not overrides


def build_v3_ab_fake(tmp_path, monkeypatch):
    build_v3_fake(tmp_path, monkeypatch)
    for fold in range(5):
        write_v3_run(tmp_path, 'all', fold, 1.9, arch_balance=True, **V3_MASKED)
        write_v3_run(tmp_path, 'all', fold, 1.3, arch_balance=True, control='no_edges', **V3_MASKED)
        write_v3_run(tmp_path, 'all', fold, 1.1, arch_balance=True, control='shuffle', **V3_MASKED)
        write_v3_baseline(tmp_path, 'all', fold, 1.5, suffix='_ab')


def test_gates_v3_cell_balanced_columns_ab_arms_and_twins(tmp_path, monkeypatch):
    build_v3_ab_fake(tmp_path, monkeypatch)
    text = '\n'.join(gates.v3(50))
    all_part = text.split('## Arch matched: pooled table')[0]
    row = {l.split(' | ')[0].lstrip('| '): l for l in all_part.splitlines() if l.startswith('| ')}
    for arm in ('R7 typed', 'R7 typed ab', 'R2 no-edges ab', 'R4 shuffle ab', 'R0 TF-IDF ab'):
        cells = row[arm].split(' | ')
        assert len(cells) == 16 and 'missing' not in row[arm], arm
        assert all(0 <= float(c.strip(' |')) <= 1 for c in cells[2:])
    assert 'R5 untyped' not in text
    got = gates.v3_cell('kfold_v3_all_gnn-mblk15-selcell', 'f1', 'all')
    q = stats.load_group('kfold_v3_all_gnn-mblk15-selcell', 'f1', ('kfold_v3', 'all'))[0][0]
    want = cell_balanced(q.y, q.prob, q.pred, q.arch)
    assert got[7] == pytest.approx(want['auc']) and got[8] == pytest.approx(want['macro_f1'])
    assert '### Arch all: paired cell-balanced differences, second minus first' in all_part
    assert 'Cell-balanced: every test sample weighted' in all_part
    twin = all_part.split('each arch-balanced arm minus its unweighted twin')[1]
    assert 'kfold_v3_all_gnn-mblk15-ab-selcell minus kfold_v3_all_gnn-mblk15-selcell' in twin
    assert 'tfidf-ab minus tfidf' in twin
    assert 'kfold_v3_all_gnn-noedges-mblk15-ab-selcell minus kfold_v3_all_gnn-noedges-mblk15-selcell' in twin
    ab_pairs = all_part.split('arch-balanced arms, paired cell-balanced differences')[1].split('### Arch all: each')[0]
    assert 'kfold_v3_all_gnn-mblk15-ab-selcell minus kfold_v3_all_gnn-noedges-mblk15-ab-selcell' in ab_pairs
    assert 'kfold_v3_all_gnn-noedges-mblk15-ab-selcell minus tfidf-ab' in ab_pairs
    plain_ab = all_part.split('arch-balanced arms, paired differences, second minus first')[1].split('### Arch all: arch')[0]
    assert 'kfold_v3_all_gnn-mblk15-ab-selcell minus kfold_v3_all_gnn-shuffle-mblk15-ab-selcell' in plain_ab
    assert 'Cell-balanced: every' not in plain_ab
    verdicts = [l for l in text.split('## Verdict')[1].splitlines() if l.startswith('- Arch all')]
    assert len(verdicts) == 4 and verdicts[3].startswith('- Arch all: cell-balanced: R7 ab beats both R2 and R4')
    assert verdicts[2].startswith('- Arch all: R7 ab beats both R2 and R4')


def test_report_cell_balanced_table_and_ab_rows(tmp_path, monkeypatch):
    build_v3_ab_fake(tmp_path, monkeypatch)
    labels = {sid: y for fold in range(5) for sid, y, *_ in fake_fold_rows(fold, 0, 1.0)}
    for fold in range(5):
        labels.update({f'tr{fold}_{i}': i % 2 for i in range(10)})
    results, _ = report.collect(labels)
    keys = {k for k, _ in results}
    assert ('kfold_v3', 'all', 'edge-typed GNN mask-blk 15% arch-bal cell-select') in keys
    assert ('kfold_v3', 'all', 'edge-typed GNN mask-blk 15% cell-select') in keys
    assert ('kfold_v3', 'all', 'tfidf_sig_api_lr arch-bal') in keys and ('kfold_v3', 'all', 'tfidf_sig_api_lr') in keys
    floor = {('kfold_v3', 'all', 'floor: x86_rule'): {0: (floors([0, 1], *cell_sample([60, 10, 20, 30]))['x86_rule'],
                                                         {}, {})}}
    lines = report.cell_table(results + sorted(floor.items()))
    assert lines[0].startswith('| config | arch | seeds | Acc | macro-F1 | AUC | PR-AUC | cells |')
    row = next(l for l in lines if l.startswith('| edge-typed GNN mask-blk 15% arch-bal cell-select | all | 1 |'))
    assert row.split(' | ')[-5] == '4' and not any('bundle' in l for l in lines)
    assert sum(l.startswith('| edge-typed GNN') for l in lines) == 3


@needs_combined
def test_report_main_writes_the_cell_balanced_section(tmp_path, monkeypatch):
    build_v3_ab_fake(tmp_path, monkeypatch)
    monkeypatch.setattr(report, 'HERE', tmp_path)
    labels = {f'tr{fold}_{i}': i % 2 for fold in range(5) for i in range(10)}
    monkeypatch.setattr(report, 'all_labels', lambda: labels)
    report.main()
    text = (tmp_path / 'RESULTS.md').read_text(encoding='utf-8')
    head, section = text.split('## Cell-balanced (each class x arch weighted equally)')
    for title in ('## Models and baselines', '## Floors on the same test manifests', '## AUC-selected checkpoint'):
        assert title in head
    assert '### F1 checkpoint at 0.5' in section and '### AUC checkpoint at the validation threshold' in section
    assert '| edge-typed GNN mask-blk 15% arch-bal cell-select | all | 1 |' in section and '| tfidf_sig_api_lr arch-bal | all | 1 |' in section
    assert section.count('| floor: x86_rule | all | 1 |') == 2 and section.count('| floor: majority | matched | 1 |') == 2
    assert 'bundle' not in section


def test_baselines_arch_balance_matches_class_weight_on_one_architecture(tmp_path):
    """On an x86-only manifest the only cells are the two classes, so N / (C n_cell) is the 'balanced' class weight:
    the arch-balanced fit must reproduce the default fit, and the output goes to a `_ab` directory."""
    from evaluation import baselines
    sizes = {k: len(v) for k, v in json.loads((baselines.CACHE / 'vocab.json').read_text(encoding='utf-8')).items()}
    baselines.run(RunConfig(protocol='kfold', fold=0, seed=0), sizes, tmp_path)
    baselines.run(RunConfig(protocol='kfold', fold=0, seed=0, arch_balance=True), sizes, tmp_path)
    plain, ab = tmp_path / 'kfold0_x86_s0', tmp_path / 'kfold0_x86_s0_ab'
    m, mab = (json.loads((d / 'metrics.json').read_text()) for d in (plain, ab))
    assert m['arch_balance'] is False and mab['arch_balance'] is True
    for name, v in mab['methods'].items():
        for key in ('test', 'test_at_val_threshold'):
            cb = v[key]['cell_balanced']
            assert cb['n_cells'] == 2 and cb['n_R_x64'] == 0 and cb['n_G_x64'] == 0 and cb['n_R_x86'] > 0
        assert v['test']['auc'] == pytest.approx(m['methods'][name]['test']['auc'], abs=1e-9)
    assert 'cell_balanced' in m['floors']['x86_rule'] and 'cell_balanced' in mab['floors']['majority']
    assert report.read_predictions(ab / 'predictions.csv', method='tfidf_sig_api_lr')


# step 11: cell-balanced selection, subgraph and weighting arms in the gates, the chain measurement, the v3 smoke

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
SELECT_CELLS = [(1, 'x86', 200), (1, 'x64', 20), (0, 'x86', 20), (0, 'x64', 200)]


def selection_fixture():
    """A validation set where architecture predicts the label (ransomware mostly x86, goodware mostly x64), and two
    epochs: epoch 1 scores by architecture only, epoch 2 by label only."""
    rows = [{'sample_id': f'{y}{a}{i}', 'label': y, 'arch': a} for y, a, n in SELECT_CELLS for i in range(n)]
    ids = [r['sample_id'] for r in rows]
    y = np.array([r['label'] for r in rows])
    arch = np.array([r['arch'] for r in rows])
    rng = np.random.default_rng(0)
    by_arch = np.clip(np.where(arch == 'x86', 0.8, 0.2) + rng.normal(0, 0.05, len(y)), 0.01, 0.99)
    by_label = np.clip(np.where(y == 1, 0.65, 0.35) + rng.normal(0, 0.2, len(y)), 0.01, 0.99)
    return rows, ids, y, arch, [by_arch, by_label]


def test_select_on_cell_picks_other_epochs_and_thresholds_than_overall():
    from training.train import evaluate, val_auc_of
    rows, ids, y, arch, epochs = selection_fixture()
    vals = [evaluate(rows, ids, p, 0.5) for p in epochs]
    pick = lambda select_on: int(np.argmax([val_auc_of(gnn(select_on=select_on), v) for v in vals]))
    assert [v['auc'] for v in vals] == sorted([v['auc'] for v in vals], reverse=True)
    assert vals[0]['cell_balanced']['auc'] < 0.6 < vals[1]['cell_balanced']['auc']
    assert pick('overall') == 0 and pick('cell') == 1
    assert val_auc_of(gnn(), vals[0]) == vals[0]['auc'] and val_auc_of(gnn(select_on='cell'), vals[0]) == (
        vals[0]['cell_balanced']['auc'])
    thr = {s: val_threshold_of(rows, ids, epochs[pick(s)], s) for s in ('overall', 'cell')}
    assert thr['overall'] != thr['cell'] and all(t in GRID for t in thr.values())
    # on the same probabilities the two rules also disagree
    assert val_threshold_of(rows, ids, epochs[0], 'overall') != val_threshold_of(rows, ids, epochs[0], 'cell')
    assert val_threshold_of(rows, ids, epochs[0]) == val_threshold_of(rows, ids, epochs[0], 'overall')


def test_cell_threshold_is_the_weighted_macro_f1_maximiser():
    rows, ids, y, arch, epochs = selection_fixture()
    w = cell_weights(y, arch)
    for p in epochs:
        want = max(GRID, key=lambda g: f1_score(y, (p >= g).astype(int), average='macro', sample_weight=w))
        assert val_threshold_of(rows, ids, p, 'cell') == want
        assert val_threshold_of(rows, ids, p, 'cell') == max(
            GRID, key=lambda g: cell_balanced(y, p, (p >= g).astype(int), arch)['macro_f1'])


def tiny_manifest():
    """Small real binaries of the v2 cache, 5 / 4 / 3 per (label, arch) cell for train / val / test."""
    from common.manifests import Manifest, load_index
    small = sorted((r for r in load_index() if int(r['num_blocks']) <= 1200), key=lambda r: r['sample_id'])
    train, val, test = [], [], []
    for label in (1, 0):
        for arch in ('x86', 'x64'):
            cell = [r for r in small if r['label'] == label and r['arch'] == arch]
            assert len(cell) >= 12, (label, arch, len(cell))
            train, val, test = train + cell[:5], val + cell[5:9], test + cell[9:12]
    return Manifest(train, val, test, []), small


@needs_cuda
@pytest.mark.parametrize('select_on', ['overall', 'cell'])
def test_run_records_and_applies_the_selection_rule(tmp_path, monkeypatch, select_on):
    """Two epochs of a d_model 16 GNN on 48 tiny binaries (a unit test of train.run, not a training run)."""
    from common.manifests import Manifest
    from training import train
    manifest, _ = tiny_manifest()
    monkeypatch.setattr(train, 'RUNS', tmp_path)
    monkeypatch.setattr(train, 'make_manifest', lambda cfg, index: Manifest(**vars(manifest)))
    cfg = gnn(d_model=16, fixed_epochs=3, batch_size=4, num_workers=0, select_on=select_on, mask_blocks=0.15)
    metrics = train.run(cfg)
    out = tmp_path / cfg.name
    assert cfg.name.endswith('-selcell_s0') == (select_on == 'cell')
    assert metrics['select_on'] == select_on and metrics['selection']['auc']['rule'] == train.AUC_RULES[select_on]
    assert json.loads((out / 'metrics.json').read_text())['select_on'] == select_on
    by_id = {r['sample_id']: r for r in manifest.val}
    epochs = {}
    for r in csv.DictReader((out / 'val_probs.csv').open()):
        epochs.setdefault(int(r['epoch']), {})[r['sample_id']] = float(r['prob'])
    score = lambda probs: train.evaluate(manifest.val, list(probs), np.array(list(probs.values())), 0.5)
    vals = {e: score(p) for e, p in epochs.items()}
    want = max(vals, key=lambda e: (train.val_auc_of(cfg, vals[e]), -e))
    assert metrics['selection']['auc']['best_epoch'] == want
    assert metrics['selection']['auc']['val_auc'] == pytest.approx(train.val_auc_of(cfg, vals[want]))
    val_rows = [r for r in csv.DictReader((out / 'predictions_auc.csv').open()) if r['split'] == 'val']
    ids, probs = [r['sample_id'] for r in val_rows], np.array([float(r['prob']) for r in val_rows])
    assert metrics['selection']['auc']['threshold_val'] == val_threshold_of(manifest.val, ids, probs, select_on)
    f1_rows = [r for r in csv.DictReader((out / 'predictions.csv').open()) if r['split'] == 'val']
    assert metrics['selection']['f1']['threshold_val'] == val_threshold_of(
        manifest.val, [r['sample_id'] for r in f1_rows], np.array([float(r['prob']) for r in f1_rows]), 'overall')
    assert metrics['best_epoch'] == metrics['selection']['f1']['best_epoch']


def test_model_labels_for_step11_fields():
    assert report.model_label(gnn(select_on='cell')) == 'edge-typed GNN cell-select'
    assert report.model_label(gnn(select_on='cell', arch_balance=True)) == 'edge-typed GNN arch-bal cell-select'
    assert report.model_label(RunConfig(select_on='cell')) == 'graph transformer cell-select'
    assert report.model_label(gnn(subgraph='lcc', budget=20000)) == 'edge-typed GNN lcc 20k'
    assert report.model_label(gnn(budget=5000)) == 'edge-typed GNN cap 5k'
    assert report.model_label(gnn(subgraph='lcc')) == 'edge-typed GNN lcc full budget'
    assert report.model_label(gnn(control='no_edges', encoder='presence')) == 'GNN no-edges enc-presence'
    assert report.model_label(gnn()) == 'edge-typed GNN'
    for kw in (dict(select_on='cell'), dict(subgraph='lcc', budget=5000), dict(budget=20000)):
        assert not {k for k in kw} & {k for k in report.DEFAULTS if k not in report.LABELLED}


SG_REF = ('mask_blocks', 0.15)


def build_subgraph_fake(root, monkeypatch, with_chain=None):
    """kfold x86 seed 0: the whole-graph reference, cap 20k (as good), cap 5k (clearly worse), lcc 20k (missing fold 4),
    no lcc 5k; the no-edge reference and its raw and presence weightings (rawlog missing)."""
    monkeypatch.setattr(stats, 'RUNS', root)
    gates.pooled.cache_clear()
    plan = {(SG_REF,): (1.6, 5), (SG_REF, ('subgraph', 'cap'), ('budget', 20000)): (1.6, 5),
            (SG_REF, ('subgraph', 'cap'), ('budget', 5000)): (0.3, 5),
            (SG_REF, ('subgraph', 'lcc'), ('budget', 20000)): (1.6, 4),
            (SG_REF, ('control', 'no_edges')): (1.4, 5),
            (SG_REF, ('control', 'no_edges'), ('encoder', 'raw')): (1.4, 5),
            (SG_REF, ('control', 'no_edges'), ('encoder', 'presence')): (0.5, 5)}
    build_fake(root, {kw: (signal, range(folds)) for kw, (signal, folds) in plan.items()})
    if with_chain is not None:
        (root / 'subgraph_chain.json').write_text(json.dumps(with_chain))


CHAIN = {'pool': 'kfold x86', 'n_binaries': 1800, 'n_blocks': 10_000_000, 'n_chain_blocks': 41_000,
         'reduction': 0.0041, 'mean_binary_reduction': 0.005, 'max_binary_reduction': 0.09}


def test_subgraph_gates_from_disk(tmp_path, monkeypatch):
    build_subgraph_fake(tmp_path, monkeypatch, CHAIN)
    text = '\n'.join(gates.subgraph(100))
    assert 'stage subgraph' in text and gates.LABEL in text
    noninf = text.split('## Subgraph policies')[1].split('## Chain merge')[0]
    ni = {l.split(' | ')[0].lstrip('| '): l for l in noninf.splitlines() if l.startswith('| R7')}
    assert ni['R7 typed, cap 20k'].endswith('| yes |') and ni['R7 typed, cap 5k'].endswith('| no |')
    assert 'missing' in ni['R7 typed, lcc 20k'] and 'missing' in ni['R7 typed, lcc 5k']
    assert 'kfold_x86_gnn-mblk15-sg-cap5k' in ni['R7 typed, cap 5k'] and 'kfold_x86_gnn-mblk15 ' in noninf
    res, _ = stats.compare('kfold_x86_gnn-mblk15', 'kfold_x86_gnn-mblk15-sg-cap20k', 'auc', 100, seeds=[0])
    assert gates.non_inferior(res) and f"{res['auc']['diff'] + 0.0:+.3f} [{res['auc']['lo'] + 0.0:+.3f}" in ni[
        'R7 typed, cap 20k']
    assert '- Chain merge (kfold x86): 41000 of 10000000 blocks in 1800 binaries could merge' in text
    assert 'reduction of 0.41%' in text and 'below the 5% bar, so no chain-merge policy is built' in text
    weights = text.split('## Block-level counting weightings')[1]
    w = {l.split(' | ')[0].lstrip('| '): l for l in weights.splitlines() if l.startswith('| R2')}
    assert set(w) == {'R2 no-edges, raw', 'R2 no-edges, presence', 'R2 no-edges, rawlog'}
    assert 'missing' in w['R2 no-edges, rawlog'] and 'missing' not in w['R2 no-edges, raw']
    assert 'kfold_x86_gnn-noedges-mblk15-enc-presence' in w['R2 no-edges, presence']
    assert 'kfold_x86_gnn-noedges-mblk15 minus' not in weights and 'kfold_x86_gnn-noedges-mblk15-enc-raw minus' in weights
    pooled = text.split('## Subgraph policies')[0]
    assert '| R7 typed, whole graph | kfold_x86_gnn-mblk15 |' in pooled and 'R2 no-edges, tfidf' in pooled


def test_subgraph_gates_without_runs_or_chain_file(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    gates.pooled.cache_clear()
    text = '\n'.join(gates.subgraph(30))
    assert 'Chain merge: not measured (subgraph_chain.json is missing' in text
    assert text.count('missing') >= 10 and 'Traceback' not in text
    chain = dict(CHAIN, reduction=0.06, n_chain_blocks=600_000)
    (tmp_path / 'subgraph_chain.json').write_text(json.dumps(chain))
    assert 'a chain-merge policy is worth building' in gates.chain_line()


def test_subgraph_gate_arms_are_run_names():
    assert gates.SG_REFERENCE[1] == 'gnn-mblk15' and gates.SG_NOEDGES == 'gnn-noedges-mblk15'
    assert [v for _, v in gates.SG_ARMS] == ['gnn-mblk15-sg-cap20k', 'gnn-mblk15-sg-cap5k', 'gnn-mblk15-sg-lcc20k',
                                              'gnn-mblk15-sg-lcc5k']
    assert [v for _, v in gates.WEIGHTINGS] == [f'gnn-noedges-mblk15-enc-{e}' for e in ('raw', 'presence', 'rawlog')]
    assert [v for _, v in gates.V3_ARMS[1:]] == ['gnn-noedges-mblk15-selcell', 'gnn-shuffle-mblk15-selcell',
                                                  'gnn-mblk15-selcell']
    assert [v for _, v in gates.V3_AB_ARMS[1:]] == ['gnn-noedges-mblk15-ab-selcell', 'gnn-shuffle-mblk15-ab-selcell',
                                                     'gnn-mblk15-ab-selcell']


def test_gates_cli_accepts_the_subgraph_stage():
    r = subprocess.run([sys.executable, 'gates.py', '--help'], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and 'subgraph' in r.stdout


def test_chain_script_writes_the_json(tmp_path, monkeypatch):
    from scripts import subgraph_chain
    (tmp_path / 'samples').mkdir()
    edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
    for sid in ('a', 'b', 'c'):
        np.savez(tmp_path / 'samples' / f'{sid}.npz', blk_feat=np.zeros((3, 6), dtype=np.float16),
                 fn_blk_ptr=np.array([0, 3], dtype=np.int32), cfg_edges=edges, edge_kind=np.array([0, 2], dtype=np.int8))
    index = [{'sample_id': 'a', 'arch': 'x86', 'fold': 0}, {'sample_id': 'b', 'arch': 'x64', 'fold': 1},
             {'sample_id': 'c', 'arch': 'x86', 'fold': None}]
    monkeypatch.setattr(subgraph_chain, 'load_index', lambda: index)
    monkeypatch.setattr(subgraph_chain, 'CACHE', tmp_path)
    monkeypatch.setattr(subgraph_chain, 'RUNS', tmp_path)
    subgraph_chain.main()
    got = json.loads((tmp_path / 'subgraph_chain.json').read_text())
    assert got['pool'] == 'kfold x86' and got['n_binaries'] == 1 and got['n_blocks'] == 3
    assert got['n_chain_blocks'] == 1 and got['reduction'] == pytest.approx(1 / 3) and got['worth_building'] is True


def test_smoke_memory_script_takes_the_v3_protocol_and_masking():
    from scripts import smoke_gnn_memory as smoke
    cfg = smoke.parse_config([])
    assert (cfg.protocol, cfg.arch, cfg.mask_blocks, cfg.model) == ('kfold', 'x86', 0.0, 'gnn')
    cfg = smoke.parse_config(['--protocol', 'kfold_v3', '--mask-blocks', '0.15'])
    assert (cfg.protocol, cfg.arch, cfg.mask_blocks, cfg.fold) == ('kfold_v3', 'all', 0.15, 0)
    assert smoke.parse_config(['--protocol', 'kfold_v3', '--arch', 'matched']).arch == 'matched'
    rows = [{'num_blocks': str(n), 'sample_id': str(n)} for n in (8_000_000, 600_000, 200_000, 90_000, 60_000, 50_000)]
    assert smoke.capped_blocks(rows[0], cfg) == 500_000
    assert [r['num_blocks'] for r in smoke.packed_rows(rows, cfg)] == ['200000', '90000']
    assert smoke.capped_blocks(rows[0], gnn(budget=20000)) == 20000
    r = subprocess.run([sys.executable, 'smoke_gnn_memory.py', '--help'], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and 'kfold_v3' in r.stdout and '--mask-blocks' in r.stdout


def test_smoke_memory_script_reads_the_combined_cache_for_kfold_v3():
    code = ('import sys; sys.argv = ["smoke_gnn_memory.py", "--protocol", "kfold_v3"]; '
            'from scripts import smoke_gnn_memory as s; print(s.CACHE)')
    env = {k: v for k, v in os.environ.items() if k != 'GT_CACHE'}
    r = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().replace('\\', '/') == 'C:/Users/chaoa/Downloads/graph_transformer_cache_combined'
