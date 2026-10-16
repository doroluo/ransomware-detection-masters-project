"""python -m pytest test_harness.py: PR-AUC, stopping rules, the paired bootstrap, gates, report tables, baselines with
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
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

import gates
import report
import stats
from config import RunConfig
from metrics import scores
from train import check_supported, parse_config, should_stop, val_threshold_of
from val_threshold import GRID

HERE = Path(__file__).parent
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
    import baselines
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
    r = subprocess.run([BASH, 'queue_gnn.sh', *args], cwd=HERE, capture_output=True, text=True,
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
    import baselines
    from data import load_index, make_manifest
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
    r = subprocess.run([BASH, 'queue_gnn.sh', '--stage', 'day1b'], cwd=HERE, capture_output=True, text=True,
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

    r = subprocess.run([BASH, 'queue_gnn.sh', '--stage', 'day1b'], cwd=HERE, capture_output=True, text=True,
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
        r = subprocess.run([BASH, 'stop_queue.sh'], cwd=HERE, capture_output=True, text=True, env=env)
        assert r.returncode == 0 and (tmp_path / 'runs' / 'STOP').exists() and 'killed' not in r.stdout
        assert victim.poll() is None
        (tmp_path / 'runs' / 'STOP').unlink()
        r = subprocess.run([BASH, 'stop_queue.sh', '--now'], cwd=HERE, capture_output=True, text=True, env=env)
        assert r.returncode == 0 and (tmp_path / 'runs' / 'STOP').exists()
        assert 'created' in r.stdout and 'killed train.py process(es):' in r.stdout
        assert victim.wait(timeout=20) is not None and bystander.poll() is None
        r = subprocess.run([BASH, 'stop_queue.sh', '--now'], cwd=HERE, capture_output=True, text=True, env=env)
        assert 'no train.py process was running' in r.stdout
        assert subprocess.run([BASH, 'stop_queue.sh', '--bogus'], cwd=HERE, env=env, capture_output=True).returncode == 2
    finally:
        for p in (victim, bystander):
            p.kill()
            p.wait()
