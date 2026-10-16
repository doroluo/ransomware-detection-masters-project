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


def write_v3_baseline(root, arch, fold, signal, threshold=0.4):
    d = root / 'baselines' / f'v3kfold{fold}_{arch}_s0'
    d.mkdir(parents=True)
    methods = {m: {'threshold_val': threshold, 'family_recall': {}, 'test': {}, 'test_at_val_threshold': {}}
               for m in ('tfidf_sig_api_lr', 'structural_lr')}
    (d / 'metrics.json').write_text(json.dumps({'protocol': 'kfold_v3', 'fold': fold, 'arch': arch, 'seed': 0,
                                                'train_fraction': 1.0, 'floors': {}, 'methods': methods}))
    with (d / 'predictions.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'split', 'method', 'label', 'prob', 'pred', 'arch', 'family', 'fold'])
        for sid, y, p, fam, a in v3_rows(fold, 0, signal):
            for method in methods:
                w.writerow([sid, 'test', method, y, f'{p:.6f}', int(p >= 0.5), a, fam, fold])
                w.writerow([sid + 'v', 'val', method, y, f'{p:.6f}', int(p >= 0.5), a, fam, fold])


FAKE_FLOORS = {'majority': {'auc': 0.5, 'pr_auc': 0.6, 'macro_f1': 0.4, 'recall_R_x86': 1.0, 'recall_R_x64': 1.0,
                            'recall_G_x86': 0.0, 'recall_G_x64': 0.0},
               'x86_rule': {'auc': 0.7, 'pr_auc': 0.8, 'macro_f1': 0.69, 'recall_R_x86': 1.0, 'recall_R_x64': 0.0,
                            'recall_G_x86': 0.0, 'recall_G_x64': 1.0}}
V3_KW = {'gnn': {}, 'gnn-noedges': dict(control='no_edges'), 'gnn-shuffle': dict(control='shuffle'),
         'gnn-untyped': dict(untyped=True)}


def build_v3_fake(tmp_path, monkeypatch):
    """arch all: R7 stronger than R2 = R4 (R4 has folds 0-2 only), no R5; arch matched: every arm equal."""
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
                write_v3_run(tmp_path, arch, fold, signal, **V3_KW[variant])
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
    r = subprocess.run([sys.executable, 'stats.py', '--help'], cwd=HERE, capture_output=True, text=True)
    assert r.returncode == 0 and '--protocol' in r.stdout and '--arch' in r.stdout and 'matched' in r.stdout


def test_v3_gates_from_disk(tmp_path, monkeypatch):
    build_v3_fake(tmp_path, monkeypatch)
    text = '\n'.join(gates.v3(50))
    assert 'stage v3' in text and gates.LABEL in text
    all_part, matched_part = text.split('## Arch matched: pooled table')
    head = all_part.split('### Arch all')[0]
    row = {l.split(' | ')[0].lstrip('| '): l for l in head.splitlines() if l.startswith('| ')}
    assert 'R x86 recall' in head and 'AUC ckpt, val thr AUC' in head
    assert row['R0 TF-IDF'].split(' | ')[1] == 'tfidf' and 'missing' not in row['R0 TF-IDF']
    assert 'missing (seed 0 folds present [0, 1, 2] of 0-4)' in row['R4 shuffle']
    assert 'missing (seed 0 folds present [] of 0-4)' in row['R5 untyped']
    cells = row['R7 typed'].split(' | ')
    assert len(cells) == 12 and 'missing' not in row['R7 typed']
    assert all(0 <= float(c.strip(' |')) <= 1 for c in cells[2:])
    assert '| floor: majority | - | 0.500 | 0.600 | 0.400 | 0.500 | 0.600 | 0.400 | 1.000 | 1.000 | 0.000 | 0.000 |' in head
    assert 'floor: x86_rule' in head and 'x86-rule macro-F1 per fold: 0.690, 0.690, 0.690, 0.690, 0.690' in head
    for pair in ('R2 no-edges vs R7 typed', 'R4 shuffle vs R7 typed', 'R5 untyped vs R7 typed',
                 'R0 TF-IDF vs R2 no-edges'):
        assert f'**{pair}**' in all_part and f'**{pair}**' in matched_part
    assert '(selection auc): missing: kfold_v3_all_gnn-shuffle' in all_part
    assert 'kfold_v3_all_gnn-noedges minus tfidf' in all_part
    assert 'kfold_v3_matched_gnn-noedges minus tfidf' in matched_part
    lines = [l for l in text.split('## Verdict')[1].splitlines() if l.startswith('- Arch')]
    assert len(lines) == 2 and all(gates.LABEL in l for l in lines)
    assert lines[0].startswith('- Arch all: cannot assess, missing: R4 shuffle')
    assert lines[1].startswith('- Arch matched: R7 does not beat both R2 and R4')


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
        write_v3_run(tmp_path, 'all', fold, 1.2, control='shuffle')
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
    assert ('kfold_v3', 'all', 'GNN no-edges') in keys and ('kfold_v3', 'matched', 'edge-typed GNN') in keys
    assert ('kfold_v3', 'all', 'tfidf_sig_api_lr') in keys and ('kfold_v3', 'matched', 'structural_lr') in keys
    assert any('GNN shuffle, kfold_v3, all, seed 0: only folds [0, 1, 2] present' in n for n in notes)
    s, _, fl = dict(results)[('kfold_v3', 'all', 'edge-typed GNN')][0]
    assert s['n'] == 600 and set(fl) == {'majority', 'x86_rule'} and s['recall_R_x64'] >= 0
    auc_results, _ = report.collect(labels, auc=True)
    assert ('kfold_v3', 'matched', 'tfidf_sig_api_lr') in {k for k, _ in auc_results}
    assert not any(k[0] == 'kfold' for k in keys)


def test_v3_val_threshold_pools_like_kfold(tmp_path, monkeypatch, capsys):
    import val_threshold
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


V3_QUEUE_ARMS = ['gnn-noedges', 'gnn', 'gnn-shuffle', 'gnn-untyped']


@pytest.mark.skipif(BASH is None, reason='needs bash')
def test_queue_v3_dry_names_match_run_configs():
    lines = queue('--stage', 'v3', '--dry')
    runs = [l.split()[1:] for l in lines if l.startswith('run ')]
    assert len(runs) == 40
    for name, *argv in runs:
        cfg, _ = parse_config(argv)
        assert cfg.name == name, (name, cfg.name)
        assert cfg.protocol == 'kfold_v3' and cfg.seed == 0 and cfg.model == 'gnn'
    names = [n for n, *_ in runs]
    assert [n.split('_')[0] for n in names[:5]] == [f'v3kfold{f}' for f in range(5)]
    assert [tuple(n.split('_')[1:3]) for n in names[::5]] == [(a, v) for a in ('all', 'matched')
                                                              for v in V3_QUEUE_ARMS]
    arms = {tuple(n.split('_')[1:3]): parse_config(a)[0] for (n, *a) in runs[::5]}
    assert arms['all', 'gnn-noedges'].control == 'no_edges' and arms['all', 'gnn'].group == 'kfold_v3_all_gnn'
    assert arms['matched', 'gnn-shuffle'].control == 'shuffle' and arms['matched', 'gnn-untyped'].untyped
    assert arms['matched', 'gnn'].arch == 'matched' and arms['all', 'gnn-untyped'].arch == 'all'
    assert lines[:2] == ['cpu baselines --protocol kfold_v3 --fold 0 1 2 3 4 --arch all --seed 0',
                         'cpu baselines --protocol kfold_v3 --fold 0 1 2 3 4 --arch matched --seed 0']
    assert lines[-3:] == ['cpu report.py', 'cpu gates.py --stage v3 > RUNS/gates_v3.md',
                          'cpu val_threshold.py > RUNS/val_threshold.md']
    assert not any(l.startswith(('skipped', 'stopped')) for l in lines)


@pytest.mark.skipif(BASH is None, reason='needs bash')
def test_queue_v3_budget_guard_and_usage():
    lines = queue('--stage', 'v3', '--dry', GPU_BUDGET_MIN='290', DRY_RUN_MIN='100')
    assert sum(l.startswith('run ') for l in lines) == 3
    assert sum(l.startswith('skipped (budget) ') for l in lines) == 37
    assert lines[-1] == 'cpu val_threshold.py > RUNS/val_threshold.md'
    assert not any(l.startswith('skipped') for l in queue('--stage', 'v3', '--dry'))
    r = subprocess.run([BASH, 'queue_gnn.sh', '--stage', 'v4', '--dry'], cwd=HERE, capture_output=True, text=True)
    assert r.returncode == 2 and 'day1|day1b|v3' in r.stdout


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
        d = runs / f'v3kfold{fold}_all_gnn-noedges_s0'
        d.mkdir()
        (d / 'metrics.json').write_text('{}')
    env = dict(GT_PY=stub.as_posix(), GT_RUNS=runs.as_posix(), GT_TRAIN_PATTERN='gtqueue_never_matches_train[.]py',
               STOP_AFTER='2', GT_CACHE='C:/somewhere/else')
    r = subprocess.run([BASH, 'queue_gnn.sh', '--stage', 'v3'], cwd=HERE, capture_output=True, text=True,
                       env={**os.environ, **env})
    assert r.returncode == 0, r.stderr + r.stdout
    out = r.stdout.splitlines()
    assert 'skip baselines v3 all' in out and any(l.startswith('start baselines v3 matched') for l in out)
    assert [l.split()[1] for l in out if l.startswith('start v3kfold')] == [
        f'v3kfold{f}_all_gnn-noedges_s0' for f in (2, 3)]
    assert [l for l in out if l.startswith('stopped before ')][0] == 'stopped before v3kfold4_all_gnn-noedges_s0'
    calls = (runs / 'calls.txt').read_text().splitlines()
    combined = 'GT_CACHE=C:/Users/chaoa/Downloads/graph_transformer_cache_combined'
    assert {c.split(' ', 1)[0] for c in calls} == {'-c', 'baselines.py', 'train.py', 'report.py', 'gates.py',
                                                   'val_threshold.py'}
    assert all(c.endswith(combined) for c in calls), calls
    assert (runs / 'gates_v3.md').exists() and out[-1].startswith('queue done')


# step 6b: truncation columns, the v3 truncation line, queue env and retry

def test_write_predictions_adds_trailing_block_columns_and_readers_still_work(tmp_path):
    import train
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
    import val_threshold
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
        d = write_v3_run(root, arch, fold, 1.5)
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
    for d in tmp_path.glob('v3kfold*_matched_gnn_s0'):
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
    p = stats.load_group('kfold_v3_matched_gnn', 'auc', ('kfold_v3', 'matched'))[0][0]
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
    r = subprocess.run([BASH, 'queue_gnn.sh', '--stage', stage], cwd=HERE, capture_output=True, text=True, env=e)
    assert r.returncode == 0, r.stderr + r.stdout
    return runs, r.stdout.splitlines()


@pytest.mark.skipif(BASH is None, reason='needs bash')
@pytest.mark.parametrize('stage', ['day1', 'day1b', 'v3'])
def test_queue_sets_expandable_segments_for_every_python_call(tmp_path, stage):
    runs, out = run_stub_queue(tmp_path, stage, PYTORCH_CUDA_ALLOC_CONF='something_else')
    calls = (runs / 'calls.txt').read_text().splitlines()
    assert calls and all(c.endswith('ALLOC=expandable_segments:True') for c in calls), calls
    assert any(c.startswith('train.py') for c in calls) and not any(l.startswith('retry') for l in out)


@pytest.mark.skipif(BASH is None, reason='needs bash')
def test_queue_retries_a_failed_run_once(tmp_path):
    runs, out = run_stub_queue(tmp_path, 'v3', FAIL_ONCE_FOLD='2')
    calls = (runs / 'calls.txt').read_text().splitlines()
    name = 'v3kfold2_all_gnn-noedges_s0'
    assert [l for l in out if l.startswith('retry')][0] == f'retry {name}' and out.count(f'retry {name}') == 1
    i = out.index(f'retry {name}')
    assert out[i - 1].startswith(f'start {name}') and out[i + 1].startswith(f'finished {name} rc=0')
    assert sum(c.startswith('train.py') and '--fold 2 --arch all' in c and '--control no_edges' in c for c in calls) == 2
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
    r = subprocess.run([BASH, 'queue_gnn.sh', '--stage', 'day1b'], cwd=HERE, capture_output=True, text=True,
                       env={**os.environ, 'GT_PY': killer.as_posix(), 'GT_RUNS': runs.as_posix(),
                            'GT_TRAIN_PATTERN': 'gtqueue_never_matches_train[.]py'})
    assert r.returncode == 0, r.stderr
    assert 'retry' not in r.stdout and (runs / 'calls.txt').read_text().count('train.py') == 1
