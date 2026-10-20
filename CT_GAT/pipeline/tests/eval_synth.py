"""Synthetic run directories for the tests of `s04_evaluation`: fake kfold / kfold_v3 runs and baselines on disk (config.json,
predictions.csv, predictions_auc.csv, metrics.json), the arms the gate stages read, and the fixture that points the evaluation
modules at the tiny cache. Not a test module."""
import csv
import json

import numpy as np
import pytest

from common.config import RunConfig
from s04_evaluation import baselines, gate_layouts, gate_tables, report, stats
from s04_evaluation.val_threshold import GRID, macro_f1
from tiny_cache import tiny  # noqa: F401  (a fixture)


def val_threshold_of(rows, ids, probs):
    """The GRID threshold with the highest validation macro-F1 (first on ties), as experiments/train.py picks it."""
    by_id = {r['sample_id']: r for r in rows}
    y = np.array([by_id[i]['label'] for i in ids])
    return float(max(GRID, key=lambda g: macro_f1(y, probs, g)))


@pytest.fixture(scope='module')
def evaltiny(tiny):  # noqa: F811
    """The tiny cache with the baselines' CACHE and the runs directory of stats / report pointed at it (`evaltiny.runs`)."""
    runs = tiny.root.parent / 'runs'
    (runs / 'baselines').mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(baselines, 'CACHE', tiny.root)
        mp.setattr(stats, 'RUNS', runs)
        mp.setattr(report, 'RUNS', runs)
        yield type('EvalTiny', (), {'root': tiny.root, 'runs': runs, 'ids': tiny.ids, 'rows': tiny.rows})


def gnn(**kw):
    return RunConfig(model='gnn', **kw)


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


def write_fake_run(root, cfg, seed_probs, selection=True, thresholds=0.4):
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
            write_fake_run(root, gnn(protocol='kfold', fold=fold, seed=0, **dict(kw)), fake_fold_rows(fold, 0, signal))


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


def build_v3_fake(tmp_path, monkeypatch):
    """arch all: R7 stronger than R2 = R4 (R4 has folds 0-2 only); arch matched: every arm equal. Every GNN arm is
    block-masked with cell-balanced selection, as in the v3 queue."""
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    monkeypatch.setattr(report, 'RUNS', tmp_path)
    monkeypatch.setattr(report, 'v3_floors', lambda arch, seed=0: (FAKE_FLOORS, [FAKE_FLOORS] * 5))
    gate_tables.pooled.cache_clear()
    gate_layouts.v3_cell.cache_clear()
    plan = {'all': {'gnn-noedges': (1.2, 5), 'gnn': (1.7, 5), 'gnn-shuffle': (1.2, 3)},
            'matched': {v: (1.2, 5) for v in V3_KW}}
    for arch, arms in plan.items():
        for variant, (signal, n_folds) in arms.items():
            for fold in range(n_folds):
                write_v3_run(tmp_path, arch, fold, signal, **V3_KW[variant], **V3_MASKED)
        for fold in range(5):
            write_v3_baseline(tmp_path, arch, fold, 1.3 if arch == 'all' else 0.6)


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


CELLS = [(1, 'x86'), (1, 'x64'), (0, 'x86'), (0, 'x64')]


def cell_sample(counts):
    """y, arch of a synthetic set with counts[i] samples in CELLS[i]."""
    y, arch = [], []
    for (label, a), n in zip(CELLS, counts):
        y += [label] * n
        arch += [a] * n
    return np.array(y), np.array(arch)


def build_v3_ab_fake(tmp_path, monkeypatch):
    build_v3_fake(tmp_path, monkeypatch)
    for fold in range(5):
        write_v3_run(tmp_path, 'all', fold, 1.9, arch_balance=True, **V3_MASKED)
        write_v3_run(tmp_path, 'all', fold, 1.3, arch_balance=True, control='no_edges', **V3_MASKED)
        write_v3_run(tmp_path, 'all', fold, 1.1, arch_balance=True, control='shuffle', **V3_MASKED)
        write_v3_baseline(tmp_path, 'all', fold, 1.5, suffix='_ab')


def arch_preds(n=400, seed=0, signal=1.0):
    p = preds(n=n, seed=seed, signal=signal)
    return stats.Preds(p.ids, p.y, p.prob, p.pred, p.cluster, np.where(np.arange(n) % 3 == 0, 'x64', 'x86'))


SG_REF = ('mask_blocks', 0.15)


def build_subgraph_fake(root, monkeypatch, with_chain=None):
    """kfold x86 seed 0: the whole-graph reference, cap 20k (as good), cap 5k (clearly worse); the no-edge reference and
    its raw and presence weightings (rawlog missing)."""
    monkeypatch.setattr(stats, 'RUNS', root)
    gate_tables.pooled.cache_clear()
    plan = {(SG_REF,): (1.6, 5), (SG_REF, ('subgraph', 'cap'), ('budget', 20000)): (1.6, 5),
            (SG_REF, ('subgraph', 'cap'), ('budget', 5000)): (0.3, 5),
            (SG_REF, ('control', 'no_edges')): (1.4, 5),
            (SG_REF, ('control', 'no_edges'), ('encoder', 'raw')): (1.4, 5),
            (SG_REF, ('control', 'no_edges'), ('encoder', 'presence')): (0.5, 5)}
    build_fake(root, {kw: (signal, range(folds)) for kw, (signal, folds) in plan.items()})
    if with_chain is not None:
        (root / 'subgraph_chain.json').write_text(json.dumps(with_chain))


CHAIN = {'pool': 'kfold x86', 'n_binaries': 1800, 'n_blocks': 10_000_000, 'n_chain_blocks': 41_000,
         'reduction': 0.0041, 'mean_binary_reduction': 0.005, 'max_binary_reduction': 0.09}


def needs_combined(fn):
    """A test of the real combined cache: marked real_data, skipped when this machine has no combined cache."""
    from common.paths import COMBINED_CACHE
    skip = pytest.mark.skipif(not (COMBINED_CACHE / 'index.csv').exists(), reason='needs the combined cache')
    return pytest.mark.real_data(skip(fn))
