"""Re-score saved predictions with the threshold that maximises validation macro-F1 (the graph2vec `oofthr`
convention), and report kfold runs per fold as well as pooled. Reads predictions.csv only; nothing is retrained."""
import csv
import statistics
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from config import RUNS, RunConfig

GRID = np.linspace(0.02, 0.98, 49)


def load(run_dir):
    with (run_dir / 'predictions.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    split = np.array([r['split'] for r in rows])
    y = np.array([int(r['label']) for r in rows])
    p = np.array([float(r['prob']) for r in rows])
    return y[split == 'val'], p[split == 'val'], y[split == 'test'], p[split == 'test']


def macro_f1(y, p, t):
    return f1_score(y, (p >= t).astype(int), average='macro', zero_division=0)


def scores(y, p, t):
    pred = (p >= t).astype(int)
    return {'macro_f1': macro_f1(y, p, t), 'recall_R': float(pred[y == 1].mean()),
            'recall_G': float(1 - pred[y == 0].mean()), 'auc': roc_auc_score(y, p)}


def fmt(values):
    return f'{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}' if len(values) > 1 else f'{values[0]:.3f}'


def main():
    groups = defaultdict(list)
    for run_dir in sorted(RUNS.iterdir()):
        if not (run_dir / 'config.json').exists() or not (run_dir / 'predictions.csv').exists():
            continue
        cfg = RunConfig.load(run_dir / 'config.json')
        if run_dir.name != cfg.name:
            continue
        yv, pv, yt, pt = load(run_dir)
        t = max(GRID, key=lambda g: macro_f1(yv, pv, g))
        groups[cfg.group].append((cfg, t, yt, pt))

    print('| config | unit | n | threshold | macro-F1 @0.5 | macro-F1 @val thr | R / G @val thr | AUC |')
    print('|---|---|---|---|---|---|---|---|')
    for group, runs in sorted(groups.items()):
        at_half = [scores(yt, pt, 0.5) for _, _, yt, pt in runs]
        at_val = [scores(yt, pt, t) for _, t, yt, pt in runs]
        unit = 'folds' if runs[0][0].protocol == 'kfold' else 'seeds'
        print(f"| {group} | {unit} | {len(runs)} | {fmt([t for _, t, _, _ in runs])} "
              f"| {fmt([s['macro_f1'] for s in at_half])} | {fmt([s['macro_f1'] for s in at_val])} "
              f"| {fmt([s['recall_R'] for s in at_val])} / {fmt([s['recall_G'] for s in at_val])} "
              f"| {fmt([s['auc'] for s in at_val])} |")
        if unit == 'folds':
            yt = np.concatenate([r[2] for r in runs])
            half = np.concatenate([r[3] for r in runs])
            pred = np.concatenate([(r[3] >= r[1]).astype(int) for r in runs])
            pooled_val = f1_score(yt, pred, average='macro')
            print(f"| {group} | pooled | {len(yt)} | per fold | {macro_f1(yt, half, 0.5):.3f} | {pooled_val:.3f} "
                  f"| {pred[yt == 1].mean():.3f} / {1 - pred[yt == 0].mean():.3f} | {roc_auc_score(yt, half):.3f} |")


if __name__ == '__main__':
    main()
