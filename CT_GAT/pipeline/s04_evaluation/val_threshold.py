"""Re-score saved predictions with the threshold that maximises validation macro-F1 (the graph2vec `oofthr`
convention), and report kfold runs per fold as well as pooled. Reads predictions.csv only; nothing is retrained.

The module also holds the other threshold rule a trainer can ask for, `--threshold-from oof` (`oof_rule`): the validation
families of a fold can saturate (validation AUC near 1.0 within a few epochs), and then every threshold in a wide band has the
same validation macro-F1 and the first of them is taken. The out-of-fold rule chooses the threshold on predictions that the
model did not train on and that are not saturated: the training binaries are split in `OOF_K` folds that keep a ransomware
family or a goodware identity group whole (`common.manifests.grouped_folds`), the model is refit on `OOF_K - 1` of them for
the epoch count of the run, and the held-out predictions of the folds are pooled."""
import argparse
import csv
import statistics
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from common.config import RUNS, RunConfig
from common.manifests import grouped_folds

GRID = np.linspace(0.02, 0.98, 49)
KFOLD = ('kfold', 'kfold_v3')
OOF_K = 3


def load(run_dir):
    with (run_dir / 'predictions.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    split = np.array([r['split'] for r in rows])
    y = np.array([int(r['label']) for r in rows])
    p = np.array([float(r['prob']) for r in rows])
    return y[split == 'val'], p[split == 'val'], y[split == 'test'], p[split == 'test']


def macro_f1(y, p, t):
    return f1_score(y, (p >= t).astype(int), average='macro', zero_division=0)


def grid_threshold(y, p):
    """The GRID threshold with the highest macro-F1 on (y, p); the first one on ties."""
    return float(max(GRID, key=lambda g: macro_f1(y, p, g)))


def oof_rule(rows, fit_predict, epochs, k=OOF_K, seed=0):
    """The out-of-fold threshold of a model.

    rows         the training rows (`label`, `family`, `identity_group`, `sample_id`)
    fit_predict  fit_predict(train_rows, held_rows, epochs, seed) -> {epoch: P(ransomware) of held_rows, in their order}: fit a
                 new model on train_rows for max(epochs) epochs and predict held_rows after each epoch in `epochs` (the model
                 owns the cost; `seed` is the seed of the refit)
    epochs       {rule: epoch}, e.g. {'f1': best_epoch, 'auc': auc_best_epoch}: the epoch count of the checkpoint each rule
                 thresholds
    Returns ({rule: threshold}, record). The record is what metrics.json keeps: the rule, k, the epochs, the number of
    out-of-fold predictions, the thresholds and their out-of-fold macro-F1."""
    held = grouped_folds(rows, k, seed)
    wanted = tuple(sorted(set(epochs.values())))
    pooled = {e: ([], []) for e in wanted}
    for j, held_rows in enumerate(held):
        held_ids = {r['sample_id'] for r in held_rows}
        train_rows = [r for r in rows if r['sample_id'] not in held_ids]
        out = fit_predict(train_rows, held_rows, wanted, seed * 1000 + j + 1)
        for e in wanted:
            probs = np.asarray(out[e], dtype=np.float64)
            if probs.shape != (len(held_rows),):
                raise ValueError(f'fit_predict gave {probs.shape} predictions for {len(held_rows)} held-out binaries')
            pooled[e][0].extend(r['label'] for r in held_rows)
            pooled[e][1].extend(probs)
    thresholds, f1s = {}, {}
    for rule, e in epochs.items():
        y, p = np.array(pooled[e][0]), np.array(pooled[e][1])
        thresholds[rule] = grid_threshold(y, p)
        f1s[rule] = float(macro_f1(y, p, thresholds[rule]))
    record = {'rule': 'oof', 'k': k, 'grouping': 'ransomware family, goodware identity group (common.manifests.grouped_folds)',
              'refit_epochs': dict(epochs), 'n_oof': len(rows), 'fold_sizes': [len(h) for h in held],
              'threshold': thresholds, 'oof_macro_f1': f1s}
    return thresholds, record


def scores(y, p, t):
    pred = (p >= t).astype(int)
    return {'macro_f1': macro_f1(y, p, t), 'recall_R': float(pred[y == 1].mean()),
            'recall_G': float(1 - pred[y == 0].mean()), 'auc': roc_auc_score(y, p)}


def fmt(values):
    return f'{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}' if len(values) > 1 else f'{values[0]:.3f}'


def main(argv=None):
    argparse.ArgumentParser(description='Re-score the saved predictions of every run at the validation threshold; prints a markdown table.').parse_args(argv)
    if not RUNS.is_dir():
        raise SystemExit(f'no run folder at {RUNS}: run a training command first (it writes there), or set GT_RUNS')
    groups = defaultdict(list)
    for run_dir in sorted(RUNS.iterdir()):
        if not (run_dir / 'config.json').exists() or not (run_dir / 'predictions.csv').exists():
            continue
        cfg = RunConfig.try_load(run_dir / 'config.json')
        if cfg is None or run_dir.name != cfg.name:
            continue
        yv, pv, yt, pt = load(run_dir)
        t = max(GRID, key=lambda g: macro_f1(yv, pv, g))
        key = f'{cfg.group}_s{cfg.seed}' if cfg.protocol in KFOLD else cfg.group
        groups[key].append((cfg, t, yt, pt))

    print('| config | unit | n | threshold | macro-F1 @0.5 | macro-F1 @val thr | R / G @val thr | AUC |')
    print('|---|---|---|---|---|---|---|---|')
    pooled_by_group = defaultdict(list)
    for group, runs in sorted(groups.items()):
        at_half = [scores(yt, pt, 0.5) for _, _, yt, pt in runs]
        at_val = [scores(yt, pt, t) for _, t, yt, pt in runs]
        unit = 'folds' if runs[0][0].protocol in KFOLD else 'seeds'
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
            if len(runs) == 5:
                pooled_by_group[runs[0][0].group].append(
                    (macro_f1(yt, half, 0.5), pooled_val, roc_auc_score(yt, half)))

    if pooled_by_group:
        print('\n| kfold config | seeds | pooled macro-F1 @0.5 | pooled macro-F1 @val thr | pooled AUC |')
        print('|---|---|---|---|---|')
        for group, seeds in sorted(pooled_by_group.items()):
            cols = list(zip(*seeds))
            print(f'| {group} | {len(seeds)} | {fmt(cols[0])} | {fmt(cols[1])} | {fmt(cols[2])} |')


if __name__ == '__main__':
    main()
