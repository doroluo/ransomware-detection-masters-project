"""Aggregate whatever run directories exist into RESULTS.md next to PLAN.md.

python report.py

Model runs are graph_transformer_runs/<RunConfig.name>/ (directories named otherwise, such as smoke runs, are
skipped). Bundle results are mean +/- sample sd over seeds; kfold results pool the out-of-fold test predictions of
all five folds per seed first, and a seed with missing folds is left out.
"""
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from config import RUNS, RunConfig
from data import load_index
from metrics import family_recall, scores

HERE = Path(__file__).resolve().parent
COLUMNS = (('accuracy', 'Acc'), ('precision', 'P'), ('recall', 'R'), ('f1', 'F1'), ('macro_f1', 'macro-F1'),
           ('auc', 'AUC'), ('recall_R_x86', 'R x86'), ('recall_G_x86', 'G x86'), ('recall_R_x64', 'R x64'),
           ('recall_G_x64', 'G x64'))
DEFAULTS = asdict(RunConfig())


def mean_sd(values):
    values = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return 'n/a'
    if len(values) == 1:
        return f'{values[0]:.3f}'
    return f'{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}'


def read_predictions(path, split='test', method=None):
    with path.open(encoding='utf-8', newline='') as f:
        return [r for r in csv.DictReader(f) if r.get('split', split) == split and r.get('method', method) == method]


def pooled(rows):
    y = np.array([int(r['label']) for r in rows])
    prob = np.array([float(r['prob']) for r in rows])
    pred = np.array([int(r['pred']) for r in rows])
    arch = [r['arch'] for r in rows]
    return scores(y, prob, pred, arch), family_recall(y, pred, [r['family'] for r in rows])


def pooled_floors(rows, train_ids_by_fold, labels):
    by_fold = defaultdict(list)
    for r in rows:
        by_fold[r['fold']].append(r)
    major = []
    for fold, members in by_fold.items():
        train_labels = [labels[i] for i in train_ids_by_fold[fold]]
        major += [int(np.mean(train_labels) >= 0.5)] * len(members)
    rows_sorted = [r for fold in by_fold for r in by_fold[fold]]
    y, arch = [int(r['label']) for r in rows_sorted], [r['arch'] for r in rows_sorted]
    pred_major = np.array(major)
    pred_x86 = np.array([int(a == 'x86') for a in arch])
    return {'majority': scores(y, pred_major.astype(float), pred_major, arch),
            'x86_rule': scores(y, pred_x86.astype(float), pred_x86, arch)}


def model_runs():
    groups = defaultdict(list)
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        if not all((d / f).exists() for f in ('config.json', 'metrics.json', 'predictions.csv')):
            continue
        cfg = RunConfig.load(d / 'config.json')
        if d.name != cfg.name:
            continue
        overrides = {k: v for k, v in asdict(cfg).items()
                     if k not in ('protocol', 'fold', 'arch', 'seed', 'no_graph', 'num_workers') and v != DEFAULTS[k]}
        label = f"{'no-graph' if cfg.no_graph else 'graph'} transformer" + (f' {overrides}' if overrides else '')
        groups[(cfg.protocol, cfg.arch, label)].append((cfg, d))
    return groups


def summarize_model_group(protocol, members, labels):
    """Per seed: (scores, family recall, floors)."""
    per_seed = {}
    if protocol == 'bundle':
        for cfg, d in members:
            m = json.loads((d / 'metrics.json').read_text(encoding='utf-8'))
            per_seed[cfg.seed] = (m['test'], m['family_recall'], m['floors'])
        return per_seed, {}
    folds = defaultdict(dict)
    for cfg, d in members:
        folds[cfg.seed][cfg.fold] = d
    incomplete = {}
    for seed, by_fold in folds.items():
        if len(by_fold) < 5:
            incomplete[seed] = sorted(by_fold)
            continue
        rows, train_ids = [], {}
        for fold, d in by_fold.items():
            rows += read_predictions(d / 'predictions.csv')
            train_ids[str(fold)] = json.loads((d / 'manifest.json').read_text(encoding='utf-8'))['train']
        s, fam = pooled(rows)
        per_seed[seed] = (s, fam, pooled_floors(rows, train_ids, labels))
    return per_seed, incomplete


def baseline_runs(labels):
    """(protocol, arch, method) -> {seed: (scores, family recall, floors)}."""
    root = RUNS / 'baselines'
    groups, kfold = defaultdict(dict), defaultdict(lambda: defaultdict(dict))
    if not root.exists():
        return groups, {}
    for d in sorted(p for p in root.iterdir() if (p / 'metrics.json').exists()):
        m = json.loads((d / 'metrics.json').read_text(encoding='utf-8'))
        for method, v in m['methods'].items():
            key = (m['protocol'], m['arch'], method)
            if m['protocol'] == 'bundle':
                groups[key][m['seed']] = (v['test'], v['family_recall'], m['floors'])
            else:
                kfold[key][m['seed']][m['fold']] = d
    incomplete = {}
    for key, seeds in kfold.items():
        for seed, by_fold in seeds.items():
            if len(by_fold) < 5:
                incomplete[(key, seed)] = sorted(by_fold)
                continue
            rows = [r for d in by_fold.values() for r in read_predictions(d / 'predictions.csv', method=key[2])]
            s, fam = pooled(rows)
            groups[key][seed] = (s, fam, {})
    return groups, incomplete


def table(rows):
    head = '| config | protocol | arch | seeds | ' + ' | '.join(t for _, t in COLUMNS) + ' |'
    lines = [head, '|' + '---|' * (4 + len(COLUMNS))]
    for (protocol, arch, label), per_seed in rows:
        cells = [mean_sd([s[0][k] for s in per_seed.values()]) for k, _ in COLUMNS]
        lines.append(f"| {label} | {protocol} | {arch} | {len(per_seed)} | " + ' | '.join(cells) + ' |')
    return lines


def main():
    labels = {r['sample_id']: r['label'] for r in load_index()}
    results = []
    notes = []
    for (protocol, arch, label), members in sorted(model_runs().items()):
        per_seed, incomplete = summarize_model_group(protocol, members, labels)
        for seed, folds_present in incomplete.items():
            notes.append(f'{label}, {protocol}, {arch}, seed {seed}: only folds {folds_present} present, left out.')
        if per_seed:
            results.append(((protocol, arch, label), per_seed))
    base, base_incomplete = baseline_runs(labels)
    for (key, seed), folds_present in base_incomplete.items():
        notes.append(f'{key[2]}, {key[0]}, {key[1]}, seed {seed}: only folds {folds_present} present, left out.')
    for key, per_seed in sorted(base.items()):
        results.append((key, per_seed))

    floor_rows = {}
    for (protocol, arch, _), per_seed in results:
        for seed, (_, _, fl) in per_seed.items():
            for name, s in fl.items():
                floor_rows.setdefault((protocol, arch, f'floor: {name}'), {})[seed] = (s, {}, {})

    out = ['# Results', '',
           'Generated by `report.py` from the run directories under `graph_transformer_runs/`. Test-set metrics; '
           'positive class ransomware; threshold 0.5. Cells are mean ± sample sd over seeds (a single value when one '
           'seed exists). kfold rows pool the out-of-fold predictions of all five folds per seed. R/G x86/x64 are '
           'recall of ransomware/goodware within each architecture.', '']
    out += ['## Models and baselines', ''] + table(results) + ['']
    out += ['## Floors on the same test manifests', ''] + table(sorted(floor_rows.items())) + ['']

    bundle = [(k, v) for k, v in results if k[0] == 'bundle']
    if bundle:
        families = sorted({f for _, per_seed in bundle for s in per_seed.values() for f in s[1]})
        out += ['## Per-family test recall, bundle protocol', '',
                '| family | ' + ' | '.join(f'{label} ({arch})' for (_, arch, label), _ in bundle) + ' |',
                '|' + '---|' * (1 + len(bundle))]
        for fam in families:
            cells = []
            for _, per_seed in bundle:
                present = [s[1][fam] for s in per_seed.values() if fam in s[1]]
                cells.append(f"{mean_sd([v['recall'] for v in present])} (n={present[0]['n']})" if present else 'n/a')
            out.append(f'| {fam} | ' + ' | '.join(cells) + ' |')
        out.append('')
    if notes:
        out += ['## Notes', ''] + [f'- {n}' for n in notes] + ['']
    (HERE / 'RESULTS.md').write_text('\n'.join(out), encoding='utf-8')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
