"""Aggregate whatever run directories exist into RESULTS.md next to PLAN.md.

python report.py

Model runs are graph_transformer_runs/<RunConfig.name>/ (directories named otherwise, such as smoke runs, are
skipped). Bundle results are mean +/- sample sd over seeds; kfold results pool the out-of-fold test predictions of
all five folds per seed first, and a seed with missing folds is left out. A second table repeats the runs that have
an AUC-selected checkpoint at their validation threshold, and the baselines at theirs.
"""
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from config import RUNS, RunConfig
from data import load_index
from metrics import family_recall, scores

HERE = Path(__file__).resolve().parent
COLUMNS = (('accuracy', 'Acc'), ('precision', 'P'), ('recall', 'R'), ('f1', 'F1'), ('macro_f1', 'macro-F1'),
           ('auc', 'AUC'), ('pr_auc', 'PR-AUC'), ('recall_R_x86', 'R x86'), ('recall_G_x86', 'G x86'),
           ('recall_R_x64', 'R x64'), ('recall_G_x64', 'G x64'))
DEFAULTS = asdict(RunConfig())
LABELLED = ('protocol', 'fold', 'arch', 'seed', 'no_graph', 'num_workers', 'model', 'encoder', 'readout', 'control',
            'untyped', 'no_call_mp', 'stack_tfidf', 'mask_tokens', 'drop_functions', 'drop_library', 'mask_blocks',
            'aux_recon', 'aux_edge')
CONTROL_LABELS = {'no_edges': 'no-edges', 'shuffle': 'shuffle', 'chain': 'chain', 'permute_types': 'permuted types'}


def model_label(cfg):
    """Readable arm name; every field in LABELLED that affects the arm shows up here, so overrides skip them."""
    if cfg.model == 'transformer':
        return f"{'no-graph' if cfg.no_graph else 'graph'} transformer"
    ablate = [CONTROL_LABELS[cfg.control]] if cfg.control else []
    ablate += [s for s, on in (('untyped', cfg.untyped), ('no call-MP', cfg.no_call_mp)) if on]
    parts = ['GNN'] if ablate else ['edge-typed GNN']
    parts += ablate
    parts += ['stacked'] if cfg.stack_tfidf else []
    parts += [f'{s} {round(100 * p)}%' for s, p in (('mask-tok', cfg.mask_tokens), ('drop-fn', cfg.drop_functions),
                                                    ('drop-lib', cfg.drop_library), ('mask-blk', cfg.mask_blocks)) if p]
    parts += ['aux'] if cfg.aux_recon > 0 or cfg.aux_edge > 0 else []
    parts += ['enc-mean'] if cfg.block_encoder == 'mean' else []
    parts += ['gated'] if cfg.readout == 'gated' else []
    return ' '.join(parts)


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


def pr_auc_of(rows):
    y = [int(r['label']) for r in rows]
    return float(average_precision_score(y, [float(r['prob']) for r in rows])) if len(set(y)) == 2 else float('nan')


def pooled(rows, thresholds=None):
    """Scores and family recall of rows pooled; `thresholds` (fold -> t) replaces the stored pred by prob >= t."""
    y = np.array([int(r['label']) for r in rows])
    prob = np.array([float(r['prob']) for r in rows])
    pred = (np.array([int(r['pred']) for r in rows]) if thresholds is None
            else (prob >= np.array([thresholds[r['fold']] for r in rows])).astype(int))
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
        overrides = {k: v for k, v in asdict(cfg).items() if k not in LABELLED and v != DEFAULTS[k]}
        groups[(cfg.protocol, cfg.arch, model_label(cfg) + (f' {overrides}' if overrides else ''))].append((cfg, d))
    return groups


def summarize_model_group(protocol, members, labels, auc=False):
    """Per seed: (scores, family recall, floors). Also {seed: reason} for the seeds left out. With `auc`, the
    AUC-selected checkpoint at its validation threshold (runs without `selection` are left out)."""
    per_seed, left_out = {}, {}
    if protocol == 'bundle':
        for cfg, d in members:
            m = json.loads((d / 'metrics.json').read_text(encoding='utf-8'))
            if auc:
                if 'selection' not in m:
                    left_out[cfg.seed] = 'no AUC-selected checkpoint (run predates it)'
                    continue
                sel = m['selection']['auc']
                per_seed[cfg.seed] = (sel['test'], sel['family_recall'], m['floors'])
            else:
                test = m['test']
                if 'pr_auc' not in test:
                    test = {**test, 'pr_auc': pr_auc_of(read_predictions(d / 'predictions.csv'))}
                per_seed[cfg.seed] = (test, m['family_recall'], m['floors'])
        return per_seed, left_out
    folds = defaultdict(dict)
    for cfg, d in members:
        folds[cfg.seed][cfg.fold] = d
    predictions = 'predictions_auc.csv' if auc else 'predictions.csv'
    for seed, by_fold in folds.items():
        have = sorted(f for f, d in by_fold.items() if (d / predictions).exists())
        if len(by_fold) < 5 or len(have) < 5:
            left_out[seed] = (f'only folds {sorted(by_fold)} present' if len(by_fold) < 5
                              else 'no AUC-selected checkpoint (run predates it)')
            continue
        rows, train_ids = [], {}
        for fold, d in by_fold.items():
            rows += read_predictions(d / predictions)
            train_ids[str(fold)] = json.loads((d / 'manifest.json').read_text(encoding='utf-8'))['train']
        s, fam = pooled(rows)
        per_seed[seed] = (s, fam, pooled_floors(rows, train_ids, labels))
    return per_seed, left_out


def baseline_runs(auc=False):
    """(protocol, arch, method) -> {seed: (scores, family recall, floors)} and {(key, seed): reason} for the rest.
    With `auc`, test scores at each baseline's validation threshold; kfold pools each fold at its own."""
    root = RUNS / 'baselines'
    groups, kfold, left_out = defaultdict(dict), defaultdict(lambda: defaultdict(dict)), {}
    if not root.exists():
        return groups, left_out
    for d in sorted(p for p in root.iterdir() if (p / 'metrics.json').exists()):
        m = json.loads((d / 'metrics.json').read_text(encoding='utf-8'))
        for method, v in m['methods'].items():
            key, seed = (m['protocol'], m['arch'], method), m['seed']
            if auc and 'test_at_val_threshold' not in v:
                left_out[(key, seed)] = 'no validation threshold (baseline predates it)'
            elif m['protocol'] == 'bundle':
                test = v['test_at_val_threshold'] if auc else v['test']
                if 'pr_auc' not in test:
                    test = {**test, 'pr_auc': pr_auc_of(read_predictions(d / 'predictions.csv', method=method))}
                groups[key][seed] = (test, v['family_recall'], m['floors'])
            else:
                kfold[key][seed][m['fold']] = (d, v.get('threshold_val'))
    for key, seeds in kfold.items():
        for seed, by_fold in seeds.items():
            if len(by_fold) < 5:
                left_out[(key, seed)] = f'only folds {sorted(by_fold)} present'
                continue
            thresholds = {str(f): t for f, (_, t) in by_fold.items()} if auc else None
            rows = [r for d, _ in by_fold.values() for r in read_predictions(d / 'predictions.csv', method=key[2])]
            s, fam = pooled(rows, thresholds)
            groups[key][seed] = (s, fam, {})
    return groups, left_out


def table(rows):
    head = '| config | protocol | arch | seeds | ' + ' | '.join(t for _, t in COLUMNS) + ' |'
    lines = [head, '|' + '---|' * (4 + len(COLUMNS))]
    for (protocol, arch, label), per_seed in rows:
        cells = [mean_sd([s[0].get(k, float('nan')) for s in per_seed.values()]) for k, _ in COLUMNS]
        lines.append(f"| {label} | {protocol} | {arch} | {len(per_seed)} | " + ' | '.join(cells) + ' |')
    return lines


def collect(labels, auc=False):
    """(table rows, notes) for the main table, or for the AUC-selected one."""
    results, notes = [], []
    for (protocol, arch, label), members in sorted(model_runs().items()):
        per_seed, left_out = summarize_model_group(protocol, members, labels, auc)
        if not auc:
            notes += [f'{label}, {protocol}, {arch}, seed {seed}: {why}, left out.' for seed, why in sorted(left_out.items())]
        else:
            by_reason = defaultdict(list)
            for seed, why in sorted(left_out.items()):
                by_reason[why].append(seed)
            notes += [f'{label}, {protocol}, {arch}, seeds {seeds}: {why}, omitted.' for why, seeds in by_reason.items()]
        if per_seed:
            results.append(((protocol, arch, label), per_seed))
    base, base_left_out = baseline_runs(auc)
    if auc:
        by_reason = defaultdict(list)
        for (key, seed), why in sorted(base_left_out.items()):
            by_reason[key, why].append(seed)
        notes += [f'{key[2]}, {key[0]}, {key[1]}, seeds {seeds}: {why}, omitted.' for (key, why), seeds in by_reason.items()]
    else:
        notes += [f'{key[2]}, {key[0]}, {key[1]}, seed {seed}: {why}, left out.'
                  for (key, seed), why in sorted(base_left_out.items())]
    results += sorted(base.items())
    return results, notes


def main():
    labels = {r['sample_id']: r['label'] for r in load_index()}
    results, notes = collect(labels)
    auc_results, auc_notes = collect(labels, auc=True)

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
    out += ['## AUC-selected checkpoint, validation threshold', '',
            'Runs with a `selection` block in metrics.json, scored from their best-validation-AUC checkpoint at the '
            'threshold that maximises validation macro-F1 (grid of 49 values); kfold rows pool each fold at its own '
            'threshold. Baselines use the same rule on their validation split.', ''] + table(auc_results) + ['']

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
    if notes or auc_notes:
        out += ['## Notes', ''] + [f'- {n}' for n in notes] + [f'- AUC table: {n}' for n in auc_notes] + ['']
    (HERE / 'RESULTS.md').write_text('\n'.join(out), encoding='utf-8')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
