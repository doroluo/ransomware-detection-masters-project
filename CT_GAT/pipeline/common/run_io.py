"""Write one run directory that `s04_evaluation/stats.py`, `report.py` and `gates.py` read like one of `experiments/train.py`'s.

`write_run` takes the predictions of a trained model in its two states (best validation macro-F1, best validation AUC)
and writes config.json, manifest.json, predictions.csv, predictions_auc.csv and metrics.json. The team trainers and the
Model 1 trainer (`s03a_transformer/token_sequence/train.py`) call it with the manifest of their own fold or block export.

`AUC_RULES`, `evaluate`, `write_predictions` and `truncated_count` are the four helpers of the GNN trainer that `write_run`
needs; they live here so that this module reads no trainer, and `experiments/train.py` imports them from here.
"""
import csv
import json
from pathlib import Path

import numpy as np

from common.config import RunConfig
from common.manifests import Manifest, load_index, make_manifest
from s04_evaluation.metrics import family_recall, floors, scores

AUC_RULES = {'overall': 'checkpoint: validation AUC; threshold: validation macro-F1',
             'cell': 'checkpoint: validation cell-balanced AUC; threshold: validation cell-balanced macro-F1'}


def auc_rule(cfg: RunConfig):
    """The text of `selection.auc.rule` in metrics.json: AUC_RULES, with the threshold clause of `threshold_from='oof'`."""
    rule = AUC_RULES[cfg.select_on]
    if cfg.threshold_from == 'oof':
        rule = rule.split('; threshold:')[0] + ('; threshold: out-of-fold macro-F1 (refits on grouped folds of the training '
                                                 'binaries, s04_evaluation/val_threshold.py oof_rule)')
    return rule


def evaluate(rows, ids, probs, threshold):
    by_id = {r['sample_id']: r for r in rows}
    y = np.array([by_id[i]['label'] for i in ids])
    arch = np.array([by_id[i]['arch'] for i in ids])
    return scores(y, probs, (probs >= threshold).astype(int), arch)


def write_predictions(path, rows, by_split, threshold):
    by_id = {r['sample_id']: r for r in rows}
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'split', 'label', 'prob', 'pred', 'arch', 'family', 'fold', 'n_segments_total',
                    'n_blocks_kept', 'n_blocks_total'])
        for split, (ids, probs, sizes) in by_split.items():
            for i, p, (n_seg, kept, total) in zip(ids, probs, sizes):
                r = by_id[i]
                w.writerow([i, split, r['label'], f'{p:.6f}', int(p >= threshold), r['arch'], r['family'], r['fold'],
                            n_seg, '' if kept is None else kept, '' if total is None else total])


def truncated_count(sizes):
    """Binaries whose kept blocks are fewer than their blocks; `sizes` are (n_segments_total, kept, total) triples
    (kept and total are None when the batch carries no block counts)."""
    return sum(kept is not None and kept < total for _, kept, total in sizes)


def _check_cover(name, by_split, manifest, excluded=()):
    """The predictions hold exactly the manifest's validation and test ids minus the explicitly `excluded` ones."""
    skip = set(excluded)
    for split, rows in (('val', manifest.val), ('test', manifest.test)):
        ids = list(by_split[split][0])
        want = sorted(r['sample_id'] for r in rows if r['sample_id'] not in skip)
        if sorted(ids) != want:
            raise AssertionError(f'{name}: {split} predictions hold {len(set(ids))} distinct of {len(ids)} ids, the '
                                 f'manifest has {len(want)}; missing {sorted(set(want) - set(ids))[:5]}, '
                                 f'extra {sorted(set(ids) - set(want))[:5]}')


def write_run(run_dir, cfg: RunConfig, rows_f1, rows_auc, thr_f1, thr_auc, extra, excluded=(), excluded_reason='',
              manifest: Manifest | None = None):
    """Write one run directory that the harness reads like its own.

    run_dir     the directory; its name must equal cfg.name
    rows_f1     {'val': (ids, probs, sizes), 'test': (ids, probs, sizes)} of the best-validation-macro-F1 state, with
                sizes = (n_segments_total, n_blocks_kept, n_blocks_total) per sample (as experiments/train.py's predict)
    rows_auc    the same for the best-validation-AUC state
    thr_f1      validation threshold (GRID, val_threshold_of) of the F1 state: metrics only. predictions.csv is
                thresholded at cfg.threshold (0.5), as experiments/train.py writes it
    thr_auc     validation threshold of the AUC state: predictions_auc.csv and the AUC block of metrics.json
    extra       must hold best_epoch and epochs_run; auc_best_epoch, val_auc, val_macro_f1 and other keys are kept
    excluded    sample ids deliberately left out of this run (a binary the arm's build could not process): the row sets
                must cover the manifest's validation and test ids minus these, no more and no fewer; metrics.json records
                them as `excluded` with `excluded_reason`, and s04_evaluation/stats.py prints them where it pairs the run
                with an arm that has those binaries. An empty list keeps the exact-coverage check.
    manifest    the train / val / test rows of the run. Default: `make_manifest(cfg, load_index())`, the manifest of the
                run's cache under cfg's fold, seed and val_seed (what the team trainers use).
    Both row sets must cover exactly the manifest's validation and test ids (less `excluded`). The metrics come from
    s04_evaluation/metrics.py through the helpers above."""
    run_dir = Path(run_dir)
    if run_dir.name != cfg.name:
        raise ValueError(f'run directory {run_dir.name} is not the RunConfig name {cfg.name}')
    for key in ('best_epoch', 'epochs_run'):
        if key not in extra:
            raise KeyError(f'extra needs {key}')
    if manifest is None:
        manifest = make_manifest(cfg, load_index())
    excluded = sorted(set(excluded))
    if excluded and not excluded_reason:
        raise ValueError('excluded ids need an excluded_reason')
    unknown = sorted(set(excluded) - {r['sample_id'] for r in manifest.train + manifest.val + manifest.test})
    if unknown:
        raise ValueError(f'excluded ids that are in no split of the fold: {unknown}')
    for name, rows in (('rows_f1', rows_f1), ('rows_auc', rows_auc)):
        _check_cover(name, rows, manifest, excluded)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / 'config.json')
    manifest.save(run_dir / 'manifest.json')
    index_rows = manifest.val + manifest.test
    write_predictions(run_dir / 'predictions.csv', index_rows, rows_f1, cfg.threshold)
    write_predictions(run_dir / 'predictions_auc.csv', index_rows, rows_auc, thr_auc)

    by_id = {r['sample_id']: r for r in manifest.test}
    labels = np.array([r['label'] for r in manifest.train])

    def test_scores(rows, threshold):
        ids, probs, _ = rows['test']
        return evaluate(manifest.test, ids, probs, threshold)

    def test_family_recall(rows, threshold):
        ids, probs, _ = rows['test']
        return family_recall([by_id[i]['label'] for i in ids], (np.asarray(probs) >= threshold).astype(int),
                             [by_id[i]['family'] for i in ids])

    ids = rows_f1['test'][0]
    metrics = {
        'run': cfg.name, 'select_on': cfg.select_on, 'best_epoch': extra['best_epoch'],
        'epochs_run': extra['epochs_run'], 'val_macro_f1': extra.get('val_macro_f1'),
        'truncated_test': truncated_count(rows_f1['test'][2]),
        'test': test_scores(rows_f1, cfg.threshold),
        'floors': floors(labels, [by_id[i]['label'] for i in ids], [by_id[i]['arch'] for i in ids]),
        'family_recall': test_family_recall(rows_f1, cfg.threshold),
        'selection': {
            'f1': {'best_epoch': extra['best_epoch'], 'threshold_val': thr_f1,
                   'test_at_val_threshold': test_scores(rows_f1, thr_f1)},
            'auc': {'rule': auc_rule(cfg), 'best_epoch': extra.get('auc_best_epoch'),
                    'val_auc': extra.get('val_auc'), 'threshold_val': thr_auc,
                    'test': test_scores(rows_auc, thr_auc), 'test_at_half': test_scores(rows_auc, 0.5),
                    'family_recall': test_family_recall(rows_auc, thr_auc)},
        },
    }
    metrics.update({k: v for k, v in extra.items() if k not in metrics})
    if excluded:
        metrics.update(excluded=excluded, excluded_reason=excluded_reason)
    (run_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    return metrics
