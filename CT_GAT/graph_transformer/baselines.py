"""TF-IDF and structural logistic regressions on the graph transformer's manifests.

python baselines.py --protocol bundle --arch x86 --seed 0 1 2 | --protocol kfold --fold 0 1 2 3 4 --arch all
| --protocol kfold_v3 --fold 0 1 2 3 4 --arch all|matched (combined cache unless GT_CACHE is set)
[--train-fraction 0.5]

Fit on the run's training manifest (validation families held out, as for the model), scored on its validation and
test sets. Writes <out-root>/<protocol[fold]>_<arch>_s<seed>/{metrics.json, predictions.csv} with out-root
graph_transformer_runs/baselines by default (`_tf<pct>` appended with --train-fraction < 1); metrics.json records
the cache path, adds the validation threshold (val_threshold.GRID rule) and the test scores at it, and predictions.csv
has val and test rows told apart by the split column.
"""
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import CACHE, RUNS, RunConfig
from data import load_index, make_manifest
from metrics import family_recall, floors, scores
from val_threshold import GRID, macro_f1


def load_features(rows, sizes):
    """Per binary: sig and api count vectors, and mean/max structural block features plus log counts."""
    counts, structural = {'sig': [], 'api': []}, []
    for r in rows:
        with np.load(CACHE / 'samples' / f"{r['sample_id']}.npz") as z:
            for k in counts:
                ids, n = np.unique(z[k], return_counts=True)
                counts[k].append(sp.csr_matrix((n, (np.zeros(len(ids)), ids)), shape=(1, sizes[k])))
            feat = z['blk_feat'].astype(np.float32)
            logs = np.log1p([len(z['fn_flags']), len(feat), len(z['cfg_edges']), len(z['mn'])])
            structural.append(np.concatenate([feat.mean(0), feat.max(0), logs]))
    return sp.hstack([sp.vstack(counts['sig']), sp.vstack(counts['api'])]).tocsr(), np.array(structural)


def run(cfg: RunConfig, sizes, out_root=None):
    manifest = make_manifest(cfg, load_index())
    train, val, test = manifest.train, manifest.val, manifest.test
    x_train_tok, x_train_st = load_features(train, sizes)
    x_val_tok, x_val_st = load_features(val, sizes)
    x_test_tok, x_test_st = load_features(test, sizes)
    keep = np.asarray((x_train_tok > 0).sum(0)).ravel() >= cfg.min_df
    y_train = np.array([r['label'] for r in train])
    y_val = np.array([r['label'] for r in val])
    y_test = np.array([r['label'] for r in test])
    arch, family = [r['arch'] for r in test], [r['family'] for r in test]

    models = {
        'tfidf_sig_api_lr': (make_pipeline(TfidfTransformer(sublinear_tf=True),
                                           LogisticRegression(max_iter=5000, class_weight='balanced')),
                             x_train_tok[:, keep], x_val_tok[:, keep], x_test_tok[:, keep]),
        'structural_lr': (make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight='balanced')),
                          x_train_st, x_val_st, x_test_st),
    }
    tf = f'_tf{round(100 * cfg.train_fraction)}' if cfg.train_fraction < 1 else ''
    out = Path(out_root or RUNS / 'baselines') / (replace(cfg, train_fraction=1.0).name.replace('_graph', '') + tf)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {'protocol': cfg.protocol, 'fold': cfg.fold, 'arch': cfg.arch, 'seed': cfg.seed,
               'train_fraction': cfg.train_fraction, 'cache': str(CACHE), 'n_train': len(train), 'n_val': len(val),
               'n_test': len(test), 'tfidf_features': int(keep.sum()),
               'floors': floors(y_train, y_test, arch), 'methods': {}}
    probs, val_probs = {}, {}
    for name, (model, xtr, xva, xte) in models.items():
        model.fit(xtr, y_train)
        probs[name], val_probs[name] = model.predict_proba(xte)[:, 1], model.predict_proba(xva)[:, 1]
        pred = (probs[name] >= 0.5).astype(int)
        threshold_val = float(max(GRID, key=lambda g: macro_f1(y_val, val_probs[name], g)))
        at_val = (probs[name] >= threshold_val).astype(int)
        metrics['methods'][name] = {'test': scores(y_test, probs[name], pred, arch),
                                    'family_recall': family_recall(y_test, pred, family),
                                    'threshold_val': threshold_val,
                                    'test_at_val_threshold': scores(y_test, probs[name], at_val, arch)}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    with (out / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'split', 'method', 'label', 'prob', 'pred', 'arch', 'family', 'fold'])
        for name in probs:
            for split, rows, p in (('val', val, val_probs[name]), ('test', test, probs[name])):
                for r, pi in zip(rows, p):
                    w.writerow([r['sample_id'], split, name, r['label'], f'{pi:.6f}', int(pi >= 0.5), r['arch'],
                                r['family'], r['fold']])
    print(json.dumps({'run': out.name, **{m: {k: round(v['test'][k], 4) for k in ('macro_f1', 'accuracy', 'auc')}
                                          for m, v in metrics['methods'].items()}}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', choices=['bundle', 'kfold', 'kfold_v3'], default='bundle')
    ap.add_argument('--fold', type=int, nargs='*', default=[None])
    ap.add_argument('--arch', choices=['x86', 'all', 'matched'], default='x86')
    ap.add_argument('--seed', type=int, nargs='*', default=[0])
    ap.add_argument('--train-fraction', type=float, default=1.0)
    ap.add_argument('--out-root', help='default: graph_transformer_runs/baselines')
    args = ap.parse_args()
    sizes = {k: len(v) for k, v in json.loads((CACHE / 'vocab.json').read_text(encoding='utf-8')).items()}
    for fold in args.fold:
        for seed in args.seed:
            run(RunConfig(protocol=args.protocol, fold=fold, arch=args.arch, seed=seed,
                          train_fraction=args.train_fraction), sizes, args.out_root)


if __name__ == '__main__':
    main()
