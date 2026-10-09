"""TF-IDF and structural logistic regressions on the graph transformer's manifests.

python baselines.py --protocol bundle --arch x86 --seed 0 1 2 | --protocol kfold --fold 0 1 2 3 4 --arch all

Fit on the run's training manifest (validation families held out, as for the model), scored once on its test set.
Writes graph_transformer_runs/baselines/<protocol[fold]>_<arch>_s<seed>/{metrics.json, predictions.csv}.
"""
import argparse
import csv
import json

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import CACHE, RUNS, RunConfig
from data import load_index, make_manifest
from metrics import family_recall, floors, scores


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


def run(cfg: RunConfig, sizes):
    manifest = make_manifest(cfg, load_index())
    train, test = manifest.train, manifest.test
    x_train_tok, x_train_st = load_features(train, sizes)
    x_test_tok, x_test_st = load_features(test, sizes)
    keep = np.asarray((x_train_tok > 0).sum(0)).ravel() >= cfg.min_df
    y_train = np.array([r['label'] for r in train])
    y_test = np.array([r['label'] for r in test])
    arch, family = [r['arch'] for r in test], [r['family'] for r in test]

    models = {
        'tfidf_sig_api_lr': (make_pipeline(TfidfTransformer(sublinear_tf=True),
                                           LogisticRegression(max_iter=5000, class_weight='balanced')),
                             x_train_tok[:, keep], x_test_tok[:, keep]),
        'structural_lr': (make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight='balanced')),
                          x_train_st, x_test_st),
    }
    out = RUNS / 'baselines' / cfg.name.replace('_graph', '')
    out.mkdir(parents=True, exist_ok=True)
    metrics = {'protocol': cfg.protocol, 'fold': cfg.fold, 'arch': cfg.arch, 'seed': cfg.seed,
               'n_train': len(train), 'n_test': len(test), 'tfidf_features': int(keep.sum()),
               'floors': floors(y_train, y_test, arch), 'methods': {}}
    probs = {}
    for name, (model, xtr, xte) in models.items():
        model.fit(xtr, y_train)
        probs[name] = model.predict_proba(xte)[:, 1]
        pred = (probs[name] >= 0.5).astype(int)
        metrics['methods'][name] = {'test': scores(y_test, probs[name], pred, arch),
                                    'family_recall': family_recall(y_test, pred, family)}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    with (out / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'method', 'label', 'prob', 'pred', 'arch', 'family', 'fold'])
        for name, p in probs.items():
            for r, pi in zip(test, p):
                w.writerow([r['sample_id'], name, r['label'], f'{pi:.6f}', int(pi >= 0.5), r['arch'], r['family'], r['fold']])
    print(json.dumps({'run': out.name, **{m: {k: round(v['test'][k], 4) for k in ('macro_f1', 'accuracy', 'auc')}
                                          for m, v in metrics['methods'].items()}}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', choices=['bundle', 'kfold'], default='bundle')
    ap.add_argument('--fold', type=int, nargs='*', default=[None])
    ap.add_argument('--arch', choices=['x86', 'all'], default='x86')
    ap.add_argument('--seed', type=int, nargs='*', default=[0])
    args = ap.parse_args()
    sizes = {k: len(v) for k, v in json.loads((CACHE / 'vocab.json').read_text(encoding='utf-8')).items()}
    for fold in args.fold:
        for seed in args.seed:
            run(RunConfig(protocol=args.protocol, fold=fold, arch=args.arch, seed=seed), sizes)


if __name__ == '__main__':
    main()
