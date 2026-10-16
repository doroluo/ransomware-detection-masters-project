"""TF-IDF and structural logistic regressions on the graph transformer's manifests.

python baselines.py --protocol bundle --arch x86 --seed 0 1 2 | --protocol kfold --fold 0 1 2 3 4 --arch all
| --protocol kfold_v3 --fold 0 1 2 3 4 --arch all|matched (combined cache unless GT_CACHE is set)
[--train-fraction 0.5] [--arch-balance] [--val-seed 0] [--stack-logits]

Fit on the run's training manifest (validation families held out, as for the model), scored on its validation and
test sets. Writes <out-root>/<protocol[fold]>_<arch>_s<seed>/{metrics.json, predictions.csv} with out-root
graph_transformer_runs/baselines by default (`_tf<pct>` appended with --train-fraction < 1); metrics.json records
the cache path, adds the validation threshold (val_threshold.GRID rule) and the test scores at it, and predictions.csv
has val and test rows told apart by the split column.

--arch-balance replaces class_weight='balanced' by per-sample weights N / (C n_cell) over the non-empty (label, arch)
cells of the training manifest (the weights train.py gives the loss), passed as sample_weight to both regressions; the
output directory gets the suffix `_ab` and metrics.json records `arch_balance`. Every `test` block carries
`cell_balanced` (each class x arch cell weighs equally).

--stack-logits writes <out-root>/<same directory>/stack_logits.csv and nothing else: the TF-IDF logistic regression's
logit (decision function) of every row of the manifest, for `train.py --stack-tfidf`. Training rows get out-of-fold
logits from an inner 5-fold CV on the training manifest, folds grouped by ransomware family and goodware identity
group and stratified by class, with the document-frequency filter refitted inside each inner fold; validation and test
rows get the logit of the model fitted on the whole training manifest. A row's logit therefore never comes from a model
that saw the row, nor another binary of its family or group. The CSV has sample_id, split, inner_fold, label, family,
logit; stack_logits.json says how it was made. Not combined with --arch-balance.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.config import CACHE, RUNS, RunConfig
from common.manifests import load_index, make_manifest
from evaluation.metrics import cell_weights, family_recall, floors, scores
from evaluation.val_threshold import GRID, macro_f1


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


def baseline_name(cfg: RunConfig):
    """Directory name of the baselines of cfg's split (protocol, fold, arch, seed, val_seed): the transformer run name
    without the model word, whatever model or tag cfg has."""
    split = RunConfig(protocol=cfg.protocol, fold=cfg.fold, arch=cfg.arch, seed=cfg.seed, val_seed=cfg.val_seed)
    return split.name.replace('_graph', '')


STACK_SPLITS = 5


def tfidf_model():
    return make_pipeline(TfidfTransformer(sublinear_tf=True),
                         LogisticRegression(max_iter=5000, class_weight='balanced'))


def fit_logits(x_fit, y_fit, x_pred, min_df):
    """Logits (decision function) of the TF-IDF regression fitted on (x_fit, y_fit) for each matrix of `x_pred`; the
    features kept are the ones in at least `min_df` rows of x_fit."""
    keep = np.asarray((x_fit > 0).sum(0)).ravel() >= min_df
    model = tfidf_model().fit(x_fit[:, keep], y_fit)
    return [model.decision_function(x[:, keep]) for x in x_pred]


def inner_groups(rows):
    """Cluster of each row: its ransomware family or goodware identity group, kept apart by class."""
    return np.array([f"fam:{r['family']}" if r['label'] == 1 else f"id:{r['identity_group']}" for r in rows])


def inner_folds(y, groups, seed, n_splits=STACK_SPLITS):
    """[(fit indices, held-out indices)] of a class-stratified inner CV in which no group is split."""
    return list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))


def oof_logits(x, y, groups, min_df, seed, n_splits=STACK_SPLITS):
    """(logits, inner fold of each row): every row's logit is from a model fitted on the other folds only."""
    logits, which = np.empty(len(y)), np.empty(len(y), dtype=int)
    for k, (fit, held) in enumerate(inner_folds(y, groups, seed, n_splits)):
        logits[held], = fit_logits(x[fit], y[fit], [x[held]], min_df)
        which[held] = k
    return logits, which


def write_stack_logits(cfg: RunConfig, sizes, out_root=None):
    """stack_logits.csv and stack_logits.json for cfg's split (see the module docstring)."""
    if cfg.arch_balance:
        raise ValueError('--stack-logits is not combined with --arch-balance')
    manifest = make_manifest(cfg, load_index())
    x_train, _ = load_features(manifest.train, sizes)
    x_val, _ = load_features(manifest.val, sizes)
    x_test, _ = load_features(manifest.test, sizes)
    y = np.array([r['label'] for r in manifest.train])
    oof, which = oof_logits(x_train, y, inner_groups(manifest.train), cfg.min_df, cfg.seed)
    val, test = fit_logits(x_train, y, [x_val, x_test], cfg.min_df)
    out = Path(out_root or RUNS / 'baselines') / baseline_name(cfg)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'stack_logits.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'split', 'inner_fold', 'label', 'family', 'logit'])
        for split, rows, logits, folds in (('train', manifest.train, oof, which), ('val', manifest.val, val, None),
                                           ('test', manifest.test, test, None)):
            for i, (r, v) in enumerate(zip(rows, logits)):
                w.writerow([r['sample_id'], split, '' if folds is None else int(folds[i]), r['label'], r['family'],
                            repr(float(v))])
    (out / 'stack_logits.json').write_text(json.dumps({
        'cache': str(CACHE), 'protocol': cfg.protocol, 'fold': cfg.fold, 'arch': cfg.arch, 'seed': cfg.seed,
        'val_seed': cfg.val_seed, 'inner_splits': STACK_SPLITS, 'inner_seed': cfg.seed, 'min_df': cfg.min_df,
        'n_train': len(manifest.train), 'n_val': len(manifest.val), 'n_test': len(manifest.test),
        'model': 'TfidfTransformer(sublinear_tf) + LogisticRegression(class_weight=balanced), decision_function',
        'train_rows': 'out of fold, StratifiedGroupKFold over family / identity group',
        'val_test_rows': 'fitted on the whole training manifest'}, indent=2), encoding='utf-8')
    print(json.dumps({'stack_logits': out.name, 'train': len(manifest.train), 'val': len(manifest.val),
                      'test': len(manifest.test)}), flush=True)
    return out


def read_stack_logits(path, manifest):
    """{sample_id: logit} from a stack_logits.csv, for the rows of `manifest`. Every train / val / test row of the
    manifest must be there in its own role (training rows out of fold); a row missing or in another role is an error."""
    if not Path(path).exists():
        raise FileNotFoundError(f'{path}: run baselines.py --stack-logits for this protocol, fold, arch and seed first')
    with Path(path).open(encoding='utf-8', newline='') as f:
        table = {r['sample_id']: (r['split'], float(r['logit'])) for r in csv.DictReader(f)}
    out = {}
    for split in ('train', 'val', 'test'):
        for r in getattr(manifest, split):
            got = table.get(r['sample_id'])
            if got is None or got[0] != split:
                found = 'missing' if got is None else f'a {got[0]} row'
                raise ValueError(f"{path}: {r['sample_id']} is {found}, the run needs it as a {split} row")
            out[r['sample_id']] = got[1]
    return out


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

    class_weight = None if cfg.arch_balance else 'balanced'
    fit_args = ({'logisticregression__sample_weight': cell_weights(y_train, [r['arch'] for r in train])}
                if cfg.arch_balance else {})
    models = {
        'tfidf_sig_api_lr': (make_pipeline(TfidfTransformer(sublinear_tf=True),
                                           LogisticRegression(max_iter=5000, class_weight=class_weight)),
                             x_train_tok[:, keep], x_val_tok[:, keep], x_test_tok[:, keep]),
        'structural_lr': (make_pipeline(StandardScaler(),
                                        LogisticRegression(max_iter=5000, class_weight=class_weight)),
                          x_train_st, x_val_st, x_test_st),
    }
    tf = f'_tf{round(100 * cfg.train_fraction)}' if cfg.train_fraction < 1 else ''
    ab = '_ab' if cfg.arch_balance else ''
    out = Path(out_root or RUNS / 'baselines') / (baseline_name(cfg) + tf + ab)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {'protocol': cfg.protocol, 'fold': cfg.fold, 'arch': cfg.arch, 'seed': cfg.seed,
               'train_fraction': cfg.train_fraction, 'arch_balance': cfg.arch_balance, 'cache': str(CACHE),
               'n_train': len(train), 'n_val': len(val), 'n_test': len(test), 'tfidf_features': int(keep.sum()),
               'floors': floors(y_train, y_test, arch), 'methods': {}}
    probs, val_probs = {}, {}
    for name, (model, xtr, xva, xte) in models.items():
        model.fit(xtr, y_train, **fit_args)
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
    ap.add_argument('--arch-balance', action='store_true')
    ap.add_argument('--val-seed', type=int, default=None)
    ap.add_argument('--stack-logits', action='store_true', help='write stack_logits.csv only (see the docstring)')
    ap.add_argument('--out-root', help='default: graph_transformer_runs/baselines')
    args = ap.parse_args()
    sizes = {k: len(v) for k, v in json.loads((CACHE / 'vocab.json').read_text(encoding='utf-8')).items()}
    for fold in args.fold:
        for seed in args.seed:
            cfg = RunConfig(protocol=args.protocol, fold=fold, arch=args.arch, seed=seed, val_seed=args.val_seed,
                            train_fraction=args.train_fraction, arch_balance=args.arch_balance)
            (write_stack_logits if args.stack_logits else run)(cfg, sizes, args.out_root)


if __name__ == '__main__':
    main()
