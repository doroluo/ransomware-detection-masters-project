"""CPU directed WL subtree counts + TF-IDF/LR on the existing IDA cache manifests.

Three fixed matched arms: h=0, h=2, and h=2 with within-function degree-preserving
shuffled CFGs. All use the same block content labels and include levels 0..h.
No register names, operand widths, RVAs, function IDs or degree features enter labels.
python wl_baseline.py --protocol kfold --fold 0 1 2 3 4 --arch x86 --seed 0 1 2
"""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import csv
from pathlib import Path
import time
import uuid
import os

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

from common.config import CACHE, RUNS, RunConfig
from pipeline.batching import load_index, make_manifest
from evaluation.metrics import cell_weights, family_recall, floors, scores
from experiments.side_arrays.side_common import sample_rng, shuffle_edges
from evaluation.val_threshold import GRID, macro_f1

METHODS = ('wl_h0_lr', 'wl_h2_lr', 'wl_h2_shuffle_lr')
N_FEATURES = 2 ** 18


def digest(value):
    return hashlib.blake2b(json.dumps(value, separators=(',', ':')).encode(), digest_size=16).hexdigest()


def block_labels(z, vocab, start=0, stop=None):
    """Sorted mnemonic/operand-category signature and API multisets. No graph structure at h=0.

    Resolve vocab IDs to strings so identical inputs in v2 and combined yield the
    same features even if the append-only vocabularies have different ID layouts.
    """
    labels = []
    stop = len(z['blk_insn_ptr']) - 1 if stop is None else stop
    for b in range(start, stop):
        content = []
        for kind, ptr in (('sig', 'blk_insn_ptr'), ('api', 'blk_api_ptr')):
            lo, hi = z[ptr][b:b + 2]
            ids, counts = np.unique(z[kind][lo:hi], return_counts=True)
            content.extend((kind, vocab[kind][i], int(n)) for i, n in zip(ids, counts))
        labels.append(digest(sorted(content)))
    return labels


def wl_counts(labels, edges, height=2):
    """Directed, untyped WL: independently sorted incoming/outgoing label multisets.

    Parallel edges in the shuffled multigraph contribute their multiplicity. All
    cached blocks/functions are retained; there is no training-dependent truncation.
    """
    if height < 0:
        raise ValueError('height must be nonnegative')
    labels = list(labels)
    pred, succ = [[] for _ in labels], [[] for _ in labels]
    for a, b in np.asarray(edges).reshape(-1, 2):
        if not 0 <= a < len(labels) or not 0 <= b < len(labels):
            raise ValueError('CFG endpoint outside label array')
        succ[a].append(b)
        pred[b].append(a)
    counts = Counter(f'0:{label}' for label in labels)
    for h in range(1, height + 1):
        labels = [digest((labels[b], sorted(labels[a] for a in pred[b]), sorted(labels[c] for c in succ[b])))
                  for b in range(len(labels))]
        counts.update(f'{h}:{label}' for label in labels)
    if not counts:
        counts['empty_graph'] = 1
    return counts


def sample_features(z, vocab, seed, sample_id):
    """Process one function at a time; bound hashing storage even for 8M-block binaries."""
    hasher = FeatureHasher(n_features=N_FEATURES, input_type='dict', alternate_sign=False)
    edges = z['cfg_edges']
    ptr = z['fn_blk_ptr']
    shuffled = shuffle_edges(edges, ptr, sample_rng(sample_id, seed))
    fn = np.searchsorted(ptr[1:], edges[:, 0], side='right')
    order = np.argsort(fn, kind='stable')
    edge_ptr = np.r_[0, np.cumsum(np.bincount(fn, minlength=len(ptr) - 1))]
    bins = np.zeros((len(METHODS), N_FEATURES), dtype=np.float64)
    for f, (lo, hi) in enumerate(zip(ptr[:-1], ptr[1:])):
        if lo == hi:
            continue
        labels = block_labels(z, vocab, lo, hi)
        idx = order[edge_ptr[f]:edge_ptr[f + 1]]
        real, null = edges[idx] - lo, shuffled[idx] - lo
        dictionaries = (wl_counts(labels, real, 0), wl_counts(labels, real, 2), wl_counts(labels, null, 2))
        for arm, counts in enumerate(dictionaries):
            hashed = hasher.transform([counts]).tocsr()
            hashed.sum_duplicates()
            bins[arm, hashed.indices] += hashed.data
    if ptr[-1] == 0:
        bins[:, :] = hasher.transform([{'empty_graph': 1}]).toarray()
    return {m: sp.csr_matrix(bins[i:i + 1]) for i, m in enumerate(METHODS)}, {
        'blocks': int(ptr[-1]), 'edges': len(edges),
        'shuffled_endpoints_changed': int(np.count_nonzero(edges[:, 1] != shuffled[:, 1]))}


def load_features(rows, cache, vocab, seed):
    """Stream binaries into fixed-size sparse hashed count vectors (no learned vocabulary).

    FeatureHasher's nonnegative 2**18 bins can collide. The SAME map is used for
    every arm. Min-DF and IDF are fitted later on training rows only.
    """
    matrices = {m: [] for m in METHODS}
    diagnostics = {'blocks': 0, 'edges': 0, 'shuffled_endpoints_changed': 0}
    for i, row in enumerate(rows):
        with np.load(Path(cache) / 'samples' / f"{row['sample_id']}.npz", allow_pickle=False) as archive:
            z = {k: archive[k] for k in ('blk_insn_ptr', 'blk_api_ptr', 'sig', 'api', 'cfg_edges', 'fn_blk_ptr')}
            features, counts = sample_features(z, vocab, seed, row['sample_id'])
            for method, values in features.items():
                matrices[method].append(values)
            for key, value in counts.items():
                diagnostics[key] += value
        if (i + 1) % 100 == 0:
            print(f'WL extracted {i + 1}/{len(rows)} binaries', flush=True)
    return {m: sp.vstack(v, format='csr') for m, v in matrices.items()}, diagnostics


def fit_predict(xtr, xva, xte, y, cfg, train_arch):
    keep = np.asarray((xtr > 0).sum(0)).ravel() >= cfg.min_df
    if not keep.any():
        raise ValueError('no WL bins survive training min_df')
    transform = TfidfTransformer(sublinear_tf=True)
    xtr = transform.fit_transform(xtr[:, keep])
    xva, xte = transform.transform(xva[:, keep]), transform.transform(xte[:, keep])
    model = LogisticRegression(C=1.0, max_iter=5000, class_weight=None if cfg.arch_balance else 'balanced',
                               random_state=cfg.seed, solver='liblinear')
    weights = cell_weights(y, train_arch) if cfg.arch_balance else None
    model.fit(xtr, y, sample_weight=weights)
    return model.predict_proba(xva)[:, 1], model.predict_proba(xte)[:, 1], int(keep.sum())


def run(cfg, cache=None, out_root=None):
    cache = Path(cache or CACHE)
    # Separate from existing baselines, keeping historical directory names untouched.
    out = Path(out_root or RUNS / 'baselines_wl') / replace(cfg, train_fraction=1.0, arch_balance=False).name.replace('_graph', '')
    suffix = (f'_tf{round(100 * cfg.train_fraction)}' if cfg.train_fraction < 1 else '') + ('_ab' if cfg.arch_balance else '')
    out = out.with_name(out.name + suffix)
    if out.exists():
        raise FileExistsError(f'use a fresh output root or remove the incomplete experiment explicitly: {out}')
    manifest = make_manifest(cfg, load_index(cache))
    train, val, test = manifest.train, manifest.val, manifest.test
    if any(not rows for rows in (train, val, test)) or len({r['label'] for r in train}) != 2:
        raise ValueError('need non-empty train/val/test and both training classes')
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = out.with_name(out.name + '.pending-' + uuid.uuid4().hex)
    stage.mkdir()
    start = time.monotonic()
    manifest.save(stage / 'manifest.json')
    vocab = json.loads((cache / 'vocab.json').read_text(encoding='utf-8'))
    matrices, diagnostics = {}, {}
    for split, rows in (('train', train), ('val', val), ('test', test)):
        matrices[split], diagnostics[split] = load_features(rows, cache, vocab, cfg.seed)
    ytr, yva, yte = (np.array([r['label'] for r in rows]) for rows in (train, val, test))
    arch, family = [r['arch'] for r in test], [r['family'] for r in test]
    metrics = dict(protocol=cfg.protocol, fold=cfg.fold, arch=cfg.arch, seed=cfg.seed,
                   train_fraction=cfg.train_fraction, arch_balance=cfg.arch_balance, cache=str(cache),
                   n_train=len(train), n_val=len(val), n_test=len(test),
                   floors=floors(ytr, yte, arch), methods={}, diagnostics=diagnostics,
                   experiment={'schema': 'ida-wl/1', 'height': 2, 'hash_bins': N_FEATURES,
                               'initial_labels': 'block sig+api multisets', 'directed': True, 'typed': False,
                               'shuffle': 'within-function directed multigraph destination permutation',
                               'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               'vocab_sha256': hashlib.sha256((cache / 'vocab.json').read_bytes()).hexdigest(),
                               'index_sha256': hashlib.sha256((cache / 'index.csv').read_bytes()).hexdigest(),
                               'min_df': cfg.min_df, 'lr_C': 1.0, 'solver': 'liblinear'})
    with (stage / 'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        w = csv.writer(stream)
        w.writerow(['sample_id', 'split', 'method', 'label', 'prob', 'pred', 'arch', 'family', 'fold'])
        for method in METHODS:
            pva, pte, n_features = fit_predict(*(matrices[s][method] for s in ('train', 'val', 'test')),
                                               ytr, cfg, [r['arch'] for r in train])
            threshold = float(max(GRID, key=lambda t: macro_f1(yva, pva, t)))
            pred = (pte >= .5).astype(int)
            metrics['methods'][method] = dict(test=scores(yte, pte, pred, arch),
                family_recall=family_recall(yte, pred, family), threshold_val=threshold,
                test_at_val_threshold=scores(yte, pte, (pte >= threshold).astype(int), arch), features=n_features)
            for split, rows, probs in (('val', val, pva), ('test', test, pte)):
                for r, p in zip(rows, probs):
                    w.writerow([r['sample_id'], split, method, r['label'], repr(float(p)), int(p >= .5),
                                r['arch'], r['family'], r['fold']])
    metrics['elapsed_seconds'] = time.monotonic() - start
    (stage / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    os.rename(stage, out)
    print(json.dumps({'out': str(out), 'seconds': metrics['elapsed_seconds'],
                      'auc': {m: v['test']['auc'] for m, v in metrics['methods'].items()}}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', choices=['bundle', 'kfold', 'kfold_v3'], default='kfold')
    ap.add_argument('--fold', nargs='+', type=int)
    ap.add_argument('--arch', choices=['x86', 'all', 'matched'], default='x86')
    ap.add_argument('--seed', nargs='+', type=int, default=[0])
    ap.add_argument('--train-fraction', type=float, default=1.0)
    ap.add_argument('--arch-balance', action='store_true')
    ap.add_argument('--out-root', type=Path)
    args = ap.parse_args()
    folds = args.fold if args.fold is not None else ([None] if args.protocol == 'bundle' else list(range(5)))
    for seed in args.seed:
        for fold in folds:
            run(RunConfig(protocol=args.protocol, fold=fold, arch=args.arch, seed=seed,
                          train_fraction=args.train_fraction, arch_balance=args.arch_balance), out_root=args.out_root)


if __name__ == '__main__':
    main()
