#!/usr/bin/env python3
"""Family-holdout evaluation of two FIXED graph2vec/CFG configurations.

    python family_holdout/run_graph2vec.py --dataset both

Both configurations are read out of `results/graph2vec/tuned/mendeley/
chosen.json` and are used verbatim -- nothing is searched here:

  baseline_h2_cap5000           the untuned baseline the track started from:
                                linear-sweep CFG, cap 5,000 blocks, no
                                cross-section targets, no import-thunk nodes,
                                WL h=2 over `class` node labels, min_df 5,
                                TF-IDF over the WL histogram, LogisticRegression
                                C=1. Same numbers as results/graph2vec/<ds>/.

  tuned_best_sparse_histogram   the post-hoc winner of the tuning study: same
                                construction plus import-thunk nodes, WL h=4
                                over `lenbucket` labels (block size bucket +
                                terminator kind), min_df 3, LogisticRegression
                                C=10 class_weight=balanced. It was selected
                                AFTER the fixed test split was opened, so its
                                0.947 macro-F1 there is not a result. Running
                                it over every family in turn is the
                                confirmation it needs, and this is that run.

Decision rules (both reported, as two model directories per configuration):
  <name>_argmax   the classifier's own argmax, i.e. p >= 0.5
  <name>_oofthr   the threshold that maximises macro-F1 on out-of-fold scores
                  computed INSIDE the training folds, with
                  StratifiedGroupKFold(5, shuffle=True, random_state=42) over
                  `group` (family for ransomware, stream/project group for
                  goodware) -- `graph2vec_pipeline.final_eval`'s rule. The test
                  fold is never consulted to pick it.

What is fit where, per fold and per LOFO training set: the min_df vocabulary
of the WL histogram comes from that fold's training rows (as in tune.py, where
it is the TRAIN split), the IDF weights are refit inside every inner fold and
again on the whole training set, and the classifier only ever sees training
rows. The WL relabelling itself is label-free and deterministic, so it is
computed once per (dataset, construction, representation) and reused.

The graph cache (`graph2vec_pipeline/graph_cache.py`, keyed by sha256) is
reused as-is: no `.asm` file is re-read and no sample is ever executed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from family_holdout.common import (DATASETS, Folds, OUT_ROOT,  # noqa: E402
                                   check_kfold, check_lofo, write_model_dir, ALL_DATASETS)
from graph2vec_pipeline import wl  # noqa: E402
from family_holdout.common import TREE_OF_CORPUS  # noqa: E402
from graph2vec_pipeline.graph_cache import (DEFAULT_CACHE, TREES,  # noqa: E402
                                            build_tree, cache_file,
                                            load_graphs)
from graph2vec_pipeline.tune import (SEED, Construction, Model,  # noqa: E402
                                     N_FOLDS, Rep, best_threshold, load_obj,
                                     make_model)

PIPELINE = "graph2vec"
CHOSEN = REPO / "results" / "graph2vec" / "tuned" / "mendeley" / "chosen.json"
WANT = ("baseline_h2_cap5000", "tuned_best_sparse_histogram")


def load_configs(names=WANT):
    doc = json.loads(CHOSEN.read_text(encoding="utf-8"))
    by_name = {p["name"]: p for p in doc["picks"]}
    out = []
    for n in names:
        if n not in by_name:
            raise KeyError(f"{n} not in {CHOSEN}")
        c, r, m = load_obj(by_name[n])
        out.append((n, c, r, m, by_name[n]))
    return out


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def restrict(M, train_mask, min_df: int):
    """Keep the columns of M whose document frequency over `train_mask` rows is
    at least min_df. Identical to wl.layer_matrix(min_df=...), but the full
    2**28-wide count matrix is built once instead of once per fold."""
    if min_df <= 0:
        return M
    ref = M[train_mask]
    uniq, cnt = np.unique(ref.indices, return_counts=True)
    keep = uniq[cnt >= min_df]
    if not len(keep):
        return sparse.csr_matrix((M.shape[0], 0), dtype=np.float32)
    pos = np.clip(np.searchsorted(keep, M.indices), 0, len(keep) - 1)
    ok = keep[pos] == M.indices
    rowid = np.repeat(np.arange(M.shape[0], dtype=np.int32), np.diff(M.indptr))
    return sparse.coo_matrix((M.data[ok], (rowid[ok], pos[ok])),
                             shape=(M.shape[0], len(keep))).tocsr()


class Layers:
    """Per-depth WL count matrices over every row of one dataset, unpruned."""

    def __init__(self, gs, folds: Folds, c: Construction, r: Rep):
        rows = np.array([gs.index[s] for s in folds.sha], dtype=np.int64)
        b = wl.assemble(gs, rows, max_blocks=c.max_blocks,
                        keep_edges=c.keep_edges())
        self.n_nodes = int(b.n_nodes)
        self.mats = [wl.layer_matrix(b.gid, lab, b.n_graphs, None, 0)
                     for lab in wl.wl_layers(b, r.h, r.labelling,
                                             r.edge_labels)]
        del b

    def build(self, train_mask, min_df: int, h: int):
        use = [restrict(m, train_mask, min_df) for m in self.mats[:h + 1]]
        use = [m for m in use if m.shape[1]]
        return sparse.hstack(use, format="csr")


def _tfidf(X, a, b):
    from sklearn.feature_extraction.text import TfidfTransformer
    tf = TfidfTransformer(sublinear_tf=True).fit(X[a])
    return tf.transform(X[a]), tf.transform(X[b])


def _with_imports(A, B, imports, tr, te):
    """Append the hashed bag-of-imports block (dense, 2,048 dims) to the
    TF-IDF'd WL histogram of the training and test rows."""
    H = sparse.csr_matrix(imports)
    return (sparse.hstack([A, H[tr]], format="csr"),
            sparse.hstack([B, H[te]], format="csr"))


def _fit(m: Model, A, ytr, B):
    est = make_model(m)
    est.fit(A, ytr)
    if hasattr(est, "predict_proba"):
        s = est.predict_proba(B)[:, 1]
    else:
        s = est.decision_function(B)
        s = s[:, 1] if getattr(s, "ndim", 1) == 2 else s
    return est.predict(B), np.asarray(s, dtype=float)


def oof_threshold(X, y, groups, tr, m: Model, imports=None):
    """macro-F1-optimal threshold on out-of-fold scores inside `tr` only, on
    the same feature matrix (WL TF-IDF, plus the imports block when given)
    that the gated model is fitted on."""
    from sklearn.model_selection import StratifiedGroupKFold
    inner = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                 random_state=SEED).split(
        np.zeros(len(tr)), y[tr], groups[tr])
    oof = np.zeros(len(tr), dtype=float)
    for a_idx, b_idx in inner:
        A, B = _tfidf(X, tr[a_idx], tr[b_idx])
        if imports is not None:
            A, B = _with_imports(A, B, imports, tr[a_idx], tr[b_idx])
        _, s = _fit(m, A, y[tr[a_idx]], B)
        oof[b_idx] = s
    thr, f1 = best_threshold(y[tr], oof, "macro_f1")
    return float(thr), float(f1)


# ---------------------------------------------------------------------------
# the graph cache
# ---------------------------------------------------------------------------
EXTRA = "g_fh_extra_{tree}_b20000_n80000_w1_xs1_ex1.npz"


def load_cache(folds: Folds, cache_dir: Path = DEFAULT_CACHE):
    """The pipeline's own cache, extended where the family-holdout pool is
    wider than the fixed split it was built for.

    `graph_cache.needed_samples()` enumerates the two fixed-split datasets, and
    the balanced one of those is expB's committed goodware membership. The
    family-holdout pools are wider: every in-cohort row of the cohort files
    (97 more balanced goodware files, and everything of the vs and hostgood
    corpora). Rows without a cached graph are built here, once, with the
    identical construction, from the extraction tree of their `corpus` column,
    and written beside the pipeline's cache files as one extra file per
    corpus - nothing already cached is touched or rebuilt.
    """
    cache_dir = Path(cache_dir)
    paths = [cache_file(cache_dir, t, 20_000, 80_000, 1, True, True)
             for t in ("mendeley", "balanced_goodware")]
    extra = [cache_dir / EXTRA.format(tree=t) for t in TREES]
    gs = load_graphs(paths + [p for p in extra if p.exists()])
    missing = [s for s in folds.sha if s not in gs.index]
    if not missing:
        return gs
    print(f"[{folds.dataset}] {len(missing)} fold rows are outside the "
          f"pipeline's graph cache; building them", flush=True)
    by_tree = {t: [] for t in TREE_OF_CORPUS}
    corpus = dict(zip(folds.sha, folds.corpus))
    for s in missing:
        t = corpus[s]
        if not (TREES[t] / "asm" / f"{s}.asm").exists():
            raise FileNotFoundError(TREES[t] / "asm" / f"{s}.asm")
        by_tree[t].append(s)
    for t, shas in by_tree.items():
        if not shas:
            continue
        out = cache_dir / EXTRA.format(tree=t)
        have = set()
        if out.exists():
            have = set(np.load(out, allow_pickle=False)["shas"].tolist())
        shas = sorted(have | set(shas))
        t0 = time.time()
        build_tree(t, shas, 20_000, 80_000, 1, True, True, out, progress=20)
        print(f"  built {len(shas)} graphs for {t} in {time.time()-t0:.0f}s",
              flush=True)
    gs = load_graphs(paths + [p for p in extra if p.exists()])
    still = [s for s in folds.sha if s not in gs.index]
    if still:
        raise RuntimeError(f"{len(still)} rows still missing, e.g. {still[:3]}")
    return gs


# ---------------------------------------------------------------------------
def run_dataset(dataset: str, out_root: Path, names=WANT, imports=None) -> None:
    t_start = time.time()
    folds = Folds(dataset)
    check_kfold(folds)
    check_lofo(folds)
    gs = load_cache(folds)
    y, groups, n = folds.y, folds.group, folds.n

    for name, c, r, m, pick in load_configs(names):
        t_cfg = time.time()
        print(f"[{dataset}] {name}: {c.key()} | {r.key()} | {m.key()}",
              flush=True)
        L = Layers(gs, folds, c, r)
        print(f"  WL done ({L.n_nodes/1e6:.1f}M nodes, "
              f"{time.time()-t_cfg:.0f}s)", flush=True)

        kf_score = np.zeros(n)
        kf_pred_a = np.zeros(n, dtype=int)
        kf_pred_t = np.zeros(n, dtype=int)
        thr_of_fold, n_feat, oof_f1 = {}, {}, {}

        for f, tr, te in folds.kfold():
            t0 = time.time()
            X = L.build(folds.fold != f, r.min_df, r.h)
            n_feat[str(f)] = int(X.shape[1])
            thr, f1 = oof_threshold(X, y, groups, tr, m, imports)
            thr_of_fold[f], oof_f1[str(f)] = thr, round(f1, 4)
            A, B = _tfidf(X, tr, te)
            if imports is not None:
                A, B = _with_imports(A, B, imports, tr, te)
            pred, score = _fit(m, A, y[tr], B)
            kf_pred_a[te], kf_score[te] = pred, score
            kf_pred_t[te] = (score >= thr).astype(int)
            print(f"  [{dataset}/{name}] fold {f}: {X.shape[1]} feats "
                  f"thr={thr:.3f} {time.time()-t0:.0f}s", flush=True)
            del X

        lofo_a, lofo_t, lofo_thr = {}, {}, {}
        fold_of = folds.fold_of_family()
        for k, (fam, tr, te) in enumerate(folds.lofo(), 1):
            t0 = time.time()
            trmask = np.zeros(n, dtype=bool)
            trmask[tr] = True
            X = L.build(trmask, r.min_df, r.h)
            A, B = _tfidf(X, tr, te)
            if imports is not None:
                A, B = _with_imports(A, B, imports, tr, te)
            pred, score = _fit(m, A, y[tr], B)
            # The threshold is a train-side quantity. Recomputing it inside
            # each LOFO training set would be a sixth of the whole run for a
            # number that moves by <0.01 between folds, so the threshold of
            # the K-fold training set that also excludes this family is
            # reused: it saw neither this family nor its fold.
            thr = thr_of_fold[fold_of[fam]]
            lofo_thr[fam] = thr
            lofo_a[fam] = (score, pred)
            lofo_t[fam] = (score, (score >= thr).astype(int))
            print(f"  [{dataset}/{name}] lofo {k}/{len(folds.families)} {fam}: n={len(te)} "
                  f"{time.time()-t0:.0f}s", flush=True)
            del X

        elapsed = time.time() - t_cfg
        base_cfg = {
            "pipeline": PIPELINE,
            "config_name": name,
            "source": str(CHOSEN.relative_to(REPO)),
            "selected_by": ("group-CV inside the fixed split's TRAIN rows "
                            "(pre-registered baseline)" if "baseline" in name
                            else "post-hoc: highest-CV sparse-histogram "
                                 "configuration of the tuning study"),
            "construction": dict(max_blocks=c.max_blocks, windows=c.windows,
                                 calls=c.calls, cross_section=c.cross_section,
                                 extern=c.extern, max_insns=c.max_insns,
                                 keep_edges=list(c.keep_edges())),
            "representation": dict(kind=r.kind, labelling=r.labelling, h=r.h,
                                   edge_labels=r.edge_labels, norm=r.norm,
                                   min_df=r.min_df, with_size=r.with_size),
            "model": {"name": m.name, "params": dict(m.params),
                      "estimator": repr(make_model(m))},
            "graph_cache": "graph2vec_pipeline/graph_cache.py b20000 n80000 "
                           "w1 xs1 ex1 (filtered down to this construction)",
            "fit_per_fold": ["WL min_df vocabulary (training rows)",
                             "TF-IDF IDF weights (training rows)",
                             "classifier (training rows)"],
            "n_features_per_kfold": n_feat,
            "imports_side_input": (None if imports is None else
                                   "family_holdout/run_imports_baseline.hash_imports, "
                                   f"{imports.shape[1]} dims, hstacked onto the WL TF-IDF block"),
            "seed": SEED,
        }
        for tag, preds, lofo, extra in (
                ("argmax", kf_pred_a, lofo_a,
                 {"decision": "argmax (default 0.5)"}),
                ("oofthr", kf_pred_t, lofo_t,
                 {"decision": "threshold from training-fold out-of-fold scores",
                  "threshold_per_fold": {str(k): round(v, 6)
                                         for k, v in thr_of_fold.items()},
                  "oof_macro_f1_per_fold": oof_f1,
                  "threshold_source_for_lofo":
                      "the K-fold training set that also excludes the family",
                  "threshold_rule":
                      "best_threshold(macro_f1) over StratifiedGroupKFold(5, "
                      "shuffle=True, random_state=42) out-of-fold scores of "
                      "the training rows"})):
            cfg = {**base_cfg, **extra}
            model_dir = f"{name}_{tag}" + ("_imports" if imports is not None else "")
            d = out_root / dataset / PIPELINE / model_dir
            write_model_dir(d, folds, PIPELINE, model_dir, kf_score, preds,
                            folds.fold, lofo, cfg, elapsed,
                            description=(f"graph2vec {name} ({r.kind}/{m.name}"
                                         f"), decision={extra['decision']}, "
                                         f"family-holdout on {dataset}"))
            print(f"  wrote {d}", flush=True)
        del L
    print(f"[{dataset}] done in {time.time()-t_start:.0f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=(*ALL_DATASETS, "both"), default="both")
    ap.add_argument("--configs", default=",".join(WANT))
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--imports", action="store_true",
                    help="append the hashed bag-of-imports to the WL block; "
                         "written to <config>_<decision>_imports/")
    a = ap.parse_args()
    names = tuple(x for x in a.configs.split(",") if x)
    for d in (DATASETS if a.dataset == "both" else (a.dataset,)):
        imports = None
        if a.imports:
            from family_holdout.run_imports_baseline import hash_imports, load_imports
            f = Folds(d)
            imports = np.vstack([hash_imports(x) for x in load_imports(f)])
        run_dataset(d, Path(a.out), names, imports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
