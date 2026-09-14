#!/usr/bin/env python3
"""Train and evaluate the CFG-graph representations on the shared split.

    python graph2vec_pipeline/train_eval.py --dataset mendeley
    python graph2vec_pipeline/train_eval.py --dataset balanced

Representations (every one fitted on TRAIN graphs only):
    wl_svd        WL subtree TF-IDF -> TruncatedSVD(128)      (graph2vec-style)
    graph2vec     PV-DBOW over the same WL documents          (graph2vec proper)
    wl_tfidf      the sparse WL TF-IDF itself                 (WL-kernel baseline)
    size_only     10 scalar graph statistics                  (shortcut control)

Classifiers: the same RF / SVM-RBF / MLP grid as llm_features_pipeline
(GridSearchCV, scoring="f1_macro", StratifiedKFold(shuffle=True, seed) over
TRAIN only), plus LogisticRegression for the WL-kernel and size-only rows.

`size_only` exists to answer one question before any of the others are worth
reading: does block/edge count alone separate the classes? If it scores near
the learned representations, the graph structure is not what is being measured.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from cnn_vit_pipeline.cohort import (add_val_fold, build_result, load_split,  # noqa: E402
                                     split_summary, write_metrics)
from graph2vec_pipeline import embed as E  # noqa: E402
from graph2vec_pipeline.build_graphs import DEFAULT_CACHE, cache_path  # noqa: E402

SEED = 42
CV = 2
SIZE_COLS = ["n_blocks", "n_edges", "insns_used", "mean_block_len",
             "back_edges", "self_loops", "call_edges", "resolved_targets",
             "unresolved_targets", "n_sections"]


# ---------------------------------------------------------------------------
def fit_and_score(model_type, Xtr, ytr, Xte, cv=CV, seed=SEED):
    """Fit, predict, and also return an out-of-fold-calibrated threshold.

    The shared split is 45% ransomware in train and 74% in test. A classifier
    left at its default 0.5 cut is therefore operating at the wrong prior, and
    on this data that costs far more than the choice of representation: several
    rows below reach ROC-AUC 0.95+ while their macro-F1 sits near 0.55. The
    threshold is picked from `cross_val_predict` scores over TRAIN with the
    same folds -- out-of-fold, never on test -- and reported as a second row
    per (representation, model) so both operating points are visible.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                         cross_val_predict)
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    if model_type == "RF":
        clf = RandomForestClassifier(random_state=seed)
        grid = {"n_estimators": [100, 400], "max_depth": [None, 20]}
    elif model_type == "SVM-RBF":
        clf = SVC(kernel="rbf", random_state=seed)
        grid = {"C": [1, 10], "gamma": [0.1, 0.01]}
    elif model_type == "MLP":
        clf = MLPClassifier(early_stopping=True, max_iter=500, random_state=seed)
        grid = {"hidden_layer_sizes": [(500,)], "learning_rate_init": [0.001]}
    elif model_type == "LR":
        clf = LogisticRegression(max_iter=4000, random_state=seed)
        grid = {"C": [0.1, 1, 10]}
    else:
        raise ValueError(model_type)

    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    gs = GridSearchCV(clf, grid, cv=folds, scoring="f1_macro", n_jobs=-1)
    gs.fit(Xtr, ytr)
    best = gs.best_estimator_
    meth = "predict_proba" if hasattr(best, "predict_proba") else "decision_function"

    def _scores(est, X):
        s = getattr(est, meth)(X)
        return s[:, 1] if s.ndim == 2 else s

    y_pred = best.predict(Xte)
    score = _scores(best, Xte)

    oof = cross_val_predict(best, Xtr, ytr, cv=folds, method=meth, n_jobs=-1)
    oof = oof[:, 1] if oof.ndim == 2 else oof
    cands = np.unique(np.quantile(oof, np.linspace(0.01, 0.99, 99)))
    thr, best_f1 = float(np.median(oof)), -1.0
    for t in cands:
        f = f1_score(ytr, (oof >= t).astype(int), average="macro")
        if f > best_f1:
            thr, best_f1 = float(t), float(f)

    return dict(y_pred=y_pred, score=score,
                best_params={k: str(v) for k, v in gs.best_params_.items()},
                cv_best_f1=float(gs.best_score_),
                threshold=thr, oof_macro_f1=best_f1,
                y_pred_cal=(score >= thr).astype(int))


def _scale(Xtr, Xte):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


# ---------------------------------------------------------------------------
def run(dataset: str, cache_dir: Path, out_dir: Path, iters: int,
        blocks: int, insns: int, min_df: int, dim: int,
        pvdbow_seconds: float, skip_pvdbow: bool) -> dict:
    t_start = time.time()
    split = add_val_fold(load_split(dataset))
    stats = pd.read_csv(REPO / "results" / "graph2vec" / "graph_stats.csv")
    stats = stats.drop_duplicates("sha256").set_index("sha256")

    caches = {}
    for tree in ("mendeley", "balanced_goodware"):
        caches.update(E.load_cache(cache_path(cache_dir, tree, iters, blocks,
                                              insns)))

    have = split["sha256"].isin(caches.keys())
    if not have.all():
        print(f"  !! {int((~have).sum())} samples have no cached graph; dropped")
        split = split[have].reset_index(drop=True)

    docs = [caches[s] for s in split["sha256"]]
    y = split["label"].to_numpy()
    is_train = (split["split"] == "train").to_numpy()
    is_test = ~is_train
    arch = split["arch"].to_numpy()
    fam = split["family"].to_numpy()

    print(f"[{dataset}] {len(split)} graphs, {is_train.sum()} train / "
          f"{is_test.sum()} test")

    X, vocab, df = E.build_counts(docs, is_train, min_df=min_df)
    print(f"  WL vocabulary (train, min_df={min_df}): {len(vocab):,} labels, "
          f"nnz={X.nnz:,}")
    Xt, _ = E.tfidf_fit_transform(X, is_train)

    reps: dict[str, tuple] = {}

    # -- WL TF-IDF (sparse) : the WL subtree kernel feature map --------------
    reps["wl_tfidf"] = (Xt[is_train], Xt[is_test], ["LR"], False)

    # -- WL TF-IDF -> SVD ---------------------------------------------------
    t0 = time.time()
    Z, sv = E.svd_fit_transform(Xt, is_train, dim=dim, seed=SEED)
    print(f"  SVD({Z.shape[1]}) in {time.time()-t0:.0f}s, "
          f"explained var {sv.explained_variance_ratio_.sum():.3f}")
    reps["wl_svd"] = (Z[is_train], Z[is_test], ["RF", "SVM-RBF", "MLP"], True)

    # -- graph2vec (PV-DBOW) ------------------------------------------------
    # Cached: the fit is the only expensive step in this script and rerunning
    # the classifier grid must not cost another 15 minutes of SGD.
    if not skip_pvdbow:
        pvc = cache_dir / (f"pvdbow_{dataset}_d{dim}_i{iters}_b{blocks}"
                           f"_n{insns}_df{min_df}_s{SEED}.npz")
        if pvc.exists():
            z = np.load(pvc)
            Gtr, Gte = z["train"], z["test"]
            print(f"  PV-DBOW loaded from {pvc.name}")
        else:
            t0 = time.time()
            pv = E.PVDBOW(dim=dim, seed=SEED, max_seconds=pvdbow_seconds)
            pv.fit(X[is_train], log=print)
            Gtr = E.l2norm(pv.Wd_train)
            Gte = E.l2norm(pv.infer(X[is_test], log=print))
            print(f"  PV-DBOW in {time.time()-t0:.0f}s")
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(pvc, train=Gtr, test=Gte)
        reps["graph2vec"] = (Gtr, Gte, ["RF", "SVM-RBF", "MLP"], True)

    # -- size-only control --------------------------------------------------
    S = stats.loc[split["sha256"], SIZE_COLS].to_numpy(dtype=np.float64)
    reps["size_only"] = (S[is_train], S[is_test], ["RF", "LR"], True)

    results = []
    for rep, (Xtr, Xte, models, scale) in reps.items():
        for m in models:
            t0 = time.time()
            A, B = (_scale(Xtr, Xte) if scale else (Xtr, Xte))
            f = fit_and_score(m, A, y[is_train], B)
            common = dict(representation=rep, model=m,
                          best_params=f["best_params"],
                          cv_best_f1_macro=round(f["cv_best_f1"], 4),
                          fit_seconds=round(time.time() - t0, 1),
                          n_features=int(Xtr.shape[1]))
            r = build_result(y[is_test], f["y_pred"], f["score"],
                             arch[is_test], fam[is_test],
                             decision="argmax (default 0.5)", **common)
            rc = build_result(y[is_test], f["y_pred_cal"], f["score"],
                              arch[is_test], fam[is_test],
                              decision="threshold from train out-of-fold scores",
                              threshold=round(f["threshold"], 6),
                              oof_macro_f1=round(f["oof_macro_f1"], 4), **common)
            results.extend([r, rc])
            print(f"  {rep:>10s} {m:8s} macroF1={r['macro_f1']:.3f} "
                  f"(calibrated {rc['macro_f1']:.3f}) auc="
                  f"{(r['roc_auc'] or 0):.3f} FPR={r['false_positive_rate']:.3f}"
                  f" ({time.time()-t0:.0f}s)")

    samples = split_summary(split, "fold")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_metrics(
        out_dir / "metrics.json",
        experiment=f"graph2vec/{dataset}",
        description=("CFG basic-block graphs from linear-sweep disassembly; "
                     "WL subtree documents; TF-IDF/SVD, PV-DBOW (graph2vec) "
                     "and a graph-size control, on the shared cohort split"),
        samples=samples,
        results=results,
        elapsed_seconds=time.time() - t_start,
        config=dict(max_blocks=blocks, max_insns=insns, wl_iterations=iters,
                    min_df=min_df, embedding_dim=dim, seed=SEED, cv_folds=CV,
                    graph2vec_impl="PV-DBOW in numpy (karateclub/gensim have "
                                   "no cp314 wheel)"),
        graph_stats=_graph_stat_block(stats.loc[split["sha256"]], y, arch),
    )
    print(f"  wrote {out_dir/'metrics.json'}")
    return {"dataset": dataset, "results": results}


def _graph_stat_block(st: pd.DataFrame, y, arch) -> dict:
    st = st.copy()
    st["label"] = y
    st["arch"] = arch
    out = {}
    for key, grp in (("by_class", ["label"]), ("by_class_arch", ["label", "arch"])):
        g = st.groupby(grp)
        blk = {}
        for name, sub in g:
            k = "|".join(str(x) for x in (name if isinstance(name, tuple) else (name,)))
            blk[k] = {
                "n": int(len(sub)),
                "blocks_median": float(sub["n_blocks"].median()),
                "blocks_mean": round(float(sub["n_blocks"].mean()), 1),
                "edges_median": float(sub["n_edges"].median()),
                "edges_mean": round(float(sub["n_edges"].mean()), 1),
                "mean_block_len": round(float(sub["mean_block_len"].mean()), 3),
                "back_edges_median": float(sub["back_edges"].median()),
                "frac_at_block_cap": round(float(sub["block_capped"].mean()), 3),
                "frac_insn_truncated": round(float(sub["truncated"].mean()), 3),
                "unresolved_target_frac": round(float(
                    (sub["unresolved_targets"] /
                     (sub["unresolved_targets"] + sub["resolved_targets"])
                     .replace(0, np.nan)).mean()), 3),
            }
        out[key] = blk
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mendeley", "balanced", "both"],
                    default="both")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--max-blocks", type=int, default=5000)
    ap.add_argument("--max-insns", type=int, default=80000)
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--pvdbow-seconds", type=float, default=600.0)
    ap.add_argument("--skip-pvdbow", action="store_true")
    ap.add_argument("--results-dir", default=str(REPO / "results" / "graph2vec"))
    a = ap.parse_args()

    dsets = ["mendeley", "balanced"] if a.dataset == "both" else [a.dataset]
    for ds in dsets:
        run(ds, Path(a.cache_dir), Path(a.results_dir) / ds, a.iters,
            a.max_blocks, a.max_insns, a.min_df, a.dim, a.pvdbow_seconds,
            a.skip_pvdbow)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
