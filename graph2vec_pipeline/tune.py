#!/usr/bin/env python3
"""Group-cross-validated hyper-parameter search for the CFG / graph2vec track.

Everything here happens on TRAIN rows only. The test half of the shared split
is never loaded by this module -- `final_eval.py` is the only place it is
touched, once, for the configuration this search selects.

Protocol
--------
* folds: `StratifiedGroupKFold(n_splits=5)` over `family_or_group`, so a whole
  ransomware family (and, on `balanced`, a whole source project) is held out.
  The old harness used `StratifiedKFold(n_splits=2)` with no groups at all,
  which is why it reported CV macro-F1 0.98 on `balanced` against a test score
  of 0.53.
* selection metric: pooled out-of-fold macro-F1 at the classifier's own
  decision (argmax). Balanced accuracy, ROC-AUC and the macro-F1 reachable by
  moving the threshold are reported next to it; the last one is optimistic by
  construction (the threshold is chosen on the same OOF scores) and is there to
  separate "bad ordering" from "bad operating point".
* every configuration evaluated is appended to `cv_search.csv`.

What is fitted where
--------------------
The WL hashing and the min_df vocabulary filter use the dataset's TRAIN split
(never test). Inside the CV loop, the IDF weights, the SVD basis, the feature
scaler and the classifier are refitted on each fold's training part. PV-DBOW is
the one exception: it is fitted once on the whole train split, because a refit
per fold costs minutes. It is unsupervised -- it never sees a label -- but a
CV fold's held-out graphs did contribute to its label vectors, so `graph2vec`
rows here are slightly optimistic relative to the sparse rows. That is called
out again wherever a graph2vec row is reported.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cnn_vit_pipeline.cohort import load_split  # noqa: E402
from graph2vec_pipeline import wl  # noqa: E402
from graph2vec_pipeline.cfg import (E_BRANCH, E_BRANCH_XSEC, E_CALL,  # noqa: E402
                                    E_CALL_XSEC, E_EXTERN, E_FALL)
from graph2vec_pipeline.graph_cache import (DEFAULT_CACHE, cache_file,  # noqa: E402
                                            load_graphs)

SEED = 42
N_FOLDS = 5


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Construction:
    """How the graph is cut out of the disassembly."""
    max_blocks: int = 20_000
    windows: int = 1
    calls: bool = True
    cross_section: bool = True
    extern: bool = True
    max_insns: int = 80_000

    def keep_edges(self):
        keep = [E_FALL, E_BRANCH]
        if self.cross_section:
            keep.append(E_BRANCH_XSEC)
        if self.calls:
            keep.append(E_CALL)
            if self.cross_section:
                keep.append(E_CALL_XSEC)
        if self.extern and self.calls:
            keep.append(E_EXTERN)
        return tuple(keep)

    def key(self):
        return (f"b{self.max_blocks}/w{self.windows}"
                f"/calls{int(self.calls)}/xs{int(self.cross_section)}"
                f"/ext{int(self.extern)}")


@dataclass(frozen=True)
class Rep:
    """How the graph becomes a feature vector."""
    kind: str = "wl_tfidf"          # wl_tfidf | wl_svd | graph2vec | size_only
    labelling: str = "class"
    h: int = 2
    edge_labels: bool = False
    norm: str = "tfidf"             # tfidf | binary | log | raw
    min_df: int = 3
    with_size: bool = False
    dim: int = 128                  # wl_svd / graph2vec
    pv_epochs: int = 12
    pv_min_count: int = 0

    def key(self):
        s = (f"{self.kind}/{self.labelling}/h{self.h}/el{int(self.edge_labels)}"
             f"/{self.norm}/df{self.min_df}/size{int(self.with_size)}")
        if self.kind in ("wl_svd", "graph2vec"):
            s += f"/d{self.dim}"
        if self.kind == "graph2vec":
            s += f"/ep{self.pv_epochs}"
        return s


@dataclass(frozen=True)
class Model:
    name: str = "LR"
    params: tuple = ()               # tuple of (k, v) pairs; hashable
    arch_weight: bool = False

    def kw(self):
        return dict(self.params)

    def key(self):
        p = ",".join(f"{k}={v}" for k, v in self.params)
        return f"{self.name}({p}){'+archw' if self.arch_weight else ''}"


def make_model(m: Model, seed: int = SEED):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC, LinearSVC
    kw = m.kw()
    if m.name == "LR":
        return LogisticRegression(max_iter=4000, random_state=seed,
                                  solver=kw.pop("solver", "liblinear"), **kw)
    if m.name == "LinearSVC":
        return LinearSVC(random_state=seed, max_iter=8000, dual="auto", **kw)
    if m.name == "RF":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **kw)
    if m.name == "SVM-RBF":
        return SVC(kernel="rbf", random_state=seed, **kw)
    if m.name == "MLP":
        return MLPClassifier(early_stopping=True, max_iter=600,
                             random_state=seed, **kw)
    raise ValueError(m.name)


SUPPORTS_WEIGHT = {"LR", "LinearSVC", "RF", "SVM-RBF"}


# ---------------------------------------------------------------------------
# feature building
# ---------------------------------------------------------------------------
class Features:
    """Builds (and memoises) feature matrices for one dataset's TRAIN rows."""

    def __init__(self, dataset: str, cache_dir: Path, rows_mask: str = "train"):
        self.dataset = dataset
        self.cache_dir = Path(cache_dir)
        split = load_split(dataset)
        self.split_all = split
        self.use = (split["split"] == "train").to_numpy() if rows_mask == "train" \
            else np.ones(len(split), bool)
        self.split = split[self.use].reset_index(drop=True)
        self.y = self.split["label"].to_numpy()
        self.groups = self.split["family_or_group"].to_numpy()
        self.arch = self.split["arch"].to_numpy()
        self.family = self.split["family"].to_numpy()
        self.is_train = (self.split["split"] == "train").to_numpy()
        self._gs: dict = {}
        self._batch: dict = {}
        self._layers: dict = {}
        self._pv: dict = {}

    # -- graphs ----------------------------------------------------------
    def graphs(self, c: Construction):
        k = (c.windows, c.max_insns)
        if k in self._gs:
            return self._gs[k]
        paths = [cache_file(self.cache_dir, t, 20_000, c.max_insns, c.windows,
                            True, True)
                 for t in ("mendeley", "balanced_goodware")]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"graph cache missing: {missing[0].name}. Run "
                f"graph2vec_pipeline/graph_cache.py --windows {c.windows}")
        # one at a time: each windows setting is ~0.7-1GB of node and edge
        # arrays, and holding w1 and w4 together is what pushes this process
        # into swap on a 32GB box that is also running the other harnesses.
        # Re-reading the .npz costs seconds; paging costs minutes.
        self._gs.clear()
        gs = load_graphs(paths)
        self._gs[k] = gs
        return gs

    def batch(self, c: Construction) -> wl.Batch:
        if c in self._batch:
            return self._batch[c]
        gs = self.graphs(c)
        rows = np.array([gs.index[s] for s in self.split["sha256"]],
                        dtype=np.int64)
        b = wl.assemble(gs, rows, max_blocks=c.max_blocks,
                        keep_edges=c.keep_edges())
        self._batch[c] = b
        if len(self._batch) > 3:                  # keep memory bounded
            for old in list(self._batch)[:-3]:
                self._batch.pop(old, None)
        return b

    # -- WL layers -------------------------------------------------------
    # Layers are cumulative: depth-h documents are depths 0..h stacked. So the
    # deepest h in a stage is computed once and every shallower h is a prefix.
    h_build = 4

    def layers(self, c: Construction, labelling: str, edge_labels: bool,
               min_df: int, h_max: int | None = None):
        h_max = self.h_build if h_max is None else h_max
        key = (c, labelling, edge_labels, h_max, min_df)
        if key in self._layers:
            return self._layers[key]
        b = self.batch(c)
        mats = [wl.layer_matrix(b.gid, lab, b.n_graphs, self.is_train, min_df)
                for lab in wl.wl_layers(b, h_max, labelling, edge_labels)]
        self._layers.clear()                      # one at a time; they are big
        self._layers[key] = mats
        return mats

    def counts(self, c: Construction, r: Rep):
        mats = self.layers(c, r.labelling, r.edge_labels, r.min_df,
                           max(r.h, self.h_build))
        use = [m for m in mats[:r.h + 1] if m.shape[1]]
        if not use:
            return sparse.csr_matrix((len(self.y), 0), dtype=np.float32)
        return sparse.hstack(use, format="csr")

    def size(self, c: Construction) -> np.ndarray:
        return wl.size_features(self.batch(c))

    # -- PV-DBOW (unsupervised, fitted once on the whole train split) -----
    def pvdbow(self, c: Construction, r: Rep) -> np.ndarray:
        key = (c, r.labelling, r.h, r.edge_labels, r.min_df, r.dim,
               r.pv_epochs)
        if key in self._pv:
            return self._pv[key]
        from graph2vec_pipeline import embed as E
        X = self.counts(c, r)
        pv = E.PVDBOW(dim=r.dim, seed=SEED, epochs=r.pv_epochs,
                      max_seconds=240.0, max_steps=6000, max_steps_infer=3000)
        Xtr = X[self.is_train]
        pv.fit(Xtr)
        Z = np.zeros((X.shape[0], r.dim), dtype=np.float32)
        Z[self.is_train] = E.l2norm(pv.Wd_train)
        if (~self.is_train).any():
            Z[~self.is_train] = E.l2norm(pv.infer(X[~self.is_train]))
        self._pv.clear()
        self._pv[key] = Z
        return Z

    # -- the thing a model actually sees ---------------------------------
    def build(self, c: Construction, r: Rep):
        """Returns (X, needs_scaling, needs_infold_tfidf)."""
        if r.kind == "size_only":
            return self.size(c), True, False
        X = self.counts(c, r)
        if r.norm == "binary":
            X = X.copy()
            X.data[:] = 1.0
        elif r.norm == "log":
            X = X.copy()
            X.data = np.log1p(X.data).astype(np.float32)
        if r.kind == "graph2vec":
            return self.pvdbow(c, r), True, False
        infold_tfidf = (r.norm == "tfidf")
        if r.with_size:
            S = self.size(c)
            return (X, S), True, infold_tfidf
        return X, (r.kind == "wl_svd"), infold_tfidf


# ---------------------------------------------------------------------------
# cross-validation
# ---------------------------------------------------------------------------
def arch_sample_weight(y, arch) -> np.ndarray:
    """Balance the four (label, arch) cells, not just the two classes.

    Ransomware in train is 95% x86: 42 x64 ransomware against 862 x86 ones.
    Plain class_weight='balanced' does nothing about that.
    """
    w = np.ones(len(y), dtype=np.float64)
    cells = {}
    for a in np.unique(arch):
        for lab in (0, 1):
            m = (arch == a) & (y == lab)
            if m.sum():
                cells[(a, lab)] = m
    target = len(y) / len(cells)
    for m in cells.values():
        w[m] = target / m.sum()
    return w / w.mean()


def _fit_fold(model: Model, Xtr, ytr, Xte, wtr=None):
    est = make_model(model)
    if wtr is not None and model.name in SUPPORTS_WEIGHT:
        est.fit(Xtr, ytr, sample_weight=wtr)
    else:
        est.fit(Xtr, ytr)
    if hasattr(est, "predict_proba"):
        s = est.predict_proba(Xte)[:, 1]
    else:
        s = est.decision_function(Xte)
        s = s[:, 1] if getattr(s, "ndim", 1) == 2 else s
    return est.predict(Xte), s


def _prep(Xpair, tr_idx, te_idx, needs_scaling, infold_tfidf, svd_dim=0):
    """In-fold IDF / SVD / scaling. Nothing here sees the held-out labels."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.preprocessing import StandardScaler

    extra = None
    if isinstance(Xpair, tuple):
        X, extra = Xpair
    else:
        X = Xpair
    if sparse.issparse(X):
        A, B = X[tr_idx], X[te_idx]
        if infold_tfidf:
            tf = TfidfTransformer(sublinear_tf=True).fit(A)
            A, B = tf.transform(A), tf.transform(B)
        if svd_dim:
            d = int(min(svd_dim, min(A.shape) - 1))
            sv = TruncatedSVD(n_components=d, random_state=SEED).fit(A)
            A, B = sv.transform(A), sv.transform(B)
            sc = StandardScaler().fit(A)
            A, B = sc.transform(A), sc.transform(B)
    else:
        A, B = X[tr_idx], X[te_idx]
        if needs_scaling:
            sc = StandardScaler().fit(A)
            A, B = sc.transform(A), sc.transform(B)
    if extra is not None:
        sc = StandardScaler().fit(extra[tr_idx])
        Ea, Eb = sc.transform(extra[tr_idx]), sc.transform(extra[te_idx])
        if sparse.issparse(A):
            A = sparse.hstack([A, sparse.csr_matrix(Ea)], format="csr")
            B = sparse.hstack([B, sparse.csr_matrix(Eb)], format="csr")
        else:
            A = np.hstack([A, Ea])
            B = np.hstack([B, Eb])
    return A, B


def best_threshold(y, s, metric="macro_f1"):
    from sklearn.metrics import balanced_accuracy_score, f1_score
    fn = (f1_score if metric == "macro_f1" else None)
    cands = np.unique(np.quantile(s, np.linspace(0.005, 0.995, 199)))
    best, thr = -1.0, float(np.median(s))
    for t in cands:
        p = (s >= t).astype(int)
        v = (f1_score(y, p, average="macro") if fn else
             balanced_accuracy_score(y, p))
        if v > best:
            best, thr = float(v), float(t)
    return thr, best


def _nbytes(M) -> int:
    if sparse.issparse(M):
        return M.data.nbytes + M.indices.nbytes + M.indptr.nbytes
    return M.nbytes


def cv_eval(F: Features, c: Construction, r: Rep, m: Model,
            folds=None, prep_cache: dict | None = None) -> dict:
    from sklearn.metrics import (balanced_accuracy_score, f1_score,
                                 roc_auc_score)
    from sklearn.model_selection import StratifiedGroupKFold
    t0 = time.time()
    svd_dim = r.dim if r.kind == "wl_svd" else 0
    Xpair, needs_scaling, infold_tfidf = F.build(c, r)
    n = len(F.y)
    if folds is None:
        folds = list(StratifiedGroupKFold(
            n_splits=N_FOLDS, shuffle=True,
            random_state=SEED).split(np.zeros(n), F.y, F.groups))
    w = arch_sample_weight(F.y, F.arch) if m.arch_weight else None

    oof_pred = np.zeros(n, dtype=int)
    oof_score = np.zeros(n, dtype=float)
    fold_f1 = []
    for fi, (tr_idx, te_idx) in enumerate(folds):
        hit = None if prep_cache is None else prep_cache.get(fi)
        if hit is None:
            A, B = _prep(Xpair, tr_idx, te_idx, needs_scaling, infold_tfidf,
                         svd_dim)
            # the in-fold IDF / SVD / scaling is the same for every model on
            # this representation; reuse it across the classifier grid.
            if (prep_cache is not None
                    and _nbytes(A) + _nbytes(B) < 400_000_000):
                prep_cache[fi] = (A, B)
        else:
            A, B = hit
        p, s = _fit_fold(m, A, F.y[tr_idx], B,
                         None if w is None else w[tr_idx])
        oof_pred[te_idx], oof_score[te_idx] = p, s
        fold_f1.append(f1_score(F.y[te_idx], p, average="macro"))

    thr_f1, f1_at_thr = best_threshold(F.y, oof_score, "macro_f1")
    thr_ba, ba_at_thr = best_threshold(F.y, oof_score, "balanced_accuracy")
    X0 = Xpair[0] if isinstance(Xpair, tuple) else Xpair
    n_feat = X0.shape[1] + (Xpair[1].shape[1] if isinstance(Xpair, tuple) else 0)
    if svd_dim:
        n_feat = svd_dim + (Xpair[1].shape[1] if isinstance(Xpair, tuple) else 0)
    return dict(
        dataset=F.dataset, construction=c.key(), representation=r.key(),
        model=m.key(), rep_kind=r.kind, model_name=m.name,
        labelling=r.labelling, h=r.h, edge_labels=r.edge_labels, norm=r.norm,
        min_df=r.min_df, with_size=r.with_size, max_blocks=c.max_blocks,
        windows=c.windows, calls=c.calls, cross_section=c.cross_section,
        extern=c.extern, arch_weight=m.arch_weight, dim=r.dim,
        n_features=int(n_feat), n_train=int(n),
        cv_macro_f1=float(f1_score(F.y, oof_pred, average="macro")),
        cv_macro_f1_foldmean=float(np.mean(fold_f1)),
        cv_macro_f1_foldstd=float(np.std(fold_f1)),
        cv_balanced_acc=float(balanced_accuracy_score(F.y, oof_pred)),
        cv_auc=float(roc_auc_score(F.y, oof_score)),
        cv_macro_f1_at_oof_thr=float(f1_at_thr),
        cv_balanced_acc_at_oof_thr=float(ba_at_thr),
        oof_threshold_f1=float(thr_f1), oof_threshold_ba=float(thr_ba),
        seconds=round(time.time() - t0, 1),
    )


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------
class Search:
    def __init__(self, F: Features, out_csv: Path, verbose=True):
        self.F = F
        self.rows: list[dict] = []
        self.objs: list[tuple] = []          # (Construction, Rep, Model)
        self.out_csv = Path(out_csv)
        self.verbose = verbose
        from sklearn.model_selection import StratifiedGroupKFold
        self.folds = list(StratifiedGroupKFold(
            n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(
                np.zeros(len(F.y)), F.y, F.groups))

    _prep_cache: dict = {}
    _prep_key = None

    def run(self, stage: str, c: Construction, r: Rep, m: Model):
        key = (c, r)
        if key != self._prep_key:
            self._prep_cache = {}
            self._prep_key = key
        row = cv_eval(self.F, c, r, m, self.folds, self._prep_cache)
        row["stage"] = stage
        self.rows.append(row)
        self.objs.append((c, r, m))
        if self.verbose:
            print(f"  [{stage}] {c.key()} | {r.key()} | {m.key()} "
                  f"-> F1={row['cv_macro_f1']:.4f} "
                  f"(thr {row['cv_macro_f1_at_oof_thr']:.4f}) "
                  f"bal={row['cv_balanced_acc']:.4f} "
                  f"auc={row['cv_auc']:.4f} [{row['seconds']}s]", flush=True)
        self.flush()
        return row

    def flush(self):
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(self.out_csv, index=False)

    def best(self, **filt):
        idx = [i for i, r in enumerate(self.rows)
               if all(r.get(k) == v for k, v in filt.items())]
        if not idx:
            return None
        return self.rows[max(idx, key=lambda i: self.rows[i]["cv_macro_f1"])]

    def best_obj(self, **filt):
        """(row, Construction, Rep, Model) for the best matching row."""
        idx = [i for i, r in enumerate(self.rows)
               if all(r.get(k) == v for k, v in filt.items())]
        if not idx:
            return None
        i = max(idx, key=lambda j: self.rows[j]["cv_macro_f1"])
        return (self.rows[i],) + self.objs[i]


# ---------------------------------------------------------------------------
# the staged search. One thing changes at a time (coordinate ascent), which is
# what "stop when the marginal CV gain is below 0.005" is defined against.
# ---------------------------------------------------------------------------
MIN_GAIN = 0.005


def _rep(base: Rep, **kw) -> Rep:
    return Rep(**{**base.__dict__, **kw})


def _con(base: Construction, **kw) -> Construction:
    return Construction(**{**base.__dict__, **kw})


def staged_search(F: Features, out_csv: Path, quick: bool = False) -> Search:
    S = Search(F, out_csv)
    LR1 = Model("LR", (("C", 1.0),))

    # -- stage 0: the current pipeline's configuration, under group CV ------
    base_c = Construction(max_blocks=5_000, windows=1, calls=True,
                          cross_section=False, extern=False)
    base_r = Rep("wl_tfidf", "class", h=2, edge_labels=False, norm="tfidf",
                 min_df=5)
    S.run("0_baseline", base_c, base_r, LR1)
    S.run("0_baseline", base_c, _rep(base_r, kind="size_only"), LR1)

    # -- stage 1: graph construction, coordinate ascent ---------------------
    cur_c, cur_r = base_c, _rep(base_r, min_df=3)
    cur = S.run("1_construction", cur_c, cur_r, LR1)["cv_macro_f1"]
    trials = [
        ("max_blocks", [20_000]),
        ("cross_section", [True]),
        ("extern", [True]),
        ("calls", [False]),
        ("windows", [4]),
    ]
    for field_name, vals in trials:
        best_v, best_s = None, cur
        for v in vals:
            cand = _con(cur_c, **{field_name: v})
            s = S.run("1_construction", cand, cur_r, LR1)["cv_macro_f1"]
            if s > best_s:
                best_v, best_s = v, s
        if best_v is not None and best_s - cur >= MIN_GAIN:
            cur_c = _con(cur_c, **{field_name: best_v})
            cur = best_s
            print(f"  -> construction takes {field_name}={best_v} "
                  f"(CV {cur:.4f})", flush=True)
    for el in (True,):
        s = S.run("1_construction", cur_c, _rep(cur_r, edge_labels=el), LR1)
        if s["cv_macro_f1"] - cur >= MIN_GAIN:
            cur_r = _rep(cur_r, edge_labels=el)
            cur = s["cv_macro_f1"]
            print(f"  -> edge labels on (CV {cur:.4f})", flush=True)

    # -- stage 2: node labelling granularity x WL depth ---------------------
    labellings = ("class", "class_noflag", "dom_term", "seq_term", "deg_dom",
                  "lenbucket")
    depths = (1, 2, 3, 4)
    if quick:
        labellings, depths = ("class", "seq_term", "deg_dom"), (2, 3)
    best2 = None
    for lb in labellings:
        for h in depths:
            row = S.run("2_labelling", cur_c, _rep(cur_r, labelling=lb, h=h),
                        LR1)
            if best2 is None or row["cv_macro_f1"] > best2["cv_macro_f1"]:
                best2 = row
    if best2["cv_macro_f1"] - cur >= MIN_GAIN:
        cur_r = _rep(cur_r, labelling=best2["labelling"], h=int(best2["h"]))
        cur = best2["cv_macro_f1"]
        print(f"  -> labelling={cur_r.labelling} h={cur_r.h} (CV {cur:.4f})",
              flush=True)

    # -- stage 3: histogram normalisation, min_df, + size scalars -----------
    for norm in ("binary", "log"):
        S.run("3_norm", cur_c, _rep(cur_r, norm=norm), LR1)
    for mdf in (2, 5, 10):
        S.run("3_norm", cur_c, _rep(cur_r, min_df=mdf), LR1)
    S.run("3_norm", cur_c, _rep(cur_r, with_size=True), LR1)
    b3 = S.best(stage="3_norm")
    if b3 and b3["cv_macro_f1"] - cur >= MIN_GAIN:
        cur_r = _rep(cur_r, norm=b3["norm"], min_df=int(b3["min_df"]),
                     with_size=bool(b3["with_size"]))
        cur = b3["cv_macro_f1"]
        print(f"  -> {cur_r.key()} (CV {cur:.4f})", flush=True)

    # -- stage 4: dense embeddings against the histogram they compress ------
    for d in (50, 100, 300):
        for mdl in (Model("RF", (("n_estimators", 400),)),
                    Model("SVM-RBF", (("C", 10.0), ("gamma", "scale"))),
                    Model("LR", (("C", 10.0),))):
            S.run("4_embedding", cur_c,
                  _rep(cur_r, kind="wl_svd", dim=d, with_size=False), mdl)
    for d in (64, 128, 256):
        for mdl in (Model("SVM-RBF", (("C", 10.0), ("gamma", "scale"))),
                    Model("RF", (("n_estimators", 400),)),
                    Model("MLP", (("hidden_layer_sizes", (256,)),))):
            S.run("4_embedding", cur_c,
                  _rep(cur_r, kind="graph2vec", dim=d, with_size=False), mdl)

    # -- the size_only control, under the same folds ------------------------
    for mdl in (Model("LR", (("C", 1.0),)),
                Model("LR", (("C", 1.0), ("class_weight", "balanced"))),
                Model("RF", (("n_estimators", 600), ("min_samples_leaf", 2)))):
        S.run("4_control", cur_c, _rep(cur_r, kind="size_only"), mdl)

    # -- stage 5: real classifier grids on the two best representations -----
    sparse_grid = ([Model("LR", (("C", c),)) for c in (0.1, 1.0, 10.0, 100.0)]
                   + [Model("LR", (("C", c), ("class_weight", "balanced")))
                      for c in (0.1, 1.0, 10.0)]
                   + [Model("LinearSVC", (("C", c),)) for c in (0.01, 0.1, 1.0)]
                   + [Model("LinearSVC", (("C", c),
                                          ("class_weight", "balanced")))
                      for c in (0.01, 0.1, 1.0)])
    for mdl in sparse_grid:
        S.run("5_clf_sparse", cur_c, cur_r, mdl)

    dense_grid = (
        [Model("RF", (("n_estimators", n), ("max_depth", d),
                      ("min_samples_leaf", l)))
         for n in (300, 1000) for d in (None, 20) for l in (1, 3)]
        + [Model("SVM-RBF", (("C", c), ("gamma", g)))
           for c in (1.0, 10.0, 100.0) for g in ("scale", 0.01, 0.1)]
        + [Model("SVM-RBF", (("C", c), ("gamma", "scale"),
                             ("class_weight", "balanced")))
           for c in (10.0, 100.0)]
        + [Model("MLP", (("hidden_layer_sizes", hl), ("alpha", a)))
           for hl in ((256,), (512,), (256, 128)) for a in (1e-4, 1e-2)])
    b4 = S.best_obj(stage="4_embedding")
    if b4 is not None:
        dense_r = b4[2]
        for mdl in dense_grid:
            S.run("5_clf_dense", cur_c, dense_r, mdl)

    # -- stage 6: architecture-aware reweighting ----------------------------
    order = sorted(range(len(S.rows)), key=lambda i: -S.rows[i]["cv_macro_f1"])
    seen = set()
    for i in order:
        row = S.rows[i]
        c2, r2, m2 = S.objs[i]
        if m2.arch_weight or m2.name not in SUPPORTS_WEIGHT:
            continue
        k = (row["representation"], row["model"])
        if k in seen:
            continue
        seen.add(k)
        S.run("6_archweight", c2, r2, Model(m2.name, m2.params, True))
        if len(seen) >= 5:
            break
    S.flush()
    return S


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mendeley", "balanced", "both"],
                    default="both")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--out-dir", default=str(REPO / "results" / "graph2vec"
                                             / "tuned"))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    dsets = ["mendeley", "balanced"] if a.dataset == "both" else [a.dataset]
    for ds in dsets:
        t0 = time.time()
        print(f"=== {ds}: group-CV search on TRAIN only ===", flush=True)
        F = Features(ds, Path(a.cache_dir))
        print(f"  {len(F.y)} train graphs, {F.y.sum()} ransomware, "
              f"{len(set(F.groups))} groups", flush=True)
        out = Path(a.out_dir) / ds / "cv_search.csv"
        S = staged_search(F, out, quick=a.quick)
        b = S.best()
        print(f"  best: {b['construction']} | {b['representation']} | "
              f"{b['model']} -> CV macro-F1 {b['cv_macro_f1']:.4f}")
        print(f"  {len(S.rows)} configurations, {time.time()-t0:.0f}s")
        write_chosen(S, out.parent / "chosen.json")
    return 0


def _obj_dict(c: Construction, r: Rep, m: Model) -> dict:
    return {"construction": dict(c.__dict__), "rep": dict(r.__dict__),
            "model": {"name": m.name, "params": [list(p) for p in m.params],
                      "arch_weight": m.arch_weight}}


def load_obj(d: dict) -> tuple:
    c = Construction(**d["construction"])
    r = Rep(**d["rep"])
    m = Model(d["model"]["name"],
              tuple((k, tuple(v) if isinstance(v, list) else v)
                    for k, v in (tuple(p) for p in d["model"]["params"])),
              bool(d["model"]["arch_weight"]))
    return c, r, m


# -- reconstructing a configuration from a cv_search.csv row ----------------
# `chosen.json`'s "picks" carry the dataclasses verbatim; its "top5" and every
# row of cv_search.csv carry the flat columns plus the rendered `model` key.
# `final_eval` needs objects for the latter too, to score the top-5 CV
# configurations on test post hoc.
def _literal(s):
    import ast
    s = str(s).strip()
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s                      # "scale", "balanced", ...


def _as_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def parse_model_key(s: str) -> Model:
    """Inverse of `Model.key()`.

    `MLP(hidden_layer_sizes=(256, 128),alpha=0.0001)` is why the argument split
    has to track bracket depth rather than `str.split(",")`.
    """
    s = str(s).strip()
    arch_w = s.endswith("+archw")
    if arch_w:
        s = s[:-len("+archw")]
    if "(" not in s or not s.endswith(")"):
        raise ValueError(f"not a model key: {s!r}")
    name, body = s[:s.index("(")], s[s.index("(") + 1:s.rindex(")")]
    parts, depth, cur = [], 0, ""
    for ch in body:
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
            continue
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        cur += ch
    if cur.strip():
        parts.append(cur)
    params = []
    for p in parts:
        if not p.strip():
            continue
        k, _, v = p.partition("=")
        params.append((k.strip(), _literal(v)))
    return Model(name, tuple(params), arch_w)


def obj_from_row(row) -> tuple:
    """(Construction, Rep, Model) from a cv_search.csv row / chosen.json top5.

    Raises if the reconstruction does not render back to the row's own
    `construction` / `representation` / `model` strings, so a silent drift
    between what the search recorded and what gets fitted on test is
    impossible.
    """
    c = Construction(max_blocks=int(row["max_blocks"]),
                     windows=int(row["windows"]),
                     calls=_as_bool(row["calls"]),
                     cross_section=_as_bool(row["cross_section"]),
                     extern=_as_bool(row["extern"]))
    r = Rep(kind=str(row["rep_kind"]), labelling=str(row["labelling"]),
            h=int(row["h"]), edge_labels=_as_bool(row["edge_labels"]),
            norm=str(row["norm"]), min_df=int(row["min_df"]),
            with_size=_as_bool(row["with_size"]), dim=int(row["dim"]))
    m = parse_model_key(row["model"])
    for got, want, what in ((c.key(), str(row["construction"]), "construction"),
                            (r.key(), str(row["representation"]), "rep"),
                            (m.key(), str(row["model"]), "model")):
        if got != want:
            raise ValueError(f"{what} round-trip failed: {got!r} != {want!r}")
    return c, r, m


def write_chosen(S: Search, path: Path) -> dict:
    """The configurations that go to the single test evaluation.

    All five are picked by CV macro-F1 on TRAIN. They exist so the summary can
    report before/after, the shortcut control, and whether the dense embedding
    beats the histogram it compresses -- each on one test run, not on a search
    over the test set.
    """
    picks: list[tuple[str, tuple]] = []
    seen = set()

    def _add(name, got):
        if got is None:
            return
        row, c, r, m = got
        k = (c.key(), r.key(), m.key())
        if k in seen:
            return
        seen.add(k)
        picks.append((name, (row, c, r, m)))

    _add("baseline_h2_cap5000", S.best_obj(stage="0_baseline",
                                           rep_kind="wl_tfidf"))
    _add("tuned_best", S.best_obj())
    _add("tuned_best_sparse_histogram", S.best_obj(rep_kind="wl_tfidf"))
    for kind in ("wl_svd", "graph2vec"):
        _add(f"tuned_best_{kind}", S.best_obj(rep_kind=kind))
    _add("size_only_control", S.best_obj(rep_kind="size_only"))

    doc = {"dataset": S.F.dataset, "n_configurations": len(S.rows),
           "selection": "pooled out-of-fold macro-F1, StratifiedGroupKFold(5) "
                        "over family_or_group, TRAIN rows only",
           "picks": [{"name": n, "cv_macro_f1": row["cv_macro_f1"],
                      "cv_balanced_acc": row["cv_balanced_acc"],
                      "cv_auc": row["cv_auc"],
                      "cv_macro_f1_at_oof_thr": row["cv_macro_f1_at_oof_thr"],
                      **_obj_dict(c, r, m)}
                     for n, (row, c, r, m) in picks],
           "top5": [{k: v for k, v in r.items()}
                    for r in sorted(S.rows, key=lambda x: -x["cv_macro_f1"])[:5]]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    print(f"  wrote {path}")
    return doc


if __name__ == "__main__":
    raise SystemExit(main())
