#!/usr/bin/env python3
"""Family-holdout evaluation of the mnemonic TF-IDF calibration baseline.

    python family_holdout/run_tfidf.py --dataset both

The configuration is `rules_pipeline/train_eval.py`'s section (c) verbatim --
the pre-registered calibration baseline, not a tuned pick:

    TfidfVectorizer(analyzer="word", token_pattern=r"\\S+", ngram_range=(1, 3),
                    min_df=5, sublinear_tf=True, max_features=300_000)
    over the first MAX_MNEMS = 30_000 mnemonics of Shared/Extract*/mn/<sha>.txt
    LinearSVC(random_state=42, max_iter=5000) and
    LogisticRegression(max_iter=4000, random_state=42)
    C selected per fit by GridSearchCV({"C": [0.1, 1, 10]},
        cv=StratifiedKFold(2, shuffle=True, random_state=42), scoring=f1_macro)

`class_weight` is left at the scikit-learn default (None) because that is what
the baseline uses; the grid over C is part of the baseline's own definition,
so it is re-run inside every training set rather than frozen to the value the
fixed split happened to pick. Nothing here is chosen by looking at a test fold.

The vectoriser is refit on the training rows of EVERY fold and of EVERY LOFO
training set, so no held-out file contributes a vocabulary entry or an IDF
weight.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from family_holdout.common import (DATASETS, Folds, OUT_ROOT,  # noqa: E402
                                   check_kfold, check_lofo, write_model_dir, ALL_DATASETS, stream_path,
                                   lofo_placeholders, strip_lofo)

SEED = 42
MAX_MNEMS = 30_000                 # rules_pipeline.train_eval.MAX_MNEMS
NGRAM = (1, 3)
MIN_DF = 5
MAX_FEATURES = 300_000
C_GRID = [0.1, 1, 10]
PIPELINE = "tfidf"

# extraction trees are registered per corpus in family_holdout.common
# (TREE_OF_CORPUS); rows of the fold file carry `corpus`, so nothing here
# needs to know which dataset a row belongs to


def read_mnemonics(path: Path, cap: int) -> list:
    out: list = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            for tok in line.split():
                out.append(tok)
                if len(out) >= cap:
                    return out
    return out


STREAMS = ("mn", "mn_api", "mn_opclass")   # see asm_tool/rewrite_streams.py


def load_texts(folds: Folds, stream: str = "mn") -> list:
    """One whitespace-joined token string per fold-file row, from the `mn/`
    tree or one of the rewritten variants (`mn_api/`, `mn_opclass/`)."""
    texts = []
    for sha, corpus in zip(folds.sha, folds.corpus):
        texts.append(" ".join(read_mnemonics(
            stream_path(corpus, sha, stream), MAX_MNEMS)))
    return texts


def make_vec(ngram_max: int = NGRAM[1], max_features=MAX_FEATURES):
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(analyzer="word", token_pattern=r"\S+",
                           ngram_range=(1, ngram_max), min_df=MIN_DF, sublinear_tf=True,
                           max_features=max_features, dtype=np.float32)


def _grid(name):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    if name == "LogReg":
        return LogisticRegression(max_iter=4000, random_state=SEED)
    if name == "LinearSVC":
        return LinearSVC(random_state=SEED, max_iter=5000)
    raise ValueError(name)


def fit_score(name, Xtr, ytr, Xte, n_jobs: int = -1):
    """GridSearchCV over C exactly as the calibration baseline does it."""
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    gs = GridSearchCV(_grid(name), {"C": C_GRID}, cv=folds,
                      scoring="f1_macro", n_jobs=n_jobs).fit(Xtr, ytr)
    est = gs.best_estimator_
    pred = est.predict(Xte)
    score = (est.predict_proba(Xte)[:, 1] if hasattr(est, "predict_proba")
             else est.decision_function(Xte))
    return pred, np.asarray(score, dtype=float), gs.best_params_["C"], \
        float(gs.best_score_)


MODELS = ("LogReg", "LinearSVC")


def run_dataset(dataset: str, out_root: Path, stream: str = "mn",
                ngram_max: int = NGRAM[1], kfold_only: bool = False,
                max_features=MAX_FEATURES, n_jobs: int = -1) -> dict:
    """`ngram_max` != 3 or `kfold_only` is the phrase-length sweep: the model
    dir gets a `_ng1-<n>` suffix and, K-fold only, its LOFO fields are blank.
    `max_features=None` removes the vocabulary cap (the sweep uses it, since
    the cap keeps the most frequent phrases and so would cut long ones first);
    `n_jobs` bounds the parallel C-search fits, each of which copies X."""
    t_start = time.time()
    folds = Folds(dataset)
    check_kfold(folds)
    check_lofo(folds)
    print(f"[{dataset}] {folds.n} files, {len(folds.families)} families; "
          f"reading mnemonics ...", flush=True)
    t0 = time.time()
    texts = load_texts(folds, stream)
    print(f"[{dataset}] read in {time.time()-t0:.0f}s", flush=True)

    n = folds.n
    kf_score = {m: np.zeros(n) for m in MODELS}
    kf_pred = {m: np.zeros(n, dtype=int) for m in MODELS}
    kf_fold = folds.fold.copy()
    chosen = {m: {"kfold": {}, "lofo": {}} for m in MODELS}
    n_feat = {}

    # ---- K-fold ---------------------------------------------------------
    for f, tr, te in folds.kfold():
        t0 = time.time()
        vec = make_vec(ngram_max, max_features)
        Xtr = vec.fit_transform(texts[i] for i in tr)
        Xte = vec.transform(texts[i] for i in te)
        del vec                     # the fitted vocabulary is the big object
        n_feat[f"kfold_{f}"] = int(Xtr.shape[1])
        for m in MODELS:
            pred, score, C, cv = fit_score(m, Xtr, folds.y[tr], Xte, n_jobs)
            kf_pred[m][te], kf_score[m][te] = pred, score
            chosen[m]["kfold"][str(f)] = {"C": C, "cv_f1_macro": round(cv, 4)}
        print(f"  [{dataset}] kfold {f}: {Xtr.shape[1]} features, "
              f"{time.time()-t0:.0f}s", flush=True)

    # ---- LOFO -----------------------------------------------------------
    lofo = {m: (lofo_placeholders(folds) if kfold_only else {}) for m in MODELS}
    for k, (fam, tr, te) in enumerate([] if kfold_only else folds.lofo(), 1):
        t0 = time.time()
        vec = make_vec(ngram_max, max_features)
        Xtr = vec.fit_transform(texts[i] for i in tr)
        Xte = vec.transform(texts[i] for i in te)
        for m in MODELS:
            pred, score, C, cv = fit_score(m, Xtr, folds.y[tr], Xte, n_jobs)
            lofo[m][fam] = (score, pred)
            chosen[m]["lofo"][fam] = {"C": C, "cv_f1_macro": round(cv, 4)}
        print(f"  [{dataset}] lofo {k}/{len(folds.families)} {fam}: n={len(te)} "
              f"{time.time()-t0:.0f}s", flush=True)

    # ---- write ----------------------------------------------------------
    elapsed = time.time() - t_start
    out = {}
    for m in MODELS:
        cfg = {
            "pipeline": PIPELINE,
            "model": f"mnemonic_tfidf_1_{ngram_max}+{m}",
            "source": "rules_pipeline/train_eval.py section (c), calibration "
                      "baseline (pre-registered; not the tuned pick)",
            "features": {
                "input": f"Shared/Extract*/{stream}/<sha256>.txt"
                         + (" (raw mnemonic stream)" if stream == "mn" else
                            " (rewritten stream, asm_tool/rewrite_streams.py)"),
                "stream": stream,
                "max_mnemonics": MAX_MNEMS,
                "vectoriser": (f"TfidfVectorizer(analyzer=word, "
                               f"token_pattern=\\S+, ngram_range={(1, ngram_max)}, "
                               f"min_df={MIN_DF}, sublinear_tf=True, "
                               f"max_features={max_features})"),
                "fit_on": "training rows of each fold / each LOFO train set only",
                "n_features_per_kfold": n_feat,
            },
            "classifier": {
                "name": m,
                "estimator": ("LogisticRegression(max_iter=4000, random_state=42)"
                              if m == "LogReg" else
                              "LinearSVC(random_state=42, max_iter=5000)"),
                "class_weight": None,
                "C_grid": C_GRID,
                "C_selection": ("GridSearchCV(cv=StratifiedKFold(2, "
                                "shuffle=True, random_state=42), "
                                "scoring=f1_macro) inside the training rows"),
                "C_chosen": chosen[m],
            },
            "decision": "argmax (the estimator's own predict)",
            "seed": SEED,
            "schemes": ["kfold"] if kfold_only else ["kfold", "lofo"],
        }
        suffix = (("" if stream == "mn" else f"_{stream}")
                  + ("" if ngram_max == NGRAM[1] else f"_ng1-{ngram_max}")
                  + ("" if max_features == MAX_FEATURES else "_nocap"))
        d = out_root / dataset / PIPELINE / f"{m}{suffix}"
        out[m] = write_model_dir(
            d, folds, PIPELINE, m, kf_score[m], kf_pred[m], kf_fold,
            lofo[m], cfg, elapsed,
            description=(f"mnemonic 1-{ngram_max}-gram TF-IDF + {m} (rules_pipeline "
                         f"calibration baseline), family-holdout K-fold and "
                         f"LOFO on the {dataset} cohort"))
        if kfold_only:
            strip_lofo(d, "phrase-length sweep: K-fold only (97 LOFO fits per "
                          "setting are outside the budget); the 1-3 model dir has LOFO")
        print(f"  wrote {d}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=(*ALL_DATASETS, "both"), default="both")
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--stream", choices=STREAMS, default="mn",
                    help="which token tree to read; a non-default stream is "
                         "written to <model>_<stream>/")
    ap.add_argument("--ngram-max", type=int, default=NGRAM[1],
                    help="longest phrase (n-gram) length; the committed baseline is 3, "
                         "other values are written to <model>_ng1-<n>/")
    ap.add_argument("--no-feature-cap", action="store_true",
                    help="keep every phrase above min_df (written to <model>..._nocap/)")
    ap.add_argument("--jobs", type=int, default=-1,
                    help="parallel fits in the inner C search (each copies X)")
    ap.add_argument("--kfold-only", action="store_true",
                    help="skip leave-one-family-out (its fields are left blank)")
    a = ap.parse_args()
    ds = DATASETS if a.dataset == "both" else (a.dataset,)
    for d in ds:
        run_dataset(d, Path(a.out), a.stream, a.ngram_max, a.kfold_only,
                    None if a.no_feature_cap else MAX_FEATURES, a.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
