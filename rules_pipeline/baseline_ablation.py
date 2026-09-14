#!/usr/bin/env python3
"""How much of the classical baseline's score comes from window and n-gram order?

    python rules_pipeline/baseline_ablation.py

`results/summary.md` reports macro-F1 0.75-0.81 (Mendeley goodware) and
0.59-0.68 (Goodware_Balanced) for tokenizer + word2vec + RF/SVM/MLP, with the
instruction stream truncated at 5,000 instructions. The rules pipeline's
calibration baseline -- plain mnemonic TF-IDF + logistic regression over 30,000
mnemonics -- scores far higher. Two things changed at once (the window and the
representation), so this sweeps them apart:

    window in {1k, 5k, 30k} mnemonics  x  n-gram order in {1, 1-2, 1-3}

on both datasets, with the same split, the same seed and the same 2-fold
f1_macro grid search over C. Writes results/rules/baseline_ablation.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from cnn_vit_pipeline.cohort import add_val_fold, build_result, load_split  # noqa: E402
from rules_pipeline.train_eval import MnemCorpus, load_encoded  # noqa: E402

WINDOWS = [1_000, 5_000, 30_000]
ORDERS = [(1, 1), (1, 2), (1, 3)]
SEED = 42


def run(dataset, corpus, out):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    split = add_val_fold(load_split(dataset))
    enc = load_encoded(split, corpus)
    y = split["label"].to_numpy()
    arch = split["arch"].to_numpy()
    fam = split["family"].to_numpy()
    tr = np.flatnonzero((split["split"] == "train").to_numpy())
    te = np.flatnonzero((split["split"] == "test").to_numpy())
    folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)

    for w in WINDOWS:
        texts = [corpus.text(a[:w]) for a in enc]
        for lo, hi in ORDERS:
            t0 = time.time()
            vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+",
                                  ngram_range=(lo, hi), min_df=5,
                                  sublinear_tf=True, max_features=300_000,
                                  dtype=np.float32)
            Xtr = vec.fit_transform(texts[i] for i in tr)
            Xte = vec.transform(texts[i] for i in te)
            gs = GridSearchCV(LogisticRegression(max_iter=4000,
                                                 random_state=SEED),
                              {"C": [0.1, 1, 10]}, cv=folds,
                              scoring="f1_macro", n_jobs=-1)
            gs.fit(Xtr, y[tr])
            est = gs.best_estimator_
            r = build_result(y[te], est.predict(Xte),
                             est.predict_proba(Xte)[:, 1], arch[te], fam[te],
                             dataset=dataset, window=w, ngram_range=f"{lo}-{hi}",
                             n_features=int(Xtr.shape[1]),
                             best_C=str(gs.best_params_["C"]),
                             cv_best_f1_macro=round(float(gs.best_score_), 4),
                             fit_seconds=round(time.time() - t0, 1))
            out.append(r)
            print(f"  {dataset:9s} window={w:<6d} ngram={lo}-{hi} "
                  f"feat={Xtr.shape[1]:>7,d} macroF1={r['macro_f1']:.4f} "
                  f"FPR={r['false_positive_rate']:.3f} "
                  f"recall_ran={r['recall_ransomware']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        del texts


def main() -> int:
    corpus = MnemCorpus()
    out: list = []
    for ds in ("mendeley", "balanced"):
        run(ds, corpus, out)
    dest = REPO / "results" / "rules" / "baseline_ablation.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(
        {"description": "mnemonic TF-IDF + logistic regression, window x "
                        "n-gram order sweep on the shared cohort split",
         "seed": SEED, "results": out}, indent=2, default=str),
        encoding="utf-8")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
