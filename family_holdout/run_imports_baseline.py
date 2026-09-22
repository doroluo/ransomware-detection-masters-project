#!/usr/bin/env python3
"""
run_imports_baseline.py - what the import table is worth ON ITS OWN, and what
it adds to the mnemonic TF-IDF baseline, under family holdout.

    python family_holdout/run_imports_baseline.py --dataset both

Why. The sequence transformer with the import side input
(`seq_transformer/seq_transformer_imports`) is the first row to beat the
mnemonic TF-IDF baseline on mendeley. That comparison is unfair as it stands:
the transformer was handed a second information source that TF-IDF never saw.
Three fixed configurations settle whether the gain came from the encoder or
from the imports:

    hashed2048       the transformer's own side input, alone: signed feature
                     hashing of the import names into 2,048 dims, L2-normalised
                     (bit-identical to seq_model/data.py::hash_imports), then
                     LogisticRegression.  "Would a linear model on the same
                     side input do as well?"
    names_tfidf      bag of import names, TF-IDF (binary presence, sublinear,
                     min_df 2), fitted on the training rows of each split;
                     LogisticRegression.  The interpretable version.
    mnem+names       the pre-registered mnemonic 1-3-gram TF-IDF baseline
                     (family_holdout/run_tfidf.py, verbatim) with the
                     names_tfidf block appended; LogisticRegression.  "Does
                     TF-IDF gain as much from imports as the transformer did?"

C is chosen per fit from {0.1, 1, 10} by 2-fold CV inside the training rows,
exactly as run_tfidf.py does it; nothing else is searched, no threshold is
moved, and nothing here was chosen after seeing a test fold. Files with no
import directory (83 cohort ransomware, 18 goodware) get an all-zero import
block - the same thing the transformer saw for them.

Input: manifests/imports/imports_flat.json ({sha256: [import names]}, from
imports/merge_imports.py). Output: results/family_holdout/<dataset>/
imports_baseline/<config>_LogReg/, the standard six files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from family_holdout.common import (DATASETS, Folds, OUT_ROOT,  # noqa: E402
                                   check_kfold, check_lofo, write_model_dir, ALL_DATASETS)
from family_holdout.run_tfidf import (C_GRID, MAX_FEATURES, MAX_MNEMS,  # noqa: E402
                                      MIN_DF, NGRAM, SEED, fit_score,
                                      load_texts, make_vec)

PIPELINE = "imports_baseline"
IMPORTS = REPO / "manifests" / "imports" / "imports_flat.json"
HASH_DIM = 2048
CONFIGS = ("hashed2048", "names_tfidf", "mnem+names")


def hash_imports(names, dim: int = HASH_DIM) -> np.ndarray:
    """Copy of seq_model/data.py::hash_imports (kept importable without torch);
    tests/test_imports_baseline.py asserts the two agree."""
    v = np.zeros(dim, dtype=np.float32)
    for name in names:
        h = hashlib.blake2b(str(name).strip().lower().encode("utf-8"),
                            digest_size=8).digest()
        k = int.from_bytes(h, "little")
        v[k % dim] += 1.0 if (k >> 63) & 1 else -1.0
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def load_imports(folds: Folds, path: Path = IMPORTS) -> list:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [s for s in folds.sha if s not in doc]
    if missing:
        raise SystemExit(f"{len(missing)} cohort files have no import record in {path}")
    return [list(doc[s]) for s in folds.sha]


def names_vec():
    from sklearn.feature_extraction.text import TfidfVectorizer
    # one "document" per file: import names joined by spaces; a name is a token
    return TfidfVectorizer(analyzer="word", token_pattern=r"\S+", binary=True,
                           sublinear_tf=True, min_df=2, dtype=np.float32)


def names_text(names: list) -> str:
    return " ".join(n.replace(" ", "_") for n in names)


def run_dataset(dataset: str, out_root: Path, configs) -> dict:
    from scipy.sparse import csr_matrix, hstack
    t_start = time.time()
    folds = Folds(dataset)
    check_kfold(folds)
    check_lofo(folds)
    imports = load_imports(folds)
    n_empty = sum(1 for x in imports if not x)
    print(f"[{dataset}] {folds.n} files, {n_empty} with no imports", flush=True)
    H = csr_matrix(np.vstack([hash_imports(x) for x in imports]))
    N = [names_text(x) for x in imports]
    texts = load_texts(folds) if "mnem+names" in configs else None

    def blocks(cfg, tr, te):
        """(Xtr, Xte) for one configuration, everything fitted on `tr` only."""
        if cfg == "hashed2048":
            return H[tr], H[te]
        nv = names_vec()
        Ntr = nv.fit_transform(N[i] for i in tr)
        Nte = nv.transform(N[i] for i in te)
        if cfg == "names_tfidf":
            return Ntr, Nte
        mv = make_vec()
        Mtr = mv.fit_transform(texts[i] for i in tr)
        Mte = mv.transform(texts[i] for i in te)
        return hstack([Mtr, Ntr]).tocsr(), hstack([Mte, Nte]).tocsr()

    kf_score = {c: np.zeros(folds.n) for c in configs}
    kf_pred = {c: np.zeros(folds.n, dtype=int) for c in configs}
    chosen = {c: {"kfold": {}, "lofo": {}} for c in configs}
    for f, tr, te in folds.kfold():
        t0 = time.time()
        for c in configs:
            Xtr, Xte = blocks(c, tr, te)
            pred, score, C, cv = fit_score("LogReg", Xtr, folds.y[tr], Xte)
            kf_pred[c][te], kf_score[c][te] = pred, score
            chosen[c]["kfold"][str(f)] = {"C": C, "cv_f1_macro": round(cv, 4),
                                          "n_features": int(Xtr.shape[1])}
        print(f"  [{dataset}] kfold {f}: {time.time()-t0:.0f}s", flush=True)
    lofo = {c: {} for c in configs}
    for k, (fam, tr, te) in enumerate(folds.lofo(), 1):
        t0 = time.time()
        for c in configs:
            Xtr, Xte = blocks(c, tr, te)
            pred, score, C, cv = fit_score("LogReg", Xtr, folds.y[tr], Xte)
            lofo[c][fam] = (score, pred)
            chosen[c]["lofo"][fam] = {"C": C, "cv_f1_macro": round(cv, 4)}
        print(f"  [{dataset}] lofo {k}/{len(folds.families)} {fam}: {time.time()-t0:.0f}s", flush=True)

    out = {}
    for c in configs:
        cfg = {
            "pipeline": PIPELINE, "model": f"{c}_LogReg", "config": c,
            "imports": str(IMPORTS.relative_to(REPO)),
            "n_files_without_imports": n_empty,
            "features": {
                "hashed2048": f"seq_model/data.py::hash_imports, dim {HASH_DIM}, L2-normalised",
                "names_tfidf": "TfidfVectorizer(binary=True, sublinear_tf=True, min_df=2) over import names, fit on training rows",
                "mnem+names": (f"hstack[ mnemonic TfidfVectorizer(ngram_range={NGRAM}, min_df={MIN_DF}, "
                               f"sublinear_tf=True, max_features={MAX_FEATURES}) over the first {MAX_MNEMS} "
                               f"mnemonics (run_tfidf.py verbatim), names_tfidf ]"),
            }[c],
            "classifier": {"name": "LogReg",
                           "estimator": "LogisticRegression(max_iter=4000, random_state=42)",
                           "C_grid": C_GRID,
                           "C_selection": "GridSearchCV(StratifiedKFold(2, shuffle, rs=42), f1_macro) inside training rows",
                           "C_chosen": chosen[c]},
            "decision": "argmax (predict)", "seed": SEED, "pre_registered": True,
            "schemes": ["kfold", "lofo"],
        }
        d = out_root / dataset / PIPELINE / f"{c}_LogReg"
        out[c] = write_model_dir(d, folds, PIPELINE, f"{c}_LogReg", kf_score[c],
                                 kf_pred[c], folds.fold.copy(), lofo[c], cfg,
                                 time.time() - t_start,
                                 description=f"{c} + LogReg, family-holdout K-fold and LOFO on the {dataset} cohort")
        print(f"  [{dataset}] {c}: fold macro-F1 {out[c]['fold_mean']['macro_f1']:.3f} +/- "
              f"{out[c]['fold_sd']['macro_f1']:.3f}, LOFO {out[c]['lofo_mean_recall']:.3f} -> {d}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=(*ALL_DATASETS, "both"), default="both")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--out", default=str(OUT_ROOT))
    a = ap.parse_args()
    configs = tuple(x for x in a.configs.split(",") if x)
    for c in configs:
        if c not in CONFIGS:
            raise SystemExit(f"unknown config {c!r}; choose from {CONFIGS}")
    for d in (DATASETS if a.dataset == "both" else (a.dataset,)):
        run_dataset(d, Path(a.out), configs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
