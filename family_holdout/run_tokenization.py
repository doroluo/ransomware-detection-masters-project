#!/usr/bin/env python3
"""Family-holdout evaluation of the expC/expD tokenization configuration.

    "C:/Users/chaoa/Downloads/Tokenization-Testing-for-Malware-Data/.venv/
     Scripts/python.exe" family_holdout/run_tokenization.py --dataset both

(gensim has no cp314 wheel, so this runner - and only this runner - needs the
3.12 interpreter of the Tokenization checkout.)

The configuration is `llm_features_pipeline/config.yaml` verbatim, i.e. the
pre-registered expC (mendeley) / expD (balanced) setting, not a tuned pick:

    mnemonic-only REVISED features, first 5,000 normalised lines per file
    vocab_size 1000, tokenizers SW / WP / WPC
    Word2Vec(vector_size=100, window=30, epochs=5, seed=42, workers=1)
    RF / SVM-RBF / MLP through run_pipeline.fit_and_score's GridSearchCV
    (cv=StratifiedKFold(2, shuffle=True, random_state=42), scoring=f1_macro)

and the three combinations run_pipeline.COMBOS actually reports:
RF/WPC, MLP/WP, SVM-RBF/SW (all on w2v: the BERT embedding is disabled in the
config, and expC/expD substituted w2v for it, so this matches them).

Everything the pipeline fits is imported from it rather than re-implemented:
`read_instructions`, `build_sequence_frame` (and through it `get_tokenizer`,
which is what makes the nondeterministic WordPiece trainer reproducible),
`fit_and_score`.

Per-fold fitting
----------------
For each of the five folds the WordPiece vocabulary and the Word2Vec model are
fit on that fold's TRAINING rows only. The trained tokenizer is cached under
`results/family_holdout/tokenizers/` keyed by the training corpus fingerprint,
exactly as run_pipeline does, so a rerun reproduces the committed numbers
instead of re-rolling the Rust HashMap.

LOFO reuses the unsupervised state of the fold that the held-out family lives
in. That state was fit on the other four folds, so it never saw the family -
it is a strictly conservative reuse (a smaller fitting corpus than the LOFO
training set), and it is what makes 38 leave-one-family-out runs per
combination affordable. The classifier is always refit on the true LOFO
training set: all other ransomware families plus ALL goodware. Its
hyper-parameters there are FIXED to the `best_params` recorded in
results/expC/metrics.json (mendeley) and results/expD/metrics.json (balanced)
rather than re-searched, because 38 x 3 GridSearchCV passes per dataset does
not fit the compute budget. The K-fold arm does run the full grid.
"""
from __future__ import annotations

import argparse
import csv
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
                                   check_kfold, check_lofo, write_model_dir)

PIPELINE = "tokenization"
SEED = 42
MAX_INSTRUCTIONS = 5000
VOCAB_SIZE = 1000
CV = 2
W2V = dict(vector_size=100, window=30, epochs=5, seed=SEED, workers=1)
COMBOS = (("RF", "WPC"), ("MLP", "WP"), ("SVM-RBF", "SW"))
EXP_OF = {"mendeley": "expC", "balanced": "expD"}

REVISED = Path("C:/Users/chaoa/Downloads/LLM_Features_Revised/Features_Extraction")
REVISED_BALANCED = Path("C:/Users/chaoa/Downloads/LLM_Features_Revised_Balanced")
TOK_CACHE = OUT_ROOT / "tokenizers"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _manifest(path: Path) -> dict:
    """sha256 -> row, keeping only the rows the revised extraction kept."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["kept"] == "1"]
    out = {}
    for r in rows:
        out.setdefault(r["sha256"], r)
    return out


def feature_paths(folds: Folds):
    """One revised mnemonic .txt path per fold-file row, in fold-file order."""
    m1 = _manifest(REVISED / "revised_manifest.csv")
    m2 = _manifest(REVISED_BALANCED / "revised_manifest.csv")
    paths, names = [], []
    for sha, lab in zip(folds.sha, folds.y):
        if folds.dataset == "balanced" and lab == 0:
            r, root = m2.get(sha), REVISED_BALANCED
        else:
            r, root = m1.get(sha), REVISED
        if r is None:
            raise KeyError(f"{sha} not in the revised manifest ({folds.dataset})")
        p = root / r["out_dir"] / r["txt_file"]
        if not p.exists():
            raise FileNotFoundError(p)
        paths.append(p)
        names.append(r["txt_file"])
    return paths, names


def build_df(folds: Folds, normalize):
    import pandas as pd
    from llm_features_pipeline.run_pipeline import read_instructions
    paths, names = feature_paths(folds)
    rows, empty = [], []
    for i, p in enumerate(paths):
        insns = read_instructions(p, normalize, MAX_INSTRUCTIONS)
        if not insns:
            empty.append(names[i])
            insns = ["<UNK>"]        # keep the row: the fold file defines the pool
        rows.append({"file": names[i], "source": folds.dataset,
                     "group": folds.group[i], "split": "train",
                     "Label": int(folds.y[i]), "Malware": int(folds.y[i]),
                     "arch": folds.arch[i], "family": folds.family[i],
                     "_insns": insns})
        if (i + 1) % 500 == 0:
            print(f"  read {i+1}/{len(paths)}", flush=True)
    if empty:
        print(f"  NOTE: {len(empty)} files had no normalisable line, e.g. "
              f"{empty[:3]}; kept with a single <UNK> token so every fold row "
              f"still gets a prediction", flush=True)
    return pd.DataFrame(rows), empty


# ---------------------------------------------------------------------------
# embedding
# ---------------------------------------------------------------------------
def w2v_model(seq_col_values):
    from gensim.models import Word2Vec
    return Word2Vec(sentences=list(seq_col_values), min_count=1, **W2V)


def w2v_vectors(model, seq_col_values) -> np.ndarray:
    def vec(text):
        v = [model.wv[w] for w in text if w in model.wv]
        if not v:
            return np.zeros(model.vector_size, dtype=np.float32)
        return np.mean(v, axis=0)
    return np.stack([vec(t) for t in seq_col_values], axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
def best_params_of(dataset: str) -> dict:
    """The committed expC / expD best_params, per (model, tokenizer)."""
    p = REPO / "results" / EXP_OF[dataset] / "metrics.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for r in doc["results"]:
        out[(r["model"], r["tokenizer"])] = {
            k: _lit(v) for k, v in r["best_params"].items()}
    return out


def _lit(s):
    import ast
    try:
        return ast.literal_eval(str(s))
    except (ValueError, SyntaxError):
        return s


def make_fixed(model_type, params, seed=SEED):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    if model_type == "RF":
        return RandomForestClassifier(random_state=seed, **params)
    if model_type == "SVM-RBF":
        return SVC(kernel="rbf", random_state=seed, **params)
    if model_type == "MLP":
        return MLPClassifier(early_stopping=True, max_iter=500,
                             random_state=seed, **params)
    raise ValueError(model_type)


def _scores(est, X, pred):
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    if hasattr(est, "decision_function"):
        return np.asarray(est.decision_function(X), dtype=float)
    return pred.astype(float)


# ---------------------------------------------------------------------------
def run_dataset(dataset: str, out_root: Path, tok_repo: Path,
                combos=COMBOS) -> None:
    import pandas as pd
    from llm_features_pipeline.run_pipeline import (build_sequence_frame,
                                                    fit_and_score,
                                                    load_repo_modules)
    t_start = time.time()
    mods = load_repo_modules(tok_repo)
    folds = Folds(dataset)
    check_kfold(folds)
    check_lofo(folds)
    print(f"[{dataset}] reading revised mnemonic features ...", flush=True)
    df, empty = build_df(folds, mods["normalize"])
    fixed = best_params_of(dataset)
    fold_of = folds.fold_of_family()
    n, y = folds.n, folds.y

    for model_type, tok in combos:
        t_combo = time.time()
        tag_model = f"{model_type}_{tok}_w2v"
        kf_score = np.zeros(n)
        kf_pred = np.zeros(n, dtype=int)
        lofo, chosen = {}, {}
        seq_cache = None

        for f, tr, te in folds.kfold():
            t0 = time.time()
            df["split"] = np.where(folds.fold != f, "train", "test")
            if tok in ("SW", "WP"):
                # fold-independent: no vocabulary is fitted for these two
                if seq_cache is None:
                    seq_cache = build_sequence_frame(
                        df, tok, mods["train_tokenizer"], VOCAB_SIZE,
                        cache_dir=TOK_CACHE, tag=f"fh_{dataset}")
                frame = seq_cache
            else:
                frame = build_sequence_frame(
                    df, tok, mods["train_tokenizer"], VOCAB_SIZE,
                    cache_dir=TOK_CACHE, tag=f"fh_{dataset}_fold{f}")
            seqs = frame["Instructions"]

            model = w2v_model(seqs.iloc[tr])
            X = w2v_vectors(model, seqs)
            del model

            # The pipeline's own selection code, unchanged. It returns the
            # metrics and the predictions but not the per-sample scores, and
            # predictions.csv needs those, so the winning estimator is refit
            # from the best_params it reports and the two predictions are
            # asserted identical - which also proves the refit reproduces
            # GridSearchCV's best_estimator_.
            m = fit_and_score(model_type, X[tr], y[tr], X[te], y[te], CV, SEED)
            gs_pred = m.pop("_y_pred")
            params = {k: _lit(v) for k, v in m["best_params"].items()}
            est = make_fixed(model_type, params)
            est.fit(X[tr], y[tr])
            pred = est.predict(X[te])
            if not np.array_equal(pred, gs_pred):
                raise RuntimeError(
                    f"{dataset}/{tag_model} fold {f}: refit with "
                    f"{params} does not reproduce GridSearchCV's estimator")
            kf_pred[te] = pred
            kf_score[te] = _scores(est, X[te], pred)
            chosen[str(f)] = m["best_params"]
            print(f"  [{dataset}/{tag_model}] fold {f}: macroF1="
                  f"{m['macro_f1']:.4f} ({time.time()-t0:.0f}s)", flush=True)

            # LOFO for every family that lives in this fold: the tokenizer and
            # the Word2Vec model above never saw a row of this fold, so they
            # never saw these families.
            fams = [fa for fa in folds.families if fold_of[fa] == f]
            for fam in fams:
                t1 = time.time()
                idx_te = np.flatnonzero((folds.family == fam) & (y == 1))
                idx_tr = np.flatnonzero((folds.family != fam) | (y == 0))
                params = fixed[(model_type, tok)]
                est = make_fixed(model_type, params)
                est.fit(X[idx_tr], y[idx_tr])
                p = est.predict(X[idx_te])
                lofo[fam] = (_scores(est, X[idx_te], p), p)
                print(f"    lofo {fam}: n={len(idx_te)} recall="
                      f"{(p == 1).mean():.3f} ({time.time()-t1:.0f}s)",
                      flush=True)
            del X
            if tok not in ("SW", "WP"):
                del frame

        cfg = {
            "pipeline": PIPELINE,
            "model": tag_model,
            "source": ("llm_features_pipeline/config.yaml, experiment "
                       f"{EXP_OF[dataset]} (pre-registered; not a tuned pick)"),
            "features": {
                "input": ("LLM_Features_Revised (mnemonic-only, cohort "
                          "pre-filtered)" if dataset == "mendeley" else
                          "LLM_Features_Revised ransomware + "
                          "LLM_Features_Revised_Balanced goodware"),
                "max_instructions": MAX_INSTRUCTIONS,
                "normalize": "Tokenization.tokenization.normalize_instruction",
                "empty_files_kept_as_UNK": len(empty),
            },
            "tokenizer": {"method": tok, "vocab_size": VOCAB_SIZE,
                          "fit_on": "training folds only",
                          "cache": str(TOK_CACHE.relative_to(REPO))},
            "embedding": {"method": "w2v", **W2V, "min_count": 1,
                          "fit_on": "training folds only"},
            "classifier": {
                "name": model_type,
                "kfold": {"selection": ("run_pipeline.fit_and_score "
                                        "GridSearchCV, cv=StratifiedKFold(2, "
                                        "shuffle=True, random_state=42), "
                                        "scoring=f1_macro"),
                          "best_params_per_fold": chosen},
                "lofo": {"selection": ("FIXED to the committed "
                                       f"results/{EXP_OF[dataset]}/metrics.json "
                                       "best_params - the per-fold grid does "
                                       "not fit the compute budget"),
                         "params": fixed[(model_type, tok)]},
            },
            "lofo_unsupervised_state": ("reused from the K-fold fold that "
                                        "contains the held-out family; that "
                                        "state was fit on the other four "
                                        "folds and never saw the family"),
            "decision": "argmax (the estimator's own predict)",
            "seed": SEED,
        }
        d = out_root / dataset / PIPELINE / tag_model
        write_model_dir(d, folds, PIPELINE, tag_model, kf_score, kf_pred,
                        folds.fold, lofo, cfg, time.time() - t_combo,
                        description=(f"{model_type}/{tok}/w2v on revised "
                                     f"mnemonic features ({EXP_OF[dataset]} "
                                     f"configuration), family-holdout on "
                                     f"{dataset}"))
        print(f"  wrote {d} ({time.time()-t_combo:.0f}s)", flush=True)
        seq_cache = None
    print(f"[{dataset}] done in {time.time()-t_start:.0f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=(*DATASETS, "both"), default="both")
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--combos", default="",
                    help="comma-separated MODEL/TOKENIZER pairs; default all")
    ap.add_argument("--tok-repo",
                    default="C:/Users/chaoa/Downloads/"
                            "Tokenization-Testing-for-Malware-Data")
    a = ap.parse_args()
    combos = COMBOS if not a.combos else tuple(
        tuple(x.split("/")) for x in a.combos.split(","))
    for d in (DATASETS if a.dataset == "both" else (a.dataset,)):
        run_dataset(d, Path(a.out), Path(a.tok_repo), combos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
