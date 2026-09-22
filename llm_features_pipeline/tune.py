#!/usr/bin/env python3
"""
tune.py - hyperparameter and modelling search for expC and expD, with the test
set held out until exactly one final evaluation per experiment.

    python llm_features_pipeline/tune.py --experiment expC --stage cache
    python llm_features_pipeline/tune.py --experiment expC --stage search
    python llm_features_pipeline/tune.py --experiment expC --stage final
    python llm_features_pipeline/tune.py --experiment expC --stage all

THE PROTOCOL, which is the point of this file existing separately from
run_pipeline.py:

*   Every choice - sequence budget, positional sampler, tokenizer, vocabulary
    size, n-gram order, embedding, pooling, classifier, its hyperparameters, the
    architecture reweighting and the decision threshold - is made by GROUP
    cross-validation on the TRAIN SPLIT ONLY. `--stage search` never loads a test
    row into a model and never computes a test metric. The groups are the
    `group` column of splits.csv: the ransomware FAMILY for a positive, the
    source project (or the file itself) for a negative, so a fold holds out whole
    families, which is the shape of the real test split.
*   `--stage final` fits the single chosen configuration on the whole train
    split and scores the test set ONCE, writing the same file set as the
    committed experiments plus `cv_search.csv` - every configuration tried, with
    its CV scores, so the multiple-comparison exposure is on the record rather
    than implied.
*   The committed `results/expC/` and `results/expD/` are never touched. Tuned
    output goes to `results/expC_tuned/` and `results/expD_tuned/`.

Seeds are fixed everywhere, Word2Vec runs at `workers=1` (see
docs/tokenization_audit.md §1.7) and the subword tokenizers come from the
committed cache (§1.10), so `--stage final` re-run gives a byte-identical
metrics.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from llm_features_pipeline import data as D            # noqa: E402
from llm_features_pipeline import run_pipeline as RP   # noqa: E402
from llm_features_pipeline import tuning as T          # noqa: E402

# The largest budget the search considers. The cache stores exactly this many
# tokens per file for each sampler; every smaller budget is a prefix (head) or
# an even subsample (strided) of it, so the corpus is read once.
MAX_BUDGET = 50_000
SEED = 42


def cache_root() -> Path:
    """Where the token cache lives.

    Not under `results/`: it is hundreds of megabytes of int16 and it is
    derivable from the corpus in one pass, so it is a scratch artifact, not a
    result. `RDMP_TUNE_CACHE` overrides.
    """
    env = os.environ.get("RDMP_TUNE_CACHE")
    if env:
        return Path(env)
    import tempfile
    return Path(tempfile.gettempdir()) / "rdmp_tune_cache"


# ---------------------------------------------------------------------------
# Stage 1: read the corpus once, at the largest budget, both samplers
# ---------------------------------------------------------------------------


def build_cache(cfg, name: str, outroot: Path, force=False) -> Path:
    """Read every file of one experiment once and store its token ids.

    Two sequences per file: the first MAX_BUDGET normalized lines (`head`, what
    the committed pipeline reads, only 10x further in) and MAX_BUDGET lines
    spread evenly over the whole file (`strided`). Tokens are stored as int16
    ids into one shared mnemonic vocabulary; the revised corpus uses ~1,300
    distinct mnemonics, so 16 bits is ample and the cache is half the size of
    the int32 it would otherwise take.
    """
    dest = cache_root() / f"{name}.npz"
    meta_p = cache_root() / f"{name}.meta.json"
    if dest.is_file() and meta_p.is_file() and not force:
        print(f"cache hit: {dest}")
        return dest

    mods = RP.load_repo_modules(Path(cfg["paths"]["tokenization_repo"]))
    normalize = mods["normalize"]
    samples, dedup, notes = RP.build_samples(cfg, name, outroot)
    summary = D.summarize(samples)
    if dedup is not None:
        summary["cross_source_dedup"] = dedup
    summary.update(notes)
    summary["content_leak"] = D.content_leak(samples)
    summary["arch"] = D.arch_breakdown(samples)
    summary["baselines"] = D.baselines(summary["arch"])
    summary["ransomware_families"] = D.family_breakdown(samples)
    summary["goodware_arch"] = {
        sp: dict(sorted(__import__("collections").Counter(
            s.meta.get("arch", "") or "unknown"
            for s in samples if s.split == sp and s.label == 0).items()))
        for sp in ("train", "test")}
    summary["test_goodware_arch"] = summary["goodware_arch"]["test"]
    if summary["group_overlap"]:
        sys.exit(f"group leak across splits: {summary['group_overlap'][:5]}")

    vocab: dict[str, int] = {}
    head_parts, stride_parts = [], []
    head_len, stride_len = [], []
    rows = []
    t0 = time.time()
    for i, s in enumerate(samples, 1):
        n_lines = T.count_lines(s.path)
        head = T.read_tokens(s.path, normalize, MAX_BUDGET, "head")
        if n_lines <= MAX_BUDGET:
            stride = head          # identical by construction; skip the re-read
        else:
            stride = T.read_tokens(s.path, normalize, MAX_BUDGET, "strided",
                                   n_lines=n_lines)
        if not head:
            print(f"  dropping empty {s.name}")
            continue
        for seq, parts, lens in ((head, head_parts, head_len),
                                 (stride, stride_parts, stride_len)):
            ids = np.fromiter((vocab.setdefault(t, len(vocab)) for t in seq),
                              dtype=np.int32, count=len(seq))
            parts.append(ids.astype(np.int16))
            lens.append(len(seq))
        rows.append({"file": s.name, "source": s.source, "label": s.label,
                     "group": s.group, "split": s.split,
                     "arch": s.meta.get("arch", "") or "",
                     "family": s.meta.get("family", "") or "",
                     "cohort_tag": s.meta.get("cohort_tag", "") or "",
                     "in_cohort": int(bool(s.meta.get("in_cohort"))),
                     "n_lines": n_lines})
        if i % 200 == 0:
            print(f"  read {i}/{len(samples)} "
                  f"({time.time()-t0:.0f}s, vocab {len(vocab)})", flush=True)

    if len(vocab) > 32767:
        sys.exit(f"vocabulary {len(vocab)} does not fit int16; widen the cache")

    cache_root().mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest,
        head=np.concatenate(head_parts), head_len=np.asarray(head_len, np.int64),
        stride=np.concatenate(stride_parts),
        stride_len=np.asarray(stride_len, np.int64))
    meta_p.write_text(json.dumps({
        "experiment": name, "max_budget": MAX_BUDGET,
        "vocab": sorted(vocab, key=vocab.get), "rows": rows,
        "samples_summary": summary,
    }), encoding="utf-8")
    print(f"wrote {dest} ({dest.stat().st_size/1e6:.0f} MB) and {meta_p} "
          f"in {time.time()-t0:.0f}s; {len(rows)} files, vocab {len(vocab)}")
    return dest


def load_cache(name: str):
    z = np.load(cache_root() / f"{name}.npz")
    meta = json.loads((cache_root() / f"{name}.meta.json").read_text(encoding="utf-8"))
    out = {}
    for tag in ("head", "stride"):
        flat, lens = z[tag], z[f"{tag}_len"]
        off = np.concatenate([[0], np.cumsum(lens)])
        out[tag] = [flat[off[i]:off[i + 1]].astype(np.int32)
                    for i in range(len(lens))]
    frame = pd.DataFrame(meta["rows"])
    return out, frame, meta


def budget_seqs(cache, sampling: str, budget: int):
    """Token-id sequences at one budget, derived from the cached MAX_BUDGET."""
    if sampling == "head":
        return [s[:budget] for s in cache["head"]]
    if sampling == "strided":
        return [T.even_subsample(s, budget) for s in cache["stride"]]
    raise ValueError(sampling)


# ---------------------------------------------------------------------------
# Subword tokenizers over the mnemonic stream (the SW / WP / WPC / BPE axis)
# ---------------------------------------------------------------------------


def ensure_tokenization_repo(cfg) -> None:
    """Put the tokenization repo AHEAD of this repo on sys.path.

    `Tokenization.tokenization` lives only in the external checkout (the stale
    vendored copy this repo used to carry was deleted); this keeps the import
    order explicit for the tuning driver, which reaches `train_tokenizer`
    without going through `run_pipeline.load_repo_modules`.
    """

    repo = str(Path(cfg["paths"]["tokenization_repo"]))
    if sys.path and sys.path[0] == repo:
        return
    while repo in sys.path:
        sys.path.remove(repo)
    sys.path.insert(0, repo)
    for mod in [m for m in sys.modules if m == "Tokenization"
                or m.startswith("Tokenization.")]:
        del sys.modules[mod]


def subword_ids(seqs, vocab_words, alg: str, vocab_size: int,
                train_rows, tag: str, cache_dir: Path):
    """Re-encode each document with a WordPiece/BPE tokenizer fit on TRAIN rows.

    Returns (id sequences, piece vocabulary). The tokenizer is trained on the
    `<SEP>`-joined mnemonic text of the training documents only, exactly as
    `run_pipeline.build_sequence_frame` does, and cached by corpus fingerprint
    under `results/tokenizers/` for the same reason: the `tokenizers` trainer
    picks a different vocabulary in every process (§1.10).

    Fit on the train SPLIT, not per CV fold. On mnemonic-only input the audit
    (§2.7) measures WPC as whole-word tokenization to within 0.0015% of token
    occurrences, so this axis is a control - "does subword tokenization buy
    anything here" - rather than a contender, and a per-fold refit would cost
    five tokenizer trainings per row of a table whose answer is already known to
    be "no". It is recorded as such in the search output.
    """
    from Tokenization.tokenization import train_tokenizer

    words = np.asarray(vocab_words, dtype=object)
    text = pd.Series([" <SEP> ".join(words[s].tolist()) for s in seqs])
    fit_df = pd.DataFrame({"Instructions": text.iloc[train_rows].reset_index(drop=True)})
    tok = RP.get_tokenizer(alg, train_tokenizer, fit_df, vocab_size,
                           cache_dir, tag)
    enc = tok.encode_batch(text.tolist())
    piece_vocab = tok.get_vocab()
    inv = np.empty(max(piece_vocab.values()) + 1, dtype=object)
    for w, i in piece_vocab.items():
        inv[i] = w
    out = [np.asarray(e.ids, dtype=np.int32) for e in enc]
    return out, [w if w is not None else "" for w in inv.tolist()]


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------


def make_clf(kind: str, params: dict, seed: int = SEED):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC, LinearSVC
    p = dict(params)
    if kind == "LogReg":
        return LogisticRegression(max_iter=4000, random_state=seed, **p)
    if kind == "LinearSVC":
        return LinearSVC(max_iter=8000, random_state=seed, dual="auto", **p)
    if kind == "RF":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **p)
    if kind == "SVM-RBF":
        return SVC(kernel="rbf", random_state=seed, **p)
    if kind == "MLP":
        return MLPClassifier(max_iter=600, random_state=seed, **p)
    raise ValueError(kind)


def decision_scores(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


def arch_sample_weight(y, arch, mode: str):
    """Per-sample weights: class balance, optionally x architecture balance.

    `class` is sklearn's `balanced`, written out so it can be multiplied.
    `class_arch` additionally balances the architectures INSIDE each class, so
    the 42 x64 ransomware training files weigh as much in aggregate as the 862
    x86 ones. The confound this is aimed at is stated in results/summary.md: the
    ransomware side is ~95% x86 in train while the goodware side is 57% x86
    (Mendeley) or 18% (Balanced), so "x86" is evidence for ransomware before a
    single opcode is read, and the x64 positives are the rows that force the
    model to read the code instead.
    """
    y = np.asarray(y)
    w = np.ones(len(y), dtype=np.float64)
    if mode == "none":
        return w
    for c in np.unique(y):
        sel = y == c
        w[sel] = len(y) / (len(np.unique(y)) * sel.sum())
    if mode == "class":
        return w
    if mode != "class_arch":
        raise ValueError(mode)
    arch = np.asarray([a or "unknown" for a in arch])
    for c in np.unique(y):
        sel = y == c
        sub = arch[sel]
        groups = np.unique(sub)
        adj = np.ones(sel.sum())
        for g in groups:
            m = sub == g
            adj[m] = sub.size / (groups.size * m.sum())
        w[sel] = w[sel] * adj
    return w


def _fit(clf, X, y, sw):
    if sw is None:
        clf.fit(X, y)
    else:
        try:
            clf.fit(X, y, sample_weight=sw)
        except TypeError:
            clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# Stage 2: the CV search (train split only)
# ---------------------------------------------------------------------------


TFIDF_CLFS = [
    ("LogReg", {"C": c, "class_weight": cw})
    for c in (0.1, 1.0, 10.0) for cw in (None, "balanced")
] + [
    ("LinearSVC", {"C": c, "class_weight": cw})
    for c in (0.01, 0.1, 1.0) for cw in (None, "balanced")
]

DENSE_CLFS = [
    ("RF", {"n_estimators": 600, "max_depth": None, "min_samples_leaf": 1,
            "class_weight": "balanced"}),
    ("RF", {"n_estimators": 600, "max_depth": 20, "min_samples_leaf": 2,
            "class_weight": "balanced"}),
    ("RF", {"n_estimators": 1000, "max_depth": None, "min_samples_leaf": 1,
            "class_weight": None}),
    ("SVM-RBF", {"C": 1.0, "gamma": "scale", "class_weight": "balanced"}),
    ("SVM-RBF", {"C": 10.0, "gamma": "scale", "class_weight": "balanced"}),
    ("SVM-RBF", {"C": 10.0, "gamma": 0.1, "class_weight": None}),
    ("SVM-RBF", {"C": 100.0, "gamma": "scale", "class_weight": "balanced"}),
    ("MLP", {"hidden_layer_sizes": (500,), "alpha": 1e-4,
             "learning_rate_init": 1e-3, "early_stopping": True}),
    ("MLP", {"hidden_layer_sizes": (256, 128), "alpha": 1e-3,
             "learning_rate_init": 1e-3, "early_stopping": True}),
    ("MLP", {"hidden_layer_sizes": (500,), "alpha": 1e-2,
             "learning_rate_init": 1e-3, "early_stopping": False}),
    ("LogReg", {"C": 1.0, "class_weight": "balanced"}),
]


# The sequence budgets and samplers the search sweeps. 5,000/head is exactly
# what the committed expC and expD runs read, so it is in the table as the
# reference point rather than as a candidate that happens to be nearby.
BUDGETS = [5_000, 20_000, 50_000]
SAMPLERS = ["head", "strided"]
NGRAMS = [(1, 1), (1, 2), (1, 3)]
# min_df / max_features are fixed to the calibration baseline's values
# (rules_pipeline/train_eval.py: min_df=5, max_features=300_000,
# sublinear_tf=True) so the "tokenizer output + TF-IDF" row is a like-for-like
# comparison against the number it has to beat, and not a different vectoriser
# wearing the same name.
MIN_DF = 5
MAX_FEATURES = 300_000

TFIDF_CLFS_SPECS = [(k, p, "none") for k, p in TFIDF_CLFS] + [
    ("LogReg", {"C": 1.0, "class_weight": None}, "class_arch"),
    ("LinearSVC", {"C": 0.1, "class_weight": None}, "class_arch"),
]
TOKENIZER_SPECS = [("LogReg", {"C": 1.0, "class_weight": "balanced"}, "none"),
                   ("LinearSVC", {"C": 0.1, "class_weight": "balanced"}, "none")]
DENSE_SPECS = [(k, p, "none") for k, p in DENSE_CLFS] + [
    ("SVM-RBF", {"C": 10.0, "gamma": "scale", "class_weight": None}, "class_arch"),
    ("MLP", {"hidden_layer_sizes": (500,), "alpha": 1e-4,
             "learning_rate_init": 1e-3, "early_stopping": True}, "class_arch"),
]

# (sampler, budget, dim, window, epochs, min_count). Not a full cross product:
# one Word2Vec per fold at workers=1 is the single most expensive thing in the
# search, and the TF-IDF track reaches a higher CV score for a fraction of the
# cost, so the compute goes where the gain is. The committed setting
# (head/5000/dim100/window30/epochs5/min_count1) is the first row, so the table
# always contains the configuration the tuned run is being compared against.
W2V_PLAN = [
    ("head", 5_000, 100, 30, 5, 1),
    ("head", 5_000, 100, 5, 5, 1),
    ("head", 5_000, 300, 5, 5, 1),
    ("head", 5_000, 300, 30, 5, 1),
    ("head", 5_000, 300, 10, 20, 1),
    ("head", 5_000, 200, 10, 10, 5),
    ("strided", 5_000, 100, 30, 5, 1),
    ("strided", 5_000, 300, 5, 5, 1),
    ("strided", 5_000, 300, 10, 20, 1),
    ("head", 20_000, 100, 30, 5, 1),
    ("head", 20_000, 300, 5, 5, 1),
    ("strided", 20_000, 100, 30, 5, 1),
    ("strided", 20_000, 300, 5, 5, 1),
]


def _spec_label(kind, params, weight_mode):
    ps = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
    return f"{kind}({ps})|w={weight_mode}"


def evaluate_block(per_fold_X, y, arch, folds, specs):
    """Cross-validate many classifier configurations over shared features.

    `per_fold_X(k, train_idx)` returns `{variant: X}` for fold `k`, fitted on
    `train_idx` only and evaluated on every training row. One fold's features
    are built once and handed to every (variant, classifier) pair, then freed -
    which is what keeps a 50,000-token trigram matrix from having to exist five
    times over, and what makes the number of configurations affordable at all.

    Returns, per (variant, classifier), the OUT-OF-FOLD macro-F1 at the
    threshold chosen on those same out-of-fold scores, the balanced accuracy and
    AUC that go with it, and the fold-to-fold spread at the untuned cut. No test
    row is touched anywhere in here.
    """
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

    oof: dict = {}
    fold_scores: dict = {}
    default_cut: dict = {}
    for k, (tr, te) in enumerate(folds):
        Xs = per_fold_X(k, tr)
        for variant, X in Xs.items():
            for si, (kind, params, wmode) in enumerate(specs):
                key = (variant, si)
                if key not in oof:
                    oof[key] = np.full(len(y), np.nan)
                    fold_scores[key] = []
                sw = (None if wmode == "none"
                      else arch_sample_weight(y[tr], arch[tr], wmode))
                clf = _fit(make_clf(kind, params), X[tr], y[tr], sw)
                s = decision_scores(clf, X[te])
                oof[key][te] = s
                cut = 0.5 if hasattr(clf, "predict_proba") else 0.0
                default_cut[key] = cut
                fold_scores[key].append(
                    f1_score(y[te], (s >= cut).astype(int),
                             average="macro", zero_division=0))
        del Xs

    out = {}
    for (variant, si), scores in oof.items():
        kind, params, wmode = specs[si]
        cut = default_cut[(variant, si)]
        rules = T.nested_threshold_scores(y, scores, folds, default_cut=cut)
        # RANKING IS AT THE UNTUNED CUT. Two reasons, both on the record.
        #
        # (1) It is the decision rule the committed expC/expD runs use, so the
        #     before/after comparison moves the representation and the
        #     classifier and nothing else. A tuned operating point is a
        #     different kind of change and is reported as its own axis rather
        #     than folded into the headline.
        # (2) The task's own stopping rule - drop anything whose marginal CV
        #     gain is under 0.005 - disqualifies it. Fitting the threshold on
        #     out-of-fold scores is worth +0.010 CV macro-F1 on the best
        #     configuration and +0.001 to +0.009 on the ones below it.
        #
        # Both fitted rules are scored here for every configuration, nested, and
        # carried through to the test table, so the reader can apply whichever
        # rule they prefer and nothing is hidden by the choice.
        sel = rules["default"]
        try:
            auc = float(roc_auc_score(y, scores))
        except Exception:
            auc = float("nan")
        out[(variant, si)] = {
            "classifier": kind,
            "clf_params": json.dumps({k: str(v) for k, v in params.items()},
                                     sort_keys=True),
            "weighting": wmode,
            "spec": _spec_label(kind, params, wmode),
            "cv_macro_f1": round(sel["macro_f1"], 6),
            "cv_balanced_accuracy": round(sel["balanced_accuracy"], 6),
            "cv_roc_auc": round(auc, 6),
            "threshold": cut,
            "threshold_rule": "default",
            "cv_macro_f1_bal_threshold":
                round(rules["oof_balanced_accuracy"]["macro_f1"], 6),
            "cv_macro_f1_f1_threshold":
                round(rules["oof_macro_f1"]["macro_f1"], 6),
            "oof_threshold_bal_rule":
                round(rules["oof_balanced_accuracy"]["threshold"], 8),
            "oof_threshold_f1_rule":
                round(rules["oof_macro_f1"]["threshold"], 8),
            "cv_macro_f1_fold_std":
                round(float(np.std(fold_scores[(variant, si)])), 6),
        }
    return out


def _w2v_matrix(seqs_tr, vocab_words, dim, window, epochs, min_count, seed=SEED):
    """Word2Vec over the fold's training documents; returns (W, present_mask).

    `workers=1` is not a performance choice - gensim's default races its worker
    threads over the corpus and a fixed seed then does NOT fix the model
    (docs/tokenization_audit.md §1.7). Trained on the fold's TRAINING documents
    only, so no held-out document contributes to the vectors that score it.
    """
    from gensim.models import Word2Vec

    words = np.asarray(vocab_words, dtype=object)
    sentences = [words[s].tolist() for s in seqs_tr]
    m = Word2Vec(sentences=sentences, vector_size=dim, window=window,
                 min_count=min_count, workers=1, epochs=epochs, seed=seed)
    V = len(vocab_words)
    W = np.zeros((V, dim), dtype=np.float64)
    present = np.zeros(V, dtype=bool)
    for i, w in enumerate(vocab_words):
        if w in m.wv:
            W[i] = m.wv[w]
            present[i] = True
    return W, present


def _pool_variants(counts, keys, W, present, idf=None):
    """The three poolings, all as products against the unigram count matrix.

    `keys[j]` is the token id of column j - `build_count_matrix` keeps only the
    n-grams that actually occur, so the columns are not 0..V-1 and the embedding
    rows have to be gathered through `keys`.

    Columns for tokens the Word2Vec model never saw are dropped first, so the
    mean is over the tokens that HAVE a vector - which is exactly what
    `Embedding/build_masked_embeddings.tokens_to_mean_vectors` does token by
    token, and what `--check-pooling` asserts numerically against it.
    """
    sel = present[keys]
    C = counts[:, sel].tocsr()
    Wp = W[keys[sel]]
    out = {"mean": T.pool_mean(C, Wp), "mean_max": T.pool_mean_max(C, Wp)}
    if idf is not None:
        out["tfidf_mean"] = T.pool_weighted_mean(C, Wp, idf[sel])
    return out


def _unigram_idf(counts, train_rows):
    """Smoothed IDF of each unigram column, from training rows only."""
    tr = counts[train_rows].tocsc()
    df = np.diff(tr.indptr).astype(np.float64)
    return 1.0 + np.log((1.0 + tr.shape[0]) / (1.0 + df))


# ---------------------------------------------------------------------------
# Stage 2: the search itself. Train split only, start to finish.
# ---------------------------------------------------------------------------

CV_COLS = ["cv_rank", "track", "tokenizer", "vocab_size", "sampling", "budget",
           "ngram", "embedding", "w2v", "pooling", "n_features", "classifier",
           "clf_params", "weighting", "cv_macro_f1", "cv_balanced_accuracy",
           "cv_roc_auc", "cv_macro_f1_bal_threshold", "cv_macro_f1_f1_threshold",
           "cv_macro_f1_fold_std", "threshold", "threshold_rule",
           "oof_threshold_bal_rule", "oof_threshold_f1_rule", "spec"]


def _write_search(out, name, records, folds_n, fold_report, elapsed):
    """Rewrite cv_search.csv and search_records.json, ranked, from scratch."""
    ranked = sorted(records, key=lambda r: -r["cv_macro_f1"])
    for i, r in enumerate(ranked, 1):
        r["cv_rank"] = i
    with (out / "cv_search.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CV_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(ranked)
    (out / "search_records.json").write_text(
        json.dumps({"experiment": name, "n_configurations": len(ranked),
                    "folds": folds_n, "seed": SEED, "fold_report": fold_report,
                    "selection": ("out-of-fold macro-F1 at the untuned "
                                  "decision cut"),
                    "elapsed_seconds": round(elapsed, 1),
                    "records": ranked}, indent=1), encoding="utf-8")



def run_search(cfg, name: str, outroot: Path, folds_n=5, quick=False):
    cache, frame, meta = load_cache(name)
    vocab_words = meta["vocab"]
    V = len(vocab_words)
    tr_rows = np.flatnonzero((frame["split"] == "train").to_numpy())
    y = frame["label"].to_numpy()[tr_rows]
    groups = frame["group"].to_numpy()[tr_rows]
    arch = frame["arch"].to_numpy()[tr_rows]
    folds = T.group_folds(y, groups, n_splits=folds_n, seed=SEED)
    print(f"{name}: {len(tr_rows)} train rows, {len(set(groups))} groups, "
          f"{folds_n} group folds; vocab {V} mnemonics")
    fold_report = []
    for k, (a, b) in enumerate(folds):
        fams = sorted({g for g in groups[b] if not g.startswith("file:")})
        fold_report.append({"fold": k, "fit": len(a), "held_out": len(b),
                            "held_out_ransomware": int(y[b].sum()),
                            "held_out_goodware": int((y[b] == 0).sum()),
                            "held_out_groups": len(set(groups[b])),
                            "held_out_families": fams})
        print(f"  fold {k}: {len(a)} fit / {len(b)} held out "
              f"({int(y[b].sum())} ransomware, {int((y[b] == 0).sum())} goodware, "
              f"{len(set(groups[b]))} groups, families {fams})")

    records = []
    t_start = time.time()
    out = outroot / f"{name}_tuned"
    out.mkdir(parents=True, exist_ok=True)

    def emit(base, res):
        for (variant, si), m in sorted(res.items(),
                                       key=lambda kv: -kv[1]["cv_macro_f1"]):
            rec = dict(base)
            rec.update(m)
            rec["pooling"] = variant if variant != "tfidf" else ""
            records.append(rec)
        best = max(res.values(), key=lambda m: m["cv_macro_f1"])
        print(f"    best {best['spec']} cv_macro_f1={best['cv_macro_f1']:.4f} "
              f"({time.time()-t_start:.0f}s elapsed, {len(records)} configs)",
              flush=True)
        # Written after every block, not once at the end: a search this long
        # has to be inspectable while it runs and has to survive being stopped.
        _write_search(out, name, records, folds_n, fold_report,
                      time.time() - t_start)

    # ---- Track 1: n-gram TF-IDF over the tokenizer's output ---------------
    # The fair "tokenization" analogue of the audited calibration baseline
    # (rules_pipeline: mnemonic TF-IDF 1-3 + LogReg, macro-F1 0.955 dedup-clean
    # on Mendeley, 0.80 on Balanced). Same vectoriser settings, fed the token
    # stream this pipeline produces.
    for sampling in SAMPLERS:
        all_seqs = budget_seqs(cache, sampling, BUDGETS[-1])
        for budget in BUDGETS:
            seqs = [all_seqs[i][:budget] if sampling == "head"
                    else T.even_subsample(all_seqs[i], budget) for i in tr_rows]
            for ngram in NGRAMS:
                t0 = time.time()
                counts, _keys = T.build_count_matrix(seqs, ngram, V,
                                                     min_df_global=MIN_DF)
                print(f"  tfidf {sampling}/{budget}/{ngram}: "
                      f"{counts.shape[1]} features, {counts.nnz/1e6:.1f}M nnz "
                      f"({time.time()-t0:.0f}s)", flush=True)

                def per_fold(k, tr, _c=counts):
                    X, _ = T.fold_tfidf(_c, tr, min_df=MIN_DF, sublinear=True,
                                        max_features=MAX_FEATURES)
                    return {"tfidf": X}

                res = evaluate_block(per_fold, y, arch, folds, TFIDF_CLFS_SPECS)
                emit({"track": "tfidf", "tokenizer": "SW", "vocab_size": "",
                      "sampling": sampling, "budget": budget,
                      "ngram": f"{ngram[0]}-{ngram[1]}", "embedding": "tfidf",
                      "n_features": int(counts.shape[1]), "w2v": ""}, res)
                del counts
        del all_seqs

    # ---- Track 2: the SW / WP / WPC / BPE axis ---------------------------
    # Held at the committed pipeline's own budget so this block answers "does
    # the tokenizer matter" and nothing else. WP is adjacent-mnemonic bigrams,
    # which is what run_pipeline.build_sequence_frame builds; WPC and BPE are
    # subword models trained on the train split's <SEP>-joined mnemonic text.
    ensure_tokenization_repo(cfg)
    tok_cache = REPO / cfg["tokenization"].get("tokenizer_cache",
                                               "results/tokenizers")
    for budget in ([5_000] if quick else [5_000, 20_000]):
        base_seqs = budget_seqs(cache, "head", budget)
        for alg, vsizes in (("WP", [""]), ("WPC", [500, 1000, 2000, 4000]),
                            ("BPE", [1000, 4000])):
            for vs in vsizes:
                t0 = time.time()
                if alg == "WP":
                    seqs_tr = [T.ngram_keys(base_seqs[i].astype(np.int64), 2, V)
                               for i in tr_rows]
                    vv, ngram = V * V, (1, 1)
                else:
                    ids, pieces = subword_ids(
                        base_seqs, vocab_words, alg, int(vs), tr_rows,
                        f"{name}_tuned_b{budget}_v{vs}", tok_cache)
                    seqs_tr = [ids[i] for i in tr_rows]
                    vv, ngram = len(pieces), (1, 2)
                counts, _ = T.build_count_matrix(seqs_tr, ngram, vv,
                                                 min_df_global=MIN_DF)
                print(f"  tok {alg}/{vs}/{budget}: {counts.shape[1]} features "
                      f"({time.time()-t0:.0f}s)", flush=True)

                def per_fold(k, tr, _c=counts):
                    X, _ = T.fold_tfidf(_c, tr, min_df=MIN_DF, sublinear=True,
                                        max_features=MAX_FEATURES)
                    return {"tfidf": X}

                res = evaluate_block(per_fold, y, arch, folds, TOKENIZER_SPECS)
                emit({"track": "tokenizer", "tokenizer": alg, "vocab_size": vs,
                      "sampling": "head", "budget": budget,
                      "ngram": f"{ngram[0]}-{ngram[1]}", "embedding": "tfidf",
                      "n_features": int(counts.shape[1]), "w2v": ""}, res)
                del counts
        del base_seqs

    # ---- Track 3: Word2Vec + pooling + the dense classifiers --------------
    # The committed pipeline's representation, with a real grid on top. One
    # Word2Vec per fold per setting at `workers=1` is the search's dominant
    # cost, so W2V_PLAN is a chosen list rather than a cross product, and the
    # reason is recorded there and in the audit's tuning section.
    by_source: dict = {}
    for row in W2V_PLAN:
        by_source.setdefault(row[:2], []).append(row[2:])
    for (sampling, budget), grid in by_source.items():
        all_seqs = budget_seqs(cache, sampling, BUDGETS[-1])
        seqs = [all_seqs[i][:budget] if sampling == "head"
                else T.even_subsample(all_seqs[i], budget) for i in tr_rows]
        del all_seqs
        counts1, keys1 = T.build_count_matrix(seqs, (1, 1), V, min_df_global=1)
        for (dim, win, ep, mc) in grid:
            t0 = time.time()

            def per_fold(k, tr, _s=seqs, _c=counts1, _k=keys1,
                         _p=(dim, win, ep, mc)):
                W, present = _w2v_matrix([_s[i] for i in tr], vocab_words,
                                         _p[0], _p[1], _p[2], _p[3])
                return _pool_variants(_c, _k, W, present, _unigram_idf(_c, tr))

            res = evaluate_block(per_fold, y, arch, folds, DENSE_SPECS)
            emit({"track": "w2v", "tokenizer": "SW", "vocab_size": "",
                  "sampling": sampling, "budget": budget, "ngram": "1-1",
                  "embedding": "w2v", "n_features": dim,
                  "w2v": f"dim={dim},window={win},epochs={ep},min_count={mc}"},
                 res)
            print(f"  w2v {sampling}/{budget}/dim{dim}/win{win}/ep{ep} "
                  f"took {time.time()-t0:.0f}s", flush=True)
        del counts1, keys1, seqs

    _write_search(out, name, records, folds_n, fold_report,
                  time.time() - t_start)
    records.sort(key=lambda r: -r["cv_macro_f1"])
    print(f"\n{len(records)} configurations cross-validated in "
          f"{time.time()-t_start:.0f}s; wrote {out/'cv_search.csv'}")
    print("top 10 by CV macro-F1:")
    for r in records[:10]:
        print(f"  {r['cv_macro_f1']:.4f} {r['track']:9s} {r['tokenizer']:4s} "
              f"{r['sampling']:7s} b={r['budget']:<6} {r['ngram']} "
              f"{r['pooling']:10s} {r['spec']}")
    return records


# ---------------------------------------------------------------------------
# Stage 3: ONE final evaluation per chosen configuration
# ---------------------------------------------------------------------------


def _features_for(rec, cache, frame, meta, tr_rows, cfg):
    """Train and test matrices for one recorded configuration.

    The vectoriser, the Word2Vec model and the IDF are all fitted on the TRAIN
    rows and then applied to the test rows, never the other way round. This is
    the only function in the file that is allowed to look at a test row at all,
    and it is called once per chosen configuration.
    """
    V = len(meta["vocab"])
    budget, sampling = int(rec["budget"]), rec["sampling"]
    seqs = budget_seqs(cache, sampling, budget)

    if rec["track"] == "tokenizer" and rec["tokenizer"] != "SW":
        alg, vs = rec["tokenizer"], rec["vocab_size"]
        if alg == "WP":
            seqs = [T.ngram_keys(s.astype(np.int64), 2, V) for s in seqs]
            vv, ngram = V * V, (1, 1)
        else:
            ensure_tokenization_repo(cfg)
            tok_cache = REPO / cfg["tokenization"].get("tokenizer_cache",
                                                       "results/tokenizers")
            seqs, pieces = subword_ids(
                seqs, meta["vocab"], alg, int(vs), tr_rows,
                f"{meta['experiment']}_tuned_b{budget}_v{vs}", tok_cache)
            vv, ngram = len(pieces), (1, 2)
    else:
        vv = V
        lo, hi = rec["ngram"].split("-")
        ngram = (int(lo), int(hi))

    if rec["embedding"] == "tfidf":
        counts, _ = T.build_count_matrix(seqs, ngram, vv, min_df_global=MIN_DF)
        X, keep = T.fold_tfidf(counts, tr_rows, min_df=MIN_DF, sublinear=True,
                               max_features=MAX_FEATURES)
        return X, int(keep.sum())

    p = dict(kv.split("=") for kv in rec["w2v"].split(","))
    counts1, keys1 = T.build_count_matrix(seqs, (1, 1), vv, min_df_global=1)
    W, present = _w2v_matrix([seqs[i] for i in tr_rows], meta["vocab"],
                             int(p["dim"]), int(p["window"]), int(p["epochs"]),
                             int(p["min_count"]))
    variants = _pool_variants(counts1, keys1, W, present,
                              _unigram_idf(counts1, tr_rows))
    X = variants[rec["pooling"] or "mean"]
    return X, X.shape[1]


def run_final(cfg, name: str, outroot: Path, top_k=5):
    """Fit the chosen configurations on the whole train split, score the test
    set once each, and write the tuned results directory.

    `top_k` configurations are scored, not one: the deliverable asks for the
    top five by CV WITH their test numbers, which is the only way to show
    whether the CV ranking transfers. They were all chosen before any test
    metric existed; the HEADLINE result is `cv_rank == 1`, and the other four
    are reported as what they are - the honest picture of how much of the CV
    ordering is noise.
    """
    out = outroot / f"{name}_tuned"
    src = out / "search_records.json"
    if not src.is_file():
        # `--results-dir <scratch> --stage final` is the reproducibility check:
        # re-run the chosen configurations somewhere else and diff. It has to
        # read the COMMITTED search record rather than demand the search be
        # repeated, or it is not checking the thing it claims to.
        src = REPO / "results" / f"{name}_tuned" / "search_records.json"
        print(f"  no search record under {out}; reading the committed one at {src}")
    recs = json.loads(src.read_text(encoding="utf-8"))
    cache, frame, meta = load_cache(name)
    summary = meta["samples_summary"]

    split = frame["split"].to_numpy()
    tr_rows = np.flatnonzero(split == "train")
    te_rows = np.flatnonzero(split == "test")
    y = frame["label"].to_numpy()
    arch = frame["arch"].to_numpy()
    te = pd.DataFrame({"file": frame["file"].to_numpy()[te_rows],
                       "Label": y[te_rows], "arch": arch[te_rows],
                       "family": frame["family"].to_numpy()[te_rows]})

    chosen = recs["records"][:top_k]
    results, pred_rows = [], []
    t0 = time.time()
    for rec in chosen:
        X, nfeat = _features_for(rec, cache, frame, meta, tr_rows, cfg)
        params = {k: _literal(v) for k, v in json.loads(rec["clf_params"]).items()}
        sw = (None if rec["weighting"] == "none"
              else arch_sample_weight(y[tr_rows], arch[tr_rows], rec["weighting"]))
        clf = _fit(make_clf(rec["classifier"], params), X[tr_rows], y[tr_rows], sw)
        scores = decision_scores(clf, X[te_rows])
        thr = float(rec["threshold"])
        y_pred = (scores >= thr).astype(int)

        m = RP.binary_metrics(y[te_rows], y_pred, scores)
        m.update(model=rec["classifier"],
                 tokenizer=rec["tokenizer"],
                 embedding=rec["embedding"], mask_rate=0.0,
                 n_train=len(tr_rows), n_test=len(te_rows),
                 best_params={k: str(v) for k, v in params.items()},
                 cv_rank=rec["cv_rank"], selected=rec["cv_rank"] == 1,
                 sampling=rec["sampling"], budget=rec["budget"],
                 ngram=rec["ngram"], pooling=rec["pooling"],
                 vocab_size=rec["vocab_size"], w2v=rec["w2v"],
                 weighting=rec["weighting"], n_features=nfeat,
                 threshold=thr, threshold_rule=rec["threshold_rule"],
                 cv_macro_f1=rec["cv_macro_f1"],
                 cv_balanced_accuracy=rec["cv_balanced_accuracy"],
                 cv_roc_auc=rec["cv_roc_auc"],
                 cv_minus_test_macro_f1=round(rec["cv_macro_f1"] - m["macro_f1"], 6))
        m["goodware_recall_by_arch"] = RP.goodware_recall_by_arch(te, y_pred)
        m["per_arch"] = RP.per_arch_metrics(te, y_pred)
        m["ransomware_recall_by_family"] = RP.ransomware_recall_by_family(te, y_pred)
        # The two FITTED threshold rules on the same model, so the audit can
        # price the operating point on the test set as well as in CV. The
        # reported row is the untuned cut; these are the alternatives, carried
        # so the choice is checkable rather than asserted.
        for rule, key in (("bal", "oof_threshold_bal_rule"),
                          ("f1", "oof_threshold_f1_rule")):
            cut = float(rec[key])
            alt = RP.binary_metrics(y[te_rows], (scores >= cut).astype(int), scores)
            m[f"threshold_{rule}_rule"] = cut
            m[f"macro_f1_at_{rule}_threshold"] = round(alt["macro_f1"], 6)
            m[f"recall_goodware_at_{rule}_threshold"] = round(alt["recall_goodware"], 6)
            m[f"cv_macro_f1_at_{rule}_threshold"] = rec[f"cv_macro_f1_{rule}_threshold"]
        results.append(m)
        tag = f"rank{rec['cv_rank']}"
        for f_, l_, a_, fam_, p_ in zip(te["file"], te["Label"], te["arch"],
                                        te["family"], y_pred):
            pred_rows.append({"model": rec["classifier"], "tokenizer": rec["tokenizer"],
                              "embedding": rec["embedding"], "mask_rate": 0.0,
                              "cv_rank": rec["cv_rank"], "config": tag,
                              "file": f_, "label": int(l_), "arch": a_ or "",
                              "family": fam_ or "", "pred": int(p_)})
        print(f"  rank {rec['cv_rank']}: cv {rec['cv_macro_f1']:.4f} -> test "
              f"macro-F1 {m['macro_f1']:.4f}  bal-acc {m['balanced_accuracy']:.4f} "
              f"AUC {m['roc_auc']:.4f}  [{rec['track']}/{rec['sampling']}/"
              f"{rec['budget']}/{rec['ngram']}/{rec['spec']}]", flush=True)
        del X

    payload = {
        "experiment": f"{name}_tuned",
        "description": (f"{cfg['experiments'][name]['description']} - tuned by "
                        f"{recs['n_configurations']}-configuration group-CV "
                        f"search on the train split only"),
        "task": "binary: 0=goodware, 1=ransomware",
        "samples": summary,
        "results": results,
        "tuning": {
            "configurations_cross_validated": recs["n_configurations"],
            "configurations_evaluated_on_test": len(chosen),
            "selection_metric": ("out-of-fold macro-F1 at the untuned decision cut - the same rule the committed runs use. The two fitted-threshold rules are scored for every configuration and reported beside it; see tuning.THRESHOLD_RULES"),
            "cv": f"StratifiedGroupKFold(n_splits={recs['folds']}, shuffle=True, "
                  f"random_state={SEED}) on the train split",
            "groups": "ransomware family / goodware source project (splits.csv `group`)",
            "fold_report": recs["fold_report"],
            "search_elapsed_seconds": recs["elapsed_seconds"],
            "baseline_committed_run": f"results/{name}/metrics.json",
            "cv_to_test_gap_macro_f1": [
                {"cv_rank": r["cv_rank"], "cv": r["cv_macro_f1"],
                 "test": round(m["macro_f1"], 6),
                 "gap": m["cv_minus_test_macro_f1"]}
                for r, m in zip(chosen, results)],
        },
        "config": cfg,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(payload, indent=2))
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "sample_counts.json").write_text(json.dumps(summary, indent=2))
    with (out / "splits.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "source", "label", "group", "split", "arch",
                    "family", "cohort_tag", "in_cohort"])
        for r in frame.itertuples(index=False):
            w.writerow([r.file, r.source, r.label, r.group, r.split, r.arch,
                        r.family, r.cohort_tag, r.in_cohort])
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pred_rows[0]))
        w.writeheader()
        w.writerows(pred_rows)
    print(f"wrote {out/'metrics.json'}")
    return payload


def _literal(v: str):
    """Recover a parameter value from the string form stored in cv_search.csv."""
    import ast
    if v in ("None", "none"):
        return None
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def check_pooling(cfg, name: str) -> int:
    """Assert the count-matrix pooling equals the repo's token-by-token mean.

    The whole search rests on "mean pooling is a sparse matmul". This proves it
    against `Embedding.build_masked_embeddings.build_word2vec_embeddings` on a
    real slice of the corpus rather than asserting it in a comment.
    """
    RP.load_repo_modules(Path(cfg["paths"]["tokenization_repo"]))
    from Embedding.build_masked_embeddings import build_word2vec_embeddings

    cache, frame, meta = load_cache(name)
    words = np.asarray(meta["vocab"], dtype=object)
    seqs = [s[:2000] for s in budget_seqs(cache, "head", 2000)[:120]]
    tr = list(range(80))
    te = list(range(80, 120))
    tr_df = pd.DataFrame({"Instructions": [words[seqs[i]].tolist() for i in tr],
                          "Label": [0] * len(tr)})
    te_df = pd.DataFrame({"Instructions": [words[seqs[i]].tolist() for i in te],
                          "Label": [0] * len(te)})
    Xtr_ref, Xte_ref = build_word2vec_embeddings(
        tr_df, te_df, mask_rate=0.0, seed=SEED, vector_size=50, window=10,
        epochs=3, workers=1)

    counts, keys = T.build_count_matrix(seqs, (1, 1), len(meta["vocab"]),
                                        min_df_global=1)
    W, present = _w2v_matrix([seqs[i] for i in tr], meta["vocab"], 50, 10, 3, 1,
                             seed=SEED)
    mine = _pool_variants(counts, keys, W, present)["mean"]

    # Two comparisons, because they answer different questions.
    #
    # Against a token-by-token mean in float64: this is the arithmetic identity
    # and it has to hold to machine precision.
    ref64 = np.stack([
        np.mean([W[t] for t in s if present[t]], axis=0) if any(present[t] for t in s)
        else np.zeros(W.shape[1]) for s in seqs])
    d_exact = float(np.abs(mine - ref64).max())
    # Against the repo's own function, which accumulates in float32: agreement
    # here is only ever to float32 rounding, and saying so is the point.
    d_tr = float(np.abs(mine[tr] - Xtr_ref).max())
    d_te = float(np.abs(mine[te] - Xte_ref).max())
    scale = float(np.abs(ref64).max())
    print(f"max |matmul mean - float64 token-by-token mean| = {d_exact:.3e}")
    print(f"max |matmul mean - repo float32 mean|: train {d_tr:.3e} "
          f"test {d_te:.3e} (values up to {scale:.3f}; float32 eps x that "
          f"is {scale*1.2e-7:.1e} per term)")
    ok = d_exact < 1e-9 and max(d_tr, d_te) < 1e-3 * max(scale, 1.0)
    print("POOLING EQUIVALENCE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--stage", default="all",
                    choices=["cache", "search", "final", "all", "check-pooling"])
    ap.add_argument("--results-dir", default="")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--quick", action="store_true",
                    help="tiny search; for smoke-testing the harness only")
    ap.add_argument("--force-cache", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    outroot = Path(args.results_dir or cfg["paths"]["results"])
    if not outroot.is_absolute():
        outroot = REPO / outroot
    outroot.mkdir(parents=True, exist_ok=True)

    if args.stage == "check-pooling":
        return check_pooling(cfg, args.experiment)
    if args.stage in ("cache", "all"):
        build_cache(cfg, args.experiment, outroot, force=args.force_cache)
    if args.stage in ("search", "all"):
        run_search(cfg, args.experiment, outroot, folds_n=args.folds,
                   quick=args.quick)
    if args.stage in ("final", "all"):
        run_final(cfg, args.experiment, outroot, top_k=args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
