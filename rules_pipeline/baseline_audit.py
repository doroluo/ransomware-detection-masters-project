#!/usr/bin/env python3
"""Leakage audit and characterisation of the mnemonic TF-IDF calibration baseline.

    python rules_pipeline/baseline_audit.py --dataset both

`rules_pipeline/train_eval.py` (c) reports macro-F1 0.96-0.97 on the Mendeley
family-disjoint test split for mnemonic 1-3-gram TF-IDF + LogReg / LinearSVC.
That is far above every other track measured on the same cohort (tokenization
best 0.93, graph2vec best 0.84, CNN-ViT ~0.63), so before it is quoted anywhere
it has to be shown not to be an artefact. This script answers, in order:

1. **Is the fit clean?**  vocabulary and IDF from TRAIN rows only; TEST never
   touched by `fit`; the feature matrix is the TF-IDF and nothing else -- no
   file length, no file size, no architecture, no family, no path.
2. **Are the rows disjoint?**  by sha256, by exact bytes of the `mn/` file, and
   by the capped token stream the vectoriser actually reads; families disjoint;
   groups disjoint; and the same cohort rows as every other pipeline.
3. **What survives deduplication?**  the same model re-scored on the test rows
   that are *not* verbatim copies of a training row, and retrained on a
   content-deduplicated train set.
4. **Why does it work?**  top +/- n-grams by coefficient, per-family test
   recall, per-architecture metrics, and a model trained and tested inside x64
   only (where the class/arch confound in train runs the other way).
5. **Is it real?**  label permutation (must collapse to chance), drop the
   top-50 features by |coef|, and unigram-only TF-IDF.

Writes `results/rules/baseline_audit.json`. Everything is derived, nothing here
is typed in by hand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from cnn_vit_pipeline.cohort import (_core_metrics, add_val_fold,  # noqa: E402
                                     build_result, load_split)
from rules_pipeline.train_eval import (MAX_MNEMS, TREES, MnemCorpus,  # noqa: E402
                                       _tree, load_encoded)

SEED = 42
NGRAM = (1, 3)
MIN_DF = 5
MAX_FEATURES = 300_000
PERMUTATIONS = 3
TOP_DROP = 50
EXPC_COUNTS = REPO / "results" / "expC" / "sample_counts.json"


# ---------------------------------------------------------------------------
# vectoriser, always fitted on the rows it is handed and no others
# ---------------------------------------------------------------------------
def make_vec(ngram=NGRAM):
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(analyzer="word", token_pattern=r"\S+",
                           ngram_range=ngram, min_df=MIN_DF, sublinear_tf=True,
                           max_features=MAX_FEATURES, dtype=np.float32)


def fit_logreg(Xtr, ytr, C=None, grid=(0.1, 1, 10)):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    if C is not None:
        est = LogisticRegression(max_iter=4000, random_state=SEED, C=C)
        return est.fit(Xtr, ytr), C, None
    folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    gs = GridSearchCV(LogisticRegression(max_iter=4000, random_state=SEED),
                      {"C": list(grid)}, cv=folds, scoring="f1_macro",
                      n_jobs=-1).fit(Xtr, ytr)
    return gs.best_estimator_, gs.best_params_["C"], round(float(gs.best_score_), 4)


def fit_svc(Xtr, ytr, C=None, grid=(0.1, 1, 10)):
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.svm import LinearSVC
    if C is not None:
        return LinearSVC(random_state=SEED, max_iter=5000, C=C).fit(Xtr, ytr), C, None
    folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    gs = GridSearchCV(LinearSVC(random_state=SEED, max_iter=5000),
                      {"C": list(grid)}, cv=folds, scoring="f1_macro",
                      n_jobs=-1).fit(Xtr, ytr)
    return gs.best_estimator_, gs.best_params_["C"], round(float(gs.best_score_), 4)


def _score(est, X):
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    return est.decision_function(X)


def quick(est, X, y, arch=None, fam=None, **extra):
    pred = est.predict(X)
    sc = _score(est, X)
    if arch is None:
        m = _core_metrics(y, pred, sc)
    else:
        m = build_result(y, pred, sc, arch, fam)
    keep = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc",
            "recall_goodware", "recall_ransomware", "precision_ransomware",
            "false_positive_rate", "confusion_matrix", "support_goodware",
            "support_ransomware", "per_arch", "ransomware_recall_by_family")
    out = {k: (round(v, 4) if isinstance(v, float) else v)
           for k, v in m.items() if k in keep}
    out.update(extra)
    return out


# ---------------------------------------------------------------------------
# 1. hygiene of the fit
# ---------------------------------------------------------------------------
def vectoriser_hygiene(texts, tr, te, corpus) -> dict:
    """Prove the vocabulary and the IDF come from TRAIN rows and only those."""
    v_train = make_vec()
    v_train.fit([texts[i] for i in tr])

    v_all = make_vec()
    v_all.fit([texts[i] for i in range(len(texts))])

    # the production path: fit_transform(train) then transform(test)
    v_prod = make_vec()
    v_prod.fit_transform(texts[i] for i in tr)
    idf_before = v_prod.idf_.copy()
    vocab_before = dict(v_prod.vocabulary_)
    v_prod.transform(texts[i] for i in te)          # must be a no-op on state

    mnem = set(corpus.vocab) | {"<oov>"}
    bad_tokens, order_hist = [], Counter()
    for t in v_prod.vocabulary_:
        parts = t.split()
        order_hist[len(parts)] += 1
        if any(p not in mnem for p in parts):
            bad_tokens.append(t)

    return {
        "vectoriser": (f"TfidfVectorizer(analyzer=word, token_pattern=\\S+, "
                       f"ngram_range={NGRAM}, min_df={MIN_DF}, sublinear_tf=True, "
                       f"max_features={MAX_FEATURES})"),
        "n_features_fit_on_train": int(len(vocab_before)),
        "n_features_if_fit_on_train_plus_test": int(len(v_all.vocabulary_)),
        "features_only_test_would_have_added": int(
            len(set(v_all.vocabulary_) - set(vocab_before))),
        "production_vocab_equals_train_only_vocab":
            bool(set(vocab_before) == set(v_train.vocabulary_)),
        "production_idf_equals_train_only_idf": bool(np.allclose(
            idf_before,
            v_train.idf_[[v_train.vocabulary_[t] for t in
                          sorted(vocab_before, key=vocab_before.get)]])),
        "transform_on_test_left_vocabulary_unchanged":
            bool(dict(v_prod.vocabulary_) == vocab_before),
        "transform_on_test_left_idf_unchanged":
            bool(np.array_equal(v_prod.idf_, idf_before)),
        "feature_order_histogram": {str(k): int(v) for k, v in
                                    sorted(order_hist.items())},
        "non_mnemonic_tokens_in_vocabulary": bad_tokens[:20],
        "n_non_mnemonic_tokens": len(bad_tokens),
        "note": ("the only input to fit/transform is the mnemonic text; no "
                 "length, size, architecture, family, filename or path column "
                 "is concatenated to X anywhere in train_eval.py (c)"),
    }


# ---------------------------------------------------------------------------
# 2. duplication
# ---------------------------------------------------------------------------
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def duplication(split: pd.DataFrame, texts, tr, te) -> dict:
    """Exact-duplicate audit on three keys: the cohort sha256, the raw bytes of
    the `mn/` file, and the capped token stream the vectoriser reads."""
    raw = []
    for r in split.itertuples():
        p = TREES[_tree(r.source)] / "mn" / f"{r.sha256}.txt"
        raw.append(_sha(p.read_bytes()) if p.exists() else f"MISSING:{r.sha256}")
    stream = [_sha(t.encode()) for t in texts]

    y = split["label"].to_numpy()
    lab = split["label"].to_numpy()
    is_tr = np.zeros(len(split), bool); is_tr[tr] = True
    is_te = np.zeros(len(split), bool); is_te[te] = True

    out: dict = {}
    for key, h in (("raw_mn_file_bytes", raw),
                   (f"capped_token_stream_{MAX_MNEMS}", stream)):
        h = np.asarray(h, dtype=object)
        tr_h = set(h[is_tr].tolist())
        leak = {}
        for cls, name in ((0, "goodware"), (1, "ransomware")):
            m = is_te & (lab == cls)
            n = int(m.sum())
            dup = int(sum(1 for x in h[m] if x in tr_h))
            leak[name] = {"test_n": n, "duplicated_in_train": dup,
                          "leak_rate": round(dup / n, 4) if n else 0.0}
        groups = defaultdict(list)
        for i, x in enumerate(h):
            groups[x].append(i)
        both = [g for x, g in groups.items()
                if len({lab[i] for i in g}) > 1]
        big = sorted((g for g in groups.values() if len(g) > 1),
                     key=len, reverse=True)[:6]
        out[key] = {
            "unique_streams": len(groups),
            "total_files": int(len(split)),
            "unique_streams_train": len(tr_h),
            "test_leak": leak,
            "streams_labelled_both_classes": len(both),
            "files_in_two_label_streams": sorted(
                str(split["filename"].iloc[i]) for g in both for i in g)[:10],
            "largest_duplicate_groups": [
                {"size": len(g),
                 "in_train": int(is_tr[g].sum()),
                 "in_test": int(is_te[g].sum()),
                 "labels": sorted({int(lab[i]) for i in g}),
                 "example_filenames": [str(split["filename"].iloc[i])
                                       for i in g[:3]]}
                for g in big],
        }
    out["clean_test_index_key"] = f"capped_token_stream_{MAX_MNEMS}"
    tr_h = set(np.asarray(stream, dtype=object)[is_tr].tolist())
    out["_clean_test"] = np.array(
        [i for i in te if stream[i] not in tr_h], dtype=np.int64)
    out["_stream_hash"] = stream
    del y
    return out


# ---------------------------------------------------------------------------
# 3-5. the run
# ---------------------------------------------------------------------------
def run(dataset: str, corpus: MnemCorpus) -> dict:
    t0 = time.time()
    split = add_val_fold(load_split(dataset))
    enc = load_encoded(split, corpus)
    texts = [corpus.text(a) for a in enc]
    y = split["label"].to_numpy()
    arch = split["arch"].to_numpy()
    fam = split["family"].to_numpy()
    grp = split["family_or_group"].to_numpy()
    tr = np.flatnonzero((split["split"] == "train").to_numpy())
    te = np.flatnonzero((split["split"] == "test").to_numpy())
    print(f"[{dataset}] {len(split)} rows, {len(tr)} train / {len(te)} test",
          flush=True)

    rep: dict = {"dataset": dataset, "seed": SEED}

    # ---- 1. split integrity ------------------------------------------------
    counts = {sp: {"n": int(len(ix)),
                   "goodware": int((y[ix] == 0).sum()),
                   "ransomware": int((y[ix] == 1).sum())}
              for sp, ix in (("train", tr), ("test", te))}
    expc = None
    if EXPC_COUNTS.exists() and dataset == "mendeley":
        d = json.loads(EXPC_COUNTS.read_text(encoding="utf-8"))
        expc = {k: {kk: d[k][kk] for kk in ("n", "goodware", "ransomware")}
                for k in ("train", "test")}
    shas = split["sha256"].to_numpy()
    rep["split_integrity"] = {
        "counts": counts,
        "expC_sample_counts": expc,
        "matches_expC_cohort": (expc == counts) if expc else None,
        "sha256_overlap_train_test": sorted(
            set(shas[tr]) & set(shas[te]))[:10],
        "n_sha256_overlap_train_test": len(set(shas[tr]) & set(shas[te])),
        "duplicate_sha256_anywhere": int(len(shas) - len(set(shas))),
        "ransomware_families_train": sorted(set(fam[tr][y[tr] == 1])),
        "ransomware_families_test": sorted(set(fam[te][y[te] == 1])),
        "family_overlap": sorted(set(fam[tr][y[tr] == 1]) &
                                 set(fam[te][y[te] == 1])),
        "group_overlap": sorted(set(grp[tr]) & set(grp[te]))[:10],
        "n_group_overlap": len(set(grp[tr]) & set(grp[te])),
        "arch_by_split_and_class": {
            f"{sp}/{cls}": dict(Counter(arch[ix][y[ix] == c]))
            for sp, ix in (("train", tr), ("test", te))
            for c, cls in ((0, "goodware"), (1, "ransomware"))},
    }

    # ---- 2. hygiene --------------------------------------------------------
    print("  vectoriser hygiene", flush=True)
    rep["vectoriser_hygiene"] = vectoriser_hygiene(texts, tr, te, corpus)

    # ---- 3. duplication ----------------------------------------------------
    print("  duplication", flush=True)
    dup = duplication(split, texts, tr, te)
    clean_te = dup.pop("_clean_test")
    dup.pop("_stream_hash")
    rep["duplication"] = dup

    # ---- 4. headline reproduction -----------------------------------------
    print("  headline fit", flush=True)
    vec = make_vec()
    Xtr = vec.fit_transform(texts[i] for i in tr)
    Xte = vec.transform(texts[i] for i in te)
    lr, C_lr, cv_lr = fit_logreg(Xtr, y[tr])
    sv, C_sv, cv_sv = fit_svc(Xtr, y[tr])
    rep["headline"] = {
        "n_features": int(Xtr.shape[1]),
        "LogReg": quick(lr, Xte, y[te], arch[te], fam[te],
                        best_C=str(C_lr), cv_best_f1_macro=cv_lr),
        "LinearSVC": quick(sv, Xte, y[te], arch[te], fam[te],
                           best_C=str(C_sv), cv_best_f1_macro=cv_sv),
    }

    # ---- 5. what survives deduplication ------------------------------------
    print(f"  dedup re-eval ({len(clean_te)}/{len(te)} test rows are not "
          f"verbatim copies of a train row)", flush=True)
    rows_clean = {i: k for k, i in enumerate(te)}
    sel = np.array([rows_clean[i] for i in clean_te], dtype=np.int64)
    ded = {"clean_test_n": int(len(clean_te)),
           "dropped_as_duplicates_of_train": int(len(te) - len(clean_te)),
           "same_model_on_clean_test_only": {
               "LogReg": quick(lr, Xte[sel], y[clean_te], arch[clean_te],
                               fam[clean_te]),
               "LinearSVC": quick(sv, Xte[sel], y[clean_te], arch[clean_te],
                                  fam[clean_te])}}

    # retrain on a content-deduplicated train set, evaluate on the clean test
    stream = [_sha(t.encode()) for t in texts]
    seen, keep = set(), []
    for i in tr:
        if stream[i] not in seen:
            seen.add(stream[i]); keep.append(i)
    tr_d = np.array(keep, dtype=np.int64)
    vec_d = make_vec()
    Xtr_d = vec_d.fit_transform(texts[i] for i in tr_d)
    Xte_d = vec_d.transform(texts[i] for i in clean_te)
    lr_d, C_d, cv_d = fit_logreg(Xtr_d, y[tr_d])
    ded["dedup_train"] = {
        "train_n": int(len(tr_d)), "dropped": int(len(tr) - len(tr_d)),
        "goodware": int((y[tr_d] == 0).sum()),
        "ransomware": int((y[tr_d] == 1).sum()),
        "n_features": int(Xtr_d.shape[1]),
        "LogReg_on_clean_test": quick(lr_d, Xte_d, y[clean_te], arch[clean_te],
                                      fam[clean_te], best_C=str(C_d),
                                      cv_best_f1_macro=cv_d),
    }
    rep["deduplicated"] = ded

    # ---- 6. characterisation ----------------------------------------------
    print("  characterisation", flush=True)
    names = np.array(vec.get_feature_names_out())
    coef = lr.coef_.ravel()
    order = np.argsort(coef)
    rep["characterisation"] = {
        "top_ransomware_ngrams": [
            {"ngram": str(names[j]), "coef": round(float(coef[j]), 3),
             "order": len(str(names[j]).split())}
            for j in order[::-1][:30]],
        "top_goodware_ngrams": [
            {"ngram": str(names[j]), "coef": round(float(coef[j]), 3),
             "order": len(str(names[j]).split())}
            for j in order[:30]],
        "per_family_test_recall": rep["headline"]["LogReg"].pop(
            "ransomware_recall_by_family"),
        "per_architecture": rep["headline"]["LogReg"].pop("per_arch"),
    }
    rep["headline"]["LinearSVC"].pop("ransomware_recall_by_family", None)
    rep["headline"]["LinearSVC"].pop("per_arch", None)

    # a model that never sees the other architecture
    per_arch_models = {}
    for a in sorted(set(arch)):
        tra = np.array([i for i in tr if arch[i] == a])
        tea = np.array([i for i in te if arch[i] == a])
        if len(tea) < 20 or len(set(y[tra])) < 2 or len(set(y[tea])) < 2:
            per_arch_models[a] = {"skipped": "too few rows or one class only",
                                  "train_n": int(len(tra)),
                                  "test_n": int(len(tea))}
            continue
        v = make_vec()
        A = v.fit_transform(texts[i] for i in tra)
        Bm = v.transform(texts[i] for i in tea)
        m, Ca, cva = fit_logreg(A, y[tra])
        per_arch_models[a] = quick(
            m, Bm, y[tea], arch[tea], fam[tea], best_C=str(Ca),
            cv_best_f1_macro=cva, train_n=int(len(tra)),
            train_goodware=int((y[tra] == 0).sum()),
            train_ransomware=int((y[tra] == 1).sum()),
            n_features=int(A.shape[1]))
        per_arch_models[a].pop("per_arch", None)
        per_arch_models[a].pop("ransomware_recall_by_family", None)
    rep["characterisation"]["within_architecture_models"] = per_arch_models

    # ---- 7. ablations ------------------------------------------------------
    print("  ablations", flush=True)
    abl: dict = {}

    perms = []
    rng = np.random.default_rng(SEED)
    for p in range(PERMUTATIONS):
        yp = y.copy()
        sh = y[tr].copy(); rng.shuffle(sh); yp[tr] = sh
        m, _, cvp = fit_logreg(Xtr, yp[tr], C=C_lr)
        perms.append(quick(m, Xte, y[te], permutation=p))
    abl["label_permutation_train_only"] = {
        "n_permutations": PERMUTATIONS,
        "C": str(C_lr),
        "runs": perms,
        "mean_macro_f1": round(float(np.mean([r["macro_f1"] for r in perms])), 4),
        "chance_macro_f1_all_ransomware": round(
            float(_core_metrics(y[te], np.ones(len(te), int))["macro_f1"]), 4),
    }

    drop = np.argsort(-np.abs(coef))[:TOP_DROP]
    mask = np.ones(Xtr.shape[1], bool); mask[drop] = False
    m, _, _ = fit_logreg(Xtr[:, mask], y[tr], C=C_lr)
    abl[f"drop_top_{TOP_DROP}_by_abs_coef"] = quick(
        m, Xte[:, mask], y[te], dropped=[str(names[j]) for j in drop[:15]],
        n_features=int(mask.sum()))

    drop2 = np.argsort(-np.abs(coef))[:500]
    mask2 = np.ones(Xtr.shape[1], bool); mask2[drop2] = False
    m2, _, _ = fit_logreg(Xtr[:, mask2], y[tr], C=C_lr)
    abl["drop_top_500_by_abs_coef"] = quick(m2, Xte[:, mask2], y[te],
                                            n_features=int(mask2.sum()))

    v1 = make_vec(ngram=(1, 1))
    A1 = v1.fit_transform(texts[i] for i in tr)
    B1 = v1.transform(texts[i] for i in te)
    m1, C1, cv1 = fit_logreg(A1, y[tr])
    abl["unigram_only"] = quick(m1, B1, y[te], n_features=int(A1.shape[1]),
                                best_C=str(C1), cv_best_f1_macro=cv1)
    abl["unigram_only"].pop("per_arch", None)

    # length-only control: is it just "ransomware binaries are shorter"?
    L = np.array([[len(a), len(set(a.tolist()))] for a in enc], dtype=float)
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=400, random_state=SEED).fit(
        L[tr], y[tr])
    abl["length_only_control"] = quick(
        rf, L[te], y[te],
        features="[n_mnemonics_read (cap 30k), n_distinct_mnemonics]",
        median_mnemonics_goodware_train=float(np.median(L[tr][y[tr] == 0, 0])),
        median_mnemonics_ransomware_train=float(np.median(L[tr][y[tr] == 1, 0])),
        median_mnemonics_goodware_test=float(np.median(L[te][y[te] == 0, 0])),
        median_mnemonics_ransomware_test=float(np.median(L[te][y[te] == 1, 0])))
    rep["ablations"] = abl

    rep["elapsed_seconds"] = round(time.time() - t0, 1)
    for k, v in (("LogReg", rep["headline"]["LogReg"]),
                 ("LinearSVC", rep["headline"]["LinearSVC"])):
        print(f"    headline {k:10s} macroF1={v['macro_f1']:.4f}", flush=True)
    print(f"    clean-test LogReg   macroF1="
          f"{ded['same_model_on_clean_test_only']['LogReg']['macro_f1']:.4f}",
          flush=True)
    print(f"    dedup-train LogReg  macroF1="
          f"{ded['dedup_train']['LogReg_on_clean_test']['macro_f1']:.4f}",
          flush=True)
    print(f"    permuted labels     macroF1="
          f"{abl['label_permutation_train_only']['mean_macro_f1']:.4f}",
          flush=True)
    print(f"    unigram only        macroF1="
          f"{abl['unigram_only']['macro_f1']:.4f}", flush=True)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mendeley", "balanced", "both"],
                    default="both")
    ap.add_argument("--out", default=str(REPO / "results" / "rules" /
                                         "baseline_audit.json"))
    a = ap.parse_args()
    corpus = MnemCorpus()
    doc = {"description": ("leakage audit and characterisation of the mnemonic "
                           "1-3-gram TF-IDF calibration baseline, on the shared "
                           "cohort split"),
           "max_mnemonics_per_file": MAX_MNEMS,
           "datasets": {}}
    for ds in (["mendeley", "balanced"] if a.dataset == "both" else [a.dataset]):
        doc["datasets"][ds] = run(ds, corpus)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=2, default=str),
                           encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
