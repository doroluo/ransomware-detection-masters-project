#!/usr/bin/env python3
"""
run_pipeline.py - one command, one config, six experiments.

    python llm_features_pipeline/run_pipeline.py --config llm_features_pipeline/config.yaml
    python llm_features_pipeline/run_pipeline.py --experiment expC
    python llm_features_pipeline/run_pipeline.py --dry-run          # counts only
    python llm_features_pipeline/run_pipeline.py --summary-only     # rebuild summary.md

The six move two variables one at a time:

    expA         Mendeley as shipped,   full-instruction lines  (the baseline)
    expA_cohort  cohort-filtered,       full-instruction lines
    expC         cohort-filtered,       mnemonic-only lines
    expB         + Goodware_Balanced,   full-instruction lines
    expB_cohort  cohort-filtered,       full-instruction lines
    expD         cohort-filtered,       mnemonic-only lines

expB_cohort and expD reuse expB's committed goodware membership rather than
re-splitting, so they must run after expB (the config order guarantees it).

BINARY TASK. Label 0 = goodware, label 1 = ransomware. The tokenization and
embedding code is imported from Tokenization-Testing-for-Malware-Data rather
than copied, so there is exactly one implementation of each. What is NOT
reused is that repo's data loader, which drops filenames (making dedup and
error analysis impossible), iterates directories unsorted (so the embedding
matrix and the label vector are matched by an unstable row order), and offers
no way to keep related samples out of opposite splits. See
docs/tokenization_audit.md.

Writes per experiment under results/<name>/ (or --results-dir):
    metrics.json          every model x tokenizer x embedding x mask rate,
                          plus per-architecture metrics for both classes,
                          per-family ransomware recall, the sample counts and
                          the config that made them
    sample_counts.json    counts, group overlap, leak rates, dedup report,
                          cohort annotation/filter reports, arch and family
                          breakdowns for both classes and both splits
    splits.csv            every sample: label, group, split, arch, family,
                          cohort tag, in_cohort
    predictions.csv       one row per (model, tokenizer, embedding, mask, test
                          file): label, arch, family, prediction. Saved so a new
                          question about an old run can be answered without a
                          re-run.
    config_used.yaml

REPRODUCIBILITY: two runs of the same command must give byte-identical
metrics.json results. That holds only with `embedding.w2v.workers: 1` (§1.7)
AND a warm tokenizer cache under results/tokenizers/ (§1.10) -- the WordPiece
trainer has no seed and picks a different vocabulary in every process. Both are
in docs/tokenization_audit.md. Verify with:

    python llm_features_pipeline/run_pipeline.py --results-dir /tmp/repro
    diff results/expB/metrics.json /tmp/repro/expB/metrics.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from llm_features_pipeline import data as D  # noqa: E402

# (classifier, tokenization method, embedding method) - the combos the
# original Classification/GridSearch.py sweeps.
COMBOS = [("RF", "WPC", "w2v"), ("MLP", "WP", "w2v"), ("SVM-RBF", "SW", "bert")]


def load_repo_modules(tok_repo: Path):
    if not tok_repo.is_dir():
        sys.exit(f"tokenization_repo not found: {tok_repo}")
    sys.path.insert(0, str(tok_repo))
    from Tokenization.tokenization import (normalize_instruction,
                                           train_tokenizer)
    from Embedding.build_masked_embeddings import (build_word2vec_embeddings,
                                                   build_bert_embeddings)
    return dict(normalize=normalize_instruction, train_tokenizer=train_tokenizer,
                w2v=build_word2vec_embeddings, bert=build_bert_embeddings)


def read_instructions(path: Path, normalize, cap: int) -> list:
    out = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            n = normalize(line)
            if n:
                out.append(n)
                if cap and len(out) >= cap:
                    break
    return out


def build_frames(samples, normalize, cap: int):
    """One row per sample, provenance kept alongside the sequence so a
    prediction can always be traced back to a file."""
    rows = []
    dropped = []
    for i, s in enumerate(samples, 1):
        insns = read_instructions(s.path, normalize, cap)
        if not insns:
            dropped.append(s.name)
            continue
        rows.append({
            "file": s.name, "source": s.source, "group": s.group,
            "split": s.split, "Label": s.label, "Malware": s.label,
            # Now known for EVERY file of every set: the cohort CSVs carry the
            # architecture for both corpora, kept rows and dropped ones alike
            # (data.annotate_cohort). It used to be available only for
            # Goodware_Balanced, via opcode_manifest.csv.
            "arch": (s.meta.get("arch") or ""),
            # Split group for ransomware, "" for goodware. Never a target.
            "family": (s.meta.get("family") or ""),
            "_insns": insns,
        })
        if i % 400 == 0:
            print(f"  read {i}/{len(samples)}", flush=True)
    if dropped:
        print(f"  dropped {len(dropped)} empty files, e.g. {dropped[:3]}")
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("no samples survived reading")
    return df


def _fingerprint(alg: str, vocab_size: int, texts) -> str:
    h = hashlib.sha256(f"{alg}|{vocab_size}|".encode())
    for t in texts:
        h.update(t.encode("utf-8", "replace"))
        h.update(b"\0")
    return h.hexdigest()


def get_tokenizer(alg, train_tokenizer, fit_df, vocab_size, cache_dir, tag):
    """Train the subword tokenizer, or reuse a cached one trained on exactly
    this corpus.

    The cache is a REPRODUCIBILITY mechanism, not a speed one.
    `tokenizers`' WordPiece trainer is nondeterministic across processes: the
    vocabulary it selects at the size cutoff depends on Rust HashMap iteration
    order, which is randomly seeded per process. Measured on the real corpus,
    two runs of the identical command gave RF/WPC macro-F1 0.6044 and 0.6629 -
    a 0.06 swing, larger than several of the differences the experiments are
    trying to measure. `PYTHONHASHSEED` does not touch it (that is Python's
    hash, not Rust's) and neither does `RAYON_NUM_THREADS=1`: it is not a
    thread-count race. It cannot be fixed from Python.

    So the trained tokenizer is written next to the results and reused whenever
    the training corpus is byte-for-byte the same. The committed metrics are
    then exactly reproducible from the committed artifacts, and the residual
    nondeterminism is confined to the first run on a new corpus, where it is
    announced in the log.

    Delete `results/tokenizers/` to force a retrain. Everything else in the
    pipeline - Word2Vec, the folds, every classifier - is already
    bit-reproducible; see docs/tokenization_audit.md §1.7-§1.10.
    """
    from tokenizers import Tokenizer

    fp = _fingerprint(alg, vocab_size, fit_df["Instructions"])
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tok_path = cache_dir / f"{tag}_{alg}.json"
    fp_path = cache_dir / f"{tag}_{alg}.fingerprint"

    if tok_path.is_file() and fp_path.is_file() \
            and fp_path.read_text(encoding="utf-8").strip() == fp:
        print(f"  reusing cached {alg} tokenizer {tok_path.name} "
              f"(corpus fingerprint {fp[:12]})", flush=True)
        return Tokenizer.from_file(str(tok_path))

    why = "no cache" if not tok_path.is_file() else "training corpus changed"
    print(f"  training {alg} on {len(fit_df)} train rows (vocab {vocab_size}); "
          f"{why}. NOTE: this step is not reproducible across processes - the "
          f"result is cached to {tok_path.name} so later runs match it.",
          flush=True)
    tok = train_tokenizer(alg, fit_df, vocab_size, "<UNK>",
                          ["<UNK>", "<SEP>", "<MASK>", "<CLS>", "<HEX>", "<OFFSET>"],
                          1000)
    tok.save(str(tok_path))
    fp_path.write_text(fp + "\n", encoding="utf-8")
    return tok


def build_sequence_frame(df, alg, train_tokenizer, vocab_size: int,
                         cache_dir=None, tag=""):
    """Materialise the sequence column for ONE tokenization method.

    Built one method at a time on purpose. With 2,600 samples x 5,000
    instructions, holding SW (raw instructions), WP (bigrams) and WPC (subword
    tokens) simultaneously is several GB of Python strings; the caller frees
    each frame before asking for the next.

    Tokenizers are fit on TRAIN ROWS ONLY - fitting on everything leaks test
    vocabulary into the model.
    """
    if alg == "SW":
        return df.assign(Instructions=df["_insns"], Opcodes=df["_insns"])

    if alg == "WP":
        def bigrams(x):
            if len(x) >= 2:
                return [f"{x[i]}_{x[i+1]}" for i in range(len(x) - 1)]
            return list(x)
        col = df["_insns"].map(bigrams)
        return df.assign(Instructions=col, Opcodes=col)

    if alg in ("WPC", "BPE"):
        text = df["_insns"].map(lambda x: " <SEP> ".join(x))
        fit_df = pd.DataFrame(
            {"Instructions": text[df["split"] == "train"].reset_index(drop=True)})
        tok = get_tokenizer(alg, train_tokenizer, fit_df, vocab_size,
                            cache_dir, tag)
        enc = tok.encode_batch(text.tolist())
        col = pd.Series([e.tokens for e in enc], index=df.index)
        del text, enc
        return df.assign(Instructions=col, Opcodes=col)

    raise ValueError(f"unknown tokenization method: {alg}")


def binary_metrics(y_true, y_pred, scores):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, precision_recall_fscore_support,
                                 roc_auc_score)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1],
                                                   zero_division=0)
    # sup can be 0 for a class in a per-architecture slice; guard below.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # The Mendeley test split is 74% ransomware, so plain accuracy is
        # inflated by ~0.74 for guessing the majority class every time.
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "majority_class_accuracy": float(max(np.mean(y_true), 1 - np.mean(y_true))),
        "precision_goodware": float(p[0]), "recall_goodware": float(r[0]),
        "f1_goodware": float(f[0]), "support_goodware": int(sup[0]),
        "precision_ransomware": float(p[1]), "recall_ransomware": float(r[1]),
        "f1_ransomware": float(f[1]), "support_ransomware": int(sup[1]),
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "macro_f1": float(np.mean(f)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    try:
        m["roc_auc"] = float(roc_auc_score(y_true, scores))
    except Exception:
        m["roc_auc"] = None
    return m


def fit_and_score(model_type, X_train, y_train, X_test, y_test, cv, seed):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    if model_type == "RF":
        clf = RandomForestClassifier(random_state=seed)
        grid = {"n_estimators": [100, 400], "max_depth": [None, 20]}
    elif model_type == "SVM-RBF":
        # probability=False keeps the fit fast; decision_function is enough for
        # ROC-AUC and is monotone in the same direction as the probability.
        clf = SVC(kernel="rbf", random_state=seed)
        grid = {"C": [1, 10], "gamma": [0.1, 0.01]}
    elif model_type == "MLP":
        clf = MLPClassifier(early_stopping=True, max_iter=500, random_state=seed)
        grid = {"hidden_layer_sizes": [(500,)], "learning_rate_init": [0.001]}
    else:
        raise ValueError(model_type)

    # Explicit, seeded folds. `cv=<int>` gives StratifiedKFold(shuffle=False),
    # which slices the matrix in row order - and row order here is
    # ransomware-then-goodware, sorted by filename, i.e. sorted by FAMILY. Fold
    # 1 would be roughly avaddon..makop and fold 2 maze..zeppelin. That is an
    # arbitrary, undocumented grouping of the model-selection folds; shuffling
    # with a fixed seed makes the folds reproducible *and* representative.
    #
    # Folds are taken over X_train only, so the test matrix is never seen
    # during selection.
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    gs = GridSearchCV(clf, grid, cv=folds, scoring="f1_macro", n_jobs=-1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    if hasattr(best, "predict_proba"):
        scores = best.predict_proba(X_test)[:, 1]
    elif hasattr(best, "decision_function"):
        scores = best.decision_function(X_test)
    else:
        scores = y_pred
    out = binary_metrics(y_test, y_pred, scores)
    out["best_params"] = {k: str(v) for k, v in gs.best_params_.items()}
    out["_y_pred"] = y_pred
    return out


def per_arch_metrics(te_frame, y_pred) -> dict:
    """Test metrics for BOTH classes, computed separately inside each
    architecture slice.

    This is the check the earlier audit could only ask for. The ransomware side
    is ~95% x86 in train and ~80% x86 in test, while the goodware sides run from
    57% x86 (Mendeley) down to 18% x86 (Goodware_Balanced), so "x64 implies
    benign" is a shortcut that scores well on the pooled test set. Splitting the
    test set by architecture and scoring each slice on its own removes that
    shortcut's payoff: a model that is genuinely reading behaviour keeps its
    macro-F1 inside both slices, and a model that is reading bitness collapses
    inside the slice where bitness no longer separates the classes.

    Reported per slice: n, the class supports, recall for each class, accuracy
    and macro-F1. Slices where one class is absent get `macro_f1: null` rather
    than a number computed against an empty class.

    The architecture comes from the cohort CSVs, which carry it for every file
    of every set (`data.annotate_cohort`). Before those existed only
    Goodware_Balanced had a per-file architecture, which is why the previous
    version of this function could break out the goodware side only.
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    y_pred = np.asarray(y_pred)
    y_true = te_frame["Label"].to_numpy()
    archs = np.asarray([a or "unknown" for a in te_frame["arch"].to_numpy()])
    out = {}
    for a in sorted(set(archs.tolist())):
        sel = archs == a
        yt, yp = y_true[sel], y_pred[sel]
        n_good = int((yt == 0).sum())
        n_ran = int((yt == 1).sum())
        p, r, f, _ = precision_recall_fscore_support(yt, yp, labels=[0, 1],
                                                     zero_division=0)
        both = n_good > 0 and n_ran > 0
        out[a] = {
            "n": int(sel.sum()),
            "support_goodware": n_good, "support_ransomware": n_ran,
            "recall_goodware": float(r[0]) if n_good else None,
            "recall_ransomware": float(r[1]) if n_ran else None,
            "accuracy": float(accuracy_score(yt, yp)) if len(yt) else None,
            # Averaging an f1 over a class with zero support is averaging in a
            # zero, which reads as "the model failed" when the truth is "the
            # question was not asked". Only report it where both classes exist.
            "macro_f1": float(np.mean(f)) if both else None,
        }
    return out


def ransomware_recall_by_family(te_frame, y_pred) -> dict:
    """Per-family recall on the ransomware half of the test set.

    The family is a split group, never a label (see data.py). It is used here
    only to show WHERE the ransomware recall comes from: the Mendeley test
    families are entirely unseen in training, so a single pooled recall hides
    the fact that a couple of large families can carry the whole number.
    """
    y_pred = np.asarray(y_pred)
    mask = te_frame["Label"].to_numpy() == 1
    fams = te_frame["family"].to_numpy()[mask]
    hit = (y_pred[mask] == 1)
    out = {}
    for fam in sorted(set(f or "unknown" for f in fams.tolist())):
        sel = np.asarray([(f or "unknown") == fam for f in fams])
        out[fam] = {"n": int(sel.sum()), "recall": float(hit[sel].mean()),
                    "detected": int(hit[sel].sum())}
    return out


def goodware_recall_by_arch(te_frame, y_pred) -> dict:
    """Goodware recall (1 - FPR) split by the architecture of the source binary.

    The whole A-vs-B comparison is confounded by architecture: the Mendeley
    ransomware is 96% x86 in train and 76% x86 in test (WRITEUP.md §3.2-3.3),
    the Mendeley goodware 56% x86, and Goodware_Balanced only ~23% x86. A model
    that has partly learned "x64 means benign" would show it here as a large
    gap between the two rows.

    Only the goodware side is broken out, and only where `arch` is known:
    `opcode_manifest.csv` records it for every Goodware_Balanced file, but the
    Mendeley feature files carry no architecture and their `good_test` binaries
    are not present on this machine (0 of 131 match by name), so the Exp A row
    is honestly reported as `unknown` rather than guessed. The ransomware side
    cannot be broken out at all: those binaries are VM-only and WRITEUP.md has
    aggregate counts only.
    """
    mask = te_frame["Label"].to_numpy() == 0
    archs = te_frame["arch"].to_numpy()[mask]
    correct = (np.asarray(y_pred)[mask] == 0)
    out = {}
    for a in sorted(set(archs)):
        sel = archs == a
        n = int(sel.sum())
        out[a or "unknown"] = {"n": n,
                               "recall_goodware": float(correct[sel].mean())}
    return out


def _feature_root(paths, kind: str) -> Path:
    """Traditional or REVISED feature tree for a Mendeley-shaped source."""
    return Path(paths["llm_features_revised" if kind.endswith("_revised")
                      else "llm_features"])


def build_samples(cfg, name, outroot: Path):
    """Resolve one experiment's sample set, split and cohort metadata.

    Six experiments come out of three orthogonal switches:

        ransomware:           mendeley | mendeley_revised
        goodware:             mendeley | mendeley_revised | balanced | balanced_revised
        cohort_filter:        false | true          (traditional corpora only)
        reuse_goodware_split: absent | <experiment> (take a committed membership)

    `_revised` swaps extract.py's full-instruction lines for
    extract_unified.py's mnemonic-only lines AND, because
    asm_tool/mn_to_features.py applies the cohort filter as it writes, swaps the
    shipped corpus for the cohort. Those two changes are separated by running
    `cohort_filter: true` on the traditional corpus as its own experiment.
    """
    exp = cfg["experiments"][name]
    paths = cfg["paths"]
    sp = cfg["split"]
    notes: dict = {}

    ran = D.load_mendeley(_feature_root(paths, exp["ransomware"]), "ransomware")

    good_kind = exp["goodware"]
    dedup = None
    if good_kind in ("mendeley", "mendeley_revised"):
        good = D.load_mendeley(_feature_root(paths, good_kind), "goodware")
    else:
        revised = good_kind.endswith("_revised")
        pool_dir = Path(paths["balanced_goodware_revised" if revised
                              else "balanced_goodware"])
        manifest = Path(paths["balanced_revised_manifest" if revised
                              else "balanced_manifest"])
        pool = D.load_balanced(pool_dir, manifest,
                               sp["goodware_group_field"],
                               sp.get("goodware_ungrouped_entries", []))
        # The dedup reference is the Mendeley goodware in the SAME feature
        # format. Comparing a mnemonic-only stream against a full-instruction
        # one could never match, so the content-hash channel would silently
        # stop working if the traditional tree were used for the revised run.
        ref = D.load_mendeley(_feature_root(
            paths, "mendeley_revised" if revised else "mendeley"), "goodware")
        pool_before_dedup = list(pool)
        if sp.get("dedup_goodware_sources", True):
            pool, dedup = D.dedup_goodware_sources(
                pool, ref, manifest, Path(paths["mendeley_goodware_sha256"]))
            print(f"  cross-source dedup: {dedup['removed_total']} of "
                  f"{dedup['pool_in']} balanced goodware files also appear in the "
                  f"Mendeley goodware set "
                  f"({len(dedup['removed_by_source_sha256'])} by binary sha256, "
                  f"{len(dedup['removed_by_content_hash'])} by opcode-stream hash)")
            for n in (dedup["removed_by_source_sha256"]
                      + dedup["removed_by_content_hash"])[:10]:
                print(f"    removed {n}")

        reuse = exp.get("reuse_goodware_split")
        if reuse:
            committed = outroot / reuse / "splits.csv"
            if not committed.is_file():
                committed = REPO / cfg["paths"]["results"] / reuse / "splits.csv"
            good, rep = D.reuse_split(pool, committed, "balanced_goodware")
            if dedup is not None and dedup["removed_total"]:
                # What the membership would have been WITHOUT the dedup guard.
                # It matters for Exp D: the mnemonic-only representation drops
                # the operands that used to tell two Inno/NSIS installer stubs
                # apart, so 13 Goodware_Balanced files become byte-identical to
                # a Mendeley goodware file that the full-instruction form kept
                # distinct. The guard removes them, which is the right call and
                # also the reason Exp D's goodware counts sit a few below Exp
                # B_cohort's. Both numbers are recorded so the gap is visible
                # rather than inferred.
                _, cf = D.reuse_split(pool_before_dedup, committed,
                                      "balanced_goodware")
                rep["would_select_train_without_dedup"] = cf["selected_train"]
                rep["would_select_test_without_dedup"] = cf["selected_test"]
                rep["removed_from_membership_by_dedup"] = (
                    cf["selected"] - rep["selected"])
            notes["reused_goodware_split"] = rep
            print(f"  reusing {reuse}'s goodware membership from {committed}: "
                  f"{rep['selected_train']} train / {rep['selected_test']} test "
                  f"selected from a pool of {rep['pool']} "
                  f"({rep['named_but_absent_from_pool']} of {rep['named_by_split']} "
                  f"named files are not in this pool)")
        else:
            if sp.get("match_counts_to"):
                n_tr = sum(1 for s in ref if s.split == "train")
                n_te = sum(1 for s in ref if s.split == "test")
            else:
                n_te = round(len(pool) * 0.105)
                n_tr = len(pool) - n_te
            print(f"  group-splitting {len(pool)} balanced goodware "
                  f"-> {n_tr} train / {n_te} test")
            good = D.group_split(pool, n_tr, n_te, sp["seed"])

    samples = ran + good

    # --- cohort metadata ---------------------------------------------------
    # Annotation runs for EVERY experiment, filtering only where asked. That is
    # what lets Exp A and Exp B - which are not cohort experiments - still be
    # reported per architecture, closing the question results/summary.md had to
    # leave open.
    coh_m = D.load_cohort(Path(paths["cohort_mendeley"]))
    coh_b = D.load_cohort(Path(paths["cohort_balanced"]))
    notes["cohort_annotation"] = {
        "mendeley": D.annotate_cohort(
            [s for s in samples if s.source.startswith("mendeley")], coh_m),
        "balanced": D.annotate_cohort(
            [s for s in samples if s.source == "balanced_goodware"], coh_b),
    }
    for side, rep in notes["cohort_annotation"].items():
        if rep["annotated"] or rep["unmatched"]:
            print(f"  cohort ({side}): {rep['annotated']} annotated, "
                  f"{rep['unmatched']} unmatched, {rep['in_cohort']} in cohort")
        if rep["unmatched"]:
            print(f"    e.g. {rep['unmatched_examples'][:3]}")

    if exp.get("cohort_filter"):
        before = D.summarize(samples)
        samples, rep = D.filter_cohort(samples)
        rep["applied"] = True
        rep["train_before"], rep["test_before"] = before["train"]["n"], before["test"]["n"]
        notes["cohort_filter"] = rep
        print(f"  cohort filter: kept {rep['out']} of {rep['in']} "
              f"(dropped {rep['dropped']}: {rep['dropped_by_reason']})")
    else:
        n_in = sum(1 for s in samples if s.meta.get("in_cohort"))
        notes["cohort_filter"] = {
            "applied": False, "in": len(samples), "out": len(samples),
            "would_keep_if_applied": n_in,
            # The revised folders ARE the cohort - mn_to_features.py filtered as
            # it wrote - so this reads len == would_keep for Exp C and Exp D and
            # is the assertion that nothing outside the cohort slipped in.
            "already_cohort_only": n_in == len(samples)}
    return samples, dedup, notes


def run_experiment(cfg, name, mods, outroot: Path, dry: bool):
    print("\n" + "=" * 72)
    print(f"{name}: {cfg['experiments'][name]['description']}")
    print("=" * 72)
    samples, dedup, notes = build_samples(cfg, name, outroot)
    summary = D.summarize(samples)
    if dedup is not None:
        summary["cross_source_dedup"] = dedup
    summary.update(notes)
    print("measuring exact-duplicate leakage ...", flush=True)
    summary["content_leak"] = D.content_leak(samples)
    summary["goodware_arch"] = {
        sp: dict(sorted(collections.Counter(
            s.meta.get("arch", "") or "unknown"
            for s in samples if s.split == sp and s.label == 0).items()))
        for sp in ("train", "test")}
    summary["test_goodware_arch"] = summary["goodware_arch"]["test"]
    # Both classes, both splits - the confound stated as counts.
    summary["arch"] = D.arch_breakdown(samples)
    # Two trivial rules scored on this test set, so no model number below is
    # read without a floor beside it. Derived from summary["arch"] alone, which
    # is what lets write_summary recover them from an older metrics.json.
    summary["baselines"] = D.baselines(summary["arch"])
    summary["ransomware_families"] = D.family_breakdown(samples)
    print(json.dumps(summary, indent=2))
    if summary["group_overlap"]:
        sys.exit(f"group leak across splits: {summary['group_overlap'][:5]}")

    outdir = outroot / name
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "splits.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        # `arch`, `family` and `cohort_tag` are new columns. Anything reading
        # this file by header name (results/cnn_vit/expB_splits_snapshot.csv's
        # consumer, the tests) is unaffected; anything reading it positionally
        # was already wrong.
        w.writerow(["file", "source", "label", "group", "split",
                    "arch", "family", "cohort_tag", "in_cohort"])
        for s in samples:
            w.writerow([s.name, s.source, s.label, s.group, s.split,
                        s.meta.get("arch", ""), s.meta.get("family", ""),
                        s.meta.get("cohort_tag", ""),
                        int(bool(s.meta.get("in_cohort")))])
    (outdir / "sample_counts.json").write_text(json.dumps(summary, indent=2))
    if dry:
        print("dry run: stopping before tokenization")
        return {"experiment": name,
                "description": cfg["experiments"][name]["description"],
                "samples": summary, "results": []}

    t0 = time.time()
    print("reading instruction text ...")
    df = build_frames(samples, mods["normalize"], cfg["tokenization"]["max_instructions"])
    methods = cfg["tokenization"]["methods"]

    embed_methods = cfg["embedding"]["methods"]
    w2v_cfg = cfg["embedding"]["w2v"]
    results = []
    pred_rows: list[dict] = []

    for model_type, tok_method, embed in COMBOS:
        if model_type not in cfg["classification"]["models"]:
            continue
        if tok_method not in methods:
            print(f"skip {model_type}/{tok_method}: method not enabled")
            continue
        use = embed
        if embed not in embed_methods:
            if "w2v" not in embed_methods:
                print(f"skip {model_type}/{tok_method}/{embed}: embedding disabled")
                continue
            use = "w2v"
            print(f"note: {embed} disabled, substituting w2v for {tok_method}")

        cache = cfg["tokenization"].get("tokenizer_cache", "results/tokenizers")
        cache = Path(cache)
        if not cache.is_absolute():
            cache = REPO / cache
        f = build_sequence_frame(df, tok_method, mods["train_tokenizer"],
                                 cfg["tokenization"]["vocab_size"],
                                 cache_dir=cache, tag=name)
        tr = f[f["split"] == "train"].reset_index(drop=True)
        te = f[f["split"] == "test"].reset_index(drop=True)
        del f

        for rate in cfg["embedding"]["mask_rates"]:
            print(f"\n--- {model_type} | {tok_method} | {use} | mask={rate} ---",
                  flush=True)
            if use == "w2v":
                # workers=1 is NOT a performance choice. gensim's default
                # workers=4 makes Word2Vec nondeterministic even with a fixed
                # seed: the worker threads consume the corpus in a racing order,
                # so two runs of the identical command produce different
                # vectors (measured: max abs difference 3.8e-2 per component on
                # a 300-token toy corpus; identical at workers=1). With
                # workers=4 the metrics in results/ cannot be reproduced.
                Xtr, Xte = mods["w2v"](tr, te, mask_rate=rate,
                                       seed=w2v_cfg["seed"],
                                       vector_size=w2v_cfg["vector_size"],
                                       window=w2v_cfg["window"],
                                       epochs=w2v_cfg["epochs"],
                                       workers=w2v_cfg.get("workers", 1))
            else:
                Xtr, Xte = mods["bert"](tr, te, mask_rate=rate, seed=w2v_cfg["seed"])
            assert len(Xtr) == len(tr) and len(Xte) == len(te), \
                "embedding rows out of step with labels"
            m = fit_and_score(model_type, Xtr, tr["Label"].to_numpy(),
                              Xte, te["Label"].to_numpy(),
                              cfg["classification"]["cv"],
                              cfg["classification"]["seed"])
            m.update(model=model_type, tokenizer=tok_method, embedding=use,
                     mask_rate=rate, n_train=len(tr), n_test=len(te))
            y_pred = m.pop("_y_pred")
            m["goodware_recall_by_arch"] = goodware_recall_by_arch(te, y_pred)
            m["per_arch"] = per_arch_metrics(te, y_pred)
            m["ransomware_recall_by_family"] = ransomware_recall_by_family(te, y_pred)
            # Predictions are saved, not just summarised. Without them a new
            # question about an old run (per architecture, per family, per
            # source) needs a full re-run to answer, and a re-run is only
            # trustworthy when the tokenizer cache is warm.
            for f_, l_, a_, fam_, p_ in zip(te["file"], te["Label"], te["arch"],
                                            te["family"], y_pred):
                pred_rows.append({"model": model_type, "tokenizer": tok_method,
                                  "embedding": use, "mask_rate": rate,
                                  "file": f_, "label": int(l_), "arch": a_ or "",
                                  "family": fam_ or "", "pred": int(p_)})
            results.append(m)
            auc = "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.4f}"
            print(f"    acc {m['accuracy']:.4f}  bal-acc {m['balanced_accuracy']:.4f}"
                  f"  macro-F1 {m['macro_f1']:.4f}  AUC {auc}"
                  f"  (majority baseline {m['majority_class_accuracy']:.4f})")
        del tr, te
        gc.collect()

    payload = {"experiment": name,
               "description": cfg["experiments"][name]["description"],
               "task": "binary: 0=goodware, 1=ransomware",
               "samples": summary, "results": results,
               # The config is embedded as well as written to config_used.yaml
               # so metrics.json is self-contained: a result and the settings
               # that produced it cannot be separated by copying one file.
               "config": cfg,
               "elapsed_seconds": round(time.time() - t0, 1)}
    (outdir / "metrics.json").write_text(json.dumps(payload, indent=2))
    (outdir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if pred_rows:
        with (outdir / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pred_rows[0]))
            w.writeheader()
            w.writerows(pred_rows)
        print(f"wrote {outdir / 'predictions.csv'} ({len(pred_rows)} rows)")
    print(f"\nwrote {outdir / 'metrics.json'}")
    return payload


# Reading order for the six-experiment tables. It is not the config order:
# the config runs expA and expB first because expB_cohort and expD reuse expB's
# committed goodware membership, but the comparison reads
# A -> A_cohort -> C along the Mendeley-goodware axis and
# B -> B_cohort -> D along the Goodware_Balanced axis.
DISPLAY_ORDER = ["expA", "expA_cohort", "expC", "expB", "expB_cohort", "expD"]

# The three one-variable moves the four new experiments buy. Each pair differs
# in exactly one thing; anything else in the table differs in two or more.
CONTRASTS = [
    ("expA", "expA_cohort", "sample set only (Mendeley goodware)"),
    ("expB", "expB_cohort", "sample set only (Goodware_Balanced)"),
    ("expA_cohort", "expC", "feature form only (Mendeley goodware)"),
    ("expB_cohort", "expD", "feature form only (Goodware_Balanced)"),
    ("expA", "expC", "both at once (Mendeley goodware)"),
    ("expB", "expD", "both at once (Goodware_Balanced)"),
]


def _ordered(payloads: dict) -> list:
    seen = [n for n in DISPLAY_ORDER if n in payloads]
    return seen + [n for n in payloads if n not in seen]


def _key(r) -> tuple:
    return (r["model"], r["tokenizer"], r["embedding"], r["mask_rate"])


def _by_key(p) -> dict:
    return {_key(r): r for r in p["results"]}


def _pct(d: dict) -> str:
    tot = sum(d.values()) or 1
    return ", ".join(f"{v} {k} ({100*v/tot:.1f}%)" for k, v in d.items())


def _arch_row(samples_block: dict, split: str, tag: str) -> str:
    d = samples_block.get("arch", {}).get(f"{split}_{tag}", {})
    return _pct(d) if d else "-"


def _floors(p) -> dict:
    """The two trivial-rule floors for one experiment.

    Taken from the payload when the run that produced it recorded them, and
    otherwise recomputed from its `arch` counts - which is the same arithmetic
    on the same numbers, so a summary rebuilt from an older metrics.json says
    exactly what a fresh run would say.
    """
    s = p.get("samples", {})
    return s.get("baselines") or D.baselines(s.get("arch", {}))


FLOOR_ROWS = [("majority_class", "*floor* - majority class"),
              ("x86_is_ransomware", "*floor* - x86 means ransomware")]


def _arch_block(p) -> list:
    """Per-architecture test metrics for BOTH classes.

    The old version of this table could only show goodware recall, and only for
    Goodware_Balanced, because nothing else carried a per-file architecture.
    The cohort CSVs carry it for every file of every set, so every experiment
    now gets both classes and both slices.
    """
    lines = ["", "#### Test metrics inside each architecture", ""]
    archs = sorted({a for r in p["results"] for a in r.get("per_arch", {})})
    if not archs:
        return lines + ["No per-architecture breakdown was recorded for this run.", ""]
    head = "| model / tok |"
    sep = "|---|"
    for a in archs:
        head += f" {a} n | {a} rec(good) | {a} rec(ran) | {a} acc | {a} macro-F1 |"
        sep += "---|---|---|---|---|"
    lines += [head, sep]
    for r in p["results"]:
        cells = [f"| {r['model']}/{r['tokenizer']} |"]
        for a in archs:
            s = r.get("per_arch", {}).get(a)
            if not s:
                cells.append(" - | - | - | - | - |")
                continue

            def f(v):
                return "-" if v is None else f"{v:.4f}"
            cells.append(f" {s['n']} | {f(s['recall_goodware'])} "
                         f"({s['support_goodware']}) | {f(s['recall_ransomware'])} "
                         f"({s['support_ransomware']}) | {f(s['accuracy'])} | "
                         f"{f(s['macro_f1'])} |")
        lines.append("".join(cells))
    lines += ["",
              "`rec(good)` and `rec(ran)` carry their class support in brackets. "
              "A slice holding only one class gets `macro-F1 -`: averaging an F1 "
              "over a class with no support reports \"the model failed\" when the "
              "truth is \"the question was not asked\".", ""]
    return lines


def _family_block(p) -> list:
    """Per-family ransomware recall on the test set.

    The Mendeley test families are entirely unseen in training, so the pooled
    ransomware recall is an average over generalisation to 14 different
    strangers. Two large families can carry it on their own.
    """
    fams = sorted({f for r in p["results"]
                   for f in r.get("ransomware_recall_by_family", {})})
    if not fams:
        return []
    supports = {}
    for r in p["results"]:
        for f, v in r.get("ransomware_recall_by_family", {}).items():
            supports[f] = v["n"]
    lines = ["", "#### Ransomware test recall by family", "",
             "| family | n | " + " | ".join(f"{r['model']}/{r['tokenizer']}"
                                            for r in p["results"]) + " |",
             "|---|---|" + "---|" * len(p["results"])]
    for f in sorted(fams, key=lambda f: (-supports.get(f, 0), f)):
        row = [f"| {f} | {supports.get(f, 0)} |"]
        for r in p["results"]:
            v = r.get("ransomware_recall_by_family", {}).get(f)
            row.append(f" {v['recall']:.4f} |" if v else " - |")
        lines.append("".join(row))
    lines.append("")
    return lines


def _counts_section(name, p) -> list:
    s = p["samples"]
    lines = [f"## {name} - {p['description']}", "",
             "| split | goodware | ransomware | total | groups |",
             "|---|---|---|---|---|"]
    for sp in ("train", "test"):
        d = s[sp]
        lines.append(f"| {sp} | {d['goodware']} | {d['ransomware']} | "
                     f"{d['n']} | {d['groups']} |")
    lines += ["", "| split | class | x86 / x64 |", "|---|---|---|"]
    for sp in ("train", "test"):
        for tag in ("ransomware", "goodware"):
            lines.append(f"| {sp} | {tag} | {_arch_row(s, sp, tag)} |")
    lines.append("")

    cf = s.get("cohort_filter", {})
    if cf.get("applied"):
        lines += [f"Cohort filter: kept **{cf['out']} of {cf['in']}** files "
                  f"(dropped {cf['dropped']}: "
                  + ", ".join(f"{k} {v}" for k, v in cf["dropped_by_reason"].items())
                  + ").", ""]
    elif cf:
        lines += [("The feature folders are already cohort-only, and this run "
                   "verified it: all "
                   f"{cf['out']} files carry `in_cohort == 1`."
                   if cf.get("already_cohort_only") else
                   f"No cohort filter. {cf.get('would_keep_if_applied')} of "
                   f"{cf['out']} files would survive one."), ""]

    ru = s.get("reused_goodware_split")
    if ru:
        lines += [f"Goodware membership reused verbatim from `{ru['source']}`'s "
                  f"committed `splits.csv`: **{ru['selected_train']} train / "
                  f"{ru['selected_test']} test** selected out of a pool of "
                  f"{ru['pool']}; {ru['named_but_absent_from_pool']} of the "
                  f"{ru['named_by_split']} files that split names are not in "
                  f"this pool."]
        if "would_select_train_without_dedup" in ru:
            lines += ["",
                      f"Without the cross-source dedup guard it would have been "
                      f"{ru['would_select_train_without_dedup']} train / "
                      f"{ru['would_select_test_without_dedup']} test; the guard "
                      f"removed {ru['removed_from_membership_by_dedup']} files "
                      f"from the membership."]
        lines.append("")

    dd = s.get("cross_source_dedup")
    if dd and dd["removed_total"]:
        lines += [f"Cross-source dedup removed **{dd['removed_total']} of "
                  f"{dd['pool_in']}** pool files "
                  f"({len(dd['removed_by_source_sha256'])} by binary sha256, "
                  f"{len(dd['removed_by_content_hash'])} by opcode-stream hash): "
                  "each is byte-identical, as an opcode stream, to a Mendeley "
                  "goodware file. Examples: "
                  + ", ".join(f"`{n}`" for n in
                              (dd["removed_by_source_sha256"]
                               + dd["removed_by_content_hash"])[:4]) + ".", ""]

    cl = s.get("content_leak", {})
    if cl:
        lines += [f"Exact-duplicate leakage: **{cl['test_goodware_duplicated_in_train']}"
                  f"/{cl['test_goodware_n']} goodware** and "
                  f"**{cl['test_ransomware_duplicated_in_train']}"
                  f"/{cl['test_ransomware_n']} ransomware** test files have a "
                  f"byte-identical opcode stream in train "
                  f"({cl['unique_streams']}/{cl['total_files']} streams are unique; "
                  f"{cl['streams_labelled_both_classes']} appear under both labels).",
                  ""]

    lines += ["", "| model | tok | emb | mask | acc | bal-acc | macro-P | "
              "macro-R | macro-F1 | AUC | recall(ran) | FPR |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in p["results"]:
        auc = "-" if r["roc_auc"] is None else f"{r['roc_auc']:.4f}"
        fpr = "-" if r["false_positive_rate"] is None else f"{r['false_positive_rate']:.4f}"
        lines.append(f"| {r['model']} | {r['tokenizer']} | {r['embedding']} | "
                     f"{r['mask_rate']} | {r['accuracy']:.4f} | "
                     f"{r['balanced_accuracy']:.4f} | "
                     f"{r['macro_precision']:.4f} | {r['macro_recall']:.4f} | "
                     f"{r['macro_f1']:.4f} | {auc} | "
                     f"{r['recall_ransomware']:.4f} | {fpr} |")
    fl = _floors(p)
    for key, label in FLOOR_ROWS:
        b = fl[key]
        lines.append(f"| {label} | - | - | - | {b['accuracy']:.4f} | "
                     f"{b['balanced_accuracy']:.4f} | - | - | "
                     f"{b['macro_f1']:.4f} | - | "
                     f"{b['recall_ransomware']:.4f} | "
                     f"{1 - b['recall_goodware']:.4f} |")
    lines += ["",
              f"The two floor rows read no opcodes. `majority class` calls "
              f"every test file "
              f"{fl['majority_class']['predicts']}; `x86 means ransomware` "
              f"reads only the architecture recorded in the cohort CSV "
              f"({fl['x86_is_ransomware']['test_x86']} of "
              f"{sum(fl['x86_is_ransomware']['confusion_matrix'].values())} "
              f"test files are x86"
              + (f", {fl['x86_is_ransomware']['test_unknown_arch']} are "
                 f"`unknown` and count as not-x86"
                 if fl["x86_is_ransomware"]["test_unknown_arch"] else "")
              + "). A model row is evidence about opcodes only to the extent "
              "that it clears both.", ""]
    lines += _arch_block(p)
    lines += _family_block(p)
    return lines


def _six_way_table(payloads) -> list:
    order = _ordered(payloads)
    keyed = {n: _by_key(payloads[n]) for n in order}
    combos = []
    for n in order:
        for k in keyed[n]:
            if k not in combos:
                combos.append(k)
    floors = {n: _floors(payloads[n]) for n in order}

    def floor_rows(metric):
        out = []
        for key, label in FLOOR_ROWS:
            out.append(f"| {label} | " + " | ".join(
                f"{floors[n][key][metric]:.4f}" for n in order) + " |")
        return out

    lines = ["## Six-experiment comparison", "",
             "Matched rows: the same classifier, tokenizer, embedding and mask "
             "rate, with the same seeds, in every experiment. Cells are "
             "**macro-F1** on the test set. The last two rows are floors: two "
             "rules that read no opcodes at all, scored on the same test set. "
             "A model row that does not clear them has not been shown to use "
             "the code.", "",
             "| model / tok / emb / mask | " + " | ".join(order) + " |",
             "|---|" + "---|" * len(order)]
    for k in combos:
        cells = []
        for n in order:
            r = keyed[n].get(k)
            cells.append(f"{r['macro_f1']:.4f}" if r else "-")
        lines.append(f"| {' / '.join(map(str, k))} | " + " | ".join(cells) + " |")
    lines += floor_rows("macro_f1")
    lines += ["", "Same rows, **balanced accuracy**:", "",
              "| model / tok / emb / mask | " + " | ".join(order) + " |",
              "|---|" + "---|" * len(order)]
    for k in combos:
        cells = []
        for n in order:
            r = keyed[n].get(k)
            cells.append(f"{r['balanced_accuracy']:.4f}" if r else "-")
        lines.append(f"| {' / '.join(map(str, k))} | " + " | ".join(cells) + " |")
    lines += floor_rows("balanced_accuracy")
    says = {floors[n]["majority_class"]["predicts"] for n in order}
    which = (f"{says.pop()} in all {len(order)}" if len(says) == 1 else
             ", ".join(f"{n}: {floors[n]['majority_class']['predicts']}"
                       for n in order))
    lines += ["",
              f"`majority class` predicts the larger test class for every file "
              f"({which}), so its balanced accuracy is 0.5000 by construction "
              f"and only its macro-F1 moves with the class ratio. "
              f"`x86 means ransomware` reads the architecture column of the "
              f"cohort CSV and nothing else; it is the confound of the section "
              f"below, priced.", ""]

    lines += ["### One-variable deltas (macro-F1)", "",
              "Each block moves exactly one thing. Positive means the second "
              "experiment scores higher.", "",
              "| contrast | what moves | " +
              " | ".join(" / ".join(map(str, k)) for k in combos) + " |",
              "|---|---|" + "---|" * len(combos)]
    for a, b, what in CONTRASTS:
        if a not in payloads or b not in payloads:
            continue
        cells = []
        for k in combos:
            ra, rb = keyed[a].get(k), keyed[b].get(k)
            cells.append(f"{rb['macro_f1'] - ra['macro_f1']:+.4f}"
                         if ra and rb else "-")
        lines.append(f"| {a} -> {b} | {what} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _six_way_arch_table(payloads) -> list:
    order = _ordered(payloads)
    lines = ["### Per-architecture test metrics, all experiments", "",
             "macro-F1 computed inside each architecture slice. A model that "
             "has learned \"x64 means benign\" cannot use that shortcut inside "
             "a slice where every sample has the same bitness, so a large drop "
             "from the pooled number is the signature of exactly that.", "",
             "| experiment | model / tok | pooled macro-F1 | x86 n | x86 macro-F1 "
             "| x64 n | x64 macro-F1 | x86 rec(ran) | x64 rec(ran) "
             "| x86 rec(good) | x64 rec(good) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in order:
        for r in payloads[n]["results"]:
            pa = r.get("per_arch", {})

            def cell(a, f):
                s = pa.get(a)
                if not s or s.get(f) is None:
                    return "-"
                return f"{s[f]:.4f}"

            def nn(a):
                return str(pa.get(a, {}).get("n", "-"))
            lines.append(
                f"| {n} | {r['model']}/{r['tokenizer']} | {r['macro_f1']:.4f} | "
                f"{nn('x86')} | {cell('x86', 'macro_f1')} | "
                f"{nn('x64')} | {cell('x64', 'macro_f1')} | "
                f"{cell('x86', 'recall_ransomware')} | "
                f"{cell('x64', 'recall_ransomware')} | "
                f"{cell('x86', 'recall_goodware')} | "
                f"{cell('x64', 'recall_goodware')} |")
    lines.append("")
    return lines


def _six_way_family_table(payloads) -> list:
    order = _ordered(payloads)
    fams, supports = set(), {}
    for n in order:
        for r in payloads[n]["results"]:
            for f, v in r.get("ransomware_recall_by_family", {}).items():
                fams.add(f)
                supports.setdefault((n, f), v["n"])
    if not fams:
        return []
    lines = ["### Ransomware test recall by family, RF/WPC row of each experiment",
             "",
             "One model per experiment keeps the table readable; the per-experiment "
             "sections above carry all three. `n` differs between the traditional "
             "and revised columns because the cohort filter and the two extractors "
             "keep different files.", "",
             "| family | " + " | ".join(f"{n} n | {n} recall" for n in order) + " |",
             "|---|" + "---|---|" * len(order)]
    picked = {}
    for n in order:
        rs = payloads[n]["results"]
        picked[n] = next((r for r in rs if r["model"] == "RF"), rs[0] if rs else None)
    order_f = sorted(fams, key=lambda f: (-max(supports.get((n, f), 0) for n in order), f))
    for f in order_f:
        row = [f"| {f} |"]
        for n in order:
            r = picked[n]
            v = (r or {}).get("ransomware_recall_by_family", {}).get(f)
            row.append(f" {v['n']} | {v['recall']:.4f} |" if v else " - | - |")
        lines.append("".join(row))
    lines += ["",
              "Row order is by family size. A family present in one column and "
              "absent from another was removed entirely by the cohort filter "
              "(`thanos`) or has no surviving member in that feature tree.", ""]
    return lines


def _arch_verdict(payloads) -> list:
    """State what the per-architecture numbers show, rather than leaving the
    reader to eyeball the table."""
    rows = []
    for name, p in payloads.items():
        for r in p["results"]:
            pa = r.get("per_arch", {})
            x86, x64 = pa.get("x86"), pa.get("x64")
            if not (x86 and x64):
                continue
            if x86.get("macro_f1") is None or x64.get("macro_f1") is None:
                continue
            rows.append({
                "exp": name, "combo": f"{r['model']}/{r['tokenizer']}",
                "pooled": r["macro_f1"],
                "x86": x86["macro_f1"], "x64": x64["macro_f1"],
                "good86": x86["recall_goodware"], "good64": x64["recall_goodware"],
            })
    if not rows:
        return ["No experiment produced both architecture slices with both "
                "classes present, so the comparison stays open."]
    gaps = [r["x64"] - r["x86"] for r in rows]
    good_gaps = [r["good64"] - r["good86"] for r in rows
                 if r["good64"] is not None and r["good86"] is not None]
    worst = max(rows, key=lambda r: r["x64"] - r["x86"])
    drops = [max(r["pooled"] - r["x86"], r["pooled"] - r["x64"]) for r in rows]
    out = [f"Across the {len(rows)} model/experiment pairs where both slices "
           f"hold both classes, within-x64 macro-F1 runs from "
           f"{min(gaps):+.4f} to {max(gaps):+.4f} relative to within-x86 "
           f"(largest gap: {worst['exp']} {worst['combo']}, {worst['x64']:.4f} "
           f"on x64 against {worst['x86']:.4f} on x86)."]
    if good_gaps:
        out.append(f"On the goodware class alone the x64-minus-x86 recall gap "
                   f"runs from {min(good_gaps):+.4f} to {max(good_gaps):+.4f}.")
    out.append(f"Splitting the test set by bitness costs up to "
               f"{max(drops):.4f} macro-F1 against the pooled score, which is "
               f"the size of the shortcut the pooled number was buying.")
    return out


def _floor_verdict(payloads) -> list:
    """How the model rows stand against the architecture-only rule.

    The rule reads the cohort CSV's `arch` column and nothing else, so anything
    it beats has not been shown to use the opcode stream at all. Whether it is
    a hard floor or a trivial one is itself an experiment-level fact, because
    the two goodware sources have opposite bitness mixes on the test side.
    """
    order = _ordered(payloads)
    lines = []
    for n in order:
        fl = _floors(payloads[n])["x86_is_ransomware"]
        rows = payloads[n]["results"]
        beaten = [r for r in rows if r["macro_f1"] <= fl["macro_f1"]]
        verdict = ("every model row clears it" if not beaten else
                   ", ".join(f"{r['model']}/{r['tokenizer']} {r['macro_f1']:.4f}"
                             for r in beaten) +
                   (" does not clear it" if len(beaten) == 1
                    else " do not clear it"))
        lines.append(f"- **{n}**: the architecture-only rule scores macro-F1 "
                     f"{fl['macro_f1']:.4f} / balanced accuracy "
                     f"{fl['balanced_accuracy']:.4f} on this test set; "
                     f"{verdict}.")
    worst = max(order, key=lambda n: _floors(payloads[n])["x86_is_ransomware"]["macro_f1"])
    best = min(order, key=lambda n: _floors(payloads[n])["x86_is_ransomware"]["macro_f1"])
    lines += ["",
              f"The rule is not equally strong everywhere, and the reason is in "
              f"the table above rather than in the models: it is worth "
              f"macro-F1 "
              f"{_floors(payloads[worst])['x86_is_ransomware']['macro_f1']:.4f} "
              f"in `{worst}`, where the goodware half is Goodware_Balanced and "
              f"mostly x64, and only "
              f"{_floors(payloads[best])['x86_is_ransomware']['macro_f1']:.4f} "
              f"in `{best}`, where Mendeley `good_test` is ~91% x86 and so "
              f"looks, to a bitness rule, exactly like the ransomware. A "
              f"Mendeley-goodware experiment therefore cannot be cleared of the "
              f"confound by pointing at this floor: the shortcut is in its "
              f"TRAINING mix (~57% x86 goodware against ~95% x86 ransomware) "
              f"and it MISFIRES on its own test set, which is what the collapsed "
              f"x64 slices in the per-architecture table are."]
    return lines


def _architecture_note(payloads) -> list:
    order = _ordered(payloads)
    lines = ["", "### Architecture: the confound, now measured rather than assumed", "",
             "Every count below comes from `cohort_mendeley.csv` / "
             "`cohort_balanced.csv`, which record the architecture of every "
             "input binary of every set. The previous version of this document "
             "had to leave `good_test` and the whole ransomware side as "
             "`unknown`, because the feature files carry no architecture and "
             "those binaries are not on this machine. That gap is closed.", "",
             "| experiment | train ransomware | test ransomware | train goodware "
             "| test goodware |",
             "|---|---|---|---|---|"]
    for n in order:
        s = payloads[n]["samples"]
        lines.append(f"| {n} | {_arch_row(s, 'train', 'ransomware')} | "
                     f"{_arch_row(s, 'test', 'ransomware')} | "
                     f"{_arch_row(s, 'train', 'goodware')} | "
                     f"{_arch_row(s, 'test', 'goodware')} |")
    lines += ["",
              "**State it plainly.** The ransomware side is overwhelmingly x86 "
              "(~96% of train, ~80% of test once the cohort filter has removed "
              "the packed samples). Mendeley goodware is ~57% x86 in train but "
              "~91% x86 in test - the two goodware splits do not even match each "
              "other. Goodware_Balanced is ~18-23% x86. So in every experiment "
              "here, \"x64\" is evidence for benign and \"x86\" is evidence for "
              "ransomware before a single opcode is read - in TRAINING. The "
              "imbalance is different on each side of each split, and that "
              "difference decides whether the shortcut still pays on the test "
              "set:",
              "",]
    lines += _floor_verdict(payloads)
    lines += ["",
              "Bitness is not hidden from the model. In the TRADITIONAL feature "
              "form it is written all over the operands (`rbp`, `r8`-`r15`, "
              "rip-relative addressing, the register calling convention) and "
              "survives `normalize_instruction` into the token stream. The "
              "REVISED form drops operands for exactly this reason - a mnemonic "
              "line is `mov`, not `mov rbp, rsp` - but it does not remove the "
              "signal entirely: the x64-only instruction set (`vpxor`, `rorx`, "
              "`cmpxchg16b`, the AVX/AVX-512 mnemonics that dominate the "
              "candidate-only vocabulary in the schema check) is still visible "
              "as a mnemonic.",
              "",
              "The tables above are the test: score each architecture slice on "
              "its own, where bitness separates nothing, and see how much is "
              "left.", ""]
    lines += _arch_verdict(payloads)
    lines.append("")
    return lines


def _mnemonic_note(payloads) -> list:
    return [
        "", "### Mnemonic-only input: what the tokenizers actually do", "",
        "The revised feature files hold one mnemonic per line. Measured on the "
        "Exp C training corpus under the pipeline's own 5,000-instruction cap:",
        "",
        "| corpus | distinct normalized lines | example lines |",
        "|---|---|---|",
        "| traditional (Exp A train) | 36,802 | `push ebp`, `mov ebp esp`, "
        "`add byte ptr [eax] al`, `call <HEX>` |",
        "| revised (Exp C train) | 466 | `mov`, `push`, `add`, `call`, `int3` |",
        "",
        "That changes what each tokenizer means, and the three of them stop "
        "being three different things:",
        "",
        "* **SW** - one token per line. Unchanged in kind; the token is now a "
        "bare mnemonic.",
        "* **WP** - adjacent-line bigrams, `mov_push`. Still a real second view "
        "of the stream, and on mnemonic-only input it is the only one that "
        "carries any order information.",
        "* **WPC** - WordPiece over the `<SEP>`-joined line text. **On "
        "mnemonic-only input it is whole-word tokenization to within a rounding "
        "error.** The cached tokenizer that produced the committed Exp C "
        "numbers (`results/tokenizers/expC_WPC.json`) holds 941 entries against "
        "the 1,000 asked for - 667 word-initial and 274 `##` continuation "
        "pieces - so the trainer stopped short of the cap. The corpus holds 478 "
        "distinct mnemonics, and **466 of them encode as a single whole-word "
        "token**. The 12 that fragment (`cvtpd2ps` -> `cvt ##pd ##2ps`, "
        "`xacquire` -> six pieces) are *exactly* the 12 that occur only in the "
        "test split, which the tokenizer - correctly fit on train rows only - "
        "has never seen as a word. Between them they account for **176 of "
        "11,952,897** mnemonic occurrences, 0.0015%. There are no `<UNK>` "
        "tokens at all: the `##` machinery is doing nothing but keeping 12 "
        "unseen-at-training mnemonics out of `<UNK>`.",
        "",
        "So in Exp C and Exp D, **RF/WPC and any SW-based row are reading all "
        "but the same token stream** - 176 occurrences in 11.95 million differ "
        "- and the subword tokenizer is contributing next to nothing. That is "
        "worth knowing before reading a WPC row in the revised columns as "
        "evidence about subword tokenization: it is evidence about whole-word "
        "mnemonic tokenization wearing a WordPiece label.",
        "",
        "One consequence, tested rather than argued. The WPC nondeterminism "
        "that forced the tokenizer cache (docs/tokenization_audit.md §1.10) "
        "comes from the tie-break at the vocabulary size cutoff, and on "
        "mnemonic-only input the trainer stops short of that cutoff. It is "
        "still not bit-deterministic: a vocabulary trained fresh in another "
        "process differs from the cached one in 12 of its 941 entries and in "
        "the ids of 102 more. But every one of those differences is an unused "
        "fragment - both vocabularies encode all 478 distinct mnemonics to the "
        "same token strings - so Exp C, re-run twice with an empty tokenizer "
        "cache deleted between the runs, produced a `predictions.csv` "
        "byte-identical to the committed one both times. The cache is still "
        "used, and is still required for the four traditional-feature "
        "experiments, where the cutoff is reached.",
        "",
        "The whole-corpus mnemonic vocabularies, from `check_schema.py` (which "
        "reads every line, not the first 5,000): revised Mendeley good_train "
        "605, good_test 514, mal_train 698, mal_test 611; revised "
        "Goodware_Balanced 1,292. All five directories pass every hard schema "
        "check - one instruction per line, no address prefix, no tabs, no "
        "comments, no directives, no empty files. The mnemonic-set difference "
        "against the traditional reference is large by design (Jaccard 0.47-0.72) "
        "and is reported, not treated as a failure: the revised extractor sweeps "
        "past undecodable bytes instead of stopping at them, so it reaches "
        "AVX/AVX-512 code the linear sweep never got to.", ""]


def _deviation_note(cfg) -> list:
    return ["", "### Deviation from the plan's split procedure", "",
            "The plan asked for \"the same stratified 80/20 split procedure\" in "
            "both experiments. Three departures, all deliberate:", "",
            "1. **The ransomware split is the Mendeley release's own "
            "family-disjoint split, not a random stratified 80/20.** "
            "`mal_train` holds 25 families and `mal_test` 15 entirely different "
            "ones, with zero overlap - a genuine unseen-family generalisation "
            "test. A random stratified re-split destroys it: `dharma` alone "
            "contributes 45 byte-identical samples, `phobos` 37, `lockbit` 35, "
            "so copies of one stream land on both sides and the task collapses "
            "into near-duplicate retrieval. Measured on the shipped split, "
            "ransomware test leakage is 1/382; a random re-split would take it "
            "far higher. `config.yaml` sets `split.ransomware: "
            "preserve_mendeley`, and the identical ransomware split is reused "
            "verbatim in every experiment, so it contributes nothing to any "
            "difference between them. The cohort filter removes whole families "
            "only where every member was packed: `thanos` is gone from train "
            "and `night sky` from test, leaving 24 and 14 families.",
            "2. **Exp A's goodware split is also the release's own**, not a "
            "re-split. It has no group discipline, which is precisely what "
            "§2.1 of the audit measures: 64/131 of `good_test` is a verbatim "
            "copy of a training file. It is kept as shipped so Exp A remains "
            "the published baseline to compare against. Exp B's goodware side "
            "*is* split the way the plan intends - grouped by `entry_id` so no "
            "source project straddles train and test, stratified by bucket so "
            "`everyday`/`hard_negative`/`system` keep their proportions on both "
            "sides, with a fixed seed - and its counts are matched to Exp A's "
            "exactly (`split.match_counts_to: expA`).",
            "3. **Exp B_cohort and Exp D do not re-split at all.** They take Exp "
            "B's committed goodware membership out of `results/expB/splits.csv` "
            "and intersect it with the pool in front of them "
            "(`data.reuse_split`). Re-running `group_split` on a pool the cohort "
            "filter has changed would move whole source projects across the "
            "train/test line, and the result would then differ from Exp B for "
            "two reasons at once - which is the one thing these four "
            "experiments exist to avoid.", "",
            "Net effect: A is the baseline on its own terms, B is the same "
            "ransomware task with a harder, properly grouped goodware half, and "
            "the four new experiments move one variable at a time off those "
            "two."]


def write_summary(cfg, payloads, outroot: Path):
    order = _ordered(payloads)
    lines = ["# Experiment summary: six variants of one binary task", "",
             "Binary task throughout: **0 = goodware, 1 = ransomware**. "
             "The ransomware family prefix is a split group, never a label.", "",
             "Two variables are separated here, and they used to move together:",
             "",
             "| | sample set | feature form |",
             "|---|---|---|",
             "| **expA** | Mendeley as shipped | full instruction per line |",
             "| **expA_cohort** | cohort-filtered | full instruction per line |",
             "| **expC** | cohort-filtered | mnemonic only |",
             "| **expB** | Mendeley ransomware + Goodware_Balanced, as shipped "
             "| full instruction per line |",
             "| **expB_cohort** | cohort-filtered | full instruction per line |",
             "| **expD** | cohort-filtered | mnemonic only |",
             "",
             "The **cohort** drops .NET, entropy-packed, UPX-unrecoverable, "
             "no-code, odd-architecture, broken and duplicate samples, and the "
             "`thanos` family. The **revised** feature form comes from "
             "`extract_unified.py` (capstone skip-data sweep, uncapped) via "
             "`asm_tool/mn_to_features.py`, which applies the cohort filter as "
             "it writes - so expC and expD are cohort-only by construction and "
             "differ from expA_cohort and expB_cohort in the feature form alone.",
             "",
             "Everything else is held fixed across all six: the same tokenizer, "
             "embedding and classifier settings, the same seeds, the same "
             "ransomware split, and - for expB, expB_cohort and expD - the same "
             "goodware split membership.", ""]

    for name in order:
        lines += _counts_section(name, payloads[name])

    if len(payloads) > 1:
        lines += _six_way_table(payloads)
        lines += _six_way_arch_table(payloads)
        lines += _six_way_family_table(payloads)
        lines += _reading(payloads)
        lines += _architecture_note(payloads)
        lines += _mnemonic_note(payloads)
        lines += _deviation_note(cfg)

    out = outroot / "summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def _reading(payloads) -> list:
    """What the two moves actually do, in numbers pulled from this run."""
    keyed = {n: _by_key(payloads[n]) for n in payloads}
    combos = sorted({k for d in keyed.values() for k in d})

    def mean_delta(a, b):
        vals = [keyed[b][k]["macro_f1"] - keyed[a][k]["macro_f1"]
                for k in combos if k in keyed.get(a, {}) and k in keyed.get(b, {})]
        return (sum(vals) / len(vals), min(vals), max(vals), len(vals)) if vals else None

    lines = ["## Reading the six", ""]

    def block(title, pairs, body):
        out = [f"### {title}", ""]
        for a, b in pairs:
            d = mean_delta(a, b)
            if d:
                out.append(f"- `{a}` -> `{b}`: macro-F1 moves by "
                           f"**{d[0]:+.4f}** on average across {d[3]} matched "
                           f"rows (range {d[1]:+.4f} to {d[2]:+.4f}).")
        out += [""] + body + [""]
        return out

    lines += block(
        "What the cohort filter does",
        [("expA", "expA_cohort"), ("expB", "expB_cohort")],
        ["The filter removes samples, not information: 104 files leave Exp A "
         "(88 entropy-packed, 15 duplicate rows, 1 `thanos`) and 102 leave Exp "
         "B. Those are overwhelmingly the packed ransomware, whose opcode "
         "stream is the packer's stub rather than the payload's. Removing them "
         "should make the task HARDER in the sense that a trivially separable "
         "group of positives is gone, and EASIER in the sense that the "
         "remaining positives are the ones a model can actually learn "
         "something about; the sign of the measured move is in the table above "
         "rather than in this sentence.",
         "",
         "The filter also changes the architecture mix: it removes proportionally "
         "more x64 ransomware from the test set than x86, and it removes the "
         "`thanos` family outright."])

    lines += block(
        "What the traditional -> revised change does",
        [("expA_cohort", "expC"), ("expB_cohort", "expD")],
        ["Two things move at once inside this one contrast, and they should not "
         "be conflated:",
         "",
         "1. **Operands are gone.** A line is `mov`, not `mov rbp, rsp`. That "
         "removes the most direct architecture tell (`rbp`/`r8`-`r15`/"
         "rip-relative) and collapses the distinct-line vocabulary from 36,802 "
         "to 466, so the feature space is far smaller and far less able to "
         "memorise a specific binary.",
         "2. **The sweep is different.** `extract_unified.py` skips undecodable "
         "bytes and keeps going instead of stopping at the first one, and it is "
         "uncapped. Under the pipeline's 5,000-instruction cap that mostly means "
         "the 5,000 instructions come from further into the binary, and from "
         "code the linear sweep could not reach at all - which is why the "
         "revised mnemonic vocabulary contains hundreds of AVX/AVX-512 "
         "mnemonics the traditional one never saw.",
         "",
         "A consequence worth stating on its own: under mnemonic-only features, "
         "**13 Goodware_Balanced files become byte-identical to a Mendeley "
         "goodware file** that the full-instruction form kept distinct. They are "
         "Inno Setup and NSIS installer stubs - re-measured 14 September 2026, "
         "**six** distinct streams cover all 13, and each of the six is shared "
         "with between one and five Mendeley goodware files - and the "
         "cross-source dedup guard removes them, which is why Exp D's goodware "
         "counts sit a little under Exp B_cohort's. The traditional form removed "
         "zero. Dropping operands makes the installer-stub problem from audit "
         "§2.1 strictly more visible, not less.",
         "",
         "Exp D's test goodware count follows from that in two steps, and both "
         "are readable off other experiments: of Exp B's 131 test goodware "
         "files, **4** have no revised feature file at all - they are exactly "
         "the four the cohort filter drops in Exp B_cohort, which is why that "
         "experiment has 127 - and **4** more are removed by the cross-source "
         "dedup guard, leaving 123."])

    lines += block(
        "Both at once",
        [("expA", "expC"), ("expB", "expD")],
        ["This is the comparison that was available before the two cohort "
         "experiments existed, and it is the one that cannot be interpreted: "
         "the sample set and the feature form move together. It is kept in the "
         "table only so the decomposition above can be checked against it."])
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--experiment", default="", help="run just one (expA|expB)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve splits and stop, without tokenizing")
    ap.add_argument("--results-dir", default="",
                    help="write under this directory instead of paths.results "
                         "(use it to reproduce a run without overwriting the "
                         "committed results)")
    ap.add_argument("--summary-only", action="store_true",
                    help="rebuild summary.md from the metrics.json files "
                         "already on disk, running nothing. A full sweep is "
                         "~30 minutes and the prose in summary.md is edited "
                         "far more often than the numbers are recomputed.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    outroot = Path(args.results_dir or cfg["paths"]["results"])
    if not outroot.is_absolute():
        outroot = REPO / outroot
    outroot.mkdir(parents=True, exist_ok=True)

    names = [args.experiment] if args.experiment else list(cfg["experiments"])

    if args.summary_only:
        payloads = {}
        for n in cfg["experiments"]:
            f = outroot / n / "metrics.json"
            if f.is_file():
                payloads[n] = json.loads(f.read_text(encoding="utf-8"))
            else:
                print(f"  no {f}, leaving {n} out of the summary")
        if not payloads:
            sys.exit(f"nothing to summarise under {outroot}")
        write_summary(cfg, payloads, outroot)
        return 0

    if args.dry_run:
        mods = {"normalize": lambda s: (s.strip().lower() or None)}
    else:
        mods = load_repo_modules(Path(cfg["paths"]["tokenization_repo"]))

    payloads = {}
    for n in names:
        payloads[n] = run_experiment(cfg, n, mods, outroot, args.dry_run)
    if args.dry_run:
        return 0
    if len(payloads) < len(cfg["experiments"]):
        # Don't clobber a six-experiment summary with a one-experiment file.
        # Re-running a single experiment is a normal thing to do while
        # iterating; silently replacing summary.md with a sixth of it is not.
        # `--summary-only` rebuilds it from whatever is on disk.
        print(f"ran {len(payloads)} of {len(cfg['experiments'])} experiments; "
              f"leaving {outroot / 'summary.md'} alone "
              f"(rebuild it with --summary-only)")
        return 0
    write_summary(cfg, payloads, outroot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
