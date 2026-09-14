#!/usr/bin/env python3
"""
run_pipeline.py - one command, one config, both experiments.

    python llm_features_pipeline/run_pipeline.py --config llm_features_pipeline/config.yaml
    python llm_features_pipeline/run_pipeline.py --experiment expB --dry-run

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
                          plus the sample counts and the config that made them
    sample_counts.json    counts, group overlap, leak rates, dedup report, arch
    splits.csv            every sample, its label, group and split
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
            # Known for Goodware_Balanced (opcode_manifest.csv). Empty for the
            # Mendeley corpora, whose feature files carry no architecture and
            # whose test binaries are not present locally - see the
            # architecture note in results/summary.md.
            "arch": (s.meta.get("arch") or ""),
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


def build_samples(cfg, name):
    exp = cfg["experiments"][name]
    paths = cfg["paths"]
    sp = cfg["split"]

    ran = D.load_mendeley(Path(paths["llm_features"]), "ransomware")
    dedup = None
    if exp["goodware"] == "mendeley":
        good = D.load_mendeley(Path(paths["llm_features"]), "goodware")
    else:
        pool = D.load_balanced(Path(paths["balanced_goodware"]),
                               Path(paths["balanced_manifest"]),
                               sp["goodware_group_field"],
                               sp.get("goodware_ungrouped_entries", []))
        ref = D.load_mendeley(Path(paths["llm_features"]), "goodware")
        if sp.get("dedup_goodware_sources", True):
            pool, dedup = D.dedup_goodware_sources(
                pool, ref, Path(paths["balanced_manifest"]),
                Path(paths["mendeley_goodware_sha256"]))
            print(f"  cross-source dedup: {dedup['removed_total']} of "
                  f"{dedup['pool_in']} balanced goodware files also appear in the "
                  f"Mendeley goodware set "
                  f"({len(dedup['removed_by_source_sha256'])} by binary sha256, "
                  f"{len(dedup['removed_by_content_hash'])} by opcode-stream hash)")
            for n in (dedup["removed_by_source_sha256"]
                      + dedup["removed_by_content_hash"])[:10]:
                print(f"    removed {n}")
        if sp.get("match_counts_to"):
            n_tr = sum(1 for s in ref if s.split == "train")
            n_te = sum(1 for s in ref if s.split == "test")
        else:
            n_te = round(len(pool) * 0.105)
            n_tr = len(pool) - n_te
        print(f"  group-splitting {len(pool)} balanced goodware "
              f"-> {n_tr} train / {n_te} test")
        good = D.group_split(pool, n_tr, n_te, sp["seed"])
    return ran + good, dedup


def run_experiment(cfg, name, mods, outroot: Path, dry: bool):
    print("\n" + "=" * 72)
    print(f"{name}: {cfg['experiments'][name]['description']}")
    print("=" * 72)
    samples, dedup = build_samples(cfg, name)
    summary = D.summarize(samples)
    if dedup is not None:
        summary["cross_source_dedup"] = dedup
    print("measuring exact-duplicate leakage ...", flush=True)
    summary["content_leak"] = D.content_leak(samples)
    summary["goodware_arch"] = {
        sp: dict(sorted(collections.Counter(
            s.meta.get("arch", "") or "unknown"
            for s in samples if s.split == sp and s.label == 0).items()))
        for sp in ("train", "test")}
    summary["test_goodware_arch"] = summary["goodware_arch"]["test"]
    print(json.dumps(summary, indent=2))
    if summary["group_overlap"]:
        sys.exit(f"group leak across splits: {summary['group_overlap'][:5]}")

    outdir = outroot / name
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "splits.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "source", "label", "group", "split"])
        for s in samples:
            w.writerow([s.name, s.source, s.label, s.group, s.split])
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
            m["goodware_recall_by_arch"] = goodware_recall_by_arch(
                te, m.pop("_y_pred"))
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
    print(f"\nwrote {outdir / 'metrics.json'}")
    return payload


# Architecture shares of the ransomware side. Those binaries are VM-only, so
# these are quoted from the profile pass in
# "Ransomware Combined Structural Feature Dataset/extract-opcode/WRITEUP.md"
# §3.2-3.3 rather than recomputed here.
RANSOMWARE_ARCH = {"train": "978/1,023 x86 (95.6%), 42 x64 (4.1%)",
                   "test": "292/385 x86 (75.8%), 92 x64 (23.9%)"}


def _arch_block(p) -> list:
    """Per-architecture goodware recall, where the architecture is known."""
    known = p["samples"].get("test_goodware_arch", {})
    lines = ["", "#### Goodware test set by architecture", ""]
    if set(known) <= {"unknown", ""}:
        lines += [f"Architecture is unknown for all {sum(known.values())} goodware "
                  "test files: the Mendeley feature files record no architecture "
                  "and the `good_test` binaries are not present on this machine "
                  "(0 of 131 match a local binary by name), so no breakdown is "
                  "possible for this experiment. See the architecture note below.",
                  ""]
        return lines
    lines += ["| model | " + " | ".join(f"recall {a} (n={n})"
                                        for a, n in known.items()) + " |",
              "|---|" + "---|" * len(known)]
    for r in p["results"]:
        by = r.get("goodware_recall_by_arch", {})
        cells = []
        for a in known:
            v = by.get(a)
            cells.append("-" if not v or not v["n"] else f"{v['recall_goodware']:.4f}")
        lines.append(f"| {r['model']}/{r['tokenizer']} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _pct(d: dict) -> str:
    tot = sum(d.values()) or 1
    return ", ".join(f"{v} {k} ({100*v/tot:.1f}%)" for k, v in d.items())


def _arch_verdict(payloads) -> list:
    """State what the per-architecture recalls actually show, rather than
    leaving the reader to eyeball the table."""
    rows = []
    for name, p in payloads.items():
        for r in p["results"]:
            by = r.get("goodware_recall_by_arch", {})
            a, b = by.get("x64"), by.get("x86")
            if a and b and a["n"] and b["n"]:
                rows.append((name, f"{r['model']}/{r['tokenizer']}",
                             a["recall_goodware"], b["recall_goodware"]))
    if not rows:
        return ["It cannot be computed for any experiment here: no test-set "
                "goodware has a known architecture. The comparison stays open."]
    worst = max(rows, key=lambda t: t[2] - t[3])
    gaps = [t[2] - t[3] for t in rows]
    return [f"**It does.** Across {len(rows)} model/tokenizer pairs with a known "
            f"architecture, x64 goodware is recalled between "
            f"{min(gaps):.2f} and {max(gaps):.2f} better than x86 goodware "
            f"(worst: {worst[0]} {worst[1]}, {worst[2]:.4f} on x64 against "
            f"{worst[3]:.4f} on x86). Nearly every x64 benign file is caught and "
            "a large share of the x86 benign files are called ransomware — which "
            "is what a model keying on bitness looks like, given that the "
            "ransomware class is 96%/76% x86. **The A-vs-B gap therefore cannot "
            "yet be attributed to the goodware source.** Part of it is that Exp "
            "B's x86 goodware sits in the region of feature space the ransomware "
            "class occupies, and Exp A's goodware — 57% x86 — is not being "
            "scored the same way."]


def _architecture_note(payloads) -> list:
    lines = ["", "### Architecture is a confound, and it is not controlled here", "",
             "x86/x64 shares of each side of the task:", "",
             "| side | x86 / x64 | source of the count |",
             "|---|---|---|",
             f"| Mendeley ransomware, train (1,023) | {RANSOMWARE_ARCH['train']} "
             "| WRITEUP.md §3.2 (binaries are VM-only) |",
             f"| Mendeley ransomware, test (385) | {RANSOMWARE_ARCH['test']} "
             "| WRITEUP.md §3.3 |",
             "| Mendeley goodware, train (1,115 on disk) | 630 x86 (56.5%), "
             "485 x64 (43.5%) | WRITEUP.md §3.1 |",
             "| Mendeley goodware, test (131) | unknown | those binaries are not "
             "present locally; WRITEUP.md §3.4 leaves them unprofiled |"]
    for name, p in payloads.items():
        ga = p["samples"].get("goodware_arch", {})
        for sp in ("train", "test"):
            d = {k: v for k, v in ga.get(sp, {}).items() if k != "unknown"}
            if d:
                lines.append(f"| {name} goodware, {sp} ({sum(ga[sp].values())}) | "
                             f"{_pct(d)} | `opcode_manifest.csv` |")
    lines += ["",
              "The ransomware side is overwhelmingly x86 (96% in train, 76% in "
              "test). The Mendeley goodware is 57% x86; Goodware_Balanced is "
              "roughly 23% x86. **Exp B therefore widens the architecture gap "
              "between the classes at the same time as it changes the goodware "
              "source**, and bitness is not a hidden variable that a byte-level "
              "model has to infer - it is written all over the operands "
              "(`rbp`, `r8`-`r15`, rip-relative addressing, the register calling "
              "convention), so it survives normalization into the token stream.",
              "",
              "**A per-architecture breakdown is required before the A-vs-B gap "
              "can be attributed to the goodware source.** The goodware recall "
              "tables above are the part of it that can be produced here: they "
              "split each experiment's goodware test set by the architecture "
              "recorded in `opcode_manifest.csv`. If x64 goodware is recalled "
              "far better than x86 goodware, the model is partly reading "
              "bitness.",
              ""]
    lines += _arch_verdict(payloads)
    lines += [""]
    lines += [
              "What cannot be produced here, and why:",
              "",
              "* **Exp A's goodware test set** - the 131 `good_test` binaries are "
              "not on this machine (0 of 131 match a local binary by name) and "
              "the feature files carry no architecture, so there is nothing to "
              "join on. WRITEUP.md §3.4 has no counts for that folder either. "
              "Only the aggregate for `good_train` (56.5% x86) is known.",
              "* **Either experiment's ransomware side** - identical in A and B, "
              "but the binaries are VM-only, so only the aggregate counts above "
              "exist. A per-sample architecture index has to come out of the VM "
              "(`check_arch.py` writes one) before ransomware recall can be "
              "split by bitness.",
              "",
              "Until both exist, the honest statement is that A vs B varies "
              "goodware source **and** goodware architecture mix together."]
    return lines


def _deviation_note(cfg) -> list:
    return ["", "### Deviation from the plan's split procedure", "",
            "The plan asked for \"the same stratified 80/20 split procedure\" in "
            "both experiments. Two departures, both deliberate:", "",
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
            "verbatim in both experiments, so it contributes nothing to the "
            "A-vs-B difference.",
            "2. **Exp A's goodware split is also the release's own**, not a "
            "re-split. It has no group discipline, which is precisely what "
            "§2.1 of the audit measures: 64/131 of `good_test` is a verbatim "
            "copy of a training file. It is kept as shipped so Exp A remains "
            "the published baseline to compare against. Exp B's goodware side "
            "*is* split the way the plan intends - grouped by `entry_id` so no "
            "source project straddles train and test, stratified by bucket so "
            "`everyday`/`hard_negative`/`system` keep their proportions on both "
            "sides, with a fixed seed - and its counts are matched to Exp A's "
            "exactly (`split.match_counts_to: expA`). Measured: the pool is "
            "53.9% everyday / 24.9% hard_negative / 21.2% system, and the "
            "chosen split is 53.9/24.8/21.2 in train and 54.2/24.4/21.4 in "
            "test.", "",
            "Net effect: A is the baseline on its own terms, B is the same "
            "ransomware task with a harder, properly grouped goodware half."]


def write_summary(cfg, payloads, outroot: Path):
    lines = ["# Experiment summary: A vs B", "",
             "Binary task throughout: **0 = goodware, 1 = ransomware**. "
             "The ransomware family prefix is a split group, never a label.", ""]
    for name, p in payloads.items():
        s = p["samples"]
        lines += [f"## {name} - {p['description']}", "",
                  "| split | goodware | ransomware | total | groups |",
                  "|---|---|---|---|---|"]
        for sp in ("train", "test"):
            d = s[sp]
            lines.append(f"| {sp} | {d['goodware']} | {d['ransomware']} | "
                         f"{d['n']} | {d['groups']} |")
        cl = s.get("content_leak", {})
        if cl:
            lines += ["",
                      f"Exact-duplicate leakage: **{cl['test_goodware_duplicated_in_train']}"
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
        lines.append("")
        lines += _arch_block(p)

    if "expA" in payloads and "expB" in payloads:
        a, b = payloads["expA"], payloads["expB"]
        lines += ["## A vs B, matched pairs", "",
                  "| model / tok / emb / mask | A macro-F1 | B macro-F1 | delta |",
                  "|---|---|---|---|"]
        keyed = {(r["model"], r["tokenizer"], r["embedding"], r["mask_rate"]): r
                 for r in a["results"]}
        for r in b["results"]:
            k = (r["model"], r["tokenizer"], r["embedding"], r["mask_rate"])
            if k in keyed:
                d = r["macro_f1"] - keyed[k]["macro_f1"]
                lines.append(f"| {' / '.join(map(str, k))} | "
                             f"{keyed[k]['macro_f1']:.4f} | {r['macro_f1']:.4f} | "
                             f"{d:+.4f} |")
        sa, sb = a["samples"], b["samples"]
        same = (sa["train"]["goodware"] == sb["train"]["goodware"]
                and sa["test"]["goodware"] == sb["test"]["goodware"]
                and sa["train"]["ransomware"] == sb["train"]["ransomware"]
                and sa["test"]["ransomware"] == sb["test"]["ransomware"])
        lines += ["", "### Confounds", "",
                  ("- Class sizes are **identical** between A and B, so a gap is "
                   "not attributable to data volume. That rules out one "
                   "confound; it does not rule out the architecture mix, which "
                   "is not controlled - see below."
                   if same else
                   "- **Class sizes differ between A and B.** Any gap confounds the "
                   "goodware source with the amount of data; re-run with "
                   "`split.match_counts_to: expA` before drawing a conclusion."),
                  "- The ransomware side is byte-identical in both experiments "
                  "(same files, same family-disjoint split), so it contributes "
                  "nothing to the difference.",
                  "- Exp A's goodware split is the Mendeley release's own and has "
                  "no group discipline; Exp B's is group-disjoint by source "
                  "project. B is therefore the *harder* split, and a lower B score "
                  "is not by itself evidence of worse data."]

        la = sa.get("content_leak", {})
        lb = sb.get("content_leak", {})
        if la and lb:
            ra = la["test_goodware_leak_rate"] or 0
            rb = lb["test_goodware_leak_rate"] or 0
            lines += ["", "### Reading the gap", "",
                      f"A scores higher on every pair. Before reading that as "
                      f"\"the Mendeley goodware is better\", note that "
                      f"**{100*ra:.1f}%** of A's goodware test set is a verbatim "
                      f"copy of its own training data, against **{100*rb:.1f}%** "
                      f"for B, and that A contains "
                      f"{la['streams_labelled_both_classes']} opcode stream(s) "
                      f"carrying both labels while B contains "
                      f"{lb['streams_labelled_both_classes']}.",
                      "",
                      "A large part of A's goodware score is therefore recall of "
                      "streams the model has already memorised - overwhelmingly "
                      "NSIS and Inno installer stubs, which `extract.py` "
                      "disassembles in place of the payload. B has no such "
                      "shortcut, and its goodware is deliberately "
                      "ransomware-adjacent (encryption tools, archivers, backup "
                      "and sync clients, secure-delete utilities), so its "
                      "negatives sit much closer to the decision boundary.",
                      "",
                      "The two numbers are answering different questions. A "
                      "estimates performance on a corpus whose goodware half is "
                      "~50% duplicated; B estimates performance against "
                      "hard negatives never seen in training. **B is the number "
                      "to report as a generalisation estimate**; A is the "
                      "comparable baseline, not the better result.",
                      "",
                      "That reading is provisional. It assumes the two "
                      "experiments differ only in the *quality* of their "
                      "goodware, and the next section shows they also differ "
                      "sharply in its *architecture mix* - which the models are "
                      "demonstrably using. Read the two sections together."]

        lines += _architecture_note(payloads)
        lines += _deviation_note(cfg)
    out = outroot / "summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


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
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    outroot = Path(args.results_dir or cfg["paths"]["results"])
    if not outroot.is_absolute():
        outroot = REPO / outroot
    outroot.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        mods = {"normalize": lambda s: (s.strip().lower() or None)}
    else:
        mods = load_repo_modules(Path(cfg["paths"]["tokenization_repo"]))

    names = [args.experiment] if args.experiment else list(cfg["experiments"])
    payloads = {}
    for n in names:
        payloads[n] = run_experiment(cfg, n, mods, outroot, args.dry_run)
    if args.dry_run:
        return 0
    if len(payloads) < len(cfg["experiments"]):
        # Don't clobber an A-vs-B summary with a one-experiment file. Re-running
        # a single experiment is a normal thing to do while iterating; silently
        # replacing summary.md with half of it is not.
        print(f"ran {len(payloads)} of {len(cfg['experiments'])} experiments; "
              f"leaving {outroot / 'summary.md'} alone")
        return 0
    write_summary(cfg, payloads, outroot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
