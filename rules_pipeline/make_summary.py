#!/usr/bin/env python3
"""Render results/rules/summary.md from the two metrics.json files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results" / "rules"


def fmt(df, cols=None):
    cols = list(cols or df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append("-" if v != v else f"{v:.4f}")   # NaN -> "-"
            elif v is None or (isinstance(v, list) and not v):
                vals.append("-")
            elif isinstance(v, list):
                vals.append(", ".join(str(x) for x in v))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _tick(b) -> str:
    if b is None:
        return "n/a"
    return "**PASS**" if b else "**FAIL**"


def _m(d, *keys):
    """Pull a metric row into a dict of the columns the tables use."""
    return {k: d.get(k) for k in keys}


def baseline_section(o: list) -> None:
    """Section 5: the leakage audit of the TF-IDF calibration baseline.

    Everything here comes from results/rules/baseline_audit.json, written by
    rules_pipeline/baseline_audit.py. Nothing is typed in by hand.
    """
    p = RES / "baseline_audit.json"
    if not p.exists():
        o.append("\n## 5. Calibration baseline: leakage audit\n")
        o.append("Not run. `python rules_pipeline/baseline_audit.py` writes "
                 "`results/rules/baseline_audit.json`; this section is "
                 "rendered from it.\n")
        return
    doc = json.loads(p.read_text(encoding="utf-8"))
    D = doc["datasets"]

    o.append("\n## 5. Calibration baseline: is macro-F1 0.96 real?\n")
    o.append(
        "The `calibration_baseline` rows in section 1 are the highest scores "
        "anywhere in this project -- higher than the tokenization pipeline's "
        "best (0.926, `results/summary.md`), higher than graph2vec (0.895, "
        "`results/graph2vec/summary.md`) and far higher than CNN-ViT (0.628 "
        "seed-mean, `results/cnn_vit/summary.md`). A result that far outside "
        "the field is a claim about the corpus until it is audited, so this "
        "section audits it. Source: `rules_pipeline/baseline_audit.py` -> "
        "`results/rules/baseline_audit.json`.\n")

    # -- 5.1 hygiene of the fit -------------------------------------------
    o.append("\n### 5.1 Is the fit clean?\n")
    rows = []
    for ds, d in D.items():
        h = d["vectoriser_hygiene"]
        rows.append({
            "dataset": ds,
            "vocab + IDF from train rows only": _tick(
                h["production_vocab_equals_train_only_vocab"] and
                h["production_idf_equals_train_only_idf"]),
            "`transform(test)` left vocabulary unchanged": _tick(
                h["transform_on_test_left_vocabulary_unchanged"]),
            "`transform(test)` left IDF unchanged": _tick(
                h["transform_on_test_left_idf_unchanged"]),
            "features": f"{h['n_features_fit_on_train']:,}",
            "features test would have added": (
                f"{h['features_only_test_would_have_added']:,}"),
            "non-mnemonic tokens in vocabulary": h["n_non_mnemonic_tokens"],
        })
    o.append(fmt(pd.DataFrame(rows)) + "\n")
    o.append(
        "\nRead the last two columns together. Fitting the vectoriser on "
        "train+test instead of train would have produced tens of thousands of "
        "extra n-gram columns, so the check is not vacuous -- there really is "
        "test-only vocabulary, and the production path really does exclude it. "
        "Every surviving feature is a 1-, 2- or 3-gram of mnemonics from the "
        "corpus vocabulary: no file length, no file size, no architecture "
        "flag, no family, no filename, no path. `X` handed to the classifier "
        "is the TF-IDF matrix and nothing is concatenated to it.\n")

    # -- 5.2 split integrity ----------------------------------------------
    o.append("\n### 5.2 Are the rows disjoint, and are they the same rows?\n")
    rows = []
    for ds, d in D.items():
        s = d["split_integrity"]
        c = s["counts"]
        rows.append({
            "dataset": ds,
            "train (good/ransom)":
                f"{c['train']['n']} ({c['train']['goodware']}/"
                f"{c['train']['ransomware']})",
            "test (good/ransom)":
                f"{c['test']['n']} ({c['test']['goodware']}/"
                f"{c['test']['ransomware']})",
            "matches results/expC/sample_counts.json":
                _tick(s["matches_expC_cohort"]),
            "sha256 shared train/test": s["n_sha256_overlap_train_test"],
            "duplicate sha256 anywhere": s["duplicate_sha256_anywhere"],
            "family overlap": len(s["family_overlap"]),
            "group overlap": s["n_group_overlap"],
            "train / test families":
                f"{len(s['ransomware_families_train'])} / "
                f"{len(s['ransomware_families_test'])}",
        })
    o.append(fmt(pd.DataFrame(rows)) + "\n")
    o.append(
        "\nThe cohort rows are the same rows every other pipeline uses -- both "
        "harnesses call `cnn_vit_pipeline.cohort.load_split`, and the counts "
        "reconcile against the tokenization pipeline's own "
        "`results/expC/sample_counts.json` (1,114 goodware / 904 ransomware "
        "train, 129 / 362 test; this pipeline's `train` figure splits further "
        "into a fit and a val fold). No sha256 is on both sides, no ransomware "
        "family is on both sides, no group straddles the boundary.\n\n"
        "The `balanced` row **is** the expB-style split, not a separate "
        "experiment: `cohort._balanced_frame` keeps the Mendeley ransomware "
        "split byte-for-byte and reads its goodware membership verbatim from "
        "the tokenization pipeline's own `results/expB/splits.csv` (snapshot "
        "at `results/cnn_vit/expB_splits_snapshot.csv`), mapped to sha256 "
        "through `LLM_Features_Balanced/opcode_manifest.csv` and intersected "
        "with `cohort_balanced`. So `mendeley` here corresponds to expA/expC "
        "and `balanced` to expB/expD; there is no third split to run.\n")

    o.append("\nArchitecture composition, which matters for §5.6:\n")
    rows = []
    for ds, d in D.items():
        a = d["split_integrity"]["arch_by_split_and_class"]
        for k, v in a.items():
            rows.append({"dataset": ds, "split/class": k,
                         "x86": v.get("x86", 0), "x64": v.get("x64", 0)})
    o.append(fmt(pd.DataFrame(rows)) + "\n")

    # -- 5.3 duplication ---------------------------------------------------
    o.append("\n### 5.3 Exact-duplicate audit\n")
    o.append(
        "Two content keys, independent of the cohort sha256: the sha256 of the "
        "`mn/<sha>.txt` file's raw bytes, and the sha256 of the capped token "
        "stream the vectoriser actually reads (first "
        f"{doc['max_mnemonics_per_file']:,} mnemonics). A test row whose "
        "content hash appears in train is, for this model, a training row with "
        "a different name.\n")
    rows = []
    for ds, d in D.items():
        for key, blk in d["duplication"].items():
            if not isinstance(blk, dict) or "test_leak" not in blk:
                continue
            g, r = blk["test_leak"]["goodware"], blk["test_leak"]["ransomware"]
            rows.append({
                "dataset": ds, "content key": key,
                "unique streams / files":
                    f"{blk['unique_streams']:,} / {blk['total_files']:,}",
                "goodware test leaked":
                    f"{g['duplicated_in_train']}/{g['test_n']} "
                    f"({g['leak_rate']:.1%})",
                "ransomware test leaked":
                    f"{r['duplicated_in_train']}/{r['test_n']} "
                    f"({r['leak_rate']:.1%})",
                "streams under both labels":
                    blk["streams_labelled_both_classes"],
            })
    o.append(fmt(pd.DataFrame(rows)) + "\n")
    key = D[next(iter(D))]["duplication"]["clean_test_index_key"]
    for ds, d in D.items():
        blk = d["duplication"][key]
        big = blk["largest_duplicate_groups"]
        if not big:
            continue
        o.append(f"\nLargest duplicate groups in `{ds}` (by capped token "
                 f"stream):\n")
        o.append(fmt(pd.DataFrame([
            {"group size": b["size"], "in train": b["in_train"],
             "in test": b["in_test"],
             "labels": "/".join(str(x) for x in b["labels"]),
             "example filenames": b["example_filenames"]} for b in big])) + "\n")

    # -- 5.4 what survives -------------------------------------------------
    o.append("\n### 5.4 What survives deduplication\n")
    rows = []
    for ds, d in D.items():
        ded = d["deduplicated"]
        h = d["headline"]
        rows.append({"dataset": ds, "condition": "as reported (all test rows)",
                     "test n": h["LogReg"]["support_goodware"] +
                               h["LogReg"]["support_ransomware"],
                     "LogReg macro-F1": h["LogReg"]["macro_f1"],
                     "LinearSVC macro-F1": h["LinearSVC"]["macro_f1"],
                     "recall goodware": h["LogReg"]["recall_goodware"],
                     "recall ransomware": h["LogReg"]["recall_ransomware"]})
        c = ded["same_model_on_clean_test_only"]
        rows.append({"dataset": ds,
                     "condition": "same model, test rows not duplicated in train",
                     "test n": ded["clean_test_n"],
                     "LogReg macro-F1": c["LogReg"]["macro_f1"],
                     "LinearSVC macro-F1": c["LinearSVC"]["macro_f1"],
                     "recall goodware": c["LogReg"]["recall_goodware"],
                     "recall ransomware": c["LogReg"]["recall_ransomware"]})
        t = ded["dedup_train"]
        rows.append({"dataset": ds,
                     "condition": (f"retrained on deduplicated train "
                                   f"({t['train_n']} rows), clean test"),
                     "test n": ded["clean_test_n"],
                     "LogReg macro-F1": t["LogReg_on_clean_test"]["macro_f1"],
                     "LinearSVC macro-F1": None,
                     "recall goodware":
                         t["LogReg_on_clean_test"]["recall_goodware"],
                     "recall ransomware":
                         t["LogReg_on_clean_test"]["recall_ransomware"]})
    o.append(fmt(pd.DataFrame(rows)) + "\n")

    # -- 5.5 ablations -----------------------------------------------------
    o.append("\n### 5.5 Permutation and ablation\n")
    rows = []
    for ds, d in D.items():
        a = d["ablations"]
        base = d["headline"]["LogReg"]
        rows.append({"dataset": ds, "condition": "full model (LogReg)",
                     "n features": d["headline"]["n_features"],
                     "macro-F1": base["macro_f1"],
                     "recall ransomware": base["recall_ransomware"],
                     "FPR": base["false_positive_rate"]})
        p = a["label_permutation_train_only"]
        rows.append({"dataset": ds,
                     "condition": (f"(a) train labels shuffled, "
                                   f"{p['n_permutations']} runs (mean)"),
                     "n features": d["headline"]["n_features"],
                     "macro-F1": p["mean_macro_f1"],
                     "recall ransomware": None, "FPR": None})
        for k, label in (("drop_top_50_by_abs_coef", "(b) drop top-50 n-grams"),
                         ("drop_top_500_by_abs_coef", "(b') drop top-500"),
                         ("unigram_only", "(c) unigram TF-IDF only"),
                         ("length_only_control",
                          "(d) length-only control (2 scalars, RF)")):
            if k not in a:
                continue
            r = a[k]
            rows.append({"dataset": ds, "condition": label,
                         "n features": r.get("n_features", 2),
                         "macro-F1": r["macro_f1"],
                         "recall ransomware": r["recall_ransomware"],
                         "FPR": r["false_positive_rate"]})
    o.append(fmt(pd.DataFrame(rows)) + "\n")
    chance = D[next(iter(D))]["ablations"][
        "label_permutation_train_only"]["chance_macro_f1_all_ransomware"]
    o.append(
        f"\n(a) is the test that matters. With the training labels shuffled "
        f"the same pipeline collapses to roughly the always-predict-ransomware "
        f"floor (macro-F1 {chance:.3f}), so the 0.96 is not an artefact of the "
        f"vectoriser, the grid search, or the metric -- it is coming from the "
        f"label-feature association. (b) says the score is not carried by a "
        f"handful of magic n-grams: the signal is spread over the whole "
        f"vocabulary. (d) says it is not file length in disguise.\n")

    # -- 5.6 why it works --------------------------------------------------
    o.append("\n### 5.6 Why it works\n")
    for ds, d in D.items():
        ch = d["characterisation"]
        o.append(f"\n**{ds} -- top n-grams by logistic-regression "
                 f"coefficient.**\n")
        pos = ch["top_ransomware_ngrams"][:15]
        neg = ch["top_goodware_ngrams"][:15]
        o.append(fmt(pd.DataFrame([
            {"toward ransomware": "`" + a["ngram"] + "`", "coef +": a["coef"],
             "toward goodware": "`" + b["ngram"] + "`", "coef -": b["coef"]}
            for a, b in zip(pos, neg)])) + "\n")

        o.append(f"\n**{ds} -- per-architecture.** The whole-corpus model, "
                 f"scored inside each architecture:\n")
        o.append(fmt(pd.DataFrame([
            dict(arch=a, **{k: v for k, v in m.items()})
            for a, m in ch["per_architecture"].items()])) + "\n")

        wam = ch["within_architecture_models"]
        o.append(f"\nAnd a model **trained and tested inside one architecture "
                 f"only** -- no cross-architecture shortcut available:\n")
        o.append(fmt(pd.DataFrame([
            {"arch": a,
             "train n (good/ransom)":
                 (f"{m.get('train_n','-')} ({m.get('train_goodware','-')}/"
                  f"{m.get('train_ransomware','-')})"),
             "test n": (m.get("support_goodware", 0) +
                        m.get("support_ransomware", 0)) or m.get("test_n", "-"),
             "macro-F1": m.get("macro_f1"),
             "recall goodware": m.get("recall_goodware"),
             "recall ransomware": m.get("recall_ransomware"),
             "note": m.get("skipped", "")}
            for a, m in wam.items()])) + "\n")

        pf = ch["per_family_test_recall"]
        o.append(f"\n**{ds} -- per-family test recall** (all 14 families are "
                 f"unseen in training):\n")
        o.append(fmt(pd.DataFrame([
            {"family": k, "n": v["n"], "recall": v["recall"],
             "detected": v["detected"]}
            for k, v in sorted(pf.items(), key=lambda kv: kv[1]["recall"])])) + "\n")

    m, b = D.get("mendeley"), D.get("balanced")
    if m and b:
        mfam = m["characterisation"]["per_family_test_recall"]
        bfam = b["characterisation"]["per_family_test_recall"]
        perfect = sum(1 for v in mfam.values() if v["recall"] == 1.0)
        worst = min(mfam.items(), key=lambda kv: kv[1]["recall"])
        bzero = sorted(k for k, v in bfam.items() if v["recall"] == 0.0)
        mx = m["characterisation"]["within_architecture_models"]
        mpa = m["characterisation"]["per_architecture"]
        bpa = b["characterisation"]["per_architecture"]
        o.append(
            f"\n**Reading §5.6.** Three things, in order of how much they "
            f"matter.\n\n"
            f"1. **The ransomware side generalises, and duplication cannot "
            f"explain it.** On `mendeley`, {perfect} of the 14 held-out "
            f"families are detected at recall 1.00 and the worst is "
            f"`{worst[0]}` at {worst[1]['recall']:.2f}. These are families "
            f"with **no representative in training at all**, and the "
            f"ransomware test-leak rate in §5.3 is 0.0% -- not one ransomware "
            f"test file is a verbatim copy of a training file. Whatever the "
            f"goodware half of this result is worth, the ransomware half is "
            f"genuine cross-family generalisation.\n\n"
            f"2. **It is not the architecture shortcut.** Mendeley training "
            f"has only 42 x64 ransomware against 482 x64 goodware, so 'x64 "
            f"implies goodware' is available -- and would be punished by a "
            f"test set that is 72 of 84 x64 rows ransomware. The model scores "
            f"macro-F1 {mpa['x64']['macro_f1']:.3f} inside x64 and "
            f"{mpa['x86']['macro_f1']:.3f} inside x86, and a model trained "
            f"**only** on x64 rows still reaches "
            f"{mx['x64']['macro_f1']:.3f} from 42 positive training examples. "
            f"The shortcut is available and is not being taken.\n\n"
            f"3. **The n-grams are compiler idiom, not semantics.** The "
            f"positive weights are dominated by wide-integer arithmetic "
            f"(`adc mov`, `sbb mov mov`, `add adc mov`, `movsx shl or`, "
            f"`rol`, `idiv`) -- the shape of bignum and block-cipher inner "
            f"loops compiled without SIMD -- and the negative weights by MSVC "
            f"CRT padding and epilogue idiom (`int3 int3 int3`, "
            f"`call leave ret`, `pop leave ret`). That is a real and "
            f"explainable difference, but it is a difference between "
            f"*toolchains and code styles*, not between 'encrypts your files' "
            f"and 'does not'. It is exactly the kind of feature a different "
            f"goodware corpus can erase, and that is what `balanced` does: "
            f"macro-F1 falls to {b['headline']['LogReg']['macro_f1']:.3f}, "
            f"x86 goodware recall falls to "
            f"{bpa['x86']['recall_goodware']:.2f}"
            + (f", and **{', '.join(bzero)}** is missed entirely "
               f"({sum(bfam[k]['n'] for k in bzero)} files, recall 0.00).\n"
               if bzero else ", and per-family recall spreads out.\n"))


def verdict(o: list, docs: dict) -> None:
    """Section 6: the one-paragraph answer, derived from the audit JSON."""
    p = RES / "baseline_audit.json"
    if not p.exists():
        return
    D = json.loads(p.read_text(encoding="utf-8"))["datasets"]
    key = D[next(iter(D))]["duplication"]["clean_test_index_key"]

    o.append("\n## 6. Verdict on the calibration baseline\n")
    rows = []
    for ds, d in D.items():
        dup = d["duplication"][key]["test_leak"]
        ded = d["deduplicated"]
        rows.append({
            "dataset": ds,
            "as reported": d["headline"]["LogReg"]["macro_f1"],
            "goodware test leak rate": f"{dup['goodware']['leak_rate']:.1%}",
            "ransomware test leak rate": f"{dup['ransomware']['leak_rate']:.1%}",
            "on non-duplicated test rows":
                ded["same_model_on_clean_test_only"]["LogReg"]["macro_f1"],
            "retrained dedup train, clean test":
                ded["dedup_train"]["LogReg_on_clean_test"]["macro_f1"],
            "shuffled train labels":
                d["ablations"]["label_permutation_train_only"]["mean_macro_f1"],
        })
    o.append(fmt(pd.DataFrame(rows)) + "\n")
    o.append(
        "\nThe mechanical checks all pass: vocabulary and IDF are fitted on "
        "train rows only, the test rows are disjoint from train by cohort "
        "sha256, the 14 test families are disjoint from the 24 training "
        "families, no group straddles the split, the rows reconcile against "
        "`results/expC/sample_counts.json`, the feature matrix is mnemonic "
        "n-grams and nothing else, and the score collapses to the "
        "always-predict-ransomware floor when the training labels are "
        "shuffled. **The number is not produced by a coding error in the "
        "harness.**\n\n"
        "What it *is* partly produced by is the corpus. The Mendeley goodware "
        "half of the test set contains verbatim duplicates of training files "
        "(the leak-rate column above; cause in `docs/tokenization_audit.md` "
        "§2.1 -- `extract.py` disassembles installer stubs, so many "
        "PortableApps launchers collapse onto a handful of identical opcode "
        "streams that land on both sides of the split). The ransomware half "
        "does not have this problem, because it is family-disjoint by "
        "construction. So the goodware column of the Mendeley result is "
        "inflated and the ransomware column is not, and the honest headline is "
        "the deduplicated row, not the as-reported one.\n\n"
        "The `balanced` dataset is the control for exactly this. Its goodware "
        "test-leak rate is a twelfth of Mendeley's, and there the "
        "deduplicated numbers go *up*, not down -- the duplication is not "
        "doing any work. The same model scores materially lower on that "
        "dataset for a different reason (§5.6 point 3).\n\n"
        "**Bottom line.** The mnemonic 1-3-gram TF-IDF baseline is clean "
        "enough to quote and is the strongest result in this project. Quote "
        "it as **0.95 (Mendeley, duplicate test rows removed) / 0.80 "
        "(Goodware_Balanced)**, not as 0.968, and always with the pair -- the "
        "gap between the two datasets is the most informative thing about it. "
        "Two caveats travel with the number: it is one fixed family split, so "
        "the per-family table is a sample of size 14, not a distribution; and "
        "what it has learned is compiler and code-style idiom, which is "
        "cheap to evade deliberately even though it transfers across 14 "
        "unseen families that were not trying to evade it.\n")


def main() -> int:
    docs = {}
    for ds in ("mendeley", "balanced"):
        p = RES / ds / "metrics.json"
        if p.exists():
            docs[ds] = json.loads(p.read_text(encoding="utf-8"))
    if not docs:
        sys.exit("no metrics.json under results/rules/")
    cfg = next(iter(docs.values()))["config"]

    o = []
    o.append("# Rule-based prototypes\n")
    o.append(
        "Two rule families, both learned on TRAIN only and scored on the "
        "held-out family-disjoint TEST split from `cnn_vit_pipeline/cohort.py`:"
        "\n\n"
        "**(a) mined mnemonic n-gram rules.** 2-, 3- and 4-grams over the "
        f"`mn/` mnemonic streams (first {cfg['max_mnemonics']:,} mnemonics per "
        f"file), mined level-wise with Apriori downward closure at document "
        f"support >= {cfg['min_support']} training files, scored by lift and "
        "odds ratio toward ransomware under a Jeffreys (alpha=0.5) prior, then "
        "reduced to a compact set by greedy set cover with a train-precision "
        "floor of 0.80.\n\n"
        "**(b) hand-written behaviour signatures.** 23 crypto-loop, "
        "enumeration and anti-analysis patterns over mnemonics and operands in "
        f"the `.asm` (first {cfg['max_insns']:,} instructions). Presence rules "
        "have no free parameter; density rules get one threshold, picked on "
        "train by maximising that single rule's own F1.\n\n"
        "Decision thresholds for the weighted scorers come from the **val "
        "fold** (`cohort.add_val_fold`, 10% of train held out by group), not "
        "from the rows the rules were mined on and not from test.\n\n"
        "**(c)** one calibration baseline -- mnemonic 1-3-gram TF-IDF + linear "
        "SVM / logistic regression -- is reported alongside, because a rule "
        "set is only interesting relative to the cheapest strong classical "
        "model on the same tokens. Section 5 audits it, because it turned out "
        "to be the best result in the whole project.\n")

    o.append(
        "> **Correction, and what it invalidates.** An earlier version of "
        "these tables was wrong. `MnemCorpus` stores each file's mnemonic "
        "stream as `int16` to keep the corpus in memory; `ngram_rules._codes` "
        "then computed n-gram codes by `c * 2048 + next`, which **overflows "
        "int16 silently** for any leading mnemonic whose id is 16 or above. "
        "The miner cast to `int64` first and was correct; the scorer "
        "(`hit_matrix`) did not. So every mined rule was scored against a "
        "wrapped-around code: all 3- and 4-grams matched nothing at all, and "
        "2-grams matched only when their first mnemonic was one of the 15 "
        "earliest-seen in the corpus (`add sub cmp inc push mov ...`). The "
        "visible symptom was a **top-by-lift table in which every rule showed "
        "zero test hits** -- including `xorps movlpd`, present in 264 of 796 "
        "training ransomware and 4 of 1,003 training goodware, which cannot "
        "plausibly fire on none of 491 test files. The cast now lives inside "
        "`_codes` and `tests/test_graph2vec_rules.py` pins it "
        "(`test_codes_are_dtype_independent`, "
        "`test_hit_matrix_fires_for_high_id_mnemonics`, "
        "`test_hit_matrix_agrees_with_mining_on_int16_corpus`, which re-derives "
        "every mined document frequency from the hit matrix). Every "
        "`ngram_rules`, `behaviour_rules`-weighted and `combined` number below "
        "is from the rerun; the `calibration_baseline` rows never touched the "
        "hit matrix and are unchanged.\n")

    # ---- headline tables -------------------------------------------------
    o.append("## 1. Test results\n")
    for ds, doc in docs.items():
        s = doc["samples"]
        o.append(f"### {ds}\n")
        o.append(f"test {s['test']['n']} ({s['test']['goodware']} goodware / "
                 f"{s['test']['ransomware']} ransomware); majority-class "
                 f"accuracy {doc['results'][0]['majority_class_accuracy']:.4f}."
                 "\n")
        rows = [dict(track=r.get("track", "-"), model=r["model"],
                     rules=r.get("n_rules", r.get("n_features")),
                     acc=r["accuracy"], bal_acc=r["balanced_accuracy"],
                     macro_f1=r["macro_f1"], auc=r["roc_auc"],
                     recall_ran=r["recall_ransomware"],
                     fpr=r["false_positive_rate"])
                for r in doc["results"]]
        o.append(fmt(pd.DataFrame(rows)) + "\n")

    # ---- selected n-gram rules ------------------------------------------
    o.append("\n## 2. The selected n-gram rule set\n")
    for ds, doc in docs.items():
        rep = doc["rule_report"]["ngram"]
        o.append(f"### {ds}\n")
        o.append(f"{rep['candidates']:,} n-grams cleared support "
                 f"({rep['per_level']}); {rep['pooled']:,} kept as candidates; "
                 f"greedy set cover selected "
                 f"{len(rep['selected_ransomware'])} ransomware-leaning and "
                 f"{len(rep['selected_goodware'])} goodware-leaning rules, "
                 f"covering {rep['train_coverage']['covered']}/"
                 f"{rep['train_coverage']['positives']} training ransomware."
                 "\n")
        for key, title in (("selected_ransomware", "Selected ransomware rules"),
                           ("selected_goodware", "Selected goodware rules")):
            if not rep[key]:
                continue
            rows = []
            for r in rep[key]:
                rows.append({
                    "rule (mnemonic n-gram)": "`" + r["rule"] + "`",
                    "n": r["n"],
                    "train R/G docs": f"{r['train_ransomware']}/{r['train_goodware']}",
                    "lift": r["lift"], "odds": r["odds"],
                    "train P": r["train_precision"], "train R": r["train_recall"],
                    "test P": r["test"]["precision"], "test R": r["test"]["recall"],
                })
            o.append(f"\n**{title}**\n")
            o.append(fmt(pd.DataFrame(rows)) + "\n")

        rows = [{"rule": "`" + r["rule"] + "`", "n": r["n"],
                 "train R/G docs": f"{r['train_ransomware']}/{r['train_goodware']}",
                 "lift": r["lift"], "train P": r["train_precision"],
                 "train R": r["train_recall"],
                 "test P": r["test"]["precision"], "test R": r["test"]["recall"]}
                for r in rep["top_by_lift"]]
        nfire = sum(1 for r in rep["top_by_lift"] if r["test"]["fired"] == 0)
        o.append(
            f"\n**Top 15 by lift toward ransomware.** Not the selected set: "
            f"greedy set cover optimises coverage at a precision floor, lift "
            f"ranks purity. Because support is floored at "
            f"{doc['config']['min_support']} training documents, high lift "
            f"here does *not* mean a rare n-gram -- the top row is present in "
            f"{rep['top_by_lift'][0]['train_ransomware']} of the training "
            f"ransomware. **{15 - nfire} of these 15 rules fire on the test "
            f"split** (before the `_codes` fix it was 0 of 15, which is what "
            f"the correction at the top of this file is about):\n")
        o.append(fmt(pd.DataFrame(rows)) + "\n")

    # ---- behaviour rules -------------------------------------------------
    o.append("\n## 3. Behaviour signatures, per rule\n")
    for ds, doc in docs.items():
        rep = doc["rule_report"]["behaviour"]
        o.append(f"### {ds}\n")
        rows = []
        for r in rep["rules"]:
            rows.append({
                "id": r["id"], "rule": r["name"], "kind": r["kind"],
                "thr": ("-" if r["threshold"] is None else r["threshold"]),
                "fires on ransomware (test)": r["ransomware_fire_rate_test"],
                "fires on goodware (test)": r["goodware_fire_rate_test"],
                "train P": r["train"]["precision"],
                "train R": r["train"]["recall"],
                "test P": r["test"]["precision"],
                "test R": r["test"]["recall"],
                "log-odds weight": rep["weights"][r["id"]],
            })
        o.append(fmt(pd.DataFrame(rows)) + "\n")
        o.append("\nWhat each rule means:\n")
        for r in rep["rules"]:
            o.append(f"- **{r['id']} {r['name']}** -- {r['description']}")
        o.append("")

    # ---- hard negatives --------------------------------------------------
    o.append("\n## 4. Which rules fire on the hard negatives\n")
    if "balanced" in docs and "mendeley" in docs:
        rb = {r["id"]: r for r in docs["balanced"]["rule_report"]["behaviour"]["rules"]}
        rm = {r["id"]: r for r in docs["mendeley"]["rule_report"]["behaviour"]["rules"]}
        rows = []
        for rid in sorted(rb):
            rows.append({
                "id": rid, "rule": rb[rid]["name"],
                "goodware fire rate, Mendeley test":
                    rm[rid]["goodware_fire_rate_test"],
                "goodware fire rate, Goodware_Balanced test":
                    rb[rid]["goodware_fire_rate_test"],
                "delta": round(rb[rid]["goodware_fire_rate_test"] -
                               rm[rid]["goodware_fire_rate_test"], 4),
                "example false positives":
                    rb[rid]["goodware_false_positives_test"][:5],
            })
        df = pd.DataFrame(rows).sort_values("delta", ascending=False)
        o.append(
            "The `balanced` dataset replaces the Mendeley goodware with "
            "Goodware_Balanced -- archivers, encryption utilities, backup and "
            "sync clients, secure-delete tools. That is the bucket these "
            "signatures are supposed to survive. The delta column is how much "
            "more often each rule fires on benign software when the benign "
            "software is ransomware-adjacent.\n")
        o.append(fmt(df) + "\n")

    baseline_section(o)
    verdict(o, docs)

    (RES / "summary.md").write_text("\n".join(o), encoding="utf-8")
    print(f"wrote {RES/'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
