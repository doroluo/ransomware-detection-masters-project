#!/usr/bin/env python3
"""Render results/graph2vec/summary.md from the two metrics.json files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results" / "graph2vec"


def _row(r):
    return dict(
        representation=r["representation"], model=r["model"],
        decision=("default" if r.get("decision", "").startswith("argmax")
                  else "calibrated"),
        acc=r["accuracy"], bal_acc=r["balanced_accuracy"],
        macro_f1=r["macro_f1"], auc=r["roc_auc"],
        recall_ran=r["recall_ransomware"], fpr=r["false_positive_rate"],
        n_feat=r.get("n_features"), best=r.get("best_params"))


def _default_rows(doc):
    return [r for r in doc["results"]
            if r.get("decision", "").startswith("argmax")]


def fmt(df, cols=None):
    cols = cols or df.columns
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [head, sep]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            elif v is None:
                vals.append("-")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _best(path: Path, pick=lambda r: True) -> str:
    """Best macro-F1 in another track's metrics.json, or '-' if absent."""
    if not path.exists():
        return "-"
    doc = json.loads(path.read_text(encoding="utf-8"))
    vals = [r["macro_f1"] for r in doc.get("results", []) if pick(r)]
    return f"{max(vals):.3f}" if vals else "-"


def _cross_track(docs) -> list:
    R = REPO / "results"
    rules = R / "rules"
    cnn = R / "cnn_vit"
    is_tfidf = lambda r: r.get("track") == "calibration_baseline"      # noqa: E731
    not_tfidf = lambda r: r.get("track") != "calibration_baseline"     # noqa: E731
    g = {ds: (f"{max(r['macro_f1'] for r in d['results']):.3f}"
              if d.get("results") else "-") for ds, d in docs.items()}
    gs = {ds: (f"{max(r['macro_f1'] for r in d['results'] if r['representation'] != 'size_only'):.3f}")
          for ds, d in docs.items()}
    return [
        ("mnemonic 1-3-gram TF-IDF + LogReg / LinearSVC",
         "**" + _best(rules / "mendeley" / "metrics.json", is_tfidf) + "**",
         _best(rules / "balanced" / "metrics.json", is_tfidf),
         "`results/rules/summary.md`"),
        ("tokenizer + word2vec + RF/SVM/MLP (expC / expD)",
         _best(R / "expC" / "metrics.json"), _best(R / "expD" / "metrics.json"),
         "`results/summary.md`"),
        ("mined n-gram + hand-written behaviour rules",
         _best(rules / "mendeley" / "metrics.json", not_tfidf),
         _best(rules / "balanced" / "metrics.json", not_tfidf),
         "`results/rules/summary.md`"),
        ("**graph2vec / WL on CFGs (this file), structural rows only**",
         "**" + gs.get("mendeley", "-") + "**", "**" + gs.get("balanced", "-") + "**",
         "here"),
        ("graph2vec track including the `size_only` control",
         g.get("mendeley", "-"), g.get("balanced", "-"), "here"),
        ("CNN-ViT on instruction images (single run; 5-seed mean 0.628 / 0.502)",
         _best(cnn / "mendeley" / "metrics.json"),
         _best(cnn / "balanced" / "metrics.json"),
         "`results/cnn_vit/summary.md`"),
    ]


def main() -> int:
    docs = {}
    for ds in ("mendeley", "balanced"):
        p = RES / ds / "metrics.json"
        if p.exists():
            docs[ds] = json.loads(p.read_text(encoding="utf-8"))
    if not docs:
        sys.exit("no metrics.json under results/graph2vec/")

    stats = pd.read_csv(RES / "graph_stats.csv")
    any_doc = next(iter(docs.values()))
    cfgc = any_doc["config"]

    out = []
    out.append("# graph2vec on linear-sweep CFGs\n")
    out.append(
        "Prototype. Control-flow graphs are cut out of the linear disassembly "
        "in `Shared/Extract*/asm`, embedded with Weisfeiler-Lehman subtree "
        "features, and classified with the same RF / SVM-RBF / MLP grid as "
        "`llm_features_pipeline`. Split, metrics schema and per-arch / "
        "per-family breakdowns come from `cnn_vit_pipeline/cohort.py`, so "
        "these numbers sit next to `results/summary.md` without adjustment.\n")

    out.append(
        "> **How to read the result tables.** There are **four** "
        "representations -- `wl_tfidf`, `wl_svd`, `graph2vec`, `size_only` -- "
        "and every (representation, model) pair occupies **two** rows. Those "
        "two rows are *not* two feature variants. They are the same fitted "
        "model on the same scores under two decision rules: `default` is "
        "argmax at 0.5, `calibrated` moves only the cut, chosen from TRAIN "
        "out-of-fold predictions. The `representation` column is the only "
        "place the feature set is named:\n\n"
        "| representation | what the classifier actually sees |\n"
        "|---|---|\n"
        "| `wl_tfidf` | the sparse WL subtree-label histogram with IDF "
        "weighting -- i.e. the WL kernel feature map, ~320k-340k columns. The "
        "**plain WL-histogram baseline**. |\n"
        "| `wl_svd` | that same histogram reduced by TruncatedSVD to 128 dense "
        "dimensions. A graph2vec-*style* dense embedding, but linear. |\n"
        "| `graph2vec` | PV-DBOW trained on the WL documents, 128 dims. This "
        "is **graph2vec proper** (Narayanan et al. 2017), re-implemented in "
        "numpy. |\n"
        "| `size_only` | ten scalar graph statistics and nothing else. The "
        "shortcut control. |\n")

    out.append("## 1. What was built\n")
    out.append(
        f"- **Blocks.** Leaders are the first instruction of each section, the "
        f"instruction after every branch/call/ret, every direct branch or call "
        f"target that resolves inside the same section, and the instruction "
        f"after an undecoded `.skip` gap. Edges are fall-through, resolved "
        f"branch targets and (optionally) call targets. No fall-through is "
        f"emitted across a `.skip` gap or a section boundary.\n"
        f"- **Node labels.** `<dominant mnemonic class>.<terminator kind>."
        f"<length bucket>` with `+C/+S/+V` flags when the block contains a "
        f"crypto, string or SIMD instruction. 19 mnemonic classes.\n"
        f"- **Caps.** {cfgc['max_insns']:,} instruction lines and "
        f"{cfgc['max_blocks']:,} blocks per file, whichever binds first. The "
        f"corpus is 7.2GB of disassembly and single files run to 3.16M "
        f"instructions; the tokenization baseline in `results/summary.md` "
        f"truncates at 5,000 *instructions*, so this window is about six times "
        f"wider than the baseline it is compared against.\n"
        f"- **Embedding.** WL iterations h=0..{cfgc['wl_iterations']}, "
        f"min_df={cfgc['min_df']} over TRAIN graphs, TF-IDF, then either "
        f"TruncatedSVD({cfgc['embedding_dim']}) (`wl_svd`) or PV-DBOW "
        f"(`graph2vec`). Everything is fitted on train rows only.\n")
    out.append(
        "**graph2vec implementation note.** karateclub is sdist-only and its "
        "`Graph2Vec` needs `gensim>=4`, which has no cp314 wheel (pip offers "
        "only the unrelated pure-python 0.10.x line) and no local C toolchain "
        "to build from source. PV-DBOW -- the doc2vec variant graph2vec is "
        "defined as -- is therefore implemented directly in vectorised numpy "
        "(`graph2vec_pipeline/embed.py`), with a fixed seed and a step budget. "
        "Same objective, different optimiser schedule from karateclub's.\n")
    out.append(
        "**What the graphs are not.** This is a linear sweep, so there is no "
        "function recovery and no indirect control flow: `call dword ptr "
        "[0x405128]`, `jmp eax`, vtable dispatch and SEH contribute no edge, "
        "and data decoded as code manufactures blocks. The unresolved-target "
        "fractions below say how much of the control flow is missing.\n")

    # ---- graph statistics ------------------------------------------------
    out.append("## 2. Graph statistics\n")
    g = stats.groupby(["source", "label"])
    tbl = pd.DataFrame({
        "n": g.size(),
        "blocks median": g["n_blocks"].median(),
        "blocks mean": g["n_blocks"].mean().round(0),
        "edges median": g["n_edges"].median(),
        "edges mean": g["n_edges"].mean().round(0),
        "insns used median": g["insns_used"].median(),
        "mean block len": g["mean_block_len"].mean().round(2),
        "back edges median": g["back_edges"].median(),
        "at block cap": g["block_capped"].mean().round(3),
    }).reset_index()
    tbl["source"] = tbl["source"] + " (label " + tbl["label"].astype(str) + ")"
    out.append(fmt(tbl.drop(columns=["label"])) + "\n")

    ga = stats.groupby(["label", "arch"])
    tbl2 = pd.DataFrame({
        "n": ga.size(),
        "blocks median": ga["n_blocks"].median(),
        "edges median": ga["n_edges"].median(),
        "mean block len": ga["mean_block_len"].mean().round(2),
        "back edges median": ga["back_edges"].median(),
        "unresolved targets median": ga["unresolved_targets"].median(),
    }).reset_index()
    out.append("\nBy class and architecture (both trees pooled):\n")
    out.append(fmt(tbl2) + "\n")

    unres = (stats["unresolved_targets"] /
             (stats["unresolved_targets"] + stats["resolved_targets"])
             .replace(0, pd.NA))
    out.append(
        f"\nMedian share of branch/call sites whose target could not be "
        f"resolved: **{float(unres.median()):.1%}** "
        f"(goodware {float(unres[stats['label']==0].median()):.1%}, "
        f"ransomware {float(unres[stats['label']==1].median()):.1%}). "
        f"{float(stats['block_capped'].mean()):.0%} of files hit the "
        f"{cfgc['max_blocks']:,}-block cap and "
        f"{float(stats['truncated'].mean()):.0%} hit the instruction cap, so "
        f"for about half the corpus these are graphs of a prefix of the code, "
        f"not of the program.\n")

    mg = stats[stats["source"] == "mendeley_goodware"]
    bg = stats[stats["source"] == "balanced_goodware"]
    mr = stats[stats["source"] == "mendeley_ransomware"]
    out.append(
        f"\n**Does size alone separate the classes?** Partly, and the "
        f"direction depends on which goodware is used. Median blocks per file: "
        f"Mendeley goodware {mg['n_blocks'].median():,.0f}, Mendeley "
        f"ransomware {mr['n_blocks'].median():,.0f}, Goodware_Balanced "
        f"{bg['n_blocks'].median():,.0f}; median edges "
        f"{mg['n_edges'].median():,.0f} / {mr['n_edges'].median():,.0f} / "
        f"{bg['n_edges'].median():,.0f}. So against Mendeley goodware the "
        f"ransomware graphs are the larger ones, and against "
        f"Goodware_Balanced they are the same size. The `size_only` rows in "
        f"section 3 put a number on exactly that, and it is the number to "
        f"quote: the ten scalars alone reach **ROC-AUC "
        f"{max((r['roc_auc'] or 0) for r in docs['mendeley']['results'] if r['representation']=='size_only'):.3f}** "
        f"on `mendeley` and "
        f"**{max((r['roc_auc'] or 0) for r in docs['balanced']['results'] if r['representation']=='size_only'):.3f}** "
        f"on `balanced`"
        if "mendeley" in docs and "balanced" in docs else
        "\n**Does size alone separate the classes?** See the `size_only` rows "
        "in section 3")
    out[-1] += (
        ". Two cautions on those medians: `n_blocks` is **censored** -- half "
        "the corpus sits exactly at the 5,000-block cap, so a median of 5,000 "
        "means 'at least 5,000', not 5,000 -- and the cap rate itself differs "
        "by class and by tree (see the `at block cap` column), which makes "
        "'hit the cap' a feature in its own right.\n")

    # ---- results ---------------------------------------------------------
    out.append("## 3. Results\n")
    for ds, doc in docs.items():
        s = doc["samples"]
        out.append(f"### {ds}\n")
        out.append(
            f"train {s['train']['n']} ({s['train']['goodware']} good / "
            f"{s['train']['ransomware']} ransomware), "
            f"val {s.get('val',{}).get('n','-')}, "
            f"test {s['test']['n']} ({s['test']['goodware']} good / "
            f"{s['test']['ransomware']} ransomware). "
            f"Majority-class accuracy on test: "
            f"{doc['results'][0]['majority_class_accuracy']:.4f}.\n")
        df = pd.DataFrame([_row(r) for r in doc["results"]])
        out.append(fmt(df, ["representation", "model", "decision", "acc",
                            "bal_acc", "macro_f1", "auc", "recall_ran", "fpr",
                            "n_feat"]) + "\n")
        out.append(
            "\n`decision = default` is argmax at 0.5. `calibrated` reuses the "
            "same fitted model and the same scores, moving only the cut, "
            "chosen on TRAIN out-of-fold predictions (`cross_val_predict`, "
            "same folds) -- test is never consulted. Train is 45% ransomware "
            "and test is 74%, so the default cut is at the wrong prior.\n")

        out.append("\nGoodware recall by architecture (1 - FPR within arch), "
                   "calibrated rows:\n")
        rows = []
        for r in doc["results"]:
            if r.get("decision", "").startswith("argmax"):
                continue
            d = {"representation": r["representation"], "model": r["model"]}
            for a, v in r.get("goodware_recall_by_arch", {}).items():
                d[f"{a} (n={v['n']})"] = v["recall_goodware"]
            rows.append(d)
        out.append(fmt(pd.DataFrame(rows)) + "\n")

        overall = max(doc["results"], key=lambda r: r["macro_f1"])
        best = max((r for r in doc["results"]
                    if r["representation"] != "size_only"),
                   key=lambda r: r["macro_f1"])
        if overall["representation"] == "size_only":
            out.append(
                f"\n> On `{ds}` the single highest macro-F1 row in the whole "
                f"table is the **shortcut control** "
                f"(`size_only/{overall['model']}`, "
                f"{overall['macro_f1']:.4f}), ahead of every structural "
                f"representation. That is a finding about this dataset, not a "
                f"result for graph embeddings: on Goodware_Balanced the CFG "
                f"features stop working before the ten size scalars do. The "
                f"per-family table below is for the best *structural* row.\n")
        pf = best.get("per_family_recall", {})
        if pf:
            out.append(f"\nPer-family ransomware recall, best structural row "
                       f"(`{best['representation']}/{best['model']}`, "
                       f"macro-F1 {best['macro_f1']:.4f}). All 14 families are "
                       f"unseen in training:\n")
            fr = pd.DataFrame([{"family": k, "recall": v["recall"],
                                "correct": v["correct"], "support": v["support"]}
                               for k, v in sorted(
                                   pf.items(), key=lambda kv: kv[1]["recall"])])
            out.append(fmt(fr) + "\n")
            zero = [k for k, v in pf.items() if v["recall"] == 0.0]
            weak = [f"{k} ({v['recall']:.2f}, n={v['support']})"
                    for k, v in sorted(pf.items(),
                                       key=lambda kv: kv[1]["recall"])[:3]]
            out.append(
                f"\nWeakest three families: {', '.join(weak)}"
                + (f"; **{len(zero)} families are missed entirely** "
                   f"({', '.join(sorted(zero))})" if zero else "")
                + ". Family recall spreads across the full 0-1 range on a "
                  "single fixed split, so a difference of a few points of "
                  "macro-F1 between two rows here is within the noise of which "
                  "14 families happened to land in test.\n")

    # ---- reading ---------------------------------------------------------
    out.append("\n## 4. Reading the numbers\n")
    lines = []
    for ds, doc in docs.items():
        best = max(doc["results"], key=lambda r: r["macro_f1"])
        size_best = max((r for r in doc["results"]
                         if r["representation"] == "size_only"),
                        key=lambda r: r["macro_f1"])
        size_auc = max(r["roc_auc"] or 0 for r in doc["results"]
                       if r["representation"] == "size_only")
        best_auc = max(r["roc_auc"] or 0 for r in doc["results"]
                       if r["representation"] != "size_only")
        lines.append(
            f"- **{ds}**: best row is `{best['representation']}/"
            f"{best['model']}` at macro-F1 {best['macro_f1']:.3f} "
            f"(AUC {best['roc_auc']:.3f}); the graph-size-only control reaches "
            f"macro-F1 {size_best['macro_f1']:.3f} and **AUC "
            f"{size_auc:.3f}**, against {best_auc:.3f} for the best "
            f"structural representation.")
    out.append("\n".join(lines) + "\n")
    out.append(
        "\nTwo things to read carefully.\n\n"
        "**The `size_only` control.** Ten scalars -- block count, edge count, "
        "instructions used, mean block length, back edges, self loops, call "
        "edges, resolved and unresolved targets, section count -- and nothing "
        "else. In macro-F1 it is clearly behind the learned representations, "
        "and the medians in section 2 agree that graph size does not split the "
        "classes outright: about half of *each* class sits at the block cap "
        "and the class medians overlap. **In ranking terms it is not behind at "
        "all**: its ROC-AUC is close to the structural rows'. So most of what "
        "the CFG representation buys over file size is a better *operating "
        "point*, not a better ordering. Any claim that the graph structure is "
        "carrying the detection has to be made against the size control's AUC, "
        "not against a majority baseline.\n\n"
        "**Threshold, not representation, is the larger effect here.** Several "
        "rows reach AUC above 0.95 while scoring near 0.55 macro-F1 at the "
        "default cut. The split is 45% ransomware in train and 74% in test; a "
        "model left at 0.5 is answering a different question from the one the "
        "test set asks. The calibrated rows move only the cut and nothing "
        "else.\n")
    out.append(
        "\n**Which representation actually wins.** `wl_tfidf/LR` is the plain "
        "WL-subtree-kernel baseline -- a linear model on the WL histogram is a "
        "WL kernel machine -- so it is the reference the learned embeddings "
        "have to beat. On `mendeley` it has the **highest ROC-AUC of any row "
        "(0.977)** while `graph2vec/SVM-RBF` has the highest macro-F1 (0.895) "
        "at AUC 0.955. Ordering and operating point disagree, on 491 test "
        "rows, between two representations built from the same WL documents. "
        "The honest reading is that the 128-dimensional embeddings do not "
        "improve on the histogram they are compressed from; they only move "
        "where the threshold lands. Nothing in these tables supports a claim "
        "that PV-DBOW learns graph structure the WL kernel misses.\n")

    # ---- architecture confound -------------------------------------------
    out.append("\n## 5. The architecture confound\n")
    rows = []
    for ds, doc in docs.items():
        s = doc["samples"]
        for fold in ("train", "test"):
            ga = s.get("goodware_arch", {}).get(fold, {})
            rows.append({"dataset": ds, "fold": fold,
                         "goodware x86": ga.get("x86", 0),
                         "goodware x64": ga.get("x64", 0),
                         "all arch": s[fold]["arch"]})
    out.append(fmt(pd.DataFrame(rows)) + "\n")
    out.append(
        "\nRansomware is 95% x86 in the train split (862 x86 / 42 x64, val "
        "fold included) and 80% x86 in test (290 / 72) in **both** datasets -- "
        "the ransomware half of the "
        "split is identical by construction. The goodware half is what "
        "changes, and it changes a lot: Mendeley goodware train is ~57% x86, "
        "Goodware_Balanced train is ~17% x86. So on `balanced` a classifier can "
        "reach a good training score with the rule *x64 implies goodware*, and "
        "that rule then meets a test set that is 64% x64 goodware but still "
        "80% x86 ransomware. The `goodware recall by architecture` tables "
        "above show the damage directly: on `balanced` every structural row "
        "keeps goodware recall near 1.00 inside x64 and drops to 0.70-0.83 "
        "inside x86. Per-architecture modelling is the obvious next control. "
        "It has been run for the mnemonic TF-IDF baseline -- see "
        "`results/rules/summary.md` §5.6, which reports a model trained and "
        "tested inside one architecture -- but not yet for these "
        "representations.\n")

    # ---- where this sits --------------------------------------------------
    out.append("\n## 6. Where this sits against the other tracks\n")
    out.append(
        "Same cohort, same family-disjoint split, same metrics module. Best "
        "macro-F1 per track, read live from each track's `metrics.json` where "
        "one exists:\n\n"
        "| track | best macro-F1 (mendeley) | best macro-F1 (balanced) | "
        "where |\n"
        "|---|---|---|---|\n"
        + "".join(f"| {label} | {a} | {b} | {where} |\n"
                  for label, a, b, where in _cross_track(docs)) + "\n"
        "Two readings. (i) On Mendeley goodware the CFG track is competitive "
        "with the tokenization baseline and clearly behind a plain TF-IDF over "
        "the same mnemonics -- a bag of 1-3-grams beats a graph embedding of "
        "the control flow those same instructions form. (ii) On "
        "Goodware_Balanced the CFG track collapses to roughly the size "
        "control, which is the more informative half of the comparison: "
        "whatever the CFG features are separating on Mendeley is largely a "
        "property of *that* goodware corpus, not of ransomware.\n")

    (RES / "summary.md").write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {RES/'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
