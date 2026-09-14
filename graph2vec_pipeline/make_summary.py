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
            if v is None or (isinstance(v, float) and v != v):   # None / NaN
                vals.append("-")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
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


def _tuned_best_f1(ds: str, structural_only: bool) -> str:
    """Best committed tuned row for `ds`. `posthoc_top5` is excluded: it was
    scored after the test set was opened and is not a result."""
    p = REPO / "results" / "graph2vec" / "tuned" / ds / "metrics.json"
    if not p.exists():
        return "-"
    rows = json.loads(p.read_text(encoding="utf-8")).get("results", [])
    if structural_only:
        rows = [r for r in rows if r["representation"] != "size_only"]
    return f"{max(r['macro_f1'] for r in rows):.3f}" if rows else "-"


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
        ("graph2vec / WL on CFGs, **untuned** prototype, structural rows only",
         gs.get("mendeley", "-"), gs.get("balanced", "-"), "§3 here"),
        ("graph2vec / WL on CFGs, **untuned**, incl. the `size_only` control",
         g.get("mendeley", "-"), g.get("balanced", "-"), "§3 here"),
        ("**graph2vec / WL on CFGs, tuned, structural rows only**",
         "**" + _tuned_best_f1("mendeley", True) + "**",
         "**" + _tuned_best_f1("balanced", True) + "**", "§7 here"),
        ("graph2vec / WL on CFGs, tuned, incl. the `size_only` control",
         _tuned_best_f1("mendeley", False), _tuned_best_f1("balanced", False),
         "§7 here"),
        ("CNN-ViT on instruction images (single run; 5-seed mean 0.628 / 0.502)",
         _best(cnn / "mendeley" / "metrics.json"),
         _best(cnn / "balanced" / "metrics.json"),
         "`results/cnn_vit/summary.md`"),
    ]


# ---------------------------------------------------------------------------
# section 7: the tuned run (results/graph2vec/tuned/)
# ---------------------------------------------------------------------------
TUNED = RES / "tuned"

# calibration numbers from the other tracks, for the same cohort and split
CALIBRATION = (
    ("mnemonic 1-3-gram TF-IDF + LogReg", "0.955", "0.800",
     "results/rules/summary.md"),
    ("tokenizer + word2vec (expC)", "0.926", "-", "results/summary.md"),
    ("tokenizer + word2vec (expD)", "0.844", "-", "results/summary.md"),
)


def _t_rows(doc, decision):
    want = "argmax" if decision == "argmax" else "threshold"
    return [r for r in doc["results"] if r.get("decision", "").startswith(want)]


def _tuned_docs() -> dict:
    out = {}
    for ds in ("mendeley", "balanced"):
        p = TUNED / ds / "metrics.json"
        if p.exists():
            out[ds] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _before_after(ds: str, tuned: dict, untuned: dict) -> str:
    rows = []
    best_old = max(untuned["results"], key=lambda r: r["macro_f1"])
    best_old_struct = max(
        (r for r in untuned["results"] if r["representation"] != "size_only"),
        key=lambda r: r["macro_f1"])
    for tag, r in (("before (best row of the untuned run)", best_old),
                   ("before (best *structural* row)", best_old_struct)):
        rows.append(dict(
            row=tag, config=f'{r["representation"]}/{r["model"]}',
            cv="-", acc=r["accuracy"], bal_acc=r["balanced_accuracy"],
            macro_f1=r["macro_f1"], auc=r["roc_auc"],
            fpr=r["false_positive_rate"], gap="-"))
    for r in tuned["results"]:
        dec = ("argmax" if r["decision"].startswith("argmax") else "oof-thr")
        rows.append(dict(
            row=f'after: {r["config_name"]} [{dec}]',
            config=f'{r["representation"]}/{r["model"]}',
            cv=f'{r["cv_macro_f1"]:.4f}', acc=r["accuracy"],
            bal_acc=r["balanced_accuracy"], macro_f1=r["macro_f1"],
            auc=r["roc_auc"], fpr=r["false_positive_rate"],
            gap=f'{r["cv_to_test_macro_f1_gap"]:+.4f}'))
    for name, f in tuned.get("floors", {}).items():
        rows.append(dict(row=f"floor: {name}", config="-", cv="-",
                         acc=f["accuracy"], bal_acc=f["balanced_accuracy"],
                         macro_f1=f["macro_f1"], auc=None,
                         fpr=f["false_positive_rate"], gap="-"))
    return fmt(pd.DataFrame(rows),
               ["row", "config", "cv", "acc", "bal_acc", "macro_f1", "auc",
                "fpr", "gap"])


# The search is a coordinate ascent that keeps a change only when it buys at
# least this much CV macro-F1 (graph2vec_pipeline/tune.py MIN_GAIN), so "what
# mattered" is replayed against the configuration in force at the time, not
# against a global maximum -- once a dimension is fixed, later stages only ever
# explore the winner, and a marginal table over the whole CSV would read that
# as evidence.
MIN_GAIN = 0.005
_DIMS = ("max_blocks", "windows", "calls", "cross_section", "extern",
         "labelling", "h", "edge_labels", "norm", "min_df", "with_size",
         "rep_kind", "dim", "model")


def _diff(cur, row) -> str:
    bits = [f"{d}: {cur[d]} -> {row[d]}" for d in _DIMS
            if str(cur[d]) != str(row[d])]
    return "; ".join(bits) if bits else "(no change)"


def _what_mattered(ds: str) -> str:
    """Replay the coordinate ascent: each trial against the one in force."""
    p = TUNED / ds / "cv_search.csv"
    if not p.exists():
        return ""
    cv = pd.read_csv(p)
    rows = []
    cur = cv[cv.stage == "1_construction"].iloc[0]
    curf = float(cur["cv_macro_f1"])
    rows.append(dict(stage="start",
                     change="the baseline construction and node labelling, "
                            "under group CV, min_df 3",
                     cv_macro_f1=curf, delta="-", kept="-"))

    # stage 1 is one trial per construction field, in order, each against the
    # construction in force -- so it replays as a sequence.
    for _, r in cv[cv.stage == "1_construction"].iloc[1:].iterrows():
        d = float(r["cv_macro_f1"]) - curf
        keep = d >= MIN_GAIN
        rows.append(dict(stage="graph construction", change=_diff(cur, r),
                         cv_macro_f1=float(r["cv_macro_f1"]),
                         delta=f"{d:+.4f}", kept="yes" if keep else "no"))
        if keep:
            cur, curf = r, float(r["cv_macro_f1"])

    # the rest are grids: every cell is scored against the same configuration,
    # and only the winner can be adopted.
    def grid(stage, label, describe, shortlist=None, ascent=True):
        """`ascent=False` for the last two stages: they do not feed the
        coordinate ascent at all. The configuration that goes to test is the
        global CV argmax over every row of the search, so a classifier grid
        does not have to clear MIN_GAIN to be the one that is used."""
        nonlocal cur, curf
        s = cv[cv.stage == stage]
        if not len(s):
            return
        win = s.loc[s["cv_macro_f1"].idxmax()]
        show = s.loc[shortlist(s)] if shortlist else s.loc[[win.name]]
        for _, r in show.iterrows():
            d = float(r["cv_macro_f1"]) - curf
            rows.append(dict(
                stage=label, change=describe(r),
                cv_macro_f1=float(r["cv_macro_f1"]), delta=f"{d:+.4f}",
                kept=("-" if not ascent else
                      "yes" if (r.name == win.name and d >= MIN_GAIN)
                      else "no")))
        d = float(win["cv_macro_f1"]) - curf
        if ascent and d >= MIN_GAIN:
            cur, curf = win, float(win["cv_macro_f1"])

    grid("2_labelling", "node labelling x WL depth (24 cells)",
         lambda r: f'best cell: labelling {r["labelling"]}, h={r["h"]}')
    grid("3_norm", "normalisation / min_df / size scalars (6 trials)",
         lambda r: _diff(cur, r), lambda s: list(s.index))
    grid("4_embedding", "dense embedding vs the histogram it compresses",
         lambda r: (f'best {r["rep_kind"]}: d={r["dim"]}, {r["model"]}'),
         lambda s: [g["cv_macro_f1"].idxmax()
                    for _, g in s.groupby("rep_kind")])
    for st, lab in (("5_clf_sparse", "classifier grid on the histogram"),
                    ("5_clf_dense", "classifier grid on the embedding")):
        grid(st, lab, lambda r: f'best model: {r["model"]} on {r["rep_kind"]}',
             ascent=False)
    grid("6_archweight", "architecture-aware sample reweighting",
         lambda r: f'best reweighted model: {r["model"]}', ascent=False)
    return fmt(pd.DataFrame(rows),
               ["stage", "change", "cv_macro_f1", "delta", "kept"])


def _labelling_grid(ds: str) -> str:
    p = TUNED / ds / "cv_search.csv"
    if not p.exists():
        return ""
    s = pd.read_csv(p)
    s = s[s.stage == "2_labelling"]
    if not len(s):
        return ""
    g = s.pivot_table(index="labelling", columns="h",
                      values="cv_macro_f1").round(4).reset_index()
    g.columns = ["labelling"] + [f"h={c}" for c in g.columns[1:]]
    return fmt(g)


def _size_control(ds: str) -> str:
    p = TUNED / ds / "cv_search.csv"
    if not p.exists():
        return ""
    cv = pd.read_csv(p)
    rows = []
    for kind, lab in (("size_only", "size_only (the shortcut control)"),
                      ("wl_tfidf", "wl_tfidf histogram"),
                      ("wl_svd", "wl_svd"), ("graph2vec", "graph2vec")):
        s = cv[cv.rep_kind == kind]
        if not len(s):
            continue
        b = s.loc[s["cv_macro_f1"].idxmax()]
        rows.append(dict(representation=lab, n_configs=int(len(s)),
                         best_cv_macro_f1=float(b["cv_macro_f1"]),
                         best_cv_auc=float(s["cv_auc"].max()),
                         model=b["model"]))
    return fmt(pd.DataFrame(rows))


def _top5(ds: str, tuned: dict) -> str:
    """CV -> test for the five highest-CV configurations.

    Only the row that is also `tuned_best` was committed to before the test
    set was opened; the other four were scored afterwards (metrics.json's
    `posthoc_top5`) and the `committed` column says which is which.
    """
    p = TUNED / ds / "cv_search.csv"
    if not p.exists():
        return ""
    cv = pd.read_csv(p).sort_values("cv_macro_f1", ascending=False)
    seen, keep = set(), []
    for _, r in cv.iterrows():
        k = (r["construction"], r["representation"], r["model"])
        if k in seen:
            continue
        seen.add(k)
        keep.append(r)
        if len(keep) == 5:
            break
    committed, tested = {}, {}
    for r in _t_rows(tuned, "argmax"):
        committed[(r["construction"], r["rep_spec"], r["model_spec"])] = r
    tested.update(committed)
    for r in tuned.get("posthoc_top5") or []:
        if r.get("decision", "").startswith("argmax"):
            tested.setdefault(
                (r["construction"], r["rep_spec"], r["model_spec"]), r)
    rows = []
    for r in keep:
        k = (r["construction"], r["representation"], r["model"])
        t = tested.get(k)
        rows.append(dict(
            construction=r["construction"], representation=r["representation"],
            model=r["model"], cv_macro_f1=float(r["cv_macro_f1"]),
            cv_bal_acc=float(r["cv_balanced_acc"]),
            cv_auc=float(r["cv_auc"]),
            test_macro_f1=(t["macro_f1"] if t else None),
            test_auc=(t["roc_auc"] if t else None),
            gap=(round(float(r["cv_macro_f1"]) - t["macro_f1"], 4)
                 if t else None),
            committed=("yes" if k in committed else "post hoc")))
    return fmt(pd.DataFrame(rows))


def _reading(ds: str, tuned: dict, untuned: dict) -> str:
    """The three comparisons that decide what the tuned row is worth:
    the untuned prototype, the shortcut controls, and the floors."""
    thr = _t_rows(tuned, "threshold")
    struct = [r for r in thr if r["representation"] != "size_only"]
    size = [r for r in thr if r["representation"] == "size_only"]
    if not struct:
        return ""
    best = max(struct, key=lambda r: r["macro_f1"])
    base = next((r for r in thr if r["config_name"] == "baseline_h2_cap5000"),
                None)
    old = max(untuned["results"], key=lambda r: r["macro_f1"])
    fl = tuned.get("floors", {})
    x86 = fl.get("x86_rule", {}).get("macro_f1")
    maj = fl.get("majority", {}).get("macro_f1")
    bits = [
        f'Best committed row: `{best["config_name"]}` '
        f'(`{best["representation"]}/{best["model"]}`) at macro-F1 '
        f'**{best["macro_f1"]:.4f}**, AUC {best["roc_auc"]:.4f}, '
        f'FPR {best["false_positive_rate"]:.4f}.']
    bits.append(
        f'Untuned best was {old["macro_f1"]:.4f} '
        f'(`{old["representation"]}/{old["model"]}`), so tuning moves '
        f'macro-F1 by {best["macro_f1"] - old["macro_f1"]:+.4f}.')
    if base:
        bits.append(
            f'The same untuned *configuration* re-scored here under the '
            f'group-CV threshold gives {base["macro_f1"]:.4f}.')
    if size:
        s = max(size, key=lambda r: r["macro_f1"])
        verb = "clears" if best["macro_f1"] > s["macro_f1"] else "does NOT clear"
        bits.append(
            f'It {verb} the `size_only` shortcut control '
            f'({s["macro_f1"]:.4f} macro-F1, AUC {s["roc_auc"]:.4f}) by '
            f'{best["macro_f1"] - s["macro_f1"]:+.4f}.')
    if x86 is not None and maj is not None:
        rel = "above" if best["macro_f1"] > x86 else "**below**"
        bits.append(
            f'Against the floors -- majority {maj:.4f}, x86-rule '
            f'{x86:.4f} -- it sits {rel} the architecture shortcut '
            f'({best["macro_f1"] - x86:+.4f}).')
        if best["macro_f1"] <= x86:
            bits.append(
                "That is the number to quote for this dataset: a rule that "
                "reads nothing but the ELF/PE machine field scores higher "
                "than the best searched CFG model, so nothing here "
                "demonstrates that the control-flow structure is carrying "
                "the detection.")
    gap = best["cv_macro_f1"] - best["macro_f1"]
    bits.append(
        f'CV->test gap on that row: {gap:+.4f} (CV '
        f'{best["cv_macro_f1"]:.4f}).')
    return " ".join(bits) + "\n"


def _per_arch(tuned: dict) -> str:
    rows = []
    for r in _t_rows(tuned, "threshold"):
        d = dict(config=r["config_name"],
                 model=f'{r["representation"]}/{r["model"]}')
        for a in ("x86", "x64"):
            pa = r["per_architecture"].get(a)
            if not pa:
                continue
            d[f"{a} n"] = pa["n"]
            d[f"{a} macro_f1"] = pa["macro_f1"]
            d[f"{a} rec_ran"] = pa["recall_ransomware"]
            d[f"{a} rec_good"] = pa["recall_goodware"]
        rows.append(d)
    return fmt(pd.DataFrame(rows))


def _per_family(tuned: dict) -> tuple[str, str]:
    struct = [r for r in _t_rows(tuned, "threshold")
              if r["representation"] != "size_only"]
    if not struct:
        return "", ""
    best = max(struct, key=lambda r: r["macro_f1"])
    fam = pd.DataFrame([
        dict(family=f, recall=v["recall"], correct=v["correct"],
             support=v["support"])
        for f, v in best["per_family_recall"].items()]).sort_values("recall")
    return (f'{best["config_name"]} (`{best["representation"]}/{best["model"]}`'
            f', macro-F1 {best["macro_f1"]:.4f})'), fmt(fam)


def _hard_negatives(ds: str) -> str:
    """Which goodware the best tuned model calls ransomware, and whether size
    explains it. `family` on the balanced goodware is the bucket the corpus was
    assembled from: everyday / system / hard_negative."""
    p = TUNED / ds / "predictions.csv"
    if not p.exists():
        return ""
    pr = pd.read_csv(p)
    best_name = None
    tuned = _tuned_docs().get(ds)
    if tuned:
        struct = [r for r in _t_rows(tuned, "threshold")
                  if r["representation"] != "size_only"]
        if struct:
            best_name = max(struct, key=lambda r: r["macro_f1"])["config_name"]
    sub = pr[pr["config_name"] == best_name] if best_name else pr
    good = sub[sub["label"] == 0]
    if not len(good):
        return ""
    if "n_blocks" not in good.columns:            # pre-size-column predictions
        stats = pd.read_csv(RES / "graph_stats.csv").drop_duplicates("sha256")
        good = good.merge(
            stats[["sha256", "n_blocks", "n_edges", "block_capped"]],
            on="sha256", how="left")
    else:
        good = good.rename(columns={"at_block_cap": "block_capped"})
    by_bucket = good.groupby("family").agg(
        n=("label", "size"),
        false_positives=("pred_threshold", "sum"),
        median_blocks=("n_blocks", "median")).reset_index()
    by_bucket["fp_rate"] = by_bucket["false_positives"] / by_bucket["n"]
    by_bucket = by_bucket.rename(columns={"family": "goodware bucket"})
    proj = good.groupby("family_or_group").agg(
        n=("label", "size"), false_positives=("pred_threshold", "sum"),
        median_blocks=("n_blocks", "median")).reset_index()
    proj = proj[proj["false_positives"] > 0].sort_values(
        "false_positives", ascending=False).head(10)
    proj = proj.rename(columns={"family_or_group": "project"})
    fp = good[good["pred_threshold"] == 1]
    tn = good[good["pred_threshold"] == 0]
    size_line = ""
    if len(fp) and len(tn):
        # rank statistic, because n_blocks is censored at the block cap and a
        # mean is not readable through that.
        auc = {}
        for col in ("n_blocks", "n_edges", "insns"):
            if col not in good.columns:
                continue
            a, b = fp[col].to_numpy(float), tn[col].to_numpy(float)
            gt = (a[:, None] > b[None, :]).sum()
            eq = (a[:, None] == b[None, :]).sum()
            auc[col] = (gt + 0.5 * eq) / (len(a) * len(b))
        size_line = (
            f"Median blocks: false positives {fp['n_blocks'].median():.0f} "
            f"(n={len(fp)}), correctly kept goodware "
            f"{tn['n_blocks'].median():.0f} (n={len(tn)}); at the block cap "
            f"{fp['block_capped'].mean():.2f} vs "
            f"{tn['block_capped'].mean():.2f}. Rank separation of the false "
            "positives from the rest of the test goodware by size alone "
            "(AUC, 0.5 = size says nothing): "
            + ", ".join(f"{k} {v:.3f}" for k, v in auc.items()) + ".")
        nb = auc.get("n_blocks")
        if nb is not None:
            if nb < 0.45:
                size_line += (
                    " So size does not explain the false positives, and not "
                    "in the direction the block cap would suggest either: an "
                    "AUC below 0.5 means the misclassified goodware is "
                    "*smaller* than the goodware the model keeps, the "
                    "opposite of 'big binaries look like ransomware'. The "
                    "false positives concentrate by project instead -- see "
                    "the table above -- which points at what the code is, "
                    "not how much of it there is.")
            elif nb > 0.55:
                size_line += (
                    " So the false positives are the larger graphs, and the "
                    "block cap is a live confound for this row: part of what "
                    "the model calls ransomware is simply a binary big "
                    "enough to be truncated.")
            else:
                size_line += (
                    " So size alone does not separate the false positives "
                    "from the rest of the test goodware; the errors "
                    "concentrate by project instead.")
    return (f"Best structural row: `{best_name}`.\n\n"
            + fmt(by_bucket, ["goodware bucket", "n", "false_positives",
                              "fp_rate", "median_blocks"])
            + "\n\nProjects contributing false positives:\n\n"
            + fmt(proj, ["project", "n", "false_positives", "median_blocks"])
            + "\n\n" + size_line + "\n")


def tuned_section(untuned: dict) -> list:
    tuned = _tuned_docs()
    if not tuned:
        return []
    any_t = next(iter(tuned.values()))
    sel = any_t.get("selection", {})
    sel_n = sel.get("n_configurations_searched", "?")
    out = ["\n## 7. Tuned run\n"]
    out.append(
        "Section 3 is the *untuned* prototype and is kept as written. This "
        "section is a second pass whose only rule was that the test half of "
        "the split is opened once, at the end, for the configurations a "
        "cross-validated search on TRAIN had already committed to.\n")
    out.append(
        "**The search protocol, and why the old CV numbers were not usable.** "
        "Section 3's `cv_best_f1_macro` column came from "
        "`StratifiedKFold(n_splits=2)` with no groups: ransomware families "
        "straddled the fold boundary, so a model could recognise a family it "
        "had already seen. That is why the untuned `balanced` rows report CV "
        "macro-F1 near 0.98 against a test macro-F1 of 0.53. Here every "
        "selection decision uses "
        f"`StratifiedGroupKFold(n_splits={sel.get('cv_folds', 5)})` over "
        "`family_or_group`, so a whole family (and on `balanced` a whole "
        "source project) is held out; the reported CV number is the pooled "
        "out-of-fold macro-F1. "
        f"{sel.get('n_configurations_searched', '?')} configurations were "
        "evaluated on `mendeley` and the same protocol on `balanced`; all of "
        "them are in `results/graph2vec/tuned/<dataset>/cv_search.csv` with "
        "their CV macro-F1, balanced accuracy and AUC.\n")
    out.append(
        "**What the search was allowed to change.** Graph construction (block "
        "cap 5,000 vs 20,000; a single 80k-instruction prefix vs four "
        "20k-instruction windows spread over the file; call edges on/off; "
        "cross-section branch targets on/off; import thunks as shared extern "
        "nodes on/off; edge kinds salted into the WL relabelling or not), node "
        "labelling (`class` = dominant mnemonic class + terminator + length "
        "bucket + crypto/string/SIMD flags, `class_noflag`, `dom_term`, "
        "`seq_term` = hashed 8-mnemonic-class sequence, `deg_dom` = "
        "(in-degree, out-degree, class, terminator), `lenbucket` = length "
        "bucket + terminator only), WL depth h=1..4, histogram normalisation "
        "(TF-IDF / binary / log) and min_df, whether the size scalars are "
        "concatenated onto the histogram, SVD and PV-DBOW dimensions, the "
        "classifier grids, and architecture-aware sample reweighting.\n")

    for ds, doc in tuned.items():
        out.append(f"\n### 7.{1 if ds == 'mendeley' else 2} {ds}\n")
        s = doc["samples"]
        out.append(
            f'test {s["test"]["n"]} ({s["test"]["goodware"]} good / '
            f'{s["test"]["ransomware"]} ransomware), train {s["train"]["n"]}.\n')
        out.append("**Before / after.** `cv` is the group-CV macro-F1 the "
                   "configuration was selected on; `gap` is CV minus test.\n")
        out.append(_before_after(ds, doc, untuned[ds]) + "\n")
        out.append("**Reading that table.** " + _reading(ds, doc, untuned[ds]))
        out.append("Calibration, same cohort and split:\n\n"
                   "| track | mendeley macro-F1 | balanced macro-F1 | where |\n"
                   "|---|---|---|---|\n"
                   + "".join(f"| {a} | {b} | {c} | {d} |\n"
                             for a, b, c, d in CALIBRATION) + "\n")
        out.append(
            "**What helped, and what did not.** The search is a coordinate "
            "ascent, so each trial below is scored against the configuration "
            "in force when it ran, not against a global best; a change is "
            f"kept only if it buys at least {MIN_GAIN} CV macro-F1. Where a "
            "stage is a grid rather than a sequence of trials only its winner "
            "is shown. The last three stages carry `kept = -` because they do "
            "not feed the ascent: the configuration sent to test is the "
            "global CV argmax over all "
            f'{sel_n} rows, whatever the marginal gain.\n')
        out.append(_what_mattered(ds) + "\n")
        out.append("Node labelling against WL depth, the one full grid in the "
                   "search (CV macro-F1):\n")
        out.append(_labelling_grid(ds) + "\n")
        out.append("**Does the structural model beat graph size?** Best CV "
                   "reached by each representation anywhere in the search:\n")
        out.append(_size_control(ds) + "\n")
        out.append("**Top 5 configurations by CV macro-F1**, and what they "
                   "score on test. Only the `committed = yes` row was "
                   "selected before the test set was opened; the rest were "
                   "scored afterwards purely to size the CV->test gap, and "
                   "nothing is chosen from this table.\n")
        out.append(_top5(ds, doc) + "\n")
        out.append("**Per architecture**, out-of-fold-threshold rows:\n")
        out.append(_per_arch(doc) + "\n")
        label, table = _per_family(doc)
        if table:
            out.append(f"**Per-family ransomware recall**, {label}. All test "
                       "families are unseen in training:\n")
            out.append(table + "\n")
        if ds == "balanced":
            hn = _hard_negatives(ds)
            if hn:
                out.append("**Hard negatives.** Which goodware the best "
                           "structural model calls ransomware, and whether "
                           "graph size explains it:\n")
                out.append(hn)
    out.append("\n### 7.3 Reproducing\n")
    out.append(
        "```\n"
        "python graph2vec_pipeline/graph_cache.py --max-blocks 20000\n"
        "python graph2vec_pipeline/graph_cache.py --max-blocks 20000 "
        "--windows 4\n"
        "python graph2vec_pipeline/tune.py --dataset both\n"
        "python graph2vec_pipeline/final_eval.py --dataset both\n"
        "python graph2vec_pipeline/make_summary.py\n"
        "```\n"
        f'Seed {sel.get("seed", 42)} throughout; the WL relabelling uses a '
        "fixed 64-bit mixer and no RNG, the graph cache is a pure function of "
        "the disassembly, and PV-DBOW seeds its SGD from "
        "`np.random.default_rng(42)`. `config_used.yaml` in each tuned "
        "directory records exactly what was fitted.\n")
    rep = [(ds, d["reproducibility"]) for ds, d in tuned.items()
           if d.get("reproducibility")]
    if rep:
        out.append(
            "\n**Reproducibility.** The selected configuration was refitted "
            "from scratch in a separate process "
            "(`final_eval.py --repro-check`) -- graph assembly, WL "
            "relabelling, the min_df vocabulary, IDF, SVD, scaler and "
            "classifier -- and compared against `predictions.csv` sample by "
            "sample:\n\n"
            "| dataset | test samples | max abs score difference | identical "
            "predictions | macro-F1 run 1 | macro-F1 run 2 |\n"
            "|---|---|---|---|---|---|\n"
            + "".join(
                f'| {ds} | {r.get("n_test_samples", "-")} | '
                f'{r["max_abs_score_difference"]:.3g} | '
                f'{"yes" if r.get("identical_predictions_threshold") else "NO"}'
                f' | {r["macro_f1_run1"]:.4f} | {r["macro_f1_run2"]:.4f} |\n'
                for ds, r in rep) + "\n")
    for c in any_t.get("caveats", []):
        out.append(f"- {c}\n")
    out.append(
        "- The `graph2vec` (PV-DBOW) rows are the one part of this table that "
        "does **not** reproduce exactly. Its SGD stops on a wall-clock budget "
        "(`max_seconds`, checked every 200 steps), so a busier machine takes "
        "fewer steps and lands on different document vectors from the same "
        "seed. Two runs of the `mendeley` PV-DBOW pick a few hours apart gave "
        "test macro-F1 0.617 and 0.486 at argmax. Everything else here -- the "
        "graph build, the WL relabelling, the vocabulary, IDF, SVD and every "
        "classifier -- is seed-determined and did reproduce to the last bit "
        "(see the table above). Read the PV-DBOW row as a range, not a "
        "number, and do not rank it against the others on a single run.\n")
    return out


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
        "The two graph2vec pairs are the same code on the same split; what "
        "separates them is that the tuned rows chose their construction, node "
        "labelling, WL depth, normalisation and classifier by grouped "
        "cross-validation on TRAIN (§7) while the untuned rows are one "
        "hand-picked configuration. Read the untuned pair as what the "
        "prototype did, not as what the representation can do.\n\n"
        "Two readings of the untuned pair. (i) On Mendeley goodware the CFG "
        "track was competitive with the tokenization baseline and behind a "
        "plain TF-IDF over the same mnemonics. (ii) On Goodware_Balanced it "
        "collapsed to roughly the size control, which looked like the more "
        "informative half of the comparison: whatever the CFG features "
        "separated on Mendeley looked like a property of *that* goodware "
        "corpus. §7 puts a searched configuration next to both; its "
        "before/after and `size_only` tables are what the second reading has "
        "to be checked against.\n")

    out.extend(tuned_section(docs))

    (RES / "summary.md").write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {RES/'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
