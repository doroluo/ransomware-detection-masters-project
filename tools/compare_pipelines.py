#!/usr/bin/env python3
"""
compare_pipelines.py - one table across every pipeline that was scored on the
shared cohort and family-disjoint split.

Reads metrics.json from:
    results/exp*/                     tokenization pipeline (six experiments)
    results/cnn_vit/{mendeley,balanced}/  + results/cnn_vit/seeds/*/seed*/   CNN-ViT
    results/graph2vec/{mendeley,balanced}/
    results/rules/{mendeley,balanced}/
    results/ember/{...}/              if present (teammate)

and writes results/COMPARISON.md. Every row carries the two floors that make a
score meaningful on a 74%-ransomware test set: majority-class accuracy and the
machine-only rule (predict ransomware iff x86), both computed from the row's
own test set when the metrics file records support and per-arch blocks.

    python tools/compare_pipelines.py
"""
from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
R = REPO / "results"

DATASET_OF = {
    "expA": "mendeley (traditional)", "expA_cohort": "mendeley, cohort (traditional)",
    "expC": "mendeley, cohort (revised)",
    "expB": "balanced (traditional)", "expB_cohort": "balanced, cohort (traditional)",
    "expD": "balanced, cohort (revised)",
}


def results_of(path: Path):
    m = json.loads(path.read_text(encoding="utf-8"))
    res = m["results"] if isinstance(m["results"], list) else list(m["results"].values())
    return m, res


def floors(r: dict):
    """majority-class accuracy and machine-only (x86 -> ransomware) accuracy."""
    sg, sr = r.get("support_goodware"), r.get("support_ransomware")
    maj = r.get("majority_class_accuracy")
    if maj is None and sg is not None and sr is not None and (sg + sr):
        maj = max(sg, sr) / (sg + sr)
    pa = r.get("per_arch") or r.get("per_architecture") or {}
    mach = None
    if pa and sg is not None and sr is not None and (sg + sr):
        correct = 0
        for a, blk in pa.items():
            g, rr = blk.get("support_goodware", 0), blk.get("support_ransomware", 0)
            correct += rr if a == "x86" else g
        mach = correct / (sg + sr)
    return maj, mach


def fmt(x, nd=3):
    return "" if x is None else f"{x:.{nd}f}"


def arch_cells(r: dict):
    pa = r.get("per_arch") or r.get("per_architecture") or {}
    out = []
    for a in ("x86", "x64"):
        blk = pa.get(a)
        out.append("" if not blk else f"{blk.get('recall_ransomware', float('nan')):.2f}/{blk.get('recall_goodware', float('nan')):.2f}")
    return out


def label_of(r: dict):
    parts = [r.get("model") or r.get("name") or "?"]
    for k in ("tokenizer", "tok", "embedding", "emb", "representation", "features", "variant", "track", "decision"):
        if r.get(k):
            parts.append(str(r[k]))
    if r.get("n_rules") is not None:
        parts.append(f"{r['n_rules']} rules")
    if r.get("threshold") is not None:
        parts.append(f"thr {float(r['threshold']):.2f}")
    return "/".join(parts)


def main() -> int:
    rows = []  # (pipeline, dataset, model, acc, bal, f1, auc, maj, mach, x86 r/g, x64 r/g, note)

    # tokenization
    for exp, ds in DATASET_OF.items():
        p = R / exp / "metrics.json"
        if not p.is_file():
            continue
        _, res = results_of(p)
        for r in res:
            maj, mach = floors(r)
            rows.append(("tokenization " + exp, ds, label_of(r), r.get("accuracy"), r.get("balanced_accuracy"),
                         r.get("macro_f1"), r.get("roc_auc"), maj, mach, *arch_cells(r), ""))

    # cnn-vit: seed-1337 run + seed sweep mean/sd
    for ds in ("mendeley", "balanced"):
        p = R / "cnn_vit" / ds / "metrics.json"
        if p.is_file():
            _, res = results_of(p)
            r = res[0]
            maj, mach = floors(r)
            rows.append(("cnn_vit (seed 1337)", f"{ds}, cohort (unified asm)", "HierarchicalMalwareNet",
                         r.get("accuracy"), r.get("balanced_accuracy"), r.get("macro_f1"), r.get("roc_auc"),
                         maj, mach, *arch_cells(r), ""))
        seeds = sorted(glob.glob(str(R / "cnn_vit" / "seeds" / ds / "seed*" / "metrics.json")))
        if seeds:
            rs = [results_of(Path(s))[1][0] for s in seeds]
            def ms(k):
                v = [x[k] for x in rs if x.get(k) is not None]
                return (st.mean(v), st.pstdev(v)) if v else (None, None)
            maj, mach = floors(rs[0])
            cells = []
            for k in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc"):
                m_, s_ = ms(k)
                cells.append(None if m_ is None else m_)
            rows.append((f"cnn_vit (mean of {len(rs)} seeds)", f"{ds}, cohort (unified asm)", "HierarchicalMalwareNet",
                         *cells, maj, mach, "", "", "sd macro-F1 " + fmt(ms("macro_f1")[1])))

    # graph2vec, rules, ember
    for pipe in ("graph2vec", "rules", "ember"):
        for p in sorted((R / pipe).glob("*/metrics.json")):
            ds = p.parent.name
            _, res = results_of(p)
            for r in res:
                maj, mach = floors(r)
                rows.append((pipe, f"{ds}, cohort", label_of(r), r.get("accuracy"), r.get("balanced_accuracy"),
                             r.get("macro_f1"), r.get("roc_auc"), maj, mach, *arch_cells(r), ""))

    # write
    L = ["# Cross-pipeline comparison", "",
         "All rows below are scored on the shared cohort (tag plain / upx_unpacked, Thanos excluded) and the",
         "Mendeley family-disjoint split (24 train / 14 test ransomware families), except the two `expA`/`expB`",
         "rows, which are the pre-cohort baselines. `mendeley` = Mendeley goodware; `balanced` = Goodware_Balanced",
         "on expB's project-grouped goodware split. Ransomware is identical in every row.", "",
         "Floors: **majority** = predict the majority class (ransomware) everywhere; **machine** = predict ransomware",
         "iff the file is x86, from the cohort architecture. A score is only interesting relative to these.",
         "`x86 r/g` and `x64 r/g` = ransomware recall / goodware recall inside that architecture.", "",
         "| pipeline | dataset | model | acc | bal-acc | macro-F1 | AUC | majority | machine | x86 r/g | x64 r/g | note |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        pipe, ds, model, acc, bal, f1, auc, maj, mach, x86, x64, note = row
        L.append(f"| {pipe} | {ds} | {model} | {fmt(acc)} | {fmt(bal)} | {fmt(f1)} | {fmt(auc)} | {fmt(maj)} | {fmt(mach)} | {x86} | {x64} | {note} |")

    # best per pipeline family, by macro-F1, per dataset family
    L += ["", "## Best macro-F1 per pipeline and dataset", "", "| pipeline | mendeley-type dataset | balanced-type dataset |", "|---|---|---|"]
    fam = {}
    for row in rows:
        pipe = row[0].split(" (")[0].split(" exp")[0]
        dskey = "balanced" if row[1].startswith("balanced") else "mendeley"
        f1 = row[5]
        if f1 is None:
            continue
        cur = fam.setdefault(pipe, {}).get(dskey)
        if cur is None or f1 > cur[0]:
            fam[pipe][dskey] = (f1, f"{row[0]} · {row[2]}")
    for pipe, d in fam.items():
        m_ = d.get("mendeley"); b_ = d.get("balanced")
        L.append(f"| {pipe} | {fmt(m_[0]) + ' (' + m_[1] + ')' if m_ else ''} | {fmt(b_[0]) + ' (' + b_[1] + ')' if b_ else ''} |")

    out = R / "COMPARISON.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out} with {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
