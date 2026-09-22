#!/usr/bin/env python3
"""
aggregate.py - one table across every pipeline evaluated under family holdout.

Reads results/family_holdout/<dataset>/<pipeline>/<model>/{metrics.json,
fold_metrics.csv, per_family.csv} and writes results/family_holdout/summary.md:

  1. per dataset: fold mean +/- sd and pooled macro-F1 / balanced accuracy /
     AUC per model, with the per-fold floors (majority, x86 rule) and the
     pooled per-architecture ransomware / goodware recall
  2. per-family recall matrix (K-fold held-out recall, and LOFO recall) across
     models, sorted by the median K-fold recall, so the families every
     pipeline fails on are visible at a glance
  3. the fixed-split number from results/COMPARISON.md for the same model where
     it can be matched by name, for the "does the one split mislead" question

    python family_holdout/aggregate.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
from family_holdout.common import FOLD_DIR as FH  # noqa: E402  ($RANSOM_FH_DIR or results/family_holdout)


def read_csv(p: Path):
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def f(x, nd=3):
    try:
        return "" if x is None or x == "" else f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def mean_sd(rows, key):
    v = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    if not v:
        return None, None
    # sample sd (ddof=1), the same definition common.py stores in metrics.json
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def main() -> int:
    L = [f"# Family-holdout evaluation ({FH.name})", "",
         "Every in-cohort ransomware family is held out exactly once in a 5-fold scheme (families assigned",
         "whole; goodware assigned by duplicate-stream group or source project).",
         "`fold mean +/- sd` is over the five held-out folds (sample sd); `pooled` scores the union of all held-out predictions.",
         "LOFO = leave-one-family-out (train on every other family + all goodware), a recall study only.",
         "Floors are per-fold means over the training folds: majority-class accuracy and the x86 rule (ransomware iff x86).",
         f"Fold files: `{FH}`. See `family_holdout/folds.py`.", ""]

    families = {}
    for ds in sorted(d.name for d in FH.iterdir() if d.is_dir() and any(d.glob("*/*/metrics.json"))):
        models = sorted((FH / ds).glob("*/*/metrics.json"))
        if not models:
            continue
        L += [f"## Dataset: {ds}", "",
              "| pipeline / model | folds macro-F1 mean +/- sd | pooled macro-F1 | pooled bal-acc | pooled AUC | "
              "pooled recall R / G | x86 R / G | x64 R / G | majority floor | x86-rule floor | LOFO mean recall |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for mp in models:
            pipe, model = mp.parent.parent.name, mp.parent.name
            m = json.loads(mp.read_text(encoding="utf-8"))
            res = m["results"][0] if isinstance(m.get("results"), list) else m
            fm = read_csv(mp.parent / "fold_metrics.csv") if (mp.parent / "fold_metrics.csv").is_file() else []
            mu, sd = mean_sd(fm, "macro_f1")
            maj, _ = mean_sd(fm, "majority_floor")
            x86r, _ = mean_sd(fm, "x86_rule_floor")
            pa = res.get("per_arch") or res.get("per_architecture") or {}
            def rg(a):
                b = pa.get(a, {})
                return f"{f(b.get('recall_ransomware'), 2)} / {f(b.get('recall_goodware'), 2)}" if b else ""
            L.append(f"| {pipe} / {model} | {f(mu)} +/- {f(sd)} | {f(res.get('macro_f1'))} | {f(res.get('balanced_accuracy'))} | "
                     f"{f(res.get('roc_auc'))} | {f(res.get('recall_ransomware'), 2)} / {f(res.get('recall_goodware'), 2)} | "
                     f"{rg('x86')} | {rg('x64')} | {f(maj)} | {f(x86r)} | {f(m.get('lofo_mean_recall'))} |")
            pf = mp.parent / "per_family.csv"
            if pf.is_file():
                for r in read_csv(pf):
                    families.setdefault(r["family"], {"n": r.get("n"), "n_x64": r.get("n_x64"), "fold": r.get("fold"),
                                                      "corpus": r.get("corpus") or "mendeley"})
                    families[r["family"]][f"{ds}:{pipe}/{model}"] = (r.get("recall_kfold"), r.get("recall_lofo"))
        L.append("")

    if families:
        cols = sorted({k for v in families.values() for k in v if ":" in k})
        L += ["## Per-family recall (K-fold held-out / LOFO), all models", "",
              "| family | corpus | n | x64 | fold | " + " | ".join(cols) + " |",
              "|---|---|---|---|---|" + "---|" * len(cols)]
        def med(fam):
            v = [float(families[fam][c][0]) for c in cols if c in families[fam] and families[fam][c][0] not in (None, "")]
            return st.median(v) if v else -1
        for fam in sorted(families, key=lambda x: (med(x), x)):
            d = families[fam]
            cells = [f"{f(d[c][0], 2)} / {f(d[c][1], 2)}" if c in d else "" for c in cols]
            L.append(f"| {fam} | {d['corpus']} | {d['n']} | {d['n_x64']} | {d['fold']} | " + " | ".join(cells) + " |")
        L.append("")
        L.append("Sorted by the median K-fold recall across models; families at the top are the ones every pipeline misses.")
        L.append("")

    comp = REPO / "results" / "COMPARISON.md"
    if comp.is_file():
        L += ["## Fixed split, for reference", "",
              "The single family-disjoint split (24 train / 14 test families) numbers are in `results/COMPARISON.md`.",
              "A model whose family-holdout pooled macro-F1 differs from its fixed-split number by more than the fold sd",
              "was being scored on a lucky or unlucky choice of test families.", ""]

    out = FH / "summary.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
