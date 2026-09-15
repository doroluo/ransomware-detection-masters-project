#!/usr/bin/env python3
"""Turn the family-holdout model directories into one summary per pipeline.

    python family_holdout/make_summary.py --pipeline all

Reads `results/family_holdout/<dataset>/<pipeline>/<model>/{metrics.json,
fold_metrics.csv,per_family.csv}` and writes
`results/family_holdout/summary_<pipeline>.md` with

  * a per-dataset table: fold mean +/- sd, pooled, both floors, and the
    fixed-split number for the SAME model out of results/COMPARISON.md
  * a per-family recall table, K-fold and LOFO side by side, sorted by recall
  * a per-arch table over the pooled K-fold predictions

Nothing here recomputes a metric: every number is read back from the files the
runners wrote, so the summary cannot drift from them.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from family_holdout.common import DATASETS, OUT_ROOT  # noqa: E402

COMPARISON = REPO / "results" / "COMPARISON.md"

# model directory -> how to find the same model's fixed-split row in
# results/COMPARISON.md: (pipeline substring, dataset substring, model column
# PREFIX, note substring). A pipeline spec beginning with "=" must match the
# pipeline column exactly - "tokenization expD" is a prefix of "tokenization
# expD_tuned", and the tuned rows are a different experiment.
# The model column carries the chosen threshold
# ("/thr 0.38"), which differs per dataset, so it is matched by prefix.
# A spec that matches no row, or more than one, yields "n/a" rather than a
# number: that model was never run on the fixed split.
FIXED_SPLIT = {
    "tfidf": {
        "LogReg": ("rules", "{ds}", "mnemonic_tfidf_1_3+LogReg/calibration_baseline", ""),
        "LinearSVC": ("rules", "{ds}", "mnemonic_tfidf_1_3+LinearSVC/calibration_baseline", ""),
    },
    "tokenization": {
        "RF_WPC_w2v": ("=tokenization exp{CD}", "", "RF/WPC/w2v", ""),
        "MLP_WP_w2v": ("=tokenization exp{CD}", "", "MLP/WP/w2v", ""),
        "SVM-RBF_SW_w2v": ("=tokenization exp{CD}", "", "SVM-RBF/SW/w2v", ""),
    },
    "graph2vec": {
        "baseline_h2_cap5000_argmax":
            ("graph2vec (tuned)", "{ds}", "LR/wl_tfidf/argmax (default 0.5)",
             "baseline_h2_cap5000"),
        "baseline_h2_cap5000_oofthr":
            ("graph2vec (tuned)", "{ds}",
             "LR/wl_tfidf/threshold from train out-of-fold scores",
             "baseline_h2_cap5000"),
        "tuned_best_sparse_histogram_argmax":
            ("graph2vec (tuned)", "{ds}", "LR/wl_tfidf/argmax (default 0.5)",
             "tuned_best_sparse_histogram"),
        "tuned_best_sparse_histogram_oofthr":
            ("graph2vec (tuned)", "{ds}",
             "LR/wl_tfidf/threshold from train out-of-fold scores",
             "tuned_best_sparse_histogram"),
    },
}
DS_WORD = {"mendeley": "mendeley", "balanced": "balanced"}
CD = {"mendeley": "C", "balanced": "D"}

ORDER = {
    "tfidf": ["LogReg", "LinearSVC"],
    "tokenization": ["MLP_WP_w2v", "SVM-RBF_SW_w2v", "RF_WPC_w2v"],
    "graph2vec": ["baseline_h2_cap5000_argmax", "baseline_h2_cap5000_oofthr",
                  "tuned_best_sparse_histogram_argmax",
                  "tuned_best_sparse_histogram_oofthr"],
}


# ---------------------------------------------------------------------------
def comparison_rows():
    rows = []
    for line in COMPARISON.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 12 or cells[0] == "pipeline":
            continue
        rows.append(dict(pipeline=cells[0], dataset=cells[1], model=cells[2],
                         acc=cells[3], bal=cells[4], f1=cells[5],
                         auc=cells[6], majority=cells[7], machine=cells[8],
                         x86=cells[9], x64=cells[10], note=cells[11]))
    return rows


def fixed_split(pipeline: str, model: str, dataset: str, rows):
    spec = FIXED_SPLIT.get(pipeline, {}).get(model)
    if spec is None:
        return None
    pl, ds, mod, note = spec
    pl = pl.replace("{CD}", CD[dataset])
    ds = ds.replace("{ds}", DS_WORD[dataset])
    match_pl = ((lambda v: v == pl[1:]) if pl.startswith("=")
                else (lambda v: pl in v))
    hits = [r for r in rows
            if match_pl(r["pipeline"]) and ds in r["dataset"]
            and r["model"].startswith(mod) and note in r["note"]]
    if len(hits) != 1:
        return None
    return hits[0]


def load_model(dataset: str, pipeline: str, model: str, root: Path):
    d = root / dataset / pipeline / model
    if not (d / "metrics.json").exists():
        return None
    doc = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    with (d / "fold_metrics.csv").open(encoding="utf-8", newline="") as fh:
        folds = list(csv.DictReader(fh))
    with (d / "per_family.csv").open(encoding="utf-8", newline="") as fh:
        fams = list(csv.DictReader(fh))
    return {"dir": d, "doc": doc, "res": doc["results"][0], "folds": folds,
            "fams": fams}


def _f(x, nd=3):
    if x is None or x == "":
        return "n/a"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _mean(vals):
    v = [float(x) for x in vals if x not in ("", None)]
    return sum(v) / len(v) if v else None


# ---------------------------------------------------------------------------
def dataset_table(pipeline, dataset, models, rows, root) -> list:
    L = [f"### {dataset}", "",
         "| model | fold mean macro-F1 +/- sd | pooled macro-F1 | pooled bal-acc "
         "| pooled acc | pooled AUC | majority floor (acc / macro-F1) | x86-rule "
         "floor (acc / macro-F1) | LOFO mean recall | fixed-split macro-F1 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        got = load_model(dataset, pipeline, m, root)
        if got is None:
            L.append(f"| {m} | (not run) |  |  |  |  |  |  |  |  |")
            continue
        r, doc = got["res"], got["doc"]
        fl = doc["floors"]["pooled"]
        fs = fixed_split(pipeline, m, dataset, rows)
        L.append(
            f"| {m} | {_f(r['fold_mean']['macro_f1'])} +/- "
            f"{_f(r['fold_sd']['macro_f1'])} | **{_f(r['macro_f1'])}** | "
            f"{_f(r['balanced_accuracy'])} | {_f(r['accuracy'])} | "
            f"{_f(r['roc_auc'])} | {_f(fl['majority']['accuracy'])} / "
            f"{_f(fl['majority']['macro_f1'])} | "
            f"{_f(fl['x86_rule']['accuracy'])} / "
            f"{_f(fl['x86_rule']['macro_f1'])} | "
            f"{_f(r['lofo_mean_recall'])} | "
            f"{(fs['f1'] if fs else 'n/a')} |")
    L.append("")
    L.append("Per-fold spread of the other headline metrics:")
    L.append("")
    L.append("| model | acc (mean +/- sd) | bal-acc (mean +/- sd) | "
             "AUC (mean +/- sd) |")
    L.append("|---|---|---|---|")
    for m in models:
        got = load_model(dataset, pipeline, m, root)
        if got is None:
            continue
        r = got["res"]
        L.append(f"| {m} | {_f(r['fold_mean']['accuracy'])} +/- "
                 f"{_f(r['fold_sd']['accuracy'])} | "
                 f"{_f(r['fold_mean']['balanced_accuracy'])} +/- "
                 f"{_f(r['fold_sd']['balanced_accuracy'])} | "
                 f"{_f(r['fold_mean']['roc_auc'])} +/- "
                 f"{_f(r['fold_sd']['roc_auc'])} |")
    L.append("")
    return L


def arch_table(pipeline, dataset, models, root) -> list:
    L = [f"| model | arch | n | support r / g | recall ransomware | "
         f"recall goodware | accuracy | macro-F1 |", "|---|---|---|---|---|---|---|---|"]
    any_row = False
    for m in models:
        got = load_model(dataset, pipeline, m, root)
        if got is None:
            continue
        pa = got["res"]["per_arch"]
        for a in sorted(pa):
            v = pa[a]
            any_row = True
            L.append(f"| {m} | {a} | {v['n']} | {v['support_ransomware']} / "
                     f"{v['support_goodware']} | {_f(v['recall_ransomware'])} | "
                     f"{_f(v['recall_goodware'])} | {_f(v['accuracy'])} | "
                     f"{_f(v['macro_f1'])} |")
    return L if any_row else []


def family_table(pipeline, dataset, models, root) -> list:
    got = {m: load_model(dataset, pipeline, m, root) for m in models}
    got = {m: g for m, g in got.items() if g}
    if not got:
        return []
    fams = {}
    for m, g in got.items():
        for row in g["fams"]:
            e = fams.setdefault(row["family"],
                                {"n": row["n"], "n_x64": row["n_x64"],
                                 "fold": row["fold"], "k": {}, "l": {}})
            e["k"][m] = float(row["recall_kfold"])
            e["l"][m] = float(row["recall_lofo"])
    ms = list(got)
    head = ("| family | n | n_x64 | fold | "
            + " | ".join(f"{m} K-fold | {m} LOFO" for m in ms)
            + " | mean LOFO |")
    L = [head, "|" + "---|" * (4 + 2 * len(ms) + 1)]
    ordered = sorted(fams.items(), key=lambda kv: (_mean(kv[1]["l"].values()),
                                                   _mean(kv[1]["k"].values())))
    for fam, e in ordered:
        cells = " | ".join(f"{_f(e['k'].get(m), 2)} | {_f(e['l'].get(m), 2)}"
                           for m in ms)
        L.append(f"| {fam} | {e['n']} | {e['n_x64']} | {e['fold']} | {cells} | "
                 f"{_f(_mean(e['l'].values()), 2)} |")
    return L


def delta_table(pipeline, models, rows, root) -> list:
    L = ["| dataset | model | pooled macro-F1 (K-fold) | fixed-split macro-F1 "
         "| delta | headroom over the x86-rule floor | LOFO mean recall | "
         "families with LOFO recall < 0.5 |",
         "|---|---|---|---|---|---|---|---|"]
    any_row = False
    for ds in DATASETS:
        for m in models:
            got = load_model(ds, pipeline, m, root)
            if got is None:
                continue
            any_row = True
            r = got["res"]
            fl = got["doc"]["floors"]["pooled"]["x86_rule"]["macro_f1"]
            fs = fixed_split(pipeline, m, ds, rows)
            dl = ("n/a" if fs is None
                  else f"{r['macro_f1'] - float(fs['f1']):+.3f}")
            bad = sorted(f for f, v in r["lofo_recall_by_family"].items()
                         if v < 0.5)
            L.append(f"| {ds} | {m} | {_f(r['macro_f1'])} | "
                     f"{(fs['f1'] if fs else 'n/a')} | {dl} | "
                     f"{r['macro_f1'] - fl:+.3f} | "
                     f"{_f(r['lofo_mean_recall'])} | "
                     f"{len(bad)}: {', '.join(bad) if bad else '-'} |")
    return (L + [""]) if any_row else []


def write_summary(pipeline: str, root: Path) -> Path:
    rows = comparison_rows()
    models = ORDER[pipeline]
    L = [f"# Family-holdout: `{pipeline}`", "",
         "Every ransomware family in the cohort is held out, twice over, with "
         "the fold definition in `family_holdout/folds.py` "
         "(`results/family_holdout/folds_<dataset>.csv`):", "",
         "* **K-fold** - families are assigned whole to 5 folds; train on four, "
         "test on the fifth. Every file gets exactly one held-out prediction, "
         "so the pooled row is a complete cross-validated pass over the cohort.",
         "* **LOFO** - leave one family out: train on every other ransomware "
         "family plus ALL goodware, test on that family alone. No goodware in "
         "the test set, so it is a recall study only (no FPR, no AUC).", "",
         "Nothing here is tuned. The configuration is the pre-registered one "
         "for each pipeline; see `config_used.yaml` in each model directory.",
         "",
         "> **Architecture caveat, carried into every table below.** x64 "
         "ransomware concentrates in fold 2: 59 of the 114 x64 ransomware "
         "files land there, and Hive alone is 43 of them. A per-fold x64 "
         "number outside fold 2 rests on a handful of files, and the per-fold "
         "spread of any x64 metric is not a sampling spread.", "",
         "## Headline", ""]
    for ds in DATASETS:
        L += dataset_table(pipeline, ds, models, rows, root)
    L += ["## Per architecture (pooled K-fold predictions)", ""]
    for ds in DATASETS:
        t = arch_table(pipeline, ds, models, root)
        if t:
            L += [f"### {ds}", ""] + t + [""]
    L += ["## Per family: K-fold recall vs LOFO recall", "",
          "Sorted by mean LOFO recall, worst first. `fold` is the K-fold the "
          "family is assigned to; `n_x64` is how many of its files are x64.", ""]
    for ds in DATASETS:
        t = family_table(pipeline, ds, models, root)
        if t:
            L += [f"### {ds}", ""] + t + [""]
    L += ["## Against the fixed split", "",
          "The `fixed-split macro-F1` column of the headline table is the same "
          "model's row in `results/COMPARISON.md`, i.e. the single Mendeley "
          "family-disjoint split (24 train / 14 test families). The two are "
          "not the same experiment: the fixed split tests 14 families with a "
          "74%-ransomware test set, the K-fold pooled row tests all 38 with a "
          "roughly balanced one, and LOFO gives every family the largest "
          "possible training set. A family-holdout number above the "
          "fixed-split one usually means the fixed split's 14 test families "
          "were the harder half, not that the model improved.", ""]
    L += delta_table(pipeline, models, rows, root)
    out = root / f"summary_{pipeline}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pipeline", default="all",
                    choices=("all", "tfidf", "tokenization", "graph2vec"))
    ap.add_argument("--out", default=str(OUT_ROOT))
    a = ap.parse_args()
    ps = list(ORDER) if a.pipeline == "all" else [a.pipeline]
    for p in ps:
        print("wrote", write_summary(p, Path(a.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
