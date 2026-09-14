#!/usr/bin/env python3
"""Turn the tuning artefacts into the markdown that goes in summary.md.

Reads only what is already on disk - nothing here trains anything - so the
tables in `results/cnn_vit/summary.md` can be regenerated and diffed rather
than hand-typed:

  search   top-N configurations per dataset by CV macro-F1, plus a per-axis
           effect table (how much each search dimension moved CV macro-F1
           away from the base configuration it was varied from).
  gap      CV -> test gap for the top-3 CV configurations. These are *post
           hoc*: the test fold was scored after selection was already
           finished, purely to measure how far CV over-/under-states test.
  final    before/after against the untuned 5-seed sweep in
           results/cnn_vit/seeds/, with the majority-accuracy floor, the
           per-architecture split and the per-family recalls.

    python cnn_vit_pipeline/tuned_report.py search --top 10
    python cnn_vit_pipeline/tuned_report.py gap
    python cnn_vit_pipeline/tuned_report.py final
    python cnn_vit_pipeline/tuned_report.py all > /tmp/tuned_tables.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results" / "cnn_vit"
TUNED = RESULTS / "tuned"
UNTUNED_SEEDS = RESULTS / "seeds"
DATASETS = ("mendeley", "balanced")

#: Config fields, in the order a reader wants to scan them. Anything not in
#: this list is bookkeeping and never differs between search rows.
KNOBS = ["variant", "row_align", "crops", "tta", "lr", "weight_decay",
         "batch_size", "epochs", "patience", "warmup_epochs", "schedule",
         "grad_clip", "label_smoothing", "loss", "focal_gamma", "sampler",
         "amp", "monitor", "dropout", "drop_path", "width", "depth", "heads",
         "mlp_ratio", "aug_p", "aug_max_shift", "threshold"]

#: which search stage each knob belongs to, for the "what mattered" table
AXIS = {
    "encoding": ["variant", "row_align", "crops", "tta"],
    "optimisation": ["lr", "weight_decay", "batch_size", "epochs", "patience",
                     "warmup_epochs", "schedule", "grad_clip", "monitor",
                     "label_smoothing", "amp"],
    "loss / sampling": ["loss", "focal_gamma", "sampler"],
    "regularisation / capacity": ["dropout", "drop_path", "width", "depth",
                                  "heads", "mlp_ratio", "aug_p",
                                  "aug_max_shift"],
    "decision rule": ["threshold"],
}


# ------------------------------------------------------------------ utils ---
def load_search(dataset: str) -> pd.DataFrame:
    p = TUNED / dataset / "cv_search.csv"
    if not p.exists():
        raise SystemExit(f"no search table at {p}")
    df = pd.read_csv(p)
    df["row"] = np.arange(len(df))
    return df


def base_row(df: pd.DataFrame) -> pd.Series:
    """The unvaried configuration every one-factor row was varied from.

    It is the first row whose knobs equal the trainer's defaults, i.e. the row
    that every stage used as its starting point.
    """
    from cnn_vit_pipeline.tuned_train import Config
    d = Config()
    m = np.ones(len(df), dtype=bool)
    for k in KNOBS:
        if k not in df.columns:
            continue
        want = getattr(d, k)
        col = df[k]
        m &= (col.astype(str) == str(want)).to_numpy()
    if not m.any():
        return df.sort_values("cv_macro_f1", ascending=False).iloc[0]
    return df[m].iloc[0]


def diff_vs(row: pd.Series, base: pd.Series) -> dict:
    out = {}
    for k in KNOBS:
        if k in row.index and str(row[k]) != str(base[k]):
            out[k] = row[k]
    return out


def fmt_diff(d: dict) -> str:
    if not d:
        return "_(base)_"
    return " ".join(f"`{k}={_short(v)}`" for k, v in d.items())


def _short(v):
    if isinstance(v, float):
        return f"{v:g}"
    return v


def md_table(header: list[str], rows: list[list], align: str = "") -> str:
    align = align or ("l" * len(header))
    sep = {"l": "---", "r": "---:", "c": ":---:"}
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(sep[a] for a in align) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# ----------------------------------------------------------------- search ---
def report_search(top: int = 10) -> str:
    out = []
    for ds in DATASETS:
        df = load_search(ds)
        base = base_row(df)
        out.append(f"\n**`{ds}`** - {len(df)} configurations, "
                   f"{int(base['folds'])}-fold group CV on the train split "
                   f"({int(base['cv_n'])} samples), CV seed {int(base['seed'])}. "
                   f"Base configuration CV macro-F1 "
                   f"**{base['cv_macro_f1']:.4f}**.\n")
        best = rank_configs(df, base).head(top)
        rows = []
        for i, (_, r) in enumerate(best.iterrows(), 1):
            cv = (f"{r['cv_macro_f1']:.4f}" if r["cv_seeds"] == 1 else
                  f"{r['cv_macro_f1']:.4f} ± {r['cv_sd']:.4f}")
            rows.append([i, f"`{r['config_id']}`", r["stage"], cv,
                         int(r["cv_seeds"]),
                         f"{r['cv_macro_f1'] - base['cv_macro_f1']:+.4f}",
                         f"{r['cv_balanced_accuracy']:.4f}",
                         f"{r['cv_auc']:.4f}",
                         fmt_diff(json.loads(r["_key"]))])
        out.append(md_table(
            ["#", "config", "stage", "CV macro-F1", "CV seeds", "vs base",
             "CV bal.acc", "CV AUC", "differs from base by"],
            rows, "rlrrrrrrl"))

        # --- which axis moved the needle -----------------------------------
        # restricted to the base CV seed, so every delta is measured against
        # the same fold split
        one = df[df["seed"] == base["seed"]]
        out.append(f"\nPer-axis effect at CV seed {int(base['seed'])} - every "
                   "row that changed only knobs belonging to that axis, "
                   "against the base:\n")
        arows = []
        for axis, knobs in AXIS.items():
            sel = []
            for _, r in one.iterrows():
                d = diff_vs(r, base)
                if d and set(d) <= set(knobs):
                    sel.append(r["cv_macro_f1"] - base["cv_macro_f1"])
            if not sel:
                continue
            sel = np.array(sel)
            arows.append([axis, len(sel), f"{sel.max():+.4f}",
                          f"{sel.min():+.4f}", f"{sel.mean():+.4f}"])
        # rows that mix axes (the combine stage)
        mixed = []
        for _, r in one.iterrows():
            d = diff_vs(r, base)
            if d and not any(set(d) <= set(k) for k in AXIS.values()):
                mixed.append(r["cv_macro_f1"] - base["cv_macro_f1"])
        if mixed:
            mixed = np.array(mixed)
            arows.append(["combinations of two or more axes", len(mixed),
                          f"{mixed.max():+.4f}", f"{mixed.min():+.4f}",
                          f"{mixed.mean():+.4f}"])
        out.append(md_table(["axis", "n", "best delta", "worst delta",
                             "mean delta"], arows, "lrrrr"))
    return "\n".join(out)


# -------------------------------------------------------------------- gap ---
def _test_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def rank_configs(df: pd.DataFrame, base: pd.Series) -> pd.DataFrame:
    """One row per distinct configuration, ranked by MEAN CV macro-F1.

    A configuration re-cross-validated at a second CV seed contributes both
    runs, so a config that only won by drawing a lucky fold split falls back.
    """
    key = [json.dumps(diff_vs(r, base), sort_keys=True, default=str)
           for _, r in df.iterrows()]
    d = df.copy()
    d["_key"] = key
    g = (d.groupby("_key")
           .agg(cv_macro_f1=("cv_macro_f1", "mean"),
                cv_sd=("cv_macro_f1", "std"),
                cv_seeds=("cv_macro_f1", "size"),
                cv_balanced_accuracy=("cv_balanced_accuracy", "mean"),
                cv_auc=("cv_auc", "mean"),
                config_id=("config_id", "first"),
                stage=("stage", "first"))
           .reset_index()
           .sort_values("cv_macro_f1", ascending=False))
    return g


def report_gap() -> str:
    out = []
    for ds in DATASETS:
        df = load_search(ds)
        base = base_row(df)
        best = rank_configs(df, base).head(3)
        rows = []
        for i, (_, r) in enumerate(best.iterrows(), 1):
            # keyed by CV rank, not config_id: the previous run's two search
            # batches both restarted their numbering at 001, so `encoding_002`
            # names two different configurations on `balanced`.
            m = _test_metrics(TUNED / ds / "posthoc" / f"rank{i}" /
                              "metrics.json")
            cv = (f"{r['cv_macro_f1']:.4f}" if r["cv_seeds"] == 1 else
                  f"{r['cv_macro_f1']:.4f} (mean of {int(r['cv_seeds'])} "
                  f"CV seeds)")
            knobs = fmt_diff(json.loads(r["_key"]))
            if m is None:
                rows.append([i, f"`{r['config_id']}`", cv, "-", "-", "-",
                             knobs])
                continue
            res = m["results"][0]
            rows.append([i, f"`{r['config_id']}`", cv,
                         f"{res['macro_f1']:.4f}",
                         f"{res['macro_f1'] - r['cv_macro_f1']:+.4f}",
                         f"{res['roc_auc']:.4f}", knobs])
        out.append(f"\n**`{ds}`**\n")
        out.append(md_table(
            ["CV rank", "config", "CV macro-F1", "test macro-F1 (seed 1)",
             "test - CV", "test AUC", "differs from base by"],
            rows, "rlrrrrl"))
    return "\n".join(out)


# ------------------------------------------------------------------ final ---
def untuned_summary(ds: str) -> dict:
    """mean/sd over the untuned 5-seed sweep in results/cnn_vit/seeds/."""
    res = []
    for d in sorted((UNTUNED_SEEDS / ds).glob("seed*")):
        p = d / "metrics.json"
        if p.exists():
            res.append(json.loads(p.read_text(encoding="utf-8"))["results"][0])
    if not res:
        raise SystemExit(f"no untuned seed metrics under {UNTUNED_SEEDS/ds}")
    return _agg(res)


def tuned_summary(ds: str, sub: str = "") -> dict:
    p = (TUNED / ds / sub / "metrics.json") if sub else (TUNED / ds / "metrics.json")
    if not p.exists():
        raise SystemExit(f"no tuned metrics at {p} - run `tuned_train.py final`")
    doc = json.loads(p.read_text(encoding="utf-8"))
    agg = _agg(doc["results"])
    agg["_doc"] = doc
    return agg


def harness_base_summary(ds: str) -> dict | None:
    """The search's BASE configuration under the tuning harness, 5 seeds.

    The published untuned sweep in results/cnn_vit/seeds/ came from
    `train_eval.py` - fp32, a DataLoader, a different RNG consumption order -
    while the harness runs the same recipe in bf16 off a GPU-resident tensor.
    Same hyper-parameters, different numerics, so this is the column a
    before/after claim has to be made against.
    """
    p = TUNED / ds / "baseline" / "metrics.json"
    if not p.exists():
        return None
    return _agg(json.loads(p.read_text(encoding="utf-8"))["results"])


def _agg(res: list[dict]) -> dict:
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc",
            "recall_goodware", "recall_ransomware", "false_positive_rate",
            "precision_ransomware", "majority_class_accuracy"]
    out = {"n_seeds": len(res)}
    for k in keys:
        v = np.array([r[k] for r in res if r.get(k) is not None], float)
        if len(v):
            out[k] = (float(v.mean()),
                      float(v.std(ddof=1)) if len(v) > 1 else 0.0)
    per_arch = {}
    for a in res[0].get("per_architecture", {}):
        per_arch[a] = {"n": res[0]["per_architecture"][a]["n"]}
        for k in ("balanced_accuracy", "macro_f1", "recall_goodware",
                  "recall_ransomware"):
            v = np.array([r["per_architecture"][a][k] for r in res], float)
            per_arch[a][k] = (float(v.mean()),
                              float(v.std(ddof=1)) if len(v) > 1 else 0.0)
    out["per_architecture"] = per_arch
    fam = {}
    for f in res[0].get("per_family_recall", {}):
        v = np.array([r["per_family_recall"][f]["recall"] for r in res], float)
        fam[f] = (float(v.mean()),
                  float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                  res[0]["per_family_recall"][f]["support"])
    out["per_family_recall"] = fam
    return out


def _pm(t) -> str:
    return f"{t[0]:.3f} ± {t[1]:.3f}"


def report_final() -> str:
    out = []
    labels = [("macro_f1", "**macro F1**"),
              ("balanced_accuracy", "balanced accuracy"),
              ("accuracy", "accuracy"),
              ("roc_auc", "ROC AUC"),
              ("recall_goodware", "recall goodware"),
              ("recall_ransomware", "recall ransomware"),
              ("false_positive_rate", "false positive rate")]

    U = {ds: untuned_summary(ds) for ds in DATASETS}
    B = {ds: harness_base_summary(ds) for ds in DATASETS}
    T = {ds: tuned_summary(ds) for ds in DATASETS}
    cols = [("untuned", U)]
    if all(B.values()):
        cols.append(("base (harness)", B))
    cols.append(("tuned", T))

    head = ["metric"]
    for ds in DATASETS:
        head += [f"`{ds}` {c}" for c, _ in cols]
    rows = []
    for k, lab in labels:
        r = [lab]
        for ds in DATASETS:
            r += [_pm(src[ds][k]) for _, src in cols]
        rows.append(r)
    floor = ["**majority-class accuracy (floor)**"]
    for ds in DATASETS:
        floor += [f"**{U[ds]['majority_class_accuracy'][0]:.3f}**"] * len(cols)
    rows.append(floor)
    out.append("Test fold, mean ± sd over 5 seeds each. `untuned` is the "
               "published `train_eval.py` sweep in `results/cnn_vit/seeds/`; "
               "`base (harness)` is the *same recipe* re-run through the "
               "tuning harness (bf16, GPU-resident batching) and is the "
               "configuration every search row was varied from.\n")
    out.append(md_table(head, rows, "l" + "r" * (len(head) - 1)))

    # --- per architecture -------------------------------------------------
    out.append("\nPer architecture (test fold, mean ± sd over 5 seeds). "
               "`balanced accuracy` here is *within* the architecture, so 0.5 "
               "is chance no matter how the architecture mix is skewed.\n")
    arows = []
    for ds in DATASETS:
        for a in sorted(U[ds]["per_architecture"]):
            r = [f"`{ds}`", a, U[ds]["per_architecture"][a]["n"]]
            for _, src in cols:
                r.append(_pm(src[ds]["per_architecture"][a]["balanced_accuracy"]))
            for _, src in cols:
                r.append(_pm(src[ds]["per_architecture"][a]["macro_f1"]))
            arows.append(r)
    ahead = (["dataset", "arch", "n"]
             + [f"{c} bal.acc" for c, _ in cols]
             + [f"{c} macro-F1" for c, _ in cols])
    out.append(md_table(ahead, arows, "ll" + "r" * (len(ahead) - 2)))

    # --- per family -------------------------------------------------------
    out.append("\nPer ransomware test family, recall (mean ± sd over 5 "
               "seeds). All 14 families are disjoint from the 24 training "
               "families.\n")
    fam_u = U["mendeley"]["per_family_recall"]
    order = sorted(fam_u, key=lambda f: -fam_u[f][2])
    frows = []
    for f in order:
        r = [f, fam_u[f][2]]
        for ds in DATASETS:
            r += [_pm(U[ds]["per_family_recall"][f][:2]),
                  _pm(T[ds]["per_family_recall"][f][:2])]
        frows.append(r)
    fhead = ["family", "support"]
    for ds in DATASETS:
        fhead += [f"`{ds}` untuned", f"`{ds}` tuned"]
    out.append(md_table(fhead, frows, "lr" + "r" * (len(fhead) - 2)))

    for ds in DATASETS:
        cfg = T[ds]["_doc"]["config"]
        out.append(f"\nChosen for `{ds}`: encoding `{cfg['encoding']['variant']}`"
                   f"{' + row_align' if cfg['encoding']['row_align'] else ''}, "
                   + ", ".join(f"`{k}={_short(v)}`"
                               for k, v in cfg["training"].items()
                               if k in ("lr", "weight_decay", "batch_size",
                                        "schedule", "label_smoothing", "loss",
                                        "sampler"))
                   + f", width {cfg['model']['width']} depth "
                     f"{cfg['model']['depth']} dropout {cfg['model']['dropout']}"
                     f", decision `{cfg['decision_rule']['threshold']}`.")
    return "\n".join(out)


# ------------------------------------------------------------------- main ---
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["search", "gap", "final", "all"],
                    default="all", nargs="?")
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    if a.what in ("search", "all"):
        print("### Search\n")
        print(report_search(a.top))
    if a.what in ("gap", "all"):
        print("\n### CV -> test gap (post hoc)\n")
        print(report_gap())
    if a.what in ("final", "all"):
        print("\n### Before / after\n")
        print(report_final())
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
