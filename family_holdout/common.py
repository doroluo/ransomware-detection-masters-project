#!/usr/bin/env python3
"""Shared plumbing for the family-holdout runners.

`folds.py` decides WHICH file is in which fold; this module decides what a
runner has to write once it has scored them, so the three pipelines'
`results/family_holdout/<dataset>/<pipeline>/<model>/` directories are
literally the same shape and the coordinator can aggregate them blind.

Two schemes, both derived from the one fold file (see folds.py's docstring):

    K-fold   for f in 0..4: fit on the other four folds, score fold f. Every
             file gets exactly one held-out prediction, so the pooled
             predictions are a complete cross-validated pass over the cohort.
    LOFO     for each ransomware family: fit on every other ransomware family
             plus ALL goodware, score that family alone. No goodware in the
             test set, so this is a recall study only - no FPR, no AUC.

Floors, per test fold, from the fold file's own arch/label columns:
    majority   predict the larger class of that test fold everywhere
    x86_rule   predict ransomware iff the file is x86

The x64 caveat this whole study has to carry: x64 ransomware is concentrated
in fold 2 (59 of the 114 x64 ransomware files; Hive alone is 43 of them), so
fold 2's x64 numbers are the only ones with real support and the per-fold
spread of any x64 metric is not a sampling spread.
"""
from __future__ import annotations

import csv
import os
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cnn_vit_pipeline.cohort import (_core_metrics, _safe_div,  # noqa: E402
                                     build_result, write_metrics)

K = 5
DATASETS = ("mendeley", "balanced")          # what "--dataset both" means
ALL_DATASETS = (*DATASETS, "all")            # "all": every ransomware corpus vs every goodware source
# RANSOM_FH_DIR points a run at another fold set (e.g. results/family_holdout_v2)
# without touching the committed one.
FOLD_DIR = Path(os.environ.get("RANSOM_FH_DIR", REPO / "results" / "family_holdout"))
OUT_ROOT = FOLD_DIR

# Where each corpus's extraction outputs live (asm/, mn/, mn_api/ ... under one
# folder per corpus). Rows in the fold file carry `corpus`, so runners route by
# corpus instead of by (dataset, label).
SHARED = Path(os.environ.get("RANSOM_SHARED_DIR", r"C:/Users/chaoa/Downloads/asm and mm/Shared"))
TREE_OF_CORPUS = {
    "mendeley": SHARED / "Extract",
    "balanced": SHARED / "Extract_Goodware_Balanced",
    "vs": SHARED / "Extract_VS",
    "hostgood": SHARED / "Extract_Goodware_HostX86",
}


def stream_path(corpus: str, sha: str, stream: str = "mn", ext: str = ".txt") -> Path:
    """The token-stream file of one sample: <corpus tree>/<stream>/<sha><ext>."""
    try:
        return TREE_OF_CORPUS[corpus] / stream / f"{sha}{ext}"
    except KeyError:
        raise KeyError(f"no extraction tree registered for corpus {corpus!r}; "
                       f"add it to common.TREE_OF_CORPUS") from None

FOLD_METRIC_COLS = [
    "fold", "n_test", "n_good", "n_rans", "accuracy", "balanced_accuracy",
    "macro_f1", "roc_auc", "recall_ransomware", "recall_goodware", "fpr",
    "x86_recall_ransomware", "x86_recall_goodware", "x64_recall_ransomware",
    "x64_recall_goodware", "majority_floor", "x86_rule_floor",
]
FOLD_MEAN_KEYS = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc")


# ---------------------------------------------------------------------------
# the fold file
# ---------------------------------------------------------------------------
class Folds:
    """The fold table for one dataset, as plain aligned numpy arrays."""

    def __init__(self, dataset: str, fold_dir: Path = FOLD_DIR):
        path = Path(fold_dir) / f"folds_{dataset}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; run python family_holdout/folds.py --out "
                f"results/family_holdout")
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.dataset = dataset
        self.path = path
        self.sha = np.array([r["sha256"] for r in rows], dtype=object)
        self.y = np.array([int(r["label"]) for r in rows], dtype=int)
        self.family = np.array([r["family"] for r in rows], dtype=object)
        self.group = np.array([r["group"] for r in rows], dtype=object)
        self.arch = np.array([r["arch"] for r in rows], dtype=object)
        self.orig_set = np.array([r["orig_set"] for r in rows], dtype=object)
        self.fold = np.array([int(r["fold"]) for r in rows], dtype=int)
        self.corpus = np.array([r.get("corpus") or "mendeley" for r in rows], dtype=object)
        self.n = len(rows)
        self.families = sorted({f for f, l in zip(self.family, self.y) if l == 1})
        if len(set(self.sha)) != self.n:
            raise ValueError(f"{path}: duplicate sha256 rows; rebuild the fold file")

    # -- the two schemes ---------------------------------------------------
    def kfold(self):
        """[(fold, train_idx, test_idx)] for f in 0..K-1."""
        out = []
        for f in range(K):
            te = np.flatnonzero(self.fold == f)
            tr = np.flatnonzero(self.fold != f)
            out.append((f, tr, te))
        return out

    def lofo(self):
        """[(family, train_idx, test_idx)] - test is exactly that family."""
        out = []
        for fam in self.families:
            te = np.flatnonzero((self.family == fam) & (self.y == 1))
            tr = np.flatnonzero((self.family != fam) | (self.y == 0))
            out.append((fam, tr, te))
        return out

    def fold_of_family(self) -> dict:
        rans = self.y == 1
        return {f: int(self.fold[np.flatnonzero((self.family == f) & rans)[0]])
                for f in self.families}

    def counts(self) -> dict:
        rans = self.y == 1
        fam_n = {f: int(((self.family == f) & rans).sum()) for f in self.families}
        fam_x64 = {f: int(((self.family == f) & rans & (self.arch == "x64")).sum())
                   for f in self.families}
        return fam_n, fam_x64


# ---------------------------------------------------------------------------
# floors
# ---------------------------------------------------------------------------
def floor_blocks(y_true, arch, y_train=None) -> dict:
    """majority and x86-rule, as full metric blocks, for one test set.

    The majority class comes from y_train when given (a real baseline that
    knows only the training labels); without it the test set's own majority is
    used, which is an oracle and slightly optimistic."""
    y_true = np.asarray(y_true).astype(int)
    arch = np.asarray(arch, dtype=object)
    src = y_true if y_train is None else np.asarray(y_train).astype(int)
    maj = int(np.bincount(src, minlength=2).argmax())
    out = {"majority": _core_metrics(y_true, np.full(len(y_true), maj)),
           "x86_rule": _core_metrics(y_true, (arch == "x86").astype(int))}
    keep = ("accuracy", "balanced_accuracy", "macro_f1", "recall_goodware",
            "recall_ransomware", "false_positive_rate")
    return {k: {kk: (None if vv is None else round(vv, 4)) for kk, vv in v.items() if kk in keep}
            for k, v in out.items()}


def _recall(y_true, y_pred, mask, label) -> float:
    m = np.asarray(mask) & (y_true == label)
    if not m.sum():
        return float("nan")
    hit = (y_pred[m] == label).sum()
    return _safe_div(int(hit), int(m.sum()))


def fold_row(fold, y_true, y_pred, y_score, arch, y_train=None) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    arch = np.asarray(arch, dtype=object)
    m = _core_metrics(y_true, y_pred, y_score)
    fl = floor_blocks(y_true, arch, y_train)
    is86, is64 = arch == "x86", arch == "x64"
    return {
        "fold": fold,
        "n_test": int(len(y_true)),
        "n_good": int((y_true == 0).sum()),
        "n_rans": int((y_true == 1).sum()),
        "accuracy": round(m["accuracy"], 6),
        "balanced_accuracy": round(m["balanced_accuracy"], 6),
        "macro_f1": round(m["macro_f1"], 6),
        "roc_auc": ("" if m["roc_auc"] is None else round(m["roc_auc"], 6)),
        "recall_ransomware": round(m["recall_ransomware"], 6),
        "recall_goodware": round(m["recall_goodware"], 6),
        "fpr": round(m["false_positive_rate"], 6),
        "x86_recall_ransomware": _rnd(_recall(y_true, y_pred, is86, 1)),
        "x86_recall_goodware": _rnd(_recall(y_true, y_pred, is86, 0)),
        "x64_recall_ransomware": _rnd(_recall(y_true, y_pred, is64, 1)),
        "x64_recall_goodware": _rnd(_recall(y_true, y_pred, is64, 0)),
        "majority_floor": fl["majority"]["accuracy"],
        "x86_rule_floor": fl["x86_rule"]["accuracy"],
    }


def _rnd(v):
    return "" if v != v else round(float(v), 6)       # NaN -> ""


# ---------------------------------------------------------------------------
# writing one model directory
# ---------------------------------------------------------------------------
def _write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _mean_sd(vals):
    v = np.array([x for x in vals if x == x and x != ""], dtype=float)
    if not len(v):
        return None, None
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def write_model_dir(out_dir: Path, folds: Folds, pipeline: str, model: str,
                    kfold_score: np.ndarray, kfold_pred: np.ndarray,
                    kfold_fold: np.ndarray, lofo, config: dict,
                    elapsed: float, description: str = "", **extra) -> dict:
    """Write the six files of one `<dataset>/<pipeline>/<model>/` directory.

    kfold_score/pred/fold are aligned with `folds` row order (one held-out
    prediction per file). `lofo` maps family -> (score array, pred array)
    aligned with that family's rows in `folds` order.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y, arch, fam, sha = folds.y, folds.arch, folds.family, folds.sha

    # -- predictions.csv --------------------------------------------------
    _write_csv(out_dir / "predictions.csv",
               ["sha256", "label", "family", "arch", "fold", "score", "pred"],
               [{"sha256": sha[i], "label": int(y[i]), "family": fam[i],
                 "arch": arch[i], "fold": int(kfold_fold[i]),
                 "score": float(kfold_score[i]), "pred": int(kfold_pred[i])}
                for i in range(folds.n)])

    # -- fold_metrics.csv -------------------------------------------------
    frows = []
    for f in range(K):
        m = kfold_fold == f
        frows.append(fold_row(f, y[m], kfold_pred[m], kfold_score[m], arch[m], y_train=y[~m]))
    _write_csv(out_dir / "fold_metrics.csv", FOLD_METRIC_COLS, frows)

    # -- lofo_predictions.csv ---------------------------------------------
    lrows, lofo_recall = [], {}
    for famname in folds.families:
        idx = np.flatnonzero((fam == famname) & (y == 1))
        sc, pr = lofo[famname]
        for j, i in enumerate(idx):
            lrows.append({"sha256": sha[i], "family": famname, "arch": arch[i],
                          "score": float(sc[j]), "pred": int(pr[j])})
        lofo_recall[famname] = _safe_div(int((np.asarray(pr) == 1).sum()),
                                         len(idx))
    _write_csv(out_dir / "lofo_predictions.csv",
               ["sha256", "family", "arch", "score", "pred"], lrows)

    # -- per_family.csv ---------------------------------------------------
    fam_n, fam_x64 = folds.counts()
    fold_of = folds.fold_of_family()
    prows = []
    for famname in sorted(folds.families):
        idx = np.flatnonzero((fam == famname) & (y == 1))
        rk = _safe_div(int((kfold_pred[idx] == 1).sum()), len(idx))
        prows.append({"family": famname, "n": fam_n[famname],
                      "n_x64": fam_x64[famname], "fold": fold_of[famname],
                      "corpus": "+".join(sorted(set(folds.corpus[idx]))),
                      "recall_kfold": round(rk, 6),
                      "recall_lofo": round(lofo_recall[famname], 6)})
    _write_csv(out_dir / "per_family.csv",
               ["family", "n", "n_x64", "fold", "corpus", "recall_kfold", "recall_lofo"],
               prows)

    # -- metrics.json -----------------------------------------------------
    pooled = build_result(y, kfold_pred, kfold_score, arch, fam,
                          pipeline=pipeline, model=model,
                          scheme="kfold_pooled", n_configs=1)
    stats = {}
    for k in FOLD_MEAN_KEYS:
        mu, sd = _mean_sd([r[k] for r in frows])
        stats[f"fold_mean_{k}"] = None if mu is None else round(mu, 6)
        stats[f"fold_sd_{k}"] = None if sd is None else round(sd, 6)
    pooled["fold_mean"] = {k: stats[f"fold_mean_{k}"] for k in FOLD_MEAN_KEYS}
    pooled["fold_sd"] = {k: stats[f"fold_sd_{k}"] for k in FOLD_MEAN_KEYS}
    pooled["lofo_mean_recall"] = round(
        float(np.mean([lofo_recall[f] for f in folds.families])), 6)
    pooled["lofo_weighted_recall"] = round(_safe_div(
        sum(lofo_recall[f] * fam_n[f] for f in folds.families),
        sum(fam_n.values())), 6)
    pooled["lofo_recall_by_family"] = {f: round(lofo_recall[f], 6)
                                       for f in folds.families}
    pooled["config"] = config

    samples = {
        "total": folds.n,
        "goodware": int((y == 0).sum()),
        "ransomware": int((y == 1).sum()),
        "families": len(folds.families),
        "folds": {str(f): {"n": int((folds.fold == f).sum()),
                           "goodware": int(((folds.fold == f) & (y == 0)).sum()),
                           "ransomware": int(((folds.fold == f) & (y == 1)).sum()),
                           "x64_ransomware": int(((folds.fold == f) & (y == 1)
                                                  & (arch == "x64")).sum())}
                  for f in range(K)},
        "fold_definition": str(folds.path.relative_to(REPO)),
    }
    write_metrics(
        out_dir / "metrics.json",
        experiment=f"family_holdout/{folds.dataset}/{pipeline}/{model}",
        description=description or f"{pipeline} {model}, family-holdout K-fold "
                                   f"+ LOFO on the {folds.dataset} cohort",
        samples=samples, results=[pooled], elapsed_seconds=elapsed,
        floors={"pooled": floor_blocks(y, arch),
                "per_fold": {str(f): floor_blocks(y[folds.fold == f],
                                                  arch[folds.fold == f],
                                                  y_train=y[folds.fold != f])
                             for f in range(K)}},
        n_configs=1, config=config,
        # also at the top level: family_holdout/aggregate.py reads them there
        lofo_mean_recall=pooled["lofo_mean_recall"],
        lofo_weighted_recall=pooled["lofo_weighted_recall"],
        caveat=("x64 ransomware concentrates in fold 2: 59 of the 114 x64 "
                "ransomware files, of which Hive is 43. Per-fold x64 numbers "
                "outside fold 2 rest on a handful of files."),
        **extra)

    # -- config_used.yaml -------------------------------------------------
    (out_dir / "config_used.yaml").write_text(_yaml(config), encoding="utf-8")
    return pooled


def _yaml(obj, indent=0) -> str:
    """Tiny YAML emitter: the runners' configs are dicts of scalars, lists and
    dicts, and pyyaml is not guaranteed present under every interpreter here."""
    try:
        import yaml
        return yaml.safe_dump(obj, sort_keys=False, default_flow_style=False)
    except Exception:
        pass
    pad = "  " * indent
    if isinstance(obj, dict):
        L = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                L.append(f"{pad}{k}:")
                L.append(_yaml(v, indent + 1).rstrip("\n"))
            else:
                L.append(f"{pad}{k}: {json.dumps(v)}")
        return "\n".join(L) + "\n"
    if isinstance(obj, list):
        L = []
        for v in obj:
            if isinstance(v, (dict, list)) and v:
                L.append(f"{pad}-")
                L.append(_yaml(v, indent + 1).rstrip("\n"))
            else:
                L.append(f"{pad}- {json.dumps(v)}")
        return "\n".join(L) + "\n"
    return f"{pad}{json.dumps(obj)}\n"


# ---------------------------------------------------------------------------
# invariants the runners assert before they spend an hour
# ---------------------------------------------------------------------------
def check_kfold(folds: Folds) -> dict:
    """Train/test disjoint by sha, by family and by goodware group, per fold."""
    report = {}
    for f, tr, te in folds.kfold():
        s_tr, s_te = set(folds.sha[tr]), set(folds.sha[te])
        fam_tr = {x for x, l in zip(folds.family[tr], folds.y[tr]) if l == 1}
        fam_te = {x for x, l in zip(folds.family[te], folds.y[te]) if l == 1}
        g_tr = {x for x, l in zip(folds.group[tr], folds.y[tr]) if l == 0}
        g_te = {x for x, l in zip(folds.group[te], folds.y[te]) if l == 0}
        report[f] = {"sha_overlap": len(s_tr & s_te),
                     "family_overlap": sorted(fam_tr & fam_te),
                     "goodware_group_overlap": len(g_tr & g_te)}
        assert not s_tr & s_te, f"fold {f}: sha overlap"
        assert not fam_tr & fam_te, f"fold {f}: family overlap {fam_tr & fam_te}"
        assert not g_tr & g_te, f"fold {f}: goodware group overlap"
    return report


def check_lofo(folds: Folds) -> dict:
    report = {}
    for fam, tr, te in folds.lofo():
        assert set(folds.family[te]) == {fam}, f"LOFO {fam}: test not the family"
        assert set(folds.y[te]) == {1}, f"LOFO {fam}: goodware in test"
        assert fam not in set(folds.family[tr]), f"LOFO {fam}: family in train"
        assert int((folds.y[tr] == 0).sum()) == int((folds.y == 0).sum()), \
            f"LOFO {fam}: goodware dropped from train"
        assert not set(folds.sha[tr]) & set(folds.sha[te])
        report[fam] = {"n_test": int(len(te)), "n_train": int(len(tr))}
    return report
