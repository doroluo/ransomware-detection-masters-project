#!/usr/bin/env python3
"""
run_stacker.py - a LEARNED two-member combiner under family holdout, fitted
by cross-validated stacking over the members' held-out scores.

    python family_holdout/run_stacker.py --dataset both \
        --members tfidf/LogReg,seq_transformer/seq_transformer_imports

What it fits. Three numbers: a logistic regression over the two members'
logit scores, `sigmoid(w1*logit(p1) + w2*logit(p2) + b)`, decided at 0.5.
Nothing else - no architecture feature (that would be a bitness detector),
no per-family anything, no C search (C=1, three parameters cannot overfit
2,000 rows in any way a C grid would fix).

How it is evaluated, and the one caveat to carry.
    K-fold   for outer fold f: fit the combiner on the members' K-fold
             held-out scores of the OTHER four folds, apply it to fold f's
             held-out scores. Fold f's own scores were produced by member
             models that never saw fold f, so the test rows are clean. The
             combiner's TRAINING rows, however, carry scores from member
             models whose training folds included fold f. That is the
             standard cross-validated-stacking approximation; the strictly
             nested version retrains every member inside every outer fold
             (five more transformer studies, about 25 GPU-hours) and was
             not done. With three parameters the leak has almost nowhere to
             go, and the oracle bound in summary_ensemble*.md says how much
             there was to gain in the first place.
    LOFO     for family F: fit the combiner on the K-fold held-out scores of
             every file NOT in F (all goodware, the other 37 families), apply
             it to F's LOFO member scores. Same caveat, same reading: a
             recall study only.

Output: results/family_holdout/<dataset>/stacker/<member1>+<member2>_logitlr/
(the standard six files, with the per-fold weights recorded in metrics.json)
and results/family_holdout/summary_stacker<suffix>.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from family_holdout.common import DATASETS, Folds, OUT_ROOT, write_model_dir  # noqa: E402, ALL_DATASETS
from family_holdout.run_ensemble import MEMBERS as DEFAULT_MEMBERS  # noqa: E402
from family_holdout.run_ensemble import _member_metrics, _fm, load_member  # noqa: E402

PIPELINE = "stacker"
EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def features(scores: np.ndarray) -> np.ndarray:
    """scores (n_members, n) -> (n, n_members) logit features."""
    return logit(np.asarray(scores)).T


def fit(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0, max_iter=1000).fit(X, y)


def run_dataset(dataset: str, out_root: Path, members) -> dict:
    t0 = time.time()
    folds = Folds(dataset)
    y = folds.y
    kf, lofo = [], []
    for p, m in members:
        sc, lo = load_member(folds, out_root, p, m)
        kf.append(sc); lofo.append(lo)
    X = features(np.vstack(kf))                       # (n, 2) held-out logits

    score = np.zeros(folds.n)
    weights = {}
    for f in range(5):
        tr, te = folds.fold != f, folds.fold == f
        est = fit(X[tr], y[tr])
        score[te] = est.predict_proba(X[te])[:, 1]
        weights[str(f)] = {"w": [round(float(w), 4) for w in est.coef_[0]],
                           "b": round(float(est.intercept_[0]), 4)}
    pred = (score >= 0.5).astype(int)

    lofo_out, lofo_w = {}, {}
    for fam in folds.families:
        idx = np.flatnonzero((folds.family == fam) & (y == 1))
        tr = ~((folds.family == fam) & (y == 1))
        est = fit(X[tr], y[tr])
        Xf = features(np.vstack([lo[fam] for lo in lofo]))
        s = est.predict_proba(Xf)[:, 1]
        lofo_out[fam] = (s, (s >= 0.5).astype(int))
        lofo_w[fam] = [round(float(w), 4) for w in est.coef_[0]]

    tag = "+".join(f"{p}_{m}".lower() for p, m in members)
    cfg = {"pipeline": PIPELINE, "model": f"{tag}_logitlr",
           "members": [{"pipeline": p, "model": m} for p, m in members],
           "combiner": "LogisticRegression(C=1) over [logit(p_member1), logit(p_member2)], decision 0.5",
           "protocol": {"kfold": "fit on the other four folds' held-out member scores, apply to the held-out fold",
                        "lofo": "fit on the K-fold held-out scores of every file outside the family, apply to the family's LOFO member scores",
                        "caveat": "cross-validated stacking: the combiner's training rows carry member scores from models whose training data included the test fold; members were not retrained per outer fold"},
           "kfold_weights": weights, "lofo_weights": lofo_w, "pre_registered": True}
    d = out_root / dataset / PIPELINE / f"{tag}_logitlr"
    res = write_model_dir(d, folds, PIPELINE, f"{tag}_logitlr", score, pred, folds.fold.copy(),
                          lofo_out, cfg, time.time() - t0,
                          description=f"learned logit combiner of {tag}, cross-validated stacking, family holdout on {dataset}")
    print(f"  [{dataset}] stacker: fold macro-F1 {res['fold_mean']['macro_f1']:.3f} +/- "
          f"{res['fold_sd']['macro_f1']:.3f}, LOFO {res['lofo_mean_recall']:.3f}; weights "
          f"{[weights[k]['w'] for k in sorted(weights)]} -> {d}", flush=True)
    return {"tag": tag, "weights": weights}


def write_summary(out_root: Path, results: dict, members, suffix: str) -> Path:
    L = ["# Learned combiner (cross-validated stacking) under family holdout", "",
         "Logistic regression over the two members' logit scores, three parameters, decided at 0.5. K-fold: fitted on the",
         "other four folds' held-out member scores. LOFO: fitted on every file outside the family. Caveat: the combiner's",
         "training rows carry member scores from models whose training data included the test fold (members were not",
         "retrained per outer fold), so this is the standard cross-validated-stacking approximation, not nested stacking.",
         "Written by `family_holdout/run_stacker.py`.", ""]
    for ds, r in results.items():
        tag = r["tag"]
        rows = [(f"{p} / {m}", _member_metrics(out_root, ds, p, m)) for p, m in members]
        for rule in ("mean", "max"):
            d = out_root / ds / "ensemble" / f"{tag}_{rule}"
            if (d / "metrics.json").is_file():
                rows.append((f"ensemble / {rule} (fixed rule)", _member_metrics(out_root, ds, "ensemble", f"{tag}_{rule}")))
        rows.append(("stacker / logit LR (learned)", _member_metrics(out_root, ds, PIPELINE, f"{tag}_logitlr")))
        L += [f"## Dataset: {ds}", "",
              "| model | folds macro-F1 mean +/- sd | pooled macro-F1 | pooled AUC | recall R / G | FPR | LOFO mean recall |",
              "|---|---|---|---|---|---|---|"]
        for name, m in rows:
            L.append(f"| {name} | {_fm(m['fold_mean'])} +/- {_fm(m['fold_sd'])} | {_fm(m['pooled'])} | {_fm(m['auc'])} | "
                     f"{_fm(m['rr'], 2)} / {_fm(m['rg'], 2)} | {_fm(m['fpr'])} | {_fm(m['lofo'])} |")
        w = r["weights"]
        L += ["", "Per-fold weights (w on member 1, w on member 2, bias): " +
              "; ".join(f"fold {k}: {w[k]['w'][0]:.2f}, {w[k]['w'][1]:.2f}, {w[k]['b']:.2f}" for k in sorted(w)), ""]
    p = out_root / f"summary_stacker{suffix}.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=(*ALL_DATASETS, "both"), default="both")
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--members", default=",".join(f"{p}/{m}" for p, m in DEFAULT_MEMBERS))
    ap.add_argument("--summary-suffix", default="")
    a = ap.parse_args()
    members = tuple(tuple(x.split("/", 1)) for x in a.members.split(","))
    if len(members) != 2 or any(len(m) != 2 for m in members):
        raise SystemExit("--members needs exactly two pipeline/model entries")
    out_root = Path(a.out)
    ds = DATASETS if a.dataset == "both" else (a.dataset,)
    results = {d: run_dataset(d, out_root, members) for d in ds}
    if a.dataset == "both":
        print(f"wrote {write_summary(out_root, results, members, a.summary_suffix)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
