#!/usr/bin/env python3
"""
run_ensemble.py - a fixed-rule ensemble of two already-scored pipelines under
family holdout, built ONLY from their held-out predictions.

    python family_holdout/run_ensemble.py --dataset both

Why. Under family holdout the mnemonic TF-IDF baseline and the mnemonic
sequence transformer fail on DIFFERENT families (results/family_holdout/
summary.md, per-family matrix): TF-IDF misses Makop and Stop on mendeley,
the transformer misses Maze and WastedLocker in K-fold, and so on. Whether
that difference is worth anything is a one-line question - average the two
scores and re-score the same held-out rows - and it needs no training, no
GPU and no new split.

What is pre-registered (written before any ensemble row was scored):

    members   tfidf/LogReg              P(ransomware) from predict_proba
              seq_transformer/seq_transformer   seed-mean P(ransomware)
    rules     mean   score = (p_tfidf + p_seq) / 2,  pred = score >= 0.5   PRIMARY
              max    score = max(p_tfidf, p_seq),    pred = score >= 0.5   secondary,
                     the recall-oriented OR rule; it can only raise recall
                     and can only raise the false-positive rate
    no weights, no fitted threshold, no stacking. Both members are the
    pre-registered configurations, so nothing in this file was chosen by
    looking at a test fold, and there is nothing here that COULD be.

Both members already carry one held-out prediction per file (K-fold) and one
LOFO prediction per ransomware file, so the ensemble's K-fold and LOFO rows
are exactly as leakage-free as the members'. Alignment is by sha256 against
the fold file; label, family, arch and fold are asserted equal to the fold
file for every row, so a stale member directory fails loudly.

Output: results/family_holdout/<dataset>/ensemble/<member1>+<member2>_<rule>/
in the same six-file shape as every other runner (family_holdout/common.py),
plus results/family_holdout/summary_ensemble.md with the members, the two
rules, the per-family rows where the ensemble differs from BOTH members, and
the oracle ("either member catches it") upper bound that says how much
complementarity there was to harvest in the first place.
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from family_holdout.common import (DATASETS, Folds, OUT_ROOT,  # noqa: E402
                                   write_model_dir)

PIPELINE = "ensemble"
MEMBERS = (("tfidf", "LogReg"), ("seq_transformer", "seq_transformer"))
RULES = ("mean", "max")
PRIMARY = "mean"
THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# the two rules
# ---------------------------------------------------------------------------
def combine(scores: np.ndarray, rule: str) -> np.ndarray:
    """scores: (n_members, n) matrix of P(ransomware). Returns one score per row."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores must be (n_members, n)")
    if rule == "mean":
        return scores.mean(axis=0)
    if rule == "max":
        return scores.max(axis=0)
    raise ValueError(f"unknown rule {rule!r}")


def decide(score: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    return (np.asarray(score, dtype=float) >= threshold).astype(int)


# ---------------------------------------------------------------------------
# loading a member's held-out predictions, aligned to the fold file
# ---------------------------------------------------------------------------
def _read(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_member(folds: Folds, out_root: Path, pipeline: str, model: str):
    """Returns (kfold_score[n], lofo: {family: score[len(family)]}) aligned to
    `folds` row order. Asserts label/family/arch/fold agree with the fold file
    for every row and that every ransomware file has a LOFO score."""
    d = Path(out_root) / folds.dataset / pipeline / model
    if not (d / "predictions.csv").is_file():
        raise FileNotFoundError(f"{d / 'predictions.csv'} not found")
    rows = {r["sha256"]: r for r in _read(d / "predictions.csv")}
    if set(rows) != set(folds.sha):
        missing = set(folds.sha) - set(rows)
        extra = set(rows) - set(folds.sha)
        raise ValueError(f"{d}: predictions do not cover the fold file "
                         f"(missing {len(missing)}, extra {len(extra)})")
    score = np.zeros(folds.n)
    for i, sha in enumerate(folds.sha):
        r = rows[sha]
        assert int(r["label"]) == int(folds.y[i]), f"{d}: label mismatch {sha}"
        assert r["family"] == folds.family[i], f"{d}: family mismatch {sha}"
        assert r["arch"] == folds.arch[i], f"{d}: arch mismatch {sha}"
        assert int(r["fold"]) == int(folds.fold[i]), f"{d}: fold mismatch {sha}"
        score[i] = float(r["score"])

    lrows = {r["sha256"]: r for r in _read(d / "lofo_predictions.csv")}
    lofo = {}
    for fam in folds.families:
        idx = np.flatnonzero((folds.family == fam) & (folds.y == 1))
        sc = np.zeros(len(idx))
        for j, i in enumerate(idx):
            r = lrows.get(folds.sha[i])
            if r is None:
                raise ValueError(f"{d}: no LOFO score for {folds.sha[i]} ({fam})")
            assert r["family"] == fam
            sc[j] = float(r["score"])
        lofo[fam] = sc
    return score, lofo


def _pred_of_member(folds: Folds, out_root: Path, pipeline: str, model: str):
    """The member's OWN pred column (its own decision rule), K-fold and LOFO."""
    d = Path(out_root) / folds.dataset / pipeline / model
    rows = {r["sha256"]: int(r["pred"]) for r in _read(d / "predictions.csv")}
    kf = np.array([rows[s] for s in folds.sha], dtype=int)
    lrows = {r["sha256"]: int(r["pred"]) for r in _read(d / "lofo_predictions.csv")}
    lofo = {}
    for fam in folds.families:
        idx = np.flatnonzero((folds.family == fam) & (folds.y == 1))
        lofo[fam] = np.array([lrows[folds.sha[i]] for i in idx], dtype=int)
    return kf, lofo


# ---------------------------------------------------------------------------
# one dataset
# ---------------------------------------------------------------------------
def run_dataset(dataset: str, out_root: Path, members=MEMBERS) -> dict:
    t0 = time.time()
    folds = Folds(dataset)
    names = [f"{p}/{m}" for p, m in members]
    print(f"[{dataset}] members: {', '.join(names)}", flush=True)

    kf_scores, lofo_scores, member_preds = [], [], []
    for p, m in members:
        sc, lo = load_member(folds, out_root, p, m)
        kf_scores.append(sc)
        lofo_scores.append(lo)
        member_preds.append(_pred_of_member(folds, out_root, p, m))
    kf_scores = np.vstack(kf_scores)

    # oracle and disagreement, K-fold: how much complementarity exists
    y = folds.y
    preds = np.vstack([mp[0] for mp in member_preds])
    disagree = preds[0] != preds[1]
    oracle_pred = np.where(y == 1, preds.max(axis=0), preds.min(axis=0))
    oracle = {
        "disagreement_rate": round(float(disagree.mean()), 6),
        "n_disagree": int(disagree.sum()),
        "oracle_recall_ransomware": round(float(oracle_pred[y == 1].mean()), 6),
        "oracle_recall_goodware": round(float(1 - oracle_pred[y == 0].mean()), 6),
        "member_recall_ransomware": [round(float(preds[k][y == 1].mean()), 6)
                                     for k in range(len(members))],
        "member_recall_goodware": [round(float(1 - preds[k][y == 0].mean()), 6)
                                   for k in range(len(members))],
        "note": ("oracle = a per-file choice of whichever member is right; an "
                 "upper bound on any combination rule, not a result"),
    }

    out = {}
    tag = "+".join(f"{p}_{m}".lower() for p, m in members)
    for rule in RULES:
        sc = combine(kf_scores, rule)
        pr = decide(sc)
        lofo = {}
        for fam in folds.families:
            s = combine(np.vstack([lo[fam] for lo in lofo_scores]), rule)
            lofo[fam] = (s, decide(s))
        cfg = {
            "pipeline": PIPELINE,
            "model": f"{tag}_{rule}",
            "members": [{"pipeline": p, "model": m,
                         "source": f"results/family_holdout/{dataset}/{p}/{m}/"
                                   "{predictions,lofo_predictions}.csv",
                         "score": ("predict_proba P(ransomware)" if p == "tfidf"
                                   else "seed-mean P(ransomware), 3 seeds K-fold / 1 seed LOFO")}
                        for p, m in members],
            "rule": rule,
            "rule_definition": ("score = mean of member P(ransomware)" if rule == "mean"
                                else "score = max of member P(ransomware)"),
            "decision": f"score >= {THRESHOLD} (fixed; no threshold fitted anywhere)",
            "primary": rule == PRIMARY,
            "pre_registered": True,
            "trained": False,
            "oracle_kfold": oracle,
            "schemes": ["kfold", "lofo"],
        }
        d = out_root / dataset / PIPELINE / f"{tag}_{rule}"
        out[rule] = write_model_dir(
            d, folds, PIPELINE, f"{tag}_{rule}", sc, pr, folds.fold.copy(),
            lofo, cfg, time.time() - t0,
            description=(f"{rule} of {' and '.join(names)} held-out "
                         f"P(ransomware), argmax at {THRESHOLD}; family-holdout "
                         f"K-fold and LOFO on the {dataset} cohort"))
        print(f"  [{dataset}] {rule}: fold macro-F1 "
              f"{out[rule]['fold_mean']['macro_f1']:.3f} +/- "
              f"{out[rule]['fold_sd']['macro_f1']:.3f}, pooled "
              f"{out[rule]['macro_f1']:.3f}, LOFO {out[rule]['lofo_mean_recall']:.3f}"
              f"  -> {d}", flush=True)
    out["_oracle"] = oracle
    out["_tag"] = tag
    out["_members"] = list(members)
    return out


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def _fm(x, nd=3):
    return "" if x is None else f"{float(x):.{nd}f}"


def _member_metrics(out_root: Path, dataset: str, pipeline: str, model: str) -> dict:
    import json
    d = out_root / dataset / pipeline / model
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    res = m["results"][0]
    fm = _read(d / "fold_metrics.csv")
    v = [float(r["macro_f1"]) for r in fm]
    pf = {r["family"]: (float(r["recall_kfold"]), float(r["recall_lofo"]))
          for r in _read(d / "per_family.csv")}
    return {"fold_mean": st.mean(v), "fold_sd": (st.stdev(v) if len(v) > 1 else 0.0),
            "pooled": res["macro_f1"], "auc": res.get("roc_auc"),
            "rr": res.get("recall_ransomware"), "rg": res.get("recall_goodware"),
            "fpr": res.get("false_positive_rate"),
            "lofo": m.get("lofo_mean_recall"), "per_family": pf}


def write_summary(out_root: Path, results: dict, suffix: str = "") -> Path:
    members = next(iter(results.values()))["_members"]
    L = ["# Ensemble of " + " and ".join(f"{p}/{m}" for p, m in members) + " under family holdout", "",
         "Two pre-registered, parameter-free rules over the members' held-out P(ransomware):",
         "`mean` (primary) and `max` (the OR rule, secondary). No weight, threshold or stacker is fitted;",
         "argmax at 0.5 everywhere. Members are the pre-registered TF-IDF/LogReg baseline and the frozen",
         "sequence-transformer configuration; both K-fold and LOFO rows reuse their held-out predictions",
         "unchanged, so the ensemble is exactly as leakage-free as its members. `fold mean +/- sd` is over",
         "the five held-out folds (population sd, as in summary.md). Written by `family_holdout/run_ensemble.py`.", ""]
    for ds, res in results.items():
        tag = res["_tag"]
        rows = [(f"{p} / {m}", _member_metrics(out_root, ds, p, m)) for p, m in res["_members"]]
        for rule in RULES:
            rows.append((f"ensemble / {rule}" + (" (primary)" if rule == PRIMARY else ""),
                         _member_metrics(out_root, ds, PIPELINE, f"{tag}_{rule}")))
        L += [f"## Dataset: {ds}", "",
              "| model | folds macro-F1 mean +/- sd | pooled macro-F1 | pooled AUC | recall R / G | FPR | LOFO mean recall |",
              "|---|---|---|---|---|---|---|"]
        for name, m in rows:
            L.append(f"| {name} | {_fm(m['fold_mean'])} +/- {_fm(m['fold_sd'])} | {_fm(m['pooled'])} | "
                     f"{_fm(m['auc'])} | {_fm(m['rr'], 2)} / {_fm(m['rg'], 2)} | {_fm(m['fpr'])} | {_fm(m['lofo'])} |")
        o = res["_oracle"]
        L += ["", f"K-fold disagreement between the two members: {o['n_disagree']} files "
              f"({100 * o['disagreement_rate']:.1f}%). Oracle (pick whichever member is right per file): "
              f"ransomware recall {o['oracle_recall_ransomware']:.3f} against members "
              f"{' / '.join(f'{x:.3f}' for x in o['member_recall_ransomware'])}; goodware recall "
              f"{o['oracle_recall_goodware']:.3f} against {' / '.join(f'{x:.3f}' for x in o['member_recall_goodware'])}. "
              "The oracle is an upper bound on any combination rule, not a result.", ""]
        # per-family rows where the primary ensemble differs from BOTH members by >= 0.05 on K-fold or LOFO
        a, b = rows[0][1]["per_family"], rows[1][1]["per_family"]
        e = rows[2][1]["per_family"]
        diff = []
        for fam in sorted(e):
            ek, el = e[fam]
            ak, al = a[fam]
            bk, bl = b[fam]
            if (abs(ek - ak) >= 0.05 and abs(ek - bk) >= 0.05) or (abs(el - al) >= 0.05 and abs(el - bl) >= 0.05):
                diff.append((fam, ak, al, bk, bl, ek, el))
        if diff:
            L += ["Families where the primary (`mean`) ensemble differs from BOTH members by at least 0.05 "
                  "(K-fold recall / LOFO recall):", "",
                  "| family | tfidf / LogReg | seq_transformer | ensemble mean |", "|---|---|---|---|"]
            for fam, ak, al, bk, bl, ek, el in diff:
                L.append(f"| {fam} | {ak:.2f} / {al:.2f} | {bk:.2f} / {bl:.2f} | {ek:.2f} / {el:.2f} |")
            L.append("")
    p = out_root / (f"summary_ensemble{suffix}.md")
    p.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=(*DATASETS, "both"), default="both")
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--members", default=",".join(f"{p}/{m}" for p, m in MEMBERS),
                    help="two 'pipeline/model' result directories, comma-separated")
    ap.add_argument("--summary-suffix", default="",
                    help="written as summary_ensemble<suffix>.md (default: the pre-registered pair)")
    a = ap.parse_args()
    out_root = Path(a.out)
    members = tuple(tuple(x.split("/", 1)) for x in a.members.split(","))
    if len(members) != 2 or any(len(m) != 2 for m in members):
        raise SystemExit("--members needs exactly two pipeline/model entries")
    ds = DATASETS if a.dataset == "both" else (a.dataset,)
    results = {d: run_dataset(d, out_root, members) for d in ds}
    if a.dataset == "both":
        print(f"wrote {write_summary(out_root, results, a.summary_suffix)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
