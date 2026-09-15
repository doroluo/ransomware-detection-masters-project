#!/usr/bin/env python3
"""
folds.py - fold definition for family-holdout evaluation over every cohort family.

Why: the fixed Mendeley split (24 train / 14 test families) gave every pipeline
one number, and the tuning study showed that group cross-validation inside the
24 training families does not predict the 14 held-out ones. Evaluating on
every family in turn removes the dependence on which families happened to
land on which side.

Pool
    ransomware  every in-cohort file from mal_train and mal_test: 1,266 files,
                38 families (Thanos is .NET, Night Sky is 100% packed; neither
                can enter the opcode track, so "all 40" is 38 here).
    goodware    dataset "mendeley": in-cohort good_train + good_test (1,243)
                dataset "balanced": in-cohort Goodware_Balanced (1,337)

Folds (K = 5, deterministic)
    ransomware  families assigned whole to folds by greedy count balancing
                (largest family first, into the currently smallest fold).
                The assignment is shared by both datasets.
    goodware    assigned by GROUP, never by file, so that duplicates and
                same-project files never straddle a fold:
                  mendeley  group = SHA-256 of the mnemonic stream (60/47/35-file
                            duplicate groups exist; 1,024 distinct streams)
                  balanced  group = source project (entry_id, 103 projects)
                greedy count balancing as above.

Two evaluation schemes are derived from the same file:
    K-fold        train on 4 folds, test on the 5th; report mean +/- sd over
                  folds and pooled per-family / per-arch metrics (every file
                  gets exactly one held-out prediction).
    LOFO          leave one family out: train on all other ransomware families
                  plus ALL goodware, test on the held-out family alone; gives
                  per-family recall with maximal training data. FPR is not
                  defined for LOFO (no goodware in the test set), so LOFO is a
                  recall study only; use the K-fold numbers for everything else.

Output (--out DIR)
    folds_mendeley.csv, folds_balanced.csv   dataset, sha256, label, family,
                                             group, arch, orig_set, fold
    families.csv                             family, n, n_x64, fold

Usage
    python family_holdout/folds.py --out results/family_holdout
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
from pathlib import Path

K = 5
SHARED = Path(r"C:/Users/chaoa/Downloads/asm and mm/Shared")
COHORT = {"mendeley": SHARED / "cohort_mendeley.csv", "balanced": SHARED / "cohort_balanced.csv"}
UNIFIED_MANIFEST = SHARED / "Extract" / "manifest.csv"
CORPUS_INDEX = Path(r"C:/Users/chaoa/Downloads/Goodware_Balanced/corpus_index.csv")
FIELDS = ["dataset", "sha256", "label", "family", "group", "arch", "orig_set", "fold"]


def _read(path: Path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def assign_greedy(counts: dict[str, int], k: int = K) -> dict[str, int]:
    """Largest group first into the currently smallest fold. Ties broken by name,
    so the result is a pure function of the input."""
    folds = [0] * k
    out = {}
    for name in sorted(counts, key=lambda n: (-counts[n], n)):
        f = min(range(k), key=lambda i: (folds[i], i))
        out[name] = f
        folds[f] += counts[name]
    return out


def ransomware_rows():
    rows = [r for r in _read(COHORT["mendeley"]) if r["in_cohort"] == "1" and r["label"] == "1"]
    fam_counts = collections.Counter(r["family"] for r in rows)
    fam_fold = assign_greedy(fam_counts)
    return [{"sha256": r["sha256"], "label": 1, "family": r["family"], "group": r["family"],
             "arch": r["arch"], "orig_set": r["set"], "fold": fam_fold[r["family"]]} for r in rows], fam_fold


def mendeley_goodware_rows():
    rows = [r for r in _read(COHORT["mendeley"]) if r["in_cohort"] == "1" and r["label"] == "0"]
    mn_of = {}
    for m in _read(UNIFIED_MANIFEST):
        if m["mn_file"]:
            mn_of.setdefault(m["sha256"], m["mn_file"])
    stream = {}
    for r in rows:
        p = SHARED / "Extract" / mn_of[r["sha256"]]
        stream[r["sha256"]] = "mn:" + hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    grp_fold = assign_greedy(collections.Counter(stream.values()))
    return [{"sha256": r["sha256"], "label": 0, "family": "goodware", "group": stream[r["sha256"]],
             "arch": r["arch"], "orig_set": r["set"], "fold": grp_fold[stream[r["sha256"]]]} for r in rows]


def balanced_goodware_rows():
    rows = [r for r in _read(COHORT["balanced"]) if r["in_cohort"] == "1"]
    entry = {r["sha256"]: r["entry_id"] for r in _read(CORPUS_INDEX)}
    groups = {r["sha256"]: "proj:" + (entry.get(r["sha256"]) or "unknown:" + r["sha256"][:8]) for r in rows}
    grp_fold = assign_greedy(collections.Counter(groups.values()))
    return [{"sha256": r["sha256"], "label": 0, "family": "goodware", "group": groups[r["sha256"]],
             "arch": r["arch"], "orig_set": r["family"], "fold": grp_fold[groups[r["sha256"]]]} for r in rows]


def build(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rans, fam_fold = ransomware_rows()
    datasets = {"mendeley": rans + mendeley_goodware_rows(), "balanced": rans + balanced_goodware_rows()}
    for ds, rows in datasets.items():
        with (out / f"folds_{ds}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for r in sorted(rows, key=lambda r: (r["fold"], r["label"], r["family"], r["sha256"])):
                w.writerow({"dataset": ds, **r})
    fam_n = collections.Counter(r["family"] for r in rans)
    fam_x64 = collections.Counter(r["family"] for r in rans if r["arch"] == "x64")
    with (out / "families.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "n", "n_x64", "fold", "orig_set"])
        orig = {r["family"]: r["orig_set"] for r in rans}
        for f in sorted(fam_n, key=lambda f: (fam_fold[f], f)):
            w.writerow([f, fam_n[f], fam_x64[f], fam_fold[f], orig[f]])
    return datasets


def load(ds: str, out: Path):
    return _read(out / f"folds_{ds}.csv")


def summary(datasets: dict) -> str:
    L = []
    for ds, rows in datasets.items():
        L.append(f"{ds}: {len(rows)} files")
        for f in range(K):
            fr = [r for r in rows if r["fold"] == f]
            g = sum(r["label"] == 0 for r in fr); m = len(fr) - g
            fams = sorted({r["family"] for r in fr if r["label"] == 1})
            x64 = sum(r["arch"] == "x64" for r in fr if r["label"] == 1)
            L.append(f"  fold {f}: {len(fr):4d} files  goodware {g:4d}  ransomware {m:4d} (x64 {x64:3d})  families {len(fams):2d}: {', '.join(fams)}")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/family_holdout")
    a = ap.parse_args()
    datasets = build(Path(a.out))
    print(summary(datasets))
    print(f"wrote folds_mendeley.csv, folds_balanced.csv, families.csv under {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
