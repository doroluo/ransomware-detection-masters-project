#!/usr/bin/env python3
"""
arch_balance.py - measure and fix the x86/x64 split between the two classes.

The confound this repo keeps meeting: ransomware is mostly x86, goodware mostly
x64, so "x64 means goodware" is a free 0.66-0.84 accuracy. This tool reads any
set of cohort files (in-cohort rows only) and

  1. prints the x86/x64 counts and the x64 share per class, per corpus and per
     ransomware family, plus the x86-rule accuracy on that pool;
  2. with --match, writes a selection CSV (sha256, keep) in which both classes
     have identical (x86, x64) counts. The class that is over-represented in an
     architecture is thinned per file, deterministically (seed), spread over
     the ransomware families in proportion to their size so no family vanishes
     and the fold assignment stays valid;
  3. with --need, prints how many x64 ransomware files would be needed to reach
     a target x64 share without dropping anything, per family where a family
     ships x64 builds at all.

    python tools/arch_balance.py --cohort <Shared>/cohort_mendeley.csv \
        --cohort <Shared>/cohort_balanced.csv --cohort <Shared>/cohort_vs.csv
    python tools/arch_balance.py ... --match results/arch_matched.csv --seed 1
    python tools/arch_balance.py ... --need 0.40

A runner that honours the selection reads `keep == 1` rows only; the fold file
itself is not changed, so the same folds serve both the full and the matched
evaluation.
"""
from __future__ import annotations

import argparse
import collections
import csv
import random
from pathlib import Path

ARCHES = ("x86", "x64")


def read_cohorts(paths):
    rows, seen = [], set()
    for p in paths:
        with Path(p).open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if r["in_cohort"] != "1" or r["arch"] not in ARCHES:
                    continue
                sha = r["sha256"].lower()
                if sha in seen:
                    continue
                seen.add(sha)
                rows.append({"sha256": sha, "label": int(r["label"]), "arch": r["arch"],
                             "family": r["family"] if int(r["label"]) == 1 else "goodware",
                             "corpus": r.get("corpus") or Path(p).stem.replace("cohort_", "")})
    return rows


def counts(rows):
    c = collections.Counter((r["label"], r["arch"]) for r in rows)
    return {lab: {a: c[(lab, a)] for a in ARCHES} for lab in (0, 1)}


def x64_share(d):
    n = d["x86"] + d["x64"]
    return d["x64"] / n if n else float("nan")


def report(rows) -> str:
    L = []
    c = counts(rows)
    L.append(f"{'class':12} {'x86':>6} {'x64':>6} {'x64 share':>10}")
    for lab, name in ((1, "ransomware"), (0, "goodware")):
        L.append(f"{name:12} {c[lab]['x86']:6d} {c[lab]['x64']:6d} {x64_share(c[lab]):10.3f}")
    n = len(rows)
    x86_rule = sum((r["arch"] == "x86") == (r["label"] == 1) for r in rows) / n if n else float("nan")
    L.append(f"x86-rule accuracy on this pool: {x86_rule:.3f}  (n = {n})")
    by_corpus = collections.defaultdict(collections.Counter)
    for r in rows:
        by_corpus[(r["corpus"], r["label"])][r["arch"]] += 1
    L.append(f"\n{'corpus':12} {'class':12} {'x86':>6} {'x64':>6} {'x64 share':>10}")
    for (corpus, lab), d in sorted(by_corpus.items()):
        L.append(f"{corpus:12} {'ransomware' if lab else 'goodware':12} {d['x86']:6d} {d['x64']:6d} "
                 f"{x64_share(d):10.3f}")
    fam = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["label"] == 1:
            fam[r["family"]][r["arch"]] += 1
    L.append(f"\n{'family':16} {'x86':>5} {'x64':>5}")
    for f in sorted(fam, key=lambda f: (-fam[f]["x64"], f)):
        L.append(f"{f:16} {fam[f]['x86']:5d} {fam[f]['x64']:5d}")
    return "\n".join(L)


def match(rows, seed: int = 1) -> dict:
    """sha256 -> keep (0/1) with identical (x86, x64) counts in both classes.

    For each architecture both classes keep min(n_ransomware, n_goodware)
    files, so the scarce class in that architecture is kept in full and the
    other is thinned. Thinning is per file, spread over families in proportion
    to how many files of that architecture each family has (largest-remainder
    quotas, deterministic under `seed`), so no family disappears and the fold
    assignment stays valid; goodware is one "family" and is thinned at random.
    """
    rng = random.Random(seed)
    c = counts(rows)
    target = {a: min(c[0][a], c[1][a]) for a in ARCHES}
    keep = {r["sha256"]: 1 for r in rows}
    for a in ARCHES:
        for lab in (0, 1):
            excess = c[lab][a] - target[a]
            if excess <= 0:
                continue
            by_fam = collections.defaultdict(list)
            for r in rows:
                if r["label"] == lab and r["arch"] == a:
                    by_fam[r["family"]].append(r["sha256"])
            total = sum(len(v) for v in by_fam.values())
            quota = {f: excess * len(v) / total for f, v in by_fam.items()}
            drop = {f: int(q) for f, q in quota.items()}
            for f in sorted(by_fam, key=lambda f: (-(quota[f] - drop[f]), f))[:excess - sum(drop.values())]:
                drop[f] += 1
            for f in sorted(by_fam):
                files = sorted(by_fam[f])
                rng.shuffle(files)
                for sha in files[:drop[f]]:
                    keep[sha] = 0
    return keep


def need(rows, target_share: float) -> str:
    c = counts(rows)
    n86 = c[1]["x86"]
    want = int(round(target_share / (1 - target_share) * n86)) - c[1]["x64"]
    L = [f"ransomware x64 share {x64_share(c[1]):.3f} -> target {target_share:.2f} needs "
         f"{max(0, want)} more x64 ransomware files (keeping all {n86} x86)"]
    fam = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["label"] == 1:
            fam[r["family"]][r["arch"]] += 1
    ships = [f for f in fam if fam[f]["x64"] > 0]
    L.append(f"families that ship x64 builds in this pool: {len(ships)} of {len(fam)}: "
             + ", ".join(sorted(ships)))
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", action="append", required=True, metavar="CSV")
    ap.add_argument("--match", metavar="OUT.csv", help="write an arch-matched selection (sha256, keep)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--need", type=float, metavar="SHARE", help="x64 share of ransomware to aim for")
    a = ap.parse_args()
    rows = read_cohorts(a.cohort)
    print(report(rows))
    if a.need is not None:
        print()
        print(need(rows, a.need))
    if a.match:
        keep = match(rows, a.seed)
        kept = [r for r in rows if keep[r["sha256"]]]
        out = Path(a.match)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["sha256", "keep"])
            for r in sorted(rows, key=lambda r: r["sha256"]):
                w.writerow([r["sha256"], keep[r["sha256"]]])
        print(f"\narch-matched selection: {len(kept)} of {len(rows)} files kept -> {out}")
        print(report(kept))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
