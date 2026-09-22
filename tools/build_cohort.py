#!/usr/bin/env python3
"""
build_cohort.py - turn an extract_unified.py manifest into a cohort file.

The cohort file is what every pipeline and the fold builder read
(corpus, sha256, set, label, family, filename, arch, tag, in_cohort,
exclude_reason). This applies the inclusion rules of docs/DATASET_PLAN.md §2:

    in_cohort = 1   tag in {plain, upx_unpacked}
                    and arch in {x86, x64}
                    and family not in --exclude-family (Thanos, .NET-only)
    exclude_reason  the tag, "arch:<value>" or "family:<name>" otherwise

Family names are normalised with family_holdout.folds.normalise_family (lower
case, alphanumerics, aliases onto the Mendeley spelling), so the same family
from two corpora is one family.

    python tools/build_cohort.py --manifest "<Shared>/Extract_VS/manifest.csv" \
        --corpus vs --set mal_train --out "<Shared>/cohort_vs.csv"

Prints the per-family x86/x64 counts of the rows that made it in.
"""
from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from family_holdout.folds import normalise_family  # noqa: E402

COLUMNS = ["corpus", "sha256", "set", "label", "family", "filename", "arch", "tag",
           "in_cohort", "exclude_reason"]
OPCODE_TAGS = ("plain", "upx_unpacked")
ARCHES = ("x86", "x64")


def build(manifest_rows, corpus: str, set_name: str | None, label: int | None,
          exclude_families=("thanos",), index: dict | None = None) -> list[dict]:
    """`index` maps sha256 -> extra columns (e.g. group, is_dll) appended to
    every row; rows without an index entry get empty values."""
    excl = {normalise_family(f) for f in exclude_families}
    extra_cols = sorted({k for v in (index or {}).values() for k in v})
    out = []
    for r in manifest_rows:
        lab = int(r["label"]) if label is None else label
        fam = normalise_family(r["family"]) if lab == 1 else "goodware"
        reason = ""
        if r["tag"] not in OPCODE_TAGS:
            reason = r["tag"]
        elif r["arch"] not in ARCHES:
            reason = f"arch:{r['arch'] or 'unknown'}"
        elif lab == 1 and fam in excl:
            reason = f"family:{fam}"
        row = {"corpus": corpus, "sha256": r["sha256"].lower(),
               "set": set_name or r["set"], "label": lab, "family": fam,
               "filename": r["filename"], "arch": r["arch"], "tag": r["tag"],
               "in_cohort": 0 if reason else 1, "exclude_reason": reason}
        if extra_cols:
            ex = (index or {}).get(r["sha256"].lower(), {})
            row.update({k: ex.get(k, "") for k in extra_cols})
        out.append(row)
    return out


def summary(rows) -> str:
    kept = [r for r in rows if r["in_cohort"] == 1]
    reasons = collections.Counter(r["exclude_reason"] for r in rows if r["in_cohort"] == 0)
    L = [f"{len(kept)} of {len(rows)} rows in cohort; excluded: "
         + (", ".join(f"{k}={v}" for k, v in sorted(reasons.items())) or "none")]
    fam = collections.defaultdict(lambda: [0, 0])
    for r in kept:
        if r["label"] == 1:
            fam[r["family"]][r["arch"] == "x64"] += 1
    if fam:
        L.append(f"{'family':16} {'x86':>5} {'x64':>5}")
        for f in sorted(fam, key=lambda f: (-sum(fam[f]), f)):
            L.append(f"{f:16} {fam[f][0]:5d} {fam[f][1]:5d}")
        L.append(f"{'total':16} {sum(v[0] for v in fam.values()):5d} {sum(v[1] for v in fam.values()):5d}")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--set", dest="set_name", help="override the manifest's set column (e.g. mal_train)")
    ap.add_argument("--label", type=int, choices=(0, 1), help="override the manifest's label column")
    ap.add_argument("--exclude-family", action="append", default=["thanos"], metavar="NAME")
    ap.add_argument("--index", type=Path, help="CSV with sha256 plus extra columns (group, is_dll) to append")
    a = ap.parse_args()
    with a.manifest.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    index = None
    if a.index:
        with a.index.open(encoding="utf-8", newline="") as fh:
            index = {r["sha256"].lower(): {k: v for k, v in r.items()
                                           if k in ("group", "is_dll")}
                     for r in csv.DictReader(fh)}
    cohort = build(rows, a.corpus, a.set_name, a.label, a.exclude_family, index)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cohort[0]) if cohort else COLUMNS)
        w.writeheader()
        w.writerows(cohort)
    print(summary(cohort))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
