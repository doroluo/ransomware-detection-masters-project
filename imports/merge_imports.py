#!/usr/bin/env python3
"""
merge_imports.py - flatten the per-corpus import JSONs into the one file the
sequence model reads, and report cohort coverage.

    python imports/merge_imports.py \
        --out manifests/imports/imports_flat.json \
        manifests/imports/mendeley_mal_train.json \
        manifests/imports/mendeley_mal_test.json \
        manifests/imports/mendeley_good_train.json \
        manifests/imports/mendeley_good_test.json \
        manifests/imports/goodware_balanced.json

`imports/extract_imports.py` writes `{sha256: {"imports": [...], "iat": {...},
...}}` per corpus; `seq_model/run_family_holdout.py --imports` wants one
`{sha256: [import names]}` over every file of both cohorts. This does that
and nothing else: no renaming, no filtering of names, records with a parse
error or no import directory become an empty list (the hashed vector is then
all zeros, which is what "this file imports nothing we can see" should look
like). If a sha256 appears in more than one input the lists must agree, else
the run stops.

Coverage is checked against results/family_holdout/folds_{mendeley,balanced}.csv
and printed per (dataset, label); the run refuses to write when any cohort
file is missing unless --allow-missing is given, because a silent gap would
turn the imports feature into a "which corpus half is this" feature.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FOLDS = REPO / "results" / "family_holdout"


def load_one(path: Path) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    out, n_err = {}, 0
    for sha, rec in doc.items():
        if isinstance(rec, list):                 # already flat
            out[sha] = [str(x) for x in rec]
        else:
            out[sha] = [str(x) for x in (rec.get("imports") or [])]
            n_err += bool(rec.get("error"))
    if n_err:
        # a parse failure and a genuinely empty import table both flatten to
        # []; say how many of the former there are so a label-skewed failure
        # rate cannot pass unnoticed
        print(f"{path}: {n_err} of {len(doc)} records are parse errors (flattened to [])")
    return out


def merge(paths) -> tuple[dict, dict]:
    merged, sources = {}, {}
    for p in paths:
        d = load_one(Path(p))
        for sha, names in d.items():
            if sha in merged and merged[sha] != names:
                raise SystemExit(f"{sha} differs between {sources[sha]} and {p}")
            merged.setdefault(sha, names)
            sources.setdefault(sha, str(p))
    return merged, sources


def coverage(merged: dict) -> list:
    rows = []
    fold_files = sorted(FOLDS.glob("folds_*.csv"))
    if not fold_files:
        raise SystemExit(f"no folds_*.csv under {FOLDS}; run family_holdout/folds.py first "
                         "(the coverage guarantee cannot be checked without it)")
    for f in fold_files:
        ds = f.stem[len("folds_"):]
        with f.open(encoding="utf-8", newline="") as fh:
            recs = list(csv.DictReader(fh))
        for label in ("1", "0"):
            shas = [r["sha256"] for r in recs if r["label"] == label]
            have = sum(1 for s in shas if s in merged)
            empty = sum(1 for s in shas if s in merged and not merged[s])
            rows.append((ds, "ransomware" if label == "1" else "goodware",
                         len(shas), have, empty))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()

    merged, _ = merge(a.inputs)
    n_names = collections.Counter(len(v) for v in merged.values())
    print(f"{len(merged)} files; {sum(1 for v in merged.values() if not v)} with no imports; "
          f"median names per file {sorted(len(v) for v in merged.values())[len(merged)//2]}")
    missing = 0
    for ds, lab, n, have, empty in coverage(merged):
        print(f"  {ds:9s} {lab:10s} {have:5d} / {n:5d} covered   ({empty} with an empty list)")
        missing += n - have
    if missing and not a.allow_missing:
        print(f"{missing} cohort files have no import record; refusing to write "
              f"(pass --allow-missing to override)", file=sys.stderr)
        return 1
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        fh.write("{\n")
        for i, sha in enumerate(sorted(merged)):
            fh.write(json.dumps(sha) + ": " + json.dumps(merged[sha])
                     + (",\n" if i < len(merged) - 1 else "\n"))
        fh.write("}\n")
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
