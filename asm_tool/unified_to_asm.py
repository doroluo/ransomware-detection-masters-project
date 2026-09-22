#!/usr/bin/env python3
"""Turn the revised extractor's full disassembly into an `asm_parse.py`-shaped tree.

Why this exists
---------------
The CNN-ViT pipeline renders images with `asm_parser.py`, which reads a tree of
`.asm` files produced by `asm_parse.py`. On this host that tree cannot be used
for a cohort-faithful run:

  * the ransomware `.asm` trees do not exist here at all (the binaries are
    VM-only), and
  * the host Mendeley-goodware tree was built from a copy in which 67 UPX
    files are still packed, so its sha256s do not match the cohort CSVs.

The revised extractor (`Shared/extract_unified.py`) *does* have output for
every class and every set, keyed by sha256, so this script re-shapes that
output into something `asm_parser.py` can consume unmodified:

    Shared/Extract/asm/<sha256>.asm          ->  OUT/<set>/<family>/<sha256>.asm
    Shared/Extract_Goodware_Balanced/asm/..  ->  OUT/goodware_balanced/<bucket>/<sha256>.asm

What is dropped, and why
------------------------
The revised extractor writes three kinds of line:

    ; section .text va=0x401000 size=107515      section header
    0x00401000:  push\tebx                       an instruction
    0x00404112:  .skip\t1 bytes                  a collapsed undecodable run

`asm_parser.parse_asm_line` already drops the `;` headers, but it would turn a
`.skip` line into a real token triplet -- opcode `.SKIP` is not in TOKEN_MAP,
so it becomes UNKNOWN_OPCODE (id 1) with a bogus operand. Those are not
instructions and `asm_parse.py` never emits them, so they are removed here
instead. Everything else is copied through byte for byte: the
`0x<va>:  <mnemonic>\t<operands>` form is identical on both sides, so the
token stream `asm_parser.py` builds is the same one it would have built from
an `asm_parse.py` file covering the same bytes.

Instruction cap
---------------
Capped at 100,000 instructions per file, matching `asm_parse.py`'s own cap, so
neither side gets a longer stream than the other for reasons of tooling.
(`asm_parser.py` only ever reads the first ~21,845 instructions anyway: its
canvas is 256*256 = 65,536 tokens at 3 tokens per instruction.)

Usage
-----
    python asm_tool/unified_to_asm.py \
        --extract "C:/Users/chaoa/Downloads/asm and mm/Shared/Extract" \
        --out     "C:/Users/chaoa/Downloads/asm_output/unified_mendeley"

    python asm_tool/unified_to_asm.py \
        --extract ".../Shared/Extract_Goodware_Balanced" \
        --out     ".../asm_output/unified_goodware_balanced"

and likewise Extract_VS -> unified_vs, Extract_Goodware_HostX86 ->
unified_goodware_hostx86 (CORPUS_OF_TREE below maps a tree to its corpus).

Add `--cohort-only` to convert just the rows the shared cohort keeps
(`in_cohort == 1` in `Shared/cohort_<corpus>.csv`). Without it every
row with `disassembled == 1` is converted, which is the default because the
extra files cost little and keep the tree reusable for other splits.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_DOWNLOADS = REPO_ROOT.parent
SHARED_DIR = Path(os.environ.get("RANSOM_SHARED_DIR", _DOWNLOADS / "asm and mm" / "Shared"))

# asm_parse.py's cap, reproduced so the two extractors truncate at the same place.
MAX_INSNS = 100_000

# `0x00401000:  mov\teax, ebx` -- the address prefix both extractors emit.
INSN_RE = re.compile(r"^0x[0-9a-fA-F]+:\s")
# The mnemonic is whatever follows the colon, up to the first tab or space.
MNEMONIC_RE = re.compile(r"^0x[0-9a-fA-F]+:\s+(\S+)")

# Collapsed-undecodable markers. `.skip` is what extract_unified.py writes;
# the others are listed so a future marker cannot slip through as an opcode.
SKIP_MNEMONICS = {".skip", ".byte", ".data", "(bad)"}

MANIFEST_COLUMNS = ["sha256", "set", "family", "label", "arch",
                    "n_lines_in", "n_insns_out", "asm_path",
                    "filename", "tag", "n_skip_dropped", "capped"]


# --------------------------------------------------------------- corpora ---
# extraction-tree directory name -> corpus name, as registered in
# family_holdout.common.TREE_OF_CORPUS (the cohort file is cohort_<corpus>.csv)
CORPUS_OF_TREE = {"extract": "mendeley", "extract_goodware_balanced": "balanced",
                  "extract_vs": "vs", "extract_goodware_hostx86": "hostgood"}
GOODWARE_ONLY = ("balanced", "hostgood")


def detect_corpus(extract_dir: Path) -> str:
    """The corpus an extraction tree belongs to, from its directory name."""
    try:
        return CORPUS_OF_TREE[extract_dir.name.lower()]
    except KeyError:
        raise SystemExit(f"{extract_dir.name}: not a registered extraction tree "
                         f"({', '.join(sorted(CORPUS_OF_TREE))})") from None


def rel_dir_for(corpus: str, row: dict) -> str:
    """Where one sample's .asm goes, relative to --out.

    mendeley, vs:  <set>/<family>       good_train/root, mal_test/conti, ...
    balanced, hostgood (goodware only): goodware_<corpus>/<bucket>   the
              manifest's `set` column is a constant there and carries no split
              information, so the bucket in `family` (a category, or the host
              program's folder) is the only meaningful level.
    """
    if corpus in GOODWARE_ONLY:
        return f"goodware_{corpus}/{_safe(row['family'])}"
    return f"{_safe(row['set'])}/{_safe(row['family'])}"


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def _safe(part: str) -> str:
    part = (part or "unknown").strip()
    part = _UNSAFE.sub("_", part)
    return part or "unknown"


def load_cohort_shas(corpus: str, cohort_csv: Path | None) -> set[str]:
    path = cohort_csv or SHARED_DIR / f"cohort_{corpus}.csv"
    if not path.exists():
        raise SystemExit(f"--cohort-only needs {path}; pass --cohort-csv to override")
    keep = set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("in_cohort", "").strip() == "1":
                keep.add(row["sha256"].strip().lower())
    return keep


# ------------------------------------------------------------ conversion ---
def convert_one(src: Path, dst: Path, max_insns: int = MAX_INSNS) -> dict:
    """Copy instruction lines only. Returns per-file counters.

    Lines are written through unchanged (same bytes, same tabs); only whole
    lines are ever dropped.
    """
    n_in = n_out = n_skip = n_comment = n_other = 0
    capped = False
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "r", encoding="utf-8", errors="ignore", newline="") as fin, \
         open(dst, "w", encoding="utf-8", newline="") as fout:
        for line in fin:
            n_in += 1
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(";"):
                n_comment += 1
                continue
            m = MNEMONIC_RE.match(stripped)
            if m is None:
                # Not an address-prefixed instruction and not a comment. Nothing
                # in the current extractor output looks like this; count it so
                # the caller can see if that ever changes.
                n_other += 1
                continue
            if m.group(1).lower() in SKIP_MNEMONICS:
                n_skip += 1
                continue
            if n_out >= max_insns:
                capped = True
                break
            fout.write(line if line.endswith("\n") else line + "\n")
            n_out += 1
    return {"n_lines_in": n_in, "n_insns_out": n_out, "n_skip_dropped": n_skip,
            "n_comment_dropped": n_comment, "n_other_dropped": n_other,
            "capped": int(capped)}


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def run(extract_dir: Path, out_dir: Path, cohort_only: bool,
        cohort_csv: Path | None, limit: int, max_insns: int) -> int:
    manifest_path = extract_dir / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"no manifest.csv under {extract_dir}")
    corpus = detect_corpus(extract_dir)
    rows = read_manifest(manifest_path)
    print(f"=== unified_to_asm  corpus={corpus} ===")
    print(f"extract : {extract_dir}")
    print(f"out     : {out_dir}")
    print(f"manifest: {len(rows)} rows")

    wanted = [r for r in rows if r.get("disassembled", "").strip() == "1"]
    print(f"disassembled == 1: {len(wanted)}")

    if cohort_only:
        keep = load_cohort_shas(corpus, cohort_csv)
        before = len(wanted)
        wanted = [r for r in wanted if r["sha256"].strip().lower() in keep]
        print(f"--cohort-only: {len(wanted)} of {before} are in_cohort == 1")
    if limit:
        wanted = wanted[:limit]
        print(f"--limit: {len(wanted)}")

    out_rows, totals = [], {"n_lines_in": 0, "n_insns_out": 0, "n_skip_dropped": 0,
                            "n_comment_dropped": 0, "n_other_dropped": 0,
                            "capped": 0}
    empty, absent = [], []
    per_dir: dict[str, int] = {}

    for i, r in enumerate(wanted, 1):
        sha = r["sha256"].strip().lower()
        src = extract_dir / (r.get("asm_file") or f"asm/{sha}.asm")
        if not src.exists():
            absent.append(sha)
            continue
        rel = rel_dir_for(corpus, r)
        rel_path = f"{rel}/{sha}.asm"
        dst = out_dir / rel_path
        stats = convert_one(src, dst, max_insns)
        for k in totals:
            totals[k] += stats[k]
        if stats["n_insns_out"] == 0:
            empty.append(sha)
        per_dir[rel] = per_dir.get(rel, 0) + 1
        out_rows.append({
            "sha256": sha,
            "set": r.get("set", ""),
            "family": r.get("family", ""),
            "label": r.get("label", ""),
            "arch": r.get("arch", ""),
            "n_lines_in": stats["n_lines_in"],
            "n_insns_out": stats["n_insns_out"],
            "asm_path": rel_path,
            "filename": r.get("filename", ""),
            "tag": r.get("tag", ""),
            "n_skip_dropped": stats["n_skip_dropped"],
            "capped": stats["capped"],
        })
        if i % 250 == 0 or i == len(wanted):
            print(f"  [{i}/{len(wanted)}] {totals['n_insns_out']:,} instructions written",
                  flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    man_out = out_dir / "asm_manifest.csv"
    with open(man_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(out_rows)

    print(f"\nwrote {len(out_rows)} .asm files under {out_dir}")
    for d in sorted(per_dir):
        print(f"  {d:<36} {per_dir[d]:5d}")
    print(f"\ninstruction lines kept : {totals['n_insns_out']:,}")
    print(f"`;` header lines dropped: {totals['n_comment_dropped']:,}")
    print(f"`.skip` lines dropped   : {totals['n_skip_dropped']:,}")
    print(f"unrecognised lines      : {totals['n_other_dropped']:,}"
          + ("  <-- investigate" if totals["n_other_dropped"] else ""))
    print(f"files hitting the {max_insns:,} cap: {totals['capped']}")
    if empty:
        print(f"files with zero instructions: {len(empty)}  e.g. {empty[:5]}")
    if absent:
        print(f"manifest rows whose .asm is missing on disk: {len(absent)}  "
              f"e.g. {absent[:5]}")
    print(f"manifest: {man_out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", required=True,
                    help="a Shared/Extract* directory (holds manifest.csv and asm/)")
    ap.add_argument("--out", required=True, help="destination tree")
    ap.add_argument("--cohort-only", action="store_true",
                    help="convert only rows with in_cohort == 1")
    ap.add_argument("--cohort-csv", default=None,
                    help="override the cohort CSV --cohort-only reads")
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (testing)")
    ap.add_argument("--max-insns", type=int, default=MAX_INSNS)
    a = ap.parse_args()

    extract = Path(a.extract)
    if not extract.is_dir():
        raise SystemExit(f"not a directory: {extract}")
    return run(extract, Path(a.out), a.cohort_only,
               Path(a.cohort_csv) if a.cohort_csv else None,
               a.limit, a.max_insns)


if __name__ == "__main__":
    sys.exit(main())
