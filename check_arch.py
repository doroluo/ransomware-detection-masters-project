#!/usr/bin/env python3
"""
check_arch.py - report the architecture mix of a PE directory. Read-only.

The Mendeley p3v94dft2y description does not state whether its ransomware
samples are 32- or 64-bit, so this has to be measured. Architecture matters
because if the ransomware class is predominantly x86 and the goodware class
is predominantly x64 (or vice versa), `machine` becomes a shortcut feature
and the model can separate the classes without learning anything about
behaviour.

Nothing is executed, modified, or written except the optional --csv report;
files are opened read-only and only their headers are parsed.

The directory walk excludes the same paths asm_parse.py excludes, so the two
tools describe the same corpus. Pointed at Goodware_Balanced, counting
everything gives 1,525 PEs and 25 "UPX packed"; the corpus is 1,500 PEs with
zero UPX, because `_upx_packed/` holds the packed originals of files already
unpacked in place and `.tools/` holds upx.exe itself. Pass --no-skip for the
old, everything-under-the-root behaviour.

Usage:
    python check_arch.py --dir <ransomware dir>
    python check_arch.py --dir <dir> --csv arch_report.csv
    python check_arch.py --dir <dir> --compare "C:/Users/chaoa/Downloads/Goodware_Balanced/corpus_index.csv"
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import pefile

MACHINE = {0x14C: "x86", 0x8664: "x64", 0xAA64: "arm64",
           0x1C0: "arm", 0x1C4: "armnt", 0x200: "ia64"}

# Mirrors asm_parse.SKIP_DIRS / SKIP_SUFFIXES. Duplicated rather than imported
# so this stays a pefile-only script with no capstone dependency.
SKIP_DIRS = {"_upx_packed", ".tools", "flagged", "quarantined"}
SKIP_SUFFIXES = {".csv", ".md", ".json"}


def scan(root: Path, skip: bool = True) -> list[dict]:
    out = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if skip and (any(part in SKIP_DIRS for part in rel.parts)
                     or path.suffix.lower() in SKIP_SUFFIXES):
            continue
        rec = {"rel_path": str(path.relative_to(root)),
               "size": path.stat().st_size, "arch": "", "dotnet": 0,
               "upx": 0, "subsystem": "", "status": "ok"}
        try:
            with path.open("rb") as fh:
                if fh.read(2) != b"MZ":
                    rec["status"] = "not_pe"
                    out.append(rec)
                    continue
            pe = pefile.PE(str(path), fast_load=True)
            pe.parse_data_directories(directories=[
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR"]])
            rec["arch"] = MACHINE.get(pe.FILE_HEADER.Machine,
                                      hex(pe.FILE_HEADER.Machine))
            rec["subsystem"] = str(pe.OPTIONAL_HEADER.Subsystem)
            try:
                cd = pe.OPTIONAL_HEADER.DATA_DIRECTORY[
                    pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR"]]
                rec["dotnet"] = int(bool(cd.VirtualAddress and cd.Size))
            except (IndexError, AttributeError, KeyError):
                pass
            rec["upx"] = int(any(b"UPX" in s.Name for s in pe.sections))
            pe.close()
        except Exception as exc:
            rec["status"] = f"error:{type(exc).__name__}"
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--csv", default="")
    ap.add_argument("--compare", default="",
                    help="a corpus_index.csv to compare against")
    ap.add_argument("--no-skip", action="store_true",
                    help="count every file under --dir, including "
                         f"{'/'.join(sorted(SKIP_DIRS))} and index files. "
                         "Default is to exclude them, matching asm_parse.py.")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    rows = scan(root, skip=not args.no_skip)
    pe_rows = [r for r in rows if r["status"] == "ok"]
    n = len(pe_rows)
    if not n:
        raise SystemExit("no PE files found")

    print(f"{root}\n{len(rows)} files, {n} parsed as PE\n")
    arch = collections.Counter(r["arch"] for r in pe_rows)
    print("architecture:")
    for k, v in arch.most_common():
        print(f"  {k:<8}{v:>6}  {100*v/n:5.1f}%")
    print(f"\n.NET managed: {sum(r['dotnet'] for r in pe_rows)} "
          f"({100*sum(r['dotnet'] for r in pe_rows)/n:.1f}%)")
    print(f"UPX packed:   {sum(r['upx'] for r in pe_rows)} "
          f"({100*sum(r['upx'] for r in pe_rows)/n:.1f}%)")
    bad = [r for r in rows if r["status"] != "ok"]
    if bad:
        print(f"unparsed:     {len(bad)}")

    if args.compare:
        other = list(csv.DictReader(Path(args.compare).open(encoding="utf-8")))
        o = collections.Counter(r.get("arch") for r in other)
        on = sum(o.values())
        print(f"\ncompare with {args.compare} ({on} samples):")
        print(f"  {'':<10}{'this dir':>12}{'compare':>12}{'delta':>10}")
        for k in ("x86", "x64"):
            a = 100 * arch.get(k, 0) / n
            b = 100 * o.get(k, 0) / on if on else 0
            print(f"  {k:<10}{a:>11.1f}%{b:>11.1f}%{a-b:>+9.1f}")
        skew = abs(100 * arch.get("x86", 0) / n
                   - (100 * o.get("x86", 0) / on if on else 0))
        print()
        if skew > 25:
            print(f"  WARNING: {skew:.0f} point gap in x86 share. `machine` is a "
                  f"strong shortcut feature at this spread -\n"
                  f"  rebalance one class or drop architecture-dependent features.")
        elif skew > 10:
            print(f"  CAUTION: {skew:.0f} point gap in x86 share - worth a "
                  f"per-architecture accuracy breakdown.")
        else:
            print(f"  OK: architecture shares are within {skew:.0f} points.")

    if args.csv:
        with Path(args.csv).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
