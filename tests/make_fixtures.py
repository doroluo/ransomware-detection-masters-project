#!/usr/bin/env python3
"""
make_fixtures.py - assemble the benign fixture corpus that tests/test_asm_parse.py
runs against.

Fixtures are binaries, so they are gitignored and rebuilt on demand rather than
committed. Every real sample is copied from Goodware_Balanced, which is
VirusTotal-clean; nothing here is malware and nothing is ever executed.

    python tests/make_fixtures.py --corpus ../Goodware_Balanced

Produces tests/fixtures/:
    pe32.bin            smallest x86 PE with a code section
    pe64.bin            smallest x64 PE with a code section
    dll64.bin           smallest x64 .dll
    dotnet.bin          IL-only managed assembly
    upx.bin             UPX-packed (copied from _upx_packed/, pre-unpack)
    no_exec.bin         PE with no MEM_EXECUTE section (API-set forwarder)
    highentropy.bin     pe64/pe32 with its code section overwritten by random
                        bytes - a packer-shaped PE that was never packed, and
                        never executable as one
    truncated.bin       first 512 bytes of pe32.bin - valid MZ, unparseable
    garbage_mz.bin      "MZ" followed by random bytes
    notpe.txt           plain text, no MZ
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

import pefile

FIXTURES = Path(__file__).resolve().parent / "fixtures"
IMAGE_SCN_MEM_EXECUTE = 0x20000000


def make_high_entropy(src: Path, dest: Path) -> int | None:
    """Copy `src` and fill its largest executable section with random bytes.

    This is how a synthetic positive for the entropy branch of
    asm_parse.packing_profile is built without obtaining a real packed sample
    and without running a packer. The result is a structurally valid PE whose
    code section is noise; it is data for a disassembler and nothing else.
    """
    try:
        pe = pefile.PE(str(src), fast_load=True)
    except Exception:
        return None
    try:
        execs = [s for s in pe.sections
                 if (s.Characteristics & IMAGE_SCN_MEM_EXECUTE)
                 and s.SizeOfRawData]
        if not execs:
            return None
        sec = max(execs, key=lambda s: s.SizeOfRawData)
        off, size = sec.PointerToRawData, sec.SizeOfRawData
    finally:
        pe.close()

    data = bytearray(src.read_bytes())
    if off + size > len(data):
        size = len(data) - off
    if size <= 0:
        return None
    data[off:off + size] = os.urandom(size)
    dest.write_bytes(bytes(data))
    return size


def pick(rows, out_name, **criteria):
    """Smallest row matching every criterion, as (row, out_name)."""
    cand = [r for r in rows
            if all(r.get(k) == v for k, v in criteria.items())]
    if not cand:
        return None
    best = min(cand, key=lambda r: int(r["size"]))
    return best, out_name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../Goodware_Balanced")
    args = ap.parse_args()

    corpus = Path(args.corpus).resolve()
    index = corpus / "corpus_index.csv"
    if not index.is_file():
        raise SystemExit(f"corpus_index.csv not found under {corpus}")

    FIXTURES.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(index.open(encoding="utf-8", newline="")))

    wanted = [
        pick(rows, "pe32.bin", arch="x86", pipeline="ok"),
        pick(rows, "pe64.bin", arch="x64", pipeline="ok"),
        pick(rows, "dotnet.bin", pipeline="dotnet"),
        pick(rows, "no_exec.bin", pipeline="no_code_section"),
    ]
    dlls = [r for r in rows
            if r["pipeline"] == "ok" and r["arch"] == "x64"
            and r["filename"].lower().endswith(".dll")]
    if dlls:
        wanted.append((min(dlls, key=lambda r: int(r["size"])), "dll64.bin"))

    made = []
    for item in wanted:
        if item is None:
            continue
        row, out_name = item
        src = corpus / row["rel_path"]
        if not src.is_file():
            print(f"  missing source, skipped: {row['rel_path']}")
            continue
        shutil.copy2(src, FIXTURES / out_name)
        made.append((out_name, row["filename"], int(row["size"])))

    packed_dir = corpus / "_upx_packed"
    if packed_dir.is_dir():
        packed = sorted((p for p in packed_dir.rglob("*") if p.is_file()),
                        key=lambda p: p.stat().st_size)
        if packed:
            shutil.copy2(packed[0], FIXTURES / "upx.bin")
            made.append(("upx.bin", packed[0].name, packed[0].stat().st_size))

    # Degenerate cases are synthesised, so the suite still has teeth if the
    # corpus is absent.
    pe32 = FIXTURES / "pe32.bin"
    pe64 = FIXTURES / "pe64.bin"
    base = pe64 if pe64.is_file() else pe32
    if base.is_file():
        n = make_high_entropy(base, FIXTURES / "highentropy.bin")
        if n:
            made.append(("highentropy.bin",
                         f"{base.name} + {n}B random code section",
                         (FIXTURES / "highentropy.bin").stat().st_size))
    if pe32.is_file():
        (FIXTURES / "truncated.bin").write_bytes(pe32.read_bytes()[:512])
        made.append(("truncated.bin", "first 512B of pe32.bin", 512))
    (FIXTURES / "garbage_mz.bin").write_bytes(b"MZ" + os.urandom(4094))
    (FIXTURES / "notpe.txt").write_text(
        "this is not a PE file\nmov eax, ebx\n", encoding="utf-8")
    made += [("garbage_mz.bin", "synthesised", 4096),
             ("notpe.txt", "synthesised", 37)]

    print(f"fixtures in {FIXTURES}")
    for name, origin, size in made:
        print(f"  {name:<16}{size:>12,}  <- {origin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
