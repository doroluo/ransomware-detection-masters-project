#!/usr/bin/env python3
"""
asm_to_opcodes.py - convert an asm_parse.py `.asm` tree into the flat
instruction-text format that LLM_Features / Tokenization-Testing consume.

Why this exists
---------------
Two disassembly formats are in play and they are NOT interchangeable:

    extract.py   ->  "mov eax, ebx"                  (LLM_Features/*.txt)
    asm_parse.py ->  "0x00401000:  mov\teax, ebx"    (asm_output/<dataset>/*.asm)

The address prefix is the whole problem. `Tokenization/tokenization.py`
rewrites `0x...` to `<HEX>`, so an un-stripped `.asm` line tokenizes to
`<HEX>: mov eax ebx` while the same instruction from `extract.py` tokenizes to
`mov eax ebx`. If ransomware features come from LLM_Features (no prefix) and
goodware features come from a raw `.asm` tree (prefix on every single line),
that prefix is present in exactly one class and absent in the other: a
perfect, content-free shortcut. A classifier would score ~1.00 by learning
"does line 1 start with <HEX>:".

So the conversion is not cosmetic. Run it before mixing the two sources.

Output naming follows extract.py's `{family_prefix}_{filename}.txt`, where
family_prefix is the sample's directory relative to the ASM root - for
asm_output/goodware_balanced that is the bucket (everyday / system /
hard_negative). `node.exe` appears in two buckets, so the prefix is what keeps
them apart. A flat tree such as asm_output/mendeley_goodware gets "root".

Usage:
    python asm_tool/asm_to_opcodes.py \
        --asm-dir  ../asm_output/goodware_balanced \
        --out-dir  ../LLM_Features_Goodware_Balanced/good_all \
        --index    ../Goodware_Balanced/corpus_index.csv \
        --manifest ../LLM_Features_Goodware_Balanced/opcode_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

# "0x00401000:  " - asm_parse.py always emits lowercase hex and two spaces,
# but accept any run of whitespace so a hand-edited fixture still parses.
ADDR_RE = re.compile(r"^0x[0-9a-fA-F]+:\s*")


def convert_line(line: str) -> str | None:
    """One `.asm` line -> one extract.py-style instruction, or None to drop."""
    line = line.rstrip("\n\r")
    if not line.strip():
        return None
    line = ADDR_RE.sub("", line)
    # asm_parse.py separates mnemonic from operands with a tab; extract.py uses
    # a single space and strips, so `ret\t` must collapse to `ret`.
    line = line.replace("\t", " ").strip()
    return line or None


def convert_file(src: Path, dest: Path) -> tuple[int, int]:
    """Returns (lines_in, instructions_out)."""
    n_in = n_out = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8", errors="replace") as fh, \
         dest.open("w", encoding="utf-8", newline="\n") as out:
        for raw in fh:
            n_in += 1
            conv = convert_line(raw)
            if conv is None:
                continue
            out.write(conv + "\n")
            n_out += 1
    return n_in, n_out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_index(path: Path) -> dict[str, dict]:
    """corpus_index.csv keyed by rel_path, with backslashes normalised."""
    if not path or not path.is_file():
        return {}
    out = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row.get("rel_path") or "").replace("\\", "/")
            if key:
                out[key] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asm-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--index", default="",
                    help="corpus_index.csv, to carry vendor/arch/sha256 into the manifest")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-instructions", type=int, default=0,
                    help="0 = keep all. The tokenizer truncates at 5,000 anyway.")
    args = ap.parse_args()

    asm_dir = Path(args.asm_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not asm_dir.is_dir():
        sys.exit(f"not a directory: {asm_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    index = load_index(Path(args.index)) if args.index else {}
    rows = []
    collisions: dict[str, str] = {}

    asm_files = sorted(p for p in asm_dir.rglob("*.asm") if p.is_file())
    if args.limit:
        asm_files = asm_files[: args.limit]
    print(f"{len(asm_files)} .asm files under {asm_dir}")

    for i, src in enumerate(asm_files, 1):
        rel = src.relative_to(asm_dir)
        # "everyday/node.exe.asm" -> family "everyday", source PE "node.exe"
        family = "_".join(rel.parts[:-1]) if len(rel.parts) > 1 else "root"
        pe_name = rel.name[:-4] if rel.name.endswith(".asm") else rel.name
        txt_name = f"{family}_{pe_name}.txt"

        prior = collisions.get(txt_name)
        if prior:
            # Should be impossible given the family prefix, but a silent
            # overwrite here would delete a sample, so fail loudly instead.
            sys.exit(f"output name collision: {txt_name}\n  {prior}\n  {rel}")
        collisions[txt_name] = str(rel)

        n_in, n_out = convert_file(src, out_dir / txt_name)

        pe_rel = str(rel)[: -len(".asm")].replace("\\", "/")
        meta = index.get(pe_rel, {})
        rows.append({
            "txt_file": txt_name,
            "asm_rel_path": str(rel).replace("\\", "/"),
            "pe_rel_path": pe_rel,
            "family": family,
            "bucket": meta.get("bucket", family),
            "subcategory": meta.get("subcategory", ""),
            "entry_id": meta.get("entry_id", ""),
            "vendor": meta.get("vendor", ""),
            "arch": meta.get("arch", ""),
            "sha256": meta.get("sha256", ""),
            "lines_in": n_in,
            "instructions": n_out,
        })
        if i % 200 == 0 or i == len(asm_files):
            print(f"  {i}/{len(asm_files)}", flush=True)

    if args.manifest:
        mpath = Path(args.manifest)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        with mpath.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"manifest: {mpath}")

    empty = [r for r in rows if r["instructions"] == 0]
    no_meta = [r for r in rows if index and not r["sha256"]]
    print(f"\nwrote {len(rows)} .txt to {out_dir}")
    print(f"  empty (0 instructions): {len(empty)}")
    if index:
        print(f"  unmatched in corpus_index: {len(no_meta)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
