#!/usr/bin/env python3
"""
extract_imports.py - the import-table side channel for the sequence model.

Two modes, both read-only; a sample is *parsed*, never executed.

Extraction
----------
    python imports/extract_imports.py \
        --in  C:/Users/chaoa/Downloads/Goodware_Balanced \
        --out manifests/imports/goodware_balanced.json \
        --manifest manifests/imports/goodware_balanced.csv \
        --exclude _upx_packed --exclude flagged

Walks a folder of PEs (hidden directories and `--exclude`d names pruned) and
writes, keyed by SHA-256:

    {"<sha256>": {"imports": ["kernel32.dll!createfilew", ...],
                  "iat":     {"0x4291f0": "kernel32.dll!createfilew", ...},
                  "dll_count": 12, "func_count": 310, "error": ""}}

`imports` is the ordered, de-duplicated import list (import directory first,
then delay-load). `iat` maps the *virtual address* of each IAT slot -
ImageBase + slot RVA, formatted the way the disassembler prints operands
(`0x` + lowercase hex, unpadded) - to the same `dll!func` string, so that a
`call dword ptr [0x4291f0]` or `call qword ptr [rip + 0x1234]` in
`asm_output/unified_*` can be turned back into an API name. Names are
lowercased, the DLL prefix is kept, and an ordinal import is `dll!#123`.

Verification
------------
    python imports/extract_imports.py --verify-asm ASM_ROOT --json FILE.json \
        [--limit 20] [--seed 0]

Samples `.asm` transcripts named `<sha256>.asm`, pulls every memory-indirect
`call`/`jmp` target out of them, and reports what fraction resolves through
the `iat` map. Absolute operands (`[0x4291f0]`) are used as-is; RIP-relative
operands (`[rip + 0xd639]`) are resolved against the *next* instruction's
address, which is the address printed on the following line of the transcript.

Parsing is `pefile.PE(fast_load=True)` plus the two import directories only,
so a 1,100-file corpus takes seconds rather than minutes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

import pefile

# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

_IMPORT_DIRS = [
    pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
    pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"],
]

MANIFEST_FIELDS = ["sha256", "rel_path", "status", "dll_count", "func_count"]

#: Default cap (MB) on the main JSON before the IAT map is spilled to its own
#: `<stem>.iat.json`.  The sequence model loads `imports` for every sample but
#: only needs `iat` when it is resolving a transcript.
DEFAULT_IAT_SPLIT_MB = 50.0


def _decode(raw: bytes | None) -> str:
    """Bytes from a PE string table -> a lowercase str, never raising."""
    if not raw:
        return ""
    return raw.split(b"\x00", 1)[0].decode("latin-1", "replace").strip().lower()


def _symbol(dll: str, imp) -> str:
    """`dll!func`, or `dll!#123` for an ordinal import."""
    if getattr(imp, "import_by_ordinal", False) or not imp.name:
        ordinal = getattr(imp, "ordinal", None)
        return f"{dll}!#{ordinal}" if ordinal is not None else f"{dll}!#?"
    return f"{dll}!{_decode(imp.name)}"


def format_va(address: int) -> str:
    """IAT slot VA the way the disassembly prints it: `0x` + lowercase hex."""
    return f"0x{address:x}"


def _entries(pe, attr: str):
    for entry in getattr(pe, attr, None) or []:
        dll = _decode(getattr(entry, "dll", None))
        if not dll:
            continue
        for imp in entry.imports or []:
            yield dll, imp


def extract_pe(data: bytes) -> dict:
    """Import table + IAT of one PE image.  Raises only on unparseable input."""
    pe = pefile.PE(data=data, fast_load=True)
    try:
        pe.parse_data_directories(directories=_IMPORT_DIRS)
        base = int(pe.OPTIONAL_HEADER.ImageBase)
        span = int(getattr(pe.OPTIONAL_HEADER, "SizeOfImage", 0) or 0)

        symbols: dict[str, None] = {}          # ordered set
        iat: dict[str, str] = {}
        dlls: dict[str, None] = {}

        for attr in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT"):
            for dll, imp in _entries(pe, attr):
                sym = _symbol(dll, imp)
                dlls[dll] = None
                symbols[sym] = None
                addr = getattr(imp, "address", None)
                if not addr:
                    continue
                addr = int(addr)
                # pefile already folds ImageBase into `address`; a handful of
                # old-style delay-load tables store plain RVAs, which show up
                # as an address below ImageBase.  Fold those in by hand.
                if span and addr < base:
                    addr += base
                if span and not (base <= addr < base + span):
                    continue
                iat[format_va(addr)] = sym

        return {
            "imports": list(symbols),
            "iat": iat,
            "dll_count": len(dlls),
            "func_count": len(symbols),
            "error": "",
        }
    finally:
        pe.close()


def empty_record(error: str = "") -> dict:
    return {"imports": [], "iat": {}, "dll_count": 0, "func_count": 0,
            "error": error}


def iter_files(root: Path, exclude: set[str]):
    """Regular files under `root`, hidden and excluded directories pruned."""
    lowered = {e.lower() for e in exclude}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith(".") and d.lower() not in lowered)
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            yield Path(dirpath) / name


def is_mz(data: bytes) -> bool:
    return data[:2] == b"MZ"


def extract_dir(root: Path, exclude: set[str], progress_every: int = 250):
    """-> (records keyed by sha256, manifest rows).  One row per input file."""
    records: dict[str, dict] = {}
    rows: list[dict] = []
    seen = 0

    for path in iter_files(root, exclude):
        seen += 1
        rel = path.relative_to(root).as_posix()
        try:
            data = path.read_bytes()
        except OSError as exc:
            rows.append({"sha256": "", "rel_path": rel, "status": "read_error",
                         "dll_count": 0, "func_count": 0})
            print(f"  read_error {rel}: {exc}", file=sys.stderr)
            continue

        sha = hashlib.sha256(data).hexdigest()
        if not is_mz(data):
            rows.append({"sha256": sha, "rel_path": rel, "status": "not_pe",
                         "dll_count": 0, "func_count": 0})
            continue

        try:
            rec = extract_pe(data)
            status = "ok" if rec["func_count"] else "no_imports"
        except Exception as exc:                        # pefile raises broadly
            rec = empty_record(f"{type(exc).__name__}: {exc}"[:300])
            status = "parse_error"

        # Duplicate content under two paths: keep the first, still emit a row.
        records.setdefault(sha, rec)
        rows.append({"sha256": sha, "rel_path": rel, "status": status,
                     "dll_count": rec["dll_count"],
                     "func_count": rec["func_count"]})
        if progress_every and seen % progress_every == 0:
            print(f"  {seen} files, {len(records)} unique PEs", file=sys.stderr)

    return records, rows


def write_manifest(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def dumps(mapping: dict) -> str:
    """A JSON object with one compact line per sha256 - small, and still
    greppable by hash without loading 40 MB into a viewer."""
    if not mapping:
        return "{}\n"
    lines = [f"{json.dumps(sha)}:{json.dumps(val, separators=(',', ':'))}"
             for sha, val in sorted(mapping.items())]
    return "{\n" + ",\n".join(lines) + "\n}\n"


def write_json(records: dict, out: Path, split_mb: float = DEFAULT_IAT_SPLIT_MB):
    """Write `out`, spilling the IAT maps to `<stem>.iat.json` if it is large.

    -> (main_path, iat_path or None).
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = dumps(records)
    iat_path = None
    if split_mb and len(blob.encode("utf-8")) > split_mb * 1e6:
        iat_path = out.with_suffix(".iat.json")
        iat_path.write_text(
            dumps({sha: rec.get("iat", {}) for sha, rec in records.items()}),
            encoding="utf-8")
        # Copy rather than pop: `records` belongs to the caller.
        blob = dumps({sha: {k: v for k, v in rec.items() if k != "iat"}
                      for sha, rec in records.items()})
    out.write_text(blob, encoding="utf-8")
    return out, iat_path


# ---------------------------------------------------------------------------
# .asm transcript resolution
# ---------------------------------------------------------------------------

LINE_RE = re.compile(r"^\s*0x([0-9a-fA-F]+)\s*:\s*(\S+)\s*(.*?)\s*$")
ABS_MEM_RE = re.compile(r"\[\s*(0x[0-9a-fA-F]+)\s*\]")
RIP_MEM_RE = re.compile(r"\[\s*rip\s*([+-])\s*(0x[0-9a-fA-F]+)\s*\]")
INDIRECT_MNEMONICS = {"call", "jmp", "bnd"}


def parse_asm(text: str):
    """`0xADDR:  mnem\\toperands` lines -> [(addr, mnemonic, operands)]."""
    out = []
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        out.append((int(m.group(1), 16), m.group(2).lower(), m.group(3)))
    return out


def indirect_targets(instructions):
    """Memory-indirect call/jmp slot addresses in a parsed transcript.

    -> [(instr_addr, kind, target or None)] where `kind` is "abs" or "rip".
    A RIP-relative target needs the address of the *next* instruction, which
    is the address printed on the following line; at the end of a transcript
    (or a `.asm` that was truncated by the instruction cap) there is no next
    line and the target is reported as None.
    """
    found = []
    for i, (addr, mnem, ops) in enumerate(instructions):
        if mnem not in INDIRECT_MNEMONICS:
            continue
        if mnem == "bnd":                     # `bnd jmp qword ptr [rip + ..]`
            parts = ops.split(None, 1)
            if len(parts) != 2 or parts[0].lower() not in ("jmp", "call"):
                continue
            ops = parts[1]
        rip = RIP_MEM_RE.search(ops)
        if rip:
            nxt = instructions[i + 1][0] if i + 1 < len(instructions) else None
            disp = int(rip.group(2), 16)
            target = None if nxt is None else (
                nxt + disp if rip.group(1) == "+" else nxt - disp)
            found.append((addr, "rip", target))
            continue
        absolute = ABS_MEM_RE.search(ops)
        if absolute:
            found.append((addr, "abs", int(absolute.group(1), 16)))
    return found


def resolve_asm(text: str, iat: dict[str, str]):
    """-> (resolved, total, [(instr_addr, kind, target)] unresolved)."""
    targets = indirect_targets(parse_asm(text))
    resolved, misses = 0, []
    for addr, kind, target in targets:
        if target is not None and format_va(target) in iat:
            resolved += 1
        else:
            misses.append((addr, kind, target))
    return resolved, len(targets), misses


def classify_image(path: Path):
    """-> (cfg guard pointer VAs, [(section name, start VA, end VA)]).

    The Control Flow Guard check/dispatch pointers live in the load-config
    directory, not the import table, but MSVC calls them exactly like an
    import (`call qword ptr [__guard_dispatch_icall_fptr]`) in front of every
    indirect call - which makes them the bulk of the unresolved targets.
    """
    pe = pefile.PE(str(path), fast_load=True)
    try:
        pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY[
                "IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG"]])
        base = int(pe.OPTIONAL_HEADER.ImageBase)
        guards = set()
        lc = getattr(pe, "DIRECTORY_ENTRY_LOAD_CONFIG", None)
        if lc is not None:
            for field in ("GuardCFCheckFunctionPointer",
                          "GuardCFDispatchFunctionPointer"):
                va = int(getattr(lc.struct, field, 0) or 0)
                if va:
                    guards.add(va)
        sections = [(s.Name.rstrip(b"\x00").decode("latin-1"),
                     base + s.VirtualAddress,
                     base + s.VirtualAddress
                     + max(s.Misc_VirtualSize, s.SizeOfRawData))
                    for s in pe.sections]
        return guards, sections
    finally:
        pe.close()


def _verify(asm_root: Path, records: dict, limit: int, seed: int,
            corpus: Path | None = None, index: Path | None = None) -> int:
    asms = sorted(p for p in asm_root.rglob("*.asm")
                  if p.stem in records and records[p.stem].get("iat"))
    if not asms:
        print("no .asm transcripts whose sha256 is in the JSON", file=sys.stderr)
        return 1
    random.Random(seed).shuffle(asms)
    asms = asms[:limit]

    rel_by_sha: dict[str, str] = {}
    if index:
        with Path(index).open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                rel_by_sha.setdefault(row["sha256"], row["rel_path"])

    tot_res = tot = 0
    by_kind = {"abs": [0, 0], "rip": [0, 0]}      # [resolved, total]
    reasons: dict[str, int] = {}
    hot: dict[str, int] = {}
    print(f"{'sha256':<18}{'resolved':>10}{'targets':>9}  rate")
    for path in sorted(asms):
        sha = path.stem
        iat = records[sha]["iat"]
        targets = indirect_targets(
            parse_asm(path.read_text(encoding="utf-8", errors="replace")))

        guards, sections = set(), []
        if corpus and sha in rel_by_sha:
            try:
                guards, sections = classify_image(corpus / rel_by_sha[sha])
            except Exception as exc:
                print(f"  (load-config unreadable for {sha[:16]}: {exc})",
                      file=sys.stderr)

        res = 0
        for _addr, kind, target in targets:
            by_kind[kind][1] += 1
            if target is not None and format_va(target) in iat:
                res += 1
                by_kind[kind][0] += 1
                continue
            if target is None:
                reason = "no next instruction (transcript end / 100k cap)"
            elif target in guards:
                reason = "CFG guard pointer (__guard_check/dispatch_icall)"
            elif sections:
                sec = next((n for n, lo, hi in sections if lo <= target < hi),
                           "<outside image>")
                reason = f"function pointer in {sec}"
            else:
                reason = "unresolved"
                key = f"{sha[:16]}@{format_va(target)}"
                hot[key] = hot.get(key, 0) + 1
            reasons[reason] = reasons.get(reason, 0) + 1
        tot_res += res
        tot += len(targets)
        rate = f"{res / len(targets):.1%}" if targets else "-"
        print(f"{sha[:16]:<18}{res:>10}{len(targets):>9}  {rate}")

    print(f"\n{len(asms)} transcripts: {tot_res}/{tot} memory-indirect "
          f"call/jmp targets resolved"
          + (f" ({tot_res / tot:.2%})" if tot else ""))
    for kind, (r, t) in by_kind.items():
        if t:
            print(f"  {kind:<4} {r}/{t} ({r / t:.2%})")
    if reasons:
        print("  misses:")
        for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {n:>6}  {reason}")
    if hot:
        print("  most-repeated unclassified miss slots "
              "(pass --corpus/--index to classify):")
        for key, n in sorted(hot.items(), key=lambda kv: -kv[1])[:5]:
            print(f"    {n:>6}  {key}")
    return 0


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="PE import table / IAT extraction and .asm IAT resolution")
    ap.add_argument("--in", dest="in_dir", help="folder of PE files to walk")
    ap.add_argument("--out", help="output JSON keyed by sha256")
    ap.add_argument("--manifest", help="output CSV, one row per input file")
    ap.add_argument("--exclude", action="append", default=[],
                    metavar="NAME", help="directory name to prune (repeatable)")
    ap.add_argument("--iat-split-mb", type=float, default=DEFAULT_IAT_SPLIT_MB,
                    help="spill the iat maps to <stem>.iat.json above this "
                         "size; 0 disables (default: %(default)s)")
    ap.add_argument("--verify-asm", metavar="DIR",
                    help="verify mode: resolve call/jmp targets in <sha>.asm "
                         "transcripts under DIR against --json")
    ap.add_argument("--json", help="verify mode: the JSON written by --out")
    ap.add_argument("--limit", type=int, default=20,
                    help="verify mode: transcripts to sample (default: 20)")
    ap.add_argument("--seed", type=int, default=0, help="verify mode: rng seed")
    ap.add_argument("--corpus", metavar="DIR",
                    help="verify mode: corpus root, so unresolved targets can "
                         "be classified against the load-config directory and "
                         "the section table (needs --index)")
    ap.add_argument("--index", metavar="CSV",
                    help="verify mode: the --manifest CSV, for sha256 -> path")
    a = ap.parse_args(argv)

    if a.verify_asm:
        if not a.json:
            ap.error("--verify-asm needs --json")
        records = json.loads(Path(a.json).read_text(encoding="utf-8"))
        side = Path(a.json).with_suffix(".iat.json")
        if side.is_file():
            for sha, iat in json.loads(side.read_text(encoding="utf-8")).items():
                if sha in records:
                    records[sha]["iat"] = iat
        return _verify(Path(a.verify_asm), records, a.limit, a.seed,
                       Path(a.corpus) if a.corpus else None,
                       Path(a.index) if a.index else None)

    if not a.in_dir or not a.out:
        ap.error("--in and --out are required")
    root = Path(a.in_dir)
    if not root.is_dir():
        ap.error(f"not a directory: {root}")

    records, rows = extract_dir(root, set(a.exclude))
    out, iat_path = write_json(records, Path(a.out), a.iat_split_mb)
    if a.manifest:
        write_manifest(rows, Path(a.manifest))

    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print(f"{len(rows)} files, {len(records)} unique PEs -> {out} "
          f"({out.stat().st_size / 1e6:.1f} MB)")
    if iat_path:
        print(f"  iat maps spilled to {iat_path} "
              f"({iat_path.stat().st_size / 1e6:.1f} MB)")
    if a.manifest:
        print(f"  manifest -> {a.manifest}")
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
