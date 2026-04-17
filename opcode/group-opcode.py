#!/usr/bin/env python3
"""
Goal: Scan a folder of ransomware assembly files, extract opcodes, group them into categories, and save summaries.

Usage:
    python group_opcodes.py /path/to/asm_folder

Outputs:
    results/opcode_counts_per_file.csv
    results/opcode_category_counts_per_file.csv
    results/overall_opcode_counts.csv
    results/overall_category_counts.csv
    results/summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Common x86/x64 opcode categories.
OPCODE_GROUPS = {
    "data_transfer": {
        "mov", "movzx", "movsx", "lea", "xchg", "cmovz", "cmovnz", "cmova",
        "cmovb", "cmovg", "cmovl", "push", "pop", "pusha", "popa"
    },
    "arithmetic": {
        "add", "sub", "inc", "dec", "mul", "imul", "div", "idiv", "adc", "sbb",
        "neg", "cmp"
    },
    "logic_bitwise": {
        "and", "or", "xor", "not", "test", "shl", "shr", "sar", "sal", "rol",
        "ror", "rcl", "rcr", "bt", "bts", "btr", "btc"
    },
    "control_flow": {
        "jmp", "je", "jne", "jz", "jnz", "ja", "jb", "jg", "jl", "jge", "jle",
        "jo", "jno", "js", "jns", "jc", "jnc", "call", "ret", "loop", "loope",
        "loopne"
    },
    "stack_frame": {
        "enter", "leave"
    },
    "string_ops": {
        "movs", "movsb", "movsw", "movsd", "movsq", "stos", "stosb", "stosw",
        "stosd", "stosq", "lods", "lodsb", "lodsw", "lodsd", "lodsq", "scas",
        "scasb", "scasw", "scasd", "scasq", "cmps", "cmpsb", "cmpsw", "cmpsd",
        "cmpsq"
    },
    "flags": {
        "clc", "stc", "cmc", "cld", "std", "cli", "sti", "lahf", "sahf", "pushf",
        "popf"
    },
    "system": {
        "int", "syscall", "sysenter", "sysexit", "cpuid", "hlt", "nop", "ud2"
    },
    "floating_point_simd": {
        "fld", "fst", "fstp", "fadd", "fsub", "fmul", "fdiv", "pxor", "movdqa",
        "movdqu", "movaps", "movups", "paddb", "paddw", "paddd"
    },
    "crypto_like": {
        # Not true "crypto instructions" in all cases, but useful for malware triage
        "aesenc", "aesenclast", "aesdec", "aesdeclast", "aesimc", "aeskeygenassist",
        "rdrand", "rdseed"
    },
}

# Flatten reverse lookup.
OPCODE_TO_GROUP: Dict[str, str] = {}
for group_name, opcodes in OPCODE_GROUPS.items():
    for opcode in opcodes:
        OPCODE_TO_GROUP[opcode] = group_name

# Regex helpers.
LABEL_RE = re.compile(r"^\s*[A-Za-z_.$?@][\w.$?@]*:\s*$")
HEX_PREFIX_RE = re.compile(r"^(loc_|sub_|off_|byte_|word_|dword_|qword_|unk_)", re.IGNORECASE)
MNEMONIC_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")

# Lines often found in disassembly output that are not opcodes.
NON_OPCODE_TOKENS = {
    "db", "dw", "dd", "dq", "dt",
    "align", "assume", "end", "ends", "segment", "proc", "endp",
    "public", "extrn", "extern", "model", "include", "equ",
}


def is_probable_opcode(token: str) -> bool:
    token = token.lower().strip()
    if not token:
        return False
    if token in NON_OPCODE_TOKENS:
        return False
    if HEX_PREFIX_RE.match(token):
        return False
    if not MNEMONIC_RE.match(token):
        return False
    return True


def normalize_opcode(token: str) -> str:
    token = token.lower().strip()
    # Normalize REP-prefixed string ops if present like "rep movsb"
    return token


def extract_opcode_from_line(line: str) -> str | None:
    # Remove comments ; ... and # ...
    line = line.split(";", 1)[0].split("#", 1)[0].strip()
    if not line:
        return None

    if LABEL_RE.match(line):
        return None

    # Split by whitespace and commas.
    parts = line.split()
    if not parts:
        return None

    # Handle possible address/bytes prefixes in disassembler output.
    # Example: "00401000 8B45FC mov eax, [ebp-4]"
    # or "text:00401000 mov eax, ebx"
    candidates = []

    for part in parts[:6]:
        cleaned = part.strip().rstrip(":").lower()
        if cleaned:
            candidates.append(cleaned)

    for i, token in enumerate(candidates):
        # Skip addresses and hex bytes.
        if re.fullmatch(r"[0-9a-f]+h?", token):
            continue
        if re.fullmatch(r"[0-9a-f]{2}", token):
            continue
        if ":" in token:
            continue

        if is_probable_opcode(token):
            return normalize_opcode(token)

    # Fallback: sometimes opcode is the first token before operands with tabs.
    first = re.split(r"[\s,]+", line)[0].lower()
    if is_probable_opcode(first):
        return normalize_opcode(first)

    return None


def group_opcode(opcode: str) -> str:
    return OPCODE_TO_GROUP.get(opcode, "other")


def scan_file(path: Path) -> Tuple[Counter, Counter]:
    opcode_counts: Counter = Counter()
    category_counts: Counter = Counter()

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                opcode = extract_opcode_from_line(line)
                if opcode:
                    opcode_counts[opcode] += 1
                    category_counts[group_opcode(opcode)] += 1
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")

    return opcode_counts, category_counts


def write_csv(path: Path, header: List[str], rows: Iterable[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Group opcodes from assembly files.")
    parser.add_argument("input_folder", help="Folder containing assembly files")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".asm", ".s", ".txt"],
        help="File extensions to scan (default: .asm .s .txt)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Top N opcodes to display per file in summary JSON (default: 15)",
    )
    parser.add_argument(
        "--output",
        default="opcode_results",
        help="Output folder (default: opcode_results)",
    )
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output)

    if not input_folder.exists() or not input_folder.is_dir():
        raise SystemExit(f"Input folder does not exist or is not a directory: {input_folder}")

    extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.extensions}

    per_file_opcode_counts: Dict[str, Counter] = {}
    per_file_category_counts: Dict[str, Counter] = {}
    overall_opcode_counts: Counter = Counter()
    overall_category_counts: Counter = Counter()

    files_scanned = 0

    for root, _, files in os.walk(input_folder):
        for filename in files:
            path = Path(root) / filename
            if path.suffix.lower() not in extensions:
                continue

            files_scanned += 1
            opcode_counts, category_counts = scan_file(path)

            rel_path = str(path.relative_to(input_folder))
            per_file_opcode_counts[rel_path] = opcode_counts
            per_file_category_counts[rel_path] = category_counts
            overall_opcode_counts.update(opcode_counts)
            overall_category_counts.update(category_counts)

    if files_scanned == 0:
        raise SystemExit(f"No matching files found in {input_folder} for extensions: {sorted(extensions)}")

    # Write per-file opcode counts.
    opcode_rows = []
    for file_name, counts in per_file_opcode_counts.items():
        total = sum(counts.values())
        for opcode, count in counts.most_common():
            opcode_rows.append([file_name, opcode, count, total])

    write_csv(
        output_folder / "opcode_counts_per_file.csv",
        ["file", "opcode", "count", "total_opcodes_in_file"],
        opcode_rows,
    )

    # Write per-file category counts.
    category_rows = []
    for file_name, counts in per_file_category_counts.items():
        total = sum(counts.values())
        for category, count in counts.most_common():
            category_rows.append([file_name, category, count, total])

    write_csv(
        output_folder / "opcode_category_counts_per_file.csv",
        ["file", "category", "count", "total_classified_opcodes_in_file"],
        category_rows,
    )

    # Write overall opcode counts.
    write_csv(
        output_folder / "overall_opcode_counts.csv",
        ["opcode", "count"],
        [[opcode, count] for opcode, count in overall_opcode_counts.most_common()],
    )

    # Write overall category counts.
    write_csv(
        output_folder / "overall_category_counts.csv",
        ["category", "count"],
        [[category, count] for category, count in overall_category_counts.most_common()],
    )

    # JSON summary.
    summary = {
        "files_scanned": files_scanned,
        "extensions": sorted(extensions),
        "overall_top_opcodes": overall_opcode_counts.most_common(args.top),
        "overall_category_counts": dict(overall_category_counts.most_common()),
        "per_file": {},
    }

    for file_name in sorted(per_file_opcode_counts):
        summary["per_file"][file_name] = {
            "top_opcodes": per_file_opcode_counts[file_name].most_common(args.top),
            "category_counts": dict(per_file_category_counts[file_name].most_common()),
            "total_opcodes": sum(per_file_opcode_counts[file_name].values()),
        }

    output_folder.mkdir(parents=True, exist_ok=True)
    with (output_folder / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Console summary.
    print(f"Scanned {files_scanned} files")
    print("\nTop opcodes overall:")
    for opcode, count in overall_opcode_counts.most_common(args.top):
        print(f"  {opcode:<12} {count}")

    print("\nOpcode groups overall:")
    for category, count in overall_category_counts.most_common():
        print(f"  {category:<20} {count}")

    print(f"\nResults written to: {output_folder.resolve()}")


if __name__ == "__main__":
    main()
