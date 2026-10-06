#!/usr/bin/env python3
"""
opcode_sequences.py

Shared readers for the opcode index written by extract_opcodes.py.

Every later step (tokenization, CNN-ViT images, obfuscation, the GIN graphs)
starts from that index, so the train / val / test assignment stays the one
extract_opcodes.py or split_dataset.py already chose.
"""

import csv
from pathlib import Path


INDEX_FIELDS = [
    "sha256",
    "split",
    "label",
    "label_name",
    "family",
    "arch",
    "n_instructions",
    "n_unique_opcodes",
    "source_path",
    "opcode_file",
]

SPLITS = ("train", "val", "test")

# Capstone reports these as part of the mnemonic ("lock xadd", "rep movsb", "bnd ret").
# They are not operands. Operand lines from --include-operands look like
# "mov eax, ebx" or "push ebp".
PREFIXES = {"lock", "rep", "repe", "repz", "repne", "repnz", "bnd"}


def line_has_operands(line):
    """True when a line is `mnemonic operands` rather than a bare mnemonic."""
    parts = line.split()
    if len(parts) <= 1:
        return False
    if "," in line:
        return True
    if parts[0].lower() in PREFIXES:
        return False
    return True


def opcode_tokens(line):
    """
    Opcode pieces from one line. Operand text is dropped. A prefixed mnemonic
    such as `lock xadd` contributes both pieces.
    """
    parts = line.split()
    if not parts:
        return []
    if line_has_operands(line):
        return [parts[0].lower()]
    return [part.lower() for part in parts]


def read_index(path):
    """Reads an opcode index CSV. Raises SystemExit if it is missing or empty."""
    index_path = Path(path)
    if not index_path.is_file():
        raise SystemExit(f"{index_path} not found. Run extract_opcodes.py first.")
    with open(index_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{index_path} has no rows")
    if "split" not in rows[0] or "opcode_file" not in rows[0]:
        raise SystemExit(
            f"{index_path} needs 'split' and 'opcode_file' columns. "
            "Run extract_opcodes.py first."
        )
    return rows


def apply_limit(rows, limit):
    """Keeps up to `limit` rows of each split, in index order. 0 keeps every row."""
    if not limit:
        return rows
    seen = {}
    kept = []
    for row in rows:
        split = row["split"]
        if seen.get(split, 0) >= limit:
            continue
        seen[split] = seen.get(split, 0) + 1
        kept.append(row)
    return kept


def read_mnemonics(path, max_instructions):
    """Reads one opcode file into a list of non-empty lines, optionally truncated."""
    mnemonics = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            mnemonics.append(line)
            if max_instructions and len(mnemonics) >= max_instructions:
                break
    return mnemonics


def load_sequences(rows, max_instructions):
    """
    Returns (loaded, missing).

    loaded is a list of (row, mnemonics) in index order.
    missing is a list of sha256 values whose opcode file could not be read.
    """
    loaded = []
    missing = []
    total = len(rows)
    for i, row in enumerate(rows, start=1):
        path = row["opcode_file"]
        try:
            mnemonics = read_mnemonics(path, max_instructions)
        except OSError:
            missing.append(row["sha256"])
            continue
        if not mnemonics:
            missing.append(row["sha256"])
            continue
        loaded.append((row, mnemonics))
        if i % 250 == 0 or i == total:
            print(f"  {i}/{total} opcode files read", flush=True)
    return loaded, missing


def write_index(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
