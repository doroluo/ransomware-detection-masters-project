#!/usr/bin/env python3
"""
obfuscate_opcodes.py

Writes a second opcode index whose chosen splits have been edited with the
functionality-preserving obfuscations from the CNN-ViT adversarial evaluation,
adapted to the opcode files this repo actually stores.

Bare mnemonic files (the default output of extract_opcodes.py) can take two
edits that stay functionality-preserving without operand text:

    nop        insert a NOP sled, capped at --max-growth of the original length
    synonym    swap assembler-equivalent mnemonics (je/jz, jne/jnz, shl/sal, ...)

Train and val are left pointing at the original opcode files unless named in
--splits, so a model fit on clean training data can be scored on an edited
test set. Rebuild both representations from the new index:

    python build_graphs.py --index data/opcodes_obfuscated/index.csv --out data/graphs_obfuscated
    python tokenize_opcodes.py --index data/opcodes_obfuscated/index.csv --out data/tokens_obfuscated

Operand substitutions from the CNN-ViT attack set (xor reg, reg → sub reg, reg,
inc → add 1, push/pop expanded into mov, and the other replacements) plus
identity dead-code lines run only when a file contains `mnemonic operands`
lines, which extract_opcodes.py writes with --include-operands. Inserting
those as bare mnemonics would look like a real `add` or `mov` to the GIN.

Usage:

    python obfuscate_opcodes.py --index data/opcodes_dataset/index.csv
    python obfuscate_opcodes.py --splits test --nop-rate 0.03 --synonym-rate 0.3
"""

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

from opcode_sequences import apply_limit, line_has_operands, read_index, read_mnemonics, write_index


# Each pair is the same instruction under two mnemonics.
SYNONYM_PAIRS = (
    ("je", "jz"),
    ("jne", "jnz"),
    ("jb", "jnae"),
    ("jae", "jnb"),
    ("jbe", "jna"),
    ("ja", "jnbe"),
    ("jl", "jnge"),
    ("jge", "jnl"),
    ("jle", "jng"),
    ("jg", "jnle"),
    ("jp", "jpe"),
    ("jnp", "jpo"),
    ("shl", "sal"),
    ("repz", "repe"),
    ("repnz", "repne"),
)

SYNONYM = {}
for _left, _right in SYNONYM_PAIRS:
    SYNONYM[_left] = _right
    SYNONYM[_right] = _left

REGISTER = re.compile(
    r"^(e?[abcd]x|e?[sd]i|e?[sb]p|esp|r\d+[dwb]?|[abcd][lh])$",
    re.IGNORECASE,
)

# Identity instructions. Only written when the file already stores operands,
# so the line still describes a no-op instead of a bare mnemonic.
DEAD_CODE = (
    "add eax, 0",
    "sub ebx, 0",
    "and ecx, ecx",
    "or edx, edx",
    "mov esi, esi",
    "xchg edi, edi",
    "shl eax, 0",
    "shr ebx, 0",
)


def split_operands(operands):
    return [part.strip() for part in operands.split(",")]


def both_registers(operands):
    parts = split_operands(operands)
    if len(parts) != 2:
        return None
    if REGISTER.fullmatch(parts[0]) and REGISTER.fullmatch(parts[1]):
        return parts
    return None


def substitute_xor(operands):
    parts = split_operands(operands)
    if len(parts) == 2 and parts[0].lower() == parts[1].lower():
        return [f"sub {parts[0]}, {parts[0]}"]
    return None


def substitute_inc(operands):
    return [f"add {operands}, 1"]


def substitute_dec(operands):
    return [f"sub {operands}, 1"]


def substitute_neg(operands):
    return [f"not {operands}", f"add {operands}, 1"]


def substitute_not(operands):
    return [f"xor {operands}, 0FFFFFFFFh"]


def substitute_test(operands):
    parts = split_operands(operands)
    if len(parts) == 2 and parts[0].lower() == parts[1].lower():
        return [f"cmp {parts[0]}, 0"]
    return None


def substitute_push(operands):
    return ["sub esp, 4", f"mov [esp], {operands}"]


def substitute_pop(operands):
    return [f"mov {operands}, [esp]", "add esp, 4"]


def substitute_mov(operands):
    parts = both_registers(operands)
    if parts is None:
        return None
    src, dst = parts[1], parts[0]
    return [f"push {src}", f"pop {dst}"]


def substitute_xchg(operands):
    parts = both_registers(operands)
    if parts is None:
        return None
    first, second = parts
    return [f"push {first}", f"push {second}", f"pop {first}", f"pop {second}"]


def substitute_lea(operands):
    parts = split_operands(operands)
    if len(parts) != 2:
        return None
    dst, src = parts
    match = re.fullmatch(r"\[(\w+)(?:\+0)?\]", src)
    if match:
        return [f"mov {dst}, {match.group(1)}"]
    return None


SUBSTITUTIONS = {
    "xor": substitute_xor,
    "inc": substitute_inc,
    "dec": substitute_dec,
    "neg": substitute_neg,
    "not": substitute_not,
    "test": substitute_test,
    "push": substitute_push,
    "pop": substitute_pop,
    "mov": substitute_mov,
    "xchg": substitute_xchg,
    "lea": substitute_lea,
}


def split_line(line):
    mnemonic, _sep, operands = line.partition(" ")
    return mnemonic.lower(), operands


def obfuscate_lines(lines, rng, nop_rate, nop_length, synonym_rate, substitution_rate, dead_code_rate, max_growth):
    """
    Returns (new_lines, stats).

    stats counts nop instructions inserted, synonym swaps, operand
    substitutions, and dead-code lines inserted.
    """
    # `lock xadd` is still a bare mnemonic. Operand mode is `mov eax, ebx`.
    operand_mode = any(line_has_operands(line) for line in lines)
    # Budget is the number of extra lines NOP sleds and dead code may add.
    # Original instructions are always kept. Operand substitutions are not
    # charged against it: one instruction legitimately becomes several.
    extra_budget = max(0, math.ceil(len(lines) * max_growth) - len(lines))
    output = []
    stats = Counter()

    for line in lines:
        mnemonic, operands = split_line(line)
        emitted = None

        if operand_mode and operands and rng.random() < substitution_rate:
            replace = SUBSTITUTIONS.get(mnemonic)
            emitted = replace(operands) if replace else None
            if emitted:
                stats["substitutions"] += 1

        if emitted is None:
            if mnemonic in SYNONYM and rng.random() < synonym_rate:
                mnemonic = SYNONYM[mnemonic]
                stats["synonyms"] += 1
            emitted = [f"{mnemonic} {operands}".strip()]

        output.extend(emitted)

        if extra_budget <= 0:
            continue
        if rng.random() < nop_rate:
            insert = min(nop_length, extra_budget)
            output.extend(["nop"] * insert)
            stats["nops"] += insert
            extra_budget -= insert
        if operand_mode and extra_budget > 0 and rng.random() < dead_code_rate:
            output.append(rng.choice(DEAD_CODE))
            stats["dead_code"] += 1
            extra_budget -= 1

    return output, stats


def sample_rng(seed, sha256):
    return random.Random(f"{seed}:{sha256}")


def write_opcodes(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".txt.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def obfuscate_row(row, out_dir, args):
    source = Path(row["opcode_file"])
    mnemonics = read_mnemonics(source, args.max_instructions)
    edited, stats = obfuscate_lines(
        mnemonics,
        sample_rng(args.seed, row["sha256"]),
        args.nop_rate,
        args.nop_length,
        args.synonym_rate,
        args.substitution_rate,
        args.dead_code_rate,
        args.max_growth,
    )
    relative = Path("opcodes") / row["label_name"] / row["family"] / f"{row['sha256']}.txt"
    destination = out_dir / relative
    write_opcodes(destination, edited)

    updated = dict(row)
    updated["opcode_file"] = str(destination)
    updated["n_instructions"] = len(edited)
    updated["n_unique_opcodes"] = len({line.split(" ", 1)[0] for line in edited})
    return updated, stats


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Write an opcode index with functionality-preserving edits on the chosen splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--index", default="data/opcodes_dataset/index.csv")
    parser.add_argument("--out", default="data/opcodes_obfuscated")
    parser.add_argument("--splits", default="test",
                        help="comma-separated splits to edit; the others keep their original opcode files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nop-rate", type=float, default=0.03,
                        help="chance of inserting a NOP sled after an instruction")
    parser.add_argument("--nop-length", type=int, default=4)
    parser.add_argument("--synonym-rate", type=float, default=0.30,
                        help="chance of swapping an assembler-equivalent mnemonic")
    parser.add_argument("--substitution-rate", type=float, default=0.30,
                        help="chance of an operand substitution, when the file has operands")
    parser.add_argument("--dead-code-rate", type=float, default=0.03,
                        help="chance of inserting an identity instruction, when the file has operands")
    parser.add_argument("--max-growth", type=float, default=1.15,
                        help="longest edited sequence as a fraction of the original length")
    parser.add_argument("--max-instructions", type=int, default=0,
                        help="truncate each sequence before editing (0 = no limit)")
    parser.add_argument("--limit", type=int, default=0,
                        help="keep only this many samples from each split (0 = all)")
    args = parser.parse_args(argv)

    splits = {part.strip() for part in args.splits.split(",") if part.strip()}
    if not splits:
        parser.error("--splits needs at least one split name")
    args.splits = splits
    if args.max_growth < 1:
        parser.error("--max-growth must be at least 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    rows = apply_limit(read_index(args.index), args.limit)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    edited_rows = []
    totals = Counter()
    edited_count = 0
    total = len(rows)
    print(f"Editing splits {sorted(args.splits)} in {total} samples...")

    for i, row in enumerate(rows, start=1):
        if row["split"] not in args.splits:
            edited_rows.append(row)
            continue
        try:
            updated, stats = obfuscate_row(row, out_dir, args)
        except OSError as exc:
            print(f"Could not read {row['opcode_file']}: {exc}", file=sys.stderr)
            return 1
        edited_rows.append(updated)
        totals.update(stats)
        edited_count += 1
        if i % 100 == 0 or i == total:
            print(f"  {i}/{total} processed, {edited_count} edited", flush=True)

    edited_rows.sort(key=lambda row: (row["split"], row["label"], row["family"], row["sha256"]))
    write_index(out_dir / "index.csv", edited_rows)
    for split in ("train", "val", "test"):
        write_index(
            out_dir / f"{split}.csv",
            [row for row in edited_rows if row["split"] == split],
        )

    report = {
        "seed": args.seed,
        "splits": sorted(args.splits),
        "nop_rate": args.nop_rate,
        "nop_length": args.nop_length,
        "synonym_rate": args.synonym_rate,
        "substitution_rate": args.substitution_rate,
        "dead_code_rate": args.dead_code_rate,
        "max_growth": args.max_growth,
        "samples_edited": edited_count,
        "nops_inserted": totals["nops"],
        "synonym_swaps": totals["synonyms"],
        "operand_substitutions": totals["substitutions"],
        "dead_code_inserted": totals["dead_code"],
    }
    with open(out_dir / "obfuscation.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print("\n=== Obfuscation ===")
    print(f"samples edited : {edited_count}")
    print(f"nops inserted  : {totals['nops']}")
    print(f"synonym swaps  : {totals['synonyms']}")
    print(f"substitutions  : {totals['substitutions']} (operand files only)")
    print(f"dead code      : {totals['dead_code']} (operand files only)")
    print(f"\nWrote {out_dir / 'index.csv'}")
    print("Train and val rows still point at the original opcode files."
          if args.splits == {"test"} else
          "Unedited splits still point at the original opcode files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
