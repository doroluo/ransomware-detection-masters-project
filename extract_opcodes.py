#!/usr/bin/env python3
"""
extract_opcodes.py

Disassembles the PE executables in the dataset and writes one opcode sequence
per sample, then splits the samples into train / validation / test sets.

Layout produced under --out:

    opcodes/goodware/goodware/<sha256>.txt
    opcodes/ransomware/<family>/<sha256>.txt
    index.csv        every kept sample, with its split assignment
    train.csv        \
    val.csv           > the same rows, one file per split
    test.csv         /
    skipped.csv      files that could not be disassembled, with the reason
    duplicates.csv   files dropped because an identical binary was kept

Samples are de-duplicated by SHA-256 before splitting, so the exact same binary
can never land in two different splits.

Packed samples should be unpacked first, otherwise the packer stub is all that
gets disassembled:

    find . -type f \\( -name "*.exe" -o -name "*.dll" \\) -exec upx -d {} +

Usage:

    python extract_opcodes.py --out data/opcodes_dataset
    python extract_opcodes.py --out /tmp/smoke --limit 40 --workers 4
"""

import argparse
import csv
import hashlib
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pefile
from capstone import (
    CS_ARCH_ARM64,
    CS_ARCH_X86,
    CS_MODE_32,
    CS_MODE_64,
    CS_MODE_ARM,
    Cs,
)

DATASET_ROOT = Path("data/Ransomware PE Header Feature Dataset/Executable Files")
DEFAULT_GOODWARE_DIR = DATASET_ROOT / "Goodware" / "goodware"
DEFAULT_RANSOMWARE_DIR = DATASET_ROOT / "Ransomware" / "Ransomware" / "rans"

IMAGE_SCN_MEM_EXECUTE = 0x20000000
IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR = 14

MACHINE_MODES = {
    0x014C: ("x86", CS_ARCH_X86, CS_MODE_32),
    0x8664: ("x64", CS_ARCH_X86, CS_MODE_64),
    0xAA64: ("arm64", CS_ARCH_ARM64, CS_MODE_ARM),
}

PACKER_SECTION_MARKERS = (b"UPX", b"ASPack", b".aspack", b"MPRESS", b".petite")

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


class SkipFile(Exception):
    """Raised when a file cannot be turned into an opcode sequence."""


def is_dotnet(pe):
    try:
        com = pe.OPTIONAL_HEADER.DATA_DIRECTORY[IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR]
    except (AttributeError, IndexError):
        return False
    return com.VirtualAddress != 0 and com.Size > 0


def packer_section(pe):
    for section in pe.sections:
        name = section.Name.rstrip(b"\x00")
        for marker in PACKER_SECTION_MARKERS:
            if marker in name:
                return name.decode("utf-8", "replace")
    return None


def disassemble(data, include_operands, max_instructions, allow_packed):
    """Returns (arch, [instruction strings]) or raises SkipFile."""
    try:
        pe = pefile.PE(data=data, fast_load=True)
    except pefile.PEFormatError as exc:
        raise SkipFile(f"not a valid PE ({exc})") from exc

    if is_dotnet(pe):
        raise SkipFile(".NET assembly (managed code, no native opcodes)")

    if not allow_packed:
        packed_as = packer_section(pe)
        if packed_as is not None:
            raise SkipFile(f"packed, section '{packed_as}' (unpack first)")

    machine = pe.FILE_HEADER.Machine
    if machine not in MACHINE_MODES:
        raise SkipFile(f"unsupported architecture ({hex(machine)})")
    arch, cs_arch, cs_mode = MACHINE_MODES[machine]

    md = Cs(cs_arch, cs_mode)
    instructions = []
    found_code = False

    for section in pe.sections:
        if not section.Characteristics & IMAGE_SCN_MEM_EXECUTE:
            continue
        found_code = True
        for _addr, _size, mnemonic, op_str in md.disasm_lite(
            section.get_data(), section.VirtualAddress
        ):
            if include_operands and op_str:
                instructions.append(f"{mnemonic} {op_str}")
            else:
                instructions.append(mnemonic)
            if max_instructions and len(instructions) >= max_instructions:
                return arch, instructions

    if not found_code:
        raise SkipFile("no executable sections")
    if not instructions:
        raise SkipFile("executable section disassembled to nothing")
    return arch, instructions


def process_file(task):
    """Worker: read one binary, write its opcode sequence, return a record."""
    (
        source_path,
        label,
        label_name,
        family,
        opcode_root,
        include_operands,
        max_instructions,
        allow_packed,
        skip_existing,
    ) = task

    source_path = Path(source_path)
    try:
        data = source_path.read_bytes()
    except OSError as exc:
        return {"status": "skipped", "source_path": str(source_path),
                "family": family, "label_name": label_name, "reason": f"unreadable ({exc})"}

    if data[:2] != b"MZ":
        return {"status": "skipped", "source_path": str(source_path), "family": family,
                "label_name": label_name, "reason": "not a PE file (no MZ header)"}

    sha256 = hashlib.sha256(data).hexdigest()
    out_path = Path(opcode_root) / label_name / family / f"{sha256}.txt"

    try:
        if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
            text = out_path.read_text(encoding="utf-8")
            instructions = text.split("\n")
            arch = "cached"
        else:
            arch, instructions = disassemble(
                data, include_operands, max_instructions, allow_packed
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.with_suffix(".txt.tmp")
            tmp_path.write_text("\n".join(instructions), encoding="utf-8")
            os.replace(tmp_path, out_path)
    except SkipFile as exc:
        return {"status": "skipped", "source_path": str(source_path), "family": family,
                "label_name": label_name, "reason": str(exc)}
    except Exception as exc:  # malformed headers surface as all sorts of errors
        return {"status": "skipped", "source_path": str(source_path), "family": family,
                "label_name": label_name, "reason": f"{type(exc).__name__}: {exc}"}

    opcodes = instructions
    if include_operands:
        opcodes = [line.split(" ", 1)[0] for line in instructions]

    return {
        "status": "ok",
        "sha256": sha256,
        "split": "",
        "label": label,
        "label_name": label_name,
        "family": family,
        "arch": arch,
        "n_instructions": len(instructions),
        "n_unique_opcodes": len(set(opcodes)),
        "source_path": str(source_path),
        "opcode_file": str(out_path),
    }


def collect_tasks(args, opcode_root):
    """Builds the worker task list from the goodware and ransomware directories."""
    tasks = []

    def walk(root, label, label_name, family_from_subdir):
        root = Path(root)
        if not root.is_dir():
            print(f"[!] {root} does not exist, skipping", file=sys.stderr)
            return
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            if family_from_subdir:
                rel = path.relative_to(root).parts
                family = rel[0] if len(rel) > 1 else "unknown"
            else:
                family = label_name
            tasks.append((
                str(path), label, label_name, family, str(opcode_root),
                args.include_operands, args.max_instructions,
                args.allow_packed, args.skip_existing,
            ))

    walk(args.goodware_dir, 0, "goodware", family_from_subdir=False)
    walk(args.ransomware_dir, 1, "ransomware", family_from_subdir=True)

    if args.limit:
        rng = random.Random(args.seed)
        rng.shuffle(tasks)
        tasks = tasks[: args.limit]
    return tasks


def split_sizes(n, ratios):
    """Largest-remainder allocation so small families still reach val/test."""
    raw = [n * r for r in ratios]
    sizes = [int(x) for x in raw]
    order = sorted(range(3), key=lambda i: raw[i] - sizes[i], reverse=True)
    for i in range(n - sum(sizes)):
        sizes[order[i % 3]] += 1
    return sizes


def assign_splits(records, ratios, seed, family_disjoint):
    """
    Default: stratify by (label, family) so every family appears in every split.
    --family-disjoint: keep whole families together, which measures how well a
    model generalises to ransomware families it has never seen.
    """
    rng = random.Random(seed)
    strata = defaultdict(lambda: defaultdict(list))

    for record in records:
        if family_disjoint:
            strata[record["label_name"]][record["family"]].append(record)
        else:
            strata[(record["label_name"], record["family"])][record["sha256"]].append(record)

    for stratum in sorted(strata, key=str):
        units = [strata[stratum][key] for key in sorted(strata[stratum])]
        rng.shuffle(units)
        train_n, val_n, _ = split_sizes(len(units), ratios)
        for i, unit in enumerate(units):
            if i < train_n:
                split = "train"
            elif i < train_n + val_n:
                split = "val"
            else:
                split = "test"
            for record in unit:
                record["split"] = split


def deduplicate(records):
    """Keeps one record per SHA-256; returns (kept, dropped)."""
    seen = {}
    kept, dropped = [], []
    for record in sorted(records, key=lambda r: r["source_path"]):
        first = seen.get(record["sha256"])
        if first is None:
            seen[record["sha256"]] = record
            kept.append(record)
        else:
            dropped.append({
                "sha256": record["sha256"],
                "dropped_path": record["source_path"],
                "kept_path": first["source_path"],
                "dropped_family": record["family"],
                "kept_family": first["family"],
            })
    return kept, dropped


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def link_splits(records, out_dir):
    """Mirrors the split assignment as a directory tree of symlinks."""
    for record in records:
        target = Path(record["opcode_file"]).resolve()
        link = out_dir / "splits" / record["split"] / record["label_name"] / record["family"] / target.name
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)


def print_summary(records, dropped, skipped):
    by_split = Counter(r["split"] for r in records)
    by_split_label = Counter((r["split"], r["label_name"]) for r in records)
    instructions = sum(r["n_instructions"] for r in records)

    print("\n=== Summary ===")
    print(f"samples kept   : {len(records)}")
    print(f"duplicates     : {len(dropped)}")
    print(f"skipped        : {len(skipped)}")
    print(f"instructions   : {instructions:,}")
    print("\nsplit      total   goodware  ransomware  families")
    for split in ("train", "val", "test"):
        families = {r["family"] for r in records if r["split"] == split and r["label"] == 1}
        print(f"{split:<10} {by_split[split]:<7} {by_split_label[(split, 'goodware')]:<9} "
              f"{by_split_label[(split, 'ransomware')]:<11} {len(families)}")

    if skipped:
        print("\ntop skip reasons:")
        for reason, count in Counter(s["reason"].split(" (")[0] for s in skipped).most_common(10):
            print(f"  {count:>5}  {reason}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract opcode sequences from PE files and split them into train/val/test sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--goodware-dir", default=str(DEFAULT_GOODWARE_DIR))
    parser.add_argument("--ransomware-dir", default=str(DEFAULT_RANSOMWARE_DIR))
    parser.add_argument("--out", default="data/opcodes_dataset",
                        help="directory for the opcode files and split CSVs")
    parser.add_argument("--ratios", default="0.7,0.15,0.15",
                        help="train,val,test ratios")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--include-operands", action="store_true",
                        help="write 'mnemonic operands' per line instead of the bare mnemonic")
    parser.add_argument("--max-instructions", type=int, default=0,
                        help="truncate each sample to this many instructions (0 = no limit)")
    parser.add_argument("--allow-packed", action="store_true",
                        help="disassemble UPX/ASPack-style binaries instead of skipping them")
    parser.add_argument("--family-disjoint", action="store_true",
                        help="put each ransomware family entirely in one split")
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="do not drop binaries that share a SHA-256 with another sample")
    parser.add_argument("--skip-existing", action="store_true",
                        help="reuse opcode files already present in --out")
    parser.add_argument("--link-splits", action="store_true",
                        help="also build a splits/<split>/<label>/<family>/ symlink tree")
    parser.add_argument("--limit", type=int, default=0,
                        help="process only this many randomly chosen files (for smoke tests)")

    args = parser.parse_args(argv)

    ratios = tuple(float(x) for x in args.ratios.split(","))
    if len(ratios) != 3 or any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        parser.error("--ratios needs three non-negative numbers summing to 1, e.g. 0.7,0.15,0.15")
    args.ratios = ratios
    return args


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out)
    opcode_root = out_dir / "opcodes"
    opcode_root.mkdir(parents=True, exist_ok=True)

    tasks = collect_tasks(args, opcode_root)
    if not tasks:
        print("No input files found. Check --goodware-dir / --ransomware-dir.", file=sys.stderr)
        return 1
    print(f"Disassembling {len(tasks)} files with {args.workers} workers...")

    records, skipped = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(process_file, tasks, chunksize=8), start=1):
            if result["status"] == "ok":
                records.append(result)
            else:
                skipped.append(result)
            if i % 100 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} processed, {len(skipped)} skipped", flush=True)

    if not records:
        print("Nothing was extracted successfully.", file=sys.stderr)
        return 1

    if args.keep_duplicates:
        dropped = []
    else:
        records, dropped = deduplicate(records)

    assign_splits(records, args.ratios, args.seed, args.family_disjoint)
    records.sort(key=lambda r: (r["split"], r["label"], r["family"], r["sha256"]))

    write_csv(out_dir / "index.csv", INDEX_FIELDS, records)
    for split in ("train", "val", "test"):
        write_csv(out_dir / f"{split}.csv", INDEX_FIELDS,
                  [r for r in records if r["split"] == split])
    if skipped:
        write_csv(out_dir / "skipped.csv",
                  ["source_path", "label_name", "family", "reason"], skipped)
    if dropped:
        write_csv(out_dir / "duplicates.csv",
                  ["sha256", "dropped_path", "kept_path", "dropped_family", "kept_family"], dropped)
    if args.link_splits:
        link_splits(records, out_dir)

    print_summary(records, dropped, skipped)
    print(f"\nWrote index.csv, train.csv, val.csv and test.csv to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
