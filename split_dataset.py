#!/usr/bin/env python3
"""
split_dataset.py

Assigns every sample in an opcode index.csv to a train / validation / test
split and writes one CSV per split.

The index is the file produced by extract_opcodes.py:

    sha256,split,label,label_name,family,arch,n_instructions,n_unique_opcodes,
    source_path,opcode_file

Two splitting modes:

  random (default)  Stratified by (label, family), so every family is present
                    in all three splits. This is the setting to use when the
                    question is "can the model recognise ransomware".

  family-disjoint   A whole ransomware family goes to exactly one split, so the
                    test set only contains families the model never trained on.
                    Goodware is still stratified, otherwise a split could end up
                    with a single class. Use this to measure generalisation to
                    new families; expect noticeably lower scores.

Either way samples are de-duplicated by SHA-256 first, so the identical binary
can never appear in two splits and inflate the score.

Usage:

    python split_dataset.py --index data/opcodes_dataset/index.csv
    python split_dataset.py --index data/opcodes_dataset/index.csv \
        --ratios 0.8,0.1,0.1 --seed 7 --out data/opcodes_dataset/split_80_10_10
    python split_dataset.py --index data/opcodes_dataset/index.csv --family-disjoint
"""

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")

REQUIRED_COLUMNS = {"sha256", "label", "label_name", "family"}


def read_index(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"{path} is missing required column(s): {', '.join(sorted(missing))}"
            )
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    if not rows:
        raise SystemExit(f"{path} has no data rows")
    if "split" not in fieldnames:
        fieldnames.insert(1, "split")
    for row in rows:
        row["label"] = int(row["label"])
    return rows, fieldnames


def deduplicate(rows):
    """Keeps the first row per SHA-256; returns (kept, dropped)."""
    seen = {}
    kept, dropped = [], []
    for row in sorted(rows, key=lambda r: r.get("source_path", r["sha256"])):
        first = seen.get(row["sha256"])
        if first is None:
            seen[row["sha256"]] = row
            kept.append(row)
        else:
            dropped.append({
                "sha256": row["sha256"],
                "dropped_path": row.get("source_path", ""),
                "kept_path": first.get("source_path", ""),
                "dropped_family": row["family"],
                "kept_family": first["family"],
            })
    return kept, dropped


def split_sizes(n, ratios):
    """
    Largest-remainder allocation, with one correction: as long as there are at
    least three units, val and test get at least one each. Plain rounding gives
    a 13-sample family 13/0/0 and it silently disappears from evaluation.
    """
    if n == 0:
        return [0, 0, 0]
    if n < 3:
        return [n, 0, 0]

    raw = [n * r for r in ratios]
    sizes = [int(x) for x in raw]
    order = sorted(range(3), key=lambda i: raw[i] - sizes[i], reverse=True)
    for i in range(n - sum(sizes)):
        sizes[order[i % 3]] += 1

    for i in (1, 2):
        if sizes[i] == 0 and ratios[i] > 0:
            donor = max(range(3), key=lambda j: sizes[j])
            if sizes[donor] > 1:
                sizes[donor] -= 1
                sizes[i] += 1
    return sizes


def assign_splits(rows, ratios, seed, family_disjoint):
    """
    Groups rows into strata, then into units inside each stratum, and hands
    whole units to a split. A unit is one sample normally, or one entire family
    when --family-disjoint is set.
    """
    rng = random.Random(seed)
    strata = defaultdict(lambda: defaultdict(list))

    for row in rows:
        if family_disjoint and row["label"] == 1:
            strata["ransomware"][row["family"]].append(row)
        else:
            strata[(row["label_name"], row["family"])][row["sha256"]].append(row)

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
            for row in unit:
                row["split"] = split


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def link_splits(rows, out_dir):
    """Mirrors the assignment as splits/<split>/<label>/<family>/<sha>.txt."""
    for row in rows:
        source = row.get("opcode_file")
        if not source:
            continue
        target = Path(source).resolve()
        link = (out_dir / "splits" / row["split"] / row["label_name"]
                / row["family"] / target.name)
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)


def check_leakage(rows):
    """A SHA-256 in two splits means the test score cannot be trusted."""
    splits_by_hash = defaultdict(set)
    for row in rows:
        splits_by_hash[row["sha256"]].add(row["split"])
    return {h: s for h, s in splits_by_hash.items() if len(s) > 1}


def print_summary(rows, dropped, family_disjoint):
    by_split = Counter(r["split"] for r in rows)
    by_split_label = Counter((r["split"], r["label_name"]) for r in rows)
    total = len(rows)

    print("\n=== Split summary ===")
    print(f"samples        : {total}")
    print(f"duplicates     : {len(dropped)}")
    print(f"mode           : {'family-disjoint' if family_disjoint else 'random (stratified)'}")
    print("\nsplit      total    share   goodware  ransomware  families")
    for split in SPLITS:
        n = by_split[split]
        families = {r["family"] for r in rows if r["split"] == split and r["label"] == 1}
        share = n / total if total else 0
        print(f"{split:<10} {n:<8} {share:>5.1%}   {by_split_label[(split, 'goodware')]:<9} "
              f"{by_split_label[(split, 'ransomware')]:<11} {len(families)}")

    missing = [
        split for split in SPLITS
        if by_split[split] and len({r["label"] for r in rows if r["split"] == split}) < 2
    ]
    if missing:
        print(f"\n[!] only one class present in: {', '.join(missing)}")

    if not family_disjoint:
        absent = sorted(
            family
            for family in {r["family"] for r in rows if r["label"] == 1}
            if not all(
                any(r["family"] == family and r["split"] == split for r in rows)
                for split in SPLITS
            )
        )
        if absent:
            print(f"\n[!] families too small to reach every split: {', '.join(absent)}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Split an opcode dataset index into train / validation / test sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--index", default="data/opcodes_dataset/index.csv",
                        help="index.csv written by extract_opcodes.py")
    parser.add_argument("--out", default=None,
                        help="output directory (default: the directory holding --index)")
    parser.add_argument("--ratios", default="0.7,0.15,0.15",
                        help="train,val,test ratios")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--family-disjoint", action="store_true",
                        help="keep each ransomware family entirely within one split")
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="do not drop rows sharing a SHA-256 with another row")
    parser.add_argument("--link-splits", action="store_true",
                        help="also build a splits/<split>/<label>/<family>/ symlink tree")

    args = parser.parse_args(argv)

    ratios = tuple(float(x) for x in args.ratios.split(","))
    if len(ratios) != 3 or any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        parser.error("--ratios needs three non-negative numbers summing to 1, e.g. 0.7,0.15,0.15")
    args.ratios = ratios
    return args


def main(argv=None):
    args = parse_args(argv)
    index_path = Path(args.index)
    if not index_path.is_file():
        print(f"{index_path} not found. Run extract_opcodes.py first.", file=sys.stderr)
        return 1

    out_dir = Path(args.out) if args.out else index_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, fieldnames = read_index(index_path)
    print(f"Read {len(rows)} rows from {index_path}")

    if args.keep_duplicates:
        dropped = []
    else:
        rows, dropped = deduplicate(rows)
        if dropped:
            print(f"Dropped {len(dropped)} duplicate binaries")

    assign_splits(rows, args.ratios, args.seed, args.family_disjoint)
    rows.sort(key=lambda r: (r["split"], r["label"], r["family"], r["sha256"]))

    leaked = check_leakage(rows)
    if leaked:
        print(f"[!] {len(leaked)} SHA-256 values appear in more than one split", file=sys.stderr)
        return 1

    write_csv(out_dir / "index.csv", fieldnames, rows)
    for split in SPLITS:
        write_csv(out_dir / f"{split}.csv", fieldnames,
                  [r for r in rows if r["split"] == split])
    if dropped:
        write_csv(out_dir / "duplicates.csv",
                  ["sha256", "dropped_path", "kept_path", "dropped_family", "kept_family"],
                  dropped)
    if args.link_splits:
        link_splits(rows, out_dir)

    print_summary(rows, dropped, args.family_disjoint)
    print(f"\nWrote index.csv, train.csv, val.csv and test.csv to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
