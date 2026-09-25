"""Batch-build GAT graphs from .asm files: CFG + Transformer token IDs.

Saves one .pt per sample:
  {
    file_id,
    label,
    asm_path,
    token_ids:          LongTensor[N, 4],  # opcode/API, op1, op2, behavior
    insn_to_block:      LongTensor[N],
    edge_index:         LongTensor[2, E],  # CFG + loop + call edges
    edge_type:          LongTensor[E],     # 0=cfg, 1=loop, 2=call
    block_crypto_count: LongTensor[B],
    block_file_count:   LongTensor[B],
    num_blocks:         int,
  }

Reads data/metadata.csv when it has rows. Otherwise uses the split CSVs
and data/asm/<file_id>.asm.

Usage:
    python build_pyg_graphs.py
    python build_pyg_graphs.py --limit 10
    python build_pyg_graphs.py path/to/file.asm --label 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from build_graphs import add_token_ids, build_cfg, parse_assembly
from token_mapping import parse_capstone_insn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
ASM_DIR = PROJECT_ROOT / "data" / "asm"
SPLIT_DIR = PROJECT_ROOT / "processed" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "processed" / "graphs"


def build_tokenized_cfg(asm_path: Path):
    instructions = parse_assembly(
        asm_path.read_text(encoding="utf-8", errors="replace").splitlines()
    )
    add_token_ids(instructions, parse_capstone_insn)
    return build_cfg(instructions)


def cfg_to_tensors(cfg):
    token_rows = []
    insn_to_block = []
    crypto_counts = []
    file_counts = []
    for block in cfg.blocks:
        crypto_counts.append(block.crypto_count)
        file_counts.append(block.file_count)
        for instruction in block.instructions:
            token_rows.append(list(instruction.token_ids))
            insn_to_block.append(block.block_id)

    token_ids = torch.tensor(token_rows, dtype=torch.long)
    insn_to_block_tensor = torch.tensor(insn_to_block, dtype=torch.long)
    if cfg.edges:
        edge_index = torch.tensor(cfg.edges, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(cfg.edge_types, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
    block_crypto_count = torch.tensor(crypto_counts, dtype=torch.long)
    block_file_count = torch.tensor(file_counts, dtype=torch.long)
    return (
        token_ids,
        insn_to_block_tensor,
        edge_index,
        edge_type,
        block_crypto_count,
        block_file_count,
        len(cfg.blocks),
    )


def save_graph(file_id: str, label: int, asm_path: Path) -> tuple[int, int]:
    cfg = build_tokenized_cfg(asm_path)
    (
        token_ids,
        insn_to_block,
        edge_index,
        edge_type,
        block_crypto_count,
        block_file_count,
        num_blocks,
    ) = cfg_to_tensors(cfg)
    torch.save(
        {
            "file_id": file_id,
            "label": int(label),
            "asm_path": str(asm_path),
            "token_ids": token_ids,
            "insn_to_block": insn_to_block,
            "edge_index": edge_index,
            "edge_type": edge_type,
            "block_crypto_count": block_crypto_count,
            "block_file_count": block_file_count,
            "num_blocks": num_blocks,
        },
        OUTPUT_DIR / f"{file_id}.pt",
    )
    return token_ids.shape[0], num_blocks


def iter_metadata_rows(limit: int | None):
    df = pd.read_csv(METADATA_PATH)
    if len(df) == 0:
        return None
    if limit is not None:
        df = df.head(limit)
    rows = []
    for _, row in df.iterrows():
        asm_path = Path(str(row["asm_path"]))
        if not asm_path.is_absolute():
            asm_path = PROJECT_ROOT / asm_path
        rows.append((str(row["file_id"]), int(row["label"]), asm_path))
    return rows


def iter_split_rows(limit: int | None):
    frames = []
    for name in ("train.csv", "validation.csv", "test.csv"):
        path = SPLIT_DIR / name
        if path.is_file():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(
            f"No samples in {METADATA_PATH} and no split CSVs in {SPLIT_DIR}"
        )
    df = pd.concat(frames, ignore_index=True).drop_duplicates("file_id")
    if limit is not None:
        df = df.head(limit)
    rows = []
    for _, row in df.iterrows():
        file_id = str(row["file_id"])
        rows.append((file_id, int(row["label"]), ASM_DIR / f"{file_id}.asm"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GAT graphs from .asm + token maps")
    parser.add_argument("asm_file", nargs="?", help="Optional single .asm path")
    parser.add_argument("--label", type=int, default=None, help="Label for a single file")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to process")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild graphs that already exist",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.asm_file:
        asm_path = Path(args.asm_file)
        if not asm_path.is_absolute():
            asm_path = (Path.cwd() / asm_path).resolve()
        file_id = asm_path.stem
        label = 0 if args.label is None else args.label
        samples = [(file_id, label, asm_path)]
        print(f"Single file mode: {asm_path} label={label}")
    else:
        samples = iter_metadata_rows(args.limit)
        if samples is None:
            print(f"{METADATA_PATH} is empty; using split CSVs + {ASM_DIR}")
            samples = iter_split_rows(args.limit)
        else:
            print(f"Using {METADATA_PATH}: {len(samples)} sample(s)")

    successful = 0
    skipped_empty = 0
    skipped_existing = 0
    failed = 0

    print("Samples:", len(samples))

    for file_id, label, asm_path in samples:
        try:
            output_path = OUTPUT_DIR / f"{file_id}.pt"
            if output_path.is_file() and not args.overwrite:
                skipped_existing += 1
                continue

            if not asm_path.is_file():
                print(f"Missing asm: {asm_path}")
                failed += 1
                continue

            num_insns, num_blocks = save_graph(file_id, label, asm_path)
            print(
                f"Saved {file_id}: {num_insns} instructions, "
                f"{num_blocks} blocks, label={label}"
            )
            successful += 1
        except ValueError as error:
            if "No assembly instructions" in str(error):
                print(f"Skipped empty file: {asm_path.name}")
                skipped_empty += 1
            else:
                print(f"Failed: {asm_path}")
                print("Error:", error)
                failed += 1
        except Exception as error:
            print(f"Failed: {asm_path}")
            print("Error:", error)
            failed += 1

    print("\nFinished.")
    print("Successful:", successful)
    print("Skipped empty:", skipped_empty)
    print("Skipped existing:", skipped_existing)
    print("Failed:", failed)
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
