"""Sanity-check a CFG built from a Capstone-style .asm file.

Prints counts, a few sample blocks, and invariant errors.

Usage:
    python validate_graph.py
    python validate_graph.py path/to/file.asm
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from build_graphs import MAX_X86_INSN_LEN, RETURNS, load_cfg, parse_assembly

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASM_DIR = PROJECT_ROOT / "data" / "asm"
SAMPLE_BLOCKS = 5


def default_asm_path() -> Path:
    asm_files = sorted(ASM_DIR.glob("*.asm"))
    if not asm_files:
        raise FileNotFoundError(f"No .asm files found in {ASM_DIR}")
    return asm_files[0]


def could_fall_through(from_address: int, to_address: int) -> bool:
    return 1 <= (to_address - from_address) <= MAX_X86_INSN_LEN


def block_type(block) -> str:
    last = block.instructions[-1]
    if last.is_unconditional_jump:
        return "unconditional jump" if last.target is not None else "indirect jump"
    if last.is_conditional_jump:
        return "conditional jump" if last.target is not None else "indirect conditional jump"
    if last.opcode in RETURNS:
        return "function return"
    return "fall-through"


def block_id_for_address(address_to_block: dict[int, int], address: int | None) -> int | None:
    if address is None:
        return None
    return address_to_block.get(address)


def collect_errors(cfg, instructions) -> list[str]:
    errors: list[str] = []
    block_ids = [block.block_id for block in cfg.blocks]
    id_set = set(block_ids)

    if len(block_ids) != len(id_set):
        errors.append("Duplicate block IDs")
    if block_ids != list(range(len(cfg.blocks))):
        errors.append("Block IDs are not 0 .. N-1 in order")

    address_to_block: dict[int, int] = {}
    for block in cfg.blocks:
        if not block.instructions:
            errors.append(f"B{block.block_id} is empty")
            continue
        for previous, current in zip(block.instructions, block.instructions[1:]):
            if current.address - previous.address > MAX_X86_INSN_LEN:
                errors.append(
                    f"B{block.block_id} glues section gap "
                    f"0x{previous.address:x} -> 0x{current.address:x}"
                )
            if current.address <= previous.address:
                errors.append(
                    f"B{block.block_id} is not strictly increasing at 0x{current.address:x}"
                )
        for instruction in block.instructions:
            if instruction.address in address_to_block:
                errors.append(
                    f"Address 0x{instruction.address:x} is in B{address_to_block[instruction.address]} "
                    f"and B{block.block_id}"
                )
            address_to_block[instruction.address] = block.block_id

    parsed_addresses = [instruction.address for instruction in instructions]
    cfg_addresses = [
        instruction.address
        for block in cfg.blocks
        for instruction in block.instructions
    ]
    if cfg_addresses != parsed_addresses:
        errors.append(
            f"Instruction coverage mismatch: parsed={len(instructions)} "
            f"in_blocks={len(cfg_addresses)}"
        )

    outgoing: dict[int, list[int]] = defaultdict(list)
    for source, destination in cfg.edges:
        if source not in id_set or destination not in id_set:
            errors.append(f"Invalid edge: {source} -> {destination}")
            continue
        outgoing[source].append(destination)

    for block_index, block in enumerate(cfg.blocks):
        last = block.instructions[-1]
        dests = set(outgoing[block.block_id])
        next_block = (
            cfg.blocks[block_index + 1] if block_index + 1 < len(cfg.blocks) else None
        )
        fall_through_id = (
            next_block.block_id
            if next_block is not None
            and could_fall_through(block.end_address, next_block.start_address)
            else None
        )
        target_id = block_id_for_address(address_to_block, last.target)

        if last.is_unconditional_jump:
            expected = {target_id} if target_id is not None else set()
            if dests != expected:
                errors.append(
                    f"B{block.block_id} jmp edges {sorted(dests)} != expected {sorted(expected)}"
                )

        elif last.is_conditional_jump:
            expected = set()
            if target_id is not None:
                expected.add(target_id)
            if fall_through_id is not None:
                expected.add(fall_through_id)
            if dests != expected:
                errors.append(
                    f"B{block.block_id} jcc edges {sorted(dests)} != expected {sorted(expected)}"
                )

        elif last.opcode in RETURNS:
            if dests:
                errors.append(f"B{block.block_id} ret has outgoing edges {sorted(dests)}")

        else:
            expected = {fall_through_id} if fall_through_id is not None else set()
            if dests != expected:
                errors.append(
                    f"B{block.block_id} fall-through edges {sorted(dests)} != expected {sorted(expected)}"
                )

    return errors


def print_block(block, outgoing: dict[int, list[int]]) -> None:
    last = block.instructions[-1]
    print(
        f"  B{block.block_id}: 0x{block.start_address:x} -> 0x{block.end_address:x} "
        f"n={len(block.instructions)}  last={last.mnemonic} {last.operands}  "
        f"[{block_type(block)}]  out={outgoing[block.block_id]}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a CFG built from an .asm file")
    parser.add_argument(
        "asm_file",
        nargs="?",
        help="Path to a .asm file. Defaults to the first file in data/asm/",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every block, not just a sample",
    )
    args = parser.parse_args()

    asm_path = Path(args.asm_file) if args.asm_file else default_asm_path()
    if not asm_path.is_file():
        print(f"Missing asm file: {asm_path}", file=sys.stderr)
        return 1

    instructions = parse_assembly(
        asm_path.read_text(encoding="utf-8", errors="replace").splitlines()
    )
    cfg = load_cfg(asm_path)

    outgoing: dict[int, list[int]] = defaultdict(list)
    for source, destination in cfg.edges:
        outgoing[source].append(destination)

    no_outgoing = sum(1 for block in cfg.blocks if not outgoing[block.block_id])
    incoming = {destination for _, destination in cfg.edges}
    no_incoming = sum(
        1 for block in cfg.blocks if block.block_id != 0 and block.block_id not in incoming
    )

    print("File:", asm_path)
    print("Instructions:", len(instructions))
    print("Blocks:", len(cfg.blocks))
    print("Edges:", len(cfg.edges))
    print("Blocks with no outgoing edges:", no_outgoing)
    print("Blocks with no incoming edges (excluding B0):", no_incoming)

    if args.verbose:
        print("\nAll blocks:")
        for block in cfg.blocks:
            print_block(block, outgoing)
    else:
        print("\nFirst blocks:")
        for block in cfg.blocks[:SAMPLE_BLOCKS]:
            print_block(block, outgoing)
        if len(cfg.blocks) > SAMPLE_BLOCKS:
            print("\nLast blocks:")
            for block in cfg.blocks[-SAMPLE_BLOCKS:]:
                print_block(block, outgoing)

    errors = collect_errors(cfg, instructions)
    if errors:
        print(f"\nFAILED: {len(errors)} problem(s)")
        for error in errors[:50]:
            print(" -", error)
        if len(errors) > 50:
            print(f" - ... {len(errors) - 50} more")
        return 1

    print("\nCFG checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
