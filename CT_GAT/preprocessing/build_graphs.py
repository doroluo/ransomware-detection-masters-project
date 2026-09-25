"""Build a control-flow graph from Capstone / Ghidra-style assembly.

Expected line format:
    0x0040101e:  jmp 0x40104d

The full .asm file is used. There is no instruction cap.

Tokenization stays separate. Attach Transformer token-map output with
``add_token_ids``; ``parse_capstone_insn`` from ``token_mapping`` is the
intended tokenizer.

Edge kinds:
  CFG  — fall-through and forward/backward control transfers
  LOOP — back-edges (jump to an earlier address / lower block id)
  CALL — direct near calls to another basic block in the same binary
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from token_mapping import (
    CRYPTO_BEHAVIOR_IDS,
    FILE_BEHAVIOR_IDS,
    BEHAVIOR_MAP,
)


# x86 instructions are at most 15 bytes. A larger address gap is a section
# hole, not fall-through to the next decoded instruction.
MAX_X86_INSN_LEN = 15

LINE_RE = re.compile(
    r"^\s*(?P<address>0x[0-9a-fA-F]+|[0-9]+):\s+"
    r"(?P<mnemonic>(?:(?:lock|rep|repe|repz|repne|repnz)\s+)*"
    r"[A-Za-z][A-Za-z0-9_.]*)\s*"
    r"(?P<operands>.*)$",
    re.IGNORECASE,
)
# Direct near/short/far target only. Memory refs like jmp dword ptr [0x401000]
# must not be treated as a jump to 0x401000.
DIRECT_TARGET_RE = re.compile(
    r"^(?:short\s+|near\s+|far\s+)?(0x[0-9a-fA-F]+|[0-9]+)$",
    re.IGNORECASE,
)

CONDITIONAL_JUMPS = {
    "ja", "jae", "jb", "jbe", "jc", "je", "jg", "jge", "jl", "jle",
    "jna", "jnae", "jnb", "jnbe", "jnc", "jne", "jng", "jnge", "jnl",
    "jnle", "jno", "jnp", "jns", "jnz", "jo", "jp", "jpe", "jpo",
    "js", "jz", "jcxz", "jecxz", "jrcxz",
    "loop", "loope", "loopne", "loopnz", "loopz",
}
UNCONDITIONAL_JUMPS = {"jmp"}
RETURNS = {"ret", "retn", "retf", "iret", "iretd", "iretq"}
PREFIXES = {"lock", "rep", "repe", "repz", "repne", "repnz"}

EDGE_CFG = 0
EDGE_LOOP = 1
EDGE_CALL = 2
EDGE_TYPE_NAMES = {EDGE_CFG: "cfg", EDGE_LOOP: "loop", EDGE_CALL: "call"}


def _could_fall_through(from_address: int, to_address: int) -> bool:
    return 1 <= (to_address - from_address) <= MAX_X86_INSN_LEN


def _direct_address(operand: str) -> int | None:
    if "[" in operand:
        return None
    match = DIRECT_TARGET_RE.fullmatch(operand.strip())
    return int(match.group(1), 0) if match else None


@dataclass
class Instruction:
    address: int
    mnemonic: str
    operands: str
    opcode_id: int | None = None
    operand1_id: int | None = None
    operand2_id: int | None = None
    behavior_id: int | None = None

    @property
    def opcode(self) -> str:
        """Mnemonic without lock/rep prefixes, used for CFG classification."""
        parts = self.mnemonic.lower().split()
        while parts and parts[0] in PREFIXES:
            parts.pop(0)
        return parts[-1] if parts else ""

    @property
    def target(self) -> int | None:
        if self.opcode not in CONDITIONAL_JUMPS | UNCONDITIONAL_JUMPS:
            return None
        return _direct_address(self.operands)

    @property
    def call_target(self) -> int | None:
        if self.opcode != "call":
            return None
        return _direct_address(self.operands)

    @property
    def is_conditional_jump(self) -> bool:
        return self.opcode in CONDITIONAL_JUMPS

    @property
    def is_unconditional_jump(self) -> bool:
        return self.opcode in UNCONDITIONAL_JUMPS

    @property
    def terminates_block(self) -> bool:
        return self.opcode in CONDITIONAL_JUMPS | UNCONDITIONAL_JUMPS | RETURNS

    @property
    def token_ids(self) -> tuple[int, int, int, int]:
        if (
            self.opcode_id is None
            or self.operand1_id is None
            or self.operand2_id is None
            or self.behavior_id is None
        ):
            raise ValueError("Call add_token_ids before reading token_ids")
        return (
            self.opcode_id,
            self.operand1_id,
            self.operand2_id,
            self.behavior_id,
        )


@dataclass
class BasicBlock:
    block_id: int
    instructions: list[Instruction] = field(default_factory=list)

    @property
    def start_address(self) -> int:
        return self.instructions[0].address

    @property
    def end_address(self) -> int:
        return self.instructions[-1].address

    @property
    def crypto_count(self) -> int:
        return sum(
            1
            for insn in self.instructions
            if insn.behavior_id in CRYPTO_BEHAVIOR_IDS
        )

    @property
    def file_count(self) -> int:
        return sum(
            1
            for insn in self.instructions
            if insn.behavior_id in FILE_BEHAVIOR_IDS
        )

    @property
    def has_file_read(self) -> bool:
        return any(
            insn.behavior_id == BEHAVIOR_MAP["FILE_READ"]
            for insn in self.instructions
        )

    @property
    def has_file_write(self) -> bool:
        return any(
            insn.behavior_id == BEHAVIOR_MAP["FILE_WRITE"]
            for insn in self.instructions
        )


@dataclass
class ControlFlowGraph:
    blocks: list[BasicBlock]
    # Directed edges represented as (source_block_id, destination_block_id).
    edges: list[tuple[int, int]]
    # Parallel to edges: EDGE_CFG / EDGE_LOOP / EDGE_CALL.
    edge_types: list[int] = field(default_factory=list)


def parse_assembly(lines: Iterable[str]) -> list[Instruction]:
    """Parse every matching instruction in the file. No length limit."""
    instructions = []
    for line in lines:
        match = LINE_RE.match(line)
        if not match:
            continue  # Ignore comments, labels, blank lines, and headers.
        instructions.append(
            Instruction(
                address=int(match.group("address"), 0),
                mnemonic=match.group("mnemonic"),
                operands=match.group("operands").strip(),
            )
        )
    instructions.sort(key=lambda instruction: instruction.address)
    if not instructions:
        raise ValueError("No assembly instructions matched the expected address format")
    return instructions


def add_token_ids(
    instructions: list[Instruction],
    tokenizer: Callable[[str, str], Sequence[int]],
) -> None:
    """Attach existing Transformer IDs to parsed instructions in place.

    ``tokenizer(mnemonic, operands)`` should return
    ``(opcode_or_api_id, operand1_id, operand2_id, behavior_id)``, matching
    ``token_mapping.parse_capstone_insn``.
    """
    for instruction in instructions:
        tokens = tokenizer(instruction.mnemonic, instruction.operands)
        if len(tokens) != 4:
            raise ValueError(
                f"tokenizer must return 4 ids, got {len(tokens)}: {tokens!r}"
            )
        opcode_id, operand1_id, operand2_id, behavior_id = tokens
        instruction.opcode_id = int(opcode_id)
        instruction.operand1_id = int(operand1_id)
        instruction.operand2_id = int(operand2_id)
        instruction.behavior_id = int(behavior_id)


def build_cfg(instructions: list[Instruction]) -> ControlFlowGraph:
    """Split instructions into basic blocks and add CFG / loop / call edges."""
    if not instructions:
        raise ValueError("Cannot build a CFG from zero instructions")

    address_to_index = {instruction.address: i for i, instruction in enumerate(instructions)}
    leaders = {instructions[0].address}

    for index, instruction in enumerate(instructions):
        if instruction.target in address_to_index:
            leaders.add(instruction.target)
        if instruction.call_target in address_to_index:
            leaders.add(instruction.call_target)
        next_exists = index + 1 < len(instructions)
        if next_exists and instruction.terminates_block:
            leaders.add(instructions[index + 1].address)
        if next_exists and not _could_fall_through(
            instruction.address, instructions[index + 1].address
        ):
            leaders.add(instructions[index + 1].address)

    blocks: list[BasicBlock] = []
    current: list[Instruction] = []
    for instruction in instructions:
        if current and instruction.address in leaders:
            blocks.append(BasicBlock(len(blocks), current))
            current = []
        current.append(instruction)
        if instruction.terminates_block:
            blocks.append(BasicBlock(len(blocks), current))
            current = []
    if current:
        blocks.append(BasicBlock(len(blocks), current))

    address_to_block = {
        instruction.address: block.block_id
        for block in blocks
        for instruction in block.instructions
    }

    edges: list[tuple[int, int]] = []
    edge_types: list[int] = []
    seen: set[tuple[int, int, int]] = set()

    def add_edge(src: int, dst: int, edge_type: int) -> None:
        key = (src, dst, edge_type)
        if key in seen:
            return
        seen.add(key)
        edges.append((src, dst))
        edge_types.append(edge_type)

    def add_fall_through(block_index: int, block: BasicBlock) -> None:
        if block_index + 1 >= len(blocks):
            return
        nxt = blocks[block_index + 1]
        if _could_fall_through(block.end_address, nxt.start_address):
            add_edge(block.block_id, nxt.block_id, EDGE_CFG)

    for block_index, block in enumerate(blocks):
        last = block.instructions[-1]
        if last.is_unconditional_jump:
            if last.target in address_to_block:
                dst = address_to_block[last.target]
                kind = EDGE_LOOP if dst <= block.block_id else EDGE_CFG
                add_edge(block.block_id, dst, kind)
        elif last.is_conditional_jump:
            if last.target in address_to_block:
                dst = address_to_block[last.target]
                kind = EDGE_LOOP if dst <= block.block_id else EDGE_CFG
                add_edge(block.block_id, dst, kind)
            add_fall_through(block_index, block)
        elif last.opcode not in RETURNS:
            add_fall_through(block_index, block)

        # Call edges: any direct near call inside the block.
        for instruction in block.instructions:
            target = instruction.call_target
            if target is None or target not in address_to_block:
                continue
            add_edge(block.block_id, address_to_block[target], EDGE_CALL)

    return ControlFlowGraph(
        blocks=blocks,
        edges=edges,
        edge_types=edge_types,
    )


def find_read_to_write_paths(
    cfg: ControlFlowGraph,
    max_paths: int = 5,
    max_depth: int = 32,
) -> list[list[int]]:
    """BFS paths from a FILE_READ block to a FILE_WRITE block."""
    read_blocks = [b.block_id for b in cfg.blocks if b.has_file_read]
    write_blocks = {b.block_id for b in cfg.blocks if b.has_file_write}
    if not read_blocks or not write_blocks:
        return []

    adjacency: dict[int, list[int]] = {b.block_id: [] for b in cfg.blocks}
    for src, dst in cfg.edges:
        adjacency[src].append(dst)

    paths: list[list[int]] = []
    for start in read_blocks:
        if len(paths) >= max_paths:
            break
        queue: list[list[int]] = [[start]]
        visited = {start}
        while queue and len(paths) < max_paths:
            path = queue.pop(0)
            node = path[-1]
            if node in write_blocks and len(path) > 1:
                paths.append(path)
                continue
            if len(path) >= max_depth:
                continue
            for nxt in adjacency[node]:
                if nxt in visited:
                    continue
                visited.add(nxt)
                queue.append(path + [nxt])
    return paths


def load_cfg(path: str | Path) -> ControlFlowGraph:
    """Load an entire .asm file and build its CFG. No instruction cap."""
    path = Path(path)
    return build_cfg(
        parse_assembly(path.read_text(encoding="utf-8", errors="replace").splitlines())
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a CFG from a Ghidra-style .asm file")
    parser.add_argument("asm_file")
    args = parser.parse_args()
    cfg = load_cfg(args.asm_file)
    for block in cfg.blocks:
        print(
            f"B{block.block_id}: 0x{block.start_address:x} -> 0x{block.end_address:x} "
            f"crypto={block.crypto_count} file={block.file_count}"
        )
        print("  " + " | ".join(f"{i.mnemonic} {i.operands}" for i in block.instructions))
    typed = [
        f"{s}->{d}:{EDGE_TYPE_NAMES.get(t, t)}"
        for (s, d), t in zip(cfg.edges, cfg.edge_types)
    ]
    print("Edges:", typed)
