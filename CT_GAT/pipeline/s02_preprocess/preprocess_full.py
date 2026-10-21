"""Sample one CFG subgraph and align it with the IDA listing.

The Mendeley export is schema ``ida-cfg-research/2.1``. Each ``.jsonl.gz``
holds a header, one record per function, string records, and a footer.
Function ``edges`` are directed pairs ``[src_block_id, dst_block_id]``. The
export normally has no edge type, so this file derives a block-CFG edge type
for each edge. Block ids restart in every function, so ``function_id`` is kept
only as a namespace: a node key is ``(function_id, block_id)``. No
function-level graph is built. Instruction and block addresses are RVAs.
``end_rva`` is exclusive: ``[start_rva, end_rva)``.

The paired ``.asm.gz`` listing stores virtual addresses on labels
(``sub_`` / ``loc_`` / ``org``) and on named procedures only when the CFG
function RVA is known. It does not store a virtual address on every
instruction. Alignment therefore buckets instructions by those label RVAs
and pairs CFG instructions with assembly lines inside each bucket.

Sampling uses block instruction counts and CFG edges only. ``--sample walk``
returns one connected subgraph grown from a random block. ``--sample signal``
returns the blocks of the top-ranked whole functions, which need not be
connected (CFG edges never leave a function, so it is usually several
components). The same selected blocks feed the sequence model, the
handcrafted-feature graph, and the tokenizer-embedding graph.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import logging
import random
import re
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn

_STAGE_DIR = Path(__file__).resolve().parent
if str(_STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_STAGE_DIR))

import tokenize_asm_gz as tok

# Opcode ids occupy 0..255. Operand category ids stop where API ids begin.
OPCODE_VOCAB_SIZE = 256
OPERAND_VOCAB_SIZE = tok.FIRST_API_ID


logger = logging.getLogger(__name__)

SCHEMA = "ida-cfg-research/2.1"
TOKEN_WIDTH = 5
BLOCK_EMBED_DIM = 64
GRAPH_FORMAT_VERSION = 3

# These are the relations consumed by the one block-level GNN. The final
# relation is a safe bucket for an explicit/exported edge type that cannot be
# normalized to one of the six CFG relations.
EDGE_TYPE_NAMES: tuple[str, ...] = (
    "fallthrough",
    "cond_taken",
    "jmp",
    "indirect",
    "extern",
    "other",
    "unknown",
)
EDGE_TYPE_TO_ID = {name: index for index, name in enumerate(EDGE_TYPE_NAMES)}
UNKNOWN_EDGE_TYPE_ID = EDGE_TYPE_TO_ID["unknown"]
# Keep a few examples for the manifest. Full gap lists are large enough to
# exhaust a small machine while json is being written.
MISMATCH_EXAMPLE_LIMIT = 32
FCB_INDJUMP = 1
FCB_EXTERN = 6

_PROC = re.compile(r"^(\S+)\s+proc\b", re.IGNORECASE)
_COLLAPSED = re.compile(r"COLLAPSED FUNCTION\s+(.+?)\]", re.IGNORECASE)
_STDCALL = re.compile(r"@\d+$")
_MNEMONIC = re.compile(r"[a-z][a-z0-9.]{0,15}")
_NOT_CODE = {
    "text", "guid", "unicode", "offset", "end", "include", "extrn", "public", "assume",
}

_OPERAND_NAMES = (
    "PADDING",
    "REG_DATA",
    "REG_STACK",
    "MEM_STACK_REF",
    "MEM_GLOBAL_REF",
    "IMM_VALUE",
)
_OPCODE_NAMES = tuple(
    name
    for name, _token_id in sorted(tok.TOKEN_MAP.items(), key=lambda item: (item[1], item[0]))
    if name not in _OPERAND_NAMES
)
_BEHAVIOR_NAMES = tuple(
    name for name, _behavior_id in sorted(tok.BEHAVIOR_MAP.items(), key=lambda item: item[1])
)

HANDCRAFTED_FEATURE_NAMES: tuple[str, ...] = (
    "instruction_count",
    "byte_size",
    "call_count",
    "import_reference_count",
    "external",
    "cfg_in_degree",
    "cfg_out_degree",
    *(f"behavior_{name}" for name in _BEHAVIOR_NAMES),
    "api_none_count",
    "api_unresolved_count",
    "api_known_count",
    *(f"opcode_{name}" for name in _OPCODE_NAMES),
    *(f"operand_{name}" for name in _OPERAND_NAMES),
)


class CfgSchemaError(ValueError):
    """The CFG export is missing a field the pipeline needs."""


@dataclass
class CfgInstruction:
    file_id: str
    function_id: int
    function_name: str
    block_id: int
    index_in_block: int
    rva: int
    size: int
    mnemonic: str
    import_ids: list[int]
    call_resolution: str | None
    is_call: bool = False
    raw_text: str | None = None
    va: int | None = None
    matched: bool = False
    token_ids: list[int] | None = None
    api_name: str | None = None
    api_id: int | None = None
    api_kind: str | None = None
    api_source: str | None = None
    behavior_id: int | None = None
    behavior_tag: str | None = None
    behavior_tags: list[str] = field(default_factory=list)
    operands: list[str] = field(default_factory=list)
    opcode: str | None = None

    @property
    def block_key(self) -> tuple[int, int]:
        return (self.function_id, self.block_id)


@dataclass
class CfgBlock:
    file_id: str
    function_id: int
    function_name: str
    block_id: int
    start_rva: int
    end_rva: int
    block_type: int | None
    external: bool
    instructions: list[CfgInstruction]

    @property
    def key(self) -> tuple[int, int]:
        return (self.function_id, self.block_id)

    @property
    def instruction_count(self) -> int:
        return len(self.instructions)

    @property
    def byte_size(self) -> int:
        return self.end_rva - self.start_rva


@dataclass
class CfgEdge:
    src: tuple[int, int]
    dst: tuple[int, int]
    edge_type: str | None = None
    is_back_edge: bool = False
    is_back_address: bool = False


@dataclass
class FullCFG:
    file_id: str
    imagebase: int
    blocks: dict[tuple[int, int], CfgBlock]
    edges: list[CfgEdge]
    imports: list[dict[str, Any]]
    function_rvas: dict[str, int]
    header: dict[str, Any]
    schema_notes: list[str] = field(default_factory=list)
    address_format_problems: list[dict[str, Any]] = field(default_factory=list)
    library_functions: set[int] = field(default_factory=set)

    @property
    def instruction_count(self) -> int:
        return sum(block.instruction_count for block in self.blocks.values())

@dataclass
class AsmInstruction:
    raw_text: str
    mnemonic: str
    anchor_rva: int | None
    record: dict[str, Any]


@dataclass
class MismatchReport:
    missing_json_instructions: list[dict[str, Any]] = field(default_factory=list)
    missing_asm_instructions: list[dict[str, Any]] = field(default_factory=list)
    duplicate_addresses: list[dict[str, Any]] = field(default_factory=list)
    address_format_problems: list[dict[str, Any]] = field(default_factory=list)
    missing_json_count: int = 0
    missing_asm_count: int = 0
    duplicate_count: int = 0
    address_format_count: int = 0

    def _keep(self, bucket: list[dict[str, Any]], item: dict[str, Any]) -> None:
        if len(bucket) < MISMATCH_EXAMPLE_LIMIT:
            bucket.append(item)

    def add_missing_json(self, item: dict[str, Any]) -> None:
        self.missing_json_count += 1
        self._keep(self.missing_json_instructions, item)

    def add_missing_asm(self, item: dict[str, Any]) -> None:
        self.missing_asm_count += 1
        self._keep(self.missing_asm_instructions, item)

    def add_duplicate(self, item: dict[str, Any]) -> None:
        self.duplicate_count += 1
        self._keep(self.duplicate_addresses, item)

    def add_format_problems(self, items: Sequence[dict[str, Any]]) -> None:
        for item in items:
            self.address_format_count += 1
            self._keep(self.address_format_problems, item)

    def to_dict(self) -> dict[str, Any]:
        return {
            "missing_json_count": self.missing_json_count,
            "missing_asm_count": self.missing_asm_count,
            "duplicate_count": self.duplicate_count,
            "address_format_count": self.address_format_count,
            "missing_json_instructions": self.missing_json_instructions,
            "missing_asm_instructions": self.missing_asm_instructions,
            "duplicate_addresses": self.duplicate_addresses,
            "address_format_problems": self.address_format_problems,
        }


@dataclass
class SampledSubgraph:
    seed: int
    selected_keys: list[tuple[int, int]]
    edges: list[CfgEdge]
    original_blocks: int
    original_edges: int
    original_instructions: int

    @property
    def sampled_blocks(self) -> int:
        return len(self.selected_keys)

    def sampled_instructions(self, cfg: FullCFG) -> int:
        return sum(cfg.blocks[key].instruction_count for key in self.selected_keys)


def parse_address(value: Any, field_name: str) -> int:
    """Read an RVA or image base from an int or a ``0x`` string."""
    if isinstance(value, bool) or value is None:
        raise CfgSchemaError(f"missing {field_name}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text, 0)
        except ValueError as error:
            raise CfgSchemaError(f"cannot parse {field_name}: {value!r}") from error
    raise CfgSchemaError(f"{field_name} has unsupported type {type(value).__name__}")


def _require(record: dict[str, Any], fields: Sequence[str], where: str) -> None:
    missing = [name for name in fields if name not in record or record[name] is None]
    if missing:
        raise CfgSchemaError(f"{where} is missing {', '.join(missing)}")


def _api_stem(name: str) -> str:
    """Fold stdcall decoration so ``GetLastError`` and ``GetLastError@0`` meet."""
    return _STDCALL.sub("", name.lower())


def iter_text_lines(path: Path) -> Iterable[str]:
    """Yield one listing line at a time so a large .asm.gz is not held as one string."""
    path = Path(path)
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            yield line.rstrip("\r\n")


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    path = Path(path)
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as error:
                raise CfgSchemaError(f"{path}:{line_number} is not JSON") from error
            if not isinstance(record, dict) or "record" not in record:
                raise CfgSchemaError(f"{path}:{line_number} has no record type")
            yield record


def _imagebase_of(header: dict[str, Any]) -> int:
    if header.get("imagebase") is not None:
        return parse_address(header["imagebase"], "header.imagebase")
    pe = header.get("pe") if isinstance(header.get("pe"), dict) else {}
    if pe.get("imagebase") is not None:
        return parse_address(pe["imagebase"], "header.pe.imagebase")
    raise CfgSchemaError("header has no imagebase; RVA/VA normalization needs it")


def _as_rva(value: int, imagebase: int, convert_va: bool) -> int:
    """Subtract the image base when the file stored virtual addresses."""
    if convert_va and imagebase > 0 and value >= imagebase:
        return value - imagebase
    return value


def _is_conditional_mnemonic(mnemonic: str) -> bool:
    normalized = tok._mnemonic(mnemonic).lower()
    return (normalized.startswith("j") and normalized != "jmp") or normalized.startswith("loop")


def _derived_edge_type(src: CfgBlock, dst: CfgBlock) -> str:
    """Derive a useful CFG edge label when schema 2.1 provides no edge type."""
    if dst.external or dst.block_type == FCB_EXTERN:
        return "extern"
    if dst.start_rva == src.end_rva:
        return "fallthrough"
    last = src.instructions[-1].mnemonic if src.instructions else ""
    normalized = tok._mnemonic(last).lower()
    if _is_conditional_mnemonic(normalized):
        return "cond_taken"
    if normalized == "jmp":
        return "indirect" if src.block_type == FCB_INDJUMP else "jmp"
    return "other"


def _annotate_cfg_edges(
    blocks: dict[tuple[int, int], CfgBlock],
    edges: list[CfgEdge],
) -> None:
    """Add derived edge kinds and loop/back-edge flags in place.

    The exported graph remains the original directed CFG. These fields are
    the typed relations and structural flags consumed by the one block-level
    GNN. They do not create a second graph.
    """
    by_function: dict[int, list[int]] = defaultdict(list)
    for index, edge in enumerate(edges):
        by_function[edge.src[0]].append(index)
        src = blocks[edge.src]
        dst = blocks[edge.dst]
        if edge.edge_type is None:
            edge.edge_type = _derived_edge_type(src, dst)
        edge.is_back_address = dst.start_rva <= src.start_rva

    for function_id, edge_indices in by_function.items():
        adjacency: dict[tuple[int, int], list[tuple[tuple[int, int], int]]] = defaultdict(list)
        nodes: set[tuple[int, int]] = set()
        for edge_index in edge_indices:
            edge = edges[edge_index]
            nodes.add(edge.src)
            nodes.add(edge.dst)
            adjacency[edge.src].append((edge.dst, edge_index))
        if not nodes:
            continue
        roots = sorted(nodes, key=lambda key: (blocks[key].start_rva, key[1]))

        state: dict[tuple[int, int], int] = {key: 0 for key in nodes}
        for root in roots:
            if state[root] != 0:
                continue
            state[root] = 1
            stack: list[tuple[tuple[int, int], int]] = [(root, 0)]
            while stack:
                node, offset = stack[-1]
                neighbors = adjacency.get(node, [])
                if offset >= len(neighbors):
                    state[node] = 2
                    stack.pop()
                    continue
                neighbor, edge_index = neighbors[offset]
                stack[-1] = (node, offset + 1)
                if state[neighbor] == 1:
                    edges[edge_index].is_back_edge = True
                elif state[neighbor] == 0:
                    state[neighbor] = 1
                    stack.append((neighbor, 0))


def _blocks_in_file_order_from_blocks(blocks: dict[tuple[int, int], CfgBlock]) -> list[CfgBlock]:
    return sorted(blocks.values(), key=lambda block: (block.function_id, block.start_rva, block.block_id))


def load_cfg(path: Path, file_id: str | None = None) -> FullCFG:
    """Read one JSONL CFG into blocks keyed by ``(function_id, block_id)``."""
    records = list(_iter_jsonl(path))
    header = next((record for record in records if record["record"] == "header"), None)
    if header is None:
        raise CfgSchemaError(f"{path} has no header record")
    schema = header.get("schema")
    if schema != SCHEMA:
        raise CfgSchemaError(f"{path} schema is {schema!r}; this pipeline expects {SCHEMA}")
    _require(header, ("sha256",), "header")
    imagebase = _imagebase_of(header)
    sample_id = str(header["sha256"]).lower()
    if file_id is not None and file_id.lower() != sample_id:
        logger.warning("file name %s does not match header sha256 %s", file_id, sample_id)

    pe = header.get("pe") if isinstance(header.get("pe"), dict) else {}
    raw_imports = pe.get("imports") or []
    if not isinstance(raw_imports, list):
        raise CfgSchemaError("header.pe.imports is not a list")
    imports = [_import_row(row, index) for index, row in enumerate(raw_imports)]

    notes = [
        "CFG edges are [source_block_id, destination_block_id] pairs inside one function. "
        "Schema 2.1 does not store an edge type.",
        "When an edge type is absent, this pipeline derives a heuristic type from "
        "the source terminator, destination RVA, and external/block type metadata.",
        "Block ids are unique inside a function and repeat across functions. "
        "The node key is (function_id, block_id).",
        "Block end_rva is exclusive: an instruction belongs to the block when "
        "start_rva <= rva < end_rva.",
        "The IDA listing stores a VA on labels, not on every instruction. "
        "Per-instruction RVAs come from the CFG and are joined inside a label span.",
    ]
    functions = [record for record in records if record["record"] == "function"]
    if not functions:
        raise CfgSchemaError(f"{path} has no function records")
    footer = next((record for record in records if record["record"] == "footer"), None)
    if footer is None or not footer.get("complete"):
        raise CfgSchemaError(f"{path} footer is missing or marked incomplete")

    rva_values = [
        int(instruction["rva"])
        for function in functions
        for block in function.get("blocks") or []
        for instruction in block.get("instructions") or []
        if isinstance(instruction, dict) and isinstance(instruction.get("rva"), int)
    ]
    convert_va = bool(rva_values) and imagebase > 0 and all(value >= imagebase for value in rva_values)
    if convert_va:
        notes.append(
            "Every instruction rva was >= imagebase, so those values were treated as "
            "virtual addresses and converted with imagebase."
        )
    elif imagebase > 0 and any(value >= imagebase for value in rva_values):
        notes.append(
            "Some instruction rvas are >= imagebase while others are not. "
            "They were left unchanged and listed as address-format problems."
        )

    blocks: dict[tuple[int, int], CfgBlock] = {}
    edges: list[CfgEdge] = []
    function_rvas: dict[str, int] = {}
    seen_function_ids: set[int] = set()
    library_functions: set[int] = set()
    address_problems: list[dict[str, Any]] = []
    odd_rvas: list[int] = []

    for function in functions:
        _require(function, ("id", "name", "rva", "blocks", "edges"), f"function {function.get('name')}")
        function_id = int(function["id"])
        function_name = str(function["name"])
        function_rva = _as_rva(
            parse_address(function["rva"], f"function {function_name} rva"),
            imagebase,
            convert_va,
        )
        if function_name in function_rvas and function_rvas[function_name] != function_rva:
            raise CfgSchemaError(f"function name {function_name} is used at two RVAs")
        if function_id in seen_function_ids:
            raise CfgSchemaError(f"function id {function_id} appears more than once")
        seen_function_ids.add(function_id)
        if function.get("library"):
            library_functions.add(function_id)
        function_rvas[function_name] = function_rva
        block_ids: list[int] = []
        for block in function["blocks"]:
            if not isinstance(block, dict):
                raise CfgSchemaError(f"function {function_name} has a block that is not an object")
            _require(block, ("id", "start_rva", "end_rva", "instructions"), f"function {function_name} block")
            block_id = int(block["id"])
            block_ids.append(block_id)
            start_rva = _as_rva(parse_address(block["start_rva"], "start_rva"), imagebase, convert_va)
            end_rva = _as_rva(parse_address(block["end_rva"], "end_rva"), imagebase, convert_va)
            instructions: list[CfgInstruction] = []
            for index, instruction in enumerate(block["instructions"]):
                if not isinstance(instruction, dict):
                    raise CfgSchemaError(
                        f"function {function_name} block {block_id} instruction {index} is not an object"
                    )
                if "rva" not in instruction:
                    raise CfgSchemaError(
                        f"function {function_name} block {block_id} instruction {index} has no rva. "
                        "Alignment needs an explicit instruction address, not a start/end range alone."
                    )
                rva = _as_rva(parse_address(instruction["rva"], "instruction rva"), imagebase, convert_va)
                if not convert_va and imagebase > 0 and rva >= imagebase:
                    odd_rvas.append(rva)
                mnemonic = str(instruction.get("mnemonic") or "")
                if not mnemonic:
                    raise CfgSchemaError(
                        f"function {function_name} block {block_id} instruction at rva {rva} has no mnemonic"
                    )
                size = int(instruction.get("size") or 0)
                import_ids = [int(item) for item in instruction.get("import_ids") or []]
                for import_index in import_ids:
                    if not 0 <= import_index < len(imports):
                        raise CfgSchemaError(
                            f"instruction rva {rva} import id {import_index} is outside the import table"
                        )
                is_call = bool(instruction.get("is_call")) or tok._mnemonic(mnemonic).lower() == "call"
                instructions.append(
                    CfgInstruction(
                        file_id=sample_id,
                        function_id=function_id,
                        function_name=function_name,
                        block_id=block_id,
                        index_in_block=index,
                        rva=rva,
                        size=size,
                        mnemonic=mnemonic,
                        import_ids=import_ids,
                        call_resolution=(
                            str(instruction["call_resolution"])
                            if instruction.get("call_resolution") is not None
                            else None
                        ),
                        is_call=is_call,
                    )
                )
            key = (function_id, block_id)
            if key in blocks:
                raise CfgSchemaError(f"duplicate block id {block_id} in function {function_id}")
            blocks[key] = CfgBlock(
                file_id=sample_id,
                function_id=function_id,
                function_name=function_name,
                block_id=block_id,
                start_rva=start_rva,
                end_rva=end_rva,
                block_type=int(block["type"]) if block.get("type") is not None else None,
                external=bool(block.get("external")),
                instructions=instructions,
            )
        if len(block_ids) != len(set(block_ids)):
            raise CfgSchemaError(f"function {function_name} repeats a block id")
        id_set = set(block_ids)
        for edge in function["edges"]:
            src_id, dst_id, edge_type = _parse_edge(edge, function_name)
            if src_id not in id_set or dst_id not in id_set:
                raise CfgSchemaError(
                    f"function {function_name} edge {(src_id, dst_id)} points outside its blocks"
                )
            edges.append(
                CfgEdge(
                    src=(function_id, src_id),
                    dst=(function_id, dst_id),
                    edge_type=edge_type,
                )
            )
    if odd_rvas:
        address_problems.append(
            {
                "problem": "rva_ge_imagebase_left_unchanged",
                "count": len(odd_rvas),
                "rvas": odd_rvas[:50],
            }
        )

    _annotate_cfg_edges(blocks, edges)

    return FullCFG(
        file_id=sample_id,
        imagebase=imagebase,
        blocks=blocks,
        edges=edges,
        imports=imports,
        function_rvas=function_rvas,
        header=header,
        schema_notes=notes,
        address_format_problems=address_problems,
        library_functions=library_functions,
    )


def _import_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise CfgSchemaError(f"import {index} is not an object")
    return {
        "name": row.get("name"),
        "dll": row.get("dll"),
        "ordinal": row.get("ordinal"),
        "iat_rva": row.get("rva"),
        "delay": bool(row.get("delayed")),
    }


def _parse_edge(edge: Any, function_name: str) -> tuple[int, int, str | None]:
    if isinstance(edge, (list, tuple)) and len(edge) >= 2:
        edge_type = None if len(edge) < 3 or edge[2] is None else str(edge[2])
        return int(edge[0]), int(edge[1]), edge_type
    if isinstance(edge, dict):
        _require(edge, ("src", "dst"), f"function {function_name} edge")
        edge_type = edge.get("type", edge.get("edge_type"))
        return int(edge["src"]), int(edge["dst"]), None if edge_type is None else str(edge_type)
    raise CfgSchemaError(f"function {function_name} has an unreadable edge: {edge!r}")


def _accepted_asm_line(line: str, record: dict[str, Any]) -> bool:
    code = line.split(";", 1)[0]
    if "<" in code or any("=" in token for token in code.split()):
        return False
    mnemonic = tok._mnemonic(record["opcode"]).lower()
    if mnemonic in _NOT_CODE or not _MNEMONIC.fullmatch(mnemonic):
        return False
    return True


def _consume_asm_line(
    line: str,
    state: dict[str, int | None],
    function_rvas: dict[str, int],
    collapsed: list[str],
    issues: list[dict[str, Any]],
) -> AsmInstruction | None:
    """Update the label address from one listing line and return a code line."""
    collapsed_name = _COLLAPSED.search(line)
    if collapsed_name:
        collapsed.append(collapsed_name.group(1).strip())
    before = state.get("va")
    tok._note_location(line, state)
    after = state.get("va")
    code = line.split(";", 1)[0].strip()
    proc = _PROC.match(code)
    if proc:
        name = proc.group(1)
        function_rva = function_rvas.get(name)
        imagebase = state.get("imagebase") or 0
        if function_rva is not None:
            json_va = imagebase + function_rva
            if after != before and after != json_va:
                issues.append(
                    {
                        "problem": "label_va_disagrees_with_function_rva",
                        "function": name,
                        "label_va": after,
                        "function_va": json_va,
                    }
                )
            state["va"] = json_va
    record = tok.parse_ida_instruction_line(line)
    if record is None or not _accepted_asm_line(line, record):
        return None
    anchor = tok._current_rva(state)
    if anchor is None:
        issues.append({"problem": "asm_instruction_without_address", "raw_text": line.strip()})
    return AsmInstruction(
        raw_text=line.strip(),
        mnemonic=tok._mnemonic(record["opcode"]).lower(),
        anchor_rva=anchor,
        record=record,
    )


def parse_asm_instructions(
    lines: Iterable[str],
    imagebase: int,
    function_rvas: dict[str, int],
    problems: list[dict[str, Any]] | None = None,
) -> tuple[list[AsmInstruction], list[str]]:
    """Walk an IDA listing and keep code lines with the RVA of the current label."""
    issues = problems if problems is not None else []
    collapsed: list[str] = []
    state: dict[str, int | None] = {"imagebase": imagebase, "va": None}
    parsed = [
        item
        for line in lines
        if (item := _consume_asm_line(line, state, function_rvas, collapsed, issues)) is not None
    ]
    return parsed, collapsed


def align_cfg_to_asm(
    cfg: FullCFG,
    asm_instructions: Sequence[AsmInstruction],
    collapsed_functions: Sequence[str] | None = None,
) -> MismatchReport:
    """Attach assembly text to CFG instructions inside each label-address span.

    A span is ``[label_rva, next_label_rva)``. Pairing inside the span follows
    CFG RVA order and assembly file order, and it stops at the first mnemonic
    disagreement so a collapsed function cannot slide later lines onto the
    wrong addresses.
    """
    report = MismatchReport()
    collapsed = list(collapsed_functions or [])
    grouped: dict[int, list[CfgInstruction]] = defaultdict(list)
    for block in cfg.blocks.values():
        for instruction in block.instructions:
            grouped[instruction.rva].append(instruction)
    for rva, rows in sorted(grouped.items()):
        if len(rows) == 1:
            continue
        report.add_duplicate(
            {
                "rva": rva,
                "count": len(rows),
                "blocks": [list(row.block_key) for row in rows],
            }
        )

    anchors = sorted({item.anchor_rva for item in asm_instructions if item.anchor_rva is not None})
    asm_buckets: dict[int, list[AsmInstruction]] = defaultdict(list)
    for item in asm_instructions:
        if item.anchor_rva is None:
            report.add_missing_json(
                {"raw_text": item.raw_text, "reason": "asm_line_has_no_label_rva"}
            )
            continue
        asm_buckets[item.anchor_rva].append(item)

    json_buckets: dict[int, list[CfgInstruction]] = defaultdict(list)
    unplaced: list[CfgInstruction] = []
    for block in _blocks_in_file_order(cfg):
        for instruction in block.instructions:
            if len(grouped[instruction.rva]) != 1:
                continue
            bucket = _anchor_bucket(anchors, instruction.rva)
            if bucket is None:
                unplaced.append(instruction)
            else:
                json_buckets[bucket].append(instruction)

    for instruction in unplaced:
        report.add_missing_asm(_json_gap(instruction, "rva_before_any_asm_label", collapsed))

    for anchor in anchors:
        asm_rows = asm_buckets.get(anchor, [])
        json_rows = sorted(
            json_buckets.get(anchor, []),
            key=lambda item: (item.rva, item.function_id, item.block_id, item.index_in_block),
        )
        matched = 0
        while matched < len(asm_rows) and matched < len(json_rows):
            if asm_rows[matched].mnemonic != tok._mnemonic(json_rows[matched].mnemonic).lower():
                break
            _attach_asm(json_rows[matched], asm_rows[matched], cfg.imagebase)
            matched += 1
        for item in asm_rows[matched:]:
            report.add_missing_json(
                {
                    "raw_text": item.raw_text,
                    "anchor_rva": anchor,
                    "mnemonic": item.mnemonic,
                    "reason": "no_cfg_instruction_in_address_span",
                }
            )
        for item in json_rows[matched:]:
            report.add_missing_asm(
                _json_gap(item, "no_asm_line_in_address_span", collapsed)
            )
    return report


def align_sampled_blocks(
    cfg: FullCFG,
    selected_keys: Sequence[tuple[int, int]],
    lines: Iterable[str],
    problems: list[dict[str, Any]],
) -> MismatchReport:
    """Attach assembly text for the sampled blocks without keeping every code line.

    Listing lines are read one at a time. Only the open label span is stored,
    and a span is discarded when it contains none of the sampled instructions.
    Pairing inside a kept span still starts at the first instruction so a later
    sampled block keeps its original position in that span.
    """
    report = MismatchReport()
    selected = set(selected_keys)
    grouped: dict[int, list[CfgInstruction]] = defaultdict(list)
    for block in cfg.blocks.values():
        for instruction in block.instructions:
            grouped[instruction.rva].append(instruction)
    for rva, rows in sorted(grouped.items()):
        if len(rows) == 1:
            continue
        report.add_duplicate(
            {
                "rva": rva,
                "count": len(rows),
                "blocks": [list(row.block_key) for row in rows],
            }
        )

    pairable = [
        instruction
        for block in _blocks_in_file_order(cfg)
        for instruction in block.instructions
        if len(grouped[instruction.rva]) == 1
    ]
    pairable.sort(key=lambda item: (item.rva, item.function_id, item.block_id, item.index_in_block))
    pairable_rvas = [item.rva for item in pairable]
    selected_rvas = sorted(
        instruction.rva
        for key in selected_keys
        for instruction in cfg.blocks[key].instructions
        if len(grouped[instruction.rva]) == 1
    )
    collapsed: list[str] = []
    state: dict[str, int | None] = {"imagebase": cfg.imagebase, "va": None}
    span_anchor: int | None = None
    span_lines: list[AsmInstruction] = []
    opened = False
    placed: set[int] = set()

    def flush(end_rva: int | None) -> None:
        nonlocal span_lines
        if span_anchor is None:
            for item in span_lines:
                report.add_missing_json(
                    {"raw_text": item.raw_text, "reason": "asm_line_has_no_label_rva"}
                )
            span_lines = []
            return
        left = bisect_left(pairable_rvas, span_anchor)
        right = len(pairable) if end_rva is None else bisect_left(pairable_rvas, end_rva)
        json_rows = pairable[left:right]
        if not any(instruction.block_key in selected for instruction in json_rows):
            span_lines = []
            return
        for instruction in json_rows:
            placed.add(id(instruction))
        matched = 0
        while matched < len(span_lines) and matched < len(json_rows):
            if span_lines[matched].mnemonic != tok._mnemonic(json_rows[matched].mnemonic).lower():
                break
            if json_rows[matched].block_key in selected:
                _attach_asm(json_rows[matched], span_lines[matched], cfg.imagebase)
            matched += 1
        for item in span_lines[matched:]:
            report.add_missing_json(
                {
                    "raw_text": item.raw_text,
                    "anchor_rva": span_anchor,
                    "mnemonic": item.mnemonic,
                    "reason": "no_cfg_instruction_in_address_span",
                }
            )
        for item in json_rows[matched:]:
            if item.block_key in selected:
                report.add_missing_asm(_json_gap(item, "no_asm_line_in_address_span", collapsed))
        span_lines = []

    for line in lines:
        item = _consume_asm_line(line, state, cfg.function_rvas, collapsed, problems)
        anchor = tok._current_rva(state)
        if not opened:
            span_anchor = anchor
            opened = True
        elif anchor != span_anchor:
            flush(anchor)
            span_anchor = anchor
        if item is not None and (span_anchor is None or bisect_left(selected_rvas, span_anchor) < len(selected_rvas)):
            span_lines.append(item)
    if opened:
        flush(None)

    for key in selected_keys:
        for instruction in cfg.blocks[key].instructions:
            if len(grouped[instruction.rva]) != 1 or id(instruction) in placed:
                continue
            report.add_missing_asm(_json_gap(instruction, "rva_before_any_asm_label", collapsed))
    return report


def _anchor_bucket(anchors: Sequence[int], rva: int) -> int | None:
    index = bisect_right(anchors, rva) - 1
    if index < 0:
        return None
    return anchors[index]


def _attach_asm(instruction: CfgInstruction, asm: AsmInstruction, imagebase: int) -> None:
    instruction.raw_text = asm.raw_text
    instruction.va = imagebase + instruction.rva
    instruction.matched = True
    instruction.opcode = asm.record["opcode"]
    instruction.operands = list(asm.record["operands"])


def _json_gap(instruction: CfgInstruction, reason: str, collapsed: Sequence[str]) -> dict[str, Any]:
    folded = instruction.function_name.lower()
    collapsed_hit = next((name for name in collapsed if folded in name.lower() or name.lower() in folded), None)
    return {
        "file_id": instruction.file_id,
        "function_id": instruction.function_id,
        "function_name": instruction.function_name,
        "block_id": instruction.block_id,
        "rva": instruction.rva,
        "mnemonic": instruction.mnemonic,
        "reason": "collapsed_function" if collapsed_hit else reason,
        "collapsed_function": collapsed_hit,
    }


def _blocks_in_file_order(cfg: FullCFG) -> list[CfgBlock]:
    return _blocks_in_file_order_from_blocks(cfg.blocks)


def block_degrees(cfg: FullCFG) -> dict[tuple[int, int], tuple[int, int]]:
    """Return full-CFG in/out degrees for every block.

    These are node features. They do not add edges or change the sampled
    graph; they simply retain the structural context that the cache builder
    computed before sampling.
    """
    indegree: dict[tuple[int, int], int] = {key: 0 for key in cfg.blocks}
    outdegree: dict[tuple[int, int], int] = {key: 0 for key in cfg.blocks}
    for edge in cfg.edges:
        if edge.src in outdegree:
            outdegree[edge.src] += 1
        if edge.dst in indegree:
            indegree[edge.dst] += 1
    return {key: (indegree[key], outdegree[key]) for key in cfg.blocks}


def _undirected_neighbors(cfg: FullCFG) -> dict[tuple[int, int], list[tuple[int, int]]]:
    neighbors: dict[tuple[int, int], set[tuple[int, int]]] = {key: set() for key in cfg.blocks}
    for edge in cfg.edges:
        if edge.src not in neighbors or edge.dst not in neighbors:
            continue
        if edge.src == edge.dst:
            continue
        neighbors[edge.src].add(edge.dst)
        neighbors[edge.dst].add(edge.src)
    return {key: sorted(values) for key, values in neighbors.items()}


def induced_edges(edges: Sequence[CfgEdge], selected: set[tuple[int, int]]) -> list[CfgEdge]:
    """Keep the original directed edges whose endpoints were both selected."""
    kept = [edge for edge in edges if edge.src in selected and edge.dst in selected]
    return sorted(kept, key=lambda edge: (edge.src, edge.dst, edge.edge_type or ""))


def _sampling_keys(cfg: FullCFG, include_library: bool) -> list[tuple[int, int]]:
    """Blocks a sampler may pick: those of non-library functions, or every block with ``include_library``.

    The IDA listing holds no code for a library function (one COLLAPSED FUNCTION line), so its blocks can never be
    paired with assembly. When every function is a library function there is nothing else to pick and all blocks are used.
    """
    if include_library:
        return list(cfg.blocks)
    kept = [key for key in cfg.blocks if key[0] not in cfg.library_functions]
    return kept or list(cfg.blocks)


def sample_connected_subgraph(
    cfg: FullCFG,
    *,
    seed: int,
    num_blocks: int | None = None,
    max_instructions: int | None = None,
    include_library: bool = False,
) -> SampledSubgraph:
    """Grow one connected subgraph from a seeded start block.

    Neighbors come from an undirected view of the CFG so a directed edge can
    be followed either way during the walk. Saved edges keep their original
    direction. A basic block is added whole or not at all. The start block is
    kept even when it alone is larger than ``max_instructions``. ``num_blocks=0``
    sets no block budget: the walk takes its whole connected region.

    Blocks of library functions are skipped: the start block and every
    neighbor come from non-library functions, and a short walk is not topped
    up with library blocks (the listing collapses their code, and they would
    not be connected to it). Only a binary whose functions are all library
    functions is walked over its library blocks, as in
    ``sample_signal_subgraph``. ``include_library=True`` walks over every
    block.
    """
    if num_blocks is None and max_instructions is None:
        raise ValueError("set num_blocks or max_instructions")
    if num_blocks is not None and num_blocks < 0:
        raise ValueError("num_blocks must be >= 0")
    if max_instructions is not None and max_instructions < 1:
        raise ValueError("max_instructions must be >= 1")
    if not cfg.blocks:
        raise ValueError("CFG has no blocks")
    block_cap = num_blocks or None  # 0 sets no block budget

    keys = sorted(_sampling_keys(cfg, include_library), key=lambda key: (cfg.blocks[key].start_rva, key[0], key[1]))
    allowed = set(keys)
    rng = random.Random(seed)
    start = keys[rng.randrange(len(keys))]
    selected = [start]
    chosen = {start}
    total = cfg.blocks[start].instruction_count
    neighbors = _undirected_neighbors(cfg)
    queue = [start]
    while queue:
        if block_cap is not None and len(selected) >= block_cap:
            break
        node = queue.pop(0)
        ordered = list(neighbors[node])
        rng.shuffle(ordered)
        for neighbor in ordered:
            if neighbor in chosen or neighbor not in allowed:
                continue
            if block_cap is not None and len(selected) >= block_cap:
                break
            extra = cfg.blocks[neighbor].instruction_count
            if max_instructions is not None and total + extra > max_instructions:
                continue
            selected.append(neighbor)
            chosen.add(neighbor)
            total += extra
            queue.append(neighbor)
            if block_cap is not None and len(selected) >= block_cap:
                break
    return SampledSubgraph(
        seed=seed,
        selected_keys=selected,
        edges=induced_edges(cfg.edges, chosen),
        original_blocks=len(cfg.blocks),
        original_edges=len(cfg.edges),
        original_instructions=cfg.instruction_count,
    )


def _block_signal(block: CfgBlock) -> int:
    """Prefer blocks that call or reference an import. Those carry the ransomware APIs."""
    return sum(2 * len(instruction.import_ids) + int(instruction.is_call) for instruction in block.instructions)


def sample_signal_subgraph(
    cfg: FullCFG,
    *,
    seed: int,
    num_blocks: int | None = None,
    max_instructions: int | None = None,
    include_library: bool = False,
) -> SampledSubgraph:
    """Keep the blocks of whole functions that reference imports, highest signal first.

    The result is the union of the top-ranked functions' blocks, cut at the
    block budget, not one connected subgraph: CFG edges stay inside a
    function, so the kept blocks form one component per function (and
    ``connected_keys`` is never called). A random walk often stays in library
    code and never reaches the payload. Functions are taken in import-call
    order so the block budget is spent on the code the classifier can actually
    use. Edges stay the original directed CFG inside the kept blocks.

    ``num_blocks=0`` sets no block budget: every non-library block is kept.

    Library functions are skipped. A binary with fewer than ``num_blocks``
    non-library blocks gets just those blocks; it is not topped up with
    library blocks, whose code the listing collapses and which can never be
    matched. Only a binary made of library functions alone is sampled from
    them, in the same order (import-call score, then a seeded shuffle of the
    functions without imports), so it still gets a sample.
    ``include_library=True`` ranks library and other functions together, as
    before.
    """
    if num_blocks is None and max_instructions is None:
        raise ValueError("set num_blocks or max_instructions")
    if not cfg.blocks:
        raise ValueError("CFG has no blocks")
    block_cap = num_blocks or None  # 0 sets no block budget

    by_function: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for key in cfg.blocks:
        by_function[key[0]].append(key)

    library_ids = set() if include_library else cfg.library_functions
    ranked = []
    filler = []
    library_ranked = []
    library_filler = []
    for function_id, keys in by_function.items():
        ordered = sorted(keys, key=lambda key: (cfg.blocks[key].start_rva, key[1]))
        score = sum(_block_signal(cfg.blocks[key]) for key in ordered)
        start = cfg.blocks[ordered[0]].start_rva
        if function_id in library_ids:
            bucket = library_ranked if score else library_filler
        else:
            bucket = ranked if score else filler
        bucket.append((score, start, function_id, ordered))
    rng = random.Random(seed)
    for group, rest in ((ranked, filler), (library_ranked, library_filler)):
        group.sort(key=lambda item: (-item[0], item[1], item[2]))
        rng.shuffle(rest)

    selected: list[tuple[int, int]] = []
    chosen: set[tuple[int, int]] = set()
    total = 0

    def take(functions: list[tuple[int, int, int, list[tuple[int, int]]]]) -> None:
        nonlocal total
        for _score, _start, _function_id, keys in functions:
            if block_cap is not None and len(selected) >= block_cap:
                break
            block_order = sorted(
                keys,
                key=lambda key: (-_block_signal(cfg.blocks[key]), cfg.blocks[key].start_rva, key[1]),
            )
            for key in block_order:
                if block_cap is not None and len(selected) >= block_cap:
                    break
                extra = cfg.blocks[key].instruction_count
                if (
                    max_instructions is not None
                    and selected
                    and total + extra > max_instructions
                ):
                    continue
                selected.append(key)
                chosen.add(key)
                total += extra

    take(ranked + filler)
    if not selected:
        take(library_ranked + library_filler)
    if not selected:
        return sample_connected_subgraph(
            cfg,
            seed=seed,
            num_blocks=num_blocks,
            max_instructions=max_instructions,
            include_library=include_library,
        )
    return SampledSubgraph(
        seed=seed,
        selected_keys=selected,
        edges=induced_edges(cfg.edges, chosen),
        original_blocks=len(cfg.blocks),
        original_edges=len(cfg.edges),
        original_instructions=cfg.instruction_count,
    )


def connected_keys(keys: Sequence[tuple[int, int]], edges: Sequence[CfgEdge]) -> bool:
    """True when the selected nodes are one undirected connected component."""
    chosen = list(keys)
    if len(chosen) <= 1:
        return True
    neighbors: dict[tuple[int, int], set[tuple[int, int]]] = {key: set() for key in chosen}
    chosen_set = set(chosen)
    for edge in edges:
        if edge.src in chosen_set and edge.dst in chosen_set and edge.src != edge.dst:
            neighbors[edge.src].add(edge.dst)
            neighbors[edge.dst].add(edge.src)
    seen = {chosen[0]}
    stack = [chosen[0]]
    while stack:
        node = stack.pop()
        for neighbor in neighbors[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == len(chosen)


def canonical_block_order(cfg: FullCFG, keys: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Stable model order: block start RVA, then function id, then block id."""
    return sorted(keys, key=lambda key: (cfg.blocks[key].start_rva, key[0], key[1]))


def _merge_sample_imports(
    cfg: FullCFG,
    ida_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Prefer the spelling the IDA tokenizer resolves, and fill PE fields from JSON."""
    merged = tok._merge_imports(ida_rows, [])
    by_stem: dict[str, dict[str, Any]] = {}
    for info in merged.values():
        if info.get("name"):
            by_stem.setdefault(_api_stem(str(info["name"])), info)
    for row in cfg.imports:
        name = row.get("name")
        if not name:
            key = f"ordinal:{row.get('dll')}:{row.get('ordinal')}"
            merged.setdefault(
                key,
                {
                    "name": None,
                    "dll": row.get("dll"),
                    "ordinal": row.get("ordinal"),
                    "iat_rva": row.get("iat_rva"),
                    "delay": bool(row.get("delay")),
                },
            )
            continue
        current = by_stem.get(_api_stem(str(name)))
        if current is None:
            current = {
                "name": name,
                "dll": row.get("dll"),
                "ordinal": row.get("ordinal"),
                "iat_rva": row.get("iat_rva"),
                "delay": bool(row.get("delay")),
            }
            merged[str(name).lower()] = current
            by_stem[_api_stem(str(name))] = current
            continue
        for field_name in ("dll", "ordinal", "iat_rva"):
            if current.get(field_name) is None and row.get(field_name) is not None:
                current[field_name] = row[field_name]
        if row.get("delay"):
            current["delay"] = True
    return merged


def _called_names_from_cfg(cfg: FullCFG, imports: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Mark imports referenced by any CFG instruction, including blocks we do not sample."""
    by_stem = {
        _api_stem(str(info["name"])): info
        for info in imports.values()
        if info.get("name")
    }
    called: dict[str, dict[str, Any]] = {}
    for block in cfg.blocks.values():
        for instruction in block.instructions:
            for import_index in instruction.import_ids:
                row = cfg.imports[import_index]
                json_name = row.get("name")
                if not json_name:
                    continue
                info = by_stem.get(_api_stem(str(json_name)))
                name = str(info["name"]) if info and info.get("name") else str(json_name)
                slot = called.get(name.lower())
                if slot is None:
                    called[name.lower()] = {
                        "name": name,
                        "dll": (info or row).get("dll"),
                        "api_id": None,
                        "count": 1,
                    }
                else:
                    slot["count"] += 1
    return called


def tokenize_selected_blocks(
    cfg: FullCFG,
    selected_keys: Sequence[tuple[int, int]],
    asm_lines: Iterable[str],
    api_vocab: tok.ApiVocab,
    *,
    freeze_vocab: bool = False,
    ida_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Tokenize only the selected instructions, using one shared API vocabulary.

    ``ida_rows`` are the listing's import rows when the caller collected them already (``process_sample`` does,
    in the pass that aligns the blocks); without them ``asm_lines`` is read for them.
    """
    if ida_rows is None:
        ida_rows = tok.collect_ida_imports(asm_lines)
    imports = _merge_sample_imports(cfg, ida_rows)
    if not freeze_vocab:
        for info in imports.values():
            if info.get("name"):
                api_vocab.id_for(str(info["name"]))
    called = _called_names_from_cfg(cfg, imports)
    for key, slot in called.items():
        if slot["name"] in api_vocab:
            slot["api_id"] = api_vocab.id_for(slot["name"]) if not freeze_vocab else _frozen_id(api_vocab, slot["name"])
        elif not freeze_vocab:
            slot["api_id"] = api_vocab.id_for(slot["name"])
        else:
            slot["api_id"] = tok.UNK_API

    for key in selected_keys:
        for instruction in cfg.blocks[key].instructions:
            if not instruction.matched or not instruction.raw_text:
                instruction.token_ids = [
                    tok.TOKEN_MAP["UNKNOWN_OPCODE"],
                    tok.TOKEN_MAP["PADDING"],
                    tok.TOKEN_MAP["PADDING"],
                    tok.NO_API,
                    tok.BEHAVIOR_MAP["NONE"],
                ]
                instruction.api_id = tok.NO_API
                instruction.api_kind = "unmatched"
                instruction.behavior_id = tok.BEHAVIOR_MAP["NONE"]
                instruction.behavior_tag = "NONE"
                instruction.behavior_tags = []
                continue
            encoded = tok.tokenize_raw_instruction(
                instruction.raw_text,
                imports,
                api_vocab,
                va=instruction.va,
                rva=instruction.rva,
                freeze_vocab=freeze_vocab,
            )
            if encoded is None:
                instruction.matched = False
                instruction.token_ids = [
                    tok.TOKEN_MAP["UNKNOWN_OPCODE"],
                    tok.TOKEN_MAP["PADDING"],
                    tok.TOKEN_MAP["PADDING"],
                    tok.NO_API,
                    tok.BEHAVIOR_MAP["NONE"],
                ]
                instruction.api_id = tok.NO_API
                instruction.api_kind = "unmatched"
                instruction.behavior_id = tok.BEHAVIOR_MAP["NONE"]
                instruction.behavior_tag = "NONE"
                instruction.behavior_tags = []
                continue
            instruction.token_ids = list(encoded["token_ids"])
            instruction.api_name = encoded.get("api_name")
            instruction.api_id = int(encoded["token_ids"][3])
            instruction.api_kind = encoded.get("api_kind")
            instruction.api_source = encoded.get("api_source")
            instruction.behavior_id = int(encoded["token_ids"][4])
            instruction.behavior_tag = encoded.get("behavior_tag")
            instruction.behavior_tags = list(encoded.get("behavior_tags") or [])
            instruction.opcode = encoded.get("opcode")
            instruction.operands = list(encoded.get("operands") or [])
            if (
                instruction.api_kind == "known"
                and instruction.api_name
                and instruction.api_source in {"called_api", "ida_comment"}
                and instruction.api_name.lower() not in called
            ):
                called[instruction.api_name.lower()] = {
                    "name": instruction.api_name,
                    "dll": encoded.get("dll_name"),
                    "api_id": instruction.api_id,
                    "count": 1,
                }
    for slot in called.values():
        if slot.get("api_id") is None and slot.get("name"):
            slot["api_id"] = tok.UNK_API if freeze_vocab else api_vocab.id_for(slot["name"])
    inventory = tok._inventory(imports, called, tok._import_source(cfg.imports, ida_rows))
    if cfg.imports and ida_rows:
        inventory["import_source"] = "cfg_json+ida_extrn"
    elif cfg.imports:
        inventory["import_source"] = "cfg_json"
    return inventory


def _frozen_id(vocab: tok.ApiVocab, name: str) -> int:
    if name not in vocab:
        return tok.UNK_API
    return vocab.id_for(name)


def _count_into(target: list[float], names: Sequence[str], token_id: int | None, table: dict[str, int]) -> None:
    if token_id is None:
        return
    for index, name in enumerate(names):
        if table.get(name) == token_id:
            target[index] += 1
            return


def block_feature_vector(
    block: CfgBlock,
    *,
    indegree: int = 0,
    outdegree: int = 0,
) -> list[float]:
    """Fixed-width handcrafted features. API names stay beside this vector.

    The first structural fields mirror the cache builder: calls, import
    references, external status, and full-CFG degree are retained as node
    features. They are not additional graph edges.
    """
    behavior = [0.0] * len(_BEHAVIOR_NAMES)
    opcode = [0.0] * len(_OPCODE_NAMES)
    operand = [0.0] * len(_OPERAND_NAMES)
    api_none = api_unresolved = api_known = 0.0
    call_count = 0
    import_reference_count = 0
    for instruction in block.instructions:
        call_count += int(instruction.is_call)
        import_reference_count += len(instruction.import_ids)
        token_ids = instruction.token_ids or []
        behavior_id = instruction.behavior_id if instruction.behavior_id is not None else (
            token_ids[4] if len(token_ids) == TOKEN_WIDTH else None
        )
        _count_into(behavior, _BEHAVIOR_NAMES, behavior_id, tok.BEHAVIOR_MAP)
        if len(token_ids) == TOKEN_WIDTH:
            _count_into(opcode, _OPCODE_NAMES, token_ids[0], tok.TOKEN_MAP)
            _count_into(operand, _OPERAND_NAMES, token_ids[1], tok.TOKEN_MAP)
            _count_into(operand, _OPERAND_NAMES, token_ids[2], tok.TOKEN_MAP)
        kind = instruction.api_kind
        if kind == "known":
            api_known += 1
        elif kind in {"unresolved", "unseen"}:
            api_unresolved += 1
        elif kind == "none":
            api_none += 1
    return [
        float(block.instruction_count),
        float(block.byte_size),
        float(call_count),
        float(import_reference_count),
        float(block.external),
        float(indegree),
        float(outdegree),
        *behavior,
        api_none,
        api_unresolved,
        api_known,
        *opcode,
        *operand,
    ]


def api_table_size(vocab: tok.ApiVocab) -> int:
    """Rows needed so NO_API, UNK_API, and every shared API id have a vector."""
    return tok.FIRST_API_ID + len(vocab.names)


class FiveChannelEmbedder(nn.Module):
    """One embedding table per tokenizer channel.

    An instruction vector is the sum of the opcode, two operand, API, and
    behavior lookups. Index 0 stays the zero vector, which covers padding,
    ``NO_API``, and behavior ``NONE``. The tables are trainable. A saved
    block vector is the pool of these lookups, not a mean of the integer ids.

    With ``seed`` set, every row added by ``expand_api`` is drawn from a
    generator seeded by that seed and the row index, so the same API id gets
    the same vector whatever the order and size of the growth steps.
    """

    def __init__(self, api_vocab_size: int, d_model: int = BLOCK_EMBED_DIM, seed: int | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.seed = seed
        if api_vocab_size < tok.FIRST_API_ID:
            api_vocab_size = tok.FIRST_API_ID
        self.opcode_embedding = nn.Embedding(OPCODE_VOCAB_SIZE, d_model, padding_idx=0)
        self.operand_embedding = nn.Embedding(OPERAND_VOCAB_SIZE, d_model, padding_idx=0)
        self.api_embedding = nn.Embedding(api_vocab_size, d_model, padding_idx=0)
        self.behavior_embedding = nn.Embedding(tok.BEHAVIOR_VOCAB_SIZE, d_model, padding_idx=0)

    def expand_api(self, api_vocab_size: int) -> None:
        """Grow the shared API table when the vocabulary gains names."""
        if api_vocab_size <= self.api_embedding.num_embeddings:
            return
        fresh = nn.Embedding(api_vocab_size, self.d_model, padding_idx=0)
        with torch.no_grad():
            known = self.api_embedding.num_embeddings
            fresh.weight[:known] = self.api_embedding.weight
            fresh.weight[0].zero_()
            if self.seed is not None:
                for row in range(known, api_vocab_size):
                    generator = torch.Generator().manual_seed(self.seed * 1_000_003 + row)
                    fresh.weight[row] = torch.randn(self.d_model, generator=generator)
        self.api_embedding = fresh

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map ``[N, 5]`` token ids to ``[N, d_model]`` instruction vectors."""
        if token_ids.numel() == 0:
            return torch.zeros(0, self.d_model)
        opcode = token_ids[:, 0].clamp(0, self.opcode_embedding.num_embeddings - 1)
        operand1 = token_ids[:, 1].clamp(0, self.operand_embedding.num_embeddings - 1)
        operand2 = token_ids[:, 2].clamp(0, self.operand_embedding.num_embeddings - 1)
        behavior = token_ids[:, 4].clamp(0, self.behavior_embedding.num_embeddings - 1)
        api = token_ids[:, 3]
        if api.numel() and int(api.max()) >= self.api_embedding.num_embeddings:
            raise ValueError(
                f"API id {int(api.max())} is outside the shared API embedding "
                f"table of size {self.api_embedding.num_embeddings}"
            )
        return (
            self.opcode_embedding(opcode)
            + self.operand_embedding(operand1)
            + self.operand_embedding(operand2)
            + self.api_embedding(api.clamp(min=0))
            + self.behavior_embedding(behavior)
        )


def pool_block_embeddings(
    token_ids: torch.Tensor,
    insn_to_block: torch.Tensor,
    num_blocks: int,
    embedder: FiveChannelEmbedder,
) -> torch.Tensor:
    """Mean-pool instruction embedding vectors inside each block.

    A block with no instructions is the zero vector. The mean is taken after
    each channel has been embedded, so token ids are never averaged.
    """
    width = embedder.d_model
    if num_blocks == 0:
        return torch.zeros(0, width)
    if token_ids.numel() == 0:
        return torch.zeros(num_blocks, width)
    with torch.no_grad():
        vectors = embedder.embed_tokens(token_ids)
        totals = torch.zeros(num_blocks, width, dtype=vectors.dtype)
        counts = torch.zeros(num_blocks, 1, dtype=vectors.dtype)
        totals.index_add_(0, insn_to_block, vectors)
        counts.index_add_(
            0,
            insn_to_block,
            torch.ones(vectors.shape[0], 1, dtype=vectors.dtype),
        )
        return totals / counts.clamp(min=1.0)


def api_name_counts(block: CfgBlock) -> dict[str, int]:
    counts: dict[str, int] = {}
    for instruction in block.instructions:
        if instruction.api_kind == "known" and instruction.api_name:
            counts[instruction.api_name] = counts.get(instruction.api_name, 0) + 1
    return counts


def build_model_inputs(
    cfg: FullCFG,
    sample: SampledSubgraph,
    *,
    label: int,
    embedder: FiveChannelEmbedder,
) -> dict[str, Any]:
    """Pack the same sampled blocks for the three models.

    ``edge_index`` contains the sampled directed CFG. ``edge_type`` gives one
    relation ID for every column of ``edge_index`` and ``edge_attr`` is its
    one-hot form. A relation-aware single GNN can therefore distinguish, for
    example, a fallthrough edge from a conditional-taken edge without building
    a separate function/call graph.
    """
    order = canonical_block_order(cfg, sample.selected_keys)
    index_of = {key: index for index, key in enumerate(order)}
    degrees = block_degrees(cfg)

    instruction_rows: list[list[int]] = []
    instruction_rvas: list[int] = []
    insn_to_block: list[int] = []
    matched_flags: list[bool] = []
    boundaries: list[dict[str, Any]] = []
    block_features: list[list[float]] = []
    block_api_counts: list[dict[str, int]] = []
    block_types: list[int] = []
    block_external: list[bool] = []

    for key in order:
        block = cfg.blocks[key]
        start = len(instruction_rows)
        for instruction in block.instructions:
            instruction_rows.append(list(instruction.token_ids or [0, 0, 0, 0, 0]))
            instruction_rvas.append(instruction.rva)
            insn_to_block.append(index_of[key])
            matched_flags.append(bool(instruction.matched))
        boundaries.append(
            {
                "function_id": key[0],
                "block_id": key[1],
                "start_instruction": start,
                "end_instruction": len(instruction_rows),
                "start_rva": block.start_rva,
                "end_rva": block.end_rva,
            }
        )
        indegree, outdegree = degrees.get(key, (0, 0))
        block_features.append(block_feature_vector(block, indegree=indegree, outdegree=outdegree))
        block_api_counts.append(api_name_counts(block))
        block_types.append(-1 if block.block_type is None else int(block.block_type))
        block_external.append(bool(block.external))

    edge_rows: list[tuple[int, int, int, int, int]] = []
    for edge in sample.edges:
        edge_rows.append(
            (
                index_of[edge.src],
                index_of[edge.dst],
                _edge_type_id(edge.edge_type),
                int(edge.is_back_edge),
                int(edge.is_back_address),
            )
        )
    edge_rows.sort(key=lambda row: (row[0], row[1], row[2], row[3], row[4]))
    edge_src = [row[0] for row in edge_rows]
    edge_dst = [row[1] for row in edge_rows]
    edge_types = [row[2] for row in edge_rows]
    edge_back = [row[3] for row in edge_rows]
    edge_back_addr = [row[4] for row in edge_rows]

    instruction_ids = _long_matrix(instruction_rows, TOKEN_WIDTH)
    features = _float_matrix(block_features, len(HANDCRAFTED_FEATURE_NAMES))
    insn_to_block_tensor = torch.tensor(insn_to_block, dtype=torch.long)
    embeddings = pool_block_embeddings(
        instruction_ids,
        insn_to_block_tensor,
        len(order),
        embedder,
    )
    edge_index = (
        torch.tensor([edge_src, edge_dst], dtype=torch.long)
        if edge_src
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    edge_attr = _edge_one_hot(edge_type)
    edge_back_tensor = torch.tensor(edge_back, dtype=torch.bool)
    edge_back_addr_tensor = torch.tensor(edge_back_addr, dtype=torch.bool)
    shared_blocks = [{"function_id": key[0], "block_id": key[1]} for key in order]
    common_block_metadata = {
        "block_keys": shared_blocks,
        "block_types": torch.tensor(block_types, dtype=torch.long),
        "block_external": torch.tensor(block_external, dtype=torch.bool),
    }
    model1 = {
        "instruction_ids": instruction_ids,
        "insn_to_block": insn_to_block_tensor,
        "instruction_matched": torch.tensor(matched_flags, dtype=torch.bool),
        "block_boundaries": boundaries,
        **common_block_metadata,
    }
    model2 = {
        "block_features": features,
        "block_api_counts": block_api_counts,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "edge_attr": edge_attr,
        "edge_type_names": list(EDGE_TYPE_NAMES),
        "cfg_edge_back": edge_back_tensor,
        "cfg_edge_back_addr": edge_back_addr_tensor,
        **common_block_metadata,
    }
    model3 = {
        "block_embeddings": embeddings,
        "block_features": features.clone(),
        "block_api_counts": block_api_counts,
        "edge_index": edge_index.clone(),
        "edge_type": edge_type.clone(),
        "edge_attr": edge_attr.clone(),
        "edge_type_names": list(EDGE_TYPE_NAMES),
        "cfg_edge_back": edge_back_tensor.clone(),
        "cfg_edge_back_addr": edge_back_addr_tensor.clone(),
        **common_block_metadata,
    }
    return {
        "file_id": cfg.file_id,
        "label": int(label),
        "instruction_rvas": instruction_rvas,
        "graph_format_version": GRAPH_FORMAT_VERSION,
        "edge_type_names": list(EDGE_TYPE_NAMES),
        "model1": model1,
        "model2": model2,
        "model3": model3,
    }


def _edge_type_id(edge_type: str | None) -> int:
    if edge_type is None:
        return UNKNOWN_EDGE_TYPE_ID
    normalized = str(edge_type).strip().lower()
    aliases = {
        "conditional": "cond_taken",
        "conditional_taken": "cond_taken",
        "direct": "jmp",
        "direct_jump": "jmp",
        "indirect_jump": "indirect",
        "external": "extern",
        "unknown": "unknown",
    }
    normalized = aliases.get(normalized, normalized)
    return EDGE_TYPE_TO_ID.get(normalized, UNKNOWN_EDGE_TYPE_ID)


def _edge_one_hot(edge_type: torch.Tensor) -> torch.Tensor:
    """Create ``[number_of_edges, number_of_edge_types]`` relation features."""
    if edge_type.numel() == 0:
        return torch.empty((0, len(EDGE_TYPE_NAMES)), dtype=torch.float32)
    return torch.nn.functional.one_hot(
        edge_type.clamp(min=0, max=len(EDGE_TYPE_NAMES) - 1),
        num_classes=len(EDGE_TYPE_NAMES),
    ).to(dtype=torch.float32)


def _long_matrix(rows: list[list[int]], width: int) -> torch.Tensor:
    if not rows:
        return torch.empty((0, width), dtype=torch.long)
    return torch.tensor(rows, dtype=torch.long)


def _float_matrix(rows: list[list[float]], width: int) -> torch.Tensor:
    if not rows:
        return torch.empty((0, width), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def _instruction_manifest(instruction: CfgInstruction) -> dict[str, Any]:
    return {
        "rva": instruction.rva,
        "va": instruction.va,
        "size": instruction.size,
        "index_in_block": instruction.index_in_block,
        "function_id": instruction.function_id,
        "block_id": instruction.block_id,
        "raw_text": instruction.raw_text,
        "matched": instruction.matched,
        "mnemonic": instruction.mnemonic,
        "import_ids": instruction.import_ids,
        "is_call": instruction.is_call,
        "call_resolution": instruction.call_resolution,
        "opcode": instruction.opcode,
        "operands": instruction.operands,
        "token_ids": instruction.token_ids,
        "api_id": instruction.api_id,
        "api_name": instruction.api_name,
        "api_kind": instruction.api_kind,
        "api_source": instruction.api_source,
        "behavior_id": instruction.behavior_id,
        "behavior_tag": instruction.behavior_tag,
        "behavior_tags": instruction.behavior_tags,
    }


def build_manifest(
    cfg: FullCFG,
    sample: SampledSubgraph,
    models: dict[str, Any],
    report: MismatchReport,
    inventory: dict[str, Any],
    *,
    label: int,
    source_asm: str,
    source_cfg: str,
    split: str | None,
    dataset_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    order = canonical_block_order(cfg, sample.selected_keys)
    blocks = []
    for node_index, key in enumerate(order):
        block = cfg.blocks[key]
        raw_lines = [instruction.raw_text for instruction in block.instructions if instruction.raw_text]
        blocks.append(
            {
                "node_index": node_index,
                "function_id": key[0],
                "function_name": block.function_name,
                "block_id": key[1],
                "start_rva": block.start_rva,
                "end_rva": block.end_rva,
                "block_type": block.block_type,
                "external": block.external,
                "instruction_rvas": [instruction.rva for instruction in block.instructions],
                "raw_block_text": "\n".join(raw_lines),
                "api_names": [
                    instruction.api_name
                    for instruction in block.instructions
                    if instruction.api_name
                ],
                "behavior_tags": [
                    tag
                    for instruction in block.instructions
                    for tag in instruction.behavior_tags
                ],
                "api_counts": api_name_counts(block),
                "feature_vector": models["model2"]["block_features"][node_index].tolist(),
                "instructions": [_instruction_manifest(instruction) for instruction in block.instructions],
            }
        )
    return {
        "sample_id": cfg.file_id,
        "graph_format_version": GRAPH_FORMAT_VERSION,
        "edge_type_names": list(EDGE_TYPE_NAMES),
        "source_asm": source_asm,
        "source_cfg": source_cfg,
        "label": int(label),
        "split": split,
        "imagebase": cfg.imagebase,
        "random_seed": sample.seed,
        "schema_notes": cfg.schema_notes,
        "selected_block_ids": [
            {"function_id": key[0], "block_id": key[1]} for key in sample.selected_keys
        ],
        "model_block_ids": [
            {"function_id": key[0], "block_id": key[1]} for key in order
        ],
        "block_boundaries": models["model1"]["block_boundaries"],
        "instruction_rvas": models["instruction_rvas"],
        "blocks": blocks,
        "induced_edges": [
            {
                "src_function_id": edge.src[0],
                "src_block_id": edge.src[1],
                "dst_function_id": edge.dst[0],
                "dst_block_id": edge.dst[1],
                "edge_type": edge.edge_type,
                "edge_type_id": _edge_type_id(edge.edge_type),
                "is_back_edge": edge.is_back_edge,
                "is_back_address": edge.is_back_address,
            }
            for edge in sample.edges
        ],
        "dataset_metadata": dict(dataset_metadata or {}),
        "sampling": {
            "sampled_blocks": sample.sampled_blocks,
            "sampled_instructions": sample.sampled_instructions(cfg),
        },
        "original_graph_size": {
            "blocks": sample.original_blocks,
            "edges": sample.original_edges,
            "instructions": sample.original_instructions,
        },
        "sampled_graph_size": {
            "blocks": sample.sampled_blocks,
            "edges": len(sample.edges),
            "instructions": sample.sampled_instructions(cfg),
        },
        "import_inventory": inventory,
        "mismatch_report": report.to_dict(),
        "handcrafted_feature_names": list(HANDCRAFTED_FEATURE_NAMES),
    }


def process_sample(
    asm_path: Path,
    cfg_path: Path,
    *,
    label: int,
    seed: int,
    api_vocab: tok.ApiVocab,
    num_blocks: int | None = None,
    max_instructions: int | None = None,
    sample_mode: str = "walk",
    include_library: bool = False,
    split: str | None = None,
    freeze_vocab: bool = False,
    file_id: str | None = None,
    channel_embedder: FiveChannelEmbedder | None = None,
    dataset_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the full CFG, sample one subgraph, align it, and build model inputs."""
    cfg = load_cfg(cfg_path, file_id=file_id)
    sampler = sample_signal_subgraph if sample_mode == "signal" else sample_connected_subgraph
    if sample_mode not in {"walk", "signal"}:
        raise ValueError(f"sample_mode must be 'walk' or 'signal', got {sample_mode!r}")
    sample = sampler(
        cfg,
        seed=seed,
        num_blocks=num_blocks,
        max_instructions=max_instructions,
        include_library=include_library,
    )
    problems: list[dict[str, Any]] = []
    imports = tok.IdaImportCollector()

    def listing_once() -> Iterable[str]:
        for line in iter_text_lines(asm_path):
            imports.feed(line)
            yield line

    report = align_sampled_blocks(
        cfg,
        sample.selected_keys,
        listing_once(),
        problems,
    )
    report.add_format_problems(cfg.address_format_problems)
    report.add_format_problems(problems)
    inventory = tokenize_selected_blocks(
        cfg,
        sample.selected_keys,
        (),
        api_vocab,
        freeze_vocab=freeze_vocab,
        ida_rows=imports.rows,
    )
    embedder = channel_embedder or FiveChannelEmbedder(api_table_size(api_vocab))
    embedder.expand_api(api_table_size(api_vocab))
    models = build_model_inputs(cfg, sample, label=label, embedder=embedder)
    manifest = build_manifest(
        cfg,
        sample,
        models,
        report,
        inventory,
        label=label,
        source_asm=str(asm_path),
        source_cfg=str(cfg_path),
        split=split,
        dataset_metadata=dataset_metadata,
    )
    if any(report.to_dict().values()):
        logger.warning(
            "%s alignment gaps: missing asm=%s missing json=%s duplicate=%s format=%s",
            cfg.file_id,
            report.missing_asm_count,
            report.missing_json_count,
            report.duplicate_count,
            report.address_format_count,
        )
    logger.info(
        "%s sampled %s/%s blocks seed=%s",
        cfg.file_id,
        sample.sampled_blocks,
        sample.original_blocks,
        seed,
    )
    return {
        "manifest": manifest,
        "models": models,
        "cfg": cfg,
        "sample": sample,
        "embedder": embedder,
    }


def save_sample(result: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write the manifest JSON and the tensor file for one sampled subgraph."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = result["manifest"]
    models = result["models"]
    sample_id = manifest["sample_id"]
    manifest_path = output_dir / f"{sample_id}.manifest.json"
    tensor_path = output_dir / f"{sample_id}.pt"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    torch.save(
        {
            "graph_format_version": GRAPH_FORMAT_VERSION,
            "file_id": sample_id,
            "label": manifest["label"],
            "dataset_metadata": manifest.get("dataset_metadata", {}),
            "random_seed": manifest["random_seed"],
            "instruction_ids": models["model1"]["instruction_ids"],
            "insn_to_block": models["model1"]["insn_to_block"],
            "instruction_matched": models["model1"]["instruction_matched"],
            "instruction_rvas": models["instruction_rvas"],
            "block_features": models["model2"]["block_features"],
            "block_embeddings": models["model3"]["block_embeddings"],
            "block_embedding_dim": int(models["model3"]["block_embeddings"].shape[1]),
            "edge_index": models["model2"]["edge_index"],
            "edge_type": models["model2"]["edge_type"],
            "edge_attr": models["model2"]["edge_attr"],
            "edge_type_names": models["model2"]["edge_type_names"],
            "edge_back": models["model2"]["cfg_edge_back"],
            "edge_back_addr": models["model2"]["cfg_edge_back_addr"],
            "block_types": models["model2"]["block_types"],
            "block_external": models["model2"]["block_external"],
            "block_keys": models["model1"]["block_keys"],
            "model1_block_keys": models["model1"]["block_keys"],
            "model2_block_keys": models["model2"]["block_keys"],
            "model3_block_keys": models["model3"]["block_keys"],
        },
        tensor_path,
    )
    logger.info("wrote %s and %s", manifest_path, tensor_path)
    return manifest_path, tensor_path


def _cfg_path_for_asm(asm_path: Path) -> Path:
    name = asm_path.name
    if name.endswith(".asm.gz"):
        return asm_path.with_name(name[: -len(".asm.gz")] + ".jsonl.gz")
    if name.endswith(".asm"):
        return asm_path.with_suffix(".jsonl.gz")
    raise CfgSchemaError(f"cannot derive a CFG path from {asm_path}")


def _run_settings(args: argparse.Namespace) -> dict[str, Any]:
    """What decides the content of a sample file. A folder keeps one such set, in ``settings.json``."""
    return {
        "sample": args.sample,
        "seed": args.seed,
        "num_blocks": args.num_blocks,
        "max_instructions": args.max_instructions,
        "include_library": args.include_library,
        "freeze_vocab": args.freeze_vocab,
        "split_order": "train rows grow the vocabulary first, every other split is frozen",
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest()
            for path in (Path(__file__).resolve(), Path(tok.__file__).resolve())
        },
    }


def _check_settings(parser: argparse.ArgumentParser, output: Path, settings: dict[str, Any], overwrite: bool) -> None:
    """Refuse to add samples to a folder that was built with other settings, then record these settings."""
    path = output / "settings.json"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        changed = sorted(key for key in settings.keys() | previous.keys() if settings.get(key) != previous.get(key))
        if changed and not overwrite:
            parser.error(
                f"{output} was built with other settings ({', '.join(changed)}); "
                "use a new --output or --overwrite to rebuild it"
            )
    elif not overwrite and (output.is_dir() and any(output.glob("*.pt"))):
        parser.error(f"{output} has samples but no settings.json; use a new --output or --overwrite to rebuild it")
    output.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def _cfg_path_for_row(row: dict[str, str], cfg_root: Path | None) -> Path:
    """Where the CFG of a splits-CSV row is.

    The row's ``cfg_path`` column (relative to the collection root) wins; without it the CFG is named like the
    listing. ``cfg_root`` moves both to a separate tree: the column is read relative to it, and a derived name keeps
    the listing's path relative to its collection root.
    """
    root = Path(row["root"])
    base = root if cfg_root is None else cfg_root
    if row.get("cfg_path"):
        path = (base / row["cfg_path"]).resolve()
        if base.resolve() not in path.parents:
            raise CfgSchemaError(f"unsafe CFG path: {row['cfg_path']}")
        return path
    asm_path = Path(row["asm_path"])
    if cfg_root is not None:
        asm_path = cfg_root / asm_path.relative_to(root)
    return _cfg_path_for_asm(asm_path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the preprocess. The exit status is 1 when any sample failed (see ``failures.json`` in the output folder)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, dest="roots")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--architecture", choices=["x86", "x64"])
    parser.add_argument("--sample-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-blocks",
        type=int,
        help="Block budget. 0 sets none: every non-library block of the binary is kept (see --whole).",
    )
    parser.add_argument("--whole", action="store_true", help="Same as --num-blocks 0.")
    parser.add_argument("--max-instructions", type=int)
    parser.add_argument(
        "--sample",
        choices=["walk", "signal"],
        default="walk",
        help="walk grows from a random block. signal keeps import-calling functions first.",
    )
    parser.add_argument(
        "--include-library",
        action="store_true",
        help="Sample blocks of library functions too. By default they are skipped: the listing has no code for them.",
    )
    parser.add_argument(
        "--cfg-root",
        type=Path,
        help="Folder that holds the CFGs when they are not beside the listings. A cfg_path column of the splits CSV is "
        "read relative to it, otherwise the CFG keeps the listing's path relative to its collection root.",
    )
    parser.add_argument("--output", type=Path, default=tok.PROJECT_ROOT / "processed" / "cfg_subgraphs")
    parser.add_argument(
        "--max-asm-bytes",
        type=int,
        default=0,
        help="Skip an uncompressed listing larger than this many bytes (asm_bytes of the splits CSV). 0, the default, sets no cap.",
    )
    parser.add_argument(
        "--max-cfg-blocks",
        type=int,
        default=0,
        help="Skip a CFG with more blocks than this (num_blocks of the splits CSV). 0, the default, sets no cap.",
    )
    parser.add_argument(
        "--max-cfg-instructions",
        type=int,
        default=0,
        help="Skip a CFG with more instructions than this (num_instructions of the splits CSV). 0, the default, sets no cap.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild samples that already have a .pt file.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save the embedder and the shared vocabulary after this many new samples, and at the end of the run.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first sample that fails. By default the run goes on and lists every failure in failures.json.",
    )
    parser.add_argument(
        "--freeze-vocab",
        action="store_true",
        help="Keep the shared API vocabulary unchanged for every split. Unseen names use UNK_API.",
    )
    parser.add_argument("--vocab", type=Path, default=tok.API_VOCAB_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.whole:
        if args.num_blocks not in (None, 0):
            parser.error("--whole means --num-blocks 0; drop --num-blocks or set it to 0")
        args.num_blocks = 0
    if args.num_blocks is None and args.max_instructions is None:
        parser.error("set --num-blocks (0 keeps every block), --whole or --max-instructions")
    if args.num_blocks == 0 and args.sample == "walk":
        logging.warning("no block budget with --sample walk keeps one connected region; --sample signal keeps every block")

    roots = [path.resolve() for path in (args.roots or tok.DEFAULT_ROOTS)]
    selected = tok._selected_rows(roots, args.split, args.architecture, args.sample_id)
    tok.require_rows(selected, roots, args.split, args.architecture, args.sample_id)
    # The vocabulary grows on training rows only: they go first, in every root, and every other split is frozen.
    selected.sort(key=lambda row: row["split"] != "train")
    if args.limit is not None:
        selected = selected[: args.limit]
    full_run = args.limit is None and args.sample_id is None and not args.freeze_vocab
    _check_settings(parser, args.output, _run_settings(args), args.overwrite)
    vocab = tok.load_api_vocab(args.vocab)
    torch.manual_seed(args.seed)
    embedder = FiveChannelEmbedder(api_table_size(vocab), seed=args.seed)
    embedder_path = args.output / "channel_embedder.pt"
    can_resume = embedder_path.is_file() and not args.overwrite
    # checkpoint.json lists the samples whose embedder rows and vocabulary ids are on disk. It is kept when the
    # vocabulary on disk is the one the samples used: a full run writes it, a frozen run leaves it unchanged.
    # --limit and --sample-id runs never write the shared vocabulary, so they do not resume.
    checkpoint_path = args.output / "checkpoint.json"
    resumable = full_run or args.freeze_vocab
    done: set[str] = set()
    if can_resume:
        state = torch.load(embedder_path, map_location="cpu")
        embedder.expand_api(int(state["api_embedding.weight"].shape[0]))
        embedder.load_state_dict(state)
        logger.info("resumed embedder %s", embedder_path)
        if checkpoint_path.is_file():
            done = set(json.loads(checkpoint_path.read_text(encoding="utf-8"))["done"])

    def save_state() -> None:
        args.output.mkdir(parents=True, exist_ok=True)
        torch.save(embedder.state_dict(), embedder_path)
        if full_run:
            tok.save_api_vocab(vocab, args.vocab)
        if resumable:
            checkpoint_path.write_text(json.dumps({"done": sorted(done)}), encoding="utf-8")

    since_save = 0
    saved = 0
    skipped = 0
    oversized = 0
    failures: list[dict[str, str]] = []
    skipped_over_cap: list[dict[str, str]] = []
    for row in selected:
        sample_id = row["sample_id"]
        tensor_path = args.output / f"{sample_id}.pt"
        asm_bytes = int(row.get("asm_bytes") or 0)
        cfg_blocks = int(row.get("num_blocks") or 0)
        cfg_instructions = int(row.get("num_instructions") or 0)
        if can_resume and sample_id in done and tensor_path.is_file():
            logger.info("skip existing %s", sample_id)
            skipped += 1
            continue
        reasons = []
        if args.max_asm_bytes and asm_bytes > args.max_asm_bytes:
            reasons.append(f"asm_bytes={asm_bytes}")
        if args.max_cfg_blocks and cfg_blocks > args.max_cfg_blocks:
            reasons.append(f"blocks={cfg_blocks}")
        if args.max_cfg_instructions and cfg_instructions > args.max_cfg_instructions:
            reasons.append(f"instructions={cfg_instructions}")
        if reasons:
            logger.warning("skip %s: %s", sample_id, ", ".join(reasons))
            skipped_over_cap.append({"sample_id": sample_id, "reason": ", ".join(reasons)})
            oversized += 1
            continue
        asm_path = Path(row["asm_path"])
        try:
            cfg_path = _cfg_path_for_row(row, args.cfg_root)
            if not cfg_path.is_file():
                raise CfgSchemaError(f"missing CFG: {cfg_path}")
            result = process_sample(
                asm_path,
                cfg_path,
                label=int(row["label"]),
                seed=args.seed,
                api_vocab=vocab,
                num_blocks=args.num_blocks,
                max_instructions=args.max_instructions,
                sample_mode=args.sample,
                include_library=args.include_library,
                split=row["split"],
                freeze_vocab=args.freeze_vocab or row["split"] != "train",
                file_id=sample_id,
                channel_embedder=embedder,
                dataset_metadata=row,
            )
            save_sample(result, args.output)
            del result
            gc.collect()
            saved += 1
            done.add(sample_id)
            since_save += 1
            if since_save >= args.checkpoint_every:
                save_state()
                since_save = 0
        except Exception as error:
            logger.error("failed %s: %s", sample_id, error)
            failures.append({"sample_id": sample_id, "error": f"{type(error).__name__}: {error}"})
            gc.collect()
            if args.fail_fast:
                break
    failed = len(failures)
    failures_path = args.output / "failures.json"
    for path, items in ((failures_path, failures), (args.output / "skipped.json", skipped_over_cap)):
        if items:
            args.output.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(items, indent=2), encoding="utf-8")
        elif path.is_file():
            path.unlink()
    if since_save:
        save_state()
    if saved:
        logger.info("wrote %s", embedder_path)
        if full_run:
            logger.info("wrote %s", args.vocab)
    logger.info(
        "saved %s skipped %s oversized %s failed %s",
        saved,
        skipped,
        oversized,
        failed,
    )
    if failed:
        logger.error("%s sample(s) failed, listed in %s", failed, failures_path)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())