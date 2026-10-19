"""Explain a model prediction with a short, rule-guided walk over the CFG.

The transformer and GAT still make the benign/ransomware decision. This agent
does not change that decision. Around each flagged block it keeps same-function
neighbors, callers of crypto functions, and the blocks that run after those
callers return. The story looks for FILE_READ -> CRYPTO, CRYPTO -> FILE_WRITE,
and FILE_READ -> FILE_WRITE. Missing links are reported as partial evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader as GeoDataLoader

from build_graphs import (
    EDGE_CALL,
    EDGE_CFG,
    EDGE_LOOP,
    LINE_RE,
    RETURNS,
    find_read_to_write_paths,
)
from build_pyg_graphs import build_tokenized_cfg
from data_loader import _ensure_channels, behavior_window_starts, pick_behavior_windows
from graph_loader import GRAPH_DIR, load_graph
from token_mapping import BEHAVIOR_MAP, HEAD_BEHAVIOR_NAMES, parse_asm_file, parse_asm_line

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_DIR = PROJECT_ROOT / "processed" / "sequences"
ASM_DIR = PROJECT_ROOT / "data" / "asm"
CHECKPOINT_DIR = PROJECT_ROOT / "processed" / "checkpoints"

TOP_K = 5
SCORE_MIN = 0.5
MAX_PATHS = 3
MAX_DEPTH = 16
LABELS = {0: "Benign", 1: "Ransomware"}
SOURCE_MODEL = "model_seed"
SOURCE_CFG = "CFG_expansion"
SOURCE_CALL = "call_target"
SOURCE_CALLER = "caller_expansion"
SOURCE_CALLEE = "callee_expansion"
SOURCE_CRYPTO_FN = "crypto_function"
SOURCE_CRYPTO_CALLER = "crypto_caller"
SOURCE_CONTINUATION = "caller_continuation"
SOURCE_ORDER = (
    SOURCE_MODEL,
    SOURCE_CRYPTO_CALLER,
    SOURCE_CONTINUATION,
    SOURCE_CRYPTO_FN,
    SOURCE_CALL,
    SOURCE_CALLER,
    SOURCE_CALLEE,
    SOURCE_CFG,
)
PATH_KINDS = (
    ("FILE_READ -> CRYPTO", "has_file_read", "crypto"),
    ("CRYPTO -> FILE_WRITE", "crypto", "has_file_write"),
    ("FILE_READ -> FILE_WRITE", "has_file_read", "has_file_write"),
)


def _device_of(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def _load_sequence(file_id: str) -> torch.Tensor:
    path = SEQUENCE_DIR / f"{file_id}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing sequence: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["sequence"].long()


def _asm_path(file_id: str) -> Path:
    graph_path = GRAPH_DIR / f"{file_id}.pt"
    if graph_path.is_file():
        payload = torch.load(graph_path, map_location="cpu", weights_only=False)
        saved = Path(str(payload.get("asm_path", "")))
        if saved.is_file():
            return saved
    local = ASM_DIR / f"{file_id}.asm"
    if local.is_file():
        return local
    raise FileNotFoundError(f"Missing assembly for {file_id}")


def _block_info(cfg) -> dict[int, dict]:
    info = {}
    for block in cfg.blocks:
        info[block.block_id] = {
            "crypto_count": block.crypto_count,
            "file_count": block.file_count,
            "has_file_read": block.has_file_read,
            "has_file_write": block.has_file_write,
            "start_address": block.start_address,
        }
    return info


def _adjacency(cfg) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    outgoing = {block.block_id: [] for block in cfg.blocks}
    incoming = {block.block_id: [] for block in cfg.blocks}
    for (src, dst), kind in zip(cfg.edges, cfg.edge_types):
        outgoing[src].append((dst, kind))
        incoming[dst].append((src, kind))
    return outgoing, incoming


def _file_order_block_ids(asm_path: Path, cfg) -> list[int] | None:
    """Block id of each instruction in the same order as the saved sequence."""
    address_to_block = {
        instruction.address: block.block_id
        for block in cfg.blocks
        for instruction in block.instructions
    }
    block_ids = []
    for line in asm_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if parse_asm_line(line) is None:
            continue
        match = LINE_RE.match(line)
        if not match:
            return None
        block_id = address_to_block.get(int(match.group("address"), 0))
        if block_id is None:
            return None
        block_ids.append(block_id)
    return block_ids


def _has_role(block_id: int, role: str, info: dict) -> bool:
    if role == "crypto":
        return info[block_id]["crypto_count"] > 0
    return bool(info[block_id][role])


def _remember(sources: dict[int, str], block_id: int, source: str) -> None:
    """Keep the first source. model_seed is written before any expansion."""
    sources.setdefault(block_id, source)


def _intra_neighbors(block_id: int, evidence: dict) -> list[int]:
    """CFG and loop neighbors: the adjacent blocks in the same function."""
    found = []
    for dst, kind in evidence["outgoing"].get(block_id, []):
        if kind in (EDGE_CFG, EDGE_LOOP):
            found.append(dst)
    for src, kind in evidence["incoming"].get(block_id, []):
        if kind in (EDGE_CFG, EDGE_LOOP):
            found.append(src)
    return found


def _intra_region(entry: int, evidence: dict, limit: int = 48) -> list[int]:
    """Blocks reached from ``entry`` without taking a call edge."""
    outgoing = evidence["outgoing"]
    region = []
    queue = [entry]
    visited = {entry}
    while queue and len(region) < limit:
        node = queue.pop(0)
        region.append(node)
        for dst, kind in outgoing.get(node, []):
            if kind not in (EDGE_CFG, EDGE_LOOP) or dst in visited:
                continue
            visited.add(dst)
            queue.append(dst)
    return region


def _ret_blocks(cfg) -> set[int]:
    return {
        block.block_id
        for block in cfg.blocks
        if block.instructions and block.instructions[-1].opcode in RETURNS
    }


def expand_around_seeds(evidence: dict, seeds: list[int]) -> None:
    """Open neighbors of model blocks, then callers of crypto functions.

    A call edge does not come back by itself. Return edges are added from
    each ret in the crypto function to the caller and to the blocks that
    run after that caller.
    """
    sources: dict[int, str] = {}
    info = evidence["block_info"]
    for block_id in seeds:
        _remember(sources, block_id, SOURCE_MODEL)

    call_targets = []
    for seed in list(seeds):
        for dst, kind in evidence["outgoing"].get(seed, []):
            if kind == EDGE_CALL:
                _remember(sources, dst, SOURCE_CALL)
                call_targets.append(dst)
            elif kind in (EDGE_CFG, EDGE_LOOP):
                _remember(sources, dst, SOURCE_CFG)
        for src, kind in evidence["incoming"].get(seed, []):
            if kind == EDGE_CALL:
                _remember(sources, src, SOURCE_CALLER)
            elif kind in (EDGE_CFG, EDGE_LOOP):
                _remember(sources, src, SOURCE_CFG)

    for entry in call_targets:
        for neighbor in _intra_neighbors(entry, evidence):
            _remember(sources, neighbor, SOURCE_CALLEE)

    crypto_entries = [
        block_id
        for block_id in list(sources)
        if info[block_id]["crypto_count"] > 0
    ]
    return_edges: list[tuple[int, int]] = []
    ret_blocks = evidence.get("ret_blocks") or set()
    seen_returns: set[tuple[int, int]] = set()
    for entry in crypto_entries:
        region = _intra_region(entry, evidence)
        for block_id in region:
            _remember(sources, block_id, SOURCE_CRYPTO_FN)
        region_set = set(region)
        rets = [block_id for block_id in region if block_id in ret_blocks]
        callers = []
        for block_id in region_set:
            for src, kind in evidence["incoming"].get(block_id, []):
                if kind == EDGE_CALL and src not in callers:
                    callers.append(src)
        for caller in callers:
            if sources.get(caller) == SOURCE_MODEL:
                continue
            sources[caller] = SOURCE_CRYPTO_CALLER
            continuations = [caller]
            for dst, kind in evidence["outgoing"].get(caller, []):
                if kind in (EDGE_CFG, EDGE_LOOP):
                    _remember(sources, dst, SOURCE_CONTINUATION)
                    continuations.append(dst)
            for ret_block in rets:
                for dest in continuations:
                    key = (ret_block, dest)
                    if key in seen_returns or ret_block == dest:
                        continue
                    seen_returns.add(key)
                    return_edges.append(key)

    evidence["sources"] = sources
    evidence["inspected"] = set(sources)
    evidence["return_edges"] = return_edges


def _edges_inside(evidence: dict) -> dict[int, list[int]]:
    inspected = evidence["inspected"]
    adjacency: dict[int, list[int]] = {}
    for src, edges in evidence["outgoing"].items():
        if src not in inspected:
            continue
        adjacency[src] = [dst for dst, _kind in edges if dst in inspected]
    for src, dst in evidence.get("return_edges") or []:
        if src in inspected and dst in inspected:
            adjacency.setdefault(src, [])
            if dst not in adjacency[src]:
                adjacency[src].append(dst)
    return adjacency


def _paths_for_roles(evidence: dict, start_role: str, goal_role: str) -> list[list[int]]:
    """Shortest paths inside the inspected blocks from one role to another."""
    info = evidence["block_info"]
    inspected = evidence["inspected"]
    starts = sorted(block_id for block_id in inspected if _has_role(block_id, start_role, info))
    goals = {block_id for block_id in inspected if _has_role(block_id, goal_role, info)}
    if not starts or not goals:
        return []

    adjacency = _edges_inside(evidence)
    paths: list[list[int]] = []
    seen_paths: set[tuple[int, ...]] = set()
    for start in starts:
        if len(paths) >= MAX_PATHS:
            break
        if start in goals:
            key = (start,)
            if key not in seen_paths:
                seen_paths.add(key)
                paths.append([start])
                if len(paths) >= MAX_PATHS:
                    break
        queue = [[start]]
        visited = {start}
        while queue and len(paths) < MAX_PATHS:
            path = queue.pop(0)
            node = path[-1]
            if node in goals and len(path) > 1:
                key = tuple(path)
                if key not in seen_paths:
                    seen_paths.add(key)
                    paths.append(path)
                continue
            if len(path) >= MAX_DEPTH:
                continue
            for nxt in adjacency.get(node, []):
                if nxt in visited:
                    continue
                visited.add(nxt)
                queue.append(path + [nxt])
    return paths


def story_paths(evidence: dict) -> dict[str, list[list[int]]]:
    return {
        name: _paths_for_roles(evidence, start_role, goal_role)
        for name, start_role, goal_role in PATH_KINDS
    }


def run_transformer(file_id: str, transformer, sequence: torch.Tensor) -> dict:
    windows, lengths, _targets = pick_behavior_windows(sequence)
    starts = behavior_window_starts(sequence)
    device = _device_of(transformer)
    transformer.eval()
    with torch.no_grad():
        logits, window_logits, behavior_logits = transformer(
            windows.unsqueeze(0).to(device),
            lengths.unsqueeze(0).to(device),
        )
    class_prob = torch.softmax(logits, dim=-1)[0]
    prediction = int(class_prob.argmax())
    window_prob = torch.softmax(window_logits[0], dim=-1)[:, 1]
    behavior_prob = torch.sigmoid(behavior_logits[0])
    rows = []
    for index, start in enumerate(starts):
        length = int(lengths[index])
        if length <= 0:
            continue
        rows.append(
            {
                "window": index,
                "start": int(start),
                "length": length,
                "ransomware_prob": float(window_prob[index]),
                "behavior": {
                    name.lower(): float(behavior_prob[index, column])
                    for column, name in enumerate(HEAD_BEHAVIOR_NAMES)
                },
            }
        )
    return {
        "file_id": file_id,
        "prediction": prediction,
        "confidence": float(class_prob[prediction]),
        "windows": rows,
    }


def get_top_windows(transformer_result: dict) -> list[dict]:
    rows = transformer_result["windows"]
    if not rows:
        return []
    flagged = [
        row
        for row in rows
        if row["ransomware_prob"] >= SCORE_MIN
        or max(row["behavior"].values()) >= SCORE_MIN
    ]
    if not flagged:
        flagged = [max(rows, key=lambda row: row["ransomware_prob"])]
    flagged.sort(key=lambda row: row["ransomware_prob"], reverse=True)
    return flagged[:TOP_K]


def run_gat(file_id: str, gat) -> dict:
    graph_path = GRAPH_DIR / f"{file_id}.pt"
    if not graph_path.is_file():
        raise FileNotFoundError(f"Missing graph: {graph_path}")
    device = _device_of(gat)
    if device.type == "privateuseone":
        gat = gat.cpu()
        device = torch.device("cpu")
    data = next(iter(GeoDataLoader([load_graph(graph_path)], batch_size=1))).to(device)
    gat.eval()
    with torch.no_grad():
        logits, block_logits, behavior_logits = gat(data)
    class_prob = torch.softmax(logits, dim=-1)[0]
    prediction = int(class_prob.argmax())
    malware_prob = torch.softmax(block_logits, dim=-1)[:, 1].detach().cpu()
    behavior_prob = torch.sigmoid(behavior_logits).detach().cpu()
    graph = data.cpu()
    rows = []
    for block_id in range(int(graph.num_nodes)):
        rows.append(
            {
                "block_id": block_id,
                "ransomware_prob": float(malware_prob[block_id]),
                "behavior_score": float(behavior_prob[block_id].max()),
                "crypto_count": int(graph.block_crypto_count[block_id]),
                "file_count": int(graph.block_file_count[block_id]),
                "behavior": {
                    name.lower(): float(behavior_prob[block_id, column])
                    for column, name in enumerate(HEAD_BEHAVIOR_NAMES)
                },
            }
        )
    return {
        "file_id": file_id,
        "prediction": prediction,
        "confidence": float(class_prob[prediction]),
        "blocks": rows,
    }


def get_top_blocks(gat_result: dict) -> list[dict]:
    rows = gat_result["blocks"]
    if not rows:
        return []
    flagged = [
        row
        for row in rows
        if row["behavior_score"] >= SCORE_MIN or row["ransomware_prob"] >= SCORE_MIN
    ]
    if not flagged:
        flagged = [max(rows, key=lambda row: row["behavior_score"])]
    flagged.sort(
        key=lambda row: (row["behavior_score"], row["ransomware_prob"]),
        reverse=True,
    )
    return flagged[:TOP_K]


def find_crypto_blocks(cfg) -> list[int]:
    return [block.block_id for block in cfg.blocks if block.crypto_count > 0]


def find_file_blocks(cfg) -> list[int]:
    return [block.block_id for block in cfg.blocks if block.file_count > 0]


def file_tag_census(cfg) -> dict:
    """Count FILE_READ and FILE_WRITE instructions across the whole file."""
    read_id = BEHAVIOR_MAP["FILE_READ"]
    write_id = BEHAVIOR_MAP["FILE_WRITE"]
    read_insns = 0
    write_insns = 0
    read_blocks = []
    write_blocks = []
    for block in cfg.blocks:
        reads = sum(1 for insn in block.instructions if insn.behavior_id == read_id)
        writes = sum(1 for insn in block.instructions if insn.behavior_id == write_id)
        read_insns += reads
        write_insns += writes
        if reads:
            read_blocks.append(block.block_id)
        if writes:
            write_blocks.append(block.block_id)
    return {
        "read_insns": read_insns,
        "write_insns": write_insns,
        "read_blocks": read_blocks,
        "write_blocks": write_blocks,
    }


def _tagged_blocks(block_ids: list[int], evidence: dict) -> list[int]:
    info = evidence["block_info"]
    return [
        block_id
        for block_id in block_ids
        if info[block_id]["crypto_count"] or info[block_id]["file_count"]
    ]


def _align_sequence(sequence: torch.Tensor, asm_rows: torch.Tensor) -> list[int]:
    """Map each saved-sequence row onto the current assembly.

    An exact match uses the same index. Otherwise allow one inserted assembly
    instruction so a small parser difference does not throw the window away.
    """
    if sequence.shape == asm_rows.shape and torch.equal(sequence, asm_rows):
        return list(range(int(sequence.shape[0])))

    mapping = []
    asm_index = 0
    asm_len = int(asm_rows.shape[0])
    for seq_index in range(int(sequence.shape[0])):
        found = None
        for candidate in range(asm_index, min(asm_len, asm_index + 2)):
            if torch.equal(sequence[seq_index], asm_rows[candidate]):
                found = candidate
                break
        if found is None:
            mapping.append(-1)
        else:
            mapping.append(found)
            asm_index = found + 1
    return mapping


def _attach_window_blocks(
    windows: list[dict],
    asm_path: Path,
    cfg,
    sequence: torch.Tensor,
    evidence: dict,
) -> str:
    order = _file_order_block_ids(asm_path, cfg)
    asm_rows = torch.tensor(parse_asm_file(asm_path), dtype=torch.long)
    if (
        order is None
        or asm_rows.ndim != 2
        or asm_rows.shape[0] != len(order)
        or asm_rows.shape[1] != int(sequence.shape[1])
    ):
        for row in windows:
            row["blocks"] = []
        return (
            "Transformer windows could not be lined up with basic blocks, "
            "so the walk uses the GAT blocks only."
        )

    mapping = _align_sequence(sequence, asm_rows)
    fuzzy = mapping != list(range(int(sequence.shape[0])))
    info = evidence["block_info"]
    for row in windows:
        span = mapping[row["start"] : row["start"] + row["length"]]
        mapped = [index for index in span if index >= 0]
        if len(mapped) < max(1, int(row["length"] * 0.5)):
            row["blocks"] = []
            continue
        tagged = _tagged_blocks(sorted({order[index] for index in mapped}), evidence)
        tagged.sort(
            key=lambda block_id: (
                info[block_id]["crypto_count"] + info[block_id]["file_count"]
            ),
            reverse=True,
        )
        row["blocks"] = tagged

    if fuzzy:
        return (
            "The saved sequence and the current assembly differ by a few "
            "instructions. Windows were matched instruction by instruction."
        )
    return ""


def _ids(block_ids: list[int], limit: int = 12) -> str:
    shown = [f"B{block_id}" for block_id in block_ids[:limit]]
    extra = len(block_ids) - len(shown)
    text = ", ".join(shown) if shown else "(none)"
    if extra > 0:
        text += f", +{extra} more"
    return text


def _partial_lines(evidence: dict) -> list[str]:
    info = evidence["block_info"]
    sources = evidence.get("sources") or {}
    stories = evidence.get("stories") or {}
    inspected = evidence.get("inspected") or set()
    crypto = sorted(block_id for block_id in inspected if info[block_id]["crypto_count"] > 0)
    reads = sorted(block_id for block_id in inspected if info[block_id]["has_file_read"])
    writes = sorted(block_id for block_id in inspected if info[block_id]["has_file_write"])
    callers = sorted(
        block_id for block_id, source in sources.items() if source == SOURCE_CRYPTO_CALLER
    )
    continuations = sorted(
        block_id for block_id, source in sources.items() if source == SOURCE_CONTINUATION
    )
    missing = [name for name, _start, _goal in PATH_KINDS if not stories.get(name)]
    if not missing:
        return []
    lines = ["", "Partial evidence"]
    lines.append(f"  crypto blocks near the model: {_ids(crypto)}")
    lines.append(f"  file-read blocks in that set: {_ids(reads)}")
    lines.append(f"  file-write blocks in that set: {_ids(writes)}")
    lines.append(f"  callers of crypto functions: {_ids(callers)}")
    lines.append(f"  blocks after those callers: {_ids(continuations)}")
    if reads and crypto and "FILE_READ -> CRYPTO" in missing:
        lines.append("  File-read and crypto are both nearby, but no path connects them.")
    if crypto and writes and "CRYPTO -> FILE_WRITE" in missing:
        lines.append("  Crypto and file-write are both nearby, but no return path connects them.")
    if reads and writes and "FILE_READ -> FILE_WRITE" in missing:
        lines.append("  File-read and file-write are both nearby, but no path connects them.")
    if not reads and not writes:
        lines.append("  The inspected blocks have no file-read or file-write tag.")
    return lines


def format_explanation(file_id: str, evidence: dict) -> str:
    transformer = evidence["transformer_prediction"]
    gat = evidence["gat_prediction"]
    info = evidence["block_info"]
    lines = [
        f"File {file_id}",
        "",
        "Predictions",
        f"  {'model':<14}{'label':<14}confidence",
        f"  {'Transformer':<14}{LABELS[transformer['label']]:<14}{transformer['confidence']:.3f}",
        f"  {'GAT':<14}{LABELS[gat['label']]:<14}{gat['confidence']:.3f}",
        "",
        "Transformer windows",
    ]
    if not evidence["transformer_windows"]:
        lines.append("  (none)")
    else:
        header = (
            f"  {'window':<8}{'instructions':<18}{'ransomware':<12}"
            f"{'crypto':<8}{'encrypt':<9}{'decrypt':<9}{'file_read':<11}file_write"
        )
        lines.append(header)
        for row in evidence["transformer_windows"]:
            end = row["start"] + row["length"] - 1
            behavior = row["behavior"]
            lines.append(
                f"  {row['window']:<8}{str(row['start']) + '-' + str(end):<18}"
                f"{row['ransomware_prob']:<12.3f}"
                f"{behavior.get('crypto_opcode', 0):<8.2f}"
                f"{behavior.get('encrypt', 0):<9.2f}"
                f"{behavior.get('decrypt', 0):<9.2f}"
                f"{behavior.get('file_read', 0):<11.2f}"
                f"{behavior.get('file_write', 0):.2f}"
            )
            blocks = row.get("blocks") or []
            if blocks:
                shown = [
                    f"B{block_id} crypto x{info[block_id]['crypto_count']}"
                    for block_id in blocks[:5]
                ]
                extra = len(blocks) - len(shown)
                detail = ", ".join(shown)
                if extra > 0:
                    detail += f", +{extra} more"
                lines.append(f"  {'':<8}{detail}")
            elif evidence.get("alignment_note", "").startswith("Transformer windows could not"):
                lines.append(f"  {'':<8}could not align this window")
    if evidence.get("alignment_note"):
        lines.append(f"  {evidence['alignment_note']}")

    lines.extend(["", "GAT blocks"])
    if not evidence["gat_blocks"]:
        lines.append("  (none)")
    else:
        lines.append(
            f"  {'block':<8}{'ransomware':<12}{'behavior':<10}{'crypto':<8}{'file':<6}"
            f"{'crypto_op':<10}{'encrypt':<9}{'decrypt':<9}{'file_read':<11}file_write"
        )
        for row in evidence["gat_blocks"]:
            behavior = row["behavior"]
            lines.append(
                f"  B{row['block_id']:<7}{row['ransomware_prob']:<12.3f}"
                f"{row['behavior_score']:<10.3f}{row['crypto_count']:<8}"
                f"{row['file_count']:<6}"
                f"{behavior.get('crypto_opcode', 0):<10.2f}"
                f"{behavior.get('encrypt', 0):<9.2f}"
                f"{behavior.get('decrypt', 0):<9.2f}"
                f"{behavior.get('file_read', 0):<11.2f}"
                f"{behavior.get('file_write', 0):.2f}"
            )

    lines.extend(["", "Inspection"])
    sources = evidence.get("sources") or {}
    if not sources:
        lines.append("  No blocks were flagged.")
    for source in SOURCE_ORDER:
        blocks = sorted(block_id for block_id, label in sources.items() if label == source)
        if blocks:
            lines.append(f"  {source:<22}{_ids(blocks)}")

    census = evidence.get("file_tags") or {}
    lines.extend(
        [
            "",
            "Whole file",
            (
                f"  FILE_READ   {census.get('read_insns', 0)} instructions"
                f" in {len(census.get('read_blocks') or [])} blocks"
            ),
            (
                f"  FILE_WRITE  {census.get('write_insns', 0)} instructions"
                f" in {len(census.get('write_blocks') or [])} blocks"
            ),
        ]
    )

    lines.extend(["", "Story"])
    stories = evidence.get("stories") or {}
    any_story = False
    for name, _start_role, _goal_role in PATH_KINDS:
        lines.append(f"  {name}")
        found = stories.get(name) or []
        if not found:
            lines.append("    (none)")
            continue
        any_story = True
        for path in found:
            lines.append("    " + " -> ".join(f"B{block_id}" for block_id in path))
            for block_id in path:
                role = _block_role_text(info[block_id])
                source = sources.get(block_id, "on the path")
                detail = f"{role}, {source}" if role else source
                lines.append(f"      B{block_id}: {detail}")
    lines.extend(_partial_lines(evidence))
    if not any_story and evidence.get("outside_path"):
        lines.append("  A read-to-write path exists elsewhere in the file, outside these blocks.")
    return "\n".join(lines)


def _block_role_text(block: dict) -> str:
    parts = []
    if block["has_file_read"]:
        parts.append("file read")
    if block["has_file_write"]:
        parts.append("file write")
    if block["crypto_count"]:
        parts.append(f"crypto x{block['crypto_count']}")
    return ", ".join(parts)


def analyze_with_agent(file_id: str, transformer, gat, max_steps: int = 3) -> str:
    sequence = _ensure_channels(_load_sequence(file_id))
    asm_path = _asm_path(file_id)
    cfg = build_tokenized_cfg(asm_path)
    outgoing, incoming = _adjacency(cfg)

    transformer_result = run_transformer(file_id, transformer, sequence)
    gat_result = run_gat(file_id, gat)
    windows = get_top_windows(transformer_result)
    gat_blocks = get_top_blocks(gat_result)

    evidence = {
        "block_info": _block_info(cfg),
        "outgoing": outgoing,
        "incoming": incoming,
        "crypto_blocks": find_crypto_blocks(cfg),
        "file_blocks": find_file_blocks(cfg),
        "transformer_windows": windows,
        "gat_blocks": gat_blocks,
        "transformer_prediction": {
            "label": transformer_result["prediction"],
            "confidence": transformer_result["confidence"],
        },
        "gat_prediction": {
            "label": gat_result["prediction"],
            "confidence": gat_result["confidence"],
        },
        "sources": {},
        "stories": {},
        "return_edges": [],
        "ret_blocks": _ret_blocks(cfg),
        "file_tags": file_tag_census(cfg),
        "outside_path": False,
    }
    evidence["alignment_note"] = _attach_window_blocks(
        windows, asm_path, cfg, sequence, evidence
    )
    seeds = [row["block_id"] for row in gat_blocks]
    for row in windows:
        blocks = row.get("blocks") or []
        if any(block_id in seeds for block_id in blocks):
            continue
        if blocks:
            seeds.append(blocks[0])
    evidence["flagged"] = set(seeds)
    if max_steps > 0:
        expand_around_seeds(evidence, seeds)
    else:
        evidence["sources"] = {block_id: SOURCE_MODEL for block_id in seeds}
        evidence["inspected"] = set(seeds)
    evidence["stories"] = story_paths(evidence)
    if not any(evidence["stories"].values()):
        evidence["outside_path"] = bool(find_read_to_write_paths(cfg))
    return format_explanation(file_id, evidence)


def _load_models(device: torch.device):
    from gat_model import BehaviorOpcodeGAT
    from transformer_model import BehaviorOpcodeTransformer

    transformer_path = CHECKPOINT_DIR / "best_transformer_behavior.pt"
    gat_path = CHECKPOINT_DIR / "best_gat_behavior.pt"
    missing = [path.name for path in (transformer_path, gat_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing checkpoints: " + ", ".join(missing) + f" in {CHECKPOINT_DIR}"
        )

    transformer = BehaviorOpcodeTransformer()
    transformer.load_state_dict(torch.load(transformer_path, map_location="cpu", weights_only=True))
    transformer = transformer.to(device).eval()

    gat_device = device
    if device.type == "privateuseone":
        gat_device = torch.device("cpu")
    gat = BehaviorOpcodeGAT()
    gat.load_state_dict(torch.load(gat_path, map_location="cpu", weights_only=True))
    gat = gat.to(gat_device).eval()
    return transformer, gat


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain one file from the behavior models and a CFG walk"
    )
    parser.add_argument("--file-id", required=True)
    parser.add_argument("--max-steps", type=int, default=3)
    args = parser.parse_args()

    from device import get_device

    transformer, gat = _load_models(get_device())
    print(analyze_with_agent(args.file_id, transformer, gat, max_steps=args.max_steps))


if __name__ == "__main__":
    main()
