"""Build CodeBERT basic-block graphs from the IDA Pro CFG exports.

Reads the recommended train and test rows in:

    data/Combined_Goodware/splits/{train,test}.csv
    data/Combined_Ransomware/splits/{train,test}.csv

Each row points at a gzip JSONL export (schema ida-cfg-research/2.1).
Basic blocks are embedded with microsoft/codebert-base and saved as
PyTorch Geometric graphs:

    processed_graphs/{train,test}/<sha256>.pt
"""

import argparse
import csv
import gc
import gzip
import json
import math
import time
from pathlib import Path

import torch
from torch_geometric.data import Data, Dataset
from transformers import AutoModel, AutoTokenizer


SCHEMA = "ida-cfg-research/2.1"
COLLECTIONS = ("Combined_Goodware", "Combined_Ransomware")
LM_NAME = "microsoft/codebert-base"

# Closed opcode set plus one "other" bin. Counts are divided by the block length.
MNEMONICS = (
    "mov", "lea", "push", "pop", "xchg",
    "add", "sub", "inc", "dec", "neg", "cmp", "test",
    "and", "or", "xor", "not", "shl", "shr", "rol", "ror", "sar",
    "imul", "mul", "idiv", "div",
    "call", "ret", "jmp", "jz", "jnz", "je", "jne", "ja", "jb", "jg", "jl",
    "nop", "syscall", "int", "cpuid", "rdtsc",
    "pxor", "aesenc", "aesdec",
)
MNEMONIC_INDEX = {name: index for index, name in enumerate(MNEMONICS)}
API_GROUPS = (
    ("crypto", ("crypt", "bcrypt", "cipher")),
    ("file", ("createfile", "readfile", "writefile", "deletefile", "copyfile",
              "movefile", "findfirstfile", "findnextfile", "getfilesize", "setfilepointer")),
    ("registry", ("regopen", "regset", "regcreate", "regquery", "regdelete",
                  "ntopenkey", "ntsetvalue")),
    ("process", ("createprocess", "virtualalloc", "virtualprotect", "writeprocessmemory",
                 "openprocess", "createremotethread", "ntallocate", "ntwritevirtual", "ntprotect")),
    ("network", ("internetopen", "internetconnect", "httpsend", "socket",
                 "wsastartup", "urlmon", "winhttp")),
)
STRUCTURAL_FEATURES = (
    "log1p_instructions", "log1p_calls", "log1p_import_refs",
    "external", "log1p_in_degree", "log1p_out_degree",
)


def gnn_feature_names():
    names = [f"mn_{name}" for name in MNEMONICS]
    names.append("mn_other")
    names.extend(f"api_{name}" for name, _keys in API_GROUPS)
    names.append("api_other")
    names.extend(STRUCTURAL_FEATURES)
    return names


def _collection_roots(data_root: Path):
    roots = []
    for name in COLLECTIONS:
        root = data_root / name
        if not (root / "splits").is_dir():
            raise FileNotFoundError(f"Missing IDA bundle: {root}")
        roots.append(root)
    return roots


def load_split_rows(data_root: Path, split: str):
    """Recommended samples for one split, stable order, both classes."""
    rows = []
    for root in _collection_roots(data_root):
        manifest = root / "splits" / f"{split}.csv"
        with manifest.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("cfg_status") not in ("", "recovered"):
                    continue
                if row.get("eligible_for_gnn") not in ("", "1"):
                    continue
                cfg_path = (root / row["cfg_path"]).resolve()
                if root.resolve() not in cfg_path.parents:
                    raise ValueError(f"cfg path escapes bundle: {row['cfg_path']}")
                row = dict(row)
                row["label"] = int(row["label"])
                row["cfg_file"] = cfg_path
                row["collection_root"] = str(root)
                rows.append(row)
    rows.sort(key=lambda row: (row["label"], row["sample_id"]))
    return rows


def count_dataset_graphs(data_root: str = "data"):
    """Print how many recommended CFG exports each split contains."""
    data_root = Path(data_root)
    print("=== IDA CFG exports (recommended splits) ===")
    total = 0
    for split in ("train", "test"):
        rows = load_split_rows(data_root, split)
        goodware = sum(row["label"] == 0 for row in rows)
        ransomware = sum(row["label"] == 1 for row in rows)
        print(f"\n[{split.upper()}]")
        print(f"  • goodware    : {goodware:>5}")
        print(f"  • ransomware  : {ransomware:>5}")
        print(f"  Subtotal      : {len(rows):>5}")
        total += len(rows)
    print(f"\nTotal recommended graphs : {total}\n")
    return total


def _import_name(imports, index):
    if not isinstance(index, int) or not 0 <= index < len(imports):
        return None
    symbol = imports[index]
    name = symbol.get("name")
    if name:
        return name
    ordinal = symbol.get("ordinal")
    if ordinal is None:
        return None
    dll = symbol.get("dll") or "ord"
    return f"{dll}#{ordinal}"


def _operand_text(operand):
    if not isinstance(operand, dict):
        return ""
    category = operand.get("category") or ""
    register = operand.get("register") or ""
    immediate = operand.get("immediate")
    target = operand.get("target_rva")
    displacement = operand.get("displacement")

    if category == "register":
        return register
    if category == "immediate":
        return "" if immediate is None else str(immediate)
    if category == "code_target":
        if isinstance(target, int):
            return hex(target)
        return "" if target is None else str(target)
    if category in ("displacement", "phrase", "memory"):
        if register and displacement is not None:
            return f"[{register}+{displacement}]"
        if register:
            return f"[{register}]"
        if displacement is not None:
            return f"[{displacement}]"
        return "[mem]"
    if register:
        return register
    if immediate is not None:
        return str(immediate)
    if isinstance(target, int):
        return hex(target)
    return ""


def _instruction_text(instruction, imports):
    mnemonic = (instruction.get("mnemonic") or "nop").lower()
    apis = []
    for index in instruction.get("import_ids") or []:
        name = _import_name(imports, index)
        if name:
            apis.append(name)
    if apis and instruction.get("is_call"):
        return f"{mnemonic} {', '.join(apis)}"
    operands = []
    for operand in instruction.get("operands") or []:
        text = _operand_text(operand)
        if text:
            operands.append(text)
    if apis and not operands:
        operands = apis
    if operands:
        return f"{mnemonic} {', '.join(operands)}"
    return mnemonic


def _block_score(block):
    score = 0
    for instruction in block.get("instructions") or []:
        if instruction.get("import_ids"):
            score += 2
        elif instruction.get("is_call"):
            score += 1
    return score


def _read_header(handle, sample_id):
    header = json.loads(handle.readline())
    if header.get("record") != "header":
        raise ValueError("export does not start with a header")
    if header.get("schema") != SCHEMA:
        raise ValueError(f"unsupported schema: {header.get('schema')}")
    if header.get("sha256") != sample_id:
        raise ValueError("sha256 does not match the sample id")
    return header.get("pe", {}).get("imports") or []


def _open_cfg(file_path):
    return gzip.open(file_path, "rt", encoding="utf-8")


def parse_cfg_export(file_path: Path, sample_id: str, max_blocks: int):
    """Return block texts, CFG edges, function ids, and export counts.

    The file is streamed twice so a large export is not held in memory.
    The block budget keeps API-referencing blocks first, then fills the
    rest in file order. Library functions are used only when the sample
    has no other code.
    """
    summaries = []
    num_blocks_total = 0
    num_functions = 0
    footer = None
    with _open_cfg(file_path) as handle:
        _read_header(handle, sample_id)
        order = 0
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record.get("record")
            if kind == "footer":
                footer = record
                continue
            if kind != "function":
                continue
            num_functions += 1
            blocks = record.get("blocks") or []
            num_blocks_total += len(blocks)
            if record.get("thunk") and len(blocks) <= 1:
                continue
            block_meta = []
            for block in blocks:
                block_meta.append((block.get("id"), order, _block_score(block)))
                order += 1
            summaries.append((
                record.get("id"),
                bool(record.get("library")),
                block_meta,
                record.get("edges") or [],
            ))
    if not footer or not footer.get("complete"):
        raise ValueError("incomplete export")

    def ranked_from(skip_library):
        pool = []
        for func_id, library, block_meta, _edges in summaries:
            if skip_library and library:
                continue
            pool.extend((func_id, block_id, block_order, score)
                        for block_id, block_order, score in block_meta)
        pool.sort(key=lambda item: (-item[3], item[2]))
        return pool[:max_blocks]

    ranked = ranked_from(skip_library=True)
    if not ranked:
        ranked = ranked_from(skip_library=False)
    ranked.sort(key=lambda item: item[2])
    keep = {(func_id, block_id) for func_id, block_id, _order, _score in ranked}

    node_of = {}
    block_texts = [None] * len(ranked)
    func_ids = [None] * len(ranked)
    func_index = {}
    for position, (func_id, _block_id, _order, _score) in enumerate(ranked):
        if func_id not in func_index:
            func_index[func_id] = len(func_index)
        node_of[(func_id, _block_id)] = position
        func_ids[position] = func_index[func_id]

    if keep:
        with _open_cfg(file_path) as handle:
            imports = _read_header(handle, sample_id)
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("record") != "function":
                    continue
                func_id = record.get("id")
                for block in record.get("blocks") or []:
                    position = node_of.get((func_id, block.get("id")))
                    if position is None:
                        continue
                    instructions = [
                        _instruction_text(instruction, imports)
                        for instruction in (block.get("instructions") or [])
                    ]
                    block_texts[position] = " ; ".join(instructions) if instructions else "nop"

    if not block_texts:
        block_texts = ["nop"]
        func_ids = [0]
    else:
        block_texts = [text if text is not None else "nop" for text in block_texts]

    sources, targets = [], []
    for func_id, _library, _block_meta, edges in summaries:
        for edge in edges:
            if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                continue
            src = node_of.get((func_id, edge[0]))
            dst = node_of.get((func_id, edge[1]))
            if src is None or dst is None:
                continue
            sources.append(src)
            targets.append(dst)

    return {
        "block_texts": block_texts,
        "sources": sources,
        "targets": targets,
        "func_ids": func_ids,
        "num_blocks_total": num_blocks_total,
        "num_functions": num_functions,
    }


def _api_group(name):
    token = name.lower().split("!")[-1]
    for index, (_group, keys) in enumerate(API_GROUPS):
        if any(key in token for key in keys):
            return index
    return len(API_GROUPS)


def _function_summaries(file_path, sample_id):
    """One streaming pass: per-function block scores and edges, no instruction text."""
    summaries = []
    num_blocks_total = 0
    num_functions = 0
    footer = None
    with _open_cfg(file_path) as handle:
        _read_header(handle, sample_id)
        order = 0
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record.get("record")
            if kind == "footer":
                footer = record
                continue
            if kind != "function":
                continue
            num_functions += 1
            blocks = record.get("blocks") or []
            num_blocks_total += len(blocks)
            if record.get("thunk") and len(blocks) <= 1:
                continue
            block_meta = []
            score = 0
            for block in blocks:
                block_score = _block_score(block)
                block_meta.append((block.get("id"), order, block_score))
                score += block_score
                order += 1
            summaries.append({
                "id": record.get("id"),
                "library": bool(record.get("library")),
                "score": score,
                "blocks": block_meta,
                "edges": record.get("edges") or [],
            })
    if not footer or not footer.get("complete"):
        raise ValueError("incomplete export")
    return summaries, num_blocks_total, num_functions


def _select_functions(summaries, max_blocks):
    """Keep whole functions that reference imports, so CFG edges stay inside the graph."""
    def pool(skip_library):
        chosen = [item for item in summaries if not (skip_library and item["library"])]
        chosen.sort(key=lambda item: (-item["score"], item["blocks"][0][1] if item["blocks"] else 0))
        return chosen

    functions = pool(skip_library=True) or pool(skip_library=False)
    kept = []
    used = 0
    for function in functions:
        remaining = max_blocks - used
        if remaining <= 0:
            break
        blocks = function["blocks"]
        if not blocks:
            continue
        if len(blocks) > remaining and used > 0:
            continue
        take = blocks if len(blocks) <= remaining else sorted(blocks, key=lambda item: (-item[2], item[1]))[:remaining]
        take = sorted(take, key=lambda item: item[1])
        kept.append((function, take))
        used += len(take)
        if len(blocks) > remaining:
            break
    return kept


def gnn_graph_from_export(row, max_blocks):
    """Compact block features and the intra-function CFG. No language model."""
    summaries, num_blocks_total, _num_functions = _function_summaries(row["cfg_file"], row["sample_id"])
    selected = _select_functions(summaries, max_blocks)
    node_of = {}
    func_ids = []
    func_index = {}
    for function, blocks in selected:
        func_key = function["id"]
        if func_key not in func_index:
            func_index[func_key] = len(func_index)
        for block_id, _order, _score in blocks:
            node_of[(func_key, block_id)] = len(func_ids)
            func_ids.append(func_index[func_key])

    sources, targets = [], []
    incoming = [0] * len(func_ids)
    outgoing = [0] * len(func_ids)
    for function, _blocks in selected:
        func_key = function["id"]
        for edge in function["edges"]:
            if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                continue
            src = node_of.get((func_key, edge[0]))
            dst = node_of.get((func_key, edge[1]))
            if src is None or dst is None:
                continue
            sources.append(src)
            targets.append(dst)
            outgoing[src] += 1
            incoming[dst] += 1

    width = len(gnn_feature_names())
    features = [[0.0] * width for _ in func_ids]
    if node_of:
        with _open_cfg(row["cfg_file"]) as handle:
            imports = _read_header(handle, row["sample_id"])
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("record") != "function":
                    continue
                func_key = record.get("id")
                for block in record.get("blocks") or []:
                    position = node_of.get((func_key, block.get("id")))
                    if position is None:
                        continue
                    features[position] = _block_feature_vector(
                        block, imports, incoming[position], outgoing[position]
                    )

    if not features:
        features = [[0.0] * width]
        func_ids = [0]
    feature_tensor = torch.tensor(features, dtype=torch.float32)
    if sources:
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    graph = Data(
        x=feature_tensor,
        edge_index=edge_index,
        edge_index_cfg=edge_index,
        func_batch=torch.tensor(func_ids, dtype=torch.long),
        y=torch.tensor([row["label"]], dtype=torch.long),
    )
    graph.sample_id = row["sample_id"]
    graph.family = row.get("family") or ""
    graph.architecture = row.get("architecture") or ""
    graph.num_blocks_total = int(num_blocks_total)
    return graph


def _block_feature_vector(block, imports, in_degree, out_degree):
    mnemonic_counts = [0] * (len(MNEMONICS) + 1)
    api_counts = [0] * (len(API_GROUPS) + 1)
    n_instructions = 0
    n_calls = 0
    n_imports = 0
    for instruction in block.get("instructions") or []:
        n_instructions += 1
        mnemonic = (instruction.get("mnemonic") or "").lower()
        mnemonic_counts[MNEMONIC_INDEX.get(mnemonic, len(MNEMONICS))] += 1
        if instruction.get("is_call"):
            n_calls += 1
        for index in instruction.get("import_ids") or []:
            name = _import_name(imports, index)
            if not name:
                continue
            n_imports += 1
            api_counts[_api_group(name)] += 1
    scale = max(n_instructions, 1)
    values = [count / scale for count in mnemonic_counts]
    values.extend(math.log1p(count) for count in api_counts)
    values.extend((
        math.log1p(n_instructions),
        math.log1p(n_calls),
        math.log1p(n_imports),
        float(bool(block.get("external"))),
        math.log1p(in_degree),
        math.log1p(out_degree),
    ))
    return values


def _embed_blocks(block_texts, tokenizer, lm_model, device, batch_size):
    if not block_texts:
        hidden = lm_model.config.hidden_size
        return torch.zeros((1, hidden))
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(block_texts), batch_size):
            batch = block_texts[start:start + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            hidden = lm_model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pieces.append(pooled.cpu())
            del inputs, hidden, mask, pooled
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(pieces, dim=0)


def graph_from_export(row, tokenizer, lm_model, device, max_blocks, batch_size):
    parsed = parse_cfg_export(row["cfg_file"], row["sample_id"], max_blocks)
    features = _embed_blocks(parsed["block_texts"], tokenizer, lm_model, device, batch_size)
    if parsed["sources"]:
        edge_index = torch.tensor([parsed["sources"], parsed["targets"]], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    graph = Data(
        x=features,
        edge_index=edge_index,
        edge_index_cfg=edge_index,
        func_batch=torch.tensor(parsed["func_ids"], dtype=torch.long),
        y=torch.tensor([row["label"]], dtype=torch.long),
    )
    graph.sample_id = row["sample_id"]
    graph.family = row.get("family") or ""
    graph.architecture = row.get("architecture") or ""
    graph.num_blocks_total = int(parsed["num_blocks_total"])
    return graph


class MendeleyCFGDataset(Dataset):
    """Recommended-split loader for the combined IDA CFG bundles."""

    def __init__(
        self,
        root_dir: str = "data",
        split: str = "train",
        max_blocks_per_binary: int = 500,
        lm_model_name: str = LM_NAME,
        device: str | None = None,
        limit: int | None = None,
    ):
        super().__init__()
        self.split = split
        self.max_blocks = max_blocks_per_binary
        self.lm_model_name = lm_model_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.rows = load_split_rows(Path(root_dir), split)
        if limit is not None:
            self.rows = self.rows[:limit]
        self._tokenizer = None
        self._lm_model = None

    def _model(self):
        if self._lm_model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.lm_model_name)
            self._lm_model = AutoModel.from_pretrained(self.lm_model_name).to(self.device)
            self._lm_model.eval()
        return self._tokenizer, self._lm_model

    def len(self):
        return len(self.rows)

    def get(self, idx):
        tokenizer, lm_model = self._model()
        return graph_from_export(
            self.rows[idx], tokenizer, lm_model, self.device, self.max_blocks, batch_size=16
        )


def _index_path(output_dir: Path):
    return output_dir / "index.csv"


def _read_index(path: Path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def _write_index(path: Path, rows):
    fieldnames = [
        "split", "sample_id", "label", "class_name", "family", "architecture",
        "num_blocks_kept", "num_blocks_total", "num_edges", "path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_id in sorted(rows):
            writer.writerow(rows[sample_id])


def preprocess_and_save_dataset(
    data_root: str = "data",
    output_dir: str = "processed_graphs",
    max_blocks_per_binary: int = 500,
    limit: int | None = None,
    device: str | None = None,
    lm_model_name: str = LM_NAME,
    overwrite: bool = False,
    batch_size: int = 16,
):
    """Embed every recommended sample and save one graph per SHA-256."""
    output_dir = Path(output_dir)
    index = _read_index(_index_path(output_dir))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    tokenizer = None
    lm_model = None

    for split in ("train", "test"):
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        rows = load_split_rows(Path(data_root), split)
        if limit is not None:
            rows = rows[:limit]
        print(f"\n--> Pre-processing '{split}' ({len(rows)} recommended samples)...")
        saved = skipped = failed = 0
        for index_in_split, row in enumerate(rows, start=1):
            save_path = split_dir / f"{row['sample_id']}.pt"
            if save_path.exists() and not overwrite:
                skipped += 1
                if row["sample_id"] not in index:
                    index[row["sample_id"]] = {
                        "split": split,
                        "sample_id": row["sample_id"],
                        "label": row["label"],
                        "class_name": row.get("class_name", ""),
                        "family": row.get("family", ""),
                        "architecture": row.get("architecture", ""),
                        "num_blocks_kept": "",
                        "num_blocks_total": row.get("num_blocks", ""),
                        "num_edges": "",
                        "path": str(save_path),
                    }
                continue
            try:
                if lm_model is None:
                    print(f"    Loading {lm_model_name} on {device}...")
                    tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
                    lm_model = AutoModel.from_pretrained(lm_model_name).to(device)
                    lm_model.eval()
                started = time.time()
                graph = graph_from_export(
                    row, tokenizer, lm_model, device, max_blocks_per_binary, batch_size
                )
                torch.save(graph, save_path)
                index[row["sample_id"]] = {
                    "split": split,
                    "sample_id": row["sample_id"],
                    "label": row["label"],
                    "class_name": row.get("class_name", ""),
                    "family": row.get("family", ""),
                    "architecture": row.get("architecture", ""),
                    "num_blocks_kept": int(graph.num_nodes),
                    "num_blocks_total": int(graph.num_blocks_total),
                    "num_edges": int(graph.edge_index.size(1)),
                    "path": str(save_path),
                }
                saved += 1
                elapsed = time.time() - started
                print(
                    f"    [{index_in_split}/{len(rows)}] {row['sample_id'][:12]} "
                    f"blocks {graph.num_nodes}/{graph.num_blocks_total} "
                    f"edges {graph.edge_index.size(1)} ({elapsed:.1f}s)"
                )
                del graph
                gc.collect()
            except Exception as exc:
                failed += 1
                print(f"    [{index_in_split}/{len(rows)}] FAILED {row['sample_id']}: {exc}")
        _write_index(_index_path(output_dir), index)
        print(f"✓ {split}: {saved} saved, {skipped} already on disk, {failed} failed.")


def build_gnn_dataset(
    data_root: str = "data",
    output_dir: str = "processed_graphs/gnn",
    max_blocks_per_binary: int = 128,
    limit: int | None = None,
    overwrite: bool = False,
):
    """Save one compact CFG graph per recommended sample for the GNN."""
    output_dir = Path(output_dir)
    index = _read_index(_index_path(output_dir))
    for split in ("train", "test"):
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        rows = load_split_rows(Path(data_root), split)
        if limit is not None:
            rows = rows[:limit]
        print(f"\n--> GNN graphs '{split}' ({len(rows)} recommended samples)...")
        saved = skipped = failed = 0
        started_split = time.time()
        for index_in_split, row in enumerate(rows, start=1):
            save_path = split_dir / f"{row['sample_id']}.pt"
            if save_path.exists() and not overwrite:
                skipped += 1
                continue
            try:
                graph = gnn_graph_from_export(row, max_blocks_per_binary)
                torch.save(graph, save_path)
                index[row["sample_id"]] = {
                    "split": split,
                    "sample_id": row["sample_id"],
                    "label": row["label"],
                    "class_name": row.get("class_name", ""),
                    "family": row.get("family", ""),
                    "architecture": row.get("architecture", ""),
                    "num_blocks_kept": int(graph.num_nodes),
                    "num_blocks_total": int(graph.num_blocks_total),
                    "num_edges": int(graph.edge_index.size(1)),
                    "path": str(save_path),
                }
                saved += 1
                del graph
            except Exception as exc:
                failed += 1
                print(f"    [{index_in_split}/{len(rows)}] FAILED {row['sample_id']}: {exc}")
            if index_in_split % 100 == 0 or index_in_split == len(rows):
                print(
                    f"    [{index_in_split}/{len(rows)}] saved {saved}, "
                    f"skipped {skipped}, failed {failed} ({time.time() - started_split:.0f}s)"
                )
        _write_index(_index_path(output_dir), index)
        print(f"✓ {split}: {saved} saved, {skipped} already on disk, {failed} failed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Samples per split, for a short run.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default=LM_NAME)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument(
        "--gnn",
        action="store_true",
        help="Build compact opcode/API CFG graphs for train_cfg.py instead of CodeBERT embeddings.",
    )
    args = parser.parse_args()

    count_dataset_graphs(args.data_root)
    if args.count_only:
        return
    if args.gnn:
        build_gnn_dataset(
            data_root=args.data_root,
            output_dir=args.output_dir or "processed_graphs/gnn",
            max_blocks_per_binary=128 if args.max_blocks is None else args.max_blocks,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        return
    preprocess_and_save_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir or "processed_graphs",
        max_blocks_per_binary=500 if args.max_blocks is None else args.max_blocks,
        limit=args.limit,
        device=args.device,
        lm_model_name=args.model,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
