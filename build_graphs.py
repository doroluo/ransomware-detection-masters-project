#!/usr/bin/env python3
"""
build_graphs.py

Turns the opcode sequences written by extract_opcodes.py into one directed
graph per sample, ready for the GIN in train_gin.py.

Graph construction (an "opcode transition graph"):

    nodes    one per distinct opcode mnemonic in the sample
    edges    u -> v whenever v immediately follows u in the sequence,
             with the number of occurrences as the edge weight
    features per node: how often the opcode occurs and how connected it is

So a 400k-instruction binary collapses to a graph of ~60 nodes that keeps the
local ordering information a bag-of-opcodes model throws away. This is not a
control flow graph: we never resolve branch targets, and the sequence is laid
out in address order rather than execution order. A real CFG would be stronger
but needs recursive traversal disassembly, not the linear sweep we have.

The vocabulary is built from the TRAINING SPLIT ONLY. Letting rare opcodes that
only occur in test define node identities would leak information about the test
set into the model. Opcodes outside the vocabulary collapse into a single <unk>
node, which also keeps obfuscated junk mnemonics from exploding the vocabulary.

Usage:

    python build_graphs.py --index data/opcodes_dataset/index.csv --out data/graphs
    python build_graphs.py --max-vocab 128 --max-instructions 100000
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SPLITS = ("train", "val", "test")
UNK = "<unk>"

# Node feature layout, kept here so train_gin.py can report what it is training on.
NODE_FEATURE_NAMES = [
    "log1p(count)",
    "count/total",
    "log1p(out_degree)",
    "log1p(in_degree)",
    "out_strength/total",
    "in_strength/total",
]


def read_index(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} has no rows")
    if "split" not in rows[0]:
        raise SystemExit(f"{path} has no 'split' column; run split_dataset.py first")
    return rows


def read_sequence(path, max_instructions):
    """Reads one opcode-per-line file into a list of mnemonics."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    opcodes = [line for line in text.split("\n") if line]
    if max_instructions:
        opcodes = opcodes[:max_instructions]
    return opcodes


def count_document_frequency(task):
    """Worker: which opcodes appear in this one sample (for vocabulary building)."""
    path, max_instructions = task
    try:
        return set(read_sequence(path, max_instructions))
    except OSError:
        return set()


def build_vocabulary(train_rows, max_vocab, min_df, max_instructions, workers):
    """
    Keeps opcodes seen in at least --min-df training samples, most frequent
    first, capped at --max-vocab. Index 0 is always <unk>.
    """
    tasks = [(row["opcode_file"], max_instructions) for row in train_rows]
    doc_freq = Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for present in pool.map(count_document_frequency, tasks, chunksize=16):
            doc_freq.update(present)

    kept = [op for op, df in doc_freq.most_common() if df >= min_df]
    if max_vocab:
        kept = kept[: max_vocab - 1]

    vocab = {UNK: 0}
    for opcode in sorted(kept):
        vocab[opcode] = len(vocab)

    dropped = len(doc_freq) - len(kept)
    print(f"Vocabulary: {len(vocab)} opcodes kept from {len(doc_freq)} seen in train "
          f"({dropped} dropped as rare or over the cap)")
    return vocab


def build_graph(task):
    """Worker: one opcode sequence -> one graph, as plain Python containers."""
    row, vocab, max_instructions = task

    try:
        opcodes = read_sequence(row["opcode_file"], max_instructions)
    except OSError as exc:
        return {"status": "skipped", "sha256": row["sha256"], "reason": str(exc)}

    if not opcodes:
        return {"status": "skipped", "sha256": row["sha256"], "reason": "empty sequence"}

    ids = [vocab.get(op, 0) for op in opcodes]

    # Local node indices, in first-appearance order.
    local = {}
    for opcode_id in ids:
        if opcode_id not in local:
            local[opcode_id] = len(local)

    counts = Counter(ids)
    edges = Counter()
    for source, target in zip(ids, ids[1:]):
        edges[(local[source], local[target])] += 1

    n_nodes = len(local)
    total = len(ids)

    out_degree = Counter()
    in_degree = Counter()
    out_strength = Counter()
    in_strength = Counter()
    for (source, target), weight in edges.items():
        out_degree[source] += 1
        in_degree[target] += 1
        out_strength[source] += weight
        in_strength[target] += weight

    # node_ids[i] is the vocabulary id of local node i
    node_ids = [0] * n_nodes
    for opcode_id, index in local.items():
        node_ids[index] = opcode_id

    import math

    features = []
    for index in range(n_nodes):
        opcode_id = node_ids[index]
        count = counts[opcode_id]
        features.append([
            math.log1p(count),
            count / total,
            math.log1p(out_degree[index]),
            math.log1p(in_degree[index]),
            out_strength[index] / total,
            in_strength[index] / total,
        ])

    edge_list = sorted(edges)
    return {
        "status": "ok",
        "sha256": row["sha256"],
        "split": row["split"],
        "label": int(row["label"]),
        "family": row["family"],
        "node_ids": node_ids,
        "node_features": features,
        "edge_index": [[s for s, _ in edge_list], [t for _, t in edge_list]],
        "edge_weight": [float(edges[e]) for e in edge_list],
        "n_instructions": total,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build opcode transition graphs from extracted opcode sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--index", default="data/opcodes_dataset/index.csv")
    parser.add_argument("--out", default="data/graphs",
                        help="directory to write graphs.pt into")
    parser.add_argument("--max-vocab", type=int, default=256,
                        help="cap on vocabulary size, including <unk> (0 = no cap)")
    parser.add_argument("--min-df", type=int, default=3,
                        help="an opcode must appear in this many training samples to be kept")
    parser.add_argument("--max-instructions", type=int, default=0,
                        help="truncate each sequence to this many opcodes (0 = no limit)")
    parser.add_argument("--min-nodes", type=int, default=2,
                        help="drop samples whose graph has fewer nodes than this")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    import torch  # imported late so --help works without torch installed

    index_path = Path(args.index)
    if not index_path.is_file():
        print(f"{index_path} not found. Run extract_opcodes.py and split_dataset.py first.",
              file=sys.stderr)
        return 1

    rows = read_index(index_path)
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    print("Samples per split: " + ", ".join(f"{s}={len(by_split[s])}" for s in SPLITS))

    vocab = build_vocabulary(
        by_split["train"], args.max_vocab, args.min_df, args.max_instructions, args.workers
    )

    tasks = [(row, vocab, args.max_instructions) for row in rows]
    graphs, skipped = [], []
    print(f"Building {len(tasks)} graphs with {args.workers} workers...")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(build_graph, tasks, chunksize=16), start=1):
            if result["status"] != "ok":
                skipped.append(result)
            elif len(result["node_ids"]) < args.min_nodes:
                skipped.append({"sha256": result["sha256"],
                                "reason": f"only {len(result['node_ids'])} node(s)"})
            else:
                graphs.append(result)
            if i % 250 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} built, {len(skipped)} skipped", flush=True)

    if not graphs:
        print("No graphs were built.", file=sys.stderr)
        return 1

    payload = {
        "vocab": vocab,
        "node_feature_names": NODE_FEATURE_NAMES,
        "graphs": [
            {
                "sha256": g["sha256"],
                "split": g["split"],
                "label": g["label"],
                "family": g["family"],
                "node_ids": torch.tensor(g["node_ids"], dtype=torch.long),
                "x": torch.tensor(g["node_features"], dtype=torch.float),
                "edge_index": torch.tensor(g["edge_index"], dtype=torch.long),
                "edge_weight": torch.tensor(g["edge_weight"], dtype=torch.float),
            }
            for g in graphs
        ],
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "graphs.pt"
    torch.save(payload, out_path)

    nodes = [len(g["node_ids"]) for g in graphs]
    edges = [len(g["edge_weight"]) for g in graphs]
    by_split_count = Counter(g["split"] for g in graphs)
    by_split_label = Counter((g["split"], g["label"]) for g in graphs)

    print("\n=== Graph summary ===")
    print(f"graphs         : {len(graphs)}")
    print(f"skipped        : {len(skipped)}")
    print(f"vocabulary     : {len(vocab)}")
    print(f"nodes  avg/max : {sum(nodes)/len(nodes):.1f} / {max(nodes)}")
    print(f"edges  avg/max : {sum(edges)/len(edges):.1f} / {max(edges)}")
    print("\nsplit      graphs   goodware  ransomware")
    for split in SPLITS:
        print(f"{split:<10} {by_split_count[split]:<8} {by_split_label[(split, 0)]:<9} "
              f"{by_split_label[(split, 1)]}")
    if skipped:
        print("\nskip reasons:")
        for reason, count in Counter(s["reason"] for s in skipped).most_common(5):
            print(f"  {count:>4}  {reason}")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
