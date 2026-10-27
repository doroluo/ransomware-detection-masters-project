"""CFG with feature counting + GAT.

Each node is one basic block of the IDA control-flow graph. Its vector is a
count of opcodes, API groups, and block structure, not a learned embedding.
``BlockGAT`` attends over the directed CFG edges and classifies the binary.

Reads either layout:

* ``processed_graphs/gnn/{train,test}/<sha256>.pt`` from ``construct_blocks.py``
  (opcode mix, API-group counts, size, and degree)
* a flat directory of ``preprocess_full.py`` subgraphs, using the handcrafted
  block-feature counts (``--feature-mode features``)

The flat directory stores the train/test flag inside each file. This script
holds out 15% of the training graphs for early stopping, then scores the
recommended test split. Reports ransomware precision, recall, and F1.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv

PREPROCESS_DIR = Path(__file__).resolve().parent / "CT_GAT" / "preprocessing"
sys.path.insert(0, str(PREPROCESS_DIR))
from graph_full_loader import load_dir, split_graphs  # noqa: E402


def _is_sample_graph(path: Path):
    return path.suffix == ".pt" and len(path.stem) == 64 and all(
        char in "0123456789abcdef" for char in path.stem
    )


def _is_preprocess_dir(root: Path) -> bool:
    """True when ``root`` holds ``preprocess_full.py`` dict payloads, not PyG graphs."""
    sample = next((path for path in sorted(root.glob("*.pt")) if _is_sample_graph(path)), None)
    if sample is None:
        return False
    payload = torch.load(sample, map_location="cpu", weights_only=False)
    return isinstance(payload, dict) and "block_features" in payload


def load_subgraphs(root: Path, feature_mode: str, scale: bool, val_fraction: float, seed: int):
    """Load one flat ``preprocess_full.py`` directory and split it by the saved flag."""
    graphs = load_dir(root, feature_mode=feature_mode, scale=scale)
    if not graphs:
        raise SystemExit(
            f"No subgraph .pt files in {root}. Run preprocess_full.py first."
        )
    return split_graphs(graphs, val_fraction, seed)


def load_split(root: Path, split: str):
    folder = root / split
    graphs = []
    if not folder.is_dir():
        return graphs
    for path in sorted(folder.glob("*.pt")):
        if not _is_sample_graph(path):
            continue
        graph = torch.load(path, weights_only=False)
        graph.sample_id = getattr(graph, "sample_id", path.stem)
        graphs.append(graph)
    return graphs


def stratified_holdout(graphs, fraction, seed):
    by_label = {0: [], 1: []}
    for graph in graphs:
        by_label[int(graph.y.item())].append(graph)
    rng = random.Random(seed)
    train, val = [], []
    for label, group in by_label.items():
        rng.shuffle(group)
        hold = max(1, int(round(len(group) * fraction))) if len(group) > 1 else 0
        val.extend(group[:hold])
        train.extend(group[hold:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class BlockGAT(nn.Module):
    """GAT over a CFG whose nodes are counted block features.

    ``edge_dim`` is set when the graph carries ``edge_attr``: the relation
    one-hot plus the back-edge bit, on both the counted ``construct_blocks.py``
    graphs and ``preprocess_full.py`` subgraphs. CodeBERT graphs have no edge
    attributes and attend on connectivity alone.
    """

    def __init__(self, in_dim, hidden=128, heads=4, dropout=0.2, edge_dim=None):
        super().__init__()
        self.edge_dim = edge_dim
        edge_kwargs = {"edge_dim": edge_dim} if edge_dim else {}
        self.project = nn.Linear(in_dim, hidden)
        self.conv1 = GATConv(hidden, hidden, heads=heads, concat=False, dropout=dropout, **edge_kwargs)
        self.conv2 = GATConv(hidden, hidden, heads=1, concat=False, dropout=dropout, **edge_kwargs)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 2)

    def _edges(self, data, x):
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        if edge_index.numel() == 0:
            nodes = torch.arange(x.size(0), device=x.device)
            edge_index = torch.stack([nodes, nodes], dim=0)
            if self.edge_dim:
                edge_attr = torch.zeros((x.size(0), self.edge_dim), dtype=x.dtype, device=x.device)
            else:
                edge_attr = None
        elif self.edge_dim:
            edge_attr = edge_attr.to(dtype=x.dtype)
        else:
            edge_attr = None
        return edge_index, edge_attr

    def forward(self, data):
        x = self.dropout(F.relu(self.project(data.x)))
        edge_index, edge_attr = self._edges(data, x)
        x = self.norm1(x + self.dropout(F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))))
        x = self.norm2(x + self.dropout(self.conv2(x, edge_index, edge_attr=edge_attr)))
        pooled = torch.zeros(data.num_graphs, x.size(1), device=x.device)
        pooled.index_add_(0, data.batch, x)
        counts = torch.bincount(data.batch, minlength=data.num_graphs).clamp(min=1).unsqueeze(1)
        return self.classifier(pooled / counts)


def _batches(graphs, batch_size, rng):
    order = list(graphs)
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield Batch.from_data_list(order[start:start + batch_size])


def ransomware_scores(logits, labels):
    preds = logits.argmax(dim=1)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = float((preds == labels).float().mean()) if labels.numel() else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": int(labels.numel()),
    }


@torch.no_grad()
def predict_proba(model, graphs, device, batch_size):
    model.eval()
    probs, labels = [], []
    for start in range(0, len(graphs), batch_size):
        batch = Batch.from_data_list(graphs[start:start + batch_size]).to(device)
        probs.append(model(batch).softmax(dim=1)[:, 1].cpu())
        labels.append(batch.y.view(-1).cpu())
    if not probs:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(probs), torch.cat(labels)


def metrics_at(probs, labels, threshold):
    preds = probs >= threshold
    positive = labels == 1
    tp = int((preds & positive).sum())
    fp = int((preds & ~positive).sum())
    fn = int((~preds & positive).sum())
    tn = int((~preds & ~positive).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = max(int(labels.numel()), 1)
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": int(labels.numel()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def best_threshold(probs, labels):
    chosen = metrics_at(probs, labels, 0.5)
    for step in range(5, 96):
        candidate = metrics_at(probs, labels, step / 100)
        if candidate["f1"] > chosen["f1"]:
            chosen = candidate
    return chosen


@torch.no_grad()
def evaluate(model, graphs, device, batch_size):
    model.eval()
    logits, labels = [], []
    for start in range(0, len(graphs), batch_size):
        batch = Batch.from_data_list(graphs[start:start + batch_size]).to(device)
        logits.append(model(batch).cpu())
        labels.append(batch.y.view(-1).cpu())
    if not logits:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    return ransomware_scores(torch.cat(logits), torch.cat(labels))


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(args.graphs)
    if _is_preprocess_dir(root):
        train_graphs, val_graphs, test = load_subgraphs(
            root, args.feature_mode, not args.raw_features, args.val_fraction, args.seed
        )
        source = "preprocess_full subgraphs"
    else:
        graphs = load_split(root, "train")
        test = load_split(root, "test")
        if len(graphs) < 4:
            raise SystemExit(
                f"Need at least 4 training graphs in {root / 'train'}, found {len(graphs)}. "
                "Run construct_blocks.py, or pass a preprocess_full.py output directory."
            )
        train_graphs, val_graphs = stratified_holdout(graphs, args.val_fraction, args.seed)
        source = "construct_blocks graphs"
    if len(train_graphs) < 4:
        raise SystemExit(
            f"Need at least 4 training graphs in {root}, found {len(train_graphs)}."
        )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    probe = train_graphs[0]
    in_dim = int(probe.x.size(1))
    edge_attr = getattr(probe, "edge_attr", None)
    edge_dim = int(edge_attr.size(1)) if edge_attr is not None and edge_attr.dim() == 2 else None
    model = BlockGAT(
        in_dim=in_dim, hidden=args.hidden, dropout=args.dropout, edge_dim=edge_dim
    ).to(device)
    labels = torch.tensor([int(graph.y.item()) for graph in train_graphs])
    counts = torch.bincount(labels, minlength=2).float().clamp(min=1)
    weights = (counts.sum() / counts).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = random.Random(args.seed)

    best_f1 = -1.0
    best_state = None
    stale = 0
    print(
        f"{source}: train {len(train_graphs)}  val {len(val_graphs)}  test {len(test)}  "
        f"features {in_dim}  edge_dim {edge_dim}  device {device}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in _batches(train_graphs, args.batch_size, rng):
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * batch.num_graphs
            seen += batch.num_graphs
        val_metrics = evaluate(model, val_graphs, device, args.batch_size)
        print(
            f"epoch {epoch:02d}  loss {total_loss / max(seen, 1):.4f}  "
            f"val_f1 {val_metrics['f1']:.3f}  val_acc {val_metrics['accuracy']:.3f}"
        )
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stop at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_probs, val_labels = predict_proba(model, val_graphs, device, args.batch_size)
    val_at = best_threshold(val_probs, val_labels)
    test_metrics = None
    if test:
        test_probs, test_labels = predict_proba(model, test, device, args.batch_size)
        test_metrics = metrics_at(test_probs, test_labels, val_at["threshold"])
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "in_dim": in_dim,
            "edge_dim": edge_dim,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "threshold": val_at["threshold"],
            "val_f1": val_at["f1"],
            "test": test_metrics,
        },
        checkpoint,
    )
    metrics_path = checkpoint.with_suffix(".json")
    metrics_path.write_text(json.dumps({
        "checkpoint": str(checkpoint),
        "train": len(train_graphs),
        "val": len(val_graphs),
        "test": len(test),
        "in_dim": in_dim,
        "edge_dim": edge_dim,
        "source": source,
        "threshold": val_at["threshold"],
        "val_metrics": val_at,
        "test_metrics": test_metrics,
    }, indent=2) + "\n", encoding="utf-8")
    print(
        f"Saved {checkpoint} (val ransomware F1 {val_at['f1']:.3f} "
        f"at threshold {val_at['threshold']:.2f})"
    )

    if test_metrics:
        print(
            "Test ransomware  "
            f"P {test_metrics['precision']:.3f}  R {test_metrics['recall']:.3f}  "
            f"F1 {test_metrics['f1']:.3f}  acc {test_metrics['accuracy']:.3f}  "
            f"n {test_metrics['support']}  "
            f"tp {test_metrics['tp']} fp {test_metrics['fp']} "
            f"fn {test_metrics['fn']} tn {test_metrics['tn']}"
        )
    else:
        print("No test graphs yet; skipped the held-out score.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", default="processed_graphs/gnn")
    parser.add_argument(
        "--feature-mode",
        choices=("features", "both", "embeddings"),
        default="features",
        help=(
            "Which preprocess_full.py block vector to use. "
            "features is the counted block vector (opcode, operand, behaviour, API). "
            "Ignored for construct_blocks graphs, which are already counts."
        ),
    )
    parser.add_argument(
        "--raw-features",
        action="store_true",
        help="Keep raw preprocess_full.py counts. By default sizes are log-scaled and counts become rates.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/best_gnn.pt")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
