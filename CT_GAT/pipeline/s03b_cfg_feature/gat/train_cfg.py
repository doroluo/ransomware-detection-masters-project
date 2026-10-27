"""Teammate code, copied into this repository: Dorothy Luo (doroluo), branch origin/gnn, commit 331ce0b,
original path `train_cfg.py`.
sha256 of the original file: 8ee82aa89711a5c213d95a8632b40a7b964dfee1e6d062062ffb234583a7fdb6 (the CRLF checkout that the pins of
`experiments/team_pipelines/common.PINNED_SHA256` are taken from); 732fb83e692e8d9ed3d1fae16e391ecb3dc2bc8f948dcace94d4251df35e658c as git stores it (LF).
Left out: `_is_sample_graph`, `load_split`, `stratified_holdout`, `evaluate`, `train`, `main` and the command line (our fold
manifests and `experiments/team_pipelines/fit.py` replace her split, her loop and her checkpoints). Kept, each source text
unchanged: `BlockGAT` (Pipeline A's model, `train_a`), `_batches`, `ransomware_scores`, `predict_proba`, `metrics_at` and
`best_threshold`. The imports are the ones these definitions use.
`tests/test_gat_port.py` compares the source text of every definition kept here with the pinned worktree's module
(GT_TEAM_GNN_ROOT).

---- the original docstring ----

Train a graph-attention classifier on CFG graphs.

Reads processed_graphs/gnn/{train,test}/<sha256>.pt from `construct_blocks.py --gnn`,
or CodeBERT graphs from `construct_blocks.py`. Holds out 15% of the training
graphs for early stopping, then scores the recommended test split.
Reports ransomware precision, recall, and F1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv


class BlockGAT(nn.Module):
    """Project block features, then attend over the function CFGs."""

    def __init__(self, in_dim, hidden=128, heads=4, dropout=0.2):
        super().__init__()
        self.project = nn.Linear(in_dim, hidden)
        self.conv1 = GATConv(hidden, hidden, heads=heads, concat=False, dropout=dropout)
        self.conv2 = GATConv(hidden, hidden, heads=1, concat=False, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, data):
        x = self.dropout(F.relu(self.project(data.x)))
        edge_index = data.edge_index
        if edge_index.numel() == 0:
            nodes = x.size(0)
            loops = torch.arange(nodes, device=x.device)
            edge_index = torch.stack([loops, loops], dim=0)
        x = self.norm1(x + self.dropout(F.relu(self.conv1(x, edge_index))))
        x = self.norm2(x + self.dropout(self.conv2(x, edge_index)))
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
