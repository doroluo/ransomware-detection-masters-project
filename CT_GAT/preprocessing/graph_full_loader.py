"""PyG loader for the rich subgraphs saved by ``preprocess_full.py``.

``preprocess_full.py`` writes one ``<sha256>.pt`` per sampled subgraph into
``processed/cfg_subgraphs/`` (see ``save_sample``). Each payload already holds
pooled block vectors, so unlike ``graph_loader.py`` there is nothing to pool at
train time. This module turns each payload into a ``torch_geometric`` ``Data``
whose node matrix is the block vector and whose edges carry the typed one-hot
relation and the DFS back-edge flag.

Node features (``feature_mode``):
  - ``both``        : [block_embeddings | block_features]  (default)
  - ``embeddings``  : learned 5-channel pooled vector only
  - ``features``    : handcrafted feature vector only

Edges:
  - ``edge_index``  : the sampled directed CFG
  - ``edge_attr``   : one-hot edge type (7 relations) with the back-edge bit
                      appended, so ``edge_dim = num_edge_types + 1``

The ``split`` recorded by the preprocessor (train/test) is read from
``dataset_metadata``; when it is missing the loader falls back to a stratified
holdout so a short, split-less run still trains.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = PROJECT_ROOT / "processed" / "cfg_subgraphs"
BATCH_SIZE = 8

FEATURE_MODES = ("both", "embeddings", "features")


def _is_sample_graph(path: Path) -> bool:
    """True for ``<sha256>.pt`` files, matching the preprocessor's names."""
    stem = path.stem
    return (
        path.suffix == ".pt"
        and len(stem) == 64
        and all(char in "0123456789abcdef" for char in stem.lower())
    )


def _node_features(payload: dict, feature_mode: str) -> torch.Tensor:
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {FEATURE_MODES}, got {feature_mode!r}")
    embeddings = payload["block_embeddings"].float()
    features = payload["block_features"].float()
    if feature_mode == "embeddings":
        return embeddings
    if feature_mode == "features":
        return features
    if embeddings.shape[0] != features.shape[0]:
        raise ValueError(
            f"block count mismatch: {embeddings.shape[0]} embeddings vs "
            f"{features.shape[0]} feature rows"
        )
    return torch.cat([embeddings, features], dim=1)


def _edge_attr(payload: dict) -> torch.Tensor:
    """One-hot relation with the DFS back-edge bit appended as an extra column."""
    one_hot = payload["edge_attr"].float()
    if one_hot.numel() == 0:
        return one_hot.reshape(0, one_hot.shape[1] + 1 if one_hot.dim() == 2 else 1)
    back = payload.get("edge_back")
    if back is None:
        back = torch.zeros(one_hot.shape[0], dtype=torch.bool)
    back = back.to(dtype=one_hot.dtype).reshape(-1, 1)
    return torch.cat([one_hot, back], dim=1)


def load_full_graph(path: str | Path, feature_mode: str = "both") -> Data:
    """Read one ``preprocess_full.py`` payload into a PyG ``Data`` graph."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    x = _node_features(payload, feature_mode)
    num_nodes = int(x.shape[0])
    edge_index = payload["edge_index"].long()
    edge_attr = _edge_attr(payload)
    edge_type = payload["edge_type"].long()

    metadata = payload.get("dataset_metadata") or {}
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(int(payload["label"]), dtype=torch.long),
        num_nodes=num_nodes,
    )
    data.edge_type = edge_type
    data.sample_id = str(payload.get("file_id", path.stem))
    data.split = (metadata.get("split") or "").lower()
    data.arch = metadata.get("architecture") or metadata.get("arch") or ""
    return data


def load_dir(graph_dir: str | Path = GRAPH_DIR, feature_mode: str = "both") -> list[Data]:
    """Load every ``<sha256>.pt`` in ``graph_dir``."""
    graph_dir = Path(graph_dir)
    if not graph_dir.is_dir():
        raise FileNotFoundError(
            f"No subgraph directory at {graph_dir}. Run preprocess_full.py first."
        )
    graphs = []
    for path in sorted(graph_dir.glob("*.pt")):
        if _is_sample_graph(path):
            graphs.append(load_full_graph(path, feature_mode))
    return graphs


def _stratified_holdout(graphs: list[Data], fraction: float, seed: int):
    by_label: dict[int, list[Data]] = {}
    for graph in graphs:
        by_label.setdefault(int(graph.y.item()), []).append(graph)
    rng = random.Random(seed)
    held, rest = [], []
    for group in by_label.values():
        rng.shuffle(group)
        cut = max(1, round(len(group) * fraction)) if len(group) > 1 else 0
        held.extend(group[:cut])
        rest.extend(group[cut:])
    rng.shuffle(held)
    rng.shuffle(rest)
    return rest, held


def split_graphs(graphs: list[Data], val_fraction: float = 0.15, seed: int = 0):
    """Split by the recorded train/test flag, carving a validation set from train.

    Falls back to a stratified holdout for both val and test when the payloads
    carry no split (e.g. a short ``--limit`` preprocessing run).
    """
    train_pool = [g for g in graphs if g.split == "train"]
    test = [g for g in graphs if g.split == "test"]
    if not train_pool and not test:
        train_pool, test = _stratified_holdout(graphs, val_fraction, seed)
    train, val = _stratified_holdout(train_pool, val_fraction, seed)
    return train, val, test


def get_loaders(
    graph_dir: str | Path = GRAPH_DIR,
    feature_mode: str = "both",
    batch_size: int = BATCH_SIZE,
    val_fraction: float = 0.15,
    seed: int = 0,
):
    """Return ``(train_loader, val_loader, test_loader, in_dim, edge_dim)``."""
    graphs = load_dir(graph_dir, feature_mode)
    if not graphs:
        raise FileNotFoundError(f"No subgraph .pt files in {graph_dir}")
    train, val, test = split_graphs(graphs, val_fraction, seed)
    in_dim = int(graphs[0].x.shape[1])
    edge_dim = int(graphs[0].edge_attr.shape[1]) if graphs[0].edge_attr.dim() == 2 else 1
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, in_dim, edge_dim


if __name__ == "__main__":
    graphs = load_dir()
    print(f"Loaded {len(graphs)} subgraphs from {GRAPH_DIR}")
    if graphs:
        example = graphs[0]
        print("node features x:", tuple(example.x.shape))
        print("edge_index:", tuple(example.edge_index.shape))
        print("edge_attr:", tuple(example.edge_attr.shape))
        labels = [int(g.y.item()) for g in graphs]
        splits: dict[str, int] = {}
        for g in graphs:
            splits[g.split or "(none)"] = splits.get(g.split or "(none)", 0) + 1
        print("labels:", {label: labels.count(label) for label in sorted(set(labels))})
        print("splits:", splits)
