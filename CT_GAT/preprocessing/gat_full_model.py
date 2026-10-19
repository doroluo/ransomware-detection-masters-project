"""Edge-typed GAT over the rich block vectors from ``preprocess_full.py``.

``FullBlockGAT`` consumes what ``graph_full_loader.py`` produces: one
precomputed vector per block (learned 5-channel embedding concatenated with the
handcrafted feature vector) and typed, directed CFG edges whose one-hot relation
plus DFS back-edge bit ride along as ``edge_attr``.

Design choices follow the project's own measurements in
``CT_GAT/block_level_counting/docs/RESULTS.md``:
  - Edge types help a little, so the relation one-hot is fed through ``edge_dim``
    rather than discarded (unlike the baseline ``OpcodeGAT``).
  - Structure adds little over a regularised bag of blocks, so depth is shallow
    (2 layers by default) and an optional block-dropout acts as the cheap
    regulariser that moved AUC the most in the block-level study.
  - Graph readout concatenates mean and max pooling instead of mean alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool


class FullBlockGAT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.2,
        block_dropout: float = 0.0,
        num_classes: int = 2,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        self.edge_dim = int(edge_dim)
        self.block_dropout = float(block_dropout)
        self.project = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(
            GATConv(
                hidden,
                hidden,
                heads=heads if index < layers - 1 else 1,
                concat=False,
                dropout=dropout,
                edge_dim=self.edge_dim,
            )
            for index in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        # mean + max pooling -> 2 * hidden into the head.
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def _mask_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """Zero a fraction of whole block rows during training (regulariser)."""
        if not self.training or self.block_dropout <= 0.0:
            return x
        keep = (torch.rand(x.size(0), 1, device=x.device) >= self.block_dropout).to(x.dtype)
        return x * keep

    def forward(self, data):
        x = self._mask_blocks(data.x)
        x = self.dropout(F.relu(self.project(x)))

        edge_index = data.edge_index
        edge_attr = data.edge_attr
        if edge_index.numel() == 0:
            # No edges in the whole batch: fall back to self-loops so each block
            # still passes through the attention layers with zeroed relations.
            nodes = torch.arange(x.size(0), device=x.device)
            edge_index = torch.stack([nodes, nodes], dim=0)
            edge_attr = torch.zeros((x.size(0), self.edge_dim), dtype=x.dtype, device=x.device)
        else:
            edge_attr = edge_attr.to(x.dtype)

        for conv, norm in zip(self.convs, self.norms):
            message = self.dropout(F.elu(conv(x, edge_index, edge_attr=edge_attr)))
            x = norm(x + message)

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        pooled = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.classifier(pooled)


if __name__ == "__main__":
    from graph_full_loader import get_loaders, load_dir

    graphs = load_dir()
    if not graphs:
        raise SystemExit(
            "No subgraphs found. Run preprocess_full.py, then rerun this smoke test."
        )
    train_loader, _val, _test, in_dim, edge_dim = get_loaders()
    model = FullBlockGAT(in_dim=in_dim, edge_dim=edge_dim)
    batch = next(iter(train_loader))
    logits = model(batch)
    print(f"in_dim {in_dim}  edge_dim {edge_dim}")
    print("logits:", tuple(logits.shape))
