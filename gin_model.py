#!/usr/bin/env python3
"""
gin_model.py

A Graph Isomorphism Network (Xu et al., "How Powerful are Graph Neural
Networks?", ICLR 2019) for classifying opcode transition graphs as goodware or
ransomware.

The GIN update is

    h_v^(k) = MLP^(k) ( (1 + eps^(k)) * h_v^(k-1) + sum_{u in N(v)} w_uv h_u^(k-1) )

The sum aggregator is what makes GIN as discriminative as the Weisfeiler-Lehman
test; mean and max aggregators cannot tell apart graphs that differ only in how
often a neighbourhood repeats. That matters here because "how often does `xor`
follow `mov`" is exactly the kind of signal that separates a crypto loop from
ordinary code.

Graphs are small (~60 nodes, capped by the opcode vocabulary), so batches are
padded into a dense [batch, nodes, nodes] adjacency instead of using a sparse
scatter library. That keeps this file dependency-free apart from torch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adjacency(adjacency, mode):
    """
    adjacency: [B, N, N] of raw transition counts, entry [b, u, v] = count(u -> v).

    row  each row sums to 1, so the graph becomes a Markov chain over opcodes.
         Removes the raw instruction count, which otherwise dwarfs everything
         and just measures file size.
    log  log1p of the counts, keeping some magnitude without the huge dynamic
         range (counts span 1 to ~10^6).
    none raw counts.
    """
    if mode == "none":
        return adjacency
    if mode == "log":
        return torch.log1p(adjacency)
    if mode == "row":
        row_sum = adjacency.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return adjacency / row_sum
    raise ValueError(f"unknown adjacency normalisation: {mode}")


class GINLayer(nn.Module):
    """One GIN convolution: aggregate neighbours, then a 2-layer MLP."""

    def __init__(self, in_dim, out_dim, train_eps=True, dropout=0.0, direction="both"):
        super().__init__()
        self.direction = direction
        self.eps = nn.Parameter(torch.zeros(1), requires_grad=train_eps)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.BatchNorm1d(out_dim)

    def forward(self, h, adjacency, mask):
        """
        h:         [B, N, F] node features
        adjacency: [B, N, N] normalised, [b, u, v] = weight of u -> v
        mask:      [B, N] 1 for real nodes, 0 for padding
        """
        # A^T @ h gathers from predecessors, A @ h gathers from successors.
        # An opcode's meaning depends on both what precedes and what follows it.
        if self.direction == "out":
            neighbours = torch.bmm(adjacency, h)
        elif self.direction == "in":
            neighbours = torch.bmm(adjacency.transpose(1, 2), h)
        else:
            neighbours = torch.bmm(adjacency, h) + torch.bmm(adjacency.transpose(1, 2), h)

        out = (1.0 + self.eps) * h + neighbours

        # BatchNorm1d expects [*, F], so flatten the batch and node axes and
        # run it only over real nodes, otherwise padding skews the statistics.
        flat_mask = mask.reshape(-1).bool()
        flat = out.reshape(-1, out.size(-1))
        result = torch.zeros(flat.size(0), self.mlp[-1].out_features,
                             device=flat.device, dtype=flat.dtype)
        if flat_mask.any():
            result[flat_mask] = self.norm(F.relu(self.mlp(flat[flat_mask])))
        return result.view(out.size(0), out.size(1), -1) * mask.unsqueeze(-1)


class GIN(nn.Module):
    """
    Opcode embedding + numeric node features -> stack of GIN layers -> readout.

    Following the GIN paper, every layer's pooled output feeds the classifier,
    not just the last one. Shallow layers capture individual opcode frequencies
    and deep layers capture larger instruction idioms; which depth matters is
    not known in advance, so the model gets to weigh all of them.
    """

    def __init__(
        self,
        vocab_size,
        num_features,
        embedding_dim=64,
        hidden_dim=64,
        num_layers=3,
        num_classes=2,
        dropout=0.5,
        adjacency_norm="row",
        direction="both",
        pooling="sum",
    ):
        super().__init__()
        self.adjacency_norm = adjacency_norm
        self.pooling = pooling
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim + num_features, hidden_dim),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList([
            GINLayer(hidden_dim, hidden_dim, dropout=dropout, direction=direction)
            for _ in range(num_layers)
        ])

        # One head per layer, including the initial projection.
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_layers + 1)
        ])
        self.dropout = nn.Dropout(dropout)

    def pool(self, h, mask):
        h = h * mask.unsqueeze(-1)
        if self.pooling == "mean":
            return h.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        if self.pooling == "max":
            return h.masked_fill(mask.unsqueeze(-1) == 0, float("-inf")).max(dim=1).values
        return h.sum(dim=1)

    def forward(self, node_ids, features, adjacency, mask):
        """
        node_ids:  [B, N]    vocabulary id per node
        features:  [B, N, F] numeric node features
        adjacency: [B, N, N] raw transition counts
        mask:      [B, N]
        """
        adjacency = normalize_adjacency(adjacency, self.adjacency_norm)
        adjacency = adjacency * mask.unsqueeze(1) * mask.unsqueeze(2)

        h = torch.cat([self.embedding(node_ids), features], dim=-1)
        h = self.input_proj(h) * mask.unsqueeze(-1)

        logits = self.heads[0](self.dropout(self.pool(h, mask)))
        for layer, head in zip(self.layers, self.heads[1:]):
            h = layer(h, adjacency, mask)
            logits = logits + head(self.dropout(self.pool(h, mask)))
        return logits
