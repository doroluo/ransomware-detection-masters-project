"""Graph Attention Network over basic-block CFGs.

OpcodeGAT — baseline: mean-pool instructions → blocks → graph.
BehaviorOpcodeGAT — typed edges (cfg/loop/call), crypto/file counts on
blocks, per-block malware scores, and a behavior head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import scatter

from token_mapping import BEHAVIOR_VOCAB_SIZE, HEAD_BEHAVIOR_NAMES
from transformer_model import OPCODE_VOCAB_SIZE, OPERAND_VOCAB_SIZE

NUM_EDGE_TYPES = 3  # cfg, loop, call


class OpcodeGAT(nn.Module):
    """Baseline GAT: embed 3 token channels, mean-pool blocks, classify."""

    def __init__(
        self,
        opcode_vocab_size: int = OPCODE_VOCAB_SIZE,
        operand_vocab_size: int = OPERAND_VOCAB_SIZE,
        d_model: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        self.opcode_embedding = nn.Embedding(
            opcode_vocab_size, d_model, padding_idx=0
        )
        self.operand_embedding = nn.Embedding(
            operand_vocab_size, d_model, padding_idx=0
        )
        self.gat1 = GATConv(
            d_model, d_model, heads=heads, concat=False, dropout=dropout
        )
        self.gat2 = GATConv(
            d_model, d_model, heads=1, concat=False, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, data):
        token_ids = data.token_ids
        insn_x = (
            self.opcode_embedding(token_ids[:, 0])
            + self.operand_embedding(token_ids[:, 1])
            + self.operand_embedding(token_ids[:, 2])
        )

        block_x = scatter(
            insn_x,
            data.insn_to_block,
            dim=0,
            dim_size=int(data.num_nodes),
            reduce="mean",
        )
        block_x = self.dropout(block_x)
        block_x = self.norm1(F.elu(self.gat1(block_x, data.edge_index)))
        block_x = self.norm2(F.elu(self.gat2(block_x, data.edge_index)))
        graph_x = global_mean_pool(block_x, data.batch)
        return self.classifier(graph_x)


class BehaviorOpcodeGAT(nn.Module):
    """Behavior-aware GAT with typed edges and per-block scoring.

    Graph label uses mean-pooled block features (stable, like the baseline).
    Per-block malware / behavior heads are kept for explanation only.

    Returns:
      logits:          [G, 2]
      block_logits:    [num_blocks, 2]
      behavior_logits: [num_blocks, 4]
    """

    def __init__(
        self,
        opcode_vocab_size: int = OPCODE_VOCAB_SIZE,
        operand_vocab_size: int = OPERAND_VOCAB_SIZE,
        behavior_vocab_size: int = BEHAVIOR_VOCAB_SIZE,
        d_model: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
        num_classes: int = 2,
        num_behaviors: int = len(HEAD_BEHAVIOR_NAMES),
        num_edge_types: int = NUM_EDGE_TYPES,
    ):
        super().__init__()
        self.opcode_embedding = nn.Embedding(
            opcode_vocab_size, d_model, padding_idx=0
        )
        self.operand_embedding = nn.Embedding(
            operand_vocab_size, d_model, padding_idx=0
        )
        self.behavior_embedding = nn.Embedding(
            behavior_vocab_size, d_model, padding_idx=0
        )
        self.edge_embedding = nn.Embedding(num_edge_types, d_model)
        self.count_proj = nn.Linear(2, d_model)

        self.gat1 = GATConv(
            d_model, d_model, heads=heads, concat=False, dropout=dropout, edge_dim=d_model
        )
        self.gat2 = GATConv(
            d_model, d_model, heads=1, concat=False, dropout=dropout, edge_dim=d_model
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        self.block_classifier = nn.Linear(d_model, num_classes)
        self.behavior_head = nn.Linear(d_model, num_behaviors)

    def encode_blocks(self, data):
        token_ids = data.token_ids
        insn_x = (
            self.opcode_embedding(token_ids[:, 0])
            + self.operand_embedding(token_ids[:, 1])
            + self.operand_embedding(token_ids[:, 2])
            + self.behavior_embedding(token_ids[:, 3])
        )
        block_x = scatter(
            insn_x,
            data.insn_to_block,
            dim=0,
            dim_size=int(data.num_nodes),
            reduce="mean",
        )
        counts = torch.stack(
            [
                data.block_crypto_count.float(),
                data.block_file_count.float(),
            ],
            dim=-1,
        )
        block_x = block_x + self.count_proj(counts)
        edge_attr = self.edge_embedding(data.edge_type.clamp(min=0, max=NUM_EDGE_TYPES - 1))
        block_x = self.dropout(block_x)
        block_x = self.norm1(
            F.elu(self.gat1(block_x, data.edge_index, edge_attr=edge_attr))
        )
        block_x = self.norm2(
            F.elu(self.gat2(block_x, data.edge_index, edge_attr=edge_attr))
        )
        return block_x

    def forward(self, data):
        block_x = self.encode_blocks(data)
        block_logits = self.block_classifier(block_x)
        behavior_logits = self.behavior_head(block_x)
        # Stable graph decision (same idea as OpcodeGAT).
        graph_x = global_mean_pool(block_x, data.batch)
        logits = self.classifier(graph_x)
        return logits, block_logits, behavior_logits


def block_behavior_targets(batch, head_ids: tuple[int, ...]) -> torch.Tensor:
    """Per-block multi-hot targets from instruction behavior channel [num_blocks, H]."""
    num_blocks = int(batch.num_nodes)
    device = batch.token_ids.device
    targets = torch.zeros(num_blocks, len(head_ids), device=device)
    token_beh = batch.token_ids[:, 3]
    for i, behavior_id in enumerate(head_ids):
        hit = (token_beh == behavior_id).float()
        targets[:, i] = scatter(
            hit,
            batch.insn_to_block,
            dim=0,
            dim_size=num_blocks,
            reduce="max",
        )
    return targets


def top_blocks_report(
    data,
    block_logits: torch.Tensor,
    behavior_logits: torch.Tensor,
    top_k: int = 5,
) -> list[dict]:
    """Rank blocks for a single-graph batch by ransomware score."""
    if int(data.y.numel()) != 1:
        raise ValueError("top_blocks_report expects a single-graph batch")
    scores = torch.softmax(block_logits, dim=-1)[:, 1]
    behavior_prob = torch.sigmoid(behavior_logits)
    order = scores.argsort(descending=True)[:top_k]
    rows = []
    for block_id in order.tolist():
        rows.append(
            {
                "block_id": block_id,
                "ransomware_score": float(scores[block_id]),
                "crypto_count": int(data.block_crypto_count[block_id]),
                "file_count": int(data.block_file_count[block_id]),
                **{
                    name.lower(): float(behavior_prob[block_id, i])
                    for i, name in enumerate(HEAD_BEHAVIOR_NAMES)
                },
            }
        )
    return rows


if __name__ == "__main__":
    from graph_loader import GRAPH_DIR, load_graph, train_loader
    from torch_geometric.loader import DataLoader

    if len(train_loader.dataset) > 0:
        data = next(iter(train_loader))
    else:
        graphs = sorted(GRAPH_DIR.glob("*.pt"))
        if not graphs:
            raise FileNotFoundError(
                f"No graphs in {GRAPH_DIR}. Run: python build_pyg_graphs.py"
            )
        data = next(iter(DataLoader([load_graph(graphs[0])], batch_size=1)))

    baseline = OpcodeGAT()
    print("Baseline logits:", baseline(data).shape)

    model = BehaviorOpcodeGAT()
    logits, block_logits, behavior_logits = model(data)
    print("Behavior logits:", tuple(logits.shape))
    print("Block logits:", tuple(block_logits.shape))
    print("Behavior head:", tuple(behavior_logits.shape), HEAD_BEHAVIOR_NAMES)