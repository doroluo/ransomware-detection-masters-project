"""Graph Attention Network over basic-block CFGs.

Instruction token IDs are embedded like the Transformer, mean-pooled inside
each basic block, then two GAT layers classify the executable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import scatter

from transformer_model import OPCODE_VOCAB_SIZE, OPERAND_VOCAB_SIZE


class OpcodeGAT(nn.Module):
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

    model = OpcodeGAT()
    logits = model(data)
    print("token_ids:", tuple(data.token_ids.shape))
    print("blocks:", int(data.num_nodes))
    print("graphs:", int(data.y.numel()))
    print("logits:", tuple(logits.shape))
    print("labels:", data.y)
