"""PyG dataset / loader for CFG graphs saved by build_pyg_graphs.py.

Each sample is one executable. Nodes are basic blocks. Instruction token IDs
stay on the graph and are pooled inside the GAT model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "processed" / "splits"
GRAPH_DIR = PROJECT_ROOT / "processed" / "graphs"
BATCH_SIZE = 8


class OpcodeGraphData(Data):
    """Shift both CFG edges and instruction-to-block ids when batching graphs."""

    def __inc__(self, key, value, *args, **kwargs):
        if key in ("edge_index", "insn_to_block"):
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


def load_graph(path: str | Path, label: int | None = None) -> OpcodeGraphData:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    y = int(payload["label"] if label is None else label)
    token_ids = payload["token_ids"].long()
    if token_ids.shape[1] == 3:
        # Legacy graphs without behavior channel.
        pad = torch.zeros(token_ids.shape[0], 1, dtype=torch.long)
        token_ids = torch.cat([token_ids, pad], dim=1)

    num_blocks = int(payload["num_blocks"])
    edge_type = payload.get("edge_type")
    if edge_type is None:
        edge_type = torch.zeros(payload["edge_index"].shape[1], dtype=torch.long)
    else:
        edge_type = edge_type.long()

    block_crypto = payload.get("block_crypto_count")
    if block_crypto is None:
        block_crypto = torch.zeros(num_blocks, dtype=torch.long)
    else:
        block_crypto = block_crypto.long()

    block_file = payload.get("block_file_count")
    if block_file is None:
        block_file = torch.zeros(num_blocks, dtype=torch.long)
    else:
        block_file = block_file.long()

    return OpcodeGraphData(
        token_ids=token_ids,
        insn_to_block=payload["insn_to_block"].long(),
        edge_index=payload["edge_index"].long(),
        edge_type=edge_type,
        block_crypto_count=block_crypto,
        block_file_count=block_file,
        y=torch.tensor(y, dtype=torch.long),
        num_nodes=num_blocks,
        file_id=str(payload.get("file_id", Path(path).stem)),
    )


class OpcodeGraphDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str | Path, graph_dir: str | Path = GRAPH_DIR):
        self.graph_dir = Path(graph_dir)
        df = pd.read_csv(csv_path)
        self.rows = []
        missing = 0
        for _, row in df.iterrows():
            graph_path = self.graph_dir / f"{row['file_id']}.pt"
            if graph_path.is_file():
                self.rows.append((graph_path, int(row["label"])))
            else:
                missing += 1
        if missing:
            print(f"{Path(csv_path).name}: {missing} graphs missing, using {len(self.rows)}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> OpcodeGraphData:
        graph_path, label = self.rows[index]
        return load_graph(graph_path, label)


train_dataset = OpcodeGraphDataset(SPLITS_DIR / "train.csv")
val_dataset = OpcodeGraphDataset(SPLITS_DIR / "validation.csv")
test_dataset = OpcodeGraphDataset(SPLITS_DIR / "test.csv")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


if __name__ == "__main__":
    if len(train_dataset) == 0:
        graphs = sorted(GRAPH_DIR.glob("*.pt"))
        if not graphs:
            raise FileNotFoundError(
                f"No graphs in {GRAPH_DIR}. Run: python build_pyg_graphs.py"
            )
        batch = DataLoader([load_graph(graphs[0])], batch_size=1)
        data = next(iter(batch))
        print("Single-graph fallback:", graphs[0].name)
    else:
        data = next(iter(train_loader))
        print("Train graphs:", len(train_dataset))
        print("Val graphs:", len(val_dataset))
        print("Test graphs:", len(test_dataset))

    print("token_ids:", tuple(data.token_ids.shape))
    print("insn_to_block:", tuple(data.insn_to_block.shape))
    print("edge_index:", tuple(data.edge_index.shape))
    print("edge_type:", tuple(data.edge_type.shape))
    print("block_crypto_count:", tuple(data.block_crypto_count.shape))
    print("num_nodes (blocks):", int(data.num_nodes))
    print("batch size (graphs):", int(data.y.numel()))
    print("labels:", data.y)
