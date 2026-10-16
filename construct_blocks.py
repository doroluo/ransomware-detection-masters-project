import os
import json
import gzip
import gc
import warnings
import torch
from pathlib import Path
from torch_geometric.data import Data, Dataset
from transformers import AutoTokenizer, AutoModel


def count_dataset_graphs(root_dir: str = "."):
    """
    Scans the cfg directory recursively and prints the graph breakdown per split and label.
    """
    cfg_dir = Path(root_dir) / "cfg"
    splits = ["train", "test"]
    labels = ["goodware", "ransomware"]
    
    total_graphs = 0
    print("=== Graph Count Breakdown ===")
    for split in splits:
        split_total = 0
        print(f"\n[{split.upper()} SPLIT]")
        for label in labels:
            folder = cfg_dir / label / split
            if folder.exists():
                count = len(list(folder.rglob("*.jsonl*")))
                split_total += count
                print(f"  • {label:<12}: {count:>5} graphs")
            else:
                print(f"  • {label:<12}: Directory not found!")
        print(f"  Subtotal ({split:<5}) : {split_total:>5} graphs")
        total_graphs += split_total
        
    print(f"\n=============================")
    print(f"Total Dataset Graphs : {total_graphs}\n")


class MendeleyCFGDataset(Dataset):
    """
    Memory-efficient PyTorch Geometric Dataset loader for disassembly CFGs.
    """
    LABEL_MAP = {"goodware": 0, "ransomware": 1}

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        max_blocks_per_binary: int = 500,
        lm_model_name: str = "microsoft/codebert-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.split = split
        self.max_blocks = max_blocks_per_binary
        self.device = device
        
        # Load Hugging Face Transformer
        self.tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
        self.lm_model = AutoModel.from_pretrained(lm_model_name).to(self.device)
        self.lm_model.eval()

        # Collect file paths and ground truth labels (recursive rglob)
        self.samples = []
        cfg_dir = Path(root_dir) / "cfg"
        for label_name, label_idx in self.LABEL_MAP.items():
            split_path = cfg_dir / label_name / split
            if split_path.exists():
                for file_path in split_path.rglob("*.jsonl*"):
                    self.samples.append((str(file_path), label_idx))

    def len(self):
        return len(self.samples)

    def get(self, idx):
        file_path, label = self.samples[idx]
        return self._process_binary_jsonl(file_path, label)

    def _process_binary_jsonl(self, file_path: str, label: int) -> Data:
        open_fn = gzip.open if file_path.endswith(".gz") else open
        
        imports_by_id = {}
        rva_to_func = {}
        functions_data = []

        # 1. Parse Records
        with open_fn(file_path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rec_type = rec.get("record")
                
                if rec_type == "header":
                    imports = rec.get("pe", {}).get("imports", [])
                    for i, imp in enumerate(imports):
                        if imp.get("name"):
                            imports_by_id[i] = imp["name"]
                elif rec_type == "function":
                    functions_data.append(rec)
                    if rec.get("rva") and rec.get("name"):
                        rva_to_func[rec["rva"]] = rec["name"]

        # 2. Basic Block Extraction & Subgraph Selection
        node_id_map = {}
        block_texts = []
        block_to_func_id = []
        global_node_idx = 0
        func_id_counter = 0

        for func in functions_data:
            if func.get("thunk") and len(func.get("blocks", [])) <= 1:
                continue

            current_func_id = func_id_counter
            func_id_counter += 1

            for block in func.get("blocks", []):
                if global_node_idx >= self.max_blocks:
                    break
                
                node_id_map[(func["id"], block["id"])] = global_node_idx
                
                inst_strings = []
                for inst in block.get("instructions", []):
                    mnem = inst.get("mnemonic", "")
                    ops = []
                    for op in inst.get("operands", []):
                        if not isinstance(op, dict):
                            continue
                        
                        reg = op.get("register")
                        imm = op.get("immediate")
                        trva = op.get("target_rva")
                        disp = op.get("displacement")

                        if reg:
                            ops.append(str(reg))
                        elif imm is not None:
                            ops.append(hex(imm) if isinstance(imm, int) else str(imm))
                        elif trva is not None:
                            ops.append(hex(trva) if isinstance(trva, int) else str(trva))
                        elif disp is not None:
                            ops.append(f"[disp:{hex(disp) if isinstance(disp, int) else disp}]")

                    inst_strings.append(f"{mnem} {', '.join(ops)}".strip())
                
                block_texts.append(" ; ".join(inst_strings) if inst_strings else "nop")
                block_to_func_id.append(current_func_id)
                global_node_idx += 1

        # 3. Extract Block Embeddings via CodeBERT
        block_embeddings = []
        batch_size = 16

        with torch.inference_mode():
            for i in range(0, len(block_texts), batch_size):
                batch = block_texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)

                out = self.lm_model(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1)
                emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                
                block_embeddings.append(emb.cpu())
                del inputs, out, mask, emb

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        x = torch.cat(block_embeddings, dim=0) if block_embeddings else torch.zeros((1, 768))

        # 4. Construct Intra-Function CFG Edges
        cfg_src, cfg_dst = [], []
        for func in functions_data:
            f_id = func["id"]
            for edge in func.get("edges", []):
                u_key, v_key = (f_id, edge[0]), (f_id, edge[1])
                if u_key in node_id_map and v_key in node_id_map:
                    cfg_src.append(node_id_map[u_key])
                    cfg_dst.append(node_id_map[v_key])

        edge_index_cfg = torch.tensor([cfg_src, cfg_dst], dtype=torch.long)
        func_batch = torch.tensor(block_to_func_id, dtype=torch.long)

        return Data(
            x=x,
            edge_index_cfg=edge_index_cfg,
            func_batch=func_batch,
            y=torch.tensor([label], dtype=torch.long)
        )


def preprocess_and_save_dataset(root_dir: str = ".", output_dir: str = "processed_graphs"):
    """
    Pre-processes all graphs across splits (train/test) and saves them to disk as PyTorch .pt files.
    """
    for split in ["train", "test"]:
        split_out_dir = Path(output_dir) / split
        split_out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n--> Pre-processing '{split}' split...")
        dataset = MendeleyCFGDataset(root_dir=root_dir, split=split, max_blocks_per_binary=500)
        
        num_graphs = len(dataset)
        print(f"    Found {num_graphs} graph(s) to process in '{split}' split.")
        
        for i in range(num_graphs):
            data = dataset[i]
            save_path = split_out_dir / f"graph_{i}.pt"
            torch.save(data, save_path)
            
            # Print progress every 50 graphs
            if (i + 1) % 50 == 0 or (i + 1) == num_graphs:
                print(f"    Saved [{i + 1}/{num_graphs}] graphs to {split_out_dir}")


if __name__ == "__main__":
    # 1. Print Dataset Graph Breakdown
    count_dataset_graphs(root_dir=".")

    # 2. Pre-process and save all graphs to disk
    preprocess_and_save_dataset(root_dir=".", output_dir="processed_graphs")