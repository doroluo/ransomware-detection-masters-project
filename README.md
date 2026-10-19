# masters-project

## CFG Autotokenizer
Tokenizes the CFG blocks generated IDA Pro. The CFG zip file is located [here](https://drive.google.com/drive/folders/19bufYY6xL3eUWbv0abOYgvC_wi5Ak1eO?usp=drive_link).

The project uses the Hugging Face Transformers (microsoft/codebert-base) to extract contextual embeddings from basic blocks and into feature vectors.

### Install Dependencies
```bash
pip install torch torch-geometric transformers
```

### Run Autotokenizer & Graph Processing
The IDA Pro exports live in `data/Combined_Goodware` and `data/Combined_Ransomware`.
Each recommended sample is a `<sha256>.jsonl.gz` CFG (schema `ida-cfg-research/2.1`),
listed in `splits/train.csv` and `splits/test.csv`. Unassigned samples stay in
`splits/unassigned.csv` and are not used.

```bash
python construct_blocks.py --count-only
python construct_blocks.py --limit 4          # short run
python construct_blocks.py                    # all recommended train and test samples
```

Graphs are written to `processed_graphs/{train,test}/<sha256>.pt`.
Each node is a CodeBERT vector of one basic block. Edges are the intra-function CFG.
The block budget (default 500) keeps blocks that reference imports first.

### Train the GNN
Opcode and API-call features are enough for the graph network, and they stay small
enough to train on every recommended sample:

```bash
python construct_blocks.py --gnn
python train_cfg.py --epochs 20
```

Graphs land in `processed_graphs/gnn/{train,test}/<sha256>.pt`. Each node is one
basic block (opcode mix, crypto/file/registry/process/network API counts, and
block degree). Edges are the intra-function CFG. The checkpoint is
`checkpoints/best_gnn.pt`.

CodeBERT embeddings are a separate, heavier run: `python construct_blocks.py`.
`train_cfg.py --graphs processed_graphs` trains on those vectors instead.

### Inspect a Processed Graph
```bash
python inspect_graph.py processed_graphs/train/<sha256>.pt
```

## Other pipelines
- CNN-ViT: [CNN-ViT/README.md](CNN-ViT/README.md)
- CT_GAT preprocessing and the block-level counting GNN: [CT_GAT/README.md](CT_GAT/README.md) and [CT_GAT/block_level_counting/README.md](CT_GAT/block_level_counting/README.md)
