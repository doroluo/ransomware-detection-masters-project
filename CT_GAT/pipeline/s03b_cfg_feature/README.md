# Stage 3b: Model 2, the team GAT on the 77 block features

Model 2 is Dorothy's GAT over the graph of a binary with the 77 features of each block, trained on our folds by
`experiments/team_pipelines/train_b.py` (the `-feat` arms; the `-emb` arms give the GAT the 64-d block embeddings instead).
Every command runs from `CT_GAT/pipeline`; the arms, build layouts and commands are in
[`../experiments/team_pipelines/README.md`](../experiments/team_pipelines/README.md):

```bash
python -m experiments.team_pipelines.train_b --arm sig500v3-feat --fold 0 --seed 0     # Model 2, one fold, one seed
python -m experiments.team_pipelines.train_b --help
```

## The teammate GAT (`gat/`)

Models 2 and 3 are the GAT of Dorothy Luo (`doroluo`) on the block subgraphs that `s02_preprocess/preprocess_full.py` writes.
Her code is copied here from her branch `origin/gnn`, commit 331ce0b, each file with an attribution in its docstring (author,
branch, commit, original path, the sha256 of the original, what was left out and why):

| file | original | content |
|---|---|---|
| `gat/gat_full_model.py` | `CT_GAT/preprocessing/gat_full_model.py` | `FullBlockGAT`: GATConv with `edge_dim`, 2 layers, hidden 128, 4 heads, mean and max readout. The class is byte-identical; her `__main__` demo is left out |
| `gat/graph_full_loader.py` | `CT_GAT/preprocessing/graph_full_loader.py` | the whole file: `_node_features` (unit-length embedding, log and rate scaling, `both` / `embeddings` / `features`), `_edge_attr`, `load_full_graph`, `load_dir`, `split_graphs`, `get_loaders`. One edit: `GRAPH_DIR` is `CT_GAT/processed/cfg_subgraphs` (`common.catalog.PROCESSED`) |
| `gat/train_cfg.py` | `train_cfg.py` | `BlockGAT` (Pipeline A), `_batches`, `predict_proba`, `metrics_at`, `ransomware_scores`, `best_threshold`; her split, loop and command line are left out (our fold manifests and `experiments/team_pipelines/fit.py` replace them) |

Nothing else of hers is changed. The modules import `torch_geometric` when they are imported, so the team trainers import them
inside the function that uses them. `tests/test_gat_port.py` checks the attribution and, with `GT_TEAM_GNN_ROOT` set to a
worktree at 331ce0b, that the source text of every copied definition equals the original's.
