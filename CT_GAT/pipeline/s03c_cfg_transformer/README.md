# Stage 3c: Model 3, the end-to-end block transformer and GAT

Model 3 trains a transformer over the instructions of each block together with the GAT over the block graph, as one model per
fold. `block_transformer.py` holds it: Janice's `Model1TokenSequence` encodes the first 64 instructions of every block (five
channels each) into a 64-d block vector, and Dorothy's `FullBlockGAT` combines the block vectors over the CFG. It trains on the
pipeline-B builds (the preprocessed zips or `build_b`) through `experiments/team_pipelines/train_b.py`, arm `sig500v3-e2e`;
only the training binaries of a fold fit it, and API names no training binary uses map to the unknown id.

Every command runs from `CT_GAT/pipeline`:

```bash
python -m experiments.team_pipelines.train_b --arm sig500v3-e2e --fold 0 --seed 0 --smoke 8 --device cpu   # quick check
python -m experiments.team_pipelines.train_b --arm sig500v3-e2e --fold 0 --seed 0                           # one fold (GPU)
python -m experiments.queue --plan experiments/plans/team_b.txt                                             # 3 seeds x 5 folds
```

The recipe (AdamW at lr 3e-4, gradient clipping, batch 16, early stopping), the `metrics.json` entries and the comparison of
learning rates are in [`../experiments/team_pipelines/README.md`](../experiments/team_pipelines/README.md) (section
`train_b.py`); the tests are `tests/test_model3_e2e.py`. It needs `torch_geometric` (pinned in `requirements.txt`).

The `-emb` arm (`sig500v3-emb`) is the other Model 3 variant: the same GAT on the fixed 64-d block embeddings that the
preprocessing stores, with nothing trained before the GAT.

The earlier frozen-encoder version of Model 3 (CodeBERT block vectors under the hierarchical GNN) is on branch
`jsonl-reference-models`; its results are in [`../docs/history/`](../docs/history/README.md).
