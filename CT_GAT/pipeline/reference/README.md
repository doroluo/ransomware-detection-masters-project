# reference: the jsonl-only models

Three packages that are kept, runnable and tested, but are not on the main path of the pipeline (IDA export, `preprocess_full`
builds, Models 1 to 3, evaluation). They read the CFG cache that `s02_preprocess/cache` builds from the `.jsonl.gz` exports alone,
the pipeline adopted on 2026-10-19, and they produced the R-ladder of the first plan and its token-sequence and CodeBERT rows.

| package | what it is | README |
|---|---|---|
| `gnn/` | `HierGNN`, the hierarchical edge-typed GNN over the block CFG and the call graph on block token counts (`--model gnn`): the R-ladder (R0 TF-IDF, R2 no edges, R4 shuffled edges, R5 untyped, R7 typed), the subgraph budgets, the imports header, the block weightings | [`gnn/README.md`](gnn/README.md) |
| `token_sequence/` | the token-sequence transformer over the block export (`--model tokenseq`) | [`token_sequence/README.md`](token_sequence/README.md) |
| `cfg_transformer/` | the frozen-encoder path to Model 3: block export, encoder contract, CodeBERT and reference encoders, precomputed block vectors that the GNN reads with `--block-input vec` | [`cfg_transformer/README.md`](cfg_transformer/README.md) |

The tables they produced are in [`../docs/history/`](../docs/history/README.md): [`gates_noapi.md`](../docs/history/gates_noapi.md),
[`gates_v3.md`](../docs/history/gates_v3.md), [`gates_paths_codebert.md`](../docs/history/gates_paths_codebert.md),
[`gates_model1.md`](../docs/history/gates_model1.md), [`gates_imports_header.md`](../docs/history/gates_imports_header.md) and
[`gates_subgraph.md`](../docs/history/gates_subgraph.md). Their plans are in `experiments/plans/` (`day1`, `day1b`, `subgraph`,
`noapi`, `r2_noapi`, `l2_nocall_seeds`, `vocab`, `imports`, `imports_header`, `v3`, `paths`, `paths_codebert`, `model1`).

## How to run them

Every command runs from `CT_GAT/pipeline`, on the CFG cache of the root README (`build_cache`). `--model gnn` and `--model tokenseq`
still resolve through `MODELS` in `common/registry.py`, and the run names are unchanged, so `experiments/plans/*.txt` and the gate
stages (`experiments/stages/`) work as they did.

The GNN (GPU only: `experiments/train.py` moves it to CUDA; one run per fold and seed):

```bash
python -m experiments.train --protocol kfold --fold 0 --seed 0 --model gnn --mask-blocks 0.15                     # R7
python -m experiments.train --protocol kfold --fold 0 --seed 0 --model gnn --mask-blocks 0.15 --control no_edges  # R2
```

The frozen-encoder path to Model 3. The block export (CPU, minutes), the encoder checkpoints and the vectors are made once
(`HF_HOME` holds `microsoft/codebert-base`); the export is also what `tokenseq` reads:

```bash
python -m reference.cfg_transformer.block_export --protocol kfold --arch x86 --out RUNS/export/ours_cap0
python -m reference.cfg_transformer.encoders.codebert --write-checkpoints --export ours_cap0 --name codebert
python -m reference.cfg_transformer.precompute_blockvec --encoder-name codebert --export ours_cap0 --folds 0 1 2 3 4
python -m experiments.train --protocol kfold --fold 0 --seed 0 --model gnn --mask-blocks 0.15 --block-input vec --block-vec ours_cap0_codebert
```

The export records `export_sha` `e09971a020a8...`, the value of the published runs.

The token-sequence transformer (GPU for a full run; `--device cpu --smoke 8` trains 8 binaries for 2 epochs on the CPU, as a
check of the setup):

```bash
python -m experiments.train --model tokenseq --protocol kfold --fold 0 --seed 0 --device cpu --smoke 8
python -m experiments.train --model tokenseq --protocol kfold --fold 0 --seed 0
python -m experiments.train --model tokenseq --protocol kfold --fold 0 --seed 0 --threshold-from oof
```

Times, measured on the development machine: `block_export` is CPU, minutes; `--model gnn` and a full `--model tokenseq` run are
GPU runs, not timed here.

The header-imports arrays that `--imports header` and `baselines --imports` read are written by `s04_evaluation/import_side.py`,
which is not a reference module: the baselines of the paper need it.

## The rule

These packages are frozen. They change only to keep their results reproducible: a fix that alters a recorded run name, a batch
or a prediction is not a fix here. `tests/test_gnn_equivalence.py` (the batches and one step against the recorded outputs),
`tests/test_s03_equivalence.py` and `tests/test_run_names.py` pin them. A new model goes to `s03a_transformer/`,
`s03b_cfg_feature/` or `s03c_cfg_transformer/`, not here.
