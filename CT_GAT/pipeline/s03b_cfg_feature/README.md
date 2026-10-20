# Stage 3b: the CFG feature model (GNN)

A hierarchical, edge-typed message-passing network over the block CFG and the call graph of a binary. It reads the CFG cache
that `s02_preprocess/cache` writes and gives one logit per binary. Training is `experiments/train.py`, the one training command
for every model of the pipeline; this stage supplies the model, its data path and its entry in the model registry.

Every command runs from `CT_GAT/pipeline`:

```bash
python -m experiments.train --protocol kfold --fold 0 --model gnn --mask-blocks 0.15 --seed 0     # R7, one fold, one seed
python -m experiments.train --help
```

The run directory is `<runs>/<RunConfig.name>/` (`GT_RUNS`, see `common/paths.py`); the cache is the one `datasets.json` names for
the protocol, `--cache <name>` picks another dataset, `GT_CACHE` overrides both. The flags are those of the training script of the
`3-block-level-counting` branch, with the same names, types and defaults, and `RunConfig.name` is unchanged for every option.

## Layout

The steps of the plan keep their names, as subpackages of `gnn/`:

| path | step | content |
|---|---|---|
| `gnn/s05_block_counting/` | 5 | `weights.token_weights` (tf-idf, mean, raw, presence, rawlog), `encoder.BlockEncoder` / `BlockVecEncoder`, `block_vecs` (precomputed block vectors) |
| `gnn/s06_gnn_block_cfg/` | 6 | `edges` (typed block edges, controls, edge drop and flip), `layers.DirGINELayer` |
| `gnn/s07_pool_blocks_to_functions/` | 7 | `readout.FixedReadout`, `GatedReadout` |
| `gnn/s08_gnn_call_graph/` | 8 | `calls.call_pairs`, call-graph layers |
| `gnn/s09_pool_functions_to_binary/` | 9 | no module; the readouts of step 7 over the binary |
| `gnn/s10_unused_import_features/` | 10 | `imports_feat` (`ImportVocab`, `ImportBag`: `--imports header`), `import_side` (converter of the reader output to `<cache>/side/imports/`) |
| `gnn/s11_final_classifier/` | 11 | `head` (`bin_feat`, `make_head`) |
| `gnn/model.py`, `gnn/batching.py` | | `HierGNN`; per-binary arrays, collate, the block-budget sampler and the loader |
| `gnn/spec.py` | | `GnnSpec`, the registry entry |

The tokenizer, the vocabulary and the subgraph selection that the data path reads are stage 2 (`s02_preprocess/tokenizer`,
`s02_preprocess/subgraph.py`).

## The model registry

`experiments/train.py` knows no model. It asks `common/registry.py` for the `ModelSpec` of `cfg.model` and runs the same loop
for every spec: class weights, AdamW, gradient clipping, patience on the later of the F1 and AUC bests, the F1 and AUC
checkpoints, the validation thresholds, and the run directory through `common.run_io.write_run`.

A spec has

| member | meaning |
|---|---|
| `name` | the value of `RunConfig.model` it serves |
| `requires` | paths under the cache directory every run reads; checked before anything is written |
| `check(cfg)` | raises for options the model does not support |
| `build(cfg, info)` | fits what the model fits on the training manifest (`info.manifest`), writes its files to `info.out`, returns the `torch.nn.Module` |
| `loader(cfg, rows, train)` | batches over `rows`, augmenting and shuffling when `train` |
| `forward(model, batch)` | the logits of a batch that is already on the device |
| `variant_tokens(cfg)` | the dash-separated tokens of the run-name variant that the model's options produce |

**To add a model:** write one class with these members, and add one line to `MODELS` in `common/registry.py`:

```python
MODELS = {
    'gnn': 's03b_cfg_feature.gnn.spec:GnnSpec',
    'mymodel': 'my_stage.spec:MySpec',
}
```

`RunConfig.model` must accept the name (`MODELS` in `common/config.py`) and `RunConfig.variant` must produce its name tokens, which
fixes the run names. The spec class is imported on first use and instantiated once per run, so a spec may keep what it fitted in
`build` for `loader`. `GnnSpec` is the example.

## Options that need another stage

`--block-input vec` reads block vectors that `s03c_cfg_transformer/precompute_blockvec.py` writes. Where that package is not
present the command stops at once with a message that names it. `--vocab yanping|hybrid` reads `<cache>/side/yanping/`, which
`experiments/side_arrays` writes. `--stack-tfidf` reads `<runs>/baselines/<dir>/stack_logits.csv` (`s04_evaluation/baselines.py
--stack-logits`). `--imports header` reads `<cache>/side/imports/`; `gnn/s10_unused_import_features/import_side.py` writes it.

## Tests

`tests/test_gnn_equivalence.py` compares the batches and one forward/backward step on the tiny cache with the outputs of the
branch's code (`tests/fixtures/gnn_equivalence.npz`, recorded by `tests/fixtures/make_gnn_equivalence.py`).
`tests/test_train_equivalence.py` (`real_data`, `gpu`) trains two epochs of R7 on fold 0 and compares every epoch to the branch's.
