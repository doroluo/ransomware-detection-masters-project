# Stage 3a: transformers over the token sequence

Two things live here.

* `transformer_model.py` is the teammates' transformer for the sampled-subgraph path of `s02_preprocess` (`preprocess_full.py`
  writes its inputs). It is a flat script module, not part of the package below, and nothing in `experiments/` imports it.
* `token_sequence/` is Model 1: a transformer over the instruction and API tokens of a binary read as one sequence, trained on the folds of
  the block export of stage 3c. It is registered as the model `tokenseq` (`common/registry.py`), so the one training command
  runs it. `token_sequence/README.md` has the input layout, the sequence rule, the windows and the model.

Every command runs from `CT_GAT/pipeline`. Model 1 needs a block export (`s03c_cfg_transformer/README.md`, step 1), then:

```bash
python -m s03c_cfg_transformer.block_export --protocol kfold --arch x86 --out RUNS/export/ours_cap0     # once
python -m experiments.train --model tokenseq --protocol kfold --arch x86 --fold 0 --seed 0              # one fold, one seed
python -m experiments.train --model tokenseq --protocol kfold --fold 0 --smoke 8 --device cpu           # a 2-epoch smoke run
python -m experiments.queue --plan experiments/plans/model1.txt                                         # 3 seeds x 5 folds
```

`--export <name or directory>` (default `ours_cap0`), `--device cuda|cpu`, `--smoke N` and `--window-budget N` belong to
`--model tokenseq`. The run directory is `<runs>/kfold<f>_x86_tokenseq_s<seed>/`, read by `s04_evaluation` like any other; the
gate stage is `python -m s04_evaluation.gates --stage model1`. Result: Model 1 does not beat TF-IDF, the import-table baseline or
the GNN (`docs/history/gates_model1.md`).

Tests: `python -m pytest tests/test_token_sequence.py tests/test_tokenseq_config.py tests/test_model1_trainer.py`.
