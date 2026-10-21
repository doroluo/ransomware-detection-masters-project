# Stage 3a: transformers over the token sequence

Two models of the token sequence live here, each a package and each registered in `common/registry.py`, so that the one training
command (`python -m experiments.train --model <name>`) runs it.

* `token_sequence/` is Model 1 as the pipeline trains it: a transformer over the instruction and API tokens of a binary read as one
  sequence, trained on the folds of the block export of stage 3c (model `tokenseq`). `token_sequence/README.md` has the input
  layout, the sequence rule, the windows and the model.
* `five_channel/` is Janice Park's Model 1 (model `fivechannel`): her `FiveChannelEmbedder`, `Model1TokenSequence`, `Model1Dataset`
  and `model1_collate_fn` in `five_channel/model.py` (the file she wrote as `s03a_transformer/token_sequence.py`; it moved with
  `git mv` when the package of `tokenseq` took that name, and only its imports changed), trained by `five_channel/train.py` on our
  folds. Its input is the folder of `<sample_id>.pt` files of a pipeline-B build (Yanping's stage-2 output,
  `experiments/team_pipelines/build_b.py`), five token ids per instruction: opcode, two operands, API and behaviour tag.

Every command runs from `CT_GAT/pipeline`, which is the package root (there is no `pipeline` package above it: the files import
`common`, `s02_preprocess`, ... directly, as the tests do).

## fivechannel

```bash
python -m experiments.train --model fivechannel --protocol kfold --arch x86 --fold 0 --seed 0 --build <build folder>   # L128
python -m experiments.train --model fivechannel --protocol kfold --arch x86 --fold 0 --seed 0 --max-instructions 512   # -L512
python -m experiments.train --model fivechannel --protocol kfold --fold 0 --smoke 8 --device cpu                       # 2 epochs
python -m experiments.queue --plan experiments/plans/model1_variants.txt                                                # the variants
```

`--build <dir>` is the build folder; without it the `team_b_build` source of `datasets.json` (`GT_TEAM_B_BUILD`) is used. The folds
are ours: the 5-fold x86 family holdout of `common.manifests` and the validation carve of the run seed, so a run sees the same
train, validation and test binaries as the GNN and the teammates' arms of that seed. Per fold the API vocabulary of the build, which
covers all 1,810 binaries, is made a training-set one again with the logic of `experiments/team_pipelines/train_b.py` (`remap_api`):
an API id that no training binary of the fold uses becomes `UNK_API`. The loop is the class-weighted cross entropy, AdamW, gradient
clipping, patience on the validation AUC and the two checkpoints of `tokenseq`, with her model and collate unchanged;
`--max-instructions` (default 128, as she wrote it) is the number of instructions per binary the model reads, and the run name
`kfold<f>_x86_fivechannel_s<seed>` gains `-L<n>` when it is not 128. The run directory is read by `s04_evaluation` like any other.

Tests: `python -m pytest tests/test_five_channel.py`. The model is Janice Park's (commit 1da23f6 of `origin/TokenSeqModel`).

## tokenseq

Model 1 (`tokenseq`) needs a block export (`s03c_cfg_transformer/README.md`, step 1), then:

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
