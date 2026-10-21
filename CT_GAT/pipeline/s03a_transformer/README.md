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

`--build <dir>` is the build folder; without it the `team_b_build` source is used: `GT_TEAM_B_BUILD`, else `datasets.json`, else
`CT_GAT/processed/cfg_subgraphs`. A missing or empty folder stops the run with the two ways to fill it. The folds
are ours: the 5-fold x86 family holdout of `common.manifests` and the validation carve of the run seed, so a run sees the same
train, validation and test binaries as the GNN and the teammates' arms of that seed. Per fold the API vocabulary of the build, which
covers all 1,810 binaries, is made a training-set one again with the logic of `experiments/team_pipelines/train_b.py` (`remap_api`):
an API id that no training binary of the fold uses becomes `UNK_API`. The loop is the class-weighted cross entropy, AdamW, gradient
clipping, patience on the validation AUC and the two checkpoints of `tokenseq`, with her model and collate unchanged;
`--max-instructions` (default 128, as she wrote it) is the number of instructions per binary the model reads, and the run name
`kfold<f>_x86_fivechannel_s<seed>` gains `-L<n>` when it is not 128. The run directory is read by `s04_evaluation` like any other.

The build folder, with no configuration, is `CT_GAT/processed/cfg_subgraphs`, and it is filled in one of two ways:

* The fast way: unzip `Mendeley_x86_cfg_subgraphs_500blocks_v3.zip` (the Drive preprocessed set, 500 blocks) at `CT_GAT/`. It
  fills `CT_GAT/processed/` (`cfg_subgraphs/<sample_id>.pt`, `channel_embedder.pt`, `build_info/`, and `api_vocab.json` one level up);
  the folder is gitignored. With it and the CFG cache of the zero-config setup, `--model fivechannel --protocol kfold --fold 0
  --seed 0 --device cpu --smoke 8` runs with no other flag. The zip's `api_vocab.json` is not read (it is not inside the build
  folder): the API embedding takes its size from the largest API id the fold's binaries use, which for the 1,810 binaries is the
  size of that vocabulary (12,406 names), so a full run has the 933,122 parameters of the published `kfold0_x86_fivechannel_s0`.
* The exact way, from scratch, about 4 h on the CPU for the 1,810 binaries (7.6 s per binary, measured on the development machine; resumable): `python -m experiments.team_pipelines.build_b
  --source repo --sample signal --num-blocks 500 --seed 0` over the unzipped Mendeley collections and the CFG cache (build it first with
  `python -m s02_preprocess.cache.build_cache --cache mendeley_x86`). It runs the `s02_preprocess` scripts of this repository and
  writes the same `.pt` files as the published build (checked on 60 binaries: every tensor equal, see
  `experiments/team_pipelines/README.md`). It writes into an empty `CT_GAT/processed/cfg_subgraphs`, not into an unzipped one: the
  folder holds one build only. It does not write `CT_GAT/processed/api_vocab.json`, which only the untrained demo at the end of
  `five_channel/model.py` reads; `train.py` takes the vocabulary of the build folder (`api_vocab.json` inside it, which `build_b` writes).

Yanping's own `s02_preprocess/preprocess_full.py` is a different path (train rows first, the test split frozen, the 21 UPX-stub
x86 rows included) and writes `CT_GAT/processed/` itself; its output is not the published `.pt` set.

Tests: `python -m pytest tests/test_five_channel.py`. The model is Janice Park's (commit 1da23f6 of `origin/TokenSeqModel`).

## The decision threshold: `--threshold-from`

Both models pick the threshold of the AUC checkpoint (the one `predictions_auc.csv` and the gates use) on the validation families. They
saturate: the validation AUC is near 1.0 within a few epochs, every threshold of a wide band has the same validation macro-F1, the
first of the band is taken, and the test macro-F1 swings between folds (0.77 to 0.93). `--threshold-from oof` (run name suffix
`-thoof`; the default `val` changes no name and no result) chooses it on predictions the model did not train on:

1. the training binaries are split in 3 folds that keep a ransomware family or a goodware identity group whole
   (`common.manifests.grouped_folds`);
2. a new model is fitted on 2 folds for the epoch count of each checkpoint (`best_epoch` for the F1 checkpoint, `auc_best_epoch` for
   the AUC checkpoint; one refit serves both, with no early stopping) and predicts the held-out fold with every window or instruction
   it reads; for `fivechannel` the API vocabulary of a refit is that of its own training binaries;
3. the GRID threshold (`s04_evaluation/val_threshold.py`) with the highest macro-F1 on the pooled held-out predictions is the threshold.

The cost is three more fits of two thirds of the training binaries per run. `metrics.json` keeps the rule (`threshold_rule`: k, the
epochs, the thresholds, their out-of-fold macro-F1 and the validation thresholds the default would have used, and the text of
`selection.auc.rule`). The `tokenseq` refits keep the fold's token vocabulary, which was fitted on all the training binaries. The rule is
`experiments.train.choose_thresholds`; a model that gives it a `fit_predict` can use it.


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
