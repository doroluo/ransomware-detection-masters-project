# Stage 3a: Model 1, the transformer over the token sequence

`five_channel/` is Janice Park's Model 1 (model `fivechannel`, registered in `common/registry.py`, so that the one training command
`python -m experiments.train --model fivechannel` runs it): her `FiveChannelEmbedder`, `Model1TokenSequence`, `Model1Dataset` and
`model1_collate_fn` in `five_channel/model.py` (the file she wrote as `token_sequence.py` in this folder; it moved with `git mv`
and only its imports changed), trained by `five_channel/train.py` on
our folds. Its input is the folder of `<sample_id>.pt` files of a pipeline-B build (Yanping's stage-2 output,
`experiments/team_pipelines/build_b.py`), five token ids per instruction: opcode, two operands, API and behaviour tag.

Every command runs from `CT_GAT/pipeline`, which is the package root (there is no `pipeline` package above it: the files import
`common`, `s02_preprocess`, ... directly, as the tests do).

## fivechannel

```bash
python -m experiments.train --model fivechannel --protocol kfold --arch x86 --fold 0 --seed 0 --build <build folder>   # L128
python -m experiments.train --model fivechannel --protocol kfold --arch x86 --fold 0 --seed 0 --max-instructions 512   # -L512
python -m experiments.train --model fivechannel --protocol kfold --fold 0 --smoke 8 --device cpu                       # 2 epochs
python -m experiments.train --model fivechannel --protocol kfold --arch all --fold 0 --seed 0 --build RUNS/team/b_sig500_mendeley_all
python -m experiments.train --model fivechannel --protocol kfold_v3 --arch matched --arch-balance --fold 0 --seed 0 --build RUNS/team/b_sig500_combined_v3
python -m experiments.queue --plan experiments/plans/model1_variants.txt                                                # the variants
```

The flags that pick the data are `--protocol`, `--arch`, `--arch-balance` and `--build`:

| flags | binaries | build | run name |
|---|---|---|---|
| `--protocol kfold --arch x86` (the defaults) | 1,810 Mendeley x86 | `Mendeley_x86_...` zip, `CT_GAT/processed/cfg_subgraphs` | `kfold<f>_x86_fivechannel[-L<n>]_s<seed>` |
| `--protocol kfold --arch all` | 2,411 Mendeley, x86 and x64 | `Mendeley_all_...` zip, `RUNS/team/b_sig500_mendeley_all` | `kfold<f>_all_fivechannel[-L<n>][-ab]_s<seed>` |
| `--protocol kfold_v3 --arch all` | 5,652 combined, with a v3 fold | `combined_v3_...` zip, `RUNS/team/b_sig500_combined_v3` | `v3kfold<f>_all_fivechannel[-L<n>][-ab]_s<seed>` |
| `--protocol kfold_v3 --arch matched` | 4,850 of those in `arch_matched.csv` | the same build | `v3kfold<f>_matched_fivechannel[-L<n>][-ab]_s<seed>` |

The flag must be the literal `--protocol`: the CFG cache is resolved from it. A run refuses a build made for another dataset (an
`all` build serves `all` and `matched` runs, an x86 build serves x86 runs only), and a build that left binaries out on purpose
(`--max-instructions` of `build_b`) has them in no split. `--arch-balance` (needs `--arch all` or `matched`; run name suffix `-ab`)
weights each training binary's loss by `N / (C n_cell)` over the non-empty (class, architecture) cells of the training manifest in place
of `N / (2 n_class)`, and averages the losses as a weighted mean; `metrics.json` records the cell weights. `--max-instructions`,
`--tag` (`whole` for the uncapped build) and `--threshold-from` are as for kfold x86. On the whole Mendeley dataset the architecture
predicts the label (the x86 rule scores 0.650 macro-F1), so read the recall of ransomware x64 and the cell-balanced metrics of
`gates --stage team_mendeley_all`; the pipeline README gives the numbers.

`--build <dir>` is the build folder; without it the `team_b_build` source is used: `GT_TEAM_B_BUILD`, else `datasets.json`, else
`CT_GAT/processed/cfg_subgraphs`. A missing or empty folder stops the run with the two ways to fill it. The folds
are ours: the 5-fold family holdout of `common.manifests` for the dataset of the flags (x86 by default) and the validation carve of
the run seed, so a run sees the same train, validation and test binaries as the GNN and the teammates' arms of that seed. Per fold the API vocabulary of the build, which
covers all 1,810 binaries, is made a training-set one again with the logic of `experiments/team_pipelines/train_b.py` (`remap_api`):
an API id that no training binary of the fold uses becomes `UNK_API`. The loop is the class-weighted cross entropy, AdamW, gradient
clipping, patience on the validation AUC and the two checkpoints, with her model and collate unchanged;
`--max-instructions` (default 128, as she wrote it) is the number of instructions per binary the model reads, and the run name
`kfold<f>_x86_fivechannel_s<seed>` gains `-L<n>` when it is not 128. The run directory is read by `s04_evaluation` like any other.

The build folder, with no configuration, is `CT_GAT/processed/cfg_subgraphs`, and it is filled in one of two ways:

* The fast way: unzip `Mendeley_x86_cfg_subgraphs_signal_500blocks_v3.zip` (the Drive preprocessed set, 500 blocks) at `CT_GAT/`. It
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

The Model 1 trainer picks the threshold of the AUC checkpoint (the one `predictions_auc.csv` and the gates use) on the validation families. They
saturate: the validation AUC is near 1.0 within a few epochs, every threshold of a wide band has the same validation macro-F1, the
first of the band is taken, and the test macro-F1 swings between folds (0.77 to 0.93). `--threshold-from oof` (run name suffix
`-thoof`; the default `val` changes no name and no result) chooses it on predictions the model did not train on:

1. the training binaries are split in 3 folds that keep a ransomware family or a goodware identity group whole
   (`common.manifests.grouped_folds`);
2. a new model is fitted on 2 folds for the epoch count of each checkpoint (`best_epoch` for the F1 checkpoint, `auc_best_epoch` for
   the AUC checkpoint; one refit serves both, with no early stopping) and predicts the held-out fold with every instruction
   it reads; the API vocabulary of a refit is that of its own training binaries;
3. the GRID threshold (`s04_evaluation/val_threshold.py`) with the highest macro-F1 on the pooled held-out predictions is the threshold.

The cost is three more fits of two thirds of the training binaries per run. `metrics.json` keeps the rule (`threshold_rule`: k, the
epochs, the thresholds, their out-of-fold macro-F1 and the validation thresholds the default would have used, and the text of
`selection.auc.rule`). The rule is
`experiments.train.choose_thresholds`; a model that gives it a `fit_predict` can use it.

