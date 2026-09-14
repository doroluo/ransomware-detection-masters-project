# results/cnn_vit

The yanping CNN-ViT (`HierarchicalMalwareNet` from `model_train.py`) scored on
the shared cohort and the shared split, so its numbers describe the same
samples as the tokenization and EMBER pipelines.

**Start with [`summary.md`](summary.md)** - both datasets, overall, per-arch and
per-family tables, what the cohort split changed, and the extractor caveat.

Written by `cnn_vit_pipeline/train_eval.py`:

```
results/cnn_vit/<dataset>/metrics.json          # expA schema + per_architecture + per_family_recall + per-epoch history
results/cnn_vit/<dataset>/splits.csv            # sha256, file, source, label, group, fold, split, arch, family
results/cnn_vit/<dataset>/training_log.csv      # epoch, lr, train_loss, val_loss, val_acc, seconds
results/cnn_vit/<dataset>/test_predictions.csv  # per-sample y_true / y_pred / score_ransomware
results/cnn_vit/<dataset>/config_used.yaml      # the configuration the run used
```

`<dataset>` is `mendeley` or `balanced`.

Written by `cnn_vit_pipeline/tuned_train.py` (the tuning harness, selection by
group CV on the train split only):

```
results/cnn_vit/tuned/<dataset>/cv_search.csv        # every configuration tried, its CV metrics and its cost
results/cnn_vit/tuned/<dataset>/metrics.json         # chosen config, 5 seeds; shared schema + macro_f1_sd + n_seeds
results/cnn_vit/tuned/<dataset>/predictions.csv      # all 5 seeds stacked, `seed` column
results/cnn_vit/tuned/<dataset>/config_used.yaml     # the configuration and the paths it used
results/cnn_vit/tuned/<dataset>/seeds/seed<N>/       # per-seed metrics.json, predictions.csv, training_log.csv
results/cnn_vit/tuned/<dataset>/baseline/            # the BASE configuration, 5 seeds - the "before"
results/cnn_vit/tuned/<dataset>/posthoc/rank<N>/     # top-3 CV configs, test-scored AFTER selection closed
```

`posthoc/` chose nothing; it exists to measure the CV -> test gap. See
[`summary.md` section 9](summary.md) - the short version is that tuning bought
nothing on `mendeley` and did not lift `balanced` above chance within each
architecture.

No model weights live here - they are binary. Checkpoints go to
`C:/Users/chaoa/Downloads/cnn_vit_models/<dataset>/best_model.pth`.

`expB_splits_snapshot.csv` is not a result. It is a frozen copy of
`results/expB/splits.csv`, taken once at the start of this work because the
tokenization worker was editing that file concurrently. The balanced dataset's
goodware membership is read from this snapshot, so the split stays stable even
if the live file moves. `cnn_vit_pipeline/cohort.py` prefers the snapshot and
falls back to the live file.

## Caveat carried by both runs

These images were rendered by the unmodified `asm_parser.py` from the **revised
(skip-data) extractor's** disassembly, re-shaped by `asm_tool/unified_to_asm.py`,
not from `asm_parse.py`'s linear sweep that the original CNN-ViT pipeline used.
The `asm_parse.py`-format ransomware trees do not exist on this host and the
host Mendeley-goodware tree's UPX files are still packed, so their sha256s do
not match the cohort. Both classes and both corpora go through the same
extractor here, so these numbers are internally comparable and comparable with
the other cohort-faithful pipelines - but not with an older CNN-ViT figure
built from `asm_parse.py` output. See `asm_tool/README.md` section 6 and
`summary.md` section 5.
