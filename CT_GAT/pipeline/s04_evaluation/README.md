# Stage 4: evaluation

Every model of the pipeline (Model 1 `s03a_transformer/token_sequence`, Model 2 `s03b_cfg_feature`, Model 3 `s03c_cfg_transformer`,
the baselines, the teammates' pipelines in `experiments/team_pipelines`) writes the same run directory
(`common/run_io.py`: `config.json`, `manifest.json`, `predictions.csv`, `predictions_auc.csv`, `metrics.json`), and this stage reads
those directories and nothing else. A model joins the tables by writing that directory under a `RunConfig.name`; no code here
changes.

Every command runs from `CT_GAT/pipeline`, on the runs of `GT_RUNS` (`common/paths.py`):

```bash
python -m s04_evaluation.baselines --protocol kfold --fold 0 1 2 3 4 --arch x86 --seed 0     # TF-IDF and structural baselines
python -m s04_evaluation.baselines --protocol kfold --fold 0 1 2 3 4 --imports               # logistic regression on the header imports
python -m s04_evaluation.report                                                              # RESULTS.md in the run directory
python -m s04_evaluation.val_threshold                                                       # re-score at the validation threshold
python -m s04_evaluation.stats --a kfold_x86_gnn-noedges --b kfold_x86_gnn --selection auc   # one paired comparison
python -m s04_evaluation.gates --stage v3                                                    # a whole gate summary
```

| module | what it does |
|---|---|
| `metrics.py` | accuracy, precision, recall, F1, macro-F1, AUC, PR-AUC, the class x architecture "cell" weights, the majority and x86-rule floors |
| `val_threshold.py` | the threshold grid and the validation macro-F1 maximiser every AUC-checkpoint result is scored at; `oof_rule`, the out-of-fold threshold of `--threshold-from oof` |
| `baselines.py` | TF-IDF over the block tokens (and the same over another vocabulary, `--vocab`), the structural features, the header-imports logistic regression (`--imports`), the out-of-fold TF-IDF logits that `--stack-tfidf` reads (`--stack-logits`); writes `<runs>/baselines/<dir>/` |
| `report.py` | `RESULTS.md`: one row per configuration, mean +- sd over seeds of the pooled out-of-fold test metrics |
| `stats.py` | the paired cluster bootstrap between two arms (clusters are ransomware families and single goodware binaries) |
| `stage.py`, `gates.py`, `gate_tables.py`, `gate_layouts.py` | a gate stage is data (`experiments/stages/<name>.py`: arms, pairs, seeds, rule, intro); the engine lays out the pooled table and the paired differences |

The import methods (`baselines --imports`, `experiments.train --imports header`) read `<cache>/side/imports/`. Without the
team's reader output (`GT_READER_ROOT`) the arrays come from the CFG export itself, with the Mendeley data configured as for
`build_cache` (`GT_BUNDLE`, or `GT_MENDELEY_COLLECTIONS` for the two collections):

```bash
python -m s03b_cfg_feature.gnn.s10_unused_import_features.import_side --cache <cache dir> --from-cfg
```

This reads only the header record of each sample's `.jsonl.gz` (`pe.imports`). Its `header_imports` and `n_entries` equal the
reader's for all 2,411 samples of `graph_transformer_cache_v2`, and `baselines --protocol kfold --fold 0 1 2 3 4 --imports`
on such a cache gives the published `imports_header_lr` predictions of `kfold<k>_x86_s0_imp`, row for row (that method reads
`header_imports` only). `referenced_imports` is the header identities that the cache's instructions name, and the cache leaves
out the orphan instructions the reader also scans, so `unreferenced_imports` has more entries than the reader's in 194 of the
2,411 samples. The GNN `--imports header` bag reads `header_imports` too, but its two extra inputs are log1p(`unreferenced_imports`)
and log1p(`header_imports`): for those 194 samples the first differs from the reader-built value, so a `-imph` model trained
or evaluated on `--from-cfg` arrays is not bit-identical to the published `-imph` runs.

How to read a gate table, and how to add a stage: [`docs/ADDING_AN_EXPERIMENT.md`](../docs/ADDING_AN_EXPERIMENT.md). The final tables
of the stages that were run are in [`docs/history/`](../docs/history/README.md). The queue of `experiments/` ends its plans with
`report` and `gates <stage>`.

Tests: `python -m pytest tests/test_eval_*.py` (the synthetic run directories of `tests/eval_synth.py`; the tiny cache for the baselines).
