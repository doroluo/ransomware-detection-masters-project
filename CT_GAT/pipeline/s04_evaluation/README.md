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
| `val_threshold.py` | the threshold grid and the validation macro-F1 maximiser every AUC-checkpoint result is scored at |
| `baselines.py` | TF-IDF over the block tokens (and the same over another vocabulary, `--vocab`), the structural features, the header-imports logistic regression (`--imports`), the out-of-fold TF-IDF logits that `--stack-tfidf` reads (`--stack-logits`); writes `<runs>/baselines/<dir>/` |
| `report.py` | `RESULTS.md`: one row per configuration, mean +- sd over seeds of the pooled out-of-fold test metrics |
| `stats.py` | the paired cluster bootstrap between two arms (clusters are ransomware families and single goodware binaries) |
| `stage.py`, `gates.py`, `gate_tables.py`, `gate_layouts.py` | a gate stage is data (`experiments/stages/<name>.py`: arms, pairs, seeds, rule, intro); the engine lays out the pooled table and the paired differences |

How to read a gate table, and how to add a stage: [`docs/ADDING_AN_EXPERIMENT.md`](../docs/ADDING_AN_EXPERIMENT.md). The final tables
of the stages that were run are in [`docs/history/`](../docs/history/README.md). The queue of `experiments/` ends its plans with
`report` and `gates <stage>`.

Tests: `python -m pytest tests/test_eval_*.py` (the synthetic run directories of `tests/eval_synth.py`; the tiny cache for the baselines).
