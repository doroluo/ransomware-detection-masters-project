# Stage 4: evaluation

Every model of the pipeline (Models 1 to 3 in `s03a_transformer`, `s03b_cfg_feature` and `s03c_cfg_transformer`, the teammates'
pipelines in `experiments/team_pipelines`) writes the same run directory
(`common/run_io.py`: `config.json`, `manifest.json`, `predictions.csv`, `predictions_auc.csv`, `metrics.json`), and this stage reads
those directories and nothing else. A model joins the tables by writing that directory under a `RunConfig.name`; no code here
changes.

Every command runs from `CT_GAT/pipeline`, on the runs of `GT_RUNS` (`common/paths.py`):

```bash
python -m s04_evaluation.report                                                              # RESULTS.md in the run directory
python -m s04_evaluation.val_threshold                                                       # re-score at the validation threshold
python -m s04_evaluation.stats --a kfold_x86_fivechannel --b kfold_x86_fivechannel-L512 --selection auc   # one paired comparison
python -m s04_evaluation.gates --stage team                                                  # a whole gate summary
```

| module | what it does |
|---|---|
| `metrics.py` | accuracy, precision, recall, F1, macro-F1, AUC, PR-AUC, the class x architecture "cell" weights, the majority and x86-rule floors |
| `val_threshold.py` | the threshold grid and the validation macro-F1 maximiser every AUC-checkpoint result is scored at; `oof_rule`, the out-of-fold threshold of `--threshold-from oof` |
| `report.py` | `RESULTS.md`: one row per configuration, mean +- sd over seeds of the pooled out-of-fold test metrics |
| `stats.py` | the paired cluster bootstrap between two arms (clusters are ransomware families and single goodware binaries) |
| `stage.py`, `gates.py`, `gate_tables.py` | a gate stage is data (`experiments/stages/<name>.py`: arms, pairs, seeds, rule, intro); the engine lays out the pooled table and the paired differences |

The TF-IDF and import-table baselines are on branch `jsonl-reference-models`; their published numbers are in `docs/history/`. `report.py`, `stats.py` and the
gate stages read their published runs from `<runs>/baselines/<dir>/` (`metrics.json` and `predictions.csv`), which they need no
baseline code for.

How to read a gate table, and how to add a stage: [`docs/ADDING_AN_EXPERIMENT.md`](../docs/ADDING_AN_EXPERIMENT.md). The final tables
of the stages that were run are in [`docs/history/`](../docs/history/README.md). The queue of `experiments/` ends its plans with
`report` and `gates <stage>`.

Tests: `python -m pytest tests/test_eval_*.py` (the synthetic run directories of `tests/eval_synth.py`).
