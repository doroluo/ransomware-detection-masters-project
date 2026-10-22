"""The gate stages on main, one definition file each (the stages of the earlier models are on branch `jsonl-reference-models`) (see `s04_evaluation/stage.py` for what a definition holds).

`python -m s04_evaluation.gates --stage <name>` loads `experiments/stages/<name>.py`; `common.py` holds the constants that
more than one stage shares. A new stage is a new file here plus its name in `STAGES`. `team` and `model1_variants` read kfold x86;
`team_mendeley_all` reads the whole Mendeley dataset (kfold, arch all) and `team_v3_all` and `team_v3_matched` the combined dataset
(kfold_v3, arch all and matched): they set the `scope` and `balanced` options of the stage.
"""
STAGES = ('team', 'model1_variants', 'team_mendeley_all', 'team_v3_all', 'team_v3_matched')
