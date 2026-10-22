"""The gate stages on main, one definition file each (the stages of the earlier models are on branch `jsonl-reference-models`) (see `s04_evaluation/stage.py` for what a definition holds).

`python -m s04_evaluation.gates --stage <name>` loads `experiments/stages/<name>.py`; `common.py` holds the constants that
more than one stage shares. A new stage is a new file here plus its name in `STAGES`.
"""
STAGES = ('team', 'model1_variants')
