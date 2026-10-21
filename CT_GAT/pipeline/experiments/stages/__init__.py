"""The gate stages that were run, one definition file each (see `s04_evaluation/stage.py` for what a definition holds).

`python -m s04_evaluation.gates --stage <name>` loads `experiments/stages/<name>.py`; `common.py` holds the constants that
more than one stage shares. A new stage is a new file here plus its name in `STAGES`.
"""
STAGES = ('day1', 'day1b', 'v3', 'subgraph', 'paths', 'paths_codebert', 'vocab', 'imports', 'noapi', 'team', 'model1', 'model1_variants')
