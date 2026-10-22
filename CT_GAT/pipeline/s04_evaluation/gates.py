"""Gate summary from the runs on disk, one stage at a time.

    python -m s04_evaluation.gates --stage <name> [--resamples 2000]

A stage is a data file, `experiments/stages/<name>.py`, that defines `STAGE` (see `stage.py`): the arms it reads, the pairs it
compares, the seeds, the rule column and the intro text. The stages that were run are listed in
`experiments/stages/__init__.py`, and each file's docstring says what the stage reads. A new stage is a new file there (and
its name in that list); this engine and `gate_tables.py` do not change. An arm that is not on disk, or has
missing folds, is printed as missing with the folds that are present; nothing is dropped silently.

The runs are read from `common.paths.RUNS` (`GT_RUNS`), the combined cache of the kfold_v3 floors from `datasets.json`.
"""
import argparse
import importlib

from common.manifests import run_command
from s04_evaluation.gate_tables import EXTRAS, match_rate_lines, paired_stage


def stage_names():
    return tuple(importlib.import_module('experiments.stages').STAGES)


def load_stage(name):
    """The `STAGE` of experiments/stages/<name>.py."""
    if name not in stage_names():
        raise ValueError(f'unknown stage {name!r}; the stages are {", ".join(stage_names())}')
    stage = importlib.import_module(f'experiments.stages.{name}').STAGE
    if stage.name != name:
        raise ValueError(f'experiments/stages/{name}.py defines the stage {stage.name!r}')
    return stage


def render(stage, resamples=2000):
    """The markdown lines of a stage."""
    after = list(EXTRAS[stage.extra](stage.arms)) if stage.extra else []
    after += match_rate_lines(stage.options.get('match_rate_builds', ()))
    return paired_stage(stage.name, stage.intro_text(resamples), stage.arms, stage.pairs, resamples, stage.seeds,
                        after_pooled=after, rule=stage.rule)


def run_stage(name, resamples=2000):
    return render(load_stage(name), resamples)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Gate summary of one stage (experiments/stages/<name>.py) from the runs on disk.')
    ap.add_argument('--stage', choices=stage_names(), required=True)
    ap.add_argument('--resamples', type=int, default=2000)
    args = ap.parse_args(argv)
    print('\n'.join(run_stage(args.stage, args.resamples)))


if __name__ == '__main__':
    run_command(main)
