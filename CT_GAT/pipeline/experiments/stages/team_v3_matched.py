"""Stage team_v3_matched.

Stage team_v3_matched is stage team_v3_all on `--protocol kfold_v3 --arch matched`: the 4,850 binaries of the 5,652 that are in
`s02_preprocess/folds/v3/arch_matched.csv`, a subset in which the class mix of the two architectures is closer (x86-rule floor 0.606
macro-F1 against 0.690 on all). A matched run reads the `all` build `b_<family>_combined_v3`; the manifest is the filter. The rows,
pairs and tables are those of team_v3_all with the groups `kfold_v3_matched_...`; the published rows are the matched ones of
docs/history/gates_v3.md.
"""
from experiments.stages.common import SEEDS, seed_rule, team_v3_parts
from experiments.stages.team_v3_all import intro_for
from s04_evaluation.stage import Stage

ARCH = 'matched'
SCOPE = ('kfold_v3', ARCH)
ARMS, PAIRS, MATCH_RATE_BUILDS = team_v3_parts(ARCH)


def intro(resamples):
    return intro_for(ARCH, '4,850', '0.606', resamples)


STAGE = Stage('team_v3_matched', intro=intro, arms=ARMS, pairs=PAIRS, seeds=SEEDS, rule=seed_rule, extra='cells',
              options={'scope': SCOPE, 'balanced': True, 'match_rate_builds': MATCH_RATE_BUILDS})
