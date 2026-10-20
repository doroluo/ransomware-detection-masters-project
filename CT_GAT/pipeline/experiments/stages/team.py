"""Stage team.

Stage team reads the teammates' GAT pipelines run under our folds (experiments/team_pipelines; `scripts/team_runs.sh` and
`scripts/team_gpu_runs.sh`): TF-IDF, the header-imports LR, R7 and R2 with 15% block masking, our matched cap-500 arms (typed and
no-edges, `--subgraph cap --budget 500 --tag b500`), team gnn500, cb500, sig500 and sig500v2 (`team-<arm>` tags) and path 2 on the CodeBERT
vectors. The pooled seed-0 table adds accuracy, precision, recall and the confusion matrix of every arm (ransomware = positive
class, AUC checkpoint at its validation threshold); each team arm is paired against TF-IDF, R7 and our b500 typed arm, cb500
against path 2 (same vectors, their GAT against our GNN), sig500v2 (the rebuild with Yanping's streaming preprocess) against
sig500, R7 and TF-IDF, and our b500 typed arm against R7 (what the 500-block cap costs). The
paired differences use the seeds 0 1 2 complete in both arms and carry the pre-registered rule of PLAN_GNN.md section 5.
Missing arms print as missing.
"""
from common.config import RunConfig
from experiments.stages import paths_codebert
from experiments.stages.common import CB_VEC, PATHS_X86
from s04_evaluation.stage import MIN_AUC, MIN_F1, NI_AUC, NI_F1, Stage, beats, non_inferior, variant_of, x86_group

X86 = dict(protocol='kfold', arch='x86', fold=0, model='gnn')
SEEDS = (0, 1, 2)
CAP = dict(X86, mask_blocks=0.15, subgraph='cap', budget=500, tag='b500')


def team_group(arm):
    """RunConfig.group of a team arm (`kfold_x86_gnn_team-<arm>`), the one experiments/team_pipelines/common.team_config names."""
    return RunConfig(**X86, tag=f'team-{arm}').group


ARMS = (('TF-IDF', 'tfidf'), ('header-imports LR', 'imports-header'),
        ('R7 typed, whole graph', x86_group(variant_of(**PATHS_X86))),
        ('R2 no-edges, whole graph', x86_group(variant_of(**PATHS_X86, control='no_edges'))),
        ('ours b500 typed', RunConfig(**CAP).group),
        ('ours b500 no-edges', RunConfig(**CAP, control='no_edges').group),
        ('team gnn500', team_group('gnn500')), ('team cb500', team_group('cb500')),
        ('team sig500', team_group('sig500')), ('team sig500v2', team_group('sig500v2')),
        ('path 2: typed, CodeBERT vectors', dict(paths_codebert.ARMS)['path 2: typed, CodeBERT vectors']))
V2 = 'team sig500v2'
TEAM_OF = tuple(a for a, _ in ARMS if a.startswith('team ') and a != V2)
COST = ('R7 typed, whole graph', 'ours b500 typed')
PAIRS = (tuple(('TF-IDF', a) for a in TEAM_OF) + tuple(('R7 typed, whole graph', a) for a in TEAM_OF)
         + tuple(('ours b500 typed', a) for a in TEAM_OF)
         + (('path 2: typed, CodeBERT vectors', 'team cb500'), COST)
         + (('team sig500', V2), ('R7 typed, whole graph', V2), ('TF-IDF', V2)))


def rule(first, second, result, info):
    """The rule column. Our cap-500 arm against the whole graph: non-inferiority of stage subgraph (lower bounds above
    -0.01 dAUC and -0.02 dmacro-F1). Every other pair: the pre-registered minimum effect (`beats`, 3 seeds); with fewer seeds
    in both arms (TF-IDF has seed 0 only) the word is a direction, not the test."""
    seeds = info['seeds']
    if (first, second) == COST:
        return 'non-inferior' if non_inferior(result) else 'not shown non-inferior'
    word = 'beats' if beats(result) else 'does not beat'
    return word if len(seeds) == len(SEEDS) else f'{word} (seeds {seeds} only: a direction, not the test)'


def intro(resamples):
    return ("kfold x86 on the v2 cache, 15% block masking on every GNN arm of ours. `ours b500` is `--subgraph cap --budget 500 "
            "--tag b500` (whole functions ranked non-library, API references, size, up to 500 blocks per binary: the budget "
            "the teammates' pipelines use), typed and with `--control no_edges`. `team gnn500` is the teammates' BlockGAT "
            "on their 57-d opcode / API-group / structure features, `team cb500` the same GAT on the CodeBERT vectors "
            f"`{CB_VEC}` of the same 500 blocks, `team sig500` their edge-typed GAT on the 141-d subgraph features "
            "(`experiments/team_pipelines`), `team sig500v2` the same trainer on a rebuild of those subgraphs with Yanping's "
            "streaming `preprocess_full` (main c7899f0, including the two listings sig500 had to leave out; sig500 against "
            "sig500v2 is read on the binaries both have), all trained on our folds and validation split. The pooled table is seed 0; "
            f"the paired differences, second minus first for the AUC checkpoint at its validation threshold, use the seeds "
            f"complete in both arms (queue seeds {list(SEEDS)}), with the 95% paired cluster bootstrap interval "
            f"({resamples} resamples). Pre-registered rule (PLAN_GNN.md section 5): a second arm beats the first when dAUC >= "
            f"{MIN_AUC} and dmacro-F1 >= {MIN_F1} pooled, both 95% intervals excluding zero, over 3 seeds x 5 folds; TF-IDF "
            "and the header-imports LR exist for seed 0 only, so the pairs against TF-IDF carry one seed and the word in the "
            f"rule column is a direction. The cost of the 500-block cap (R7 whole graph against ours b500 typed) is read "
            f"as non-inferiority: lower bound of dAUC above -{NI_AUC} and of dmacro-F1 above -{NI_F1}. A missing arm or fold "
            "is stated.")


STAGE = Stage('team', intro=intro, arms=ARMS, pairs=PAIRS, seeds=SEEDS, rule=rule, extra='operating_points')
