"""Stage model1.

Stage model1 reads Model 1, the token-sequence transformer over the block export (`token_sequence/`, `kfold<f>_x86_tokenseq_s<seed>`),
against four references: TF-IDF, the header-imports LR, R7 typed `gnn-mblk15` and R2 no-edges `gnn-noedges-mblk15`. The pooled table is
seed 0; Model 1 is paired with each reference (second minus first, AUC checkpoint at its validation threshold) over the seeds 0 1 2
complete in both arms, and the rule column applies the pre-registered rule (dAUC >= 0.01 and dmacro-F1 >= 0.02, both 95% intervals
excluding zero) in the direction "Model 1 beats the reference". The baselines exist for seed 0 only, so those pairs are a direction.
Missing arms print as missing.
"""
from common.config import RunConfig
from experiments.stages.common import PATHS_X86
from s04_evaluation.stage import MIN_AUC, MIN_F1, Stage, beats, variant_of, x86_group

SEEDS = (0, 1, 2)
GROUP = RunConfig(protocol='kfold', fold=0, arch='x86', model='tokenseq').group
ARMS = (('TF-IDF', 'tfidf'), ('header-imports LR', 'imports-header'),
        ('R7 typed', x86_group(variant_of(**PATHS_X86))),
        ('R2 no-edges', x86_group(variant_of(**PATHS_X86, control='no_edges'))),
        ('Model 1 tokenseq', GROUP))
PAIRS = tuple((ref, 'Model 1 tokenseq') for ref, _ in ARMS[:-1])


def rule(first, second, result, info):
    """The rule column: the pre-registered minimum effect (`beats`) of Model 1 over the reference. One or two complete seeds
    (the baselines have seed 0 only): a direction, not the test."""
    word = 'beats' if beats(result) else 'does not beat'
    seeds = info['seeds']
    if len(seeds) == len(SEEDS):
        return word
    return f'{word} (seed {seeds[0]} only: a direction, not the test)' if len(seeds) == 1 else (
        f'{word} (seeds {seeds} only: a direction, not the test)')


def intro(resamples):
    return ('kfold x86 on the v2 cache. Model 1 is the token-sequence transformer (`token_sequence/`): every instruction and API '
            'reference of a binary becomes one position (E[mn] + E[sig] + E[beh], or E[api]), the stream is cut into windows of '
            '512 positions (at most 64 random windows per binary per step in training, all of them in evaluation), each window is '
            'encoded by a two-layer transformer and the window vectors are averaged per binary. It reads the block export '
            '`ours_cap0` in the fold vocabulary and has no graph; its validation split is that of the export (val_seed 0). R7 typed and R2 '
            'no-edges carry 15% block masking. The pooled table is seed 0; the paired differences, second minus first for the AUC '
            f'checkpoint at its validation threshold, use the seeds complete in both arms (queue seeds {list(SEEDS)}), with '
            f'the 95% paired cluster bootstrap interval ({resamples} resamples). The rule column is the pre-registered rule '
            f'(dAUC >= {MIN_AUC} and dmacro-F1 >= {MIN_F1}, both 95% intervals excluding zero) read as "Model 1 beats the reference". '
            'TF-IDF and the header-imports LR exist for seed 0 only, so their pairs are a direction. A missing arm or fold is stated.')


STAGE = Stage('model1', intro=intro, arms=ARMS, pairs=PAIRS, seeds=SEEDS, rule=rule)
