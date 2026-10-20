"""Stage noapi.

Stage noapi reads the api-token ablation (kfold x86, `--vocab noapi`: the blocks hold mn, sig and beh tokens and no api token;
the header-import features are a separate switch and stay off): TF-IDF (sig + api), TF-IDF noapi (`baselines.py --vocab noapi`,
mn + sig + beh), the header-imports LR, R7 typed `gnn-mblk15` and R7 noapi `gnn-mblk15-voc-noapi`. Pairs: R7 against R7 noapi and
TF-IDF against TF-IDF noapi (what the api tokens carry), R7 noapi against TF-IDF noapi and against the header-imports LR (is
anything left once the imports are out of the tokens). The paired differences use the seeds 0 1 2 complete in both arms; the
baselines exist for seed 0 only, so those pairs carry one seed and the rule column says "seed 0 only: a direction". Missing arms
print as missing.
"""
from experiments.stages.common import PATHS_X86
from s04_evaluation.stage import MIN_AUC, MIN_F1, Stage, beats, variant_of, x86_group

SEEDS = (0, 1, 2)
ARMS = (('TF-IDF', 'tfidf'), ('TF-IDF noapi', 'tfidf-noapi'), ('header-imports LR', 'imports-header'),
        ('R7 typed', x86_group(variant_of(**PATHS_X86))),
        ('R7 noapi', x86_group(variant_of(**PATHS_X86, vocab='noapi'))))
ABLATIONS = (('R7 typed', 'R7 noapi'), ('TF-IDF', 'TF-IDF noapi'))
PAIRS = ABLATIONS + (('TF-IDF noapi', 'R7 noapi'), ('header-imports LR', 'R7 noapi'))


def rule(first, second, result, info):
    """The rule column. An ablation pair (api tokens in, then out): "api loss shown" when the arm without api tokens is
    worse by the pre-registered minimum effect (dAUC <= -0.01 and dmacro-F1 <= -0.02, both upper bounds below zero). Any
    other pair: `beats` of the second over the first. One or two complete seeds: a direction, not the test."""
    if (first, second) in ABLATIONS:
        lost = all(result[k]['diff'] <= -m and result[k]['hi'] < 0 for k, m in (('auc', MIN_AUC), ('macro_f1', MIN_F1)))
        word = 'api loss shown' if lost else 'api loss not shown'
    else:
        word = 'beats' if beats(result) else 'does not beat'
    seeds = info['seeds']
    if len(seeds) == len(SEEDS):
        return word
    return f'{word} (seed {seeds[0]} only: a direction, not the test)' if len(seeds) == 1 else (
        f'{word} (seeds {seeds} only: a direction, not the test)')


def intro(resamples):
    return ('kfold x86 on the v2 cache, 15% block masking on every GNN arm. R7 noapi is `--vocab noapi`: each block holds its '
            'mn, sig and beh tokens and no api token (`gnn-mblk15-voc-noapi`); the header-import features are not used by any '
            'arm. TF-IDF is the default sig + api logistic regression; TF-IDF noapi is `baselines.py --vocab noapi` over '
            'mn + sig + beh, the tokens of R7 noapi. The pooled table is seed 0; the paired differences, second minus first '
            f'for the AUC checkpoint at its validation threshold, use the seeds complete in both arms (queue seeds '
            f'{list(SEEDS)}), with the 95% paired cluster bootstrap interval ({resamples} resamples). The rule column '
            f'is the pre-registered rule (dAUC >= {MIN_AUC} and dmacro-F1 >= {MIN_F1}, both 95% intervals excluding zero) in '
            'the direction named: "api loss shown" for the two ablation pairs, "beats" for the others. TF-IDF and the '
            'header-imports LR exist for seed 0 only, so their pairs are a direction. A missing arm or fold is stated.')


STAGE = Stage('noapi', intro=intro, arms=ARMS, pairs=PAIRS, seeds=SEEDS, rule=rule)
