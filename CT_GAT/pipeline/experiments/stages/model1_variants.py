"""Stage model1_variants.

Stage model1_variants reads the variants of Model 1 (plan `model1_variants`) against the references and against each other, kfold x86:
TF-IDF, the header-imports LR, R7 typed `gnn-mblk15`, R2 no-edges `gnn-noedges-mblk15`, Model 1 `tokenseq`, and

  * `tokenseq-thoof`: Model 1 with the decision threshold chosen on out-of-fold predictions of the training binaries
    (`--threshold-from oof`) instead of the validation families, which saturate;
  * `fivechannel`, `fivechannel-L512`: Janice Park's five-channel Model 1 on the `.pt` files of a pipeline-B build, 128 and 512
    instructions per binary (`s03a_transformer/five_channel`);
  * team sig500v3: the teammates' edge-typed GAT on the third rebuild of the pipeline-B subgraphs, when its runs exist.

Each variant is paired with `tokenseq` and with TF-IDF (second minus first, AUC checkpoint at its validation threshold) over the seeds
0 1 2 complete in both arms; the rule column applies the pre-registered rule (dAUC >= 0.01 and dmacro-F1 >= 0.02, both 95% intervals
excluding zero) in the direction "the variant beats the reference". TF-IDF has seed 0 only, so its pairs are a direction. The pooled table
is seed 0. Missing arms print as missing. The rows of TF-IDF, the header-imports LR, R7, R2 and `tokenseq` (and
`tokenseq-thoof`) read published runs from the run directory and need no model code; the code that wrote them is on branch
`jsonl-reference-models`.
"""
from common.config import RunConfig
from experiments.stages.common import PATHS_X86
from experiments.stages.team import team_group
from s04_evaluation.stage import MIN_AUC, MIN_F1, Stage, beats, variant_of, x86_group

SEEDS = (0, 1, 2)
KFOLD = dict(protocol='kfold', fold=0, arch='x86')
TOKENSEQ, TOKENSEQ_OOF = 'Model 1 tokenseq', 'Model 1 tokenseq, out-of-fold threshold'
FIVE, FIVE_512 = 'Model 1 five-channel L128', 'Model 1 five-channel L512'
ARMS = (('TF-IDF', 'tfidf'), ('header-imports LR', 'imports-header'),
        ('R7 typed', x86_group(variant_of(**PATHS_X86))),
        ('R2 no-edges', x86_group(variant_of(**PATHS_X86, control='no_edges'))),
        (TOKENSEQ, RunConfig(**KFOLD, model='tokenseq').group),
        (TOKENSEQ_OOF, RunConfig(**KFOLD, model='tokenseq', threshold_from='oof').group),
        (FIVE, RunConfig(**KFOLD, model='fivechannel').group),
        (FIVE_512, RunConfig(**KFOLD, model='fivechannel', max_instructions=512).group),
        ('team sig500v3', team_group('sig500v3')))
VARIANTS = (TOKENSEQ_OOF, FIVE, FIVE_512)
PAIRS = tuple((ref, variant) for variant in VARIANTS for ref in (TOKENSEQ, 'TF-IDF'))


def rule(first, second, result, info):
    """The pre-registered minimum effect (`beats`) of the variant over the reference. One or two complete seeds (TF-IDF has seed 0
    only): a direction, not the test."""
    word = 'beats' if beats(result) else 'does not beat'
    seeds = info['seeds']
    if len(seeds) == len(SEEDS):
        return word
    return f'{word} (seed {seeds[0]} only: a direction, not the test)' if len(seeds) == 1 else (
        f'{word} (seeds {seeds} only: a direction, not the test)')


def intro(resamples):
    return ('kfold x86 on the v2 cache. `tokenseq` is Model 1, the token-sequence transformer over the block export; '
            '`tokenseq-thoof` is the same model with the decision threshold chosen on out-of-fold predictions of the training '
            'binaries (3 refits on folds that keep a ransomware family or a goodware identity group whole, '
            '`--threshold-from oof`) instead of the validation families, which saturate; the two differ only in the threshold of the AUC '
            'checkpoint (the seed and the training are the same, so the ranking is the same and the AUC is level up to run-to-run noise). '
            '`five-channel` is Janice Park\'s Model 1 over the five-channel instruction tuples (opcode, two operands, API, behaviour) of '
            'a pipeline-B build, its first 128 (as she wrote it) or 512 instructions per binary, trained on our folds and validation '
            'split with the API vocabulary made a training-set one per fold. R7 typed and R2 no-edges carry 15% block masking. The '
            'pooled table is seed 0; the paired differences, second minus first for the AUC checkpoint at its validation threshold, use '
            f'the seeds complete in both arms (queue seeds {list(SEEDS)}), with the 95% paired cluster bootstrap interval '
            f'({resamples} resamples). The rule column is the pre-registered rule (dAUC >= {MIN_AUC} and dmacro-F1 >= {MIN_F1}, both 95% '
            'intervals excluding zero) read as "the variant beats the reference". TF-IDF exists for seed 0 only, so its pairs are a '
            'direction. A missing arm or fold is stated.')


STAGE = Stage('model1_variants', intro=intro, arms=ARMS, pairs=PAIRS, seeds=SEEDS, rule=rule)
