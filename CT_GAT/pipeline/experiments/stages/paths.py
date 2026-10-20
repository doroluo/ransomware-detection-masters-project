"""Stage paths.

Stage paths reads the kfold x86 arms of step 12: R7 with the out-of-fold TF-IDF logit (`gnn-stack-mblk15`) against the
`gnn-mblk15` reference and against the TF-IDF baseline, and the block-vector arms (path 1: no-edge GNN on precomputed
block vectors, path 2: typed GNN on them) against each other and against the token-counting no-edge arm
`gnn-noedges-mblk15` (R2 tokens). The vectors are the reference encoder's until the teammates' checkpoints arrive. An
arm that is not on disk, or has fewer than five folds (the reference arms run on fold 0 only), is reported as missing.
"""
from experiments.stages.common import PATHS_VEC, PATHS_X86, REF_VEC
from s04_evaluation.stage import Stage, variant_of, x86_group

ARMS = (('TF-IDF', 'tfidf'),
        ('R7 typed, tokens', x86_group(variant_of(**PATHS_X86))),
        ('R7 + TF-IDF stack', x86_group(variant_of(**PATHS_X86, stack_tfidf=True))),
        ('R2 no-edges, tokens', x86_group(variant_of(**PATHS_X86, control='no_edges'))),
        ('path 1: no-edges, block vectors', x86_group(variant_of(**PATHS_VEC, control='no_edges'))),
        ('path 2: typed, block vectors', x86_group(variant_of(**PATHS_VEC))))
PAIRS = (('R7 typed, tokens', 'R7 + TF-IDF stack'), ('TF-IDF', 'R7 + TF-IDF stack'),
         ('path 1: no-edges, block vectors', 'path 2: typed, block vectors'),
         ('R2 no-edges, tokens', 'path 1: no-edges, block vectors'))


def intro(resamples):
    return ('kfold x86 on the v2 cache, seed 0 pooled over the five folds, 15% block masking on every GNN arm. Stack: R7 '
            'with the standardised out-of-fold TF-IDF logit as an extra head input (`--stack-tfidf`). Paths 1 and 2 train '
            f'on the frozen block vectors `{REF_VEC}` of the reference encoder (random weights, a plumbing check, run on '
            'fold 0 only until the teammates deliver checkpoints); path 1 has no edges, path 2 has the typed edges. Differences '
            f'are second minus first for the AUC checkpoint at its validation threshold, with the 95% paired cluster '
            f'bootstrap interval ({resamples} resamples). A missing arm or fold is stated.')


STAGE = Stage('paths', intro=intro, arms=ARMS, pairs=PAIRS)
