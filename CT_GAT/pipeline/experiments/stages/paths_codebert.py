"""Stage paths_codebert.

Stage paths_codebert reads the CodeBERT arms of queue stage `paths_codebert` (frozen `microsoft/codebert-base` on the vocabulary
strings of each block, `ours_cap0_codebert`; seeds 0 1 2): path 1 (no-edge GNN on the vectors) and path 2 (typed GNN on them)
against each other, path 1 against the token-counting no-edge arm `gnn-noedges-mblk15` (R2 tokens), path 2 against R7 typed
`gnn-mblk15`, and both against TF-IDF. The pooled table is seed 0; the paired differences use the seeds complete in both arms.
An arm that is not on disk is reported as missing.
"""
from experiments.stages.common import CB_VEC, CB_VEC_ARMS, PATHS_X86
from s04_evaluation.stage import Stage, variant_of, x86_group

SEEDS = (0, 1, 2)
ARMS = (('TF-IDF', 'tfidf'),
        ('R2 no-edges, tokens', x86_group(variant_of(**PATHS_X86, control='no_edges'))),
        ('R7 typed, tokens', x86_group(variant_of(**PATHS_X86))),
        ('path 1: no-edges, CodeBERT vectors', x86_group(variant_of(**CB_VEC_ARMS, control='no_edges'))),
        ('path 2: typed, CodeBERT vectors', x86_group(variant_of(**CB_VEC_ARMS))))
PAIRS = (('path 1: no-edges, CodeBERT vectors', 'path 2: typed, CodeBERT vectors'),
         ('R2 no-edges, tokens', 'path 1: no-edges, CodeBERT vectors'),
         ('R7 typed, tokens', 'path 2: typed, CodeBERT vectors'),
         ('TF-IDF', 'path 1: no-edges, CodeBERT vectors'), ('TF-IDF', 'path 2: typed, CodeBERT vectors'))


def intro(resamples):
    return ('kfold x86 on the v2 cache, 15% block masking on every GNN arm. Paths 1 and 2 train on the frozen CodeBERT vectors '
            f'`{CB_VEC}`: `microsoft/codebert-base` on the text of each block, built from the signature and API '
            'strings of our block export and never from operands (`transformer_paths/codebert_encoder.py`); path 1 has no '
            'edges, path 2 has the typed edges. The pooled table is seed 0; the paired differences, second minus first for the AUC '
            f'checkpoint at its validation threshold, are over the seeds complete in both arms (queue seeds {list(SEEDS)}), '
            f'with the 95% paired cluster bootstrap interval ({resamples} resamples). A missing arm or fold is stated.')


STAGE = Stage('paths_codebert', intro=intro, arms=ARMS, pairs=PAIRS, seeds=SEEDS)
