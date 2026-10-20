"""Stage vocab.

Stage vocab reads the kfold x86 vocabulary arms of step 14: the no-edge GNN with 15% block masking under our vocabulary
(`gnn-noedges-mblk15`), the reader's (`-voc-yanping`) and the hybrid one (`-voc-hybrid`), and the three TF-IDF baselines over
the same token kinds (`baselines.py --vocab`): the pooled table and the paired differences. Missing arms print as missing.
"""
from s04_evaluation.stage import Stage, variant_of, x86_group

GNN = dict(mask_blocks=0.15, control='no_edges')
ARMS = (('TF-IDF, ours (sig + api)', 'tfidf'), ('TF-IDF, yanping', 'tfidf-yanping'), ('TF-IDF, hybrid', 'tfidf-hybrid'),
        ('R2 no-edges, ours', x86_group(variant_of(**GNN))),
        ('R2 no-edges, yanping', x86_group(variant_of(**GNN, vocab='yanping'))),
        ('R2 no-edges, hybrid', x86_group(variant_of(**GNN, vocab='hybrid'))))
PAIRS = (('TF-IDF, ours (sig + api)', 'TF-IDF, yanping'), ('TF-IDF, ours (sig + api)', 'TF-IDF, hybrid'),
         ('R2 no-edges, ours', 'R2 no-edges, yanping'), ('R2 no-edges, ours', 'R2 no-edges, hybrid'),
         ('R2 no-edges, yanping', 'R2 no-edges, hybrid'), ('TF-IDF, yanping', 'R2 no-edges, yanping'),
         ('TF-IDF, hybrid', 'R2 no-edges, hybrid'))


def intro(resamples):
    return ('kfold x86 on the v2 cache, seed 0 pooled over the five folds. Vocabularies: ours (mn, sig, beh, api), yanping '
            '(the reader\'s three ids per instruction plus api) and hybrid (our mn and beh, the reader\'s two operand ids, '
            'api). GNN arms are the no-edge arm with 15% block masking (`--control no_edges --mask-blocks 0.15 --vocab`); '
            'the TF-IDF rows are `baselines.py --vocab` over the same token kinds. Differences are second minus first '
            f'for the AUC checkpoint at its validation threshold, with the 95% paired cluster bootstrap interval ({resamples} '
            'resamples). A missing arm or fold is stated.')


STAGE = Stage('vocab', intro=intro, arms=ARMS, pairs=PAIRS)
