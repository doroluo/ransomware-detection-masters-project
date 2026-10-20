"""Stage imports.

Stage imports reads the kfold x86 header-import arms: the header-imports LR (`baselines.py --imports`), R7 typed and R7 + `--imports
header` (`gnn-mblk15-imph`, queue stage `imports_header`): R7 against R7 + header imports, the header-imports LR against R7 +
header imports, and TF-IDF against R7 + header imports. The unreferenced-import arms and their gate were removed in step 25.
"""
from s04_evaluation.stage import Stage, variant_of, x86_group

R7 = dict(mask_blocks=0.15)
ARMS = (('TF-IDF', 'tfidf'), ('header imports, LR', 'imports-header'),
        ('R7 typed', x86_group(variant_of(**R7))),
        ('R7 + header imports', x86_group(variant_of(**R7, imports='header'))))
PAIRS = (('R7 typed', 'R7 + header imports'), ('header imports, LR', 'R7 + header imports'),
         ('TF-IDF', 'R7 + header imports'))


def intro(resamples):
    return ('kfold x86 on the v2 cache, seed 0 pooled over the five folds. Baselines (`baselines.py --imports`, TF-IDF '
            'logistic regression on the same folds): every header import identity. GNN arms: R7 typed with 15% block masking, '
            'with and without `--imports header` (an EmbeddingBag over the header import names and two log counts in '
            'the head; queue stage `imports_header`). Differences are second minus first '
            f'for the AUC checkpoint at its validation threshold, with the 95% paired cluster bootstrap interval ({resamples} '
            'resamples). A missing arm or fold is stated.')


STAGE = Stage('imports', intro=intro, arms=ARMS, pairs=PAIRS)
