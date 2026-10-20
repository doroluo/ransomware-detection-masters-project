"""Stage v3.

Stage v3 reads seed 0 of the kfold_v3 block-masked arms (R2 no-edges, R4 shuffle, R7 typed with 15% block masking and
cell-balanced checkpoint selection, `-mblk15-selcell`) and the R0 TF-IDF baseline on the combined cache, for arch all
and arch matched: the pooled table under both checkpoint rules with per-architecture recall at the validation threshold,
cell-balanced AUC and macro-F1 (each class x arch cell weighted equally) and both floors (majority, x86 rule), the
arch-balanced (`-ab`) arms beside their unweighted twins, the paired AUC-selection differences R2-R7, R4-R7 and
R0-R2 (ordinary and cell-balanced metrics, for the plain and for the `-ab` arms), the cell-balanced difference of each
`-ab` arm against its unweighted twin, and one-line verdicts per arch on whether R7 beats R2 and R4 under the
pre-registered rule.
"""
from s04_evaluation.stage import Stage, variant_of

ARCHES = ('all', 'matched')
MASKED = dict(mask_blocks=0.15, select_on='cell')
GNN = (('R2 no-edges', dict(control='no_edges')), ('R4 shuffle', dict(control='shuffle')), ('R7 typed', {}))
ARMS = (('R0 TF-IDF', 'tfidf'),) + tuple((arm, variant_of(**kw, **MASKED)) for arm, kw in GNN)
AB_ARMS = (('R0 TF-IDF ab', 'tfidf-ab'),) + tuple(
    (f'{arm} ab', variant_of(**kw, **MASKED, arch_balance=True)) for arm, kw in GNN)
PAIRS = (('R2 no-edges', 'R7 typed'), ('R4 shuffle', 'R7 typed'), ('R0 TF-IDF', 'R2 no-edges'))
AB_PAIRS = tuple((f'{a} ab', f'{b} ab') for a, b in PAIRS)
TWINS = tuple((arm, f'{arm} ab') for arm, _ in ARMS)

INTRO = ('Combined cache, kfold_v3 (family-holdout v3 folds 0-4), seed 0 pooled over the five folds. Every GNN arm is '
         'block-masked (15% of the blocks in training, `mblk15`). Its AUC checkpoint is chosen by the validation '
         'cell-balanced AUC and its threshold by the validation cell-balanced macro-F1 (`selcell`); the F1 checkpoint '
         'rule is unchanged. The F1 checkpoint is scored at 0.5; the AUC checkpoint at its validation threshold '
         "(TF-IDF: the validation macro-F1 maximiser, per fold); recalls are at that threshold. A missing arm or fold "
         "is stated in its row. Floors: the majority "
         "class of each fold's training manifest, and the x86 rule (ransomware iff x86). `ab` arms were trained with "
         "--arch-balance (each class x arch cell weighted equally). Cell-bal columns weight each class x arch cell "
         "of the pooled test set equally; under them the x86 rule scores 0.5.")

STAGE = Stage('v3', layout='v3', intro=INTRO, arms=ARMS, pairs=PAIRS,
              options={'arches': ARCHES, 'ab_arms': AB_ARMS, 'ab_pairs': AB_PAIRS, 'twins': TWINS})
