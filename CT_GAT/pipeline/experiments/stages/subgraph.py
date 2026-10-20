"""Stage subgraph.

Stage subgraph reads the kfold x86 arms of step 11: R7 typed with 15% block masking under the subgraph policy cap
at 20k and 5k blocks (the lcc rows were removed with the policy in step 25; the old lcc runs stay on disk, unread) against the whole-graph `gnn-mblk15` reference (non-inferiority: the lower bound of the paired
95% interval of dAUC above -0.01 and of dmacro-F1 above -0.02, AUC selection), the chain-merge measurement of
RUNS/subgraph_chain.json, and the no-edge arm with the raw, presence and rawlog block weightings against
`gnn-noedges-mblk15`.
"""
from s04_evaluation.stage import Stage, variant_of

REFERENCE = ('R7 typed, whole graph', variant_of(mask_blocks=0.15))
NOEDGES = variant_of(control='no_edges', mask_blocks=0.15)
ARMS = tuple((f'R7 typed, {policy} {budget // 1000}k', variant_of(mask_blocks=0.15, subgraph=policy, budget=budget))
             for policy in ('cap',) for budget in (20000, 5000))
WEIGHTINGS = tuple((f'R2 no-edges, {enc}', variant_of(control='no_edges', mask_blocks=0.15, encoder=enc))
                   for enc in ('raw', 'presence', 'rawlog'))
CHAIN_BAR = 0.05


def intro(resamples):
    return ('kfold x86 on the v2 cache, seed 0 pooled over the five folds, 15% block masking on every arm. Differences '
            f'are arm minus reference for the AUC checkpoint at its validation threshold, with the 95% paired cluster '
            f'bootstrap interval ({resamples} resamples).')


STAGE = Stage('subgraph', layout='subgraph', intro=intro, arms=ARMS,
              options={'reference': REFERENCE, 'noedges': NOEDGES, 'weightings': WEIGHTINGS, 'chain_bar': CHAIN_BAR})
