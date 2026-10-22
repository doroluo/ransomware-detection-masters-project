"""Stage team.

Stage team reads the teammates' GAT pipelines run under our folds (experiments/team_pipelines, and the queue plans
`team` and `paths_codebert`): TF-IDF, the header-imports LR, R7 and R2 with 15% block masking, our matched cap-500 arms (typed and
no-edges, `--subgraph cap --budget 500 --tag b500`), team gnn500, cb500, sig500, sig500v2, the three sig500v3 arms (`feat` is
Model 2, `emb` Model 3), the three uncapped sigwhole arms and main's five-channel Model 1 on the uncapped build
(`--model fivechannel --max-instructions 512 --tag whole`), and path 2 on the CodeBERT vectors. The pooled seed-0 table adds accuracy,
precision, recall and the confusion matrix of every arm (ransomware = positive class, AUC checkpoint at its validation threshold)
and a line per v3 build with its match rates (`options['match_rate_builds']`: the builds and their folders); each team arm is
paired against TF-IDF, R7 and our b500 typed arm, cb500 against path 2 (same vectors, their GAT against our GNN), sig500v2 (the
rebuild with Yanping's streaming preprocess) against sig500, R7 and TF-IDF, sig500v3 against sig500v2 and sig500, R7 and our b500
no-edges arm (v3 pairs are read on the binaries both arms have), feat against emb, each sigwhole arm against its sig500v3
counterpart and against R7, sigwhole emb against feat, and our b500 typed arm against R7 (what the 500-block cap costs). The
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


FIVECHANNEL_WHOLE = RunConfig(**dict(X86, model='fivechannel'), max_instructions=512, tag='whole').group    # kfold_x86_fivechannel-L512_whole
ARMS = (('TF-IDF', 'tfidf'), ('header-imports LR', 'imports-header'),
        ('R7 typed, whole graph', x86_group(variant_of(**PATHS_X86))),
        ('R2 no-edges, whole graph', x86_group(variant_of(**PATHS_X86, control='no_edges'))),
        ('ours b500 typed', RunConfig(**CAP).group),
        ('ours b500 no-edges', RunConfig(**CAP, control='no_edges').group),
        ('team gnn500', team_group('gnn500')), ('team cb500', team_group('cb500')),
        ('team sig500', team_group('sig500')), ('team sig500v2', team_group('sig500v2')),
        ('team sig500v3', team_group('sig500v3')),
        ('team sig500v3 feat (Model 2)', team_group('sig500v3-feat')),
        ('team sig500v3 emb (Model 3)', team_group('sig500v3-emb')),
        ('team sigwhole', team_group('sigwhole')),
        ('team sigwhole feat (Model 2)', team_group('sigwhole-feat')),
        ('team sigwhole emb (Model 3)', team_group('sigwhole-emb')),
        ('fivechannel-L512 whole', FIVECHANNEL_WHOLE),
        ('path 2: typed, CodeBERT vectors', dict(paths_codebert.ARMS)['path 2: typed, CodeBERT vectors']))
V2 = 'team sig500v2'
V3 = ('team sig500v3', 'team sig500v3 feat (Model 2)', 'team sig500v3 emb (Model 3)')
WHOLE = ('team sigwhole', 'team sigwhole feat (Model 2)', 'team sigwhole emb (Model 3)')
WHOLE_ROWS = WHOLE + ('fivechannel-L512 whole',)
MATCH_RATE_BUILDS = (('b_sig500_v3', 'team/b_sig500_v3'), ('b_sigwhole_v3', 'team/b_sigwhole_v3'))   # (name, folder under the run directory)
TEAM_OF = tuple(a for a, _ in ARMS if a.startswith('team ') and a != V2 and a not in V3 + WHOLE)
COST = ('R7 typed, whole graph', 'ours b500 typed')
PAIRS = (tuple(('TF-IDF', a) for a in TEAM_OF) + tuple(('R7 typed, whole graph', a) for a in TEAM_OF)
         + tuple(('ours b500 typed', a) for a in TEAM_OF)
         + (('path 2: typed, CodeBERT vectors', 'team cb500'), COST)
         + (('team sig500', V2), ('R7 typed, whole graph', V2), ('TF-IDF', V2))
         + ((V2, V3[0]), ('team sig500', V3[0]))
         + tuple(('R7 typed, whole graph', a) for a in V3) + tuple(('ours b500 no-edges', a) for a in V3)
         + ((V3[1], V3[2]),)
         + tuple(zip(V3, WHOLE)) + tuple(('R7 typed, whole graph', a) for a in WHOLE_ROWS)
         + ((WHOLE[1], WHOLE[2]),))


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
            "sig500v2 is read on the binaries both have), `team sig500v3` the same trainer on a third build with the `endp` and "
            "library-function fixes (source main-fix, 8a971b7's s02 pins; `b_sig500_v3`), whose two ablations feed the GAT only the "
            "77 handcrafted block features (`feat`, Model 2: block features + CFG) or only the 64-d pooled channel embedding "
            "(`emb`, Model 3: block embeddings + CFG); every v3 pair is read on the binaries both arms have, all trained on "
            "our folds and validation split. `team sigwhole`, `sigwhole feat` and `sigwhole emb` are the same three arms on the uncapped "
            "rebuild (`b_sigwhole_v3`, every non-library block of every binary, trained with a per-batch node budget), paired against "
            "their 500-block v3 counterparts and against R7; `fivechannel-L512 whole` is Janice's five-channel Model 1 (main's code, "
            "512 instructions) on the same uncapped build. The pooled table is seed 0; "
            f"the paired differences, second minus first for the AUC checkpoint at its validation threshold, use the seeds "
            f"complete in both arms (queue seeds {list(SEEDS)}), with the 95% paired cluster bootstrap interval "
            f"({resamples} resamples). Pre-registered rule (PLAN_GNN.md section 5): a second arm beats the first when dAUC >= "
            f"{MIN_AUC} and dmacro-F1 >= {MIN_F1} pooled, both 95% intervals excluding zero, over 3 seeds x 5 folds; TF-IDF "
            "and the header-imports LR exist for seed 0 only, so the pairs against TF-IDF carry one seed and the word in the "
            f"rule column is a direction. The cost of the 500-block cap (R7 whole graph against ours b500 typed) is read "
            f"as non-inferiority: lower bound of dAUC above -{NI_AUC} and of dmacro-F1 above -{NI_F1}. A missing arm or fold "
            "is stated.")


STAGE = Stage('team', intro=intro, arms=ARMS, pairs=PAIRS, seeds=SEEDS, rule=rule, extra='operating_points',
              options={'match_rate_builds': MATCH_RATE_BUILDS})
