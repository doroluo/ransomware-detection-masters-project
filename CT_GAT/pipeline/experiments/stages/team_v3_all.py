"""Stage team_v3_all.

Stage team_v3_all reads Models 1 to 3 on the combined dataset, `--protocol kfold_v3 --arch all`: the 5,652 binaries of the cache
`combined_v3` that have a v3 fold (x86 and x64, Mendeley and the two balanced collections), under the family-holdout v3 folds
(experiments/plans/team_v3.txt). The rows are the published ones of docs/history/gates_v3.md (TF-IDF and the GNN arms R2 no-edges, R4
shuffle and R7 typed with 15% block masking and cell-balanced selection, plain and `ab`; the plain GNN arms are not on disk and
print as missing), the team arms of the builds `b_sig500_combined_v3` and `b_callwalk500_combined_v3` (`feat` Model 2, `emb`
Model 3, `e2e` Model 3 trained end to end) and of the 250-block builds when they exist, Model 1 (`fivechannel`) at 128 and 512
instructions on the signal build, and the `-ab` twins (`--arch-balance`) of the signal-500 groups. The `cells` table below the pooled
one gives the recall of each class x architecture cell, the cell-balanced AUC and macro-F1 and the floors. Each team arm is paired
against R7 ab, R2 ab and TF-IDF ab, each `-ab` twin against its plain arm, feat against emb, emb against e2e, callwalk against
signal at the same budget and input, 250 against 500 blocks, and Model 1 at 512 against 128 instructions; every pair is printed
twice, ordinary and cell-balanced. The last rows, plain and `-ab`, are the per-block Model 1 (`fivechannel --per-block
--max-instructions 64`) and `sig500v3-m1frozen` (Model 3 on that Model 1's frozen encoder, experiments/plans/team_v3_m1frozen.txt),
paired emb -> m1frozen, m1frozen -> e2e and Model 1 L128 -> per-block Model 1. Missing arms print as missing.
"""
from experiments.stages.common import SEEDS, seed_rule, team_v3_parts
from s04_evaluation.stage import Stage

ARCH = 'all'
SCOPE = ('kfold_v3', ARCH)
ARMS, PAIRS, MATCH_RATE_BUILDS = team_v3_parts(ARCH)


def base_intro_for(arch, binaries, rule_floor, resamples):
    return (f"The combined dataset, `--protocol kfold_v3 --arch {arch}`: {binaries} binaries with a v3 fold, five-fold family "
            "holdout over both architectures and three corpora. Floors: the majority class of each fold's training manifest and "
            f"the x86 rule (ransomware if and only if x86, macro-F1 {rule_floor} pooled); the cell-balanced columns weight each "
            "class x architecture cell equally, under which the x86 rule scores 0.5. The published GNN rows (R2 no-edges, R4 "
            "shuffle, R7 typed; `ab` = trained with --arch-balance) are block-masked (15%) and select their AUC checkpoint by the "
            "validation cell-balanced AUC and its threshold by the validation cell-balanced macro-F1 (`selcell`); TF-IDF takes its "
            "validation macro-F1 maximiser per fold. The team arms (`team sig500v3`, `team callwalk500`, ..., Model 1) keep the "
            "teammates' selection, ransomware F1 at 0.5 for one checkpoint and the validation AUC for the other, so a difference "
            "between a team row and a published GNN row mixes the model with the selection rule. The team builds are made with "
            "`--max-instructions 4000000`, which leaves 34 Balanced_Goodware x64 goodware binaries (more than four million "
            "instructions each) out of every split: they appear as excluded under each pair, and a pair is read on the binaries "
            "both arms have, while the published rows include them. The pooled table is seed 0; the paired differences, second "
            "minus first for the AUC checkpoint at its validation threshold, use the seeds complete in both arms (queue seeds "
            f"{list(SEEDS)}), with the 95% paired cluster bootstrap interval ({resamples} resamples). The rule column is the "
            "pre-registered minimum effect of PLAN_GNN.md section 5 (dAUC >= 0.01 and dmacro-F1 >= 0.02, both intervals "
            "excluding zero, 3 seeds x 5 folds); the published GNN and TF-IDF rows have seed 0 only, so a pair against them is a "
            "direction, not the test. A missing arm or fold is stated.")


def intro_for(arch, binaries, rule_floor, resamples):
    """The paragraphs of team_v3_all (arch all) and team_v3_matched (arch matched)."""
    return base_intro_for(arch, binaries, rule_floor, resamples) + '\n\n' + (
        "`team sig500v3 m1frozen` is Model 3 on a frozen Model 1 (Yanping's PR #32, "
        "experiments/plans/team_v3_m1frozen.txt): the per-block Model 1 of the same fold, seed, dataset, weighting and "
        "build (`fivechannel --per-block --max-instructions 64`, the row `Model 1 five-channel L64 per block`: it encodes"
        " the first 64 instructions of every block with Janice's network, averages the block vectors and classifies) "
        "encodes each block once, its weights never change, and only Dorothy's GAT trains on the block vectors; the arm "
        "exits unless the Model 1 trained on the arm's build, since API ids are numbered per build. It is paired against "
        "`emb` (the pooled embedding in its place) and `e2e` (what training the encoder adds), and the per-block Model 1 "
        "against Model 1 at 128 instructions (the first instructions of every block against those of the binary).")


def intro(resamples):
    return intro_for(ARCH, '5,652', '0.690', resamples)


STAGE = Stage('team_v3_all', intro=intro, arms=ARMS, pairs=PAIRS, seeds=SEEDS, rule=seed_rule, extra='cells',
              options={'scope': SCOPE, 'balanced': True, 'match_rate_builds': MATCH_RATE_BUILDS})
