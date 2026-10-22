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
twice, ordinary and cell-balanced. Missing arms print as missing.
"""
from experiments.stages.common import SEEDS, seed_rule, team_v3_parts
from s04_evaluation.stage import Stage

ARCH = 'all'
SCOPE = ('kfold_v3', ARCH)
ARMS, PAIRS, MATCH_RATE_BUILDS = team_v3_parts(ARCH)


def intro_for(arch, binaries, rule_floor, resamples):
    """The paragraph of team_v3_all (arch all) and team_v3_matched (arch matched)."""
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


def intro(resamples):
    return intro_for(ARCH, '5,652', '0.690', resamples)


STAGE = Stage('team_v3_all', intro=intro, arms=ARMS, pairs=PAIRS, seeds=SEEDS, rule=seed_rule, extra='cells',
              options={'scope': SCOPE, 'balanced': True, 'match_rate_builds': MATCH_RATE_BUILDS})
