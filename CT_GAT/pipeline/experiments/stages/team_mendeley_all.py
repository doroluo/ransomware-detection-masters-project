"""Stage team_mendeley_all.

Stage team_mendeley_all reads Models 1 to 3 on the whole Mendeley dataset: `--protocol kfold --arch all`, the 2,411 binaries (1,810
x86 and 601 x64) of the `mendeley_x86` cache, seed 0 pooled and the seeds 0 1 2 paired (experiments/plans/team_mendeley_all.txt).
The rows are TF-IDF on the same folds (`kfold{k}_all_s0`, which have no validation threshold: its AUC-checkpoint row prints as
missing); the team arms of the five builds `RUNS/team/b_<family>_mendeley_all`
(`sig500v3`, `sig250`, `callwalk500` and `callwalk250`, each with `feat` Model 2, `emb` Model 3 and `e2e` Model 3 trained end to end,
and the three arms `sigwhole`, `sigwhole-feat` and `sigwhole-emb` on the uncapped build); Model 1 (`fivechannel`) at 128 and 512
instructions on the 500-block signal build and at 512 on the uncapped one; and the `-ab` twins (`--arch-balance`) of the three
`sig500v3` inputs and of the two Model 1 groups of the signal build. Below the pooled table a `cells` table gives the recall of each
class x architecture cell, the cell-balanced AUC and macro-F1 and the floors. Each arm is paired against TF-IDF; each `-ab` twin
against its plain arm; callwalk against signal at the same budget and input (the sampler); 250 against 500 blocks (the budget);
feat against emb and emb against e2e; each uncapped arm against its 500-block one (the `sigwhole` arm has none: `sig500v3` without
a suffix is not run on this dataset) and Model 1 at 512 against 128 instructions and against the uncapped build. Every pair is
printed twice, ordinary and cell-balanced. Missing arms print as missing.
"""
from experiments.stages.common import (AB, BUILDS, INPUTS, MODEL1_LENGTHS, SEEDS, budget_pairs, input_pairs, model1_rows,
                                       sampler_pairs, seed_rule, team_label, team_rows, whole_rows)
from s04_evaluation.stage import Stage

PROTOCOL, ARCH = 'kfold', 'all'
SCOPE = (PROTOCOL, ARCH)
SIGNAL_500 = ('sig500v3',)
MODEL1 = model1_rows(PROTOCOL, ARCH)                                           # L128, L512 on b_sig500_mendeley_all; L512 whole on the uncapped
MODEL1_TWINS = model1_rows(PROTOCOL, ARCH, MODEL1_LENGTHS[:2], arch_balance=True)
TEAM = team_rows(PROTOCOL, ARCH, BUILDS)
TEAM_TWINS = team_rows(PROTOCOL, ARCH, SIGNAL_500, arch_balance=True)
WHOLE = whole_rows(PROTOCOL, ARCH)
ARMS = ((('TF-IDF', 'tfidf'),) + TEAM + WHOLE + MODEL1 + TEAM_TWINS + MODEL1_TWINS)
PLAIN = TEAM + WHOLE + MODEL1
TWINS = TEAM_TWINS + MODEL1_TWINS
MATCH_RATE_BUILDS = tuple((f'b_{family}_mendeley_all', f'team/b_{family}_mendeley_all')
                          for family in ('sig500', 'sig250', 'callwalk500', 'callwalk250', 'sigwhole'))
L128, L512, L512_WHOLE = (label for label, _ in MODEL1)

UNCAPPED_OF = tuple((team_label('sig500v3', name), label)                # feat -> sigwhole feat, emb -> sigwhole emb
                    for (_, name), (label, _) in zip(INPUTS[:2], WHOLE[1:]))
PAIRS = (tuple(('TF-IDF', label) for label, _ in PLAIN + TWINS)
         + tuple((label.removesuffix(AB), label) for label, _ in TWINS)
         + sampler_pairs() + budget_pairs()
         + input_pairs(BUILDS) + ((WHOLE[1][0], WHOLE[2][0]),)
         + UNCAPPED_OF + ((L128, L512), (L512, L512_WHOLE)))


def intro(resamples):
    return ("The whole Mendeley dataset, `--protocol kfold --arch all`: 2,411 binaries, 1,810 x86 and 601 x64, five-fold family "
            "holdout. On Mendeley the architecture is a label proxy: x86 has 1,134 ransomware and 676 goodware binaries, x64 "
            "has 108 and 493. The x86 rule (ransomware if and only if x86) alone scores 0.650 macro-F1 pooled over the folds, and 55% of "
            "the x64 ransomware are in fold 2, so a fold's x64 test set can be almost all goodware or hold most of the x64 "
            "ransomware. A model therefore gains AUC on this dataset by scoring x64 binaries as goodware. Read the recall of ransomware x64 first, "
            "then the cell-balanced AUC and macro-F1 (each class x architecture cell weighted equally; the x86 rule scores 0.5) "
            "and the `-ab` twin of an arm, which weights its training binaries by cell. `combined_v3` (x86-rule floor 0.690 on all "
            "architectures, 0.606 on the matched subset, 741 x64 ransomware) is the fair all-architecture test (stages "
            "`team_v3_all`, `team_v3_matched`); this stage reports the whole Mendeley dataset because the team asked for it, with "
            "the floor in the same table. The team arms train on the builds `b_<family>_mendeley_all`, each the x86 build of the "
            "same name extended with the 601 x64 binaries, with the teammates' recipe (class-weighted loss, checkpoint chosen by "
            "ransomware F1 at 0.5 and by validation AUC); `-ab` weights each training binary by N / (C n_cell). The pooled table "
            "is seed 0; the paired differences, second minus first for the AUC checkpoint at its validation threshold, use the "
            f"seeds complete in both arms (queue seeds {list(SEEDS)}), with the 95% paired cluster bootstrap interval "
            f"({resamples} resamples), once with every binary weighted equally and once cell-balanced. The rule column is the "
            "pre-registered minimum effect of PLAN_GNN.md section 5 (dAUC >= 0.01 and dmacro-F1 >= 0.02, both intervals "
            "excluding zero, 3 seeds x 5 folds); TF-IDF exists for seed 0 only, so a pair against it is a direction. A TF-IDF run "
            "made before the validation threshold was recorded has none to score the AUC checkpoint with (the `kfold{k}_all_s0` "
            "runs are of that kind), so its row and its pairs print as missing. A missing arm or fold is stated.")


STAGE = Stage('team_mendeley_all', intro=intro, arms=ARMS, pairs=PAIRS, seeds=SEEDS, rule=seed_rule, extra='cells',
              options={'scope': SCOPE, 'balanced': True, 'match_rate_builds': MATCH_RATE_BUILDS})
