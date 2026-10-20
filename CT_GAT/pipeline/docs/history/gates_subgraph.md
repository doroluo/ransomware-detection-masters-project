# Gate summary, stage subgraph (seed 0 only, not the pre-registered test)

kfold x86 on the v2 cache, seed 0 pooled over the five folds, 15% block masking on every arm. Differences are arm minus reference for the AUC checkpoint at its validation threshold, with the 95% paired cluster bootstrap interval (2000 resamples).

## Pooled, AUC checkpoint

| arm | group | AUC | PR-AUC | macro-F1 |
|---|---|---|---|---|
| R7 typed, whole graph | kfold_x86_gnn-mblk15 | 0.986 | 0.990 | 0.935 |
| R7 typed, cap 20k | kfold_x86_gnn-mblk15-sg-cap20k | 0.975 | 0.981 | 0.921 |
| R7 typed, cap 5k | kfold_x86_gnn-mblk15-sg-cap5k | 0.969 | 0.975 | 0.943 |
| R7 typed, lcc 20k | kfold_x86_gnn-mblk15-sg-lcc20k | 0.969 | 0.981 | 0.920 |
| R7 typed, lcc 5k | kfold_x86_gnn-mblk15-sg-lcc5k | 0.975 | 0.979 | 0.942 |
| R2 no-edges, tfidf | kfold_x86_gnn-noedges-mblk15 | 0.977 | 0.983 | 0.941 |
| R2 no-edges, raw | kfold_x86_gnn-noedges-mblk15-enc-raw | 0.972 | 0.980 | 0.935 |
| R2 no-edges, presence | kfold_x86_gnn-noedges-mblk15-enc-presence | missing (seed 0 folds present [0, 1, 2] of 0-4) |  |  |
| R2 no-edges, rawlog | kfold_x86_gnn-noedges-mblk15-enc-rawlog | missing (seed 0 folds present [] of 0-4) |  |  |

## Subgraph policies: non-inferiority against the whole graph

Reference: R7 typed, whole graph (kfold_x86_gnn-mblk15). An arm is non-inferior when the lower bound of its dAUC interval is above -0.01 and the lower bound of its dmacro-F1 interval is above -0.02.

| arm | group | dAUC [95% CI] | dmacro-F1 [95% CI] | truncated test binaries | non-inferior |
|---|---|---|---|---|---|
| R7 typed, cap 20k | kfold_x86_gnn-mblk15-sg-cap20k | -0.011 [-0.020, -0.004] | -0.013 [-0.032, +0.000] | 297 of 1810 | no |
| R7 typed, cap 5k | kfold_x86_gnn-mblk15-sg-cap5k | -0.016 [-0.033, -0.004] | +0.009 [-0.031, +0.071] | 537 of 1810 | no |
| R7 typed, lcc 20k | kfold_x86_gnn-mblk15-sg-lcc20k | -0.017 [-0.050, +0.001] | -0.014 [-0.046, +0.008] | 297 of 1810 | no |
| R7 typed, lcc 5k | kfold_x86_gnn-mblk15-sg-lcc5k | -0.011 [-0.023, -0.001] | +0.007 [-0.037, +0.069] | 537 of 1810 | no |

### kfold_x86_gnn-mblk15-sg-cap20k minus kfold_x86_gnn-mblk15 (selection auc, 2000 resamples)

a = kfold_x86_gnn-mblk15, b = kfold_x86_gnn-mblk15-sg-cap20k; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.986 | 0.975 | -0.011 | [-0.020, -0.004] |
| PR-AUC | 0.990 | 0.981 | -0.009 | [-0.021, -0.002] |
| macro-F1 | 0.935 | 0.921 | -0.013 | [-0.032, +0.000] |

### kfold_x86_gnn-mblk15-sg-cap5k minus kfold_x86_gnn-mblk15 (selection auc, 2000 resamples)

a = kfold_x86_gnn-mblk15, b = kfold_x86_gnn-mblk15-sg-cap5k; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.986 | 0.969 | -0.016 | [-0.033, -0.004] |
| PR-AUC | 0.990 | 0.975 | -0.015 | [-0.031, -0.005] |
| macro-F1 | 0.935 | 0.943 | +0.009 | [-0.031, +0.071] |

### kfold_x86_gnn-mblk15-sg-lcc20k minus kfold_x86_gnn-mblk15 (selection auc, 2000 resamples)

a = kfold_x86_gnn-mblk15, b = kfold_x86_gnn-mblk15-sg-lcc20k; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.986 | 0.969 | -0.017 | [-0.050, +0.001] |
| PR-AUC | 0.990 | 0.981 | -0.009 | [-0.028, +0.002] |
| macro-F1 | 0.935 | 0.920 | -0.014 | [-0.046, +0.008] |

### kfold_x86_gnn-mblk15-sg-lcc5k minus kfold_x86_gnn-mblk15 (selection auc, 2000 resamples)

a = kfold_x86_gnn-mblk15, b = kfold_x86_gnn-mblk15-sg-lcc5k; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.986 | 0.975 | -0.011 | [-0.023, -0.001] |
| PR-AUC | 0.990 | 0.979 | -0.011 | [-0.024, -0.002] |
| macro-F1 | 0.935 | 0.942 | +0.007 | [-0.037, +0.069] |

## Chain merge

- Chain merge (kfold x86): 52898 of 17603969 blocks in 1810 binaries could merge, a reduction of 0.30% (per binary: mean 0.35%, maximum 40.94%): below the 5% bar, so no chain-merge policy is built.

## Block-level counting weightings

No-edge arm with the raw (counts), presence (each distinct token once) and rawlog (log1p of the count) weightings against the TF-IDF no-edge arm (kfold_x86_gnn-noedges-mblk15).

| arm | group | dAUC [95% CI] | dmacro-F1 [95% CI] |
|---|---|---|---|
| R2 no-edges, raw | kfold_x86_gnn-noedges-mblk15-enc-raw | -0.004 [-0.019, +0.012] | -0.007 [-0.049, +0.045] |
| R2 no-edges, presence | kfold_x86_gnn-noedges-mblk15-enc-presence | missing | missing |
| R2 no-edges, rawlog | kfold_x86_gnn-noedges-mblk15-enc-rawlog | missing | missing |

### kfold_x86_gnn-noedges-mblk15-enc-raw minus kfold_x86_gnn-noedges-mblk15 (selection auc, 2000 resamples)

a = kfold_x86_gnn-noedges-mblk15, b = kfold_x86_gnn-noedges-mblk15-enc-raw; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.977 | 0.972 | -0.004 | [-0.019, +0.012] |
| PR-AUC | 0.983 | 0.980 | -0.004 | [-0.016, +0.008] |
| macro-F1 | 0.941 | 0.935 | -0.007 | [-0.049, +0.045] |

**R2 no-edges, presence vs kfold_x86_gnn-noedges-mblk15** (selection auc): missing: kfold_x86_gnn-noedges-mblk15-enc-presence: no complete seed among seeds [0] with all five folds (folds present: {0: [0, 1, 2]})

**R2 no-edges, rawlog vs kfold_x86_gnn-noedges-mblk15** (selection auc): missing: kfold_x86_gnn-noedges-mblk15-enc-rawlog: no complete seed among seeds [0] with all five folds

