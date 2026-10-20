# Gate summary, stage model1 (seed 0 only, not the pre-registered test)

kfold x86 on the v2 cache. Model 1 is the token-sequence transformer (`token_sequence/`): every instruction and API reference of a binary becomes one position (E[mn] + E[sig] + E[beh], or E[api]), the stream is cut into windows of 512 positions (at most 64 random windows per binary per step in training, all of them in evaluation), each window is encoded by a two-layer transformer and the window vectors are averaged per binary. It reads the block export `ours_cap0` in the fold vocabulary and has no graph; its validation split is that of the export (val_seed 0). R7 typed and R2 no-edges carry 15% block masking. The pooled table is seed 0; the paired differences, second minus first for the AUC checkpoint at its validation threshold, use the seeds complete in both arms (queue seeds [0, 1, 2]), with the 95% paired cluster bootstrap interval (2000 resamples). The rule column is the pre-registered rule (dAUC >= 0.01 and dmacro-F1 >= 0.02, both 95% intervals excluding zero) read as "Model 1 beats the reference". TF-IDF and the header-imports LR exist for seed 0 only, so their pairs are a direction. A missing arm or fold is stated.

## Pooled, AUC checkpoint

| arm | group | AUC | PR-AUC | macro-F1 |
|---|---|---|---|---|
| TF-IDF | tfidf | 0.982 | 0.990 | 0.935 |
| header-imports LR | imports-header | 0.985 | 0.991 | 0.954 |
| R7 typed | kfold_x86_gnn-mblk15 | 0.986 | 0.990 | 0.935 |
| R2 no-edges | kfold_x86_gnn-noedges-mblk15 | 0.977 | 0.983 | 0.941 |
| Model 1 tokenseq | kfold_x86_tokenseq | 0.964 | 0.980 | 0.882 |

## Paired differences, second minus first

| first | second | dAUC [95% CI] | dmacro-F1 [95% CI] | rule |
|---|---|---|---|---|
| TF-IDF | Model 1 tokenseq | -0.018 [-0.034, -0.005] | -0.053 [-0.127, +0.010] | does not beat (seed 0 only: a direction, not the test) |
| header-imports LR | Model 1 tokenseq | -0.021 [-0.039, -0.006] | -0.072 [-0.143, -0.011] | does not beat (seed 0 only: a direction, not the test) |
| R7 typed | Model 1 tokenseq | -0.009 [-0.021, +0.001] | -0.033 [-0.101, +0.030] | does not beat |
| R2 no-edges | Model 1 tokenseq | -0.005 [-0.016, +0.005] | -0.027 [-0.092, +0.028] | does not beat |

### kfold_x86_tokenseq minus tfidf (selection auc, 2000 resamples)

a = tfidf, b = kfold_x86_tokenseq; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.982 | 0.964 | -0.018 | [-0.034, -0.005] |
| PR-AUC | 0.990 | 0.980 | -0.011 | [-0.024, -0.003] |
| macro-F1 | 0.935 | 0.882 | -0.053 | [-0.127, +0.010] |

### kfold_x86_tokenseq minus imports-header (selection auc, 2000 resamples)

a = imports-header, b = kfold_x86_tokenseq; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.985 | 0.964 | -0.021 | [-0.039, -0.006] |
| PR-AUC | 0.991 | 0.980 | -0.012 | [-0.026, -0.003] |
| macro-F1 | 0.954 | 0.882 | -0.072 | [-0.143, -0.011] |

### kfold_x86_tokenseq minus kfold_x86_gnn-mblk15 (selection auc, 2000 resamples)

a = kfold_x86_gnn-mblk15, b = kfold_x86_tokenseq; seeds [0, 1, 2]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.979 | 0.970 | -0.009 | [-0.021, +0.001] |
| PR-AUC | 0.987 | 0.982 | -0.005 | [-0.014, +0.003] |
| macro-F1 | 0.935 | 0.902 | -0.033 | [-0.101, +0.030] |

### kfold_x86_tokenseq minus kfold_x86_gnn-noedges-mblk15 (selection auc, 2000 resamples)

a = kfold_x86_gnn-noedges-mblk15, b = kfold_x86_tokenseq; seeds [0, 1, 2]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.974 | 0.970 | -0.005 | [-0.016, +0.005] |
| PR-AUC | 0.982 | 0.982 | -0.001 | [-0.009, +0.007] |
| macro-F1 | 0.929 | 0.902 | -0.027 | [-0.092, +0.028] |

