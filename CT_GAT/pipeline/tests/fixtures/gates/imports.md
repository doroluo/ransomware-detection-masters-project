# Gate summary, stage imports (seed 0 only, not the pre-registered test)

kfold x86 on the v2 cache, seed 0 pooled over the five folds. Baselines (`baselines.py --imports`, TF-IDF logistic regression on the same folds): every header import identity. GNN arms: R7 typed with 15% block masking, with and without `--imports header` (an EmbeddingBag over the header import names and two log counts in the head; queue stage `imports_header`). Differences are second minus first for the AUC checkpoint at its validation threshold, with the 95% paired cluster bootstrap interval (2000 resamples). A missing arm or fold is stated.

## Pooled, AUC checkpoint

| arm | group | AUC | PR-AUC | macro-F1 |
|---|---|---|---|---|
| TF-IDF | tfidf | 0.982 | 0.990 | 0.935 |
| header imports, LR | imports-header | 0.985 | 0.991 | 0.954 |
| R7 typed | kfold_x86_gnn-mblk15 | 0.986 | 0.990 | 0.935 |
| R7 + header imports | kfold_x86_gnn-mblk15-imph | 0.985 | 0.992 | 0.940 |

## Paired differences, second minus first

| first | second | dAUC [95% CI] | dmacro-F1 [95% CI] |
|---|---|---|---|
| R7 typed | R7 + header imports | -0.001 [-0.013, +0.007] | +0.006 [-0.075, +0.084] |
| header imports, LR | R7 + header imports | +0.000 [-0.008, +0.008] | -0.014 [-0.084, +0.031] |
| TF-IDF | R7 + header imports | +0.003 [-0.003, +0.010] | +0.005 [-0.067, +0.054] |

### kfold_x86_gnn-mblk15-imph minus kfold_x86_gnn-mblk15 (selection auc, 2000 resamples)

a = kfold_x86_gnn-mblk15, b = kfold_x86_gnn-mblk15-imph; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.986 | 0.985 | -0.001 | [-0.013, +0.007] |
| PR-AUC | 0.990 | 0.992 | +0.002 | [-0.004, +0.009] |
| macro-F1 | 0.935 | 0.940 | +0.006 | [-0.075, +0.084] |

### kfold_x86_gnn-mblk15-imph minus imports-header (selection auc, 2000 resamples)

a = imports-header, b = kfold_x86_gnn-mblk15-imph; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.985 | 0.985 | +0.000 | [-0.008, +0.008] |
| PR-AUC | 0.991 | 0.992 | +0.001 | [-0.004, +0.006] |
| macro-F1 | 0.954 | 0.940 | -0.014 | [-0.084, +0.031] |

### kfold_x86_gnn-mblk15-imph minus tfidf (selection auc, 2000 resamples)

a = tfidf, b = kfold_x86_gnn-mblk15-imph; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.982 | 0.985 | +0.003 | [-0.003, +0.010] |
| PR-AUC | 0.990 | 0.992 | +0.002 | [-0.003, +0.008] |
| macro-F1 | 0.935 | 0.940 | +0.005 | [-0.067, +0.054] |

