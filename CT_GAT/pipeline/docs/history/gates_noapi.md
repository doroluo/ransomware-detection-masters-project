# Gate summary, stage noapi (seed 0 only, not the pre-registered test)

kfold x86 on the v2 cache, 15% block masking on every GNN arm. R7 noapi is `--vocab noapi`: each block holds its mn, sig and beh tokens and no api token (`gnn-mblk15-voc-noapi`); the header-import features are not used by any arm. TF-IDF is the default sig + api logistic regression; TF-IDF noapi is `baselines.py --vocab noapi` over mn + sig + beh, the tokens of R7 noapi. The pooled table is seed 0; the paired differences, second minus first for the AUC checkpoint at its validation threshold, use the seeds complete in both arms (queue seeds [0, 1, 2]), with the 95% paired cluster bootstrap interval (2000 resamples). The rule column is the pre-registered rule (dAUC >= 0.01 and dmacro-F1 >= 0.02, both 95% intervals excluding zero) in the direction named: "api loss shown" for the two ablation pairs, "beats" for the others. TF-IDF and the header-imports LR exist for seed 0 only, so their pairs are a direction. A missing arm or fold is stated.

## Pooled, AUC checkpoint

| arm | group | AUC | PR-AUC | macro-F1 |
|---|---|---|---|---|
| TF-IDF | tfidf | 0.982 | 0.990 | 0.935 |
| TF-IDF noapi | tfidf-noapi | 0.976 | 0.984 | 0.908 |
| header-imports LR | imports-header | 0.985 | 0.991 | 0.954 |
| R7 typed | kfold_x86_gnn-mblk15 | 0.986 | 0.990 | 0.935 |
| R7 noapi | kfold_x86_gnn-mblk15-voc-noapi | 0.943 | 0.968 | 0.880 |

## Paired differences, second minus first

| first | second | dAUC [95% CI] | dmacro-F1 [95% CI] | rule |
|---|---|---|---|---|
| R7 typed | R7 noapi | -0.017 [-0.044, -0.001] | -0.025 [-0.048, -0.007] | api loss shown |
| TF-IDF | TF-IDF noapi | -0.005 [-0.014, +0.007] | -0.027 [-0.081, +0.012] | api loss not shown (seed 0 only: a direction, not the test) |
| TF-IDF noapi | R7 noapi | -0.033 [-0.089, +0.001] | -0.028 [-0.123, +0.054] | does not beat (seed 0 only: a direction, not the test) |
| header-imports LR | R7 noapi | -0.041 [-0.099, -0.007] | -0.074 [-0.158, -0.008] | does not beat (seed 0 only: a direction, not the test) |

### kfold_x86_gnn-mblk15-voc-noapi minus kfold_x86_gnn-mblk15 (selection auc, 2000 resamples)

a = kfold_x86_gnn-mblk15, b = kfold_x86_gnn-mblk15-voc-noapi; seeds [0, 1, 2]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.979 | 0.961 | -0.017 | [-0.044, -0.001] |
| PR-AUC | 0.987 | 0.979 | -0.008 | [-0.023, +0.001] |
| macro-F1 | 0.935 | 0.910 | -0.025 | [-0.048, -0.007] |

### tfidf-noapi minus tfidf (selection auc, 2000 resamples)

a = tfidf, b = tfidf-noapi; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.982 | 0.976 | -0.005 | [-0.014, +0.007] |
| PR-AUC | 0.990 | 0.984 | -0.007 | [-0.015, -0.001] |
| macro-F1 | 0.935 | 0.908 | -0.027 | [-0.081, +0.012] |

### kfold_x86_gnn-mblk15-voc-noapi minus tfidf-noapi (selection auc, 2000 resamples)

a = tfidf-noapi, b = kfold_x86_gnn-mblk15-voc-noapi; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.976 | 0.943 | -0.033 | [-0.089, +0.001] |
| PR-AUC | 0.984 | 0.968 | -0.015 | [-0.046, +0.003] |
| macro-F1 | 0.908 | 0.880 | -0.028 | [-0.123, +0.054] |

### kfold_x86_gnn-mblk15-voc-noapi minus imports-header (selection auc, 2000 resamples)

a = imports-header, b = kfold_x86_gnn-mblk15-voc-noapi; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.985 | 0.943 | -0.041 | [-0.099, -0.007] |
| PR-AUC | 0.991 | 0.968 | -0.023 | [-0.055, -0.004] |
| macro-F1 | 0.954 | 0.880 | -0.074 | [-0.158, -0.008] |

