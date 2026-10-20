# Gate summary, stage day1 (seed 0 only, not the pre-registered test)

## Pooled kfold x86, seed 0

| arm | group | f1 AUC | f1 PR-AUC | f1 macro-F1 | auc AUC | auc PR-AUC | auc macro-F1 |
|---|---|---|---|---|---|---|---|
| R0 TF-IDF | tfidf | 0.982 | 0.990 | 0.939 | 0.982 | 0.990 | 0.935 |
| R2 tfidf | kfold_x86_gnn-noedges | 0.926 | 0.957 | 0.896 | 0.976 | 0.981 | 0.924 |
| R2 mean | kfold_x86_gnn-noedges-enc-mean | 0.961 | 0.979 | 0.910 | 0.975 | 0.984 | 0.917 |
| R4 shuffle | kfold_x86_gnn-shuffle | 0.940 | 0.969 | 0.901 | 0.975 | 0.984 | 0.929 |
| R5 untyped | kfold_x86_gnn-untyped | 0.959 | 0.967 | 0.880 | 0.945 | 0.969 | 0.898 |
| R7 typed | kfold_x86_gnn | 0.972 | 0.978 | 0.880 | 0.964 | 0.974 | 0.905 |

## Gate A preliminary verdict (seed 0 only, not the pre-registered test)

R2 `--encoder tfidf` against R0 TF-IDF (PLAN_GNN.md section 5): pooled AUC >= R0 - 0.01 and macro-F1 at the validation threshold >= R0 - 0.02. The AUC-checkpoint line is the one that applies the validation threshold; the F1-checkpoint line scores at 0.5. This is one seed and cannot pass the pre-registered rule.

- F1 checkpoint (macro-F1 at 0.5): FAIL. AUC 0.926 vs R0 0.982 - 0.01 = 0.972 (below); macro-F1 0.896 vs R0 0.939 - 0.02 = 0.919 (below).
- AUC checkpoint (macro-F1 at validation threshold): PASS. AUC 0.976 vs R0 0.982 - 0.01 = 0.972 (ok); macro-F1 0.924 vs R0 0.935 - 0.02 = 0.915 (ok).

## Paired differences, second minus first (seed 0 only, not the pre-registered test)

**R2 tfidf vs R4 shuffle**

### kfold_x86_gnn-shuffle minus kfold_x86_gnn-noedges (selection f1, 2000 resamples)

a = kfold_x86_gnn-noedges, b = kfold_x86_gnn-shuffle; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.926 | 0.940 | +0.014 | [-0.094, +0.124] |
| PR-AUC | 0.957 | 0.969 | +0.012 | [-0.031, +0.056] |
| macro-F1 | 0.896 | 0.901 | +0.005 | [-0.014, +0.024] |

**R2 tfidf vs R7 typed**

### kfold_x86_gnn minus kfold_x86_gnn-noedges (selection f1, 2000 resamples)

a = kfold_x86_gnn-noedges, b = kfold_x86_gnn; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.926 | 0.972 | +0.046 | [+0.003, +0.133] |
| PR-AUC | 0.957 | 0.978 | +0.021 | [+0.001, +0.057] |
| macro-F1 | 0.896 | 0.880 | -0.016 | [-0.074, +0.016] |

**R4 shuffle vs R7 typed**

### kfold_x86_gnn minus kfold_x86_gnn-shuffle (selection f1, 2000 resamples)

a = kfold_x86_gnn-shuffle, b = kfold_x86_gnn; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.940 | 0.972 | +0.033 | [-0.008, +0.118] |
| PR-AUC | 0.969 | 0.978 | +0.009 | [-0.011, +0.044] |
| macro-F1 | 0.901 | 0.880 | -0.021 | [-0.080, +0.017] |

**R5 untyped vs R7 typed**

### kfold_x86_gnn minus kfold_x86_gnn-untyped (selection f1, 2000 resamples)

a = kfold_x86_gnn-untyped, b = kfold_x86_gnn; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.959 | 0.972 | +0.013 | [+0.003, +0.026] |
| PR-AUC | 0.967 | 0.978 | +0.011 | [+0.002, +0.026] |
| macro-F1 | 0.880 | 0.880 | +0.001 | [-0.063, +0.051] |

**R0 TF-IDF vs R2 tfidf**

### kfold_x86_gnn-noedges minus tfidf (selection f1, 2000 resamples)

a = tfidf, b = kfold_x86_gnn-noedges; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.982 | 0.926 | -0.056 | [-0.156, -0.007] |
| PR-AUC | 0.990 | 0.957 | -0.033 | [-0.082, -0.009] |
| macro-F1 | 0.939 | 0.896 | -0.043 | [-0.125, +0.024] |

**R2 tfidf vs R4 shuffle**

### kfold_x86_gnn-shuffle minus kfold_x86_gnn-noedges (selection auc, 2000 resamples)

a = kfold_x86_gnn-noedges, b = kfold_x86_gnn-shuffle; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.976 | 0.975 | -0.001 | [-0.034, +0.019] |
| PR-AUC | 0.981 | 0.984 | +0.003 | [-0.015, +0.019] |
| macro-F1 | 0.924 | 0.929 | +0.005 | [-0.073, +0.088] |

**R2 tfidf vs R7 typed**

### kfold_x86_gnn minus kfold_x86_gnn-noedges (selection auc, 2000 resamples)

a = kfold_x86_gnn-noedges, b = kfold_x86_gnn; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.976 | 0.964 | -0.012 | [-0.034, +0.002] |
| PR-AUC | 0.981 | 0.974 | -0.007 | [-0.021, +0.002] |
| macro-F1 | 0.924 | 0.905 | -0.019 | [-0.078, +0.015] |

**R4 shuffle vs R7 typed**

### kfold_x86_gnn minus kfold_x86_gnn-shuffle (selection auc, 2000 resamples)

a = kfold_x86_gnn-shuffle, b = kfold_x86_gnn; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.975 | 0.964 | -0.011 | [-0.044, +0.025] |
| PR-AUC | 0.984 | 0.974 | -0.010 | [-0.032, +0.009] |
| macro-F1 | 0.929 | 0.905 | -0.024 | [-0.094, +0.018] |

**R5 untyped vs R7 typed**

### kfold_x86_gnn minus kfold_x86_gnn-untyped (selection auc, 2000 resamples)

a = kfold_x86_gnn-untyped, b = kfold_x86_gnn; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.945 | 0.964 | +0.019 | [-0.033, +0.106] |
| PR-AUC | 0.969 | 0.974 | +0.005 | [-0.021, +0.042] |
| macro-F1 | 0.898 | 0.905 | +0.007 | [-0.007, +0.026] |

**R0 TF-IDF vs R2 tfidf**

### kfold_x86_gnn-noedges minus tfidf (selection auc, 2000 resamples)

a = tfidf, b = kfold_x86_gnn-noedges; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.982 | 0.976 | -0.006 | [-0.021, +0.012] |
| PR-AUC | 0.990 | 0.981 | -0.009 | [-0.023, +0.001] |
| macro-F1 | 0.935 | 0.924 | -0.011 | [-0.082, +0.036] |

