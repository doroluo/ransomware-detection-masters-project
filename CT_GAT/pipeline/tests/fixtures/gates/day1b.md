# Gate summary, stage day1b (seed 0 only, not the pre-registered test)

## Pooled kfold x86 GNN groups, seed 0

| group | f1 AUC | f1 PR-AUC | f1 macro-F1 | auc AUC | auc PR-AUC | auc macro-F1 |
|---|---|---|---|---|---|---|
| kfold_x86_gnn | 0.972 | 0.978 | 0.880 | 0.964 | 0.974 | 0.905 |
| kfold_x86_gnn-L8 | 0.975 | 0.981 | 0.947 | 0.978 | 0.982 | 0.913 |
| kfold_x86_gnn-bwd | 0.950 | 0.971 | 0.891 | 0.980 | 0.984 | 0.887 |
| kfold_x86_gnn-dedge20 | 0.986 | 0.988 | 0.929 | 0.982 | 0.987 | 0.908 |
| kfold_x86_gnn-dfn15 | 0.967 | 0.977 | 0.924 | 0.972 | 0.974 | 0.921 |
| kfold_x86_gnn-dfn5-dlib40 | 0.964 | 0.981 | 0.927 | 0.944 | 0.972 | 0.921 |
| kfold_x86_gnn-flip20 | 0.980 | 0.986 | 0.912 | 0.986 | 0.990 | 0.888 |
| kfold_x86_gnn-fwd | 0.937 | 0.965 | 0.903 | 0.930 | 0.964 | 0.908 |
| kfold_x86_gnn-mblk15 | 0.970 | 0.975 | 0.934 | 0.986 | 0.990 | 0.935 |
| kfold_x86_gnn-mblk15-B1 | 0.958 | 0.977 | 0.931 | 0.975 | 0.985 | 0.956 |
| kfold_x86_gnn-mblk15-B3 | 0.974 | 0.982 | 0.905 | 0.977 | 0.986 | 0.934 |
| kfold_x86_gnn-mblk15-L1 | 0.970 | 0.980 | 0.935 | 0.973 | 0.981 | 0.926 |
| kfold_x86_gnn-mblk15-L2 | 0.988 | 0.992 | 0.951 | 0.985 | 0.990 | 0.969 |
| kfold_x86_gnn-mblk15-L4 | 0.948 | 0.965 | 0.938 | 0.982 | 0.987 | 0.963 |
| kfold_x86_gnn-mblk15-L6 | 0.946 | 0.970 | 0.893 | 0.950 | 0.973 | 0.940 |
| kfold_x86_gnn-mblk15-L8 | 0.983 | 0.988 | 0.950 | 0.976 | 0.985 | 0.924 |
| kfold_x86_gnn-mblk15-bv-ours_cap0_codebert | 0.898 | 0.951 | 0.878 | 0.961 | 0.979 | 0.937 |
| kfold_x86_gnn-mblk15-bv-ours_cap0_codebert_smoke | missing (seed 0 folds present [0] of 0-4) |  |  | missing (seed 0 folds present [0] of 0-4) |  |  |
| kfold_x86_gnn-mblk15-bv-ours_cap0_ref_smoke | missing (seed 0 folds present [] of 0-4) |  |  | missing (seed 0 folds present [] of 0-4) |  |  |
| kfold_x86_gnn-mblk15-imph | 0.984 | 0.992 | 0.938 | 0.985 | 0.992 | 0.940 |
| kfold_x86_gnn-mblk15-sg-cap20k | 0.976 | 0.983 | 0.921 | 0.975 | 0.981 | 0.921 |
| kfold_x86_gnn-mblk15-sg-cap500_b500 | 0.935 | 0.940 | 0.910 | 0.923 | 0.954 | 0.843 |
| kfold_x86_gnn-mblk15-sg-cap5k | 0.948 | 0.963 | 0.916 | 0.969 | 0.975 | 0.943 |
| kfold_x86_gnn-mblk15-voc-noapi | 0.969 | 0.980 | 0.888 | 0.943 | 0.968 | 0.880 |
| kfold_x86_gnn-mblk15-voc-noapi_smoke | missing (seed 0 folds present [0] of 0-4) |  |  | missing (seed 0 folds present [0] of 0-4) |  |  |
| kfold_x86_gnn-mtok15 | 0.976 | 0.980 | 0.892 | 0.957 | 0.977 | 0.910 |
| kfold_x86_gnn-mtok50 | 0.943 | 0.971 | 0.916 | 0.963 | 0.979 | 0.915 |
| kfold_x86_gnn-nocall-mblk15 | 0.961 | 0.976 | 0.937 | 0.973 | 0.982 | 0.877 |
| kfold_x86_gnn-noedges | 0.926 | 0.957 | 0.896 | 0.976 | 0.981 | 0.924 |
| kfold_x86_gnn-noedges-enc-mean | 0.961 | 0.979 | 0.910 | 0.975 | 0.984 | 0.917 |
| kfold_x86_gnn-noedges-mblk15 | 0.972 | 0.978 | 0.934 | 0.977 | 0.983 | 0.941 |
| kfold_x86_gnn-noedges-mblk15-bv-ours_cap0_codebert_smoke | missing (seed 0 folds present [0] of 0-4) |  |  | missing (seed 0 folds present [0] of 0-4) |  |  |
| kfold_x86_gnn-noedges-mblk15-bv-ours_cap0_ref_smoke | missing (seed 0 folds present [] of 0-4) |  |  | missing (seed 0 folds present [] of 0-4) |  |  |
| kfold_x86_gnn-noedges-mblk15-enc-presence | missing (seed 0 folds present [0, 1, 2] of 0-4) |  |  | missing (seed 0 folds present [0, 1, 2] of 0-4) |  |  |
| kfold_x86_gnn-noedges-mblk15-enc-raw | 0.964 | 0.974 | 0.907 | 0.972 | 0.980 | 0.935 |
| kfold_x86_gnn-noedges-mblk15-sg-cap500_b500 | 0.948 | 0.952 | 0.931 | 0.955 | 0.964 | 0.915 |
| kfold_x86_gnn-noedges-mblk15-voc-hybrid | 0.969 | 0.975 | 0.922 | 0.969 | 0.977 | 0.896 |
| kfold_x86_gnn-noedges-mblk15-voc-noapi | 0.979 | 0.985 | 0.947 | 0.983 | 0.988 | 0.938 |
| kfold_x86_gnn-noedges-mblk15-voc-yanping | 0.975 | 0.979 | 0.941 | 0.981 | 0.984 | 0.940 |
| kfold_x86_gnn-shuffle | 0.940 | 0.969 | 0.901 | 0.975 | 0.984 | 0.929 |
| kfold_x86_gnn-shuffle-mblk15 | 0.968 | 0.980 | 0.895 | 0.974 | 0.982 | 0.901 |
| kfold_x86_gnn-stack-mblk15_smoke | missing (seed 0 folds present [] of 0-4) |  |  | missing (seed 0 folds present [] of 0-4) |  |  |
| kfold_x86_gnn-tf25 | 0.930 | 0.934 | 0.856 | 0.945 | 0.950 | 0.884 |
| kfold_x86_gnn-tf50 | 0.931 | 0.953 | 0.871 | 0.956 | 0.969 | 0.874 |
| kfold_x86_gnn-undir | 0.964 | 0.971 | 0.887 | 0.928 | 0.963 | 0.878 |
| kfold_x86_gnn-untyped | 0.959 | 0.967 | 0.880 | 0.945 | 0.969 | 0.898 |
| kfold_x86_gnn_team-cb500 | 0.844 | 0.908 | 0.723 | 0.791 | 0.872 | 0.723 |
| kfold_x86_gnn_team-gnn500 | 0.950 | 0.970 | 0.836 | 0.956 | 0.975 | 0.850 |
| kfold_x86_gnn_team-sig500 | 0.939 | 0.966 | 0.830 | 0.953 | 0.974 | 0.894 |
| kfold_x86_gnn_team-sig500v2 | 0.953 | 0.975 | 0.913 | 0.948 | 0.972 | 0.874 |

Selections: f1 = F1 checkpoint (macro-F1 at 0.5); auc = AUC checkpoint (macro-F1 at validation threshold).

## Paired differences, new arm minus reference, AUC selection (seed 0 only, not the pre-registered test)

**kfold_x86_gnn-mtok15 vs R7 typed**

### kfold_x86_gnn-mtok15 minus kfold_x86_gnn (selection auc, 2000 resamples)

a = kfold_x86_gnn, b = kfold_x86_gnn-mtok15; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.964 | 0.957 | -0.007 | [-0.048, +0.017] |
| PR-AUC | 0.974 | 0.977 | +0.003 | [-0.014, +0.018] |
| macro-F1 | 0.905 | 0.910 | +0.004 | [-0.009, +0.016] |

**kfold_x86_gnn-mtok15 vs R2 tfidf**

### kfold_x86_gnn-mtok15 minus kfold_x86_gnn-noedges (selection auc, 2000 resamples)

a = kfold_x86_gnn-noedges, b = kfold_x86_gnn-mtok15; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.976 | 0.957 | -0.019 | [-0.082, +0.013] |
| PR-AUC | 0.981 | 0.977 | -0.004 | [-0.032, +0.015] |
| macro-F1 | 0.924 | 0.910 | -0.014 | [-0.076, +0.025] |

**kfold_x86_gnn-dfn5-dlib40 vs R7 typed**

### kfold_x86_gnn-dfn5-dlib40 minus kfold_x86_gnn (selection auc, 2000 resamples)

a = kfold_x86_gnn, b = kfold_x86_gnn-dfn5-dlib40; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.964 | 0.944 | -0.020 | [-0.107, +0.039] |
| PR-AUC | 0.974 | 0.972 | -0.002 | [-0.038, +0.028] |
| macro-F1 | 0.905 | 0.921 | +0.016 | [-0.027, +0.087] |

**kfold_x86_gnn-dfn5-dlib40 vs R2 tfidf**

### kfold_x86_gnn-dfn5-dlib40 minus kfold_x86_gnn-noedges (selection auc, 2000 resamples)

a = kfold_x86_gnn-noedges, b = kfold_x86_gnn-dfn5-dlib40; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.976 | 0.944 | -0.032 | [-0.117, +0.014] |
| PR-AUC | 0.981 | 0.972 | -0.009 | [-0.046, +0.015] |
| macro-F1 | 0.924 | 0.921 | -0.002 | [-0.079, +0.079] |

**kfold_x86_gnn-undir vs R7 typed**

### kfold_x86_gnn-undir minus kfold_x86_gnn (selection auc, 2000 resamples)

a = kfold_x86_gnn, b = kfold_x86_gnn-undir; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.964 | 0.928 | -0.037 | [-0.119, +0.013] |
| PR-AUC | 0.974 | 0.963 | -0.012 | [-0.049, +0.012] |
| macro-F1 | 0.905 | 0.878 | -0.027 | [-0.069, +0.007] |

**kfold_x86_gnn-undir vs R2 tfidf**

### kfold_x86_gnn-undir minus kfold_x86_gnn-noedges (selection auc, 2000 resamples)

a = kfold_x86_gnn-noedges, b = kfold_x86_gnn-undir; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.976 | 0.928 | -0.049 | [-0.137, +0.008] |
| PR-AUC | 0.981 | 0.963 | -0.019 | [-0.059, +0.009] |
| macro-F1 | 0.924 | 0.878 | -0.046 | [-0.113, +0.008] |

**kfold_x86_gnn-fwd vs R7 typed**

### kfold_x86_gnn-fwd minus kfold_x86_gnn (selection auc, 2000 resamples)

a = kfold_x86_gnn, b = kfold_x86_gnn-fwd; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.964 | 0.930 | -0.034 | [-0.106, +0.005] |
| PR-AUC | 0.974 | 0.964 | -0.011 | [-0.038, +0.008] |
| macro-F1 | 0.905 | 0.908 | +0.003 | [-0.014, +0.023] |

**kfold_x86_gnn-fwd vs R2 tfidf**

### kfold_x86_gnn-fwd minus kfold_x86_gnn-noedges (selection auc, 2000 resamples)

a = kfold_x86_gnn-noedges, b = kfold_x86_gnn-fwd; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.976 | 0.930 | -0.046 | [-0.138, +0.003] |
| PR-AUC | 0.981 | 0.964 | -0.017 | [-0.056, +0.004] |
| macro-F1 | 0.924 | 0.908 | -0.015 | [-0.060, +0.015] |

**kfold_x86_gnn-dfn15 vs R7 typed**

### kfold_x86_gnn-dfn15 minus kfold_x86_gnn (selection auc, 2000 resamples)

a = kfold_x86_gnn, b = kfold_x86_gnn-dfn15; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.964 | 0.972 | +0.007 | [-0.019, +0.041] |
| PR-AUC | 0.974 | 0.974 | -0.000 | [-0.019, +0.020] |
| macro-F1 | 0.905 | 0.921 | +0.016 | [-0.036, +0.084] |

**kfold_x86_gnn-dfn15 vs R2 tfidf**

### kfold_x86_gnn-dfn15 minus kfold_x86_gnn-noedges (selection auc, 2000 resamples)

a = kfold_x86_gnn-noedges, b = kfold_x86_gnn-dfn15; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.976 | 0.972 | -0.005 | [-0.026, +0.012] |
| PR-AUC | 0.981 | 0.974 | -0.007 | [-0.025, +0.006] |
| macro-F1 | 0.924 | 0.921 | -0.003 | [-0.082, +0.081] |

**kfold_x86_gnn-tf50 vs R7 typed**

### kfold_x86_gnn-tf50 minus kfold_x86_gnn (selection auc, 2000 resamples)

a = kfold_x86_gnn, b = kfold_x86_gnn-tf50; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.964 | 0.956 | -0.008 | [-0.027, +0.008] |
| PR-AUC | 0.974 | 0.969 | -0.006 | [-0.023, +0.008] |
| macro-F1 | 0.905 | 0.874 | -0.031 | [-0.069, -0.003] |

**kfold_x86_gnn-tf50 vs R2 tfidf**

### kfold_x86_gnn-tf50 minus kfold_x86_gnn-noedges (selection auc, 2000 resamples)

a = kfold_x86_gnn-noedges, b = kfold_x86_gnn-tf50; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.976 | 0.956 | -0.020 | [-0.046, -0.000] |
| PR-AUC | 0.981 | 0.969 | -0.013 | [-0.034, +0.002] |
| macro-F1 | 0.924 | 0.874 | -0.050 | [-0.119, -0.001] |

**kfold_x86_gnn-tf25 vs R7 typed**

### kfold_x86_gnn-tf25 minus kfold_x86_gnn (selection auc, 2000 resamples)

a = kfold_x86_gnn, b = kfold_x86_gnn-tf25; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.964 | 0.945 | -0.019 | [-0.049, +0.015] |
| PR-AUC | 0.974 | 0.950 | -0.024 | [-0.053, -0.002] |
| macro-F1 | 0.905 | 0.884 | -0.021 | [-0.078, +0.055] |

**kfold_x86_gnn-tf25 vs R2 tfidf**

### kfold_x86_gnn-tf25 minus kfold_x86_gnn-noedges (selection auc, 2000 resamples)

a = kfold_x86_gnn-noedges, b = kfold_x86_gnn-tf25; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.976 | 0.945 | -0.031 | [-0.056, -0.012] |
| PR-AUC | 0.981 | 0.950 | -0.031 | [-0.060, -0.013] |
| macro-F1 | 0.924 | 0.884 | -0.039 | [-0.122, +0.047] |

**kfold_x86_gnn-mblk15 vs R7 typed**

### kfold_x86_gnn-mblk15 minus kfold_x86_gnn (selection auc, 2000 resamples)

a = kfold_x86_gnn, b = kfold_x86_gnn-mblk15; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.964 | 0.986 | +0.022 | [+0.005, +0.054] |
| PR-AUC | 0.974 | 0.990 | +0.016 | [+0.004, +0.038] |
| macro-F1 | 0.905 | 0.935 | +0.029 | [-0.007, +0.096] |

**kfold_x86_gnn-mblk15 vs R2 tfidf**

### kfold_x86_gnn-mblk15 minus kfold_x86_gnn-noedges (selection auc, 2000 resamples)

a = kfold_x86_gnn-noedges, b = kfold_x86_gnn-mblk15; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.976 | 0.986 | +0.010 | [+0.001, +0.021] |
| PR-AUC | 0.981 | 0.990 | +0.009 | [+0.001, +0.021] |
| macro-F1 | 0.924 | 0.935 | +0.011 | [-0.067, +0.092] |

**kfold_x86_gnn-dedge20 vs R7 typed**

### kfold_x86_gnn-dedge20 minus kfold_x86_gnn (selection auc, 2000 resamples)

a = kfold_x86_gnn, b = kfold_x86_gnn-dedge20; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.964 | 0.982 | +0.018 | [+0.004, +0.038] |
| PR-AUC | 0.974 | 0.987 | +0.013 | [+0.004, +0.028] |
| macro-F1 | 0.905 | 0.908 | +0.003 | [-0.012, +0.015] |

**kfold_x86_gnn-dedge20 vs R2 tfidf**

### kfold_x86_gnn-dedge20 minus kfold_x86_gnn-noedges (selection auc, 2000 resamples)

a = kfold_x86_gnn-noedges, b = kfold_x86_gnn-dedge20; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.976 | 0.982 | +0.006 | [-0.002, +0.014] |
| PR-AUC | 0.981 | 0.987 | +0.006 | [-0.002, +0.017] |
| macro-F1 | 0.924 | 0.908 | -0.016 | [-0.077, +0.022] |

**kfold_x86_gnn-flip20 vs R7 typed**

### kfold_x86_gnn-flip20 minus kfold_x86_gnn (selection auc, 2000 resamples)

a = kfold_x86_gnn, b = kfold_x86_gnn-flip20; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.964 | 0.986 | +0.022 | [+0.003, +0.051] |
| PR-AUC | 0.974 | 0.990 | +0.015 | [+0.004, +0.036] |
| macro-F1 | 0.905 | 0.888 | -0.017 | [-0.053, +0.008] |

**kfold_x86_gnn-flip20 vs R2 tfidf**

### kfold_x86_gnn-flip20 minus kfold_x86_gnn-noedges (selection auc, 2000 resamples)

a = kfold_x86_gnn-noedges, b = kfold_x86_gnn-flip20; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.976 | 0.986 | +0.010 | [-0.000, +0.022] |
| PR-AUC | 0.981 | 0.990 | +0.009 | [-0.001, +0.022] |
| macro-F1 | 0.924 | 0.888 | -0.035 | [-0.097, +0.008] |

**kfold_x86_gnn-mtok50 vs R7 typed**

### kfold_x86_gnn-mtok50 minus kfold_x86_gnn (selection auc, 2000 resamples)

a = kfold_x86_gnn, b = kfold_x86_gnn-mtok50; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.964 | 0.963 | -0.001 | [-0.014, +0.009] |
| PR-AUC | 0.974 | 0.979 | +0.005 | [-0.004, +0.014] |
| macro-F1 | 0.905 | 0.915 | +0.009 | [-0.033, +0.071] |

**kfold_x86_gnn-mtok50 vs R2 tfidf**

### kfold_x86_gnn-mtok50 minus kfold_x86_gnn-noedges (selection auc, 2000 resamples)

a = kfold_x86_gnn-noedges, b = kfold_x86_gnn-mtok50; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.976 | 0.963 | -0.013 | [-0.045, +0.008] |
| PR-AUC | 0.981 | 0.979 | -0.002 | [-0.020, +0.011] |
| macro-F1 | 0.924 | 0.915 | -0.009 | [-0.033, +0.010] |

**kfold_x86_gnn-bwd vs R7 typed**

### kfold_x86_gnn-bwd minus kfold_x86_gnn (selection auc, 2000 resamples)

a = kfold_x86_gnn, b = kfold_x86_gnn-bwd; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.964 | 0.980 | +0.016 | [-0.000, +0.040] |
| PR-AUC | 0.974 | 0.984 | +0.009 | [+0.000, +0.026] |
| macro-F1 | 0.905 | 0.887 | -0.018 | [-0.054, +0.007] |

**kfold_x86_gnn-bwd vs R2 tfidf**

### kfold_x86_gnn-bwd minus kfold_x86_gnn-noedges (selection auc, 2000 resamples)

a = kfold_x86_gnn-noedges, b = kfold_x86_gnn-bwd; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.976 | 0.980 | +0.004 | [-0.003, +0.011] |
| PR-AUC | 0.981 | 0.984 | +0.003 | [-0.004, +0.009] |
| macro-F1 | 0.924 | 0.887 | -0.037 | [-0.102, +0.011] |

**kfold_x86_gnn-L8 vs R7 typed**

### kfold_x86_gnn-L8 minus kfold_x86_gnn (selection auc, 2000 resamples)

a = kfold_x86_gnn, b = kfold_x86_gnn-L8; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.964 | 0.978 | +0.014 | [-0.004, +0.040] |
| PR-AUC | 0.974 | 0.982 | +0.008 | [-0.005, +0.025] |
| macro-F1 | 0.905 | 0.913 | +0.008 | [-0.047, +0.077] |

**kfold_x86_gnn-L8 vs R2 tfidf**

### kfold_x86_gnn-L8 minus kfold_x86_gnn-noedges (selection auc, 2000 resamples)

a = kfold_x86_gnn-noedges, b = kfold_x86_gnn-L8; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.976 | 0.978 | +0.002 | [-0.007, +0.011] |
| PR-AUC | 0.981 | 0.982 | +0.001 | [-0.009, +0.011] |
| macro-F1 | 0.924 | 0.913 | -0.010 | [-0.049, +0.024] |

## Learning curve: training fraction (seed 0 only, not the pre-registered test)

AUC checkpoint at its validation threshold for R7; the TF-IDF logistic regression at its validation threshold. Test sets are identical across fractions.

| train fraction | model | group | AUC | PR-AUC | macro-F1 |
|---|---|---|---|---|---|
| 0.25 | R7 typed | kfold_x86_gnn-tf25 | 0.945 | 0.950 | 0.884 |
| 0.25 | TF-IDF | tfidf-tf25 | 0.970 | 0.982 | 0.918 |
| 0.5 | R7 typed | kfold_x86_gnn-tf50 | 0.956 | 0.969 | 0.874 |
| 0.5 | TF-IDF | tfidf-tf50 | 0.978 | 0.987 | 0.903 |
| 1.0 | R7 typed | kfold_x86_gnn | 0.964 | 0.974 | 0.905 |
| 1.0 | TF-IDF | tfidf | 0.982 | 0.990 | 0.935 |

