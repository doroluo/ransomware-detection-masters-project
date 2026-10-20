# Gate summary, stage team (seed 0 only, not the pre-registered test)

kfold x86 on the v2 cache, 15% block masking on every GNN arm of ours. `ours b500` is `--subgraph cap --budget 500 --tag b500` (whole functions ranked non-library, API references, size, up to 500 blocks per binary: the budget the teammates' pipelines use), typed and with `--control no_edges`. `team gnn500` is the teammates' BlockGAT on their 57-d opcode / API-group / structure features, `team cb500` the same GAT on the CodeBERT vectors `ours_cap0_codebert` of the same 500 blocks, `team sig500` their edge-typed GAT on the 141-d subgraph features (`experiments/team_pipelines`), `team sig500v2` the same trainer on a rebuild of those subgraphs with Yanping's streaming `preprocess_full` (main c7899f0, including the two listings sig500 had to leave out; sig500 against sig500v2 is read on the binaries both have), all trained on our folds and validation split. The pooled table is seed 0; the paired differences, second minus first for the AUC checkpoint at its validation threshold, use the seeds complete in both arms (queue seeds [0, 1, 2]), with the 95% paired cluster bootstrap interval (2000 resamples). Pre-registered rule (PLAN_GNN.md section 5): a second arm beats the first when dAUC >= 0.01 and dmacro-F1 >= 0.02 pooled, both 95% intervals excluding zero, over 3 seeds x 5 folds; TF-IDF and the header-imports LR exist for seed 0 only, so the pairs against TF-IDF carry one seed and the word in the rule column is a direction. The cost of the 500-block cap (R7 whole graph against ours b500 typed) is read as non-inferiority: lower bound of dAUC above -0.01 and of dmacro-F1 above -0.02. A missing arm or fold is stated.

## Pooled, AUC checkpoint

| arm | group | AUC | PR-AUC | macro-F1 |
|---|---|---|---|---|
| TF-IDF | tfidf | 0.982 | 0.990 | 0.935 |
| header-imports LR | imports-header | 0.985 | 0.991 | 0.954 |
| R7 typed, whole graph | kfold_x86_gnn-mblk15 | 0.986 | 0.990 | 0.935 |
| R2 no-edges, whole graph | kfold_x86_gnn-noedges-mblk15 | 0.977 | 0.983 | 0.941 |
| ours b500 typed | kfold_x86_gnn-mblk15-sg-cap500_b500 | 0.923 | 0.954 | 0.843 |
| ours b500 no-edges | kfold_x86_gnn-noedges-mblk15-sg-cap500_b500 | 0.955 | 0.964 | 0.915 |
| team gnn500 | kfold_x86_gnn_team-gnn500 | 0.956 | 0.975 | 0.850 |
| team cb500 | kfold_x86_gnn_team-cb500 | 0.791 | 0.872 | 0.723 |
| team sig500 | kfold_x86_gnn_team-sig500 | 0.953 | 0.974 | 0.894 |
| team sig500v2 | kfold_x86_gnn_team-sig500v2 | 0.948 | 0.972 | 0.874 |
| path 2: typed, CodeBERT vectors | kfold_x86_gnn-mblk15-bv-ours_cap0_codebert | 0.961 | 0.979 | 0.937 |

Operating point of each arm, seed 0 pooled over the five folds, AUC checkpoint at its validation threshold (TF-IDF and the header-imports LR: their own validation threshold). Ransomware is the positive class; TP / FN are ransomware binaries called ransomware / goodware, FP / TN goodware binaries called ransomware / goodware.

| arm | AUC | PR-AUC | macro-F1 | accuracy | precision | recall | TP | FN | FP | TN |
|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF | 0.982 | 0.990 | 0.935 | 0.939 | 0.951 | 0.951 | 1079 | 55 | 55 | 621 |
| header-imports LR | 0.985 | 0.991 | 0.954 | 0.957 | 0.971 | 0.960 | 1089 | 45 | 33 | 643 |
| R7 typed, whole graph | 0.986 | 0.990 | 0.935 | 0.938 | 0.965 | 0.935 | 1060 | 74 | 38 | 638 |
| R2 no-edges, whole graph | 0.977 | 0.983 | 0.941 | 0.945 | 0.964 | 0.947 | 1074 | 60 | 40 | 636 |
| ours b500 typed | 0.923 | 0.954 | 0.843 | 0.848 | 0.932 | 0.817 | 926 | 208 | 68 | 608 |
| ours b500 no-edges | 0.955 | 0.964 | 0.915 | 0.919 | 0.959 | 0.910 | 1032 | 102 | 44 | 632 |
| team gnn500 | 0.956 | 0.975 | 0.850 | 0.854 | 0.939 | 0.820 | 930 | 204 | 60 | 616 |
| team cb500 | 0.791 | 0.872 | 0.723 | 0.724 | 0.925 | 0.608 | 690 | 444 | 56 | 620 |
| team sig500 | 0.953 | 0.974 | 0.894 | 0.898 | 0.962 | 0.871 | 986 | 146 | 39 | 637 |
| team sig500v2 | 0.948 | 0.972 | 0.874 | 0.886 | 0.881 | 0.945 | 1072 | 62 | 145 | 531 |
| path 2: typed, CodeBERT vectors | 0.961 | 0.979 | 0.937 | 0.941 | 0.958 | 0.947 | 1074 | 60 | 47 | 629 |

## Paired differences, second minus first

| first | second | dAUC [95% CI] | dmacro-F1 [95% CI] | rule |
|---|---|---|---|---|
| TF-IDF | team gnn500 | -0.026 [-0.043, -0.013] | -0.085 [-0.165, -0.021] | does not beat (seeds [0] only: a direction, not the test) |
| TF-IDF | team cb500 | -0.190 [-0.273, -0.122] | -0.212 [-0.311, -0.115] | does not beat (seeds [0] only: a direction, not the test) |
| TF-IDF | team sig500 | -0.028 [-0.050, -0.011] | -0.041 [-0.111, +0.017] | does not beat (seeds [0] only: a direction, not the test) |
| R7 typed, whole graph | team gnn500 | -0.025 [-0.043, -0.009] | -0.072 [-0.136, -0.008] | does not beat |
| R7 typed, whole graph | team cb500 | -0.159 [-0.221, -0.104] | -0.184 [-0.242, -0.129] | does not beat |
| R7 typed, whole graph | team sig500 | -0.029 [-0.054, -0.010] | -0.050 [-0.118, +0.014] | does not beat |
| ours b500 typed | team gnn500 | +0.032 [-0.002, +0.071] | +0.007 [-0.069, +0.081] | does not beat (seeds [0] only: a direction, not the test) |
| ours b500 typed | team cb500 | -0.132 [-0.220, -0.059] | -0.119 [-0.205, -0.042] | does not beat (seeds [0] only: a direction, not the test) |
| ours b500 typed | team sig500 | +0.030 [-0.003, +0.068] | +0.051 [+0.013, +0.104] | does not beat (seeds [0] only: a direction, not the test) |
| path 2: typed, CodeBERT vectors | team cb500 | -0.169 [-0.255, -0.095] | -0.214 [-0.313, -0.113] | does not beat (seeds [0] only: a direction, not the test) |
| R7 typed, whole graph | ours b500 typed | -0.063 [-0.105, -0.029] | -0.092 [-0.182, -0.003] | not shown non-inferior |
| team sig500 | team sig500v2 | -0.001 [-0.011, +0.009] | -0.009 [-0.039, +0.029] | does not beat |
| R7 typed, whole graph | team sig500v2 | -0.031 [-0.058, -0.009] | -0.059 [-0.105, -0.007] | does not beat |
| TF-IDF | team sig500v2 | -0.033 [-0.075, -0.003] | -0.061 [-0.102, -0.025] | does not beat (seeds [0] only: a direction, not the test) |

### kfold_x86_gnn_team-gnn500 minus tfidf (selection auc, 2000 resamples)

a = tfidf, b = kfold_x86_gnn_team-gnn500; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.982 | 0.956 | -0.026 | [-0.043, -0.013] |
| PR-AUC | 0.990 | 0.975 | -0.016 | [-0.032, -0.006] |
| macro-F1 | 0.935 | 0.850 | -0.085 | [-0.165, -0.021] |

### kfold_x86_gnn_team-cb500 minus tfidf (selection auc, 2000 resamples)

a = tfidf, b = kfold_x86_gnn_team-cb500; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.982 | 0.791 | -0.190 | [-0.273, -0.122] |
| PR-AUC | 0.990 | 0.872 | -0.118 | [-0.196, -0.072] |
| macro-F1 | 0.935 | 0.723 | -0.212 | [-0.311, -0.115] |

### kfold_x86_gnn_team-sig500 minus tfidf (selection auc, 2000 resamples)

a = tfidf, b = kfold_x86_gnn_team-sig500; seeds [0]; 1808 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

2 excluded from both arms: bbbf38de4f40754f235441a8e6a4c8bdb9365dab7f5cfcdac77dbb4d6236360b, e8a0e80dfc520bf7e76c33a90ed6d286e8729e9defe6bb7da2f38bc2db33f399 (process_sample raises MemoryError reading a 341 MB asm.gz listing (the two largest listings, both 341 MB; peak 20.2 GB RSS, also when run alone)). The differences are on the 1808 binaries both arms have.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.982 | 0.953 | -0.028 | [-0.050, -0.011] |
| PR-AUC | 0.990 | 0.974 | -0.016 | [-0.033, -0.006] |
| macro-F1 | 0.935 | 0.894 | -0.041 | [-0.111, +0.017] |

### kfold_x86_gnn_team-gnn500 minus kfold_x86_gnn-mblk15 (selection auc, 2000 resamples)

a = kfold_x86_gnn-mblk15, b = kfold_x86_gnn_team-gnn500; seeds [0, 1, 2]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.979 | 0.954 | -0.025 | [-0.043, -0.009] |
| PR-AUC | 0.987 | 0.973 | -0.014 | [-0.031, -0.003] |
| macro-F1 | 0.935 | 0.863 | -0.072 | [-0.136, -0.008] |

### kfold_x86_gnn_team-cb500 minus kfold_x86_gnn-mblk15 (selection auc, 2000 resamples)

a = kfold_x86_gnn-mblk15, b = kfold_x86_gnn_team-cb500; seeds [0, 1, 2]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.979 | 0.820 | -0.159 | [-0.221, -0.104] |
| PR-AUC | 0.987 | 0.893 | -0.094 | [-0.152, -0.059] |
| macro-F1 | 0.935 | 0.751 | -0.184 | [-0.242, -0.129] |

### kfold_x86_gnn_team-sig500 minus kfold_x86_gnn-mblk15 (selection auc, 2000 resamples)

a = kfold_x86_gnn-mblk15, b = kfold_x86_gnn_team-sig500; seeds [0, 1, 2]; 1808 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

2 excluded from both arms: bbbf38de4f40754f235441a8e6a4c8bdb9365dab7f5cfcdac77dbb4d6236360b, e8a0e80dfc520bf7e76c33a90ed6d286e8729e9defe6bb7da2f38bc2db33f399 (process_sample raises MemoryError reading a 341 MB asm.gz listing (the two largest listings, both 341 MB; peak 20.2 GB RSS, also when run alone)). The differences are on the 1808 binaries both arms have.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.979 | 0.949 | -0.029 | [-0.054, -0.010] |
| PR-AUC | 0.987 | 0.972 | -0.014 | [-0.032, -0.004] |
| macro-F1 | 0.935 | 0.886 | -0.050 | [-0.118, +0.014] |

### kfold_x86_gnn_team-gnn500 minus kfold_x86_gnn-mblk15-sg-cap500_b500 (selection auc, 2000 resamples)

a = kfold_x86_gnn-mblk15-sg-cap500_b500, b = kfold_x86_gnn_team-gnn500; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.923 | 0.956 | +0.032 | [-0.002, +0.071] |
| PR-AUC | 0.954 | 0.975 | +0.021 | [-0.000, +0.050] |
| macro-F1 | 0.843 | 0.850 | +0.007 | [-0.069, +0.081] |

### kfold_x86_gnn_team-cb500 minus kfold_x86_gnn-mblk15-sg-cap500_b500 (selection auc, 2000 resamples)

a = kfold_x86_gnn-mblk15-sg-cap500_b500, b = kfold_x86_gnn_team-cb500; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.923 | 0.791 | -0.132 | [-0.220, -0.059] |
| PR-AUC | 0.954 | 0.872 | -0.082 | [-0.152, -0.037] |
| macro-F1 | 0.843 | 0.723 | -0.119 | [-0.205, -0.042] |

### kfold_x86_gnn_team-sig500 minus kfold_x86_gnn-mblk15-sg-cap500_b500 (selection auc, 2000 resamples)

a = kfold_x86_gnn-mblk15-sg-cap500_b500, b = kfold_x86_gnn_team-sig500; seeds [0]; 1808 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

2 excluded from both arms: bbbf38de4f40754f235441a8e6a4c8bdb9365dab7f5cfcdac77dbb4d6236360b, e8a0e80dfc520bf7e76c33a90ed6d286e8729e9defe6bb7da2f38bc2db33f399 (process_sample raises MemoryError reading a 341 MB asm.gz listing (the two largest listings, both 341 MB; peak 20.2 GB RSS, also when run alone)). The differences are on the 1808 binaries both arms have.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.923 | 0.953 | +0.030 | [-0.003, +0.068] |
| PR-AUC | 0.954 | 0.974 | +0.020 | [+0.001, +0.048] |
| macro-F1 | 0.843 | 0.894 | +0.051 | [+0.013, +0.104] |

### kfold_x86_gnn_team-cb500 minus kfold_x86_gnn-mblk15-bv-ours_cap0_codebert (selection auc, 2000 resamples)

a = kfold_x86_gnn-mblk15-bv-ours_cap0_codebert, b = kfold_x86_gnn_team-cb500; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.961 | 0.791 | -0.169 | [-0.255, -0.095] |
| PR-AUC | 0.979 | 0.872 | -0.107 | [-0.180, -0.062] |
| macro-F1 | 0.937 | 0.723 | -0.214 | [-0.313, -0.113] |

### kfold_x86_gnn-mblk15-sg-cap500_b500 minus kfold_x86_gnn-mblk15 (selection auc, 2000 resamples)

a = kfold_x86_gnn-mblk15, b = kfold_x86_gnn-mblk15-sg-cap500_b500; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.986 | 0.923 | -0.063 | [-0.105, -0.029] |
| PR-AUC | 0.990 | 0.954 | -0.036 | [-0.072, -0.016] |
| macro-F1 | 0.935 | 0.843 | -0.092 | [-0.182, -0.003] |

### kfold_x86_gnn_team-sig500v2 minus kfold_x86_gnn_team-sig500 (selection auc, 2000 resamples)

a = kfold_x86_gnn_team-sig500, b = kfold_x86_gnn_team-sig500v2; seeds [0, 1, 2]; 1808 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

2 excluded from both arms: bbbf38de4f40754f235441a8e6a4c8bdb9365dab7f5cfcdac77dbb4d6236360b, e8a0e80dfc520bf7e76c33a90ed6d286e8729e9defe6bb7da2f38bc2db33f399 (process_sample raises MemoryError reading a 341 MB asm.gz listing (the two largest listings, both 341 MB; peak 20.2 GB RSS, also when run alone)). The differences are on the 1808 binaries both arms have.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.949 | 0.948 | -0.001 | [-0.011, +0.009] |
| PR-AUC | 0.972 | 0.972 | -0.000 | [-0.006, +0.005] |
| macro-F1 | 0.886 | 0.876 | -0.009 | [-0.039, +0.029] |

### kfold_x86_gnn_team-sig500v2 minus kfold_x86_gnn-mblk15 (selection auc, 2000 resamples)

a = kfold_x86_gnn-mblk15, b = kfold_x86_gnn_team-sig500v2; seeds [0, 1, 2]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.979 | 0.948 | -0.031 | [-0.058, -0.009] |
| PR-AUC | 0.987 | 0.972 | -0.015 | [-0.034, -0.003] |
| macro-F1 | 0.935 | 0.876 | -0.059 | [-0.105, -0.007] |

### kfold_x86_gnn_team-sig500v2 minus tfidf (selection auc, 2000 resamples)

a = tfidf, b = kfold_x86_gnn_team-sig500v2; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.982 | 0.948 | -0.033 | [-0.075, -0.003] |
| PR-AUC | 0.990 | 0.972 | -0.018 | [-0.041, -0.004] |
| macro-F1 | 0.935 | 0.874 | -0.061 | [-0.102, -0.025] |

