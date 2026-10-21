# Gate summary, stage team (seed 0 only, not the pre-registered test)

kfold x86 on the v2 cache, 15% block masking on every GNN arm of ours. `ours b500` is `--subgraph cap --budget 500 --tag b500` (whole functions ranked non-library, API references, size, up to 500 blocks per binary: the budget the teammates' pipelines use), typed and with `--control no_edges`. `team gnn500` is the teammates' BlockGAT on their 57-d opcode / API-group / structure features, `team cb500` the same GAT on the CodeBERT vectors `ours_cap0_codebert` of the same 500 blocks, `team sig500` their edge-typed GAT on the 141-d subgraph features (`experiments/team_pipelines`), `team sig500v2` the same trainer on a rebuild of those subgraphs with Yanping's streaming `preprocess_full` (main c7899f0, including the two listings sig500 had to leave out; sig500 against sig500v2 is read on the binaries both have), `team sig500v3` the same trainer on a third build with the `endp` and library-function fixes (source main-fix, 8a971b7's s02 pins; `b_sig500_v3`), whose two ablations feed the GAT only the 77 handcrafted block features (`feat`, Model 2: block features + CFG) or only the 64-d pooled channel embedding (`emb`, Model 3: block embeddings + CFG); every v3 pair is read on the binaries both arms have, all trained on our folds and validation split. `team sigwhole`, `sigwhole feat` and `sigwhole emb` are the same three arms on the uncapped rebuild (`b_sigwhole_v3`, every non-library block of every binary, trained with a per-batch node budget), paired against their 500-block v3 counterparts and against R7; `fivechannel-L512 whole` is Janice's five-channel Model 1 (main's code, 512 instructions) on the same uncapped build. The pooled table is seed 0; the paired differences, second minus first for the AUC checkpoint at its validation threshold, use the seeds complete in both arms (queue seeds [0, 1, 2]), with the 95% paired cluster bootstrap interval (2000 resamples). Pre-registered rule (PLAN_GNN.md section 5): a second arm beats the first when dAUC >= 0.01 and dmacro-F1 >= 0.02 pooled, both 95% intervals excluding zero, over 3 seeds x 5 folds; TF-IDF and the header-imports LR exist for seed 0 only, so the pairs against TF-IDF carry one seed and the word in the rule column is a direction. The cost of the 500-block cap (R7 whole graph against ours b500 typed) is read as non-inferiority: lower bound of dAUC above -0.01 and of dmacro-F1 above -0.02. A missing arm or fold is stated.

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
| team sig500v3 | kfold_x86_gnn_team-sig500v3 | 0.959 | 0.979 | 0.889 |
| team sig500v3 feat (Model 2) | kfold_x86_gnn_team-sig500v3-feat | 0.936 | 0.967 | 0.842 |
| team sig500v3 emb (Model 3) | kfold_x86_gnn_team-sig500v3-emb | 0.971 | 0.985 | 0.935 |
| team sigwhole | kfold_x86_gnn_team-sigwhole | 0.969 | 0.981 | 0.912 |
| team sigwhole feat (Model 2) | kfold_x86_gnn_team-sigwhole-feat | 0.971 | 0.983 | 0.922 |
| team sigwhole emb (Model 3) | kfold_x86_gnn_team-sigwhole-emb | 0.986 | 0.992 | 0.912 |
| fivechannel-L512 whole | kfold_x86_fivechannel-L512_whole | 0.910 | 0.948 | 0.846 |
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
| team sig500v3 | 0.959 | 0.979 | 0.889 | 0.893 | 0.963 | 0.862 | 978 | 156 | 38 | 638 |
| team sig500v3 feat (Model 2) | 0.936 | 0.967 | 0.842 | 0.849 | 0.903 | 0.850 | 964 | 170 | 103 | 573 |
| team sig500v3 emb (Model 3) | 0.971 | 0.985 | 0.935 | 0.939 | 0.966 | 0.935 | 1060 | 74 | 37 | 639 |
| team sigwhole | 0.969 | 0.981 | 0.912 | 0.915 | 0.967 | 0.896 | 1016 | 118 | 35 | 641 |
| team sigwhole feat (Model 2) | 0.971 | 0.983 | 0.922 | 0.926 | 0.962 | 0.918 | 1041 | 93 | 41 | 635 |
| team sigwhole emb (Model 3) | 0.986 | 0.992 | 0.912 | 0.916 | 0.967 | 0.897 | 1017 | 117 | 35 | 641 |
| fivechannel-L512 whole | 0.910 | 0.948 | 0.846 | 0.850 | 0.946 | 0.807 | 915 | 219 | 52 | 624 |
| path 2: typed, CodeBERT vectors | 0.961 | 0.979 | 0.937 | 0.941 | 0.958 | 0.947 | 1074 | 60 | 47 | 629 |

Match rates of `b_sig500_v3` (share of each binary's selected instructions paired with a listing line): ransomware 1134 binaries, mean 0.997, below 50% 0 (0.0%), below 90% 7 (0.6%), exactly 0 0.0%; goodware 676 binaries, mean 0.995, below 50% 0 (0.0%), below 90% 2 (0.3%), exactly 0 0.0%.

Match rates of `b_sigwhole_v3` (share of each binary's selected instructions paired with a listing line): ransomware 1134 binaries, mean 0.991, below 50% 0 (0.0%), below 90% 6 (0.5%), exactly 0 0.0%; goodware 676 binaries, mean 0.994, below 50% 0 (0.0%), below 90% 2 (0.3%), exactly 0 0.0%.

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
| team sig500v2 | team sig500v3 | -0.000 [-0.022, +0.019] | +0.017 [-0.014, +0.046] | does not beat |
| team sig500 | team sig500v3 | -0.001 [-0.023, +0.018] | +0.008 [-0.020, +0.035] | does not beat |
| R7 typed, whole graph | team sig500v3 | -0.031 [-0.064, -0.006] | -0.042 [-0.095, +0.012] | does not beat |
| R7 typed, whole graph | team sig500v3 feat (Model 2) | -0.024 [-0.040, -0.010] | -0.057 [-0.121, +0.003] | does not beat |
| R7 typed, whole graph | team sig500v3 emb (Model 3) | -0.012 [-0.026, -0.000] | -0.026 [-0.075, +0.028] | does not beat |
| ours b500 no-edges | team sig500v3 | +0.003 [-0.020, +0.024] | -0.026 [-0.101, +0.050] | does not beat (seeds [0] only: a direction, not the test) |
| ours b500 no-edges | team sig500v3 feat (Model 2) | -0.020 [-0.049, +0.007] | -0.073 [-0.151, +0.002] | does not beat (seeds [0] only: a direction, not the test) |
| ours b500 no-edges | team sig500v3 emb (Model 3) | +0.016 [+0.000, +0.031] | +0.020 [-0.030, +0.077] | does not beat (seeds [0] only: a direction, not the test) |
| team sig500v3 feat (Model 2) | team sig500v3 emb (Model 3) | +0.012 [+0.000, +0.025] | +0.031 [+0.002, +0.069] | beats |
| team sig500v3 | team sigwhole | +0.022 [+0.002, +0.052] | +0.006 [-0.039, +0.056] | does not beat |
| team sig500v3 feat (Model 2) | team sigwhole feat (Model 2) | +0.015 [+0.003, +0.029] | +0.050 [+0.006, +0.108] | beats |
| team sig500v3 emb (Model 3) | team sigwhole emb (Model 3) | +0.014 [+0.004, +0.025] | -0.009 [-0.034, +0.015] | does not beat |
| R7 typed, whole graph | team sigwhole | -0.009 [-0.023, +0.002] | -0.036 [-0.074, -0.002] | does not beat |
| R7 typed, whole graph | team sigwhole feat (Model 2) | -0.009 [-0.021, +0.001] | -0.007 [-0.046, +0.038] | does not beat |
| R7 typed, whole graph | team sigwhole emb (Model 3) | +0.002 [-0.005, +0.009] | -0.035 [-0.091, +0.019] | does not beat |
| R7 typed, whole graph | fivechannel-L512 whole | -0.060 [-0.108, -0.020] | -0.072 [-0.134, -0.022] | does not beat |
| team sigwhole feat (Model 2) | team sigwhole emb (Model 3) | +0.011 [+0.002, +0.022] | -0.028 [-0.069, +0.001] | does not beat |

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

### kfold_x86_gnn_team-sig500v3 minus kfold_x86_gnn_team-sig500v2 (selection auc, 2000 resamples)

a = kfold_x86_gnn_team-sig500v2, b = kfold_x86_gnn_team-sig500v3; seeds [0, 1, 2]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.948 | 0.948 | -0.000 | [-0.022, +0.019] |
| PR-AUC | 0.972 | 0.973 | +0.002 | [-0.008, +0.013] |
| macro-F1 | 0.876 | 0.893 | +0.017 | [-0.014, +0.046] |

### kfold_x86_gnn_team-sig500v3 minus kfold_x86_gnn_team-sig500 (selection auc, 2000 resamples)

a = kfold_x86_gnn_team-sig500, b = kfold_x86_gnn_team-sig500v3; seeds [0, 1, 2]; 1808 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

2 excluded from both arms: bbbf38de4f40754f235441a8e6a4c8bdb9365dab7f5cfcdac77dbb4d6236360b, e8a0e80dfc520bf7e76c33a90ed6d286e8729e9defe6bb7da2f38bc2db33f399 (process_sample raises MemoryError reading a 341 MB asm.gz listing (the two largest listings, both 341 MB; peak 20.2 GB RSS, also when run alone)). The differences are on the 1808 binaries both arms have.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.949 | 0.948 | -0.001 | [-0.023, +0.018] |
| PR-AUC | 0.972 | 0.973 | +0.001 | [-0.009, +0.013] |
| macro-F1 | 0.886 | 0.893 | +0.008 | [-0.020, +0.035] |

### kfold_x86_gnn_team-sig500v3 minus kfold_x86_gnn-mblk15 (selection auc, 2000 resamples)

a = kfold_x86_gnn-mblk15, b = kfold_x86_gnn_team-sig500v3; seeds [0, 1, 2]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.979 | 0.948 | -0.031 | [-0.064, -0.006] |
| PR-AUC | 0.987 | 0.973 | -0.013 | [-0.032, -0.001] |
| macro-F1 | 0.935 | 0.893 | -0.042 | [-0.095, +0.012] |

### kfold_x86_gnn_team-sig500v3-feat minus kfold_x86_gnn-mblk15 (selection auc, 2000 resamples)

a = kfold_x86_gnn-mblk15, b = kfold_x86_gnn_team-sig500v3-feat; seeds [0, 1, 2]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.979 | 0.954 | -0.024 | [-0.040, -0.010] |
| PR-AUC | 0.987 | 0.976 | -0.011 | [-0.022, -0.002] |
| macro-F1 | 0.935 | 0.879 | -0.057 | [-0.121, +0.003] |

### kfold_x86_gnn_team-sig500v3-emb minus kfold_x86_gnn-mblk15 (selection auc, 2000 resamples)

a = kfold_x86_gnn-mblk15, b = kfold_x86_gnn_team-sig500v3-emb; seeds [0, 1, 2]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.979 | 0.966 | -0.012 | [-0.026, -0.000] |
| PR-AUC | 0.987 | 0.982 | -0.005 | [-0.016, +0.003] |
| macro-F1 | 0.935 | 0.909 | -0.026 | [-0.075, +0.028] |

### kfold_x86_gnn_team-sig500v3 minus kfold_x86_gnn-noedges-mblk15-sg-cap500_b500 (selection auc, 2000 resamples)

a = kfold_x86_gnn-noedges-mblk15-sg-cap500_b500, b = kfold_x86_gnn_team-sig500v3; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.955 | 0.959 | +0.003 | [-0.020, +0.024] |
| PR-AUC | 0.964 | 0.979 | +0.015 | [-0.000, +0.038] |
| macro-F1 | 0.915 | 0.889 | -0.026 | [-0.101, +0.050] |

### kfold_x86_gnn_team-sig500v3-feat minus kfold_x86_gnn-noedges-mblk15-sg-cap500_b500 (selection auc, 2000 resamples)

a = kfold_x86_gnn-noedges-mblk15-sg-cap500_b500, b = kfold_x86_gnn_team-sig500v3-feat; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.955 | 0.936 | -0.020 | [-0.049, +0.007] |
| PR-AUC | 0.964 | 0.967 | +0.004 | [-0.016, +0.028] |
| macro-F1 | 0.915 | 0.842 | -0.073 | [-0.151, +0.002] |

### kfold_x86_gnn_team-sig500v3-emb minus kfold_x86_gnn-noedges-mblk15-sg-cap500_b500 (selection auc, 2000 resamples)

a = kfold_x86_gnn-noedges-mblk15-sg-cap500_b500, b = kfold_x86_gnn_team-sig500v3-emb; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.955 | 0.971 | +0.016 | [+0.000, +0.031] |
| PR-AUC | 0.964 | 0.985 | +0.021 | [+0.008, +0.044] |
| macro-F1 | 0.915 | 0.935 | +0.020 | [-0.030, +0.077] |

### kfold_x86_gnn_team-sig500v3-emb minus kfold_x86_gnn_team-sig500v3-feat (selection auc, 2000 resamples)

a = kfold_x86_gnn_team-sig500v3-feat, b = kfold_x86_gnn_team-sig500v3-emb; seeds [0, 1, 2]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.954 | 0.966 | +0.012 | [+0.000, +0.025] |
| PR-AUC | 0.976 | 0.982 | +0.006 | [-0.001, +0.015] |
| macro-F1 | 0.879 | 0.909 | +0.031 | [+0.002, +0.069] |

### kfold_x86_gnn_team-sigwhole minus kfold_x86_gnn_team-sig500v3 (selection auc, 2000 resamples)

a = kfold_x86_gnn_team-sig500v3, b = kfold_x86_gnn_team-sigwhole; seeds [0, 1, 2]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.948 | 0.970 | +0.022 | [+0.002, +0.052] |
| PR-AUC | 0.973 | 0.982 | +0.008 | [-0.003, +0.024] |
| macro-F1 | 0.893 | 0.899 | +0.006 | [-0.039, +0.056] |

### kfold_x86_gnn_team-sigwhole-feat minus kfold_x86_gnn_team-sig500v3-feat (selection auc, 2000 resamples)

a = kfold_x86_gnn_team-sig500v3-feat, b = kfold_x86_gnn_team-sigwhole-feat; seeds [0, 1, 2]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.954 | 0.970 | +0.015 | [+0.003, +0.029] |
| PR-AUC | 0.976 | 0.983 | +0.007 | [-0.000, +0.016] |
| macro-F1 | 0.879 | 0.929 | +0.050 | [+0.006, +0.108] |

### kfold_x86_gnn_team-sigwhole-emb minus kfold_x86_gnn_team-sig500v3-emb (selection auc, 2000 resamples)

a = kfold_x86_gnn_team-sig500v3-emb, b = kfold_x86_gnn_team-sigwhole-emb; seeds [0, 1, 2]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.966 | 0.981 | +0.014 | [+0.004, +0.025] |
| PR-AUC | 0.982 | 0.989 | +0.007 | [+0.001, +0.015] |
| macro-F1 | 0.909 | 0.901 | -0.009 | [-0.034, +0.015] |

### kfold_x86_gnn_team-sigwhole minus kfold_x86_gnn-mblk15 (selection auc, 2000 resamples)

a = kfold_x86_gnn-mblk15, b = kfold_x86_gnn_team-sigwhole; seeds [0, 1, 2]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.979 | 0.970 | -0.009 | [-0.023, +0.002] |
| PR-AUC | 0.987 | 0.982 | -0.005 | [-0.016, +0.002] |
| macro-F1 | 0.935 | 0.899 | -0.036 | [-0.074, -0.002] |

### kfold_x86_gnn_team-sigwhole-feat minus kfold_x86_gnn-mblk15 (selection auc, 2000 resamples)

a = kfold_x86_gnn-mblk15, b = kfold_x86_gnn_team-sigwhole-feat; seeds [0, 1, 2]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.979 | 0.970 | -0.009 | [-0.021, +0.001] |
| PR-AUC | 0.987 | 0.983 | -0.004 | [-0.013, +0.003] |
| macro-F1 | 0.935 | 0.929 | -0.007 | [-0.046, +0.038] |

### kfold_x86_gnn_team-sigwhole-emb minus kfold_x86_gnn-mblk15 (selection auc, 2000 resamples)

a = kfold_x86_gnn-mblk15, b = kfold_x86_gnn_team-sigwhole-emb; seeds [0, 1, 2]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.979 | 0.981 | +0.002 | [-0.005, +0.009] |
| PR-AUC | 0.987 | 0.989 | +0.002 | [-0.003, +0.008] |
| macro-F1 | 0.935 | 0.901 | -0.035 | [-0.091, +0.019] |

### kfold_x86_fivechannel-L512_whole minus kfold_x86_gnn-mblk15 (selection auc, 2000 resamples)

a = kfold_x86_gnn-mblk15, b = kfold_x86_fivechannel-L512_whole; seeds [0, 1, 2]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.979 | 0.919 | -0.060 | [-0.108, -0.020] |
| PR-AUC | 0.987 | 0.955 | -0.032 | [-0.063, -0.012] |
| macro-F1 | 0.935 | 0.863 | -0.072 | [-0.134, -0.022] |

### kfold_x86_gnn_team-sigwhole-emb minus kfold_x86_gnn_team-sigwhole-feat (selection auc, 2000 resamples)

a = kfold_x86_gnn_team-sigwhole-feat, b = kfold_x86_gnn_team-sigwhole-emb; seeds [0, 1, 2]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.970 | 0.981 | +0.011 | [+0.002, +0.022] |
| PR-AUC | 0.983 | 0.989 | +0.006 | [+0.000, +0.014] |
| macro-F1 | 0.929 | 0.901 | -0.028 | [-0.069, +0.001] |

