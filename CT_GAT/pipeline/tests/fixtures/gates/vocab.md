# Gate summary, stage vocab (seed 0 only, not the pre-registered test)

kfold x86 on the v2 cache, seed 0 pooled over the five folds. Vocabularies: ours (mn, sig, beh, api), yanping (the reader's three ids per instruction plus api) and hybrid (our mn and beh, the reader's two operand ids, api). GNN arms are the no-edge arm with 15% block masking (`--control no_edges --mask-blocks 0.15 --vocab`); the TF-IDF rows are `baselines.py --vocab` over the same token kinds. Differences are second minus first for the AUC checkpoint at its validation threshold, with the 95% paired cluster bootstrap interval (2000 resamples). A missing arm or fold is stated.

## Pooled, AUC checkpoint

| arm | group | AUC | PR-AUC | macro-F1 |
|---|---|---|---|---|
| TF-IDF, ours (sig + api) | tfidf | 0.982 | 0.990 | 0.935 |
| TF-IDF, yanping | tfidf-yanping | 0.981 | 0.990 | 0.938 |
| TF-IDF, hybrid | tfidf-hybrid | 0.981 | 0.989 | 0.917 |
| R2 no-edges, ours | kfold_x86_gnn-noedges-mblk15 | 0.977 | 0.983 | 0.941 |
| R2 no-edges, yanping | kfold_x86_gnn-noedges-mblk15-voc-yanping | 0.981 | 0.984 | 0.940 |
| R2 no-edges, hybrid | kfold_x86_gnn-noedges-mblk15-voc-hybrid | 0.969 | 0.977 | 0.896 |

## Paired differences, second minus first

| first | second | dAUC [95% CI] | dmacro-F1 [95% CI] |
|---|---|---|---|
| TF-IDF, ours (sig + api) | TF-IDF, yanping | -0.001 [-0.005, +0.003] | +0.003 [-0.017, +0.023] |
| TF-IDF, ours (sig + api) | TF-IDF, hybrid | -0.001 [-0.003, +0.000] | -0.018 [-0.065, +0.012] |
| R2 no-edges, ours | R2 no-edges, yanping | +0.004 [-0.007, +0.017] | -0.002 [-0.046, +0.048] |
| R2 no-edges, ours | R2 no-edges, hybrid | -0.008 [-0.021, +0.004] | -0.046 [-0.110, +0.001] |
| R2 no-edges, yanping | R2 no-edges, hybrid | -0.012 [-0.032, +0.003] | -0.044 [-0.129, +0.027] |
| TF-IDF, yanping | R2 no-edges, yanping | +0.000 [-0.014, +0.023] | +0.001 [-0.029, +0.027] |
| TF-IDF, hybrid | R2 no-edges, hybrid | -0.012 [-0.028, -0.001] | -0.021 [-0.113, +0.060] |

### tfidf-yanping minus tfidf (selection auc, 2000 resamples)

a = tfidf, b = tfidf-yanping; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.982 | 0.981 | -0.001 | [-0.005, +0.003] |
| PR-AUC | 0.990 | 0.990 | -0.001 | [-0.004, +0.003] |
| macro-F1 | 0.935 | 0.938 | +0.003 | [-0.017, +0.023] |

### tfidf-hybrid minus tfidf (selection auc, 2000 resamples)

a = tfidf, b = tfidf-hybrid; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.982 | 0.981 | -0.001 | [-0.003, +0.000] |
| PR-AUC | 0.990 | 0.989 | -0.001 | [-0.004, +0.000] |
| macro-F1 | 0.935 | 0.917 | -0.018 | [-0.065, +0.012] |

### kfold_x86_gnn-noedges-mblk15-voc-yanping minus kfold_x86_gnn-noedges-mblk15 (selection auc, 2000 resamples)

a = kfold_x86_gnn-noedges-mblk15, b = kfold_x86_gnn-noedges-mblk15-voc-yanping; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.977 | 0.981 | +0.004 | [-0.007, +0.017] |
| PR-AUC | 0.983 | 0.984 | +0.001 | [-0.008, +0.011] |
| macro-F1 | 0.941 | 0.940 | -0.002 | [-0.046, +0.048] |

### kfold_x86_gnn-noedges-mblk15-voc-hybrid minus kfold_x86_gnn-noedges-mblk15 (selection auc, 2000 resamples)

a = kfold_x86_gnn-noedges-mblk15, b = kfold_x86_gnn-noedges-mblk15-voc-hybrid; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.977 | 0.969 | -0.008 | [-0.021, +0.004] |
| PR-AUC | 0.983 | 0.977 | -0.006 | [-0.019, +0.002] |
| macro-F1 | 0.941 | 0.896 | -0.046 | [-0.110, +0.001] |

### kfold_x86_gnn-noedges-mblk15-voc-hybrid minus kfold_x86_gnn-noedges-mblk15-voc-yanping (selection auc, 2000 resamples)

a = kfold_x86_gnn-noedges-mblk15-voc-yanping, b = kfold_x86_gnn-noedges-mblk15-voc-hybrid; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.981 | 0.969 | -0.012 | [-0.032, +0.003] |
| PR-AUC | 0.984 | 0.977 | -0.007 | [-0.022, +0.003] |
| macro-F1 | 0.940 | 0.896 | -0.044 | [-0.129, +0.027] |

### kfold_x86_gnn-noedges-mblk15-voc-yanping minus tfidf-yanping (selection auc, 2000 resamples)

a = tfidf-yanping, b = kfold_x86_gnn-noedges-mblk15-voc-yanping; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.981 | 0.981 | +0.000 | [-0.014, +0.023] |
| PR-AUC | 0.990 | 0.984 | -0.006 | [-0.018, +0.004] |
| macro-F1 | 0.938 | 0.940 | +0.001 | [-0.029, +0.027] |

### kfold_x86_gnn-noedges-mblk15-voc-hybrid minus tfidf-hybrid (selection auc, 2000 resamples)

a = tfidf-hybrid, b = kfold_x86_gnn-noedges-mblk15-voc-hybrid; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.981 | 0.969 | -0.012 | [-0.028, -0.001] |
| PR-AUC | 0.989 | 0.977 | -0.012 | [-0.029, -0.003] |
| macro-F1 | 0.917 | 0.896 | -0.021 | [-0.113, +0.060] |

