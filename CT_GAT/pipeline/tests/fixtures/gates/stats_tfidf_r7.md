### kfold_x86_gnn-mblk15 minus tfidf (selection auc, 2000 resamples)

a = tfidf, b = kfold_x86_gnn-mblk15; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.982 | 0.986 | +0.004 | [-0.005, +0.017] |
| PR-AUC | 0.990 | 0.990 | +0.000 | [-0.006, +0.006] |
| macro-F1 | 0.935 | 0.935 | -0.001 | [-0.066, +0.043] |
