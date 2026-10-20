# Gate summary, stage paths_codebert (seed 0 only, not the pre-registered test)

kfold x86 on the v2 cache, 15% block masking on every GNN arm. Paths 1 and 2 train on the frozen CodeBERT vectors `ours_cap0_codebert`: `microsoft/codebert-base` on the text of each block, built from the signature and API strings of our block export and never from operands (`transformer_paths/codebert_encoder.py`); path 1 has no edges, path 2 has the typed edges. The pooled table is seed 0; the paired differences, second minus first for the AUC checkpoint at its validation threshold, are over the seeds complete in both arms (queue seeds [0, 1, 2]), with the 95% paired cluster bootstrap interval (2000 resamples). A missing arm or fold is stated.

## Pooled, AUC checkpoint

| arm | group | AUC | PR-AUC | macro-F1 |
|---|---|---|---|---|
| TF-IDF | tfidf | 0.982 | 0.990 | 0.935 |
| R2 no-edges, tokens | kfold_x86_gnn-noedges-mblk15 | 0.977 | 0.983 | 0.941 |
| R7 typed, tokens | kfold_x86_gnn-mblk15 | 0.986 | 0.990 | 0.935 |
| path 1: no-edges, CodeBERT vectors | kfold_x86_gnn-noedges-mblk15-bv-ours_cap0_codebert | missing (seed 0 folds present [] of 0-4) |  |  |
| path 2: typed, CodeBERT vectors | kfold_x86_gnn-mblk15-bv-ours_cap0_codebert | 0.961 | 0.979 | 0.937 |

## Paired differences, second minus first

| first | second | dAUC [95% CI] | dmacro-F1 [95% CI] |
|---|---|---|---|
| path 1: no-edges, CodeBERT vectors | path 2: typed, CodeBERT vectors | missing | missing |
| R2 no-edges, tokens | path 1: no-edges, CodeBERT vectors | missing | missing |
| R7 typed, tokens | path 2: typed, CodeBERT vectors | -0.025 [-0.059, -0.005] | +0.003 [-0.042, +0.066] |
| TF-IDF | path 1: no-edges, CodeBERT vectors | missing | missing |
| TF-IDF | path 2: typed, CodeBERT vectors | -0.021 [-0.056, -0.000] | +0.002 [-0.029, +0.030] |

**path 2: typed, CodeBERT vectors vs path 1: no-edges, CodeBERT vectors** (selection auc): missing: kfold_x86_gnn-noedges-mblk15-bv-ours_cap0_codebert: no complete seed among seeds [0, 1, 2] with all five folds

**path 1: no-edges, CodeBERT vectors vs R2 no-edges, tokens** (selection auc): missing: kfold_x86_gnn-noedges-mblk15-bv-ours_cap0_codebert: no complete seed among seeds [0, 1, 2] with all five folds

### kfold_x86_gnn-mblk15-bv-ours_cap0_codebert minus kfold_x86_gnn-mblk15 (selection auc, 2000 resamples)

a = kfold_x86_gnn-mblk15, b = kfold_x86_gnn-mblk15-bv-ours_cap0_codebert; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.986 | 0.961 | -0.025 | [-0.059, -0.005] |
| PR-AUC | 0.990 | 0.979 | -0.011 | [-0.030, -0.001] |
| macro-F1 | 0.935 | 0.937 | +0.003 | [-0.042, +0.066] |

**path 1: no-edges, CodeBERT vectors vs TF-IDF** (selection auc): missing: kfold_x86_gnn-noedges-mblk15-bv-ours_cap0_codebert: no complete seed among seeds [0, 1, 2] with all five folds

### kfold_x86_gnn-mblk15-bv-ours_cap0_codebert minus tfidf (selection auc, 2000 resamples)

a = tfidf, b = kfold_x86_gnn-mblk15-bv-ours_cap0_codebert; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.982 | 0.961 | -0.021 | [-0.056, -0.000] |
| PR-AUC | 0.990 | 0.979 | -0.011 | [-0.030, -0.001] |
| macro-F1 | 0.935 | 0.937 | +0.002 | [-0.029, +0.030] |

