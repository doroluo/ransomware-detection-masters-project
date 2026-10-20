# Gate summary, stage paths (seed 0 only, not the pre-registered test)

kfold x86 on the v2 cache, seed 0 pooled over the five folds, 15% block masking on every GNN arm. Stack: R7 with the standardised out-of-fold TF-IDF logit as an extra head input (`--stack-tfidf`). Paths 1 and 2 train on the frozen block vectors `ours_cap0_ref` of the reference encoder (random weights, a plumbing check, run on fold 0 only until the teammates deliver checkpoints); path 1 has no edges, path 2 has the typed edges. Differences are second minus first for the AUC checkpoint at its validation threshold, with the 95% paired cluster bootstrap interval (2000 resamples). A missing arm or fold is stated.

## Pooled, AUC checkpoint

| arm | group | AUC | PR-AUC | macro-F1 |
|---|---|---|---|---|
| TF-IDF | tfidf | 0.982 | 0.990 | 0.935 |
| R7 typed, tokens | kfold_x86_gnn-mblk15 | 0.986 | 0.990 | 0.935 |
| R7 + TF-IDF stack | kfold_x86_gnn-stack-mblk15 | missing (seed 0 folds present [] of 0-4) |  |  |
| R2 no-edges, tokens | kfold_x86_gnn-noedges-mblk15 | 0.977 | 0.983 | 0.941 |
| path 1: no-edges, block vectors | kfold_x86_gnn-noedges-mblk15-bv-ours_cap0_ref | missing (seed 0 folds present [] of 0-4) |  |  |
| path 2: typed, block vectors | kfold_x86_gnn-mblk15-bv-ours_cap0_ref | missing (seed 0 folds present [] of 0-4) |  |  |

## Paired differences, second minus first

| first | second | dAUC [95% CI] | dmacro-F1 [95% CI] |
|---|---|---|---|
| R7 typed, tokens | R7 + TF-IDF stack | missing | missing |
| TF-IDF | R7 + TF-IDF stack | missing | missing |
| path 1: no-edges, block vectors | path 2: typed, block vectors | missing | missing |
| R2 no-edges, tokens | path 1: no-edges, block vectors | missing | missing |

**R7 + TF-IDF stack vs R7 typed, tokens** (selection auc): missing: kfold_x86_gnn-stack-mblk15: no complete seed among seeds [0] with all five folds

**R7 + TF-IDF stack vs TF-IDF** (selection auc): missing: kfold_x86_gnn-stack-mblk15: no complete seed among seeds [0] with all five folds

**path 2: typed, block vectors vs path 1: no-edges, block vectors** (selection auc): missing: kfold_x86_gnn-noedges-mblk15-bv-ours_cap0_ref: no complete seed among seeds [0] with all five folds

**path 1: no-edges, block vectors vs R2 no-edges, tokens** (selection auc): missing: kfold_x86_gnn-noedges-mblk15-bv-ours_cap0_ref: no complete seed among seeds [0] with all five folds

