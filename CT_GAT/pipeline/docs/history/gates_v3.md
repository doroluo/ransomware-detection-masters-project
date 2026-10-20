# Gate summary, stage v3 (seed 0 only, not the pre-registered test)

Combined cache, kfold_v3 (family-holdout v3 folds 0-4), seed 0 pooled over the five folds. Every GNN arm is block-masked (15% of the blocks in training, `mblk15`). Its AUC checkpoint is chosen by the validation cell-balanced AUC and its threshold by the validation cell-balanced macro-F1 (`selcell`); the F1 checkpoint rule is unchanged. The F1 checkpoint is scored at 0.5; the AUC checkpoint at its validation threshold (TF-IDF: the validation macro-F1 maximiser, per fold); recalls are at that threshold. A missing arm or fold is stated in its row. Floors: the majority class of each fold's training manifest, and the x86 rule (ransomware iff x86). `ab` arms were trained with --arch-balance (each class x arch cell weighted equally). Cell-bal columns weight each class x arch cell of the pooled test set equally; under them the x86 rule scores 0.5.

## Arch all: pooled table

| arm | group | F1 ckpt, 0.5 AUC | F1 ckpt, 0.5 PR-AUC | F1 ckpt, 0.5 macro-F1 | AUC ckpt, val thr AUC | AUC ckpt, val thr PR-AUC | AUC ckpt, val thr macro-F1 | R x86 recall | R x64 recall | G x86 recall | G x64 recall | F1 ckpt, 0.5 cell-bal AUC | F1 ckpt, 0.5 cell-bal macro-F1 | AUC ckpt, val thr cell-bal AUC | AUC ckpt, val thr cell-bal macro-F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 TF-IDF | tfidf | 0.973 | 0.979 | 0.917 | 0.973 | 0.979 | 0.885 | 0.910 | 0.607 | 0.895 | 0.976 | 0.960 | 0.902 | 0.960 | 0.846 |
| R2 no-edges | kfold_v3_all_gnn-noedges-mblk15-selcell | missing (seed 0 folds present [] of 0-4) |  |  | missing (seed 0 folds present [] of 0-4) |  |  |  |  |  |  |  |  |  |  |
| R4 shuffle | kfold_v3_all_gnn-shuffle-mblk15-selcell | missing (seed 0 folds present [] of 0-4) |  |  | missing (seed 0 folds present [] of 0-4) |  |  |  |  |  |  |  |  |  |  |
| R7 typed | kfold_v3_all_gnn-mblk15-selcell | missing (seed 0 folds present [] of 0-4) |  |  | missing (seed 0 folds present [] of 0-4) |  |  |  |  |  |  |  |  |  |  |
| R0 TF-IDF ab | tfidf-ab | 0.969 | 0.976 | 0.909 | 0.969 | 0.976 | 0.868 | 0.858 | 0.656 | 0.929 | 0.949 | 0.967 | 0.909 | 0.967 | 0.847 |
| R2 no-edges ab | kfold_v3_all_gnn-noedges-mblk15-ab-selcell | 0.957 | 0.964 | 0.891 | 0.961 | 0.966 | 0.890 | 0.838 | 0.879 | 0.952 | 0.942 | 0.954 | 0.901 | 0.957 | 0.902 |
| R4 shuffle ab | kfold_v3_all_gnn-shuffle-mblk15-ab-selcell | 0.963 | 0.967 | 0.899 | 0.956 | 0.964 | 0.907 | 0.891 | 0.885 | 0.935 | 0.929 | 0.958 | 0.896 | 0.953 | 0.910 |
| R7 typed ab | kfold_v3_all_gnn-mblk15-ab-selcell | 0.969 | 0.973 | 0.901 | 0.965 | 0.970 | 0.904 | 0.897 | 0.831 | 0.936 | 0.933 | 0.969 | 0.900 | 0.961 | 0.899 |
| floor: majority | - | 0.500 | 0.559 | 0.359 | 0.500 | 0.559 | 0.359 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.333 | 0.500 | 0.333 |
| floor: x86_rule | - | 0.688 | 0.678 | 0.690 | 0.688 | 0.678 | 0.690 | 1.000 | 0.000 | 0.000 | 1.000 | 0.500 | 0.500 | 0.500 | 0.500 |

x86-rule macro-F1 per fold: 0.607, 0.686, 0.685, 0.733, 0.715 (pooled 0.690).

- Arch all: truncation not assessed, R7 (kfold_v3_all_gnn-mblk15-selcell) has no complete seed 0.

### Arch all: paired differences, second minus first, AUC selection (seed 0 only, not the pre-registered test)

**R2 no-edges vs R7 typed** (selection auc): missing: kfold_v3_all_gnn-noedges-mblk15-selcell: no complete seed among seeds [0] with all five folds

**R4 shuffle vs R7 typed** (selection auc): missing: kfold_v3_all_gnn-shuffle-mblk15-selcell: no complete seed among seeds [0] with all five folds

**R0 TF-IDF vs R2 no-edges** (selection auc): missing: kfold_v3_all_gnn-noedges-mblk15-selcell: no complete seed among seeds [0] with all five folds

### Arch all: paired cell-balanced differences, second minus first, AUC selection (seed 0 only, not the pre-registered test)

**R2 no-edges vs R7 typed** (selection auc): missing: kfold_v3_all_gnn-noedges-mblk15-selcell: no complete seed among seeds [0] with all five folds

**R4 shuffle vs R7 typed** (selection auc): missing: kfold_v3_all_gnn-shuffle-mblk15-selcell: no complete seed among seeds [0] with all five folds

**R0 TF-IDF vs R2 no-edges** (selection auc): missing: kfold_v3_all_gnn-noedges-mblk15-selcell: no complete seed among seeds [0] with all five folds

### Arch all: arch-balanced arms, paired differences, second minus first, AUC selection (seed 0 only, not the pre-registered test)

**R2 no-edges ab vs R7 typed ab**

### kfold_v3_all_gnn-mblk15-ab-selcell minus kfold_v3_all_gnn-noedges-mblk15-ab-selcell (selection auc, 2000 resamples)

a = kfold_v3_all_gnn-noedges-mblk15-ab-selcell, b = kfold_v3_all_gnn-mblk15-ab-selcell; seeds [0]; 5652 test binaries in 2586 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.961 | 0.965 | +0.004 | [-0.006, +0.015] |
| PR-AUC | 0.966 | 0.970 | +0.004 | [-0.006, +0.015] |
| macro-F1 | 0.890 | 0.904 | +0.014 | [-0.017, +0.051] |

**R4 shuffle ab vs R7 typed ab**

### kfold_v3_all_gnn-mblk15-ab-selcell minus kfold_v3_all_gnn-shuffle-mblk15-ab-selcell (selection auc, 2000 resamples)

a = kfold_v3_all_gnn-shuffle-mblk15-ab-selcell, b = kfold_v3_all_gnn-mblk15-ab-selcell; seeds [0]; 5652 test binaries in 2586 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.956 | 0.965 | +0.009 | [-0.012, +0.030] |
| PR-AUC | 0.964 | 0.970 | +0.006 | [-0.007, +0.021] |
| macro-F1 | 0.907 | 0.904 | -0.003 | [-0.032, +0.033] |

**R0 TF-IDF ab vs R2 no-edges ab**

### kfold_v3_all_gnn-noedges-mblk15-ab-selcell minus tfidf-ab (selection auc, 2000 resamples)

a = tfidf-ab, b = kfold_v3_all_gnn-noedges-mblk15-ab-selcell; seeds [0]; 5652 test binaries in 2586 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.969 | 0.961 | -0.008 | [-0.023, +0.006] |
| PR-AUC | 0.976 | 0.966 | -0.010 | [-0.022, -0.000] |
| macro-F1 | 0.868 | 0.890 | +0.022 | [-0.028, +0.080] |

### Arch all: arch-balanced arms, paired cell-balanced differences (seed 0 only, not the pre-registered test)

**R2 no-edges ab vs R7 typed ab**

### kfold_v3_all_gnn-mblk15-ab-selcell minus kfold_v3_all_gnn-noedges-mblk15-ab-selcell (selection auc, 2000 resamples)

a = kfold_v3_all_gnn-noedges-mblk15-ab-selcell, b = kfold_v3_all_gnn-mblk15-ab-selcell; seeds [0]; 5652 test binaries in 2586 clusters. Differences are b - a, 95% percentile interval. Cell-balanced: every test sample weighted N / (C n_cell) by its class x arch cell.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.957 | 0.961 | +0.004 | [-0.011, +0.016] |
| PR-AUC | 0.955 | 0.962 | +0.007 | [-0.011, +0.022] |
| macro-F1 | 0.902 | 0.899 | -0.003 | [-0.034, +0.026] |

**R4 shuffle ab vs R7 typed ab**

### kfold_v3_all_gnn-mblk15-ab-selcell minus kfold_v3_all_gnn-shuffle-mblk15-ab-selcell (selection auc, 2000 resamples)

a = kfold_v3_all_gnn-shuffle-mblk15-ab-selcell, b = kfold_v3_all_gnn-mblk15-ab-selcell; seeds [0]; 5652 test binaries in 2586 clusters. Differences are b - a, 95% percentile interval. Cell-balanced: every test sample weighted N / (C n_cell) by its class x arch cell.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.953 | 0.961 | +0.009 | [-0.007, +0.025] |
| PR-AUC | 0.955 | 0.962 | +0.007 | [-0.005, +0.022] |
| macro-F1 | 0.910 | 0.899 | -0.011 | [-0.041, +0.017] |

**R0 TF-IDF ab vs R2 no-edges ab**

### kfold_v3_all_gnn-noedges-mblk15-ab-selcell minus tfidf-ab (selection auc, 2000 resamples)

a = tfidf-ab, b = kfold_v3_all_gnn-noedges-mblk15-ab-selcell; seeds [0]; 5652 test binaries in 2586 clusters. Differences are b - a, 95% percentile interval. Cell-balanced: every test sample weighted N / (C n_cell) by its class x arch cell.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.967 | 0.957 | -0.010 | [-0.023, +0.003] |
| PR-AUC | 0.968 | 0.955 | -0.013 | [-0.027, -0.001] |
| macro-F1 | 0.847 | 0.902 | +0.056 | [-0.010, +0.146] |

### Arch all: each arch-balanced arm minus its unweighted twin, cell-balanced (seed 0 only, not the pre-registered test)

**R0 TF-IDF vs R0 TF-IDF ab**

### tfidf-ab minus tfidf (selection auc, 2000 resamples)

a = tfidf, b = tfidf-ab; seeds [0]; 5652 test binaries in 2586 clusters. Differences are b - a, 95% percentile interval. Cell-balanced: every test sample weighted N / (C n_cell) by its class x arch cell.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.960 | 0.967 | +0.007 | [-0.001, +0.014] |
| PR-AUC | 0.959 | 0.968 | +0.009 | [+0.001, +0.018] |
| macro-F1 | 0.846 | 0.847 | +0.001 | [-0.013, +0.014] |

**R2 no-edges vs R2 no-edges ab** (selection auc): missing: kfold_v3_all_gnn-noedges-mblk15-selcell: no complete seed among seeds [0] with all five folds

**R4 shuffle vs R4 shuffle ab** (selection auc): missing: kfold_v3_all_gnn-shuffle-mblk15-selcell: no complete seed among seeds [0] with all five folds

**R7 typed vs R7 typed ab** (selection auc): missing: kfold_v3_all_gnn-mblk15-selcell: no complete seed among seeds [0] with all five folds

## Arch matched: pooled table

| arm | group | F1 ckpt, 0.5 AUC | F1 ckpt, 0.5 PR-AUC | F1 ckpt, 0.5 macro-F1 | AUC ckpt, val thr AUC | AUC ckpt, val thr PR-AUC | AUC ckpt, val thr macro-F1 | R x86 recall | R x64 recall | G x86 recall | G x64 recall | F1 ckpt, 0.5 cell-bal AUC | F1 ckpt, 0.5 cell-bal macro-F1 | AUC ckpt, val thr cell-bal AUC | AUC ckpt, val thr cell-bal macro-F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 TF-IDF | tfidf | 0.970 | 0.984 | 0.905 | 0.970 | 0.984 | 0.894 | 0.915 | 0.843 | 0.897 | 0.924 | 0.965 | 0.907 | 0.965 | 0.895 |
| R2 no-edges | kfold_v3_matched_gnn-noedges-mblk15-selcell | missing (seed 0 folds present [] of 0-4) |  |  | missing (seed 0 folds present [] of 0-4) |  |  |  |  |  |  |  |  |  |  |
| R4 shuffle | kfold_v3_matched_gnn-shuffle-mblk15-selcell | missing (seed 0 folds present [] of 0-4) |  |  | missing (seed 0 folds present [] of 0-4) |  |  |  |  |  |  |  |  |  |  |
| R7 typed | kfold_v3_matched_gnn-mblk15-selcell | missing (seed 0 folds present [] of 0-4) |  |  | missing (seed 0 folds present [] of 0-4) |  |  |  |  |  |  |  |  |  |  |
| R0 TF-IDF ab | tfidf-ab | missing (seed 0 folds present [] of 0-4) |  |  | missing (seed 0 folds present [] of 0-4) |  |  |  |  |  |  |  |  |  |  |
| R2 no-edges ab | kfold_v3_matched_gnn-noedges-mblk15-ab-selcell | missing (seed 0 folds present [] of 0-4) |  |  | missing (seed 0 folds present [] of 0-4) |  |  |  |  |  |  |  |  |  |  |
| R4 shuffle ab | kfold_v3_matched_gnn-shuffle-mblk15-ab-selcell | missing (seed 0 folds present [] of 0-4) |  |  | missing (seed 0 folds present [] of 0-4) |  |  |  |  |  |  |  |  |  |  |
| R7 typed ab | kfold_v3_matched_gnn-mblk15-ab-selcell | missing (seed 0 folds present [] of 0-4) |  |  | missing (seed 0 folds present [] of 0-4) |  |  |  |  |  |  |  |  |  |  |
| floor: majority | - | 0.500 | 0.652 | 0.395 | 0.500 | 0.652 | 0.395 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 0.333 | 0.500 | 0.333 |
| floor: x86_rule | - | 0.604 | 0.704 | 0.606 | 0.604 | 0.704 | 0.606 | 1.000 | 0.000 | 0.000 | 1.000 | 0.500 | 0.500 | 0.500 | 0.500 |

x86-rule macro-F1 per fold: 0.535, 0.598, 0.588, 0.660, 0.628 (pooled 0.606).

- Arch matched: truncation not assessed, R7 (kfold_v3_matched_gnn-mblk15-selcell) has no complete seed 0.

### Arch matched: paired differences, second minus first, AUC selection (seed 0 only, not the pre-registered test)

**R2 no-edges vs R7 typed** (selection auc): missing: kfold_v3_matched_gnn-noedges-mblk15-selcell: no complete seed among seeds [0] with all five folds

**R4 shuffle vs R7 typed** (selection auc): missing: kfold_v3_matched_gnn-shuffle-mblk15-selcell: no complete seed among seeds [0] with all five folds

**R0 TF-IDF vs R2 no-edges** (selection auc): missing: kfold_v3_matched_gnn-noedges-mblk15-selcell: no complete seed among seeds [0] with all five folds

### Arch matched: paired cell-balanced differences, second minus first, AUC selection (seed 0 only, not the pre-registered test)

**R2 no-edges vs R7 typed** (selection auc): missing: kfold_v3_matched_gnn-noedges-mblk15-selcell: no complete seed among seeds [0] with all five folds

**R4 shuffle vs R7 typed** (selection auc): missing: kfold_v3_matched_gnn-shuffle-mblk15-selcell: no complete seed among seeds [0] with all five folds

**R0 TF-IDF vs R2 no-edges** (selection auc): missing: kfold_v3_matched_gnn-noedges-mblk15-selcell: no complete seed among seeds [0] with all five folds

### Arch matched: arch-balanced arms, paired differences, second minus first, AUC selection (seed 0 only, not the pre-registered test)

**R2 no-edges ab vs R7 typed ab** (selection auc): missing: kfold_v3_matched_gnn-noedges-mblk15-ab-selcell: no complete seed among seeds [0] with all five folds

**R4 shuffle ab vs R7 typed ab** (selection auc): missing: kfold_v3_matched_gnn-shuffle-mblk15-ab-selcell: no complete seed among seeds [0] with all five folds

**R0 TF-IDF ab vs R2 no-edges ab** (selection auc): missing: tfidf-ab: no complete seed among seeds [0] with all five folds

### Arch matched: arch-balanced arms, paired cell-balanced differences (seed 0 only, not the pre-registered test)

**R2 no-edges ab vs R7 typed ab** (selection auc): missing: kfold_v3_matched_gnn-noedges-mblk15-ab-selcell: no complete seed among seeds [0] with all five folds

**R4 shuffle ab vs R7 typed ab** (selection auc): missing: kfold_v3_matched_gnn-shuffle-mblk15-ab-selcell: no complete seed among seeds [0] with all five folds

**R0 TF-IDF ab vs R2 no-edges ab** (selection auc): missing: tfidf-ab: no complete seed among seeds [0] with all five folds

### Arch matched: each arch-balanced arm minus its unweighted twin, cell-balanced (seed 0 only, not the pre-registered test)

**R0 TF-IDF vs R0 TF-IDF ab** (selection auc): missing: tfidf-ab: no complete seed among seeds [0] with all five folds

**R2 no-edges vs R2 no-edges ab** (selection auc): missing: kfold_v3_matched_gnn-noedges-mblk15-selcell: no complete seed among seeds [0] with all five folds

**R4 shuffle vs R4 shuffle ab** (selection auc): missing: kfold_v3_matched_gnn-shuffle-mblk15-selcell: no complete seed among seeds [0] with all five folds

**R7 typed vs R7 typed ab** (selection auc): missing: kfold_v3_matched_gnn-mblk15-selcell: no complete seed among seeds [0] with all five folds

## Verdict: does R7 beat R2 and R4?

Pre-registered rule (PLAN_GNN.md section 5), applied to the AUC selection: dAUC >= 0.01 and dmacro-F1 >= 0.02 pooled, both 95% intervals excluding zero. With one seed it cannot pass the rule; these lines only say which way seed 0 points. Per arch, the first line uses the ordinary metrics on the unweighted arms, the second the cell-balanced metrics on them, the third the ordinary metrics on the arch-balanced arms, the fourth the cell-balanced metrics on the arch-balanced arms.

- Arch all: cannot assess, missing: R2 no-edges: kfold_v3_all_gnn-noedges-mblk15-selcell: no complete seed among seeds [0] with all five folds; R4 shuffle: kfold_v3_all_gnn-shuffle-mblk15-selcell: no complete seed among seeds [0] with all five folds (seed 0 only, not the pre-registered test).
- Arch all: cell-balanced: cannot assess, missing: R2 no-edges: kfold_v3_all_gnn-noedges-mblk15-selcell: no complete seed among seeds [0] with all five folds; R4 shuffle: kfold_v3_all_gnn-shuffle-mblk15-selcell: no complete seed among seeds [0] with all five folds (seed 0 only, not the pre-registered test).
- Arch all: R7 ab does not beat both R2 and R4: does not beat R2 no-edges ab (dAUC +0.004 [-0.006, +0.015], dmacro-F1 +0.014 [-0.017, +0.051]); does not beat R4 shuffle ab (dAUC +0.009 [-0.012, +0.030], dmacro-F1 -0.003 [-0.032, +0.033]) (seed 0 only, not the pre-registered test).
- Arch all: cell-balanced: R7 ab does not beat both R2 and R4: does not beat R2 no-edges ab (dAUC +0.004 [-0.011, +0.016], dmacro-F1 -0.003 [-0.034, +0.026]); does not beat R4 shuffle ab (dAUC +0.009 [-0.007, +0.025], dmacro-F1 -0.011 [-0.041, +0.017]) (seed 0 only, not the pre-registered test).
- Arch matched: cannot assess, missing: R2 no-edges: kfold_v3_matched_gnn-noedges-mblk15-selcell: no complete seed among seeds [0] with all five folds; R4 shuffle: kfold_v3_matched_gnn-shuffle-mblk15-selcell: no complete seed among seeds [0] with all five folds (seed 0 only, not the pre-registered test).
- Arch matched: cell-balanced: cannot assess, missing: R2 no-edges: kfold_v3_matched_gnn-noedges-mblk15-selcell: no complete seed among seeds [0] with all five folds; R4 shuffle: kfold_v3_matched_gnn-shuffle-mblk15-selcell: no complete seed among seeds [0] with all five folds (seed 0 only, not the pre-registered test).
- Arch matched: cannot assess, missing: R2 no-edges ab: kfold_v3_matched_gnn-noedges-mblk15-ab-selcell: no complete seed among seeds [0] with all five folds; R4 shuffle ab: kfold_v3_matched_gnn-shuffle-mblk15-ab-selcell: no complete seed among seeds [0] with all five folds (seed 0 only, not the pre-registered test).
- Arch matched: cell-balanced: cannot assess, missing: R2 no-edges ab: kfold_v3_matched_gnn-noedges-mblk15-ab-selcell: no complete seed among seeds [0] with all five folds; R4 shuffle ab: kfold_v3_matched_gnn-shuffle-mblk15-ab-selcell: no complete seed among seeds [0] with all five folds (seed 0 only, not the pre-registered test).

