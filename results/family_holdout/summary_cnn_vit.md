# CNN-ViT under family holdout

`HierarchicalMalwareNet` (CNN-ViT), untuned recipe, evaluated with every one of the 38 in-cohort ransomware
families held out in turn. Fold definition: `family_holdout/folds.py` (K = 5, families assigned whole,
goodware assigned by duplicate-stream group on Mendeley and by source project on Goodware_Balanced).
3 seed(s) per fold; a fold's number is the mean over its seeds, `pooled` scores the union of the
five held-out folds using the seed-mean score at argmax. LOFO trains on the other 37 families plus all
goodware and tests on the held-out family alone - 1 seed, recall only (no goodware in its test set).

Floors are per-fold accuracies: `majority` = always predict the larger class of that fold, `x86 rule` =
predict ransomware iff the PE machine field says x86. Nothing here is tuned and no threshold is moved.

Fold mean and pooled are not the same quantity. A fold's number is one model's; the pooled column
scores every file once from the mean of 3 seeds' scores, which is a 3-seed ensemble and
is reliably the higher of the two. The fold mean +/- sd is the like-for-like comparison with the
single-model fixed-split number; the pooled column is what the per-family and per-architecture
breakdowns below are computed from, because every file has exactly one held-out prediction there.

## mendeley

2509 files: 1266 ransomware in 38 families, 1243 goodware in 1024 groups. 53 training runs (15 K-fold, 38 LOFO).

| metric | fold mean +/- sd (5 folds) | pooled | fixed split (5 seeds) |
|---|---|---|---|
| macro-F1 | 0.679 +/- 0.038 | 0.744 | 0.628 +/- 0.018 |
| balanced accuracy | 0.694 +/- 0.029 | 0.744 | - |
| accuracy | 0.694 +/- 0.029 | 0.744 | - |
| ROC AUC | 0.805 +/- 0.023 | 0.809 | - |

Per fold:

| fold | n | good | rans | macro-F1 (seed mean) | sd over seeds | bal-acc | AUC | recall R / G | FPR | majority floor | x86-rule floor | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 503 | 249 | 254 | 0.666 | 0.058 | 0.683 | 0.807 | 0.80 / 0.56 | 0.44 | 0.505 | 0.670 | 14.7 |
| 1 | 502 | 249 | 253 | 0.705 | 0.015 | 0.711 | 0.780 | 0.58 / 0.85 | 0.15 | 0.504 | 0.681 | 14.3 |
| 2 | 502 | 249 | 253 | 0.659 | 0.033 | 0.677 | 0.791 | 0.59 / 0.77 | 0.23 | 0.504 | 0.580 | 13.0 |
| 3 | 499 | 248 | 251 | 0.630 | 0.089 | 0.659 | 0.800 | 0.67 / 0.65 | 0.35 | 0.503 | 0.647 | 13.0 |
| 4 | 503 | 248 | 255 | 0.738 | 0.064 | 0.742 | 0.848 | 0.73 / 0.75 | 0.25 | 0.507 | 0.702 | 15.0 |

Pooled floors over all 2509 held-out predictions: majority 0.505 accuracy (macro-F1 0.335), x86 rule 0.656 accuracy (macro-F1 0.631, recall 0.91 ransomware / 0.40 goodware).

Against the fixed split: the fold mean is +0.051 macro-F1 from the single 24/14 split's 0.628, which is 1.4x the fold sd (0.038) - the one split was not representative of the cohort's families.

Against the architecture shortcut: pooled macro-F1 0.744 is +0.113 against the x86 rule's 0.631. The encoder clears the floor, so it is reading more than the machine field.

Per architecture, pooled:

| arch | n | good | rans | accuracy | macro-F1 | recall ransomware | recall goodware | FPR | LOFO recall |
|---|---|---|---|---|---|---|---|---|---|
| x64 | 608 | 494 | 114 | 0.788 | 0.641 | 0.395 | 0.879 | 0.121 | 0.395 |
| x86 | 1901 | 749 | 1152 | 0.730 | 0.721 | 0.748 | 0.702 | 0.298 | 0.724 |

LOFO mean recall over 38 families: 0.681 unweighted (each family counts once); file-weighted 0.694. See `lofo_predictions.csv`.

Per-family recall (K-fold held-out, LOFO):

| family | n | x64 | fold | K-fold recall | LOFO recall |
|---|---|---|---|---|---|
| darkside | 38 | 0 | 2 | 0.00 | 0.82 |
| dharma | 46 | 0 | 1 | 0.00 | 0.00 |
| makop | 30 | 0 | 1 | 0.00 | 0.10 |
| maui | 3 | 0 | 2 | 0.00 | 1.00 |
| exorcist | 17 | 4 | 4 | 0.06 | 0.06 |
| nefilim | 37 | 11 | 2 | 0.19 | 0.86 |
| doppelpaymer | 22 | 0 | 3 | 0.23 | 0.18 |
| karma | 13 | 0 | 4 | 0.23 | 0.23 |
| zeppelin | 18 | 5 | 2 | 0.28 | 0.39 |
| mountlocker | 14 | 9 | 0 | 0.29 | 0.43 |
| hive | 50 | 43 | 2 | 0.42 | 0.20 |
| blackbasta | 30 | 3 | 1 | 0.43 | 0.80 |
| wastedlocker | 36 | 0 | 3 | 0.53 | 0.42 |
| clop | 45 | 0 | 4 | 0.62 | 0.58 |
| ransomexx | 13 | 0 | 2 | 0.69 | 0.92 |
| blackbyte | 7 | 7 | 1 | 0.71 | 1.00 |
| ragnarok | 42 | 0 | 1 | 0.74 | 1.00 |
| holyghost | 4 | 3 | 0 | 0.75 | 0.50 |
| stop | 35 | 0 | 0 | 0.77 | 0.77 |
| ryuk | 47 | 12 | 3 | 0.81 | 0.70 |
| conti | 48 | 0 | 1 | 0.81 | 0.92 |
| quantum | 6 | 5 | 4 | 0.83 | 1.00 |
| avoslocker | 50 | 0 | 0 | 0.84 | 0.02 |
| lorenz | 16 | 0 | 0 | 0.88 | 1.00 |
| revil | 47 | 0 | 2 | 0.89 | 0.96 |
| maze | 47 | 1 | 3 | 0.94 | 0.98 |
| lockbit | 47 | 0 | 2 | 0.98 | 1.00 |
| gandcrab | 49 | 0 | 4 | 0.98 | 0.53 |
| phobos | 49 | 0 | 0 | 0.98 | 0.94 |
| avaddon | 49 | 0 | 4 | 1.00 | 1.00 |
| babuk | 42 | 0 | 4 | 1.00 | 1.00 |
| bianlian | 11 | 11 | 3 | 1.00 | 0.64 |
| blackcat | 50 | 0 | 1 | 1.00 | 0.94 |
| blackmatter | 43 | 0 | 0 | 1.00 | 1.00 |
| bluesky | 34 | 0 | 4 | 1.00 | 1.00 |
| netwalker | 50 | 0 | 3 | 1.00 | 0.98 |
| playcrypt | 43 | 0 | 0 | 1.00 | 1.00 |
| pysa | 38 | 0 | 3 | 1.00 | 0.00 |

Sorted by K-fold recall; the families at the top are the ones this encoder never sees.

12 of 38 families are below 0.5 K-fold recall and 4 are at zero (darkside, dharma, makop, maui). 5 are missed by both schemes (dharma, makop, exorcist, doppelpaymer, karma) - those are the ones neither more training data nor a different fold rescues.

LOFO is one seed per family, so a large K-fold/LOFO disagreement is not by itself evidence that training-set size mattered: maui 0.00 -> 1.00; pysa 1.00 -> 0.00; avoslocker 0.84 -> 0.02 are the three widest here, and the per-fold seed sd above is already up to 0.089 macro-F1 on three seeds of the same fold.

Runtime: 53 runs, 60.1 min of training wall clock, 4.91 s/epoch mean, 13.8 epochs per run (early stopping, patience 8 of a possible 80).

## balanced

2603 files: 1266 ransomware in 38 families, 1337 goodware in 103 groups. 53 training runs (15 K-fold, 38 LOFO).

| metric | fold mean +/- sd (5 folds) | pooled | fixed split (5 seeds) |
|---|---|---|---|
| macro-F1 | 0.736 +/- 0.032 | 0.784 | 0.502 +/- 0.113 |
| balanced accuracy | 0.740 +/- 0.032 | 0.784 | - |
| accuracy | 0.741 +/- 0.031 | 0.786 | - |
| ROC AUC | 0.821 +/- 0.029 | 0.817 | - |

Per fold:

| fold | n | good | rans | macro-F1 (seed mean) | sd over seeds | bal-acc | AUC | recall R / G | FPR | majority floor | x86-rule floor | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 539 | 285 | 254 | 0.798 | 0.055 | 0.801 | 0.858 | 0.85 / 0.75 | 0.25 | 0.529 | 0.796 | 13.0 |
| 1 | 516 | 263 | 253 | 0.722 | 0.005 | 0.724 | 0.816 | 0.62 / 0.83 | 0.17 | 0.510 | 0.872 | 14.0 |
| 2 | 516 | 263 | 253 | 0.706 | 0.025 | 0.707 | 0.772 | 0.65 / 0.77 | 0.23 | 0.510 | 0.769 | 13.7 |
| 3 | 514 | 263 | 251 | 0.721 | 0.067 | 0.731 | 0.843 | 0.65 / 0.82 | 0.18 | 0.512 | 0.881 | 13.0 |
| 4 | 518 | 263 | 255 | 0.735 | 0.038 | 0.737 | 0.817 | 0.65 / 0.82 | 0.18 | 0.508 | 0.873 | 15.0 |

Pooled floors over all 2603 held-out predictions: majority 0.514 accuracy (macro-F1 0.339), x86 rule 0.838 accuracy (macro-F1 0.838, recall 0.91 ransomware / 0.77 goodware).

Against the fixed split: the fold mean is +0.234 macro-F1 from the single 24/14 split's 0.502, which is 7.3x the fold sd (0.032) - the one split was not representative of the cohort's families.

Against the architecture shortcut: pooled macro-F1 0.784 is -0.053 against the x86 rule's 0.838. **Below the floor** - a classifier that reads nothing but the PE machine field scores higher than the image encoder here, so nothing in this dataset's number demonstrates that the token images carry the detection.

Per architecture, pooled:

| arch | n | good | rans | accuracy | macro-F1 | recall ransomware | recall goodware | FPR | LOFO recall |
|---|---|---|---|---|---|---|---|---|---|
| x64 | 1143 | 1029 | 114 | 0.875 | 0.553 | 0.132 | 0.957 | 0.043 | 0.096 |
| x86 | 1460 | 308 | 1152 | 0.716 | 0.614 | 0.779 | 0.481 | 0.519 | 0.738 |

LOFO mean recall over 38 families: 0.652 unweighted (each family counts once); file-weighted 0.680. See `lofo_predictions.csv`.

Per-family recall (K-fold held-out, LOFO):

| family | n | x64 | fold | K-fold recall | LOFO recall |
|---|---|---|---|---|---|
| avaddon | 49 | 0 | 4 | 0.00 | 1.00 |
| bianlian | 11 | 11 | 3 | 0.00 | 0.00 |
| blackbyte | 7 | 7 | 1 | 0.00 | 0.00 |
| dharma | 46 | 0 | 1 | 0.02 | 0.02 |
| lockbit | 47 | 0 | 2 | 0.17 | 0.15 |
| karma | 13 | 0 | 4 | 0.23 | 0.23 |
| holyghost | 4 | 3 | 0 | 0.25 | 0.25 |
| hive | 50 | 43 | 2 | 0.30 | 0.16 |
| doppelpaymer | 22 | 0 | 3 | 0.36 | 0.41 |
| blackbasta | 30 | 3 | 1 | 0.43 | 0.50 |
| quantum | 6 | 5 | 4 | 0.50 | 1.00 |
| nefilim | 37 | 11 | 2 | 0.51 | 0.49 |
| mountlocker | 14 | 9 | 0 | 0.64 | 0.29 |
| ryuk | 47 | 12 | 3 | 0.68 | 0.74 |
| playcrypt | 43 | 0 | 0 | 0.70 | 0.56 |
| netwalker | 50 | 0 | 3 | 0.70 | 0.70 |
| zeppelin | 18 | 5 | 2 | 0.72 | 0.39 |
| wastedlocker | 36 | 0 | 3 | 0.75 | 0.86 |
| exorcist | 17 | 4 | 4 | 0.76 | 0.76 |
| bluesky | 34 | 0 | 4 | 0.82 | 0.85 |
| makop | 30 | 0 | 1 | 0.83 | 0.73 |
| avoslocker | 50 | 0 | 0 | 0.84 | 0.14 |
| pysa | 38 | 0 | 3 | 0.89 | 1.00 |
| clop | 45 | 0 | 4 | 0.91 | 0.84 |
| stop | 35 | 0 | 0 | 0.91 | 1.00 |
| conti | 48 | 0 | 1 | 0.94 | 0.67 |
| maze | 47 | 1 | 3 | 0.96 | 0.98 |
| ragnarok | 42 | 0 | 1 | 0.98 | 0.98 |
| revil | 47 | 0 | 2 | 0.98 | 0.98 |
| babuk | 42 | 0 | 4 | 1.00 | 1.00 |
| blackcat | 50 | 0 | 1 | 1.00 | 0.58 |
| blackmatter | 43 | 0 | 0 | 1.00 | 0.56 |
| darkside | 38 | 0 | 2 | 1.00 | 1.00 |
| gandcrab | 49 | 0 | 4 | 1.00 | 1.00 |
| lorenz | 16 | 0 | 0 | 1.00 | 1.00 |
| maui | 3 | 0 | 2 | 1.00 | 1.00 |
| phobos | 49 | 0 | 0 | 1.00 | 0.96 |
| ransomexx | 13 | 0 | 2 | 1.00 | 1.00 |

Sorted by K-fold recall; the families at the top are the ones this encoder never sees.

10 of 38 families are below 0.5 K-fold recall and 3 are at zero (avaddon, bianlian, blackbyte). 5 are missed by both schemes (bianlian, blackbyte, dharma, lockbit, karma) - those are the ones neither more training data nor a different fold rescues.

LOFO is one seed per family, so a large K-fold/LOFO disagreement is not by itself evidence that training-set size mattered: avaddon 0.00 -> 1.00; avoslocker 0.84 -> 0.14; quantum 0.50 -> 1.00 are the three widest here, and the per-fold seed sd above is already up to 0.067 macro-F1 on three seeds of the same fold.

Runtime: 53 runs, 59.4 min of training wall clock, 4.81 s/epoch mean, 13.9 epochs per run (early stopping, patience 8 of a possible 80).

## Reading this against the fixed split

The fixed 24/14 family split gave this encoder 0.63 +/- 0.02 macro-F1 on Mendeley and 0.50 +/- 0.11 on
Goodware_Balanced (5 seeds, `results/COMPARISON.md`). Those sds are over seeds on one split; the fold sd
above is over five different choices of held-out families, which is the quantity that says whether the
single split was lucky.

Caveat on the fold sd: x64 ransomware is concentrated in fold 2 (59 of 114 x64 ransomware files; Hive
alone is 43), so part of the spread across folds measures architecture mix rather than family
difficulty. The per-architecture table above is the check on that.

Produced by `family_holdout/run_cnn_vit.py`; per-fold numbers in `fold_metrics.csv`, every held-out
prediction in `predictions.csv`, LOFO in `lofo_predictions.csv`.

