# Family-holdout: `tokenization`

Every ransomware family in the cohort is held out, twice over, with the fold definition in `family_holdout/folds.py` (`results/family_holdout/folds_<dataset>.csv`):

* **K-fold** - families are assigned whole to 5 folds; train on four, test on the fifth. Every file gets exactly one held-out prediction, so the pooled row is a complete cross-validated pass over the cohort.
* **LOFO** - leave one family out: train on every other ransomware family plus ALL goodware, test on that family alone. No goodware in the test set, so it is a recall study only (no FPR, no AUC).

Nothing here is tuned. The configuration is the pre-registered one for each pipeline; see `config_used.yaml` in each model directory.

> **Architecture caveat, carried into every table below.** x64 ransomware concentrates in fold 2: 59 of the 114 x64 ransomware files land there, and Hive alone is 43 of them. A per-fold x64 number outside fold 2 rests on a handful of files, and the per-fold spread of any x64 metric is not a sampling spread.

## Headline

### mendeley

| model | fold mean macro-F1 +/- sd | pooled macro-F1 | pooled bal-acc | pooled acc | pooled AUC | majority floor (acc / macro-F1) | x86-rule floor (acc / macro-F1) | LOFO mean recall | fixed-split macro-F1 |
|---|---|---|---|---|---|---|---|---|---|
| MLP_WP_w2v | 0.869 +/- 0.035 | **0.869** | 0.870 | 0.870 | 0.943 | 0.505 / 0.335 | 0.656 / 0.631 | 0.830 | 0.926 |
| SVM-RBF_SW_w2v | 0.844 +/- 0.048 | **0.845** | 0.849 | 0.847 | 0.935 | 0.505 / 0.335 | 0.656 / 0.631 | 0.745 | 0.878 |
| RF_WPC_w2v | 0.851 +/- 0.047 | **0.852** | 0.854 | 0.853 | 0.955 | 0.505 / 0.335 | 0.656 / 0.631 | 0.753 | 0.766 |

Per-fold spread of the other headline metrics:

| model | acc (mean +/- sd) | bal-acc (mean +/- sd) | AUC (mean +/- sd) |
|---|---|---|---|
| MLP_WP_w2v | 0.870 +/- 0.034 | 0.870 +/- 0.034 | 0.951 +/- 0.023 |
| SVM-RBF_SW_w2v | 0.847 +/- 0.045 | 0.849 +/- 0.044 | 0.941 +/- 0.039 |
| RF_WPC_w2v | 0.853 +/- 0.046 | 0.854 +/- 0.045 | 0.955 +/- 0.014 |

### balanced

| model | fold mean macro-F1 +/- sd | pooled macro-F1 | pooled bal-acc | pooled acc | pooled AUC | majority floor (acc / macro-F1) | x86-rule floor (acc / macro-F1) | LOFO mean recall | fixed-split macro-F1 |
|---|---|---|---|---|---|---|---|---|---|
| MLP_WP_w2v | 0.820 +/- 0.043 | **0.821** | 0.821 | 0.822 | 0.895 | 0.514 / 0.339 | 0.838 / 0.838 | 0.786 | 0.597 |
| SVM-RBF_SW_w2v | 0.775 +/- 0.052 | **0.775** | 0.777 | 0.780 | 0.898 | 0.514 / 0.339 | 0.838 / 0.838 | 0.685 | 0.844 |
| RF_WPC_w2v | 0.719 +/- 0.088 | **0.721** | 0.728 | 0.733 | 0.910 | 0.514 / 0.339 | 0.838 / 0.838 | 0.616 | 0.553 |

Per-fold spread of the other headline metrics:

| model | acc (mean +/- sd) | bal-acc (mean +/- sd) | AUC (mean +/- sd) |
|---|---|---|---|
| MLP_WP_w2v | 0.822 +/- 0.040 | 0.821 +/- 0.042 | 0.910 +/- 0.016 |
| SVM-RBF_SW_w2v | 0.781 +/- 0.048 | 0.777 +/- 0.051 | 0.899 +/- 0.027 |
| RF_WPC_w2v | 0.734 +/- 0.077 | 0.729 +/- 0.081 | 0.909 +/- 0.045 |

## Per architecture (pooled K-fold predictions)

### mendeley

| model | arch | n | support r / g | recall ransomware | recall goodware | accuracy | macro-F1 |
|---|---|---|---|---|---|---|---|
| MLP_WP_w2v | x64 | 608 | 114 / 494 | 0.675 | 0.992 | 0.933 | 0.875 |
| MLP_WP_w2v | x86 | 1901 | 1152 / 749 | 0.803 | 0.921 | 0.850 | 0.847 |
| SVM-RBF_SW_w2v | x64 | 608 | 114 / 494 | 0.789 | 0.998 | 0.959 | 0.927 |
| SVM-RBF_SW_w2v | x86 | 1901 | 1152 / 749 | 0.715 | 0.960 | 0.812 | 0.811 |
| RF_WPC_w2v | x64 | 608 | 114 / 494 | 0.684 | 1.000 | 0.941 | 0.889 |
| RF_WPC_w2v | x86 | 1901 | 1152 / 749 | 0.743 | 0.952 | 0.825 | 0.824 |

### balanced

| model | arch | n | support r / g | recall ransomware | recall goodware | accuracy | macro-F1 |
|---|---|---|---|---|---|---|---|
| MLP_WP_w2v | x64 | 1143 | 114 / 1029 | 0.588 | 0.972 | 0.934 | 0.801 |
| MLP_WP_w2v | x86 | 1460 | 1152 / 308 | 0.786 | 0.542 | 0.735 | 0.644 |
| SVM-RBF_SW_w2v | x64 | 1143 | 114 / 1029 | 0.702 | 0.983 | 0.955 | 0.867 |
| SVM-RBF_SW_w2v | x86 | 1460 | 1152 / 308 | 0.647 | 0.630 | 0.643 | 0.584 |
| RF_WPC_w2v | x64 | 1143 | 114 / 1029 | 0.518 | 0.994 | 0.947 | 0.815 |
| RF_WPC_w2v | x86 | 1460 | 1152 / 308 | 0.536 | 0.679 | 0.566 | 0.530 |

## Per family: K-fold recall vs LOFO recall

Sorted by mean LOFO recall, worst first. `fold` is the K-fold the family is assigned to; `n_x64` is how many of its files are x64.

### mendeley

| family | n | n_x64 | fold | MLP_WP_w2v K-fold | MLP_WP_w2v LOFO | SVM-RBF_SW_w2v K-fold | SVM-RBF_SW_w2v LOFO | RF_WPC_w2v K-fold | RF_WPC_w2v LOFO | mean LOFO |
|---|---|---|---|---|---|---|---|---|---|---|
| phobos | 49 | 0 | 0 | 0.04 | 0.04 | 0.98 | 0.04 | 0.06 | 0.06 | 0.05 |
| maze | 47 | 1 | 3 | 0.36 | 0.43 | 0.17 | 0.13 | 0.17 | 0.23 | 0.26 |
| stop | 35 | 0 | 0 | 0.29 | 0.29 | 0.29 | 0.29 | 0.29 | 0.29 | 0.29 |
| babuk | 42 | 0 | 4 | 0.71 | 0.71 | 0.00 | 0.19 | 0.24 | 0.21 | 0.37 |
| blackcat | 50 | 0 | 1 | 0.42 | 0.42 | 0.40 | 0.40 | 0.54 | 0.42 | 0.41 |
| makop | 30 | 0 | 1 | 0.53 | 0.53 | 0.50 | 0.50 | 0.53 | 0.53 | 0.52 |
| doppelpaymer | 22 | 0 | 3 | 0.59 | 0.59 | 0.64 | 0.64 | 0.45 | 0.41 | 0.55 |
| darkside | 38 | 0 | 2 | 1.00 | 1.00 | 0.47 | 0.68 | 0.00 | 0.00 | 0.56 |
| zeppelin | 18 | 5 | 2 | 0.67 | 0.67 | 0.61 | 0.61 | 0.67 | 0.67 | 0.65 |
| ryuk | 47 | 12 | 3 | 0.64 | 0.74 | 0.68 | 0.68 | 0.60 | 0.57 | 0.67 |
| avaddon | 49 | 0 | 4 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.67 |
| lockbit | 47 | 0 | 2 | 0.28 | 0.28 | 0.79 | 0.79 | 0.94 | 0.94 | 0.67 |
| mountlocker | 14 | 9 | 0 | 0.64 | 0.64 | 0.86 | 0.86 | 0.50 | 0.50 | 0.67 |
| nefilim | 37 | 11 | 2 | 0.73 | 0.92 | 0.70 | 0.70 | 0.43 | 0.46 | 0.69 |
| wastedlocker | 36 | 0 | 3 | 0.94 | 0.97 | 0.28 | 0.42 | 0.69 | 0.69 | 0.69 |
| clop | 45 | 0 | 4 | 0.80 | 0.80 | 0.84 | 0.76 | 0.80 | 0.73 | 0.76 |
| exorcist | 17 | 4 | 4 | 0.76 | 0.76 | 0.76 | 0.76 | 0.76 | 0.76 | 0.76 |
| blackmatter | 43 | 0 | 0 | 1.00 | 1.00 | 0.37 | 0.37 | 1.00 | 1.00 | 0.79 |
| quantum | 6 | 5 | 4 | 0.17 | 1.00 | 0.83 | 0.83 | 0.83 | 0.83 | 0.89 |
| blackbyte | 7 | 7 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.71 | 0.90 |
| hive | 50 | 43 | 2 | 0.96 | 1.00 | 1.00 | 1.00 | 0.86 | 0.78 | 0.93 |
| conti | 48 | 0 | 1 | 0.98 | 0.98 | 0.85 | 0.88 | 0.98 | 0.96 | 0.94 |
| revil | 47 | 0 | 2 | 0.98 | 0.96 | 0.96 | 0.94 | 0.81 | 0.94 | 0.94 |
| blackbasta | 30 | 3 | 1 | 0.97 | 0.93 | 0.97 | 0.97 | 0.93 | 0.93 | 0.94 |
| ransomexx | 13 | 0 | 2 | 1.00 | 0.92 | 0.92 | 0.92 | 1.00 | 1.00 | 0.95 |
| dharma | 46 | 0 | 1 | 0.98 | 0.98 | 0.98 | 0.98 | 1.00 | 0.98 | 0.98 |
| gandcrab | 49 | 0 | 4 | 1.00 | 0.98 | 0.98 | 1.00 | 0.94 | 0.98 | 0.99 |
| avoslocker | 50 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| bianlian | 11 | 11 | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| bluesky | 34 | 0 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| holyghost | 4 | 3 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| karma | 13 | 0 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| lorenz | 16 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| maui | 3 | 0 | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| netwalker | 50 | 0 | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| playcrypt | 43 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| pysa | 38 | 0 | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| ragnarok | 42 | 0 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### balanced

| family | n | n_x64 | fold | MLP_WP_w2v K-fold | MLP_WP_w2v LOFO | SVM-RBF_SW_w2v K-fold | SVM-RBF_SW_w2v LOFO | RF_WPC_w2v K-fold | RF_WPC_w2v LOFO | mean LOFO |
|---|---|---|---|---|---|---|---|---|---|---|
| avaddon | 49 | 0 | 4 | 1.00 | 0.00 | 1.00 | 0.04 | 0.00 | 0.00 | 0.01 |
| makop | 30 | 0 | 1 | 0.03 | 0.03 | 0.07 | 0.07 | 0.00 | 0.00 | 0.03 |
| blackcat | 50 | 0 | 1 | 0.00 | 0.00 | 0.98 | 0.58 | 0.00 | 0.00 | 0.19 |
| stop | 35 | 0 | 0 | 0.66 | 0.20 | 0.26 | 0.34 | 0.31 | 0.31 | 0.29 |
| lorenz | 16 | 0 | 0 | 0.12 | 1.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.38 |
| maze | 47 | 1 | 3 | 0.40 | 0.40 | 0.40 | 0.34 | 0.36 | 0.38 | 0.38 |
| doppelpaymer | 22 | 0 | 3 | 0.64 | 0.86 | 0.09 | 0.18 | 0.09 | 0.18 | 0.41 |
| babuk | 42 | 0 | 4 | 0.74 | 0.71 | 0.31 | 0.36 | 0.24 | 0.50 | 0.52 |
| karma | 13 | 0 | 4 | 1.00 | 1.00 | 0.77 | 0.62 | 0.00 | 0.00 | 0.54 |
| conti | 48 | 0 | 1 | 0.90 | 0.94 | 0.31 | 0.33 | 0.25 | 0.35 | 0.54 |
| gandcrab | 49 | 0 | 4 | 0.55 | 0.55 | 0.55 | 0.55 | 0.53 | 0.55 | 0.55 |
| blackmatter | 43 | 0 | 0 | 1.00 | 1.00 | 0.37 | 0.37 | 0.37 | 0.37 | 0.58 |
| zeppelin | 18 | 5 | 2 | 0.67 | 0.67 | 0.50 | 0.50 | 0.67 | 0.67 | 0.61 |
| mountlocker | 14 | 9 | 0 | 0.43 | 0.64 | 0.79 | 0.79 | 0.57 | 0.50 | 0.64 |
| maui | 3 | 0 | 2 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 0.67 |
| nefilim | 37 | 11 | 2 | 0.73 | 0.73 | 0.57 | 0.73 | 0.32 | 0.54 | 0.67 |
| phobos | 49 | 0 | 0 | 0.98 | 0.98 | 0.98 | 1.00 | 0.06 | 0.06 | 0.68 |
| netwalker | 50 | 0 | 3 | 1.00 | 0.52 | 1.00 | 1.00 | 0.94 | 0.56 | 0.69 |
| ryuk | 47 | 12 | 3 | 0.74 | 0.74 | 0.66 | 0.64 | 0.62 | 0.72 | 0.70 |
| quantum | 6 | 5 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.17 | 0.72 |
| clop | 45 | 0 | 4 | 0.80 | 0.78 | 0.76 | 0.69 | 0.80 | 0.82 | 0.76 |
| exorcist | 17 | 4 | 4 | 0.76 | 0.76 | 0.76 | 0.76 | 0.76 | 0.76 | 0.76 |
| hive | 50 | 43 | 2 | 0.76 | 0.76 | 0.82 | 0.84 | 0.70 | 0.72 | 0.77 |
| blackbasta | 30 | 3 | 1 | 0.80 | 0.80 | 0.80 | 0.80 | 0.70 | 0.80 | 0.80 |
| wastedlocker | 36 | 0 | 3 | 1.00 | 0.97 | 0.83 | 0.83 | 0.83 | 0.83 | 0.88 |
| lockbit | 47 | 0 | 2 | 1.00 | 1.00 | 0.85 | 0.85 | 0.85 | 0.85 | 0.90 |
| blackbyte | 7 | 7 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.71 | 0.71 | 0.90 |
| revil | 47 | 0 | 2 | 0.98 | 0.91 | 0.19 | 0.87 | 0.23 | 0.96 | 0.91 |
| ransomexx | 13 | 0 | 2 | 1.00 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 |
| pysa | 38 | 0 | 3 | 1.00 | 1.00 | 0.00 | 0.95 | 1.00 | 1.00 | 0.98 |
| darkside | 38 | 0 | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 0.92 | 0.95 | 0.98 |
| dharma | 46 | 0 | 1 | 0.98 | 0.98 | 1.00 | 0.98 | 1.00 | 1.00 | 0.99 |
| avoslocker | 50 | 0 | 0 | 0.12 | 1.00 | 0.12 | 1.00 | 0.12 | 1.00 | 1.00 |
| bianlian | 11 | 11 | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| bluesky | 34 | 0 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| holyghost | 4 | 3 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| playcrypt | 43 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| ragnarok | 42 | 0 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Against the fixed split

The `fixed-split macro-F1` column of the headline table is the same model's row in `results/COMPARISON.md`, i.e. the single Mendeley family-disjoint split (24 train / 14 test families). The two are not the same experiment: the fixed split tests 14 families with a 74%-ransomware test set, the K-fold pooled row tests all 38 with a roughly balanced one, and LOFO gives every family the largest possible training set. A family-holdout number above the fixed-split one usually means the fixed split's 14 test families were the harder half, not that the model improved.

| dataset | model | pooled macro-F1 (K-fold) | fixed-split macro-F1 | delta | headroom over the x86-rule floor | LOFO mean recall | families with LOFO recall < 0.5 |
|---|---|---|---|---|---|---|---|
| mendeley | MLP_WP_w2v | 0.869 | 0.926 | -0.057 | +0.238 | 0.830 | 5: blackcat, lockbit, maze, phobos, stop |
| mendeley | SVM-RBF_SW_w2v | 0.845 | 0.878 | -0.033 | +0.215 | 0.745 | 8: avaddon, babuk, blackcat, blackmatter, maze, phobos, stop, wastedlocker |
| mendeley | RF_WPC_w2v | 0.852 | 0.766 | +0.086 | +0.221 | 0.753 | 8: babuk, blackcat, darkside, doppelpaymer, maze, nefilim, phobos, stop |
| balanced | MLP_WP_w2v | 0.821 | 0.597 | +0.224 | -0.016 | 0.786 | 5: avaddon, blackcat, makop, maze, stop |
| balanced | SVM-RBF_SW_w2v | 0.775 | 0.844 | -0.069 | -0.062 | 0.685 | 10: avaddon, babuk, blackmatter, conti, doppelpaymer, lorenz, makop, maui, maze, stop |
| balanced | RF_WPC_w2v | 0.721 | 0.553 | +0.168 | -0.117 | 0.616 | 12: avaddon, blackcat, blackmatter, conti, doppelpaymer, karma, lorenz, makop, maze, phobos, quantum, stop |

