# Family-holdout: `graph2vec`

Every ransomware family in the cohort is held out, twice over, with the fold definition in `family_holdout/folds.py` (`results/family_holdout/folds_<dataset>.csv`):

* **K-fold** - families are assigned whole to 5 folds; train on four, test on the fifth. Every file gets exactly one held-out prediction, so the pooled row is a complete cross-validated pass over the cohort.
* **LOFO** - leave one family out: train on every other ransomware family plus ALL goodware, test on that family alone. No goodware in the test set, so it is a recall study only (no FPR, no AUC).

Nothing here is tuned. The configuration is the pre-registered one for each pipeline; see `config_used.yaml` in each model directory.

> **Architecture caveat, carried into every table below.** x64 ransomware concentrates in fold 2: 59 of the 114 x64 ransomware files land there, and Hive alone is 43 of them. A per-fold x64 number outside fold 2 rests on a handful of files, and the per-fold spread of any x64 metric is not a sampling spread.

## Headline

### mendeley

| model | fold mean macro-F1 +/- sd | pooled macro-F1 | pooled bal-acc | pooled acc | pooled AUC | majority floor (acc / macro-F1) | x86-rule floor (acc / macro-F1) | LOFO mean recall | fixed-split macro-F1 |
|---|---|---|---|---|---|---|---|---|---|
| baseline_h2_cap5000_argmax | 0.891 +/- 0.070 | **0.892** | 0.893 | 0.892 | 0.982 | 0.505 / 0.335 | 0.656 / 0.631 | 0.804 | 0.724 |
| baseline_h2_cap5000_oofthr | 0.942 +/- 0.037 | **0.942** | 0.942 | 0.942 | 0.982 | 0.505 / 0.335 | 0.656 / 0.631 | 0.905 | 0.885 |
| tuned_best_sparse_histogram_argmax | 0.835 +/- 0.086 | **0.837** | 0.842 | 0.840 | 0.980 | 0.505 / 0.335 | 0.656 / 0.631 | 0.707 | 0.772 |
| tuned_best_sparse_histogram_oofthr | 0.885 +/- 0.057 | **0.886** | 0.888 | 0.887 | 0.980 | 0.505 / 0.335 | 0.656 / 0.631 | 0.777 | 0.947 |

Per-fold spread of the other headline metrics:

| model | acc (mean +/- sd) | bal-acc (mean +/- sd) | AUC (mean +/- sd) |
|---|---|---|---|
| baseline_h2_cap5000_argmax | 0.892 +/- 0.068 | 0.893 +/- 0.068 | 0.982 +/- 0.015 |
| baseline_h2_cap5000_oofthr | 0.942 +/- 0.037 | 0.942 +/- 0.037 | 0.982 +/- 0.015 |
| tuned_best_sparse_histogram_argmax | 0.840 +/- 0.081 | 0.841 +/- 0.080 | 0.981 +/- 0.011 |
| tuned_best_sparse_histogram_oofthr | 0.887 +/- 0.055 | 0.888 +/- 0.055 | 0.981 +/- 0.011 |

### balanced

| model | fold mean macro-F1 +/- sd | pooled macro-F1 | pooled bal-acc | pooled acc | pooled AUC | majority floor (acc / macro-F1) | x86-rule floor (acc / macro-F1) | LOFO mean recall | fixed-split macro-F1 |
|---|---|---|---|---|---|---|---|---|---|
| baseline_h2_cap5000_argmax | 0.798 +/- 0.077 | **0.801** | 0.803 | 0.807 | 0.947 | 0.514 / 0.339 | 0.838 / 0.838 | 0.651 | 0.472 |
| baseline_h2_cap5000_oofthr | 0.867 +/- 0.059 | **0.867** | 0.867 | 0.867 | 0.947 | 0.514 / 0.339 | 0.838 / 0.838 | 0.789 | 0.568 |
| tuned_best_sparse_histogram_argmax | 0.709 +/- 0.077 | **0.712** | 0.727 | 0.734 | 0.946 | 0.514 / 0.339 | 0.838 / 0.838 | 0.484 | n/a |
| tuned_best_sparse_histogram_oofthr | 0.751 +/- 0.056 | **0.752** | 0.760 | 0.765 | 0.946 | 0.514 / 0.339 | 0.838 / 0.838 | 0.623 | n/a |

Per-fold spread of the other headline metrics:

| model | acc (mean +/- sd) | bal-acc (mean +/- sd) | AUC (mean +/- sd) |
|---|---|---|---|
| baseline_h2_cap5000_argmax | 0.807 +/- 0.069 | 0.803 +/- 0.071 | 0.945 +/- 0.020 |
| baseline_h2_cap5000_oofthr | 0.868 +/- 0.058 | 0.867 +/- 0.059 | 0.945 +/- 0.020 |
| tuned_best_sparse_histogram_argmax | 0.734 +/- 0.063 | 0.727 +/- 0.066 | 0.949 +/- 0.027 |
| tuned_best_sparse_histogram_oofthr | 0.766 +/- 0.048 | 0.759 +/- 0.051 | 0.949 +/- 0.027 |

## Per architecture (pooled K-fold predictions)

### mendeley

| model | arch | n | support r / g | recall ransomware | recall goodware | accuracy | macro-F1 |
|---|---|---|---|---|---|---|---|
| baseline_h2_cap5000_argmax | x64 | 608 | 114 / 494 | 0.632 | 1.000 | 0.931 | 0.867 |
| baseline_h2_cap5000_argmax | x86 | 1901 | 1152 / 749 | 0.825 | 0.965 | 0.880 | 0.878 |
| baseline_h2_cap5000_oofthr | x64 | 608 | 114 / 494 | 0.684 | 0.998 | 0.939 | 0.886 |
| baseline_h2_cap5000_oofthr | x86 | 1901 | 1152 / 749 | 0.935 | 0.955 | 0.943 | 0.941 |
| tuned_best_sparse_histogram_argmax | x64 | 608 | 114 / 494 | 0.605 | 0.998 | 0.924 | 0.853 |
| tuned_best_sparse_histogram_argmax | x86 | 1901 | 1152 / 749 | 0.704 | 0.981 | 0.813 | 0.813 |
| tuned_best_sparse_histogram_oofthr | x64 | 608 | 114 / 494 | 0.728 | 0.996 | 0.946 | 0.901 |
| tuned_best_sparse_histogram_oofthr | x86 | 1901 | 1152 / 749 | 0.798 | 0.976 | 0.868 | 0.867 |

### balanced

| model | arch | n | support r / g | recall ransomware | recall goodware | accuracy | macro-F1 |
|---|---|---|---|---|---|---|---|
| baseline_h2_cap5000_argmax | x64 | 1143 | 114 / 1029 | 0.474 | 0.999 | 0.947 | 0.805 |
| baseline_h2_cap5000_argmax | x86 | 1460 | 1152 / 308 | 0.661 | 0.834 | 0.698 | 0.657 |
| baseline_h2_cap5000_oofthr | x64 | 1143 | 114 / 1029 | 0.474 | 0.997 | 0.945 | 0.801 |
| baseline_h2_cap5000_oofthr | x86 | 1460 | 1152 / 308 | 0.871 | 0.568 | 0.807 | 0.715 |
| tuned_best_sparse_histogram_argmax | x64 | 1143 | 114 / 1029 | 0.509 | 0.996 | 0.948 | 0.815 |
| tuned_best_sparse_histogram_argmax | x86 | 1460 | 1152 / 308 | 0.471 | 0.922 | 0.566 | 0.552 |
| tuned_best_sparse_histogram_oofthr | x64 | 1143 | 114 / 1029 | 0.561 | 0.995 | 0.952 | 0.837 |
| tuned_best_sparse_histogram_oofthr | x86 | 1460 | 1152 / 308 | 0.551 | 0.873 | 0.619 | 0.594 |

## Per family: K-fold recall vs LOFO recall

Sorted by mean LOFO recall, worst first. `fold` is the K-fold the family is assigned to; `n_x64` is how many of its files are x64.

### mendeley

| family | n | n_x64 | fold | baseline_h2_cap5000_argmax K-fold | baseline_h2_cap5000_argmax LOFO | baseline_h2_cap5000_oofthr K-fold | baseline_h2_cap5000_oofthr LOFO | tuned_best_sparse_histogram_argmax K-fold | tuned_best_sparse_histogram_argmax LOFO | tuned_best_sparse_histogram_oofthr K-fold | tuned_best_sparse_histogram_oofthr LOFO | mean LOFO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| phobos | 49 | 0 | 0 | 0.06 | 0.06 | 1.00 | 1.00 | 0.06 | 0.06 | 0.06 | 0.06 | 0.30 |
| makop | 30 | 0 | 1 | 0.27 | 0.53 | 0.53 | 0.53 | 0.07 | 0.07 | 0.07 | 0.07 | 0.30 |
| mountlocker | 14 | 9 | 0 | 0.57 | 0.50 | 0.71 | 0.71 | 0.21 | 0.21 | 0.21 | 0.07 | 0.38 |
| wastedlocker | 36 | 0 | 3 | 0.39 | 0.39 | 0.39 | 0.39 | 0.39 | 0.39 | 0.39 | 0.39 | 0.39 |
| exorcist | 17 | 4 | 4 | 0.76 | 0.76 | 0.76 | 0.76 | 0.12 | 0.12 | 0.12 | 0.12 | 0.44 |
| revil | 47 | 0 | 2 | 0.43 | 0.45 | 0.98 | 1.00 | 0.15 | 0.17 | 0.17 | 0.17 | 0.45 |
| pysa | 38 | 0 | 3 | 0.00 | 0.00 | 1.00 | 1.00 | 0.11 | 0.11 | 1.00 | 1.00 | 0.53 |
| stop | 35 | 0 | 0 | 0.63 | 0.63 | 0.69 | 0.63 | 0.46 | 0.43 | 0.43 | 0.43 | 0.53 |
| doppelpaymer | 22 | 0 | 3 | 0.55 | 0.55 | 0.55 | 0.55 | 0.55 | 0.55 | 0.59 | 0.55 | 0.55 |
| zeppelin | 18 | 5 | 2 | 0.56 | 0.56 | 0.56 | 0.56 | 0.56 | 0.56 | 0.56 | 0.67 | 0.58 |
| darkside | 38 | 0 | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.45 | 0.45 | 0.61 |
| ryuk | 47 | 12 | 3 | 0.74 | 0.74 | 0.74 | 0.74 | 0.74 | 0.74 | 0.74 | 0.74 | 0.74 |
| quantum | 6 | 5 | 4 | 1.00 | 0.50 | 1.00 | 1.00 | 0.83 | 0.83 | 0.83 | 0.83 | 0.79 |
| holyghost | 4 | 3 | 0 | 0.75 | 0.75 | 1.00 | 1.00 | 0.75 | 0.75 | 0.75 | 0.75 | 0.81 |
| hive | 50 | 43 | 2 | 0.78 | 0.78 | 0.84 | 0.84 | 0.64 | 0.64 | 1.00 | 1.00 | 0.81 |
| ransomexx | 13 | 0 | 2 | 0.85 | 0.85 | 0.92 | 0.92 | 0.77 | 0.77 | 0.85 | 0.77 | 0.83 |
| nefilim | 37 | 11 | 2 | 0.73 | 0.81 | 0.81 | 0.97 | 0.54 | 0.62 | 0.89 | 0.95 | 0.84 |
| ragnarok | 42 | 0 | 1 | 0.95 | 0.95 | 1.00 | 1.00 | 0.55 | 0.55 | 0.90 | 0.88 | 0.85 |
| conti | 48 | 0 | 1 | 0.90 | 0.90 | 0.90 | 0.90 | 0.88 | 0.88 | 0.88 | 0.88 | 0.89 |
| blackbasta | 30 | 3 | 1 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 |
| lorenz | 16 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 0.69 | 0.88 | 0.69 | 0.88 | 0.94 |
| maze | 47 | 1 | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 0.60 | 0.79 | 1.00 | 1.00 | 0.95 |
| babuk | 42 | 0 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 0.93 | 0.93 | 1.00 | 1.00 | 0.98 |
| dharma | 46 | 0 | 1 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 1.00 | 0.98 |
| blackmatter | 43 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 0.98 | 0.98 | 0.98 | 0.99 |
| gandcrab | 49 | 0 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 0.98 | 1.00 | 1.00 | 0.99 |
| avoslocker | 50 | 0 | 0 | 0.98 | 0.98 | 1.00 | 1.00 | 0.90 | 1.00 | 0.90 | 1.00 | 0.99 |
| avaddon | 49 | 0 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| bianlian | 11 | 11 | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| blackbyte | 7 | 7 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| blackcat | 50 | 0 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| bluesky | 34 | 0 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| clop | 45 | 0 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| karma | 13 | 0 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| lockbit | 47 | 0 | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| maui | 3 | 0 | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| netwalker | 50 | 0 | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| playcrypt | 43 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### balanced

| family | n | n_x64 | fold | baseline_h2_cap5000_argmax K-fold | baseline_h2_cap5000_argmax LOFO | baseline_h2_cap5000_oofthr K-fold | baseline_h2_cap5000_oofthr LOFO | tuned_best_sparse_histogram_argmax K-fold | tuned_best_sparse_histogram_argmax LOFO | tuned_best_sparse_histogram_oofthr K-fold | tuned_best_sparse_histogram_oofthr LOFO | mean LOFO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blackcat | 50 | 0 | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| avaddon | 49 | 0 | 4 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| doppelpaymer | 22 | 0 | 3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.14 | 0.09 | 0.45 | 0.45 | 0.14 |
| avoslocker | 50 | 0 | 0 | 0.00 | 0.00 | 0.12 | 0.12 | 0.00 | 0.00 | 0.04 | 0.84 | 0.24 |
| mountlocker | 14 | 9 | 0 | 0.36 | 0.21 | 0.36 | 0.36 | 0.14 | 0.07 | 0.57 | 0.50 | 0.29 |
| makop | 30 | 0 | 1 | 0.07 | 0.53 | 0.53 | 0.53 | 0.03 | 0.03 | 0.07 | 0.07 | 0.29 |
| lorenz | 16 | 0 | 0 | 0.12 | 0.12 | 1.00 | 1.00 | 0.00 | 0.00 | 0.12 | 0.12 | 0.31 |
| nefilim | 37 | 11 | 2 | 0.51 | 0.51 | 0.51 | 0.54 | 0.19 | 0.19 | 0.32 | 0.32 | 0.39 |
| exorcist | 17 | 4 | 4 | 0.76 | 0.76 | 0.76 | 0.76 | 0.12 | 0.12 | 0.12 | 0.12 | 0.44 |
| pysa | 38 | 0 | 3 | 0.03 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.42 | 0.95 | 0.49 |
| quantum | 6 | 5 | 4 | 0.17 | 0.17 | 0.17 | 0.17 | 0.83 | 0.83 | 0.83 | 0.83 | 0.50 |
| phobos | 49 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 0.06 | 0.06 | 0.06 | 0.06 | 0.53 |
| zeppelin | 18 | 5 | 2 | 0.50 | 0.50 | 0.56 | 0.56 | 0.50 | 0.50 | 0.50 | 0.67 | 0.56 |
| blackbasta | 30 | 3 | 1 | 0.40 | 0.20 | 0.90 | 0.87 | 0.57 | 0.50 | 0.77 | 0.67 | 0.56 |
| revil | 47 | 0 | 2 | 0.94 | 0.96 | 1.00 | 0.98 | 0.15 | 0.15 | 0.15 | 0.15 | 0.56 |
| wastedlocker | 36 | 0 | 3 | 0.47 | 0.64 | 0.97 | 0.97 | 0.22 | 0.25 | 0.39 | 0.39 | 0.56 |
| maze | 47 | 1 | 3 | 0.53 | 0.83 | 1.00 | 1.00 | 0.23 | 0.23 | 0.26 | 0.26 | 0.58 |
| darkside | 38 | 0 | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.39 | 0.39 | 0.60 |
| netwalker | 50 | 0 | 3 | 0.52 | 0.52 | 1.00 | 1.00 | 0.52 | 0.52 | 0.52 | 0.52 | 0.64 |
| stop | 35 | 0 | 0 | 0.83 | 0.74 | 1.00 | 1.00 | 0.60 | 0.23 | 0.86 | 0.63 | 0.65 |
| playcrypt | 43 | 0 | 0 | 0.56 | 0.56 | 0.93 | 0.93 | 0.56 | 0.56 | 0.56 | 0.56 | 0.65 |
| hive | 50 | 43 | 2 | 0.64 | 0.64 | 0.78 | 0.78 | 0.64 | 0.64 | 0.64 | 0.64 | 0.68 |
| ransomexx | 13 | 0 | 2 | 0.85 | 0.85 | 1.00 | 1.00 | 0.15 | 0.23 | 0.69 | 0.69 | 0.69 |
| ragnarok | 42 | 0 | 1 | 1.00 | 0.98 | 1.00 | 1.00 | 0.38 | 0.38 | 0.55 | 0.55 | 0.73 |
| ryuk | 47 | 12 | 3 | 0.72 | 0.72 | 0.74 | 0.74 | 0.72 | 0.72 | 0.72 | 0.72 | 0.73 |
| holyghost | 4 | 3 | 0 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| conti | 48 | 0 | 1 | 0.88 | 0.88 | 0.94 | 0.94 | 0.81 | 0.75 | 0.90 | 0.90 | 0.86 |
| karma | 13 | 0 | 4 | 1.00 | 0.77 | 1.00 | 1.00 | 0.77 | 0.77 | 1.00 | 1.00 | 0.88 |
| clop | 45 | 0 | 4 | 0.93 | 0.93 | 0.98 | 0.98 | 0.93 | 0.93 | 0.93 | 0.93 | 0.94 |
| babuk | 42 | 0 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 | 0.90 | 0.93 | 0.98 | 0.97 |
| dharma | 46 | 0 | 1 | 0.98 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 |
| gandcrab | 49 | 0 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 0.98 | 0.98 | 1.00 | 0.99 |
| bianlian | 11 | 11 | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| blackbyte | 7 | 7 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| blackmatter | 43 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| bluesky | 34 | 0 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| lockbit | 47 | 0 | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| maui | 3 | 0 | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Against the fixed split

The `fixed-split macro-F1` column of the headline table is the same model's row in `results/COMPARISON.md`, i.e. the single Mendeley family-disjoint split (24 train / 14 test families). The two are not the same experiment: the fixed split tests 14 families with a 74%-ransomware test set, the K-fold pooled row tests all 38 with a roughly balanced one, and LOFO gives every family the largest possible training set. A family-holdout number above the fixed-split one usually means the fixed split's 14 test families were the harder half, not that the model improved.

| dataset | model | pooled macro-F1 (K-fold) | fixed-split macro-F1 | delta | headroom over the x86-rule floor | LOFO mean recall | families with LOFO recall < 0.5 |
|---|---|---|---|---|---|---|---|
| mendeley | baseline_h2_cap5000_argmax | 0.892 | 0.724 | +0.168 | +0.261 | 0.804 | 4: phobos, pysa, revil, wastedlocker |
| mendeley | baseline_h2_cap5000_oofthr | 0.942 | 0.885 | +0.057 | +0.311 | 0.905 | 1: wastedlocker |
| mendeley | tuned_best_sparse_histogram_argmax | 0.837 | 0.772 | +0.065 | +0.206 | 0.707 | 9: darkside, exorcist, makop, mountlocker, phobos, pysa, revil, stop, wastedlocker |
| mendeley | tuned_best_sparse_histogram_oofthr | 0.886 | 0.947 | -0.061 | +0.255 | 0.777 | 8: darkside, exorcist, makop, mountlocker, phobos, revil, stop, wastedlocker |
| balanced | baseline_h2_cap5000_argmax | 0.801 | 0.472 | +0.329 | -0.037 | 0.651 | 9: avaddon, avoslocker, blackbasta, blackcat, doppelpaymer, lorenz, mountlocker, pysa, quantum |
| balanced | baseline_h2_cap5000_oofthr | 0.867 | 0.568 | +0.299 | +0.030 | 0.789 | 6: avaddon, avoslocker, blackcat, doppelpaymer, mountlocker, quantum |
| balanced | tuned_best_sparse_histogram_argmax | 0.712 | n/a | n/a | -0.125 | 0.484 | 18: avaddon, avoslocker, blackcat, darkside, doppelpaymer, exorcist, lorenz, makop, maze, mountlocker, nefilim, phobos, pysa, ragnarok, ransomexx, revil, stop, wastedlocker |
| balanced | tuned_best_sparse_histogram_oofthr | 0.752 | n/a | n/a | -0.085 | 0.623 | 12: avaddon, blackcat, darkside, doppelpaymer, exorcist, lorenz, makop, maze, nefilim, phobos, revil, wastedlocker |

