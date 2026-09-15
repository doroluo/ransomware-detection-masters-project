# Family-holdout: `tfidf`

Every ransomware family in the cohort is held out, twice over, with the fold definition in `family_holdout/folds.py` (`results/family_holdout/folds_<dataset>.csv`):

* **K-fold** - families are assigned whole to 5 folds; train on four, test on the fifth. Every file gets exactly one held-out prediction, so the pooled row is a complete cross-validated pass over the cohort.
* **LOFO** - leave one family out: train on every other ransomware family plus ALL goodware, test on that family alone. No goodware in the test set, so it is a recall study only (no FPR, no AUC).

Nothing here is tuned. The configuration is the pre-registered one for each pipeline; see `config_used.yaml` in each model directory.

> **Architecture caveat, carried into every table below.** x64 ransomware concentrates in fold 2: 59 of the 114 x64 ransomware files land there, and Hive alone is 43 of them. A per-fold x64 number outside fold 2 rests on a handful of files, and the per-fold spread of any x64 metric is not a sampling spread.

## Headline

### mendeley

| model | fold mean macro-F1 +/- sd | pooled macro-F1 | pooled bal-acc | pooled acc | pooled AUC | majority floor (acc / macro-F1) | x86-rule floor (acc / macro-F1) | LOFO mean recall | fixed-split macro-F1 |
|---|---|---|---|---|---|---|---|---|---|
| LogReg | 0.954 +/- 0.021 | **0.955** | 0.955 | 0.955 | 0.989 | 0.505 / 0.335 | 0.656 / 0.631 | 0.904 | 0.968 |
| LinearSVC | 0.919 +/- 0.045 | **0.919** | 0.920 | 0.919 | 0.990 | 0.505 / 0.335 | 0.656 / 0.631 | 0.825 | 0.960 |

Per-fold spread of the other headline metrics:

| model | acc (mean +/- sd) | bal-acc (mean +/- sd) | AUC (mean +/- sd) |
|---|---|---|---|
| LogReg | 0.955 +/- 0.021 | 0.955 +/- 0.020 | 0.991 +/- 0.003 |
| LinearSVC | 0.920 +/- 0.043 | 0.920 +/- 0.043 | 0.990 +/- 0.006 |

### balanced

| model | fold mean macro-F1 +/- sd | pooled macro-F1 | pooled bal-acc | pooled acc | pooled AUC | majority floor (acc / macro-F1) | x86-rule floor (acc / macro-F1) | LOFO mean recall | fixed-split macro-F1 |
|---|---|---|---|---|---|---|---|---|---|
| LogReg | 0.942 +/- 0.032 | **0.943** | 0.942 | 0.943 | 0.980 | 0.514 / 0.339 | 0.838 / 0.838 | 0.893 | 0.796 |
| LinearSVC | 0.922 +/- 0.056 | **0.923** | 0.922 | 0.924 | 0.985 | 0.514 / 0.339 | 0.838 / 0.838 | 0.859 | 0.802 |

Per-fold spread of the other headline metrics:

| model | acc (mean +/- sd) | bal-acc (mean +/- sd) | AUC (mean +/- sd) |
|---|---|---|---|
| LogReg | 0.942 +/- 0.032 | 0.942 +/- 0.032 | 0.981 +/- 0.013 |
| LinearSVC | 0.923 +/- 0.054 | 0.922 +/- 0.054 | 0.985 +/- 0.012 |

## Per architecture (pooled K-fold predictions)

### mendeley

| model | arch | n | support r / g | recall ransomware | recall goodware | accuracy | macro-F1 |
|---|---|---|---|---|---|---|---|
| LogReg | x64 | 608 | 114 / 494 | 0.789 | 0.998 | 0.959 | 0.927 |
| LogReg | x86 | 1901 | 1152 / 749 | 0.942 | 0.971 | 0.953 | 0.951 |
| LinearSVC | x64 | 608 | 114 / 494 | 0.719 | 0.998 | 0.946 | 0.900 |
| LinearSVC | x86 | 1901 | 1152 / 749 | 0.871 | 0.973 | 0.911 | 0.909 |

### balanced

| model | arch | n | support r / g | recall ransomware | recall goodware | accuracy | macro-F1 |
|---|---|---|---|---|---|---|---|
| LogReg | x64 | 1143 | 114 / 1029 | 0.667 | 0.999 | 0.966 | 0.889 |
| LogReg | x86 | 1460 | 1152 / 308 | 0.937 | 0.880 | 0.925 | 0.891 |
| LinearSVC | x64 | 1143 | 114 / 1029 | 0.667 | 1.000 | 0.967 | 0.891 |
| LinearSVC | x86 | 1460 | 1152 / 308 | 0.879 | 0.929 | 0.890 | 0.853 |

## Per family: K-fold recall vs LOFO recall

Sorted by mean LOFO recall, worst first. `fold` is the K-fold the family is assigned to; `n_x64` is how many of its files are x64.

### mendeley

| family | n | n_x64 | fold | LogReg K-fold | LogReg LOFO | LinearSVC K-fold | LinearSVC LOFO | mean LOFO |
|---|---|---|---|---|---|---|---|---|
| exorcist | 17 | 4 | 4 | 0.59 | 0.12 | 0.12 | 0.12 | 0.12 |
| makop | 30 | 0 | 1 | 0.13 | 0.37 | 0.10 | 0.10 | 0.23 |
| pysa | 38 | 0 | 3 | 1.00 | 1.00 | 1.00 | 0.00 | 0.50 |
| stop | 35 | 0 | 0 | 0.63 | 0.63 | 0.37 | 0.40 | 0.51 |
| phobos | 49 | 0 | 0 | 1.00 | 1.00 | 0.06 | 0.06 | 0.53 |
| wastedlocker | 36 | 0 | 3 | 0.97 | 0.67 | 0.67 | 0.42 | 0.54 |
| zeppelin | 18 | 5 | 2 | 0.56 | 0.56 | 0.56 | 0.56 | 0.56 |
| ransomexx | 13 | 0 | 2 | 0.77 | 0.69 | 0.77 | 0.69 | 0.69 |
| ryuk | 47 | 12 | 3 | 0.72 | 0.72 | 0.74 | 0.74 | 0.73 |
| nefilim | 37 | 11 | 2 | 0.81 | 0.97 | 0.62 | 0.70 | 0.84 |
| mountlocker | 14 | 9 | 0 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 |
| blackbasta | 30 | 3 | 1 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 |
| conti | 48 | 0 | 1 | 0.90 | 0.94 | 0.94 | 0.94 | 0.94 |
| doppelpaymer | 22 | 0 | 3 | 0.95 | 1.00 | 0.95 | 0.91 | 0.95 |
| babuk | 42 | 0 | 4 | 1.00 | 0.95 | 1.00 | 1.00 | 0.98 |
| dharma | 46 | 0 | 1 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 |
| revil | 47 | 0 | 2 | 0.98 | 1.00 | 0.98 | 0.98 | 0.99 |
| holyghost | 4 | 3 | 0 | 1.00 | 1.00 | 0.75 | 1.00 | 1.00 |
| avoslocker | 50 | 0 | 0 | 1.00 | 1.00 | 0.90 | 1.00 | 1.00 |
| hive | 50 | 43 | 2 | 1.00 | 1.00 | 0.90 | 1.00 | 1.00 |
| avaddon | 49 | 0 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| bianlian | 11 | 11 | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| blackbyte | 7 | 7 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| blackcat | 50 | 0 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| blackmatter | 43 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| bluesky | 34 | 0 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| clop | 45 | 0 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| darkside | 38 | 0 | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| gandcrab | 49 | 0 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| karma | 13 | 0 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| lockbit | 47 | 0 | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| lorenz | 16 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| maui | 3 | 0 | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| maze | 47 | 1 | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| netwalker | 50 | 0 | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| playcrypt | 43 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| quantum | 6 | 5 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| ragnarok | 42 | 0 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### balanced

| family | n | n_x64 | fold | LogReg K-fold | LogReg LOFO | LinearSVC K-fold | LinearSVC LOFO | mean LOFO |
|---|---|---|---|---|---|---|---|---|
| blackcat | 50 | 0 | 1 | 0.60 | 0.00 | 0.00 | 0.00 | 0.00 |
| makop | 30 | 0 | 1 | 0.53 | 0.53 | 0.10 | 0.07 | 0.30 |
| exorcist | 17 | 4 | 4 | 0.76 | 0.76 | 0.12 | 0.12 | 0.44 |
| doppelpaymer | 22 | 0 | 3 | 0.41 | 0.41 | 0.50 | 0.59 | 0.50 |
| stop | 35 | 0 | 0 | 0.80 | 0.60 | 0.46 | 0.40 | 0.50 |
| hive | 50 | 43 | 2 | 0.64 | 0.70 | 0.64 | 0.64 | 0.67 |
| zeppelin | 18 | 5 | 2 | 0.67 | 0.67 | 0.72 | 0.72 | 0.69 |
| ryuk | 47 | 12 | 3 | 0.74 | 0.74 | 0.74 | 0.74 | 0.74 |
| nefilim | 37 | 11 | 2 | 0.73 | 0.78 | 0.73 | 0.73 | 0.76 |
| blackbasta | 30 | 3 | 1 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 |
| mountlocker | 14 | 9 | 0 | 1.00 | 0.93 | 0.93 | 0.93 | 0.93 |
| revil | 47 | 0 | 2 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 |
| conti | 48 | 0 | 1 | 0.98 | 0.96 | 0.96 | 0.96 | 0.96 |
| ransomexx | 13 | 0 | 2 | 1.00 | 1.00 | 0.92 | 0.92 | 0.96 |
| wastedlocker | 36 | 0 | 3 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 |
| avaddon | 49 | 0 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| avoslocker | 50 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| babuk | 42 | 0 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| bianlian | 11 | 11 | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| blackbyte | 7 | 7 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| blackmatter | 43 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| bluesky | 34 | 0 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| clop | 45 | 0 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| darkside | 38 | 0 | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| dharma | 46 | 0 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| gandcrab | 49 | 0 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| holyghost | 4 | 3 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| karma | 13 | 0 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| lockbit | 47 | 0 | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| lorenz | 16 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| maui | 3 | 0 | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| maze | 47 | 1 | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| netwalker | 50 | 0 | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| phobos | 49 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| playcrypt | 43 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| pysa | 38 | 0 | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| quantum | 6 | 5 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| ragnarok | 42 | 0 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Against the fixed split

The `fixed-split macro-F1` column of the headline table is the same model's row in `results/COMPARISON.md`, i.e. the single Mendeley family-disjoint split (24 train / 14 test families). The two are not the same experiment: the fixed split tests 14 families with a 74%-ransomware test set, the K-fold pooled row tests all 38 with a roughly balanced one, and LOFO gives every family the largest possible training set. A family-holdout number above the fixed-split one usually means the fixed split's 14 test families were the harder half, not that the model improved.

| dataset | model | pooled macro-F1 (K-fold) | fixed-split macro-F1 | delta | headroom over the x86-rule floor | LOFO mean recall | families with LOFO recall < 0.5 |
|---|---|---|---|---|---|---|---|
| mendeley | LogReg | 0.955 | 0.968 | -0.013 | +0.324 | 0.904 | 2: exorcist, makop |
| mendeley | LinearSVC | 0.919 | 0.960 | -0.041 | +0.289 | 0.825 | 6: exorcist, makop, phobos, pysa, stop, wastedlocker |
| balanced | LogReg | 0.943 | 0.796 | +0.147 | +0.105 | 0.893 | 2: blackcat, doppelpaymer |
| balanced | LinearSVC | 0.923 | 0.802 | +0.121 | +0.085 | 0.859 | 4: blackcat, exorcist, makop, stop |

