# Learned combiner (cross-validated stacking) under family holdout

Logistic regression over the two members' logit scores, three parameters, decided at 0.5. K-fold: fitted on the
other four folds' held-out member scores. LOFO: fitted on every file outside the family. Caveat: the combiner's
training rows carry member scores from models whose training data included the test fold (members were not
retrained per outer fold), so this is the standard cross-validated-stacking approximation, not nested stacking.
Written by `family_holdout/run_stacker.py`.

## Dataset: mendeley

| model | folds macro-F1 mean +/- sd | pooled macro-F1 | pooled AUC | recall R / G | FPR | LOFO mean recall |
|---|---|---|---|---|---|---|
| tfidf / LogReg | 0.954 +/- 0.018 | 0.955 | 0.989 | 0.93 / 0.98 | 0.019 | 0.904 |
| seq_transformer / seq_transformer | 0.923 +/- 0.057 | 0.924 | 0.976 | 0.89 / 0.95 | 0.045 | 0.922 |
| ensemble / mean (fixed rule) | 0.955 +/- 0.026 | 0.955 | 0.988 | 0.93 / 0.98 | 0.024 | 0.917 |
| ensemble / max (fixed rule) | 0.951 +/- 0.009 | 0.951 | 0.988 | 0.96 / 0.95 | 0.053 | 0.954 |
| stacker / logit LR (learned) | 0.959 +/- 0.006 | 0.959 | 0.985 | 0.96 / 0.96 | 0.042 | 0.944 |

Per-fold weights (w on member 1, w on member 2, bias): fold 0: 1.08, 0.08, 1.13; fold 1: 1.19, 0.24, 1.06; fold 2: 1.15, 0.12, 1.23; fold 3: 1.05, 0.15, 1.18; fold 4: 1.16, 0.10, 1.34

## Dataset: balanced

| model | folds macro-F1 mean +/- sd | pooled macro-F1 | pooled AUC | recall R / G | FPR | LOFO mean recall |
|---|---|---|---|---|---|---|
| tfidf / LogReg | 0.942 +/- 0.028 | 0.943 | 0.980 | 0.91 / 0.97 | 0.028 | 0.893 |
| seq_transformer / seq_transformer | 0.889 +/- 0.057 | 0.890 | 0.951 | 0.90 / 0.88 | 0.122 | 0.840 |
| ensemble / mean (fixed rule) | 0.930 +/- 0.036 | 0.931 | 0.975 | 0.91 / 0.95 | 0.049 | 0.878 |
| ensemble / max (fixed rule) | 0.911 +/- 0.050 | 0.911 | 0.966 | 0.95 / 0.87 | 0.126 | 0.936 |
| stacker / logit LR (learned) | 0.949 +/- 0.010 | 0.949 | 0.978 | 0.95 / 0.95 | 0.052 | 0.934 |

Per-fold weights (w on member 1, w on member 2, bias): fold 0: 1.22, 0.02, 1.18; fold 1: 1.19, 0.17, 0.79; fold 2: 1.39, 0.09, 1.02; fold 3: 1.23, 0.12, 0.81; fold 4: 1.26, 0.05, 1.02

