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
| seq_transformer / seq_transformer_imports | 0.964 +/- 0.011 | 0.965 | 0.986 | 0.96 / 0.97 | 0.027 | 0.958 |
| ensemble / mean (fixed rule) | 0.966 +/- 0.013 | 0.967 | 0.990 | 0.96 / 0.98 | 0.023 | 0.960 |
| ensemble / max (fixed rule) | 0.966 +/- 0.010 | 0.966 | 0.990 | 0.97 / 0.96 | 0.035 | 0.970 |
| stacker / logit LR (learned) | 0.968 +/- 0.009 | 0.968 | 0.986 | 0.97 / 0.97 | 0.035 | 0.959 |

Per-fold weights (w on member 1, w on member 2, bias): fold 0: 0.62, 0.32, 1.01; fold 1: 0.92, 0.44, 0.76; fold 2: 0.92, 0.26, 0.95; fold 3: 0.86, 0.28, 1.02; fold 4: 0.88, 0.26, 1.03

## Dataset: balanced

| model | folds macro-F1 mean +/- sd | pooled macro-F1 | pooled AUC | recall R / G | FPR | LOFO mean recall |
|---|---|---|---|---|---|---|
| tfidf / LogReg | 0.942 +/- 0.028 | 0.943 | 0.980 | 0.91 / 0.97 | 0.028 | 0.893 |
| seq_transformer / seq_transformer_imports | 0.927 +/- 0.042 | 0.928 | 0.970 | 0.95 / 0.91 | 0.090 | 0.875 |
| ensemble / mean (fixed rule) | 0.949 +/- 0.028 | 0.949 | 0.982 | 0.94 / 0.96 | 0.043 | 0.892 |
| ensemble / max (fixed rule) | 0.930 +/- 0.044 | 0.931 | 0.978 | 0.96 / 0.90 | 0.096 | 0.943 |
| stacker / logit LR (learned) | 0.950 +/- 0.017 | 0.950 | 0.981 | 0.95 / 0.95 | 0.048 | 0.932 |

Per-fold weights (w on member 1, w on member 2, bias): fold 0: 0.87, 0.22, 0.80; fold 1: 0.97, 0.30, 0.59; fold 2: 0.99, 0.27, 0.74; fold 3: 0.87, 0.36, 0.42; fold 4: 0.96, 0.24, 0.85

