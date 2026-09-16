# Learned combiner (cross-validated stacking) under family holdout

Logistic regression over the two members' logit scores, three parameters, decided at 0.5. K-fold: fitted on the
other four folds' held-out member scores. LOFO: fitted on every file outside the family. Caveat: the combiner's
training rows carry member scores from models whose training data included the test fold (members were not
retrained per outer fold), so this is the standard cross-validated-stacking approximation, not nested stacking.
Written by `family_holdout/run_stacker.py`.

## Dataset: mendeley

| model | folds macro-F1 mean +/- sd | pooled macro-F1 | pooled AUC | recall R / G | FPR | LOFO mean recall |
|---|---|---|---|---|---|---|
| imports_baseline / mnem+names_LogReg | 0.973 +/- 0.011 | 0.973 | 0.993 | 0.96 / 0.99 | 0.012 | 0.950 |
| seq_transformer / seq_transformer_imports | 0.964 +/- 0.011 | 0.965 | 0.986 | 0.96 / 0.97 | 0.027 | 0.958 |
| ensemble / mean (fixed rule) | 0.971 +/- 0.014 | 0.971 | 0.991 | 0.96 / 0.98 | 0.017 | 0.966 |
| ensemble / max (fixed rule) | 0.971 +/- 0.007 | 0.971 | 0.990 | 0.97 / 0.97 | 0.031 | 0.973 |
| stacker / logit LR (learned) | 0.980 +/- 0.014 | 0.980 | 0.989 | 0.98 / 0.98 | 0.019 | 0.972 |

Per-fold weights (w on member 1, w on member 2, bias): fold 0: 0.99, 0.06, 1.03; fold 1: 1.47, 0.06, 0.97; fold 2: 1.17, -0.03, 1.01; fold 3: 1.07, 0.05, 1.17; fold 4: 1.07, 0.04, 1.18

## Dataset: balanced

| model | folds macro-F1 mean +/- sd | pooled macro-F1 | pooled AUC | recall R / G | FPR | LOFO mean recall |
|---|---|---|---|---|---|---|
| imports_baseline / mnem+names_LogReg | 0.962 +/- 0.022 | 0.962 | 0.989 | 0.94 / 0.98 | 0.020 | 0.927 |
| seq_transformer / seq_transformer_imports | 0.927 +/- 0.042 | 0.928 | 0.970 | 0.95 / 0.91 | 0.090 | 0.875 |
| ensemble / mean (fixed rule) | 0.953 +/- 0.026 | 0.953 | 0.987 | 0.95 / 0.96 | 0.045 | 0.906 |
| ensemble / max (fixed rule) | 0.931 +/- 0.040 | 0.932 | 0.982 | 0.96 / 0.90 | 0.096 | 0.947 |
| stacker / logit LR (learned) | 0.959 +/- 0.022 | 0.959 | 0.987 | 0.96 / 0.96 | 0.039 | 0.947 |

Per-fold weights (w on member 1, w on member 2, bias): fold 0: 1.22, 0.02, 0.93; fold 1: 1.26, 0.12, 0.53; fold 2: 1.28, 0.08, 0.50; fold 3: 1.22, 0.19, 0.27; fold 4: 1.12, 0.13, 0.57

