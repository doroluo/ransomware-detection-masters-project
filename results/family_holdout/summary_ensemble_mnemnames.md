# Ensemble of imports_baseline/mnem+names_LogReg and seq_transformer/seq_transformer_imports under family holdout

Two pre-registered, parameter-free rules over the members' held-out P(ransomware):
`mean` (primary) and `max` (the OR rule, secondary). No weight, threshold or stacker is fitted;
argmax at 0.5 everywhere. Members are the pre-registered TF-IDF/LogReg baseline and the frozen
sequence-transformer configuration; both K-fold and LOFO rows reuse their held-out predictions
unchanged, so the ensemble is exactly as leakage-free as its members. `fold mean +/- sd` is over
the five held-out folds (population sd, as in summary.md). Written by `family_holdout/run_ensemble.py`.

## Dataset: mendeley

| model | folds macro-F1 mean +/- sd | pooled macro-F1 | pooled AUC | recall R / G | FPR | LOFO mean recall |
|---|---|---|---|---|---|---|
| imports_baseline / mnem+names_LogReg | 0.973 +/- 0.011 | 0.973 | 0.993 | 0.96 / 0.99 | 0.012 | 0.950 |
| seq_transformer / seq_transformer_imports | 0.964 +/- 0.011 | 0.965 | 0.986 | 0.96 / 0.97 | 0.027 | 0.958 |
| ensemble / mean (primary) | 0.971 +/- 0.014 | 0.971 | 0.991 | 0.96 / 0.98 | 0.017 | 0.966 |
| ensemble / max | 0.971 +/- 0.007 | 0.971 | 0.990 | 0.97 / 0.97 | 0.031 | 0.973 |

K-fold disagreement between the two members: 68 files (2.7%). Oracle (pick whichever member is right per file): ransomware recall 0.973 against members 0.959 / 0.956; goodware recall 0.992 against 0.988 / 0.973. The oracle is an upper bound on any combination rule, not a result.

## Dataset: balanced

| model | folds macro-F1 mean +/- sd | pooled macro-F1 | pooled AUC | recall R / G | FPR | LOFO mean recall |
|---|---|---|---|---|---|---|
| imports_baseline / mnem+names_LogReg | 0.962 +/- 0.022 | 0.962 | 0.989 | 0.94 / 0.98 | 0.020 | 0.927 |
| seq_transformer / seq_transformer_imports | 0.927 +/- 0.042 | 0.928 | 0.970 | 0.95 / 0.91 | 0.090 | 0.875 |
| ensemble / mean (primary) | 0.953 +/- 0.026 | 0.953 | 0.987 | 0.95 / 0.96 | 0.045 | 0.906 |
| ensemble / max | 0.931 +/- 0.040 | 0.932 | 0.982 | 0.96 / 0.90 | 0.096 | 0.947 |

K-fold disagreement between the two members: 153 files (5.9%). Oracle (pick whichever member is right per file): ransomware recall 0.961 against members 0.943 / 0.946; goodware recall 0.987 against 0.980 / 0.910. The oracle is an upper bound on any combination rule, not a result.

Families where the primary (`mean`) ensemble differs from BOTH members by at least 0.05 (K-fold recall / LOFO recall):

| family | tfidf / LogReg | seq_transformer | ensemble mean |
|---|---|---|---|
| hive | 0.86 / 1.00 | 0.92 / 0.92 | 0.80 / 0.94 |
| stop | 1.00 / 0.80 | 0.80 / 0.09 | 0.97 / 0.46 |

