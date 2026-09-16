# Ensemble of tfidf/LogReg and seq_transformer/seq_transformer_imports under family holdout

Two pre-registered, parameter-free rules over the members' held-out P(ransomware):
`mean` (primary) and `max` (the OR rule, secondary). No weight, threshold or stacker is fitted;
argmax at 0.5 everywhere. Members are the pre-registered TF-IDF/LogReg baseline and the frozen
sequence-transformer configuration; both K-fold and LOFO rows reuse their held-out predictions
unchanged, so the ensemble is exactly as leakage-free as its members. `fold mean +/- sd` is over
the five held-out folds (population sd, as in summary.md). Written by `family_holdout/run_ensemble.py`.

## Dataset: mendeley

| model | folds macro-F1 mean +/- sd | pooled macro-F1 | pooled AUC | recall R / G | FPR | LOFO mean recall |
|---|---|---|---|---|---|---|
| tfidf / LogReg | 0.954 +/- 0.018 | 0.955 | 0.989 | 0.93 / 0.98 | 0.019 | 0.904 |
| seq_transformer / seq_transformer_imports | 0.964 +/- 0.011 | 0.965 | 0.986 | 0.96 / 0.97 | 0.027 | 0.958 |
| ensemble / mean (primary) | 0.966 +/- 0.013 | 0.967 | 0.990 | 0.96 / 0.98 | 0.023 | 0.960 |
| ensemble / max | 0.966 +/- 0.010 | 0.966 | 0.990 | 0.97 / 0.96 | 0.035 | 0.970 |

K-fold disagreement between the two members: 97 files (3.9%). Oracle (pick whichever member is right per file): ransomware recall 0.968 against members 0.928 / 0.956; goodware recall 0.990 against 0.981 / 0.973. The oracle is an upper bound on any combination rule, not a result.

Families where the primary (`mean`) ensemble differs from BOTH members by at least 0.05 (K-fold recall / LOFO recall):

| family | tfidf / LogReg | seq_transformer | ensemble mean |
|---|---|---|---|
| doppelpaymer | 0.95 / 1.00 | 0.55 / 0.82 | 0.68 / 0.95 |
| stop | 0.63 / 0.63 | 1.00 / 1.00 | 1.00 / 0.91 |
| wastedlocker | 0.97 / 0.67 | 0.97 / 0.89 | 0.97 / 0.78 |
| zeppelin | 0.56 / 0.56 | 0.56 / 0.61 | 0.56 / 0.72 |

## Dataset: balanced

| model | folds macro-F1 mean +/- sd | pooled macro-F1 | pooled AUC | recall R / G | FPR | LOFO mean recall |
|---|---|---|---|---|---|---|
| tfidf / LogReg | 0.942 +/- 0.028 | 0.943 | 0.980 | 0.91 / 0.97 | 0.028 | 0.893 |
| seq_transformer / seq_transformer_imports | 0.927 +/- 0.042 | 0.928 | 0.970 | 0.95 / 0.91 | 0.090 | 0.875 |
| ensemble / mean (primary) | 0.949 +/- 0.028 | 0.949 | 0.982 | 0.94 / 0.96 | 0.043 | 0.892 |
| ensemble / max | 0.930 +/- 0.044 | 0.931 | 0.978 | 0.96 / 0.90 | 0.096 | 0.943 |

K-fold disagreement between the two members: 177 files (6.8%). Oracle (pick whichever member is right per file): ransomware recall 0.960 against members 0.912 / 0.946; goodware recall 0.978 against 0.972 / 0.910. The oracle is an upper bound on any combination rule, not a result.

Families where the primary (`mean`) ensemble differs from BOTH members by at least 0.05 (K-fold recall / LOFO recall):

| family | tfidf / LogReg | seq_transformer | ensemble mean |
|---|---|---|---|
| hive | 0.64 / 0.70 | 0.92 / 0.92 | 0.74 / 0.86 |
| stop | 0.80 / 0.60 | 0.80 / 0.09 | 0.94 / 0.26 |

