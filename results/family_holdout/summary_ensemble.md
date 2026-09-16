# Ensemble of TF-IDF and the sequence transformer under family holdout

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
| seq_transformer | 0.923 +/- 0.057 | 0.924 | 0.976 | 0.89 / 0.95 | 0.045 | 0.922 |
| ensemble / mean (primary) | 0.955 +/- 0.026 | 0.955 | 0.988 | 0.93 / 0.98 | 0.024 | 0.917 |
| ensemble / max | 0.951 +/- 0.009 | 0.951 | 0.988 | 0.96 / 0.95 | 0.053 | 0.954 |

K-fold disagreement between the two members: 167 files (6.7%). Oracle (pick whichever member is right per file): ransomware recall 0.956 against members 0.928 / 0.893; goodware recall 0.990 against 0.981 / 0.955. The oracle is an upper bound on any combination rule, not a result.

Families where the primary (`mean`) ensemble differs from BOTH members by at least 0.05 (K-fold recall / LOFO recall):

| family | tfidf / LogReg | seq_transformer | ensemble mean |
|---|---|---|---|
| doppelpaymer | 0.95 / 1.00 | 0.36 / 0.55 | 0.55 / 0.59 |
| ransomexx | 0.77 / 0.69 | 0.85 / 1.00 | 0.85 / 0.85 |
| wastedlocker | 0.97 / 0.67 | 0.33 / 0.97 | 0.67 / 0.97 |

## Dataset: balanced

| model | folds macro-F1 mean +/- sd | pooled macro-F1 | pooled AUC | recall R / G | FPR | LOFO mean recall |
|---|---|---|---|---|---|---|
| tfidf / LogReg | 0.942 +/- 0.028 | 0.943 | 0.980 | 0.91 / 0.97 | 0.028 | 0.893 |
| seq_transformer | 0.889 +/- 0.057 | 0.890 | 0.951 | 0.90 / 0.88 | 0.122 | 0.840 |
| ensemble / mean (primary) | 0.930 +/- 0.036 | 0.931 | 0.975 | 0.91 / 0.95 | 0.049 | 0.878 |
| ensemble / max | 0.911 +/- 0.050 | 0.911 | 0.966 | 0.95 / 0.87 | 0.126 | 0.936 |

K-fold disagreement between the two members: 242 files (9.3%). Oracle (pick whichever member is right per file): ransomware recall 0.950 against members 0.912 / 0.904; goodware recall 0.975 against 0.972 / 0.878. The oracle is an upper bound on any combination rule, not a result.

Families where the primary (`mean`) ensemble differs from BOTH members by at least 0.05 (K-fold recall / LOFO recall):

| family | tfidf / LogReg | seq_transformer | ensemble mean |
|---|---|---|---|
| makop | 0.53 / 0.53 | 0.23 / 0.23 | 0.27 / 0.30 |
| maze | 1.00 / 1.00 | 0.26 / 0.13 | 0.49 / 0.26 |
| nefilim | 0.73 / 0.78 | 0.97 / 0.86 | 0.97 / 0.97 |
| zeppelin | 0.67 / 0.67 | 0.72 / 0.50 | 0.72 / 0.56 |

