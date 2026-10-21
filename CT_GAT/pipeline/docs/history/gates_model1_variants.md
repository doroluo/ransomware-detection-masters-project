# Gate summary, stage model1_variants (seed 0 only, not the pre-registered test)

kfold x86 on the v2 cache. `tokenseq` is Model 1, the token-sequence transformer over the block export (plan `model1`); `tokenseq-thoof` is the same model with the decision threshold chosen on out-of-fold predictions of the training binaries (3 refits on folds that keep a ransomware family or a goodware identity group whole, `--threshold-from oof`) instead of the validation families, which saturate; the two differ only in the threshold of the AUC checkpoint (the seed and the training are the same, so the ranking is the same and the AUC is level up to run-to-run noise). `five-channel` is Janice Park's Model 1 over the five-channel instruction tuples (opcode, two operands, API, behaviour) of a pipeline-B build, its first 128 (as she wrote it) or 512 instructions per binary, trained on our folds and validation split with the API vocabulary made a training-set one per fold. R7 typed and R2 no-edges carry 15% block masking. The pooled table is seed 0; the paired differences, second minus first for the AUC checkpoint at its validation threshold, use the seeds complete in both arms (queue seeds [0, 1, 2]), with the 95% paired cluster bootstrap interval (2000 resamples). The rule column is the pre-registered rule (dAUC >= 0.01 and dmacro-F1 >= 0.02, both 95% intervals excluding zero) read as "the variant beats the reference". TF-IDF exists for seed 0 only, so its pairs are a direction. A missing arm or fold is stated.

## Pooled, AUC checkpoint

| arm | group | AUC | PR-AUC | macro-F1 |
|---|---|---|---|---|
| TF-IDF | tfidf | 0.982 | 0.990 | 0.935 |
| header-imports LR | imports-header | 0.985 | 0.991 | 0.954 |
| R7 typed | kfold_x86_gnn-mblk15 | 0.986 | 0.990 | 0.935 |
| R2 no-edges | kfold_x86_gnn-noedges-mblk15 | 0.977 | 0.983 | 0.941 |
| Model 1 tokenseq | kfold_x86_tokenseq | 0.964 | 0.980 | 0.882 |
| Model 1 tokenseq, out-of-fold threshold | kfold_x86_tokenseq-thoof | 0.971 | 0.984 | 0.921 |
| Model 1 five-channel L128 | kfold_x86_fivechannel | 0.910 | 0.952 | 0.836 |
| Model 1 five-channel L512 | kfold_x86_fivechannel-L512 | 0.919 | 0.958 | 0.878 |
| team sig500v3 | kfold_x86_gnn_team-sig500v3 | 0.959 | 0.979 | 0.889 |

## Paired differences, second minus first

| first | second | dAUC [95% CI] | dmacro-F1 [95% CI] | rule |
|---|---|---|---|---|
| Model 1 tokenseq | Model 1 tokenseq, out-of-fold threshold | -0.002 [-0.007, +0.002] | +0.016 [-0.014, +0.054] | does not beat |
| TF-IDF | Model 1 tokenseq, out-of-fold threshold | -0.011 [-0.023, -0.002] | -0.014 [-0.063, +0.021] | does not beat (seed 0 only: a direction, not the test) |
| Model 1 tokenseq | Model 1 five-channel L128 | -0.051 [-0.083, -0.024] | -0.039 [-0.097, +0.026] | does not beat |
| TF-IDF | Model 1 five-channel L128 | -0.072 [-0.117, -0.034] | -0.099 [-0.147, -0.058] | does not beat (seed 0 only: a direction, not the test) |
| Model 1 tokenseq | Model 1 five-channel L512 | -0.043 [-0.071, -0.020] | -0.024 [-0.063, +0.022] | does not beat |
| TF-IDF | Model 1 five-channel L512 | -0.063 [-0.100, -0.032] | -0.057 [-0.120, -0.006] | does not beat (seed 0 only: a direction, not the test) |

### kfold_x86_tokenseq-thoof minus kfold_x86_tokenseq (selection auc, 2000 resamples)

a = kfold_x86_tokenseq, b = kfold_x86_tokenseq-thoof; seeds [0, 1, 2]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.970 | 0.968 | -0.002 | [-0.007, +0.002] |
| PR-AUC | 0.982 | 0.981 | -0.001 | [-0.005, +0.003] |
| macro-F1 | 0.902 | 0.918 | +0.016 | [-0.014, +0.054] |

### kfold_x86_tokenseq-thoof minus tfidf (selection auc, 2000 resamples)

a = tfidf, b = kfold_x86_tokenseq-thoof; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.982 | 0.971 | -0.011 | [-0.023, -0.002] |
| PR-AUC | 0.990 | 0.984 | -0.006 | [-0.015, -0.000] |
| macro-F1 | 0.935 | 0.921 | -0.014 | [-0.063, +0.021] |

### kfold_x86_fivechannel minus kfold_x86_tokenseq (selection auc, 2000 resamples)

a = kfold_x86_tokenseq, b = kfold_x86_fivechannel; seeds [0, 1, 2]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.970 | 0.919 | -0.051 | [-0.083, -0.024] |
| PR-AUC | 0.982 | 0.954 | -0.028 | [-0.052, -0.012] |
| macro-F1 | 0.902 | 0.863 | -0.039 | [-0.097, +0.026] |

### kfold_x86_fivechannel minus tfidf (selection auc, 2000 resamples)

a = tfidf, b = kfold_x86_fivechannel; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.982 | 0.910 | -0.072 | [-0.117, -0.034] |
| PR-AUC | 0.990 | 0.952 | -0.038 | [-0.070, -0.019] |
| macro-F1 | 0.935 | 0.836 | -0.099 | [-0.147, -0.058] |

### kfold_x86_fivechannel-L512 minus kfold_x86_tokenseq (selection auc, 2000 resamples)

a = kfold_x86_tokenseq, b = kfold_x86_fivechannel-L512; seeds [0, 1, 2]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.970 | 0.926 | -0.043 | [-0.071, -0.020] |
| PR-AUC | 0.982 | 0.958 | -0.024 | [-0.045, -0.011] |
| macro-F1 | 0.902 | 0.878 | -0.024 | [-0.063, +0.022] |

### kfold_x86_fivechannel-L512 minus tfidf (selection auc, 2000 resamples)

a = tfidf, b = kfold_x86_fivechannel-L512; seeds [0]; 1810 test binaries in 712 clusters. Differences are b - a, 95% percentile interval.

| metric | a | b | b - a | 95% CI |
|---|---|---|---|---|
| AUC | 0.982 | 0.919 | -0.063 | [-0.100, -0.032] |
| PR-AUC | 0.990 | 0.958 | -0.032 | [-0.058, -0.016] |
| macro-F1 | 0.935 | 0.878 | -0.057 | [-0.120, -0.006] |

