# A mnemonic sequence model under family holdout

The CNN-ViT image encoder is replaced by a sequence model over mnemonic tokens and evaluated
under the same protocol as every other pipeline: `family_holdout/folds.py`'s 5-fold family
holdout (families assigned whole, goodware assigned by duplicate-stream group on Mendeley and
by source project on Goodware_Balanced) plus leave-one-family-out over all 38 in-cohort
families. The configuration was frozen in `seq_model/config.yaml` before any test fold was
scored; the directories under `results/family_holdout/<dataset>/seq_transformer/` are written
by `family_holdout/common.py`, the same writer the other runners use.

## What changed, and why

The image encoder's failure was measured, and it was a data-representation failure in five
separable ways. Each is addressed by one change:

| the image encoder did | the sequence model does |
|---|---|
| fed token ids to a convolution as grayscale intensities, so a categorical id became an ordinal one | `nn.Embedding` over the mnemonic vocabulary; nothing ordinal survives |
| mapped only 50 mnemonics, and its API vocabulary never fired because capstone prints numeric call targets | the full vocabulary (769 model ids on a training split), fitted on the TRAINING folds, everything unseen -> `<unk>` |
| covered the first ~21,845 instructions of a file | up to 16 windows of 4096 tokens strided evenly across the WHOLE stream |
| filled the unused canvas with a constant, which leaks file size and architecture | attention masks; `tests/test_seq_model.py` asserts that junk written into the padded region does not move the pooled representation by a single bit |
| ran a 2-D convolution over a reshaped 1-D stream | a 1-D convolutional stem, then the same six ViT blocks |

The encoder itself is deliberately NOT a new design: `seq_model/model.py` imports
`TransformerBlock` and `RobustRelativeAttention` from `CNN-ViT/model_train.py` and stacks them
at `HierarchicalMalwareNet`'s own defaults (6 blocks, dim 256, 8 heads, mlp 512, dropout 0.25).
With a 4096-token window and the stem's stride of 16 the transformer sees 256 positions - exactly
the number of patches the image ViT saw. The positional signal is the learned RELATIVE
position-bias table that `RobustRelativeAttention` already carries; no absolute encoding is
added. So the comparison below is between two ways of presenting the same bytes to the same
transformer, not between two transformers.

Pooling is masked mean over the valid block positions (the image model's "masked global average
pooling"); the file logit is the MEAN of its windows' logits - a mean and not a sum, so the
number of windows a file has cannot itself become a feature.

### How much of each file is actually read

| dataset | scheme | tokens per file looked at | token-weighted coverage | per-file mean coverage |
|---|---|---|---|---|
| mendeley | sequence model, 16 x 4096 | <= 65,536 | 0.398 | 0.855 |
| mendeley | CNN-ViT image canvas, first 21,845 | <= 21,845 | 0.215 | 0.718 |
| balanced | sequence model, 16 x 4096 | <= 65,536 | 0.123 | 0.766 |
| balanced | CNN-ViT image canvas, first 21,845 | <= 21,845 | 0.058 | 0.596 |

The cap is still a cap: 16 x 4096 tokens is 65,536 mnemonics, and the Goodware_Balanced binaries
average half a million. What changed is not only how much is read but WHERE it is read from -
the image canvas was the head of the file and nothing else, these windows are spread evenly from
the first token to the last.

## The pretraining pass is transductive - state it plainly

`seq_model/pretrain.py` ran one masked-token pass (92090 steps,
15% masking, BERT's 80/10/10 corruption) over the mnemonic streams of BOTH corpora, 3,941 files
and 961 million tokens, and every fine-tuned model starts from that checkpoint. **Labels were
never read**, but the unlabelled token stream of every test-fold file and every held-out family
WAS seen. That is transductive, and the headline numbers should be read as such. The
`no_pretrain` ablation below is the strictly inductive comparison; it is the number to quote if
the question is "what would this do on a family that does not exist yet".

Fine-tuning is fold-safe regardless: the model vocabulary is refitted on each split's training
folds (769 of
769 rows transferred from the checkpoint by mnemonic
string on the run sampled here), and unseen mnemonics map to `<unk>`.

## Dataset: mendeley

2,509 files: 1,266 ransomware in 38 families,
1,243 goodware. 3 seed(s) per fold; a file's pooled score is the mean of
its seeds' P(ransomware) and the decision is argmax at 0.5 - no threshold is moved anywhere.

| model | folds macro-F1 mean +/- sd | pooled macro-F1 | pooled bal-acc | pooled AUC | pooled recall R / G | FPR | LOFO mean recall |
|---|---|---|---|---|---|---|---|
| **seq_transformer / mnemonic sequence (this work)** | 0.923 +/- 0.064 | 0.924 | 0.924 | 0.976 | 0.89 / 0.95 | 0.045 | 0.922 |
| cnn_vit / images (the model this replaces) | 0.679 +/- 0.042 | 0.744 | 0.744 | 0.809 | 0.72 / 0.77 | 0.228 | 0.681 |
| tfidf / LogReg (the pipeline to beat) | 0.954 +/- 0.021 | 0.955 | 0.955 | 0.989 | 0.93 / 0.98 | 0.019 | 0.904 |
| tokenization / MLP+WP | 0.869 +/- 0.035 | 0.869 | 0.870 | 0.943 | 0.79 / 0.95 | 0.051 | 0.830 |
| graph2vec / WL baseline | 0.942 +/- 0.037 | 0.942 | 0.942 | 0.982 | 0.91 / 0.97 | 0.028 | 0.905 |
| _floor_: majority class | 0.505 (accuracy) | | | | | | |
| _floor_: x86 rule (ransomware iff x86) | 0.656 (accuracy) | | | | | | |

Per fold (seed-mean score):

| fold | n | good | rans | macro-F1 | bal-acc | AUC | recall R / G | FPR | majority floor | x86-rule floor |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 503 | 249 | 254 | 0.968 | 0.968 | 0.994 | 1.00 / 0.94 | 0.060 | 0.505 | 0.670 |
| 1 | 502 | 249 | 253 | 0.952 | 0.953 | 0.958 | 0.91 / 1.00 | 0.004 | 0.504 | 0.681 |
| 2 | 502 | 249 | 253 | 0.932 | 0.933 | 0.989 | 0.89 / 0.97 | 0.028 | 0.504 | 0.580 |
| 3 | 499 | 248 | 251 | 0.811 | 0.814 | 0.971 | 0.69 / 0.94 | 0.056 | 0.503 | 0.647 |
| 4 | 503 | 248 | 255 | 0.952 | 0.952 | 0.993 | 0.98 / 0.92 | 0.077 | 0.507 | 0.702 |

Per architecture, over the pooled held-out predictions:

| arch | n | good | rans | accuracy | macro-F1 | recall ransomware | recall goodware | CNN-ViT recall R | CNN-ViT recall G |
|---|---|---|---|---|---|---|---|---|---|
| x64 | 608 | 494 | 114 | 0.959 | 0.926 | 0.781 | 1.000 | 0.395 | 0.879 |
| x86 | 1901 | 749 | 1152 | 0.913 | 0.910 | 0.905 | 0.925 | 0.748 | 0.702 |

## Dataset: balanced

2,603 files: 1,266 ransomware in 38 families,
1,337 goodware. 3 seed(s) per fold; a file's pooled score is the mean of
its seeds' P(ransomware) and the decision is argmax at 0.5 - no threshold is moved anywhere.

| model | folds macro-F1 mean +/- sd | pooled macro-F1 | pooled bal-acc | pooled AUC | pooled recall R / G | FPR | LOFO mean recall |
|---|---|---|---|---|---|---|---|
| **seq_transformer / mnemonic sequence (this work)** | 0.889 +/- 0.064 | 0.890 | 0.891 | 0.951 | 0.90 / 0.88 | 0.122 | 0.840 |
| cnn_vit / images (the model this replaces) | 0.736 +/- 0.036 | 0.784 | 0.784 | 0.817 | 0.72 / 0.85 | 0.153 | 0.652 |
| tfidf / LogReg (the pipeline to beat) | 0.942 +/- 0.032 | 0.943 | 0.942 | 0.980 | 0.91 / 0.97 | 0.028 | 0.893 |
| tokenization / MLP+WP | 0.820 +/- 0.043 | 0.821 | 0.821 | 0.895 | 0.77 / 0.87 | 0.127 | 0.786 |
| graph2vec / WL baseline | 0.867 +/- 0.059 | 0.867 | 0.867 | 0.947 | 0.83 / 0.90 | 0.102 | 0.789 |
| _floor_: majority class | 0.514 (accuracy) | | | | | | |
| _floor_: x86 rule (ransomware iff x86) | 0.838 (accuracy) | | | | | | |

Per fold (seed-mean score):

| fold | n | good | rans | macro-F1 | bal-acc | AUC | recall R / G | FPR | majority floor | x86-rule floor |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 539 | 285 | 254 | 0.955 | 0.957 | 0.992 | 0.99 / 0.92 | 0.077 | 0.529 | 0.796 |
| 1 | 516 | 263 | 253 | 0.884 | 0.883 | 0.946 | 0.87 / 0.90 | 0.099 | 0.510 | 0.872 |
| 2 | 516 | 263 | 253 | 0.819 | 0.821 | 0.938 | 0.90 / 0.74 | 0.259 | 0.510 | 0.769 |
| 3 | 514 | 263 | 251 | 0.835 | 0.835 | 0.904 | 0.77 / 0.90 | 0.103 | 0.512 | 0.881 |
| 4 | 518 | 263 | 255 | 0.954 | 0.954 | 0.981 | 0.98 / 0.92 | 0.076 | 0.508 | 0.873 |

Per architecture, over the pooled held-out predictions:

| arch | n | good | rans | accuracy | macro-F1 | recall ransomware | recall goodware | CNN-ViT recall R | CNN-ViT recall G |
|---|---|---|---|---|---|---|---|---|---|
| x64 | 1143 | 1029 | 114 | 0.922 | 0.806 | 0.746 | 0.942 | 0.132 | 0.957 |
| x86 | 1460 | 308 | 1152 | 0.866 | 0.796 | 0.919 | 0.666 | 0.779 | 0.481 |

## Which of the six changes mattered

Ablations are a **mendeley K-fold** study, one seed each, written to
`results/family_holdout/mendeley/seq_transformer/ablation_<name>/`. Each changes exactly one
entry of the frozen config (`seq_model/config.py::ablation`, asserted by a test) and leaves the
rest alone. Their LOFO columns are empty on purpose: 38 leave-one-family-out runs per ablation
is more GPU time than the entire main study.

| ablation | which of the six changes | folds macro-F1 mean +/- sd | delta vs the full model | pooled macro-F1 |
|---|---|---|---|---|
| _(none)_ - the frozen configuration | all six on | 0.923 +/- 0.064 | - | 0.924 |
| no_pretrain | 4. masked-token pretraining (off: random init) | 0.797 +/- 0.131 | -0.126 | 0.806 |
| first_window_only | 3. MIL over the whole file (off: one 21,840-token head window) | 0.947 +/- 0.016 | +0.024 | 0.947 |
| no_arch_balance | 6. architecture-balanced batches (off: class-balanced only) | 0.931 +/- 0.049 | +0.008 | 0.932 |
| single_seed | 5. seed ensembling (off: seed 1 alone, from the main run) | 0.928 +/- 0.069 | +0.005 | (no pooled: one seed) |

The single-seed row is read off the main run's per-seed predictions rather than retrained, as
the brief asks - it is the first seed alone, scored the same way.

## Per-family recall

K-fold recall is over the seed-mean score with the family's fold held out; LOFO recall trains on
the other 37 families plus ALL goodware. LOFO has no goodware in its test set, so it is a recall
study only - there is no FPR to read from it.

### mendeley

| family | n | n_x64 | fold | K-fold recall | LOFO recall | CNN-ViT K-fold | CNN-ViT LOFO |
|---|---|---|---|---|---|---|---|
| makop | 30 | 0 | 1 | 0.533 | 0.067 | 0.000 | 0.100 |
| doppelpaymer | 22 | 0 | 3 | 0.364 | 0.545 | 0.227 | 0.182 |
| zeppelin | 18 | 5 | 2 | 0.500 | 0.556 | 0.278 | 0.389 |
| mountlocker | 14 | 9 | 0 | 1.000 | 0.714 | 0.286 | 0.429 |
| exorcist | 17 | 4 | 4 | 0.765 | 0.765 | 0.059 | 0.059 |
| ryuk | 47 | 12 | 3 | 0.787 | 0.766 | 0.809 | 0.702 |
| blackbasta | 30 | 3 | 1 | 0.900 | 0.900 | 0.433 | 0.800 |
| conti | 48 | 0 | 1 | 0.896 | 0.917 | 0.812 | 0.917 |
| phobos | 49 | 0 | 0 | 0.980 | 0.939 | 0.980 | 0.939 |
| stop | 35 | 0 | 0 | 1.000 | 0.971 | 0.771 | 0.771 |
| wastedlocker | 36 | 0 | 3 | 0.333 | 0.972 | 0.528 | 0.417 |
| nefilim | 37 | 11 | 2 | 0.622 | 0.973 | 0.189 | 0.865 |
| clop | 45 | 0 | 4 | 0.978 | 0.978 | 0.622 | 0.578 |
| revil | 47 | 0 | 2 | 0.957 | 0.979 | 0.894 | 0.957 |
| gandcrab | 49 | 0 | 4 | 1.000 | 0.980 | 0.980 | 0.531 |
| maze | 47 | 1 | 3 | 0.340 | 1.000 | 0.936 | 0.979 |
| ransomexx | 13 | 0 | 2 | 0.846 | 1.000 | 0.692 | 0.923 |
| dharma | 46 | 0 | 1 | 0.978 | 1.000 | 0.000 | 0.000 |
| avaddon | 49 | 0 | 4 | 1.000 | 1.000 | 1.000 | 1.000 |
| avoslocker | 50 | 0 | 0 | 1.000 | 1.000 | 0.840 | 0.020 |
| babuk | 42 | 0 | 4 | 1.000 | 1.000 | 1.000 | 1.000 |
| bianlian | 11 | 11 | 3 | 1.000 | 1.000 | 1.000 | 0.636 |
| blackbyte | 7 | 7 | 1 | 1.000 | 1.000 | 0.714 | 1.000 |
| blackcat | 50 | 0 | 1 | 1.000 | 1.000 | 1.000 | 0.940 |
| blackmatter | 43 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| bluesky | 34 | 0 | 4 | 1.000 | 1.000 | 1.000 | 1.000 |
| darkside | 38 | 0 | 2 | 1.000 | 1.000 | 0.000 | 0.816 |
| hive | 50 | 43 | 2 | 1.000 | 1.000 | 0.420 | 0.200 |
| holyghost | 4 | 3 | 0 | 1.000 | 1.000 | 0.750 | 0.500 |
| karma | 13 | 0 | 4 | 1.000 | 1.000 | 0.231 | 0.231 |
| lockbit | 47 | 0 | 2 | 1.000 | 1.000 | 0.979 | 1.000 |
| lorenz | 16 | 0 | 0 | 1.000 | 1.000 | 0.875 | 1.000 |
| maui | 3 | 0 | 2 | 1.000 | 1.000 | 0.000 | 1.000 |
| netwalker | 50 | 0 | 3 | 1.000 | 1.000 | 1.000 | 0.980 |
| playcrypt | 43 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| pysa | 38 | 0 | 3 | 1.000 | 1.000 | 1.000 | 0.000 |
| quantum | 6 | 5 | 4 | 1.000 | 1.000 | 0.833 | 1.000 |
| ragnarok | 42 | 0 | 1 | 1.000 | 1.000 | 0.738 | 1.000 |

### balanced

| family | n | n_x64 | fold | K-fold recall | LOFO recall | CNN-ViT K-fold | CNN-ViT LOFO |
|---|---|---|---|---|---|---|---|
| maui | 3 | 0 | 2 | 1.000 | 0.000 | 1.000 | 1.000 |
| stop | 35 | 0 | 0 | 0.971 | 0.057 | 0.914 | 1.000 |
| maze | 47 | 1 | 3 | 0.255 | 0.128 | 0.957 | 0.979 |
| doppelpaymer | 22 | 0 | 3 | 0.455 | 0.136 | 0.364 | 0.409 |
| makop | 30 | 0 | 1 | 0.233 | 0.233 | 0.833 | 0.733 |
| zeppelin | 18 | 5 | 2 | 0.722 | 0.500 | 0.722 | 0.389 |
| ryuk | 47 | 12 | 3 | 0.809 | 0.723 | 0.681 | 0.745 |
| exorcist | 17 | 4 | 4 | 1.000 | 0.824 | 0.765 | 0.765 |
| nefilim | 37 | 11 | 2 | 0.973 | 0.865 | 0.514 | 0.486 |
| conti | 48 | 0 | 1 | 0.896 | 0.896 | 0.938 | 0.667 |
| blackbasta | 30 | 3 | 1 | 0.900 | 0.900 | 0.433 | 0.500 |
| wastedlocker | 36 | 0 | 3 | 0.972 | 0.917 | 0.750 | 0.861 |
| mountlocker | 14 | 9 | 0 | 1.000 | 0.929 | 0.643 | 0.286 |
| phobos | 49 | 0 | 0 | 0.980 | 0.939 | 1.000 | 0.959 |
| clop | 45 | 0 | 4 | 0.933 | 0.956 | 0.911 | 0.844 |
| hive | 50 | 43 | 2 | 0.640 | 0.960 | 0.300 | 0.160 |
| revil | 47 | 0 | 2 | 1.000 | 0.979 | 0.979 | 0.979 |
| gandcrab | 49 | 0 | 4 | 0.980 | 0.980 | 1.000 | 1.000 |
| ransomexx | 13 | 0 | 2 | 0.923 | 1.000 | 1.000 | 1.000 |
| ragnarok | 42 | 0 | 1 | 0.929 | 1.000 | 0.976 | 0.976 |
| avaddon | 49 | 0 | 4 | 1.000 | 1.000 | 0.000 | 1.000 |
| avoslocker | 50 | 0 | 0 | 1.000 | 1.000 | 0.840 | 0.140 |
| babuk | 42 | 0 | 4 | 1.000 | 1.000 | 1.000 | 1.000 |
| bianlian | 11 | 11 | 3 | 1.000 | 1.000 | 0.000 | 0.000 |
| blackbyte | 7 | 7 | 1 | 1.000 | 1.000 | 0.000 | 0.000 |
| blackcat | 50 | 0 | 1 | 1.000 | 1.000 | 1.000 | 0.580 |
| blackmatter | 43 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.558 |
| bluesky | 34 | 0 | 4 | 1.000 | 1.000 | 0.824 | 0.853 |
| darkside | 38 | 0 | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| dharma | 46 | 0 | 1 | 1.000 | 1.000 | 0.022 | 0.022 |
| holyghost | 4 | 3 | 0 | 1.000 | 1.000 | 0.250 | 0.250 |
| karma | 13 | 0 | 4 | 1.000 | 1.000 | 0.231 | 0.231 |
| lockbit | 47 | 0 | 2 | 1.000 | 1.000 | 0.170 | 0.149 |
| lorenz | 16 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| netwalker | 50 | 0 | 3 | 1.000 | 1.000 | 0.700 | 0.700 |
| playcrypt | 43 | 0 | 0 | 1.000 | 1.000 | 0.698 | 0.558 |
| pysa | 38 | 0 | 3 | 1.000 | 1.000 | 0.895 | 1.000 |
| quantum | 6 | 5 | 4 | 1.000 | 1.000 | 0.500 | 1.000 |

### The two called out

| family | dataset | seq_transformer (this work) K-fold / LOFO | cnn_vit / images K-fold / LOFO | tfidf / LogReg K-fold / LOFO | tokenization / MLP+WP K-fold / LOFO | graph2vec / WL baseline K-fold / LOFO |
|---|---|---|---|---|---|---|
| phobos (n=49) | mendeley | 0.98 / 0.94 | 0.98 / 0.94 | 1.00 / 1.00 | 0.04 / 0.04 | 1.00 / 1.00 |
| phobos (n=49) | balanced | 0.98 / 0.94 | 1.00 / 0.96 | 1.00 / 1.00 | 0.98 / 0.98 | 1.00 / 1.00 |
| makop (n=30) | mendeley | 0.53 / 0.07 | 0.00 / 0.10 | 0.13 / 0.37 | 0.53 / 0.53 | 0.53 / 0.53 |
| makop (n=30) | balanced | 0.23 / 0.23 | 0.83 / 0.73 | 0.53 / 0.53 | 0.03 / 0.03 | 0.53 / 0.53 |

`makop` (30 files, all x86) is the family every pipeline in this study has
struggled with - TF-IDF+LogReg gets 0.13 K-fold recall on mendeley and the
image CNN-ViT gets 0.00 - and it is the single clearest test of whether a
representation has learnt a family or a cohort-wide shortcut. `phobos` (49
files) is the opposite case: trivially detected by TF-IDF and graph2vec (1.00
everywhere) and almost completely missed by the tokenization MLP on mendeley
(0.04), so it separates pipelines rather than families.

## Runtime, and what the budget bought

| stage | runs | GPU seconds | wall-clock |
|---|---|---|---|
| mendeley K-fold + LOFO | 53 | 7,093 | 1.97 h |
| mendeley ablation no_pretrain | 5 | 655 | 0.18 h |
| mendeley ablation first_window_only | 5 | 888 | 0.25 h |
| mendeley ablation no_arch_balance | 5 | 453 | 0.13 h |
| balanced K-fold + LOFO | 53 | 7,914 | 2.20 h |
| masked-token pretraining (one pass, both corpora) | 1 | 2,700 | 0.75 h |
| learning-rate sanity check (fold 0 val only) | 3 | 439 | 0.12 h |
| **total** | | **20,141** | **5.59 h** |

The brief budgeted about four GPU-hours and sanctioned one cut if that did not
fit - "reduce seeds to 3 or windows to 8" - with LOFO never to be cut. It did
not fit. A training run costs about 120 s, so the five-seed study projects to
~5.7 h and the three-seed one to ~5.0 h; **the K-fold seed count was cut from
five to 3** and the windowing was left at 16 x 4096. Seeds rather
than windows, because multiple-instance coverage of the whole file is one of
the six changes under test and halving it would have confounded the headline.
The decision was made on the run-time arithmetic before any test fold was
read, and the seed list is not part of a run's fingerprint, so nothing about
an individual run changed. Three seeds is also what the CNN-ViT family-holdout
run used, which keeps the headline comparison like-for-like.

Even at three seeds this lands above four hours. Two engineering problems were
fixed along the way, and the first one nearly ate the budget on its own:

* **A constant batch shape.** Files have between 1 and 16 windows. A collate
  that pads each batch to its own maximum emits a different tensor shape almost
  every step, PyTorch's pinned-memory allocator caches a block per shape and
  never releases it, and the host working set walked past 11 GB until the
  machine paged - epochs went from 9 s to 190 s mid-sweep. Padding every batch
  to a fixed `(batch, 16, 4096)` and carrying ids as int32 fixed it; the padded
  window slots cost nothing on the GPU because `forward` selects the real
  windows before the encoder runs.
* **A cached token histogram.** Fitting the vocabulary on a split's training
  folds means asking which of 1,418 mnemonic strings occur in an 800 M-token
  stream, about 250 times over the study. The per-file histogram is 1,418
  uint32, so it is computed once for the whole corpus and summed thereafter.

Leave-one-family-out is 76 of the 121 training runs across the two
datasets and about half the total GPU time; it was the thing the brief said not
to cut, and it is the thing that would have had to go if the budget had not
held.

## What was NOT done

* **Imports on real data.** `--imports PATH` accepts `{sha256: [import names]}` and
  concatenates a 2,048-d signed feature-hashing vector to the pooled representation before the
  head. The ransomware import tables need the VM, which has not produced them, so the run above
  was made WITHOUT imports and the side-input is **untested on real data**. What IS tested
  (`tests/test_seq_model.py`) is the code path: hashing determinism and normalisation, the JSON
  loader, a zero vector for files with no entry, the widened head, and that changing the imports
  changes the prediction - all on synthetic input. Treat the feature as an interface, not a
  result.
* **LOFO for the ablations**, for the compute reason given above.
* **Any tuning.** Nothing in `seq_model/config.yaml` was chosen on a test fold. The only
  pre-registration measurement is the learning-rate check below.
* **Threshold moving.** Every decision in this document is argmax at 0.5.

## The one pre-registration check

`seq_model/lr_sanity.json`, produced by `seq_model/lr_sanity.py`: three learning rates on
mendeley fold 0's group-aware VAL fold (1759 train / 247 val rows).
Fold 0's 503 test rows were never loaded.

| lr | best val macro-F1 | epoch of the best | epochs run |
|---|---|---|---|
| 0.0001 | 0.9666 | 5 | 16 |
| 0.0003 | 0.9749 | 5 | 16 |
| 0.001 | 0.9709 | 9 | 16 |

argmax lr = 0.0003; the frozen config uses 0.0003 with
`max_epochs` 12. The `epoch of the best` column is the second thing
this check is for: it says whether the epoch budget reaches the val plateau or truncates it.

## Caveats carried from the fold definition

x64 ransomware concentrates in fold 2 (59 of the 114 x64 ransomware files; Hive alone is 43 of
them), so the per-fold spread of any x64 metric is not a sampling spread, and x64 numbers outside
fold 2 rest on a handful of files. This is the same caveat every other summary in this directory
carries, and it is recorded in each `metrics.json`.

## Files

* `seq_model/config.yaml` - the frozen configuration, and `seq_model/config.py` its loader and
  the four ablation definitions.
* `seq_model/data.py` - token cache, the train-only vocabulary fit, windowing, the imports
  hashing, the samplers.
* `seq_model/model.py` - the encoder (yanping blocks imported), MIL, the masked-token head.
* `seq_model/pretrain.py`, `seq_model/train.py`, `seq_model/run_family_holdout.py`,
  `seq_model/lr_sanity.py`, `seq_model/make_summary.py`.
* `tests/test_seq_model.py` - the six claims above, asserted: the train-only
  vocabulary fit, full-stream windowing under the N cap, bit-exact padding
  invariance, sampler balance, the imports hashing path, and the fold
  invariants through `family_holdout/common.py`.
* Weights and per-run records: `C:/Users/chaoa/Downloads/seq_models/` (never under `results/`).

