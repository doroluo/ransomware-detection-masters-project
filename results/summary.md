# Experiment summary: six variants of one binary task

Binary task throughout: **0 = goodware, 1 = ransomware**. The ransomware family prefix is a split group, never a label.

Two variables are separated here, and they used to move together:

| | sample set | feature form |
|---|---|---|
| **expA** | Mendeley as shipped | full instruction per line |
| **expA_cohort** | cohort-filtered | full instruction per line |
| **expC** | cohort-filtered | mnemonic only |
| **expB** | Mendeley ransomware + Goodware_Balanced, as shipped | full instruction per line |
| **expB_cohort** | cohort-filtered | full instruction per line |
| **expD** | cohort-filtered | mnemonic only |

The **cohort** drops .NET, entropy-packed, UPX-unrecoverable, no-code, odd-architecture, broken and duplicate samples, and the `thanos` family. The **revised** feature form comes from `extract_unified.py` (capstone skip-data sweep, uncapped) via `asm_tool/mn_to_features.py`, which applies the cohort filter as it writes - so expC and expD are cohort-only by construction and differ from expA_cohort and expB_cohort in the feature form alone.

Everything else is held fixed across all six: the same tokenizer, embedding and classifier settings, the same seeds, the same ransomware split, and - for expB, expB_cohort and expD - the same goodware split membership.

## expA - Mendeley ransomware + Mendeley goodware (the LLM_Features baseline)

| split | goodware | ransomware | total | groups |
|---|---|---|---|---|
| train | 1116 | 975 | 2091 | 1141 |
| test | 131 | 382 | 513 | 146 |

| split | class | x86 / x64 |
|---|---|---|
| train | ransomware | 6 unknown (0.6%), 42 x64 (4.3%), 927 x86 (95.1%) |
| train | goodware | 3 unknown (0.3%), 483 x64 (43.3%), 630 x86 (56.5%) |
| test | ransomware | 4 unknown (1.0%), 88 x64 (23.0%), 290 x86 (75.9%) |
| test | goodware | 2 unknown (1.5%), 12 x64 (9.2%), 117 x86 (89.3%) |

No cohort filter. 2500 of 2604 files would survive one.

Exact-duplicate leakage: **64/131 goodware** and **1/382 ransomware** test files have a byte-identical opcode stream in train (1765/2604 streams are unique; 2 appear under both labels).


| model | tok | emb | mask | acc | bal-acc | macro-P | macro-R | macro-F1 | AUC | recall(ran) | FPR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RF | WPC | w2v | 0.0 | 0.7797 | 0.8019 | 0.7397 | 0.8019 | 0.7496 | 0.8826 | 0.7565 | 0.1527 |
| MLP | WP | w2v | 0.0 | 0.8343 | 0.8662 | 0.7938 | 0.8662 | 0.8098 | 0.8155 | 0.8010 | 0.0687 |
| SVM-RBF | SW | w2v | 0.0 | 0.7661 | 0.8028 | 0.7353 | 0.8028 | 0.7398 | 0.9152 | 0.7277 | 0.1221 |
| *floor* - majority class | - | - | - | 0.7446 | 0.5000 | - | - | 0.4268 | - | 1.0000 | 1.0000 |
| *floor* - x86 means ransomware | - | - | - | 0.5926 | 0.4330 | - | - | 0.4266 | - | 0.7592 | 0.8931 |

The two floor rows read no opcodes. `majority class` calls every test file ransomware; `x86 means ransomware` reads only the architecture recorded in the cohort CSV (407 of 513 test files are x86, 6 are `unknown` and count as not-x86). A model row is evidence about opcodes only to the extent that it clears both.


#### Test metrics inside each architecture

| model / tok | unknown n | unknown rec(good) | unknown rec(ran) | unknown acc | unknown macro-F1 | x64 n | x64 rec(good) | x64 rec(ran) | x64 acc | x64 macro-F1 | x86 n | x86 rec(good) | x86 rec(ran) | x86 acc | x86 macro-F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RF/WPC | 6 | 1.0000 (2) | 0.0000 (4) | 0.3333 | 0.2500 | 100 | 0.9167 (12) | 0.2045 (88) | 0.2900 | 0.2865 | 407 | 0.8376 (117) | 0.9345 (290) | 0.9066 | 0.8860 |
| MLP/WP | 6 | 1.0000 (2) | 0.0000 (4) | 0.3333 | 0.2500 | 100 | 1.0000 (12) | 0.2159 (88) | 0.3100 | 0.3066 | 407 | 0.9231 (117) | 0.9897 (290) | 0.9705 | 0.9634 |
| SVM-RBF/SW | 6 | 1.0000 (2) | 0.0000 (4) | 0.3333 | 0.2500 | 100 | 0.9167 (12) | 0.1932 (88) | 0.2800 | 0.2774 | 407 | 0.8718 (117) | 0.9000 (290) | 0.8919 | 0.8724 |

`rec(good)` and `rec(ran)` carry their class support in brackets. A slice holding only one class gets `macro-F1 -`: averaging an F1 over a class with no support reports "the model failed" when the truth is "the question was not asked".


#### Ransomware test recall by family

| family | n | RF/WPC | MLP/WP | SVM-RBF/SW |
|---|---|---|---|---|
| avoslocker | 50 | 1.0000 | 1.0000 | 1.0000 |
| blackcat | 50 | 0.7400 | 1.0000 | 0.4200 |
| hive | 50 | 0.1400 | 0.1600 | 0.1600 |
| clop | 45 | 0.8889 | 0.9333 | 1.0000 |
| playcrypt | 43 | 1.0000 | 1.0000 | 1.0000 |
| bluesky | 34 | 1.0000 | 1.0000 | 1.0000 |
| blackbasta | 30 | 0.9000 | 0.9000 | 0.9000 |
| lorenz | 16 | 1.0000 | 1.0000 | 1.0000 |
| nightsky | 14 | 0.1429 | 0.1429 | 0.1429 |
| blackbyte | 13 | 0.1538 | 0.1538 | 0.0000 |
| karma | 13 | 1.0000 | 1.0000 | 1.0000 |
| bianlian | 11 | 1.0000 | 1.0000 | 1.0000 |
| quantum | 6 | 0.0000 | 0.1667 | 0.1667 |
| holyghost | 4 | 1.0000 | 1.0000 | 1.0000 |
| maui | 3 | 1.0000 | 1.0000 | 1.0000 |

## expA_cohort - Exp A restricted to the cohort (traditional features, same folders as the split)

| split | goodware | ransomware | total | groups |
|---|---|---|---|---|
| train | 1109 | 900 | 2009 | 1133 |
| test | 129 | 362 | 491 | 143 |

| split | class | x86 / x64 |
|---|---|---|
| train | ransomware | 42 x64 (4.7%), 858 x86 (95.3%) |
| train | goodware | 482 x64 (43.5%), 627 x86 (56.5%) |
| test | ransomware | 72 x64 (19.9%), 290 x86 (80.1%) |
| test | goodware | 12 x64 (9.3%), 117 x86 (90.7%) |

Cohort filter: kept **2500 of 2604** files (dropped 104: family:thanos 1, tag:dup 15, tag:packed_other 88).

Exact-duplicate leakage: **64/129 goodware** and **1/362 ransomware** test files have a byte-identical opcode stream in train (1690/2500 streams are unique; 2 appear under both labels).


| model | tok | emb | mask | acc | bal-acc | macro-P | macro-R | macro-F1 | AUC | recall(ran) | FPR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RF | WPC | w2v | 0.0 | 0.7393 | 0.7808 | 0.7194 | 0.7808 | 0.7166 | 0.8420 | 0.6934 | 0.1318 |
| MLP | WP | w2v | 0.0 | 0.8086 | 0.8402 | 0.7734 | 0.8402 | 0.7848 | 0.8213 | 0.7735 | 0.0930 |
| SVM-RBF | SW | w2v | 0.0 | 0.7841 | 0.8162 | 0.7518 | 0.8162 | 0.7595 | 0.9152 | 0.7486 | 0.1163 |
| *floor* - majority class | - | - | - | 0.7373 | 0.5000 | - | - | 0.4244 | - | 1.0000 | 1.0000 |
| *floor* - x86 means ransomware | - | - | - | 0.6151 | 0.4471 | - | - | 0.4335 | - | 0.8011 | 0.9070 |

The two floor rows read no opcodes. `majority class` calls every test file ransomware; `x86 means ransomware` reads only the architecture recorded in the cohort CSV (407 of 491 test files are x86). A model row is evidence about opcodes only to the extent that it clears both.


#### Test metrics inside each architecture

| model / tok | x64 n | x64 rec(good) | x64 rec(ran) | x64 acc | x64 macro-F1 | x86 n | x86 rec(good) | x86 rec(ran) | x86 acc | x86 macro-F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| RF/WPC | 84 | 0.9167 (12) | 0.1944 (72) | 0.2976 | 0.2967 | 407 | 0.8632 (117) | 0.8172 (290) | 0.8305 | 0.8092 |
| MLP/WP | 84 | 1.0000 (12) | 0.2083 (72) | 0.3214 | 0.3206 | 407 | 0.8974 (117) | 0.9138 (290) | 0.9091 | 0.8925 |
| SVM-RBF/SW | 84 | 0.9167 (12) | 0.2083 (72) | 0.3095 | 0.3080 | 407 | 0.8803 (117) | 0.8828 (290) | 0.8821 | 0.8627 |

`rec(good)` and `rec(ran)` carry their class support in brackets. A slice holding only one class gets `macro-F1 -`: averaging an F1 over a class with no support reports "the model failed" when the truth is "the question was not asked".


#### Ransomware test recall by family

| family | n | RF/WPC | MLP/WP | SVM-RBF/SW |
|---|---|---|---|---|
| avoslocker | 50 | 1.0000 | 1.0000 | 1.0000 |
| blackcat | 50 | 0.4200 | 1.0000 | 0.4200 |
| hive | 50 | 0.1400 | 0.1600 | 0.1600 |
| clop | 45 | 0.8889 | 0.8667 | 0.8889 |
| playcrypt | 43 | 0.6977 | 0.6279 | 1.0000 |
| bluesky | 34 | 0.8529 | 1.0000 | 1.0000 |
| blackbasta | 30 | 0.9000 | 0.9000 | 0.9000 |
| lorenz | 16 | 1.0000 | 1.0000 | 1.0000 |
| karma | 13 | 1.0000 | 0.8462 | 1.0000 |
| bianlian | 11 | 1.0000 | 1.0000 | 1.0000 |
| blackbyte | 7 | 0.0000 | 0.0000 | 0.0000 |
| quantum | 6 | 0.0000 | 0.0000 | 0.1667 |
| holyghost | 4 | 1.0000 | 1.0000 | 1.0000 |
| maui | 3 | 1.0000 | 1.0000 | 1.0000 |

## expC - REVISED Mendeley ransomware + REVISED Mendeley goodware (mnemonic-only features)

| split | goodware | ransomware | total | groups |
|---|---|---|---|---|
| train | 1114 | 904 | 2018 | 1138 |
| test | 129 | 362 | 491 | 143 |

| split | class | x86 / x64 |
|---|---|---|
| train | ransomware | 42 x64 (4.6%), 862 x86 (95.4%) |
| train | goodware | 482 x64 (43.3%), 632 x86 (56.7%) |
| test | ransomware | 72 x64 (19.9%), 290 x86 (80.1%) |
| test | goodware | 12 x64 (9.3%), 117 x86 (90.7%) |

The feature folders are already cohort-only, and this run verified it: all 2509 files carry `in_cohort == 1`.

Exact-duplicate leakage: **62/129 goodware** and **0/362 ransomware** test files have a byte-identical opcode stream in train (1655/2509 streams are unique; 2 appear under both labels).


| model | tok | emb | mask | acc | bal-acc | macro-P | macro-R | macro-F1 | AUC | recall(ran) | FPR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RF | WPC | w2v | 0.0 | 0.7902 | 0.8228 | 0.7575 | 0.8228 | 0.7660 | 0.9482 | 0.7541 | 0.1085 |
| MLP | WP | w2v | 0.0 | 0.9430 | 0.9239 | 0.9282 | 0.9239 | 0.9260 | 0.9520 | 0.9641 | 0.1163 |
| SVM-RBF | SW | w2v | 0.0 | 0.9002 | 0.8999 | 0.8624 | 0.8999 | 0.8779 | 0.9594 | 0.9006 | 0.1008 |
| *floor* - majority class | - | - | - | 0.7373 | 0.5000 | - | - | 0.4244 | - | 1.0000 | 1.0000 |
| *floor* - x86 means ransomware | - | - | - | 0.6151 | 0.4471 | - | - | 0.4335 | - | 0.8011 | 0.9070 |

The two floor rows read no opcodes. `majority class` calls every test file ransomware; `x86 means ransomware` reads only the architecture recorded in the cohort CSV (407 of 491 test files are x86). A model row is evidence about opcodes only to the extent that it clears both.


#### Test metrics inside each architecture

| model / tok | x64 n | x64 rec(good) | x64 rec(ran) | x64 acc | x64 macro-F1 | x86 n | x86 rec(good) | x86 rec(ran) | x86 acc | x86 macro-F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| RF/WPC | 84 | 1.0000 (12) | 0.2222 (72) | 0.3333 | 0.3318 | 407 | 0.8803 (117) | 0.8862 (290) | 0.8845 | 0.8652 |
| MLP/WP | 84 | 0.9167 (12) | 0.9444 (72) | 0.9405 | 0.8897 | 407 | 0.8803 (117) | 0.9690 (290) | 0.9435 | 0.9301 |
| SVM-RBF/SW | 84 | 0.8333 (12) | 0.9861 (72) | 0.9643 | 0.9244 | 407 | 0.9060 (117) | 0.8793 (290) | 0.8870 | 0.8695 |

`rec(good)` and `rec(ran)` carry their class support in brackets. A slice holding only one class gets `macro-F1 -`: averaging an F1 over a class with no support reports "the model failed" when the truth is "the question was not asked".


#### Ransomware test recall by family

| family | n | RF/WPC | MLP/WP | SVM-RBF/SW |
|---|---|---|---|---|
| avoslocker | 50 | 1.0000 | 1.0000 | 1.0000 |
| blackcat | 50 | 0.5400 | 1.0000 | 0.5400 |
| hive | 50 | 0.3400 | 1.0000 | 1.0000 |
| clop | 45 | 0.8000 | 0.8000 | 0.7556 |
| playcrypt | 43 | 1.0000 | 1.0000 | 1.0000 |
| bluesky | 34 | 1.0000 | 1.0000 | 1.0000 |
| blackbasta | 30 | 0.9333 | 0.9333 | 0.9667 |
| lorenz | 16 | 1.0000 | 1.0000 | 1.0000 |
| karma | 13 | 1.0000 | 1.0000 | 1.0000 |
| bianlian | 11 | 0.0000 | 1.0000 | 1.0000 |
| blackbyte | 7 | 0.0000 | 0.7143 | 1.0000 |
| quantum | 6 | 0.8333 | 1.0000 | 0.8333 |
| holyghost | 4 | 0.2500 | 1.0000 | 1.0000 |
| maui | 3 | 1.0000 | 1.0000 | 1.0000 |

## expB - Mendeley ransomware + Goodware_Balanced (hard-negative-rich goodware)

| split | goodware | ransomware | total | groups |
|---|---|---|---|---|
| train | 1116 | 975 | 2091 | 295 |
| test | 131 | 382 | 513 | 59 |

| split | class | x86 / x64 |
|---|---|---|
| train | ransomware | 6 unknown (0.6%), 42 x64 (4.3%), 927 x86 (95.1%) |
| train | goodware | 915 x64 (82.0%), 201 x86 (18.0%) |
| test | ransomware | 4 unknown (1.0%), 88 x64 (23.0%), 290 x86 (75.9%) |
| test | goodware | 85 x64 (64.9%), 46 x86 (35.1%) |

No cohort filter. 2502 of 2604 files would survive one.

Exact-duplicate leakage: **1/131 goodware** and **1/382 ransomware** test files have a byte-identical opcode stream in train (1980/2604 streams are unique; 0 appear under both labels).


| model | tok | emb | mask | acc | bal-acc | macro-P | macro-R | macro-F1 | AUC | recall(ran) | FPR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RF | WPC | w2v | 0.0 | 0.6842 | 0.7378 | 0.6809 | 0.7378 | 0.6629 | 0.7781 | 0.6283 | 0.1527 |
| MLP | WP | w2v | 0.0 | 0.7037 | 0.7358 | 0.6812 | 0.7358 | 0.6756 | 0.7469 | 0.6702 | 0.1985 |
| SVM-RBF | SW | w2v | 0.0 | 0.5945 | 0.6901 | 0.6509 | 0.6901 | 0.5862 | 0.8214 | 0.4948 | 0.1145 |
| *floor* - majority class | - | - | - | 0.7446 | 0.5000 | - | - | 0.4268 | - | 1.0000 | 1.0000 |
| *floor* - x86 means ransomware | - | - | - | 0.7310 | 0.7040 | - | - | 0.6799 | - | 0.7592 | 0.3511 |

The two floor rows read no opcodes. `majority class` calls every test file ransomware; `x86 means ransomware` reads only the architecture recorded in the cohort CSV (336 of 513 test files are x86, 4 are `unknown` and count as not-x86). A model row is evidence about opcodes only to the extent that it clears both.


#### Test metrics inside each architecture

| model / tok | unknown n | unknown rec(good) | unknown rec(ran) | unknown acc | unknown macro-F1 | x64 n | x64 rec(good) | x64 rec(ran) | x64 acc | x64 macro-F1 | x86 n | x86 rec(good) | x86 rec(ran) | x86 acc | x86 macro-F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RF/WPC | 4 | - (0) | 0.0000 (4) | 0.0000 | - | 173 | 0.9882 (85) | 0.0909 (88) | 0.5318 | 0.4198 | 336 | 0.5870 (46) | 0.8000 (290) | 0.7708 | 0.6349 |
| MLP/WP | 4 | - (0) | 0.0000 (4) | 0.0000 | - | 173 | 1.0000 (85) | 0.1818 (88) | 0.5838 | 0.5051 | 336 | 0.4348 (46) | 0.8276 (290) | 0.7738 | 0.6041 |
| SVM-RBF/SW | 4 | - (0) | 0.0000 (4) | 0.0000 | - | 173 | 0.9882 (85) | 0.0227 (88) | 0.4971 | 0.3514 | 336 | 0.6957 (46) | 0.6448 (290) | 0.6518 | 0.5577 |

`rec(good)` and `rec(ran)` carry their class support in brackets. A slice holding only one class gets `macro-F1 -`: averaging an F1 over a class with no support reports "the model failed" when the truth is "the question was not asked".


#### Ransomware test recall by family

| family | n | RF/WPC | MLP/WP | SVM-RBF/SW |
|---|---|---|---|---|
| avoslocker | 50 | 0.1200 | 1.0000 | 0.9200 |
| blackcat | 50 | 1.0000 | 0.9000 | 0.1400 |
| hive | 50 | 0.0400 | 0.1400 | 0.1400 |
| clop | 45 | 0.8444 | 0.8444 | 0.9111 |
| playcrypt | 43 | 1.0000 | 1.0000 | 0.5581 |
| bluesky | 34 | 1.0000 | 0.0588 | 0.8529 |
| blackbasta | 30 | 0.8667 | 0.7000 | 0.5333 |
| lorenz | 16 | 0.9375 | 1.0000 | 0.0000 |
| nightsky | 14 | 0.1429 | 0.1429 | 0.0000 |
| blackbyte | 13 | 0.0000 | 0.0000 | 0.0000 |
| karma | 13 | 1.0000 | 1.0000 | 1.0000 |
| bianlian | 11 | 0.3636 | 1.0000 | 0.0909 |
| quantum | 6 | 0.1667 | 0.1667 | 0.0000 |
| holyghost | 4 | 0.7500 | 1.0000 | 0.5000 |
| maui | 3 | 1.0000 | 1.0000 | 1.0000 |

## expB_cohort - Exp B restricted to the cohort (traditional features, Exp B's own goodware split membership)

| split | goodware | ransomware | total | groups |
|---|---|---|---|---|
| train | 1113 | 900 | 2013 | 294 |
| test | 127 | 362 | 489 | 56 |

| split | class | x86 / x64 |
|---|---|---|
| train | ransomware | 42 x64 (4.7%), 858 x86 (95.3%) |
| train | goodware | 913 x64 (82.0%), 200 x86 (18.0%) |
| test | ransomware | 72 x64 (19.9%), 290 x86 (80.1%) |
| test | goodware | 81 x64 (63.8%), 46 x86 (36.2%) |

Cohort filter: kept **2502 of 2604** files (dropped 102: family:thanos 1, tag:dotnet 1, tag:dup 10, tag:packed_other 90).

Goodware membership reused verbatim from `balanced_goodware`'s committed `splits.csv`: **1116 train / 131 test** selected out of a pool of 1343; 0 of the 1247 files that split names are not in this pool.

Exact-duplicate leakage: **1/127 goodware** and **1/362 ransomware** test files have a byte-identical opcode stream in train (1902/2502 streams are unique; 0 appear under both labels).


| model | tok | emb | mask | acc | bal-acc | macro-P | macro-R | macro-F1 | AUC | recall(ran) | FPR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RF | WPC | w2v | 0.0 | 0.6667 | 0.7237 | 0.6722 | 0.7237 | 0.6482 | 0.7920 | 0.6050 | 0.1575 |
| MLP | WP | w2v | 0.0 | 0.6094 | 0.6570 | 0.6210 | 0.6570 | 0.5901 | 0.7315 | 0.5580 | 0.2441 |
| SVM-RBF | SW | w2v | 0.0 | 0.5419 | 0.6599 | 0.6388 | 0.6599 | 0.5396 | 0.8205 | 0.4144 | 0.0945 |
| *floor* - majority class | - | - | - | 0.7403 | 0.5000 | - | - | 0.4254 | - | 1.0000 | 1.0000 |
| *floor* - x86 means ransomware | - | - | - | 0.7587 | 0.7195 | - | - | 0.7048 | - | 0.8011 | 0.3622 |

The two floor rows read no opcodes. `majority class` calls every test file ransomware; `x86 means ransomware` reads only the architecture recorded in the cohort CSV (336 of 489 test files are x86). A model row is evidence about opcodes only to the extent that it clears both.


#### Test metrics inside each architecture

| model / tok | x64 n | x64 rec(good) | x64 rec(ran) | x64 acc | x64 macro-F1 | x86 n | x86 rec(good) | x86 rec(ran) | x86 acc | x86 macro-F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| RF/WPC | 153 | 1.0000 (81) | 0.0556 (72) | 0.5556 | 0.4048 | 336 | 0.5652 (46) | 0.7414 (290) | 0.7173 | 0.5864 |
| MLP/WP | 153 | 0.9877 (81) | 0.1944 (72) | 0.6144 | 0.5262 | 336 | 0.3478 (46) | 0.6483 (290) | 0.6071 | 0.4676 |
| SVM-RBF/SW | 153 | 1.0000 (81) | 0.0000 (72) | 0.5294 | 0.3462 | 336 | 0.7391 (46) | 0.5172 (290) | 0.5476 | 0.4864 |

`rec(good)` and `rec(ran)` carry their class support in brackets. A slice holding only one class gets `macro-F1 -`: averaging an F1 over a class with no support reports "the model failed" when the truth is "the question was not asked".


#### Ransomware test recall by family

| family | n | RF/WPC | MLP/WP | SVM-RBF/SW |
|---|---|---|---|---|
| avoslocker | 50 | 0.1200 | 1.0000 | 0.9200 |
| blackcat | 50 | 1.0000 | 0.4000 | 0.0000 |
| hive | 50 | 0.1400 | 0.1400 | 0.1400 |
| clop | 45 | 0.8444 | 0.8222 | 0.7778 |
| playcrypt | 43 | 0.8372 | 0.6977 | 0.5581 |
| bluesky | 34 | 1.0000 | 0.0588 | 0.0000 |
| blackbasta | 30 | 0.8667 | 0.7333 | 0.6667 |
| lorenz | 16 | 0.0000 | 0.1250 | 0.0000 |
| karma | 13 | 1.0000 | 1.0000 | 1.0000 |
| bianlian | 11 | 0.1818 | 1.0000 | 0.0000 |
| blackbyte | 7 | 0.0000 | 0.0000 | 0.0000 |
| quantum | 6 | 0.1667 | 0.1667 | 0.1667 |
| holyghost | 4 | 0.7500 | 1.0000 | 0.2500 |
| maui | 3 | 1.0000 | 1.0000 | 1.0000 |

## expD - REVISED Mendeley ransomware + REVISED Goodware_Balanced, Exp B's goodware split membership

| split | goodware | ransomware | total | groups |
|---|---|---|---|---|
| train | 1112 | 904 | 2016 | 294 |
| test | 123 | 362 | 485 | 56 |

| split | class | x86 / x64 |
|---|---|---|
| train | ransomware | 42 x64 (4.6%), 862 x86 (95.4%) |
| train | goodware | 913 x64 (82.1%), 199 x86 (17.9%) |
| test | ransomware | 72 x64 (19.9%), 290 x86 (80.1%) |
| test | goodware | 81 x64 (65.9%), 42 x86 (34.1%) |

The feature folders are already cohort-only, and this run verified it: all 2501 files carry `in_cohort == 1`.

Goodware membership reused verbatim from `balanced_goodware`'s committed `splits.csv`: **1112 train / 123 test** selected out of a pool of 1324; 12 of the 1247 files that split names are not in this pool.

Without the cross-source dedup guard it would have been 1113 train / 127 test; the guard removed 5 files from the membership.

Cross-source dedup removed **13 of 1337** pool files (0 by binary sha256, 13 by opcode-stream hash): each is byte-identical, as an opcode stream, to a Mendeley goodware file. Examples: `everyday_CrystalDiskInfo_9.9.2_Machine_X64_inno_en-US.exe.txt`, `everyday_Greenshot_1.3.315_User_X86_inno_en-US.exe.txt`, `everyday_helper.exe.txt`, `everyday_R for Windows_4.6.1_Machine_X64_inno_en-US.exe.txt`.

Exact-duplicate leakage: **3/123 goodware** and **0/362 ransomware** test files have a byte-identical opcode stream in train (1845/2501 streams are unique; 0 appear under both labels).


| model | tok | emb | mask | acc | bal-acc | macro-P | macro-R | macro-F1 | AUC | recall(ran) | FPR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RF | WPC | w2v | 0.0 | 0.5588 | 0.6642 | 0.6338 | 0.6642 | 0.5530 | 0.8754 | 0.4503 | 0.1220 |
| MLP | WP | w2v | 0.0 | 0.6206 | 0.6627 | 0.6232 | 0.6627 | 0.5972 | 0.7526 | 0.5773 | 0.2520 |
| SVM-RBF | SW | w2v | 0.0 | 0.8763 | 0.8608 | 0.8312 | 0.8608 | 0.8439 | 0.9044 | 0.8923 | 0.1707 |
| *floor* - majority class | - | - | - | 0.7464 | 0.5000 | - | - | 0.4274 | - | 1.0000 | 1.0000 |
| *floor* - x86 means ransomware | - | - | - | 0.7649 | 0.7298 | - | - | 0.7113 | - | 0.8011 | 0.3415 |

The two floor rows read no opcodes. `majority class` calls every test file ransomware; `x86 means ransomware` reads only the architecture recorded in the cohort CSV (332 of 485 test files are x86). A model row is evidence about opcodes only to the extent that it clears both.


#### Test metrics inside each architecture

| model / tok | x64 n | x64 rec(good) | x64 rec(ran) | x64 acc | x64 macro-F1 | x86 n | x86 rec(good) | x86 rec(ran) | x86 acc | x86 macro-F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| RF/WPC | 153 | 0.9877 (81) | 0.0694 (72) | 0.5556 | 0.4150 | 332 | 0.6667 (42) | 0.5448 (290) | 0.5602 | 0.4806 |
| MLP/WP | 153 | 0.9383 (81) | 0.0000 (72) | 0.4967 | 0.3319 | 332 | 0.3810 (42) | 0.7207 (290) | 0.6777 | 0.5132 |
| SVM-RBF/SW | 153 | 0.9383 (81) | 0.8611 (72) | 0.9020 | 0.9011 | 332 | 0.6190 (42) | 0.9000 (290) | 0.8645 | 0.7284 |

`rec(good)` and `rec(ran)` carry their class support in brackets. A slice holding only one class gets `macro-F1 -`: averaging an F1 over a class with no support reports "the model failed" when the truth is "the question was not asked".


#### Ransomware test recall by family

| family | n | RF/WPC | MLP/WP | SVM-RBF/SW |
|---|---|---|---|---|
| avoslocker | 50 | 0.1200 | 1.0000 | 0.9800 |
| blackcat | 50 | 0.0000 | 0.0000 | 1.0000 |
| hive | 50 | 0.0000 | 0.1000 | 0.8400 |
| clop | 45 | 0.9778 | 0.9556 | 0.9111 |
| playcrypt | 43 | 1.0000 | 1.0000 | 0.9535 |
| bluesky | 34 | 1.0000 | 1.0000 | 1.0000 |
| blackbasta | 30 | 0.9000 | 0.9000 | 0.8667 |
| lorenz | 16 | 0.1250 | 0.1250 | 0.1250 |
| karma | 13 | 0.0000 | 0.0000 | 0.7692 |
| bianlian | 11 | 0.0000 | 0.0000 | 1.0000 |
| blackbyte | 7 | 0.0000 | 0.0000 | 1.0000 |
| quantum | 6 | 1.0000 | 0.1667 | 1.0000 |
| holyghost | 4 | 0.2500 | 0.2500 | 1.0000 |
| maui | 3 | 0.0000 | 1.0000 | 0.0000 |

## Six-experiment comparison

Matched rows: the same classifier, tokenizer, embedding and mask rate, with the same seeds, in every experiment. Cells are **macro-F1** on the test set. The last two rows are floors: two rules that read no opcodes at all, scored on the same test set. A model row that does not clear them has not been shown to use the code.

| model / tok / emb / mask | expA | expA_cohort | expC | expB | expB_cohort | expD |
|---|---|---|---|---|---|---|
| RF / WPC / w2v / 0.0 | 0.7496 | 0.7166 | 0.7660 | 0.6629 | 0.6482 | 0.5530 |
| MLP / WP / w2v / 0.0 | 0.8098 | 0.7848 | 0.9260 | 0.6756 | 0.5901 | 0.5972 |
| SVM-RBF / SW / w2v / 0.0 | 0.7398 | 0.7595 | 0.8779 | 0.5862 | 0.5396 | 0.8439 |
| *floor* - majority class | 0.4268 | 0.4244 | 0.4244 | 0.4268 | 0.4254 | 0.4274 |
| *floor* - x86 means ransomware | 0.4266 | 0.4335 | 0.4335 | 0.6799 | 0.7048 | 0.7113 |

Same rows, **balanced accuracy**:

| model / tok / emb / mask | expA | expA_cohort | expC | expB | expB_cohort | expD |
|---|---|---|---|---|---|---|
| RF / WPC / w2v / 0.0 | 0.8019 | 0.7808 | 0.8228 | 0.7378 | 0.7237 | 0.6642 |
| MLP / WP / w2v / 0.0 | 0.8662 | 0.8402 | 0.9239 | 0.7358 | 0.6570 | 0.6627 |
| SVM-RBF / SW / w2v / 0.0 | 0.8028 | 0.8162 | 0.8999 | 0.6901 | 0.6599 | 0.8608 |
| *floor* - majority class | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| *floor* - x86 means ransomware | 0.4330 | 0.4471 | 0.4471 | 0.7040 | 0.7195 | 0.7298 |

`majority class` predicts the larger test class for every file (ransomware in all 6), so its balanced accuracy is 0.5000 by construction and only its macro-F1 moves with the class ratio. `x86 means ransomware` reads the architecture column of the cohort CSV and nothing else; it is the confound of the section below, priced.

### One-variable deltas (macro-F1)

Each block moves exactly one thing. Positive means the second experiment scores higher.

| contrast | what moves | RF / WPC / w2v / 0.0 | MLP / WP / w2v / 0.0 | SVM-RBF / SW / w2v / 0.0 |
|---|---|---|---|---|
| expA -> expA_cohort | sample set only (Mendeley goodware) | -0.0330 | -0.0250 | +0.0197 |
| expB -> expB_cohort | sample set only (Goodware_Balanced) | -0.0147 | -0.0854 | -0.0466 |
| expA_cohort -> expC | feature form only (Mendeley goodware) | +0.0494 | +0.1412 | +0.1183 |
| expB_cohort -> expD | feature form only (Goodware_Balanced) | -0.0952 | +0.0070 | +0.3043 |
| expA -> expC | both at once (Mendeley goodware) | +0.0164 | +0.1162 | +0.1380 |
| expB -> expD | both at once (Goodware_Balanced) | -0.1099 | -0.0784 | +0.2577 |

### Per-architecture test metrics, all experiments

macro-F1 computed inside each architecture slice. A model that has learned "x64 means benign" cannot use that shortcut inside a slice where every sample has the same bitness, so a large drop from the pooled number is the signature of exactly that.

| experiment | model / tok | pooled macro-F1 | x86 n | x86 macro-F1 | x64 n | x64 macro-F1 | x86 rec(ran) | x64 rec(ran) | x86 rec(good) | x64 rec(good) |
|---|---|---|---|---|---|---|---|---|---|---|
| expA | RF/WPC | 0.7496 | 407 | 0.8860 | 100 | 0.2865 | 0.9345 | 0.2045 | 0.8376 | 0.9167 |
| expA | MLP/WP | 0.8098 | 407 | 0.9634 | 100 | 0.3066 | 0.9897 | 0.2159 | 0.9231 | 1.0000 |
| expA | SVM-RBF/SW | 0.7398 | 407 | 0.8724 | 100 | 0.2774 | 0.9000 | 0.1932 | 0.8718 | 0.9167 |
| expA_cohort | RF/WPC | 0.7166 | 407 | 0.8092 | 84 | 0.2967 | 0.8172 | 0.1944 | 0.8632 | 0.9167 |
| expA_cohort | MLP/WP | 0.7848 | 407 | 0.8925 | 84 | 0.3206 | 0.9138 | 0.2083 | 0.8974 | 1.0000 |
| expA_cohort | SVM-RBF/SW | 0.7595 | 407 | 0.8627 | 84 | 0.3080 | 0.8828 | 0.2083 | 0.8803 | 0.9167 |
| expC | RF/WPC | 0.7660 | 407 | 0.8652 | 84 | 0.3318 | 0.8862 | 0.2222 | 0.8803 | 1.0000 |
| expC | MLP/WP | 0.9260 | 407 | 0.9301 | 84 | 0.8897 | 0.9690 | 0.9444 | 0.8803 | 0.9167 |
| expC | SVM-RBF/SW | 0.8779 | 407 | 0.8695 | 84 | 0.9244 | 0.8793 | 0.9861 | 0.9060 | 0.8333 |
| expB | RF/WPC | 0.6629 | 336 | 0.6349 | 173 | 0.4198 | 0.8000 | 0.0909 | 0.5870 | 0.9882 |
| expB | MLP/WP | 0.6756 | 336 | 0.6041 | 173 | 0.5051 | 0.8276 | 0.1818 | 0.4348 | 1.0000 |
| expB | SVM-RBF/SW | 0.5862 | 336 | 0.5577 | 173 | 0.3514 | 0.6448 | 0.0227 | 0.6957 | 0.9882 |
| expB_cohort | RF/WPC | 0.6482 | 336 | 0.5864 | 153 | 0.4048 | 0.7414 | 0.0556 | 0.5652 | 1.0000 |
| expB_cohort | MLP/WP | 0.5901 | 336 | 0.4676 | 153 | 0.5262 | 0.6483 | 0.1944 | 0.3478 | 0.9877 |
| expB_cohort | SVM-RBF/SW | 0.5396 | 336 | 0.4864 | 153 | 0.3462 | 0.5172 | 0.0000 | 0.7391 | 1.0000 |
| expD | RF/WPC | 0.5530 | 332 | 0.4806 | 153 | 0.4150 | 0.5448 | 0.0694 | 0.6667 | 0.9877 |
| expD | MLP/WP | 0.5972 | 332 | 0.5132 | 153 | 0.3319 | 0.7207 | 0.0000 | 0.3810 | 0.9383 |
| expD | SVM-RBF/SW | 0.8439 | 332 | 0.7284 | 153 | 0.9011 | 0.9000 | 0.8611 | 0.6190 | 0.9383 |

### Ransomware test recall by family, RF/WPC row of each experiment

One model per experiment keeps the table readable; the per-experiment sections above carry all three. `n` differs between the traditional and revised columns because the cohort filter and the two extractors keep different files.

| family | expA n | expA recall | expA_cohort n | expA_cohort recall | expC n | expC recall | expB n | expB recall | expB_cohort n | expB_cohort recall | expD n | expD recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| avoslocker | 50 | 1.0000 | 50 | 1.0000 | 50 | 1.0000 | 50 | 0.1200 | 50 | 0.1200 | 50 | 0.1200 |
| blackcat | 50 | 0.7400 | 50 | 0.4200 | 50 | 0.5400 | 50 | 1.0000 | 50 | 1.0000 | 50 | 0.0000 |
| hive | 50 | 0.1400 | 50 | 0.1400 | 50 | 0.3400 | 50 | 0.0400 | 50 | 0.1400 | 50 | 0.0000 |
| clop | 45 | 0.8889 | 45 | 0.8889 | 45 | 0.8000 | 45 | 0.8444 | 45 | 0.8444 | 45 | 0.9778 |
| playcrypt | 43 | 1.0000 | 43 | 0.6977 | 43 | 1.0000 | 43 | 1.0000 | 43 | 0.8372 | 43 | 1.0000 |
| bluesky | 34 | 1.0000 | 34 | 0.8529 | 34 | 1.0000 | 34 | 1.0000 | 34 | 1.0000 | 34 | 1.0000 |
| blackbasta | 30 | 0.9000 | 30 | 0.9000 | 30 | 0.9333 | 30 | 0.8667 | 30 | 0.8667 | 30 | 0.9000 |
| lorenz | 16 | 1.0000 | 16 | 1.0000 | 16 | 1.0000 | 16 | 0.9375 | 16 | 0.0000 | 16 | 0.1250 |
| nightsky | 14 | 0.1429 | - | - | - | - | 14 | 0.1429 | - | - | - | - |
| blackbyte | 13 | 0.1538 | 7 | 0.0000 | 7 | 0.0000 | 13 | 0.0000 | 7 | 0.0000 | 7 | 0.0000 |
| karma | 13 | 1.0000 | 13 | 1.0000 | 13 | 1.0000 | 13 | 1.0000 | 13 | 1.0000 | 13 | 0.0000 |
| bianlian | 11 | 1.0000 | 11 | 1.0000 | 11 | 0.0000 | 11 | 0.3636 | 11 | 0.1818 | 11 | 0.0000 |
| quantum | 6 | 0.0000 | 6 | 0.0000 | 6 | 0.8333 | 6 | 0.1667 | 6 | 0.1667 | 6 | 1.0000 |
| holyghost | 4 | 1.0000 | 4 | 1.0000 | 4 | 0.2500 | 4 | 0.7500 | 4 | 0.7500 | 4 | 0.2500 |
| maui | 3 | 1.0000 | 3 | 1.0000 | 3 | 1.0000 | 3 | 1.0000 | 3 | 1.0000 | 3 | 0.0000 |

Row order is by family size. A family present in one column and absent from another was removed entirely by the cohort filter (`thanos`) or has no surviving member in that feature tree.

## Reading the six

### What the cohort filter does

- `expA` -> `expA_cohort`: macro-F1 moves by **-0.0128** on average across 3 matched rows (range -0.0330 to +0.0197).
- `expB` -> `expB_cohort`: macro-F1 moves by **-0.0489** on average across 3 matched rows (range -0.0854 to -0.0147).

The filter removes samples, not information: 104 files leave Exp A (88 entropy-packed, 15 duplicate rows, 1 `thanos`) and 102 leave Exp B. Those are overwhelmingly the packed ransomware, whose opcode stream is the packer's stub rather than the payload's. Removing them should make the task HARDER in the sense that a trivially separable group of positives is gone, and EASIER in the sense that the remaining positives are the ones a model can actually learn something about; the sign of the measured move is in the table above rather than in this sentence.

The filter also changes the architecture mix: it removes proportionally more x64 ransomware from the test set than x86, and it removes the `thanos` family outright.

### What the traditional -> revised change does

- `expA_cohort` -> `expC`: macro-F1 moves by **+0.1030** on average across 3 matched rows (range +0.0494 to +0.1412).
- `expB_cohort` -> `expD`: macro-F1 moves by **+0.0720** on average across 3 matched rows (range -0.0952 to +0.3043).

Two things move at once inside this one contrast, and they should not be conflated:

1. **Operands are gone.** A line is `mov`, not `mov rbp, rsp`. That removes the most direct architecture tell (`rbp`/`r8`-`r15`/rip-relative) and collapses the distinct-line vocabulary from 36,802 to 466, so the feature space is far smaller and far less able to memorise a specific binary.
2. **The sweep is different.** `extract_unified.py` skips undecodable bytes and keeps going instead of stopping at the first one, and it is uncapped. Under the pipeline's 5,000-instruction cap that mostly means the 5,000 instructions come from further into the binary, and from code the linear sweep could not reach at all - which is why the revised mnemonic vocabulary contains hundreds of AVX/AVX-512 mnemonics the traditional one never saw.

A consequence worth stating on its own: under mnemonic-only features, **13 Goodware_Balanced files become byte-identical to a Mendeley goodware file** that the full-instruction form kept distinct. They are Inno Setup and NSIS installer stubs - re-measured 14 September 2026, **six** distinct streams cover all 13, and each of the six is shared with between one and five Mendeley goodware files - and the cross-source dedup guard removes them, which is why Exp D's goodware counts sit a little under Exp B_cohort's. The traditional form removed zero. Dropping operands makes the installer-stub problem from audit §2.1 strictly more visible, not less.

Exp D's test goodware count follows from that in two steps, and both are readable off other experiments: of Exp B's 131 test goodware files, **4** have no revised feature file at all - they are exactly the four the cohort filter drops in Exp B_cohort, which is why that experiment has 127 - and **4** more are removed by the cross-source dedup guard, leaving 123.

### Both at once

- `expA` -> `expC`: macro-F1 moves by **+0.0902** on average across 3 matched rows (range +0.0164 to +0.1380).
- `expB` -> `expD`: macro-F1 moves by **+0.0231** on average across 3 matched rows (range -0.1099 to +0.2577).

This is the comparison that was available before the two cohort experiments existed, and it is the one that cannot be interpreted: the sample set and the feature form move together. It is kept in the table only so the decomposition above can be checked against it.


### Architecture: the confound, now measured rather than assumed

Every count below comes from `cohort_mendeley.csv` / `cohort_balanced.csv`, which record the architecture of every input binary of every set. The previous version of this document had to leave `good_test` and the whole ransomware side as `unknown`, because the feature files carry no architecture and those binaries are not on this machine. That gap is closed.

| experiment | train ransomware | test ransomware | train goodware | test goodware |
|---|---|---|---|---|
| expA | 6 unknown (0.6%), 42 x64 (4.3%), 927 x86 (95.1%) | 4 unknown (1.0%), 88 x64 (23.0%), 290 x86 (75.9%) | 3 unknown (0.3%), 483 x64 (43.3%), 630 x86 (56.5%) | 2 unknown (1.5%), 12 x64 (9.2%), 117 x86 (89.3%) |
| expA_cohort | 42 x64 (4.7%), 858 x86 (95.3%) | 72 x64 (19.9%), 290 x86 (80.1%) | 482 x64 (43.5%), 627 x86 (56.5%) | 12 x64 (9.3%), 117 x86 (90.7%) |
| expC | 42 x64 (4.6%), 862 x86 (95.4%) | 72 x64 (19.9%), 290 x86 (80.1%) | 482 x64 (43.3%), 632 x86 (56.7%) | 12 x64 (9.3%), 117 x86 (90.7%) |
| expB | 6 unknown (0.6%), 42 x64 (4.3%), 927 x86 (95.1%) | 4 unknown (1.0%), 88 x64 (23.0%), 290 x86 (75.9%) | 915 x64 (82.0%), 201 x86 (18.0%) | 85 x64 (64.9%), 46 x86 (35.1%) |
| expB_cohort | 42 x64 (4.7%), 858 x86 (95.3%) | 72 x64 (19.9%), 290 x86 (80.1%) | 913 x64 (82.0%), 200 x86 (18.0%) | 81 x64 (63.8%), 46 x86 (36.2%) |
| expD | 42 x64 (4.6%), 862 x86 (95.4%) | 72 x64 (19.9%), 290 x86 (80.1%) | 913 x64 (82.1%), 199 x86 (17.9%) | 81 x64 (65.9%), 42 x86 (34.1%) |

**State it plainly.** The ransomware side is overwhelmingly x86 (~96% of train, ~80% of test once the cohort filter has removed the packed samples). Mendeley goodware is ~57% x86 in train but ~91% x86 in test - the two goodware splits do not even match each other. Goodware_Balanced is ~18-23% x86. So in every experiment here, "x64" is evidence for benign and "x86" is evidence for ransomware before a single opcode is read - in TRAINING. The imbalance is different on each side of each split, and that difference decides whether the shortcut still pays on the test set:

- **expA**: the architecture-only rule scores macro-F1 0.4266 / balanced accuracy 0.4330 on this test set; every model row clears it.
- **expA_cohort**: the architecture-only rule scores macro-F1 0.4335 / balanced accuracy 0.4471 on this test set; every model row clears it.
- **expC**: the architecture-only rule scores macro-F1 0.4335 / balanced accuracy 0.4471 on this test set; every model row clears it.
- **expB**: the architecture-only rule scores macro-F1 0.6799 / balanced accuracy 0.7040 on this test set; RF/WPC 0.6629, MLP/WP 0.6756, SVM-RBF/SW 0.5862 do not clear it.
- **expB_cohort**: the architecture-only rule scores macro-F1 0.7048 / balanced accuracy 0.7195 on this test set; RF/WPC 0.6482, MLP/WP 0.5901, SVM-RBF/SW 0.5396 do not clear it.
- **expD**: the architecture-only rule scores macro-F1 0.7113 / balanced accuracy 0.7298 on this test set; RF/WPC 0.5530, MLP/WP 0.5972 do not clear it.

The rule is not equally strong everywhere, and the reason is in the table above rather than in the models: it is worth macro-F1 0.7113 in `expD`, where the goodware half is Goodware_Balanced and mostly x64, and only 0.4266 in `expA`, where Mendeley `good_test` is ~91% x86 and so looks, to a bitness rule, exactly like the ransomware. A Mendeley-goodware experiment therefore cannot be cleared of the confound by pointing at this floor: the shortcut is in its TRAINING mix (~57% x86 goodware against ~95% x86 ransomware) and it MISFIRES on its own test set, which is what the collapsed x64 slices in the per-architecture table are.

Bitness is not hidden from the model. In the TRADITIONAL feature form it is written all over the operands (`rbp`, `r8`-`r15`, rip-relative addressing, the register calling convention) and survives `normalize_instruction` into the token stream. The REVISED form drops operands for exactly this reason - a mnemonic line is `mov`, not `mov rbp, rsp` - but it does not remove the signal entirely: the x64-only instruction set (`vpxor`, `rorx`, `cmpxchg16b`, the AVX/AVX-512 mnemonics that dominate the candidate-only vocabulary in the schema check) is still visible as a mnemonic.

The tables above are the test: score each architecture slice on its own, where bitness separates nothing, and see how much is left.

Across the 18 model/experiment pairs where both slices hold both classes, within-x64 macro-F1 runs from -0.6568 to +0.1728 relative to within-x86 (largest gap: expD SVM-RBF/SW, 0.9011 on x64 against 0.7284 on x86).
On the goodware class alone the x64-minus-x86 recall gap runs from -0.0726 to +0.6398.
Splitting the test set by bitness costs up to 0.5032 macro-F1 against the pooled score, which is the size of the shortcut the pooled number was buying.


### Mnemonic-only input: what the tokenizers actually do

The revised feature files hold one mnemonic per line. Measured on the Exp C training corpus under the pipeline's own 5,000-instruction cap:

| corpus | distinct normalized lines | example lines |
|---|---|---|
| traditional (Exp A train) | 36,802 | `push ebp`, `mov ebp esp`, `add byte ptr [eax] al`, `call <HEX>` |
| revised (Exp C train) | 466 | `mov`, `push`, `add`, `call`, `int3` |

That changes what each tokenizer means, and the three of them stop being three different things:

* **SW** - one token per line. Unchanged in kind; the token is now a bare mnemonic.
* **WP** - adjacent-line bigrams, `mov_push`. Still a real second view of the stream, and on mnemonic-only input it is the only one that carries any order information.
* **WPC** - WordPiece over the `<SEP>`-joined line text. **On mnemonic-only input it is whole-word tokenization to within a rounding error.** The cached tokenizer that produced the committed Exp C numbers (`results/tokenizers/expC_WPC.json`) holds 941 entries against the 1,000 asked for - 667 word-initial and 274 `##` continuation pieces - so the trainer stopped short of the cap. The corpus holds 478 distinct mnemonics, and **466 of them encode as a single whole-word token**. The 12 that fragment (`cvtpd2ps` -> `cvt ##pd ##2ps`, `xacquire` -> six pieces) are *exactly* the 12 that occur only in the test split, which the tokenizer - correctly fit on train rows only - has never seen as a word. Between them they account for **176 of 11,952,897** mnemonic occurrences, 0.0015%. There are no `<UNK>` tokens at all: the `##` machinery is doing nothing but keeping 12 unseen-at-training mnemonics out of `<UNK>`.

So in Exp C and Exp D, **RF/WPC and any SW-based row are reading all but the same token stream** - 176 occurrences in 11.95 million differ - and the subword tokenizer is contributing next to nothing. That is worth knowing before reading a WPC row in the revised columns as evidence about subword tokenization: it is evidence about whole-word mnemonic tokenization wearing a WordPiece label.

One consequence, tested rather than argued. The WPC nondeterminism that forced the tokenizer cache (docs/tokenization_audit.md §1.10) comes from the tie-break at the vocabulary size cutoff, and on mnemonic-only input the trainer stops short of that cutoff. It is still not bit-deterministic: a vocabulary trained fresh in another process differs from the cached one in 12 of its 941 entries and in the ids of 102 more. But every one of those differences is an unused fragment - both vocabularies encode all 478 distinct mnemonics to the same token strings - so Exp C, re-run twice with an empty tokenizer cache deleted between the runs, produced a `predictions.csv` byte-identical to the committed one both times. The cache is still used, and is still required for the four traditional-feature experiments, where the cutoff is reached.

The whole-corpus mnemonic vocabularies, from `check_schema.py` (which reads every line, not the first 5,000): revised Mendeley good_train 605, good_test 514, mal_train 698, mal_test 611; revised Goodware_Balanced 1,292. All five directories pass every hard schema check - one instruction per line, no address prefix, no tabs, no comments, no directives, no empty files. The mnemonic-set difference against the traditional reference is large by design (Jaccard 0.47-0.72) and is reported, not treated as a failure: the revised extractor sweeps past undecodable bytes instead of stopping at them, so it reaches AVX/AVX-512 code the linear sweep never got to.


### Deviation from the plan's split procedure

The plan asked for "the same stratified 80/20 split procedure" in both experiments. Three departures, all deliberate:

1. **The ransomware split is the Mendeley release's own family-disjoint split, not a random stratified 80/20.** `mal_train` holds 25 families and `mal_test` 15 entirely different ones, with zero overlap - a genuine unseen-family generalisation test. A random stratified re-split destroys it: `dharma` alone contributes 45 byte-identical samples, `phobos` 37, `lockbit` 35, so copies of one stream land on both sides and the task collapses into near-duplicate retrieval. Measured on the shipped split, ransomware test leakage is 1/382; a random re-split would take it far higher. `config.yaml` sets `split.ransomware: preserve_mendeley`, and the identical ransomware split is reused verbatim in every experiment, so it contributes nothing to any difference between them. The cohort filter removes whole families only where every member was packed: `thanos` is gone from train and `night sky` from test, leaving 24 and 14 families.
2. **Exp A's goodware split is also the release's own**, not a re-split. It has no group discipline, which is precisely what §2.1 of the audit measures: 64/131 of `good_test` is a verbatim copy of a training file. It is kept as shipped so Exp A remains the published baseline to compare against. Exp B's goodware side *is* split the way the plan intends - grouped by `entry_id` so no source project straddles train and test, stratified by bucket so `everyday`/`hard_negative`/`system` keep their proportions on both sides, with a fixed seed - and its counts are matched to Exp A's exactly (`split.match_counts_to: expA`).
3. **Exp B_cohort and Exp D do not re-split at all.** They take Exp B's committed goodware membership out of `results/expB/splits.csv` and intersect it with the pool in front of them (`data.reuse_split`). Re-running `group_split` on a pool the cohort filter has changed would move whole source projects across the train/test line, and the result would then differ from Exp B for two reasons at once - which is the one thing these four experiments exist to avoid.

Net effect: A is the baseline on its own terms, B is the same ransomware task with a harder, properly grouped goodware half, and the four new experiments move one variable at a time off those two.

---

# Tuned

Everything above is the committed six-experiment sweep: one tiny `GridSearchCV` (`cv=2`, four candidate settings per classifier) on top of a fixed representation - the first 5,000 lines of each file, Word2Vec 100d/window 30/5 epochs, mean-pooled. This section is what a real search finds on the same two revised-feature experiments.

**Protocol, stated before any number below.** Every choice - sequence budget, positional sampler, tokenizer, vocabulary size, n-gram order, embedding, pooling, classifier and its hyperparameters, and the architecture reweighting - was made by grouped cross-validation on the TRAIN SPLIT ONLY, with the ransomware family / goodware project as the group so whole families are held out, and ranked by out-of-fold macro-F1. The test set was scored once per reported configuration, after the ranking was fixed. `results/expC/` and `results/expD/` are untouched. The search is in `llm_features_pipeline/tune.py`; every configuration it tried, with its CV scores, is in `results/exp*_tuned/cv_search.csv`. Only `cv_rank == 1` is the selected result; the four rows below it were scored on test afterwards, for the CV-to-test gap table alone, and reading the best of the five would be selection on the test set. The protocol audit - what the search was allowed to see, and the one operation that touches a test row and why it is not a leak - is [docs/tokenization_audit.md](../docs/tokenization_audit.md) §6.

## Before and after

| experiment | row | sequence budget | representation | classifier | macro-F1 | bal-acc | AUC | recall(ran) | FPR |
|---|---|---|---|---|---|---|---|---|---|
| expC | committed best (MLP/WP) | 5,000 head | w2v 100d mean | MLP | 0.9260 | 0.9239 | 0.9520 | 0.9641 | 0.1163 |
| expC | committed SVM-RBF/SW | 5,000 head | w2v 100d mean | SVM-RBF | 0.8779 | 0.8999 | 0.9594 | 0.9006 | 0.1008 |
| expC | committed RF/WPC | 5,000 head | w2v 100d mean | RF | 0.7660 | 0.8228 | 0.9482 | 0.7541 | 0.1085 |
| expC | **tuned (CV rank 1)** | 50,000 head | SW TF-IDF 1-3, 87269 features | LogReg | **0.8576** | 0.8214 | 0.9759 | 0.9917 | 0.3488 |
| expC | *calibration* - mnemonic_tfidf_1_3+LinearSVC | 30,000 head | mnemonic TF-IDF 1-3 | LinearSVC | 0.9603 | 0.9568 | 0.9778 | 0.9834 | 0.0698 |
| expC | *calibration* - mnemonic_tfidf_1_3+LogReg | 30,000 head | mnemonic TF-IDF 1-3 | LogReg | 0.9680 | 0.9610 | 0.9871 | 0.9917 | 0.0698 |
| expC | *floor* - majority class | - | - | - | 0.4244 | 0.5000 | - | 1.0000 | 1.0000 |
| expC | *floor* - x86 means ransomware | - | - | - | 0.4335 | 0.4471 | - | 0.8011 | 0.9070 |
| expD | committed best (SVM-RBF/SW) | 5,000 head | w2v 100d mean | SVM-RBF | 0.8439 | 0.8608 | 0.9044 | 0.8923 | 0.1707 |
| expD | committed MLP/WP | 5,000 head | w2v 100d mean | MLP | 0.5972 | 0.6627 | 0.7526 | 0.5773 | 0.2520 |
| expD | committed RF/WPC | 5,000 head | w2v 100d mean | RF | 0.5530 | 0.6642 | 0.8754 | 0.4503 | 0.1220 |
| expD | **tuned (CV rank 1)** | 50,000 strided | SW TF-IDF 1-1, 859 features | LinearSVC | **0.6830** | 0.7449 | 0.8783 | 0.6768 | 0.1870 |
| expD | *calibration* - mnemonic_tfidf_1_3+LinearSVC | 30,000 head | mnemonic TF-IDF 1-3 | LinearSVC | 0.8023 | 0.8508 | 0.9285 | 0.8039 | 0.1024 |
| expD | *calibration* - mnemonic_tfidf_1_3+LogReg | 30,000 head | mnemonic TF-IDF 1-3 | LogReg | 0.7956 | 0.8441 | 0.9226 | 0.7983 | 0.1102 |
| expD | *floor* - majority class | - | - | - | 0.4274 | 0.5000 | - | 1.0000 | 1.0000 |
| expD | *floor* - x86 means ransomware | - | - | - | 0.7113 | 0.7298 | - | 0.8011 | 0.3415 |

The *calibration* rows are `rules_pipeline`'s audited mnemonic TF-IDF 1-3 + linear model on the SAME cohort rows (`results/rules/summary.md` §5). They read raw mnemonics, not the tokenizer's output, and they are the number this pipeline has to reach before subword tokenization can be said to have earned anything. On the Mendeley split that headline 0.968 falls to **0.955** once the 62-of-129 duplicated test goodware files are removed (§5.4), which is the figure to compare against.

### Did it help?

* **expC: -0.0685 macro-F1** against the best committed row (MLP/WP, 0.9260 -> 0.8576), so the search did not beat the fixed configuration it was meant to improve on; clears 2 of 2 floors; below the mnemonic TF-IDF calibration row (0.9680); ransomware recall 0.9917, goodware recall 0.6512.
  A lower-ranked configuration reaches 0.9680 on test, but it was not selected and is not the result - picking it after the fact is choosing on the test set. The gap (0.1104) is the price of the protocol, and it is reported rather than spent.
* **expD: -0.1609 macro-F1** against the best committed row (SVM-RBF/SW, 0.8439 -> 0.6830), so the search did not beat the fixed configuration it was meant to improve on; clears 1 of 2 floors; below the mnemonic TF-IDF calibration row (0.8023); ransomware recall 0.6768, goodware recall 0.8130.
  A lower-ranked configuration reaches 0.6834 on test, but it was not selected and is not the result - picking it after the fact is choosing on the test set. The gap (0.0004) is the price of the protocol, and it is reported rather than spent.

## expC_tuned - top five by CV, with their test scores

787 configurations were cross-validated; 5 were scored on the test set. CV is `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) on the train split`, grouped by ransomware family / goodware source project (splits.csv `group`).

| CV rank | track | budget | sampler | features | classifier | CV macro-F1 | test macro-F1 | CV - test | test bal-acc | test AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 **(selected)** | tfidf | 50,000 | head | SW TF-IDF 1-3, 87269 features | LogReg {'C': '0.1', 'class_weight': 'balanced'} | 0.9474 | 0.8576 | +0.0898 | 0.8214 | 0.9759 |
| 2 | tfidf | 20,000 | head | SW TF-IDF 1-3, 65244 features | LogReg {'C': '1.0', 'class_weight': 'balanced'} | 0.9436 | 0.9540 | -0.0104 | 0.9416 | 0.9873 |
| 3 | tfidf | 20,000 | head | SW TF-IDF 1-3, 65244 features | LinearSVC {'C': '0.1', 'class_weight': 'balanced'} | 0.9420 | 0.9540 | -0.0120 | 0.9416 | 0.9894 |
| 4 | tfidf | 50,000 | head | SW TF-IDF 1-3, 87269 features | LinearSVC {'C': '0.01', 'class_weight': 'balanced'} | 0.9418 | 0.8576 | +0.0842 | 0.8214 | 0.9753 |
| 5 | tfidf | 20,000 | head | SW TF-IDF 1-3, 65244 features | LogReg {'C': '1.0', 'class_weight': 'None'} | 0.9395 | 0.9680 | -0.0285 | 0.9610 | 0.9873 |

CV minus test runs from -0.0285 to +0.0898. The five differ by 0.0079 in CV and by 0.1104 on test, so the CV ordering inside the top five is not informative at this resolution - which is the honest reading of a 787-configuration search and the reason all five are printed rather than only the winner.

#### What the operating point is worth

The reported rows use the untuned decision cut, the same rule the committed runs use. Two fitted alternatives were scored for every configuration - the threshold that maximises out-of-fold balanced accuracy, and the one that maximises out-of-fold macro-F1 - each chosen nested (fold k's cut comes from the other folds' out-of-fold scores).

| CV rank | CV: untuned | CV: bal-acc thr | CV: macro-F1 thr | test: untuned | test: bal-acc thr | test: macro-F1 thr | test recall(good), untuned -> macro-F1 thr |
|---|---|---|---|---|---|---|---|
| 1 | 0.9474 | 0.9439 | 0.9464 | 0.8576 | 0.8576 | 0.8576 | 0.6512 -> 0.6512 |
| 2 | 0.9436 | 0.9369 | 0.9457 | 0.9540 | 0.9483 | 0.9483 | 0.8915 -> 0.8760 |
| 3 | 0.9420 | 0.9399 | 0.9482 | 0.9540 | 0.9512 | 0.9512 | 0.8915 -> 0.8837 |
| 4 | 0.9418 | 0.9489 | 0.9489 | 0.8576 | 0.8506 | 0.8506 | 0.6512 -> 0.6357 |
| 5 | 0.9395 | 0.9379 | 0.9462 | 0.9680 | 0.9483 | 0.9483 | 0.9302 -> 0.8760 |

Over these five rows, fitting the cut on out-of-fold scores moves CV macro-F1 by +0.0042 on average and test macro-F1 by -0.0070, with test goodware recall moving -0.0186. The sign of that second number is a property of the split, not of the fitting: the train half is 45% ransomware and the test half 74%, macro-F1's optimal threshold moves with the class prior, and a cut fitted on training folds carries the TRAINING prior onto a test set that does not have it. Goodware recall is where it shows, and it is why the reported rows leave the cut alone.

#### expC: what moved the number, axis by axis

| axis | setting | configs | best CV macro-F1 | median |
|---|---|---|---|---|
| sequence budget | `50000` | 84 | 0.9474 | 0.8939 |
|  | `20000` | 254 | 0.9436 | 0.8660 |
|  | `5000` | 449 | 0.9168 | 0.8333 |
| positional sampler | `head` | 466 | 0.9474 | 0.8394 |
|  | `strided` | 321 | 0.9293 | 0.8603 |
| n-gram order | `1-3` | 84 | 0.9474 | 0.9032 |
|  | `1-2` | 108 | 0.9312 | 0.8935 |
|  | `1-1` | 595 | 0.9271 | 0.8419 |
| tokenizer | `SW` | 759 | 0.9474 | 0.8511 |
|  | `WP` | 4 | 0.9267 | 0.9060 |
|  | `WPC` | 16 | 0.9057 | 0.8688 |
|  | `BPE` | 8 | 0.9052 | 0.8672 |
| subword vocabulary | `500` | 4 | 0.9057 | 0.8688 |
|  | `1000` | 8 | 0.9052 | 0.8672 |
|  | `2000` | 4 | 0.9052 | 0.8672 |
|  | `4000` | 8 | 0.9052 | 0.8672 |
| representation | `tfidf` | 280 | 0.9474 | 0.8905 |
|  | `w2v` | 507 | 0.9271 | 0.8405 |
| pooling (w2v rows only) | `mean_max` | 169 | 0.9271 | 0.8223 |
|  | `tfidf_mean` | 169 | 0.9119 | 0.8483 |
|  | `mean` | 169 | 0.9102 | 0.8428 |
| classifier | `LogReg` | 179 | 0.9474 | 0.8847 |
|  | `LinearSVC` | 140 | 0.9420 | 0.8902 |
|  | `MLP` | 156 | 0.9271 | 0.8490 |
|  | `SVM-RBF` | 195 | 0.9097 | 0.8429 |
|  | `RF` | 117 | 0.8671 | 0.8241 |
| sample weighting | `none` | 673 | 0.9474 | 0.8540 |
|  | `class_arch` | 114 | 0.9297 | 0.8445 |

## expD_tuned - top five by CV, with their test scores

787 configurations were cross-validated; 5 were scored on the test set. CV is `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) on the train split`, grouped by ransomware family / goodware source project (splits.csv `group`).

| CV rank | track | budget | sampler | features | classifier | CV macro-F1 | test macro-F1 | CV - test | test bal-acc | test AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 **(selected)** | tfidf | 50,000 | strided | SW TF-IDF 1-1, 859 features | LinearSVC {'C': '0.1', 'class_weight': 'balanced'} | 0.9237 | 0.6830 | +0.2407 | 0.7449 | 0.8783 |
| 2 | w2v/mean_max | 5,000 | strided | w2v dim=300,window=10,epochs=20,min_count=1, mean_max pooling | SVM-RBF {'C': '10.0', 'class_weight': 'balanced', 'gamma': 'scale'} | 0.9232 | 0.6599 | +0.2633 | 0.7336 | 0.8371 |
| 3 | w2v/mean_max | 5,000 | strided | w2v dim=300,window=10,epochs=20,min_count=1, mean_max pooling | MLP {'alpha': '0.001', 'early_stopping': 'True', 'hidden_layer_sizes': '(256, 128)', 'learning_rate_init': '0.001'} | 0.9222 | 0.6495 | +0.2727 | 0.7134 | 0.8174 |
| 4 | tfidf | 50,000 | head | SW TF-IDF 1-3, 105872 features | LinearSVC {'C': '0.1', 'class_weight': 'None'} | 0.9206 | 0.6834 | +0.2372 | 0.7556 | 0.8962 |
| 5 | tfidf | 50,000 | head | SW TF-IDF 1-3, 105872 features | LogReg {'C': '1.0', 'class_weight': 'None'} | 0.9202 | 0.6742 | +0.2459 | 0.7487 | 0.8843 |

CV minus test runs from +0.2372 to +0.2727. The five differ by 0.0036 in CV and by 0.0339 on test, so the CV ordering inside the top five is not informative at this resolution - which is the honest reading of a 787-configuration search and the reason all five are printed rather than only the winner.

#### What the operating point is worth

The reported rows use the untuned decision cut, the same rule the committed runs use. Two fitted alternatives were scored for every configuration - the threshold that maximises out-of-fold balanced accuracy, and the one that maximises out-of-fold macro-F1 - each chosen nested (fold k's cut comes from the other folds' out-of-fold scores).

| CV rank | CV: untuned | CV: bal-acc thr | CV: macro-F1 thr | test: untuned | test: bal-acc thr | test: macro-F1 thr | test recall(good), untuned -> macro-F1 thr |
|---|---|---|---|---|---|---|---|
| 1 | 0.9237 | 0.9217 | 0.9217 | 0.6830 | 0.6848 | 0.6848 | 0.8130 -> 0.8130 |
| 2 | 0.9232 | 0.9294 | 0.9294 | 0.6599 | 0.6720 | 0.6720 | 0.8374 -> 0.8130 |
| 3 | 0.9222 | 0.9193 | 0.9193 | 0.6495 | 0.6514 | 0.6514 | 0.7886 -> 0.7724 |
| 4 | 0.9206 | 0.8995 | 0.9015 | 0.6834 | 0.7677 | 0.7706 | 0.8537 -> 0.8130 |
| 5 | 0.9202 | 0.8976 | 0.8955 | 0.6742 | 0.7648 | 0.7648 | 0.8537 -> 0.7967 |

Over these five rows, fitting the cut on out-of-fold scores moves CV macro-F1 by -0.0085 on average and test macro-F1 by +0.0387, with test goodware recall moving -0.0276. The sign of that second number is a property of the split, not of the fitting: the train half is 45% ransomware and the test half 74%, macro-F1's optimal threshold moves with the class prior, and a cut fitted on training folds carries the TRAINING prior onto a test set that does not have it. Goodware recall is where it shows, and it is why the reported rows leave the cut alone.

#### expD: what moved the number, axis by axis

| axis | setting | configs | best CV macro-F1 | median |
|---|---|---|---|---|
| sequence budget | `50000` | 84 | 0.9237 | 0.8847 |
|  | `5000` | 449 | 0.9232 | 0.8358 |
|  | `20000` | 254 | 0.9174 | 0.8672 |
| positional sampler | `strided` | 321 | 0.9237 | 0.8740 |
|  | `head` | 466 | 0.9206 | 0.8348 |
| n-gram order | `1-1` | 595 | 0.9237 | 0.8387 |
|  | `1-3` | 84 | 0.9206 | 0.8847 |
|  | `1-2` | 108 | 0.9164 | 0.8804 |
| tokenizer | `SW` | 759 | 0.9237 | 0.8498 |
|  | `WP` | 4 | 0.9081 | 0.9000 |
|  | `WPC` | 16 | 0.8949 | 0.8849 |
|  | `BPE` | 8 | 0.8934 | 0.8841 |
| subword vocabulary | `500` | 4 | 0.8949 | 0.8867 |
|  | `2000` | 4 | 0.8934 | 0.8841 |
|  | `4000` | 8 | 0.8934 | 0.8841 |
|  | `1000` | 8 | 0.8934 | 0.8834 |
| representation | `tfidf` | 280 | 0.9237 | 0.8794 |
|  | `w2v` | 507 | 0.9232 | 0.8319 |
| pooling (w2v rows only) | `mean_max` | 169 | 0.9232 | 0.8324 |
|  | `mean` | 169 | 0.9152 | 0.8291 |
|  | `tfidf_mean` | 169 | 0.9008 | 0.8325 |
| classifier | `LinearSVC` | 140 | 0.9237 | 0.8790 |
|  | `SVM-RBF` | 195 | 0.9232 | 0.8432 |
|  | `MLP` | 156 | 0.9222 | 0.8325 |
|  | `LogReg` | 179 | 0.9202 | 0.8769 |
|  | `RF` | 117 | 0.8566 | 0.7998 |
| sample weighting | `none` | 673 | 0.9237 | 0.8630 |
|  | `class_arch` | 114 | 0.8600 | 0.7751 |

## Tuned rows inside each architecture

The pooled score of a tuned row means what the committed rows' pooled scores meant: it is partly bitness. Same treatment - score each slice on its own.

| experiment | row | pooled macro-F1 | x86 n | x86 macro-F1 | x64 n | x64 macro-F1 | x86 rec(ran) | x64 rec(ran) | x86 rec(good) | x64 rec(good) |
|---|---|---|---|---|---|---|---|---|---|---|
| expC | committed MLP/WP | 0.9260 | 407 | 0.9301 | 84 | 0.8897 | 0.9690 | 0.9444 | 0.8803 | 0.9167 |
| expC | **tuned rank 1** | 0.8576 | 407 | 0.8450 | 84 | 0.9338 | 1.0000 | 0.9583 | 0.6154 | 1.0000 |
| expD | committed SVM-RBF/SW | 0.8439 | 332 | 0.7284 | 153 | 0.9011 | 0.9000 | 0.8611 | 0.6190 | 0.9383 |
| expD | **tuned rank 1** | 0.6830 | 332 | 0.5878 | 153 | 0.5077 | 0.8034 | 0.1667 | 0.4524 | 1.0000 |

### expC: ransomware test recall by family, before and after

| family | n | committed best | tuned rank 1 | delta |
|---|---|---|---|---|
| avoslocker | 50 | 1.0000 | 1.0000 | +0.0000 |
| blackcat | 50 | 1.0000 | 1.0000 | +0.0000 |
| hive | 50 | 1.0000 | 1.0000 | +0.0000 |
| clop | 45 | 0.8000 | 1.0000 | +0.2000 |
| playcrypt | 43 | 1.0000 | 1.0000 | +0.0000 |
| bluesky | 34 | 1.0000 | 1.0000 | +0.0000 |
| blackbasta | 30 | 0.9333 | 0.9000 | -0.0333 |
| lorenz | 16 | 1.0000 | 1.0000 | +0.0000 |
| karma | 13 | 1.0000 | 1.0000 | +0.0000 |
| bianlian | 11 | 1.0000 | 1.0000 | +0.0000 |
| blackbyte | 7 | 0.7143 | 1.0000 | +0.2857 |
| quantum | 6 | 1.0000 | 1.0000 | +0.0000 |
| holyghost | 4 | 1.0000 | 1.0000 | +0.0000 |
| maui | 3 | 1.0000 | 1.0000 | +0.0000 |

### expD: ransomware test recall by family, before and after

| family | n | committed best | tuned rank 1 | delta |
|---|---|---|---|---|
| avoslocker | 50 | 0.9800 | 1.0000 | +0.0200 |
| blackcat | 50 | 1.0000 | 0.0000 | -1.0000 |
| hive | 50 | 0.8400 | 0.0200 | -0.8200 |
| clop | 45 | 0.9111 | 1.0000 | +0.0889 |
| playcrypt | 43 | 0.9535 | 1.0000 | +0.0465 |
| bluesky | 34 | 1.0000 | 1.0000 | +0.0000 |
| blackbasta | 30 | 0.8667 | 0.9000 | +0.0333 |
| lorenz | 16 | 0.1250 | 1.0000 | +0.8750 |
| karma | 13 | 0.7692 | 1.0000 | +0.2308 |
| bianlian | 11 | 1.0000 | 0.5455 | -0.4545 |
| blackbyte | 7 | 1.0000 | 0.0000 | -1.0000 |
| quantum | 6 | 1.0000 | 1.0000 | +0.0000 |
| holyghost | 4 | 1.0000 | 0.2500 | -0.7500 |
| maui | 3 | 0.0000 | 1.0000 | +1.0000 |

