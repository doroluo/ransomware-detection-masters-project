# Experiment summary: A vs B

Binary task throughout: **0 = goodware, 1 = ransomware**. The ransomware family prefix is a split group, never a label.

## expA - Mendeley ransomware + Mendeley goodware (the LLM_Features baseline)

| split | goodware | ransomware | total | groups |
|---|---|---|---|---|
| train | 1116 | 975 | 2091 | 1141 |
| test | 131 | 382 | 513 | 146 |

Exact-duplicate leakage: **64/131 goodware** and **1/382 ransomware** test files have a byte-identical opcode stream in train (1765/2604 streams are unique; 2 appear under both labels).


| model | tok | emb | mask | acc | bal-acc | macro-P | macro-R | macro-F1 | AUC | recall(ran) | FPR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RF | WPC | w2v | 0.0 | 0.7797 | 0.8019 | 0.7397 | 0.8019 | 0.7496 | 0.8826 | 0.7565 | 0.1527 |
| MLP | WP | w2v | 0.0 | 0.8343 | 0.8662 | 0.7938 | 0.8662 | 0.8098 | 0.8155 | 0.8010 | 0.0687 |
| SVM-RBF | SW | w2v | 0.0 | 0.7661 | 0.8028 | 0.7353 | 0.8028 | 0.7398 | 0.9152 | 0.7277 | 0.1221 |


#### Goodware test set by architecture

Architecture is unknown for all 131 goodware test files: the Mendeley feature files record no architecture and the `good_test` binaries are not present on this machine (0 of 131 match a local binary by name), so no breakdown is possible for this experiment. See the architecture note below.

## expB - Mendeley ransomware + Goodware_Balanced (hard-negative-rich goodware)

| split | goodware | ransomware | total | groups |
|---|---|---|---|---|
| train | 1116 | 975 | 2091 | 295 |
| test | 131 | 382 | 513 | 59 |

Exact-duplicate leakage: **1/131 goodware** and **1/382 ransomware** test files have a byte-identical opcode stream in train (1980/2604 streams are unique; 0 appear under both labels).


| model | tok | emb | mask | acc | bal-acc | macro-P | macro-R | macro-F1 | AUC | recall(ran) | FPR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RF | WPC | w2v | 0.0 | 0.6842 | 0.7378 | 0.6809 | 0.7378 | 0.6629 | 0.7781 | 0.6283 | 0.1527 |
| MLP | WP | w2v | 0.0 | 0.7037 | 0.7358 | 0.6812 | 0.7358 | 0.6756 | 0.7469 | 0.6702 | 0.1985 |
| SVM-RBF | SW | w2v | 0.0 | 0.5945 | 0.6901 | 0.6509 | 0.6901 | 0.5862 | 0.8214 | 0.4948 | 0.1145 |


#### Goodware test set by architecture

| model | recall x64 (n=85) | recall x86 (n=46) |
|---|---|---|
| RF/WPC | 0.9882 | 0.5870 |
| MLP/WP | 1.0000 | 0.4348 |
| SVM-RBF/SW | 0.9882 | 0.6957 |

## A vs B, matched pairs

| model / tok / emb / mask | A macro-F1 | B macro-F1 | delta |
|---|---|---|---|
| RF / WPC / w2v / 0.0 | 0.7496 | 0.6629 | -0.0867 |
| MLP / WP / w2v / 0.0 | 0.8098 | 0.6756 | -0.1342 |
| SVM-RBF / SW / w2v / 0.0 | 0.7398 | 0.5862 | -0.1537 |

### Confounds

- Class sizes are **identical** between A and B, so a gap is not attributable to data volume. That rules out one confound; it does not rule out the architecture mix, which is not controlled - see below.
- The ransomware side is byte-identical in both experiments (same files, same family-disjoint split), so it contributes nothing to the difference.
- Exp A's goodware split is the Mendeley release's own and has no group discipline; Exp B's is group-disjoint by source project. B is therefore the *harder* split, and a lower B score is not by itself evidence of worse data.

### Reading the gap

A scores higher on every pair. Before reading that as "the Mendeley goodware is better", note that **48.9%** of A's goodware test set is a verbatim copy of its own training data, against **0.8%** for B, and that A contains 2 opcode stream(s) carrying both labels while B contains 0.

A large part of A's goodware score is therefore recall of streams the model has already memorised - overwhelmingly NSIS and Inno installer stubs, which `extract.py` disassembles in place of the payload. B has no such shortcut, and its goodware is deliberately ransomware-adjacent (encryption tools, archivers, backup and sync clients, secure-delete utilities), so its negatives sit much closer to the decision boundary.

The two numbers are answering different questions. A estimates performance on a corpus whose goodware half is ~50% duplicated; B estimates performance against hard negatives never seen in training. **B is the number to report as a generalisation estimate**; A is the comparable baseline, not the better result.

That reading is provisional. It assumes the two experiments differ only in the *quality* of their goodware, and the next section shows they also differ sharply in its *architecture mix* - which the models are demonstrably using. Read the two sections together.

### Architecture is a confound, and it is not controlled here

x86/x64 shares of each side of the task:

| side | x86 / x64 | source of the count |
|---|---|---|
| Mendeley ransomware, train (1,023) | 978/1,023 x86 (95.6%), 42 x64 (4.1%) | WRITEUP.md §3.2 (binaries are VM-only) |
| Mendeley ransomware, test (385) | 292/385 x86 (75.8%), 92 x64 (23.9%) | WRITEUP.md §3.3 |
| Mendeley goodware, train (1,115 on disk) | 630 x86 (56.5%), 485 x64 (43.5%) | WRITEUP.md §3.1 |
| Mendeley goodware, test (131) | unknown | those binaries are not present locally; WRITEUP.md §3.4 leaves them unprofiled |
| expB goodware, train (1116) | 915 x64 (82.0%), 201 x86 (18.0%) | `opcode_manifest.csv` |
| expB goodware, test (131) | 85 x64 (64.9%), 46 x86 (35.1%) | `opcode_manifest.csv` |

The ransomware side is overwhelmingly x86 (96% in train, 76% in test). The Mendeley goodware is 57% x86; Goodware_Balanced is roughly 23% x86. **Exp B therefore widens the architecture gap between the classes at the same time as it changes the goodware source**, and bitness is not a hidden variable that a byte-level model has to infer - it is written all over the operands (`rbp`, `r8`-`r15`, rip-relative addressing, the register calling convention), so it survives normalization into the token stream.

**A per-architecture breakdown is required before the A-vs-B gap can be attributed to the goodware source.** The goodware recall tables above are the part of it that can be produced here: they split each experiment's goodware test set by the architecture recorded in `opcode_manifest.csv`. If x64 goodware is recalled far better than x86 goodware, the model is partly reading bitness.

**It does.** Across 3 model/tokenizer pairs with a known architecture, x64 goodware is recalled between 0.29 and 0.57 better than x86 goodware (worst: expB MLP/WP, 1.0000 on x64 against 0.4348 on x86). Nearly every x64 benign file is caught and a large share of the x86 benign files are called ransomware — which is what a model keying on bitness looks like, given that the ransomware class is 96%/76% x86. **The A-vs-B gap therefore cannot yet be attributed to the goodware source.** Part of it is that Exp B's x86 goodware sits in the region of feature space the ransomware class occupies, and Exp A's goodware — 57% x86 — is not being scored the same way.

What cannot be produced here, and why:

* **Exp A's goodware test set** - the 131 `good_test` binaries are not on this machine (0 of 131 match a local binary by name) and the feature files carry no architecture, so there is nothing to join on. WRITEUP.md §3.4 has no counts for that folder either. Only the aggregate for `good_train` (56.5% x86) is known.
* **Either experiment's ransomware side** - identical in A and B, but the binaries are VM-only, so only the aggregate counts above exist. A per-sample architecture index has to come out of the VM (`check_arch.py` writes one) before ransomware recall can be split by bitness.

Until both exist, the honest statement is that A vs B varies goodware source **and** goodware architecture mix together.

### Deviation from the plan's split procedure

The plan asked for "the same stratified 80/20 split procedure" in both experiments. Two departures, both deliberate:

1. **The ransomware split is the Mendeley release's own family-disjoint split, not a random stratified 80/20.** `mal_train` holds 25 families and `mal_test` 15 entirely different ones, with zero overlap - a genuine unseen-family generalisation test. A random stratified re-split destroys it: `dharma` alone contributes 45 byte-identical samples, `phobos` 37, `lockbit` 35, so copies of one stream land on both sides and the task collapses into near-duplicate retrieval. Measured on the shipped split, ransomware test leakage is 1/382; a random re-split would take it far higher. `config.yaml` sets `split.ransomware: preserve_mendeley`, and the identical ransomware split is reused verbatim in both experiments, so it contributes nothing to the A-vs-B difference.
2. **Exp A's goodware split is also the release's own**, not a re-split. It has no group discipline, which is precisely what §2.1 of the audit measures: 64/131 of `good_test` is a verbatim copy of a training file. It is kept as shipped so Exp A remains the published baseline to compare against. Exp B's goodware side *is* split the way the plan intends - grouped by `entry_id` so no source project straddles train and test, stratified by bucket so `everyday`/`hard_negative`/`system` keep their proportions on both sides, with a fixed seed - and its counts are matched to Exp A's exactly (`split.match_counts_to: expA`). Measured: the pool is 53.9% everyday / 24.9% hard_negative / 21.2% system, and the chosen split is 53.9/24.8/21.2 in train and 54.2/24.4/21.4 in test.

Net effect: A is the baseline on its own terms, B is the same ransomware task with a harder, properly grouped goodware half.
