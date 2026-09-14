# graph2vec on linear-sweep CFGs

Prototype. Control-flow graphs are cut out of the linear disassembly in `Shared/Extract*/asm`, embedded with Weisfeiler-Lehman subtree features, and classified with the same RF / SVM-RBF / MLP grid as `llm_features_pipeline`. Split, metrics schema and per-arch / per-family breakdowns come from `cnn_vit_pipeline/cohort.py`, so these numbers sit next to `results/summary.md` without adjustment.

> **How to read the result tables.** There are **four** representations -- `wl_tfidf`, `wl_svd`, `graph2vec`, `size_only` -- and every (representation, model) pair occupies **two** rows. Those two rows are *not* two feature variants. They are the same fitted model on the same scores under two decision rules: `default` is argmax at 0.5, `calibrated` moves only the cut, chosen from TRAIN out-of-fold predictions. The `representation` column is the only place the feature set is named:

| representation | what the classifier actually sees |
|---|---|
| `wl_tfidf` | the sparse WL subtree-label histogram with IDF weighting -- i.e. the WL kernel feature map, ~320k-340k columns. The **plain WL-histogram baseline**. |
| `wl_svd` | that same histogram reduced by TruncatedSVD to 128 dense dimensions. A graph2vec-*style* dense embedding, but linear. |
| `graph2vec` | PV-DBOW trained on the WL documents, 128 dims. This is **graph2vec proper** (Narayanan et al. 2017), re-implemented in numpy. |
| `size_only` | ten scalar graph statistics and nothing else. The shortcut control. |

## 1. What was built

- **Blocks.** Leaders are the first instruction of each section, the instruction after every branch/call/ret, every direct branch or call target that resolves inside the same section, and the instruction after an undecoded `.skip` gap. Edges are fall-through, resolved branch targets and (optionally) call targets. No fall-through is emitted across a `.skip` gap or a section boundary.
- **Node labels.** `<dominant mnemonic class>.<terminator kind>.<length bucket>` with `+C/+S/+V` flags when the block contains a crypto, string or SIMD instruction. 19 mnemonic classes.
- **Caps.** 80,000 instruction lines and 5,000 blocks per file, whichever binds first. The corpus is 7.2GB of disassembly and single files run to 3.16M instructions; the tokenization baseline in `results/summary.md` truncates at 5,000 *instructions*, so this window is about six times wider than the baseline it is compared against.
- **Embedding.** WL iterations h=0..2, min_df=5 over TRAIN graphs, TF-IDF, then either TruncatedSVD(128) (`wl_svd`) or PV-DBOW (`graph2vec`). Everything is fitted on train rows only.

**graph2vec implementation note.** karateclub is sdist-only and its `Graph2Vec` needs `gensim>=4`, which has no cp314 wheel (pip offers only the unrelated pure-python 0.10.x line) and no local C toolchain to build from source. PV-DBOW -- the doc2vec variant graph2vec is defined as -- is therefore implemented directly in vectorised numpy (`graph2vec_pipeline/embed.py`), with a fixed seed and a step budget. Same objective, different optimiser schedule from karateclub's.

**What the graphs are not.** This is a linear sweep, so there is no function recovery and no indirect control flow: `call dword ptr [0x405128]`, `jmp eax`, vtable dispatch and SEH contribute no edge, and data decoded as code manufactures blocks. The unresolved-target fractions below say how much of the control flow is missing.

## 2. Graph statistics

| source | n | blocks median | blocks mean | edges median | edges mean | insns used median | mean block len | back edges median | at block cap |
|---|---|---|---|---|---|---|---|---|---|
| balanced_goodware (label 0) | 1240 | 5000.0000 | 3736.0000 | 6263.0000 | 4962.0000 | 18580.0000 | 4.9300 | 421.0000 | 0.5980 |
| mendeley_goodware (label 0) | 1243 | 3477.0000 | 3306.0000 | 4840.0000 | 4624.0000 | 13867.0000 | 4.1800 | 414.0000 | 0.3640 |
| mendeley_ransomware (label 1) | 1266 | 5000.0000 | 4009.0000 | 6295.0000 | 5424.0000 | 21925.0000 | 6.1000 | 612.0000 | 0.5550 |


By class and architecture (both trees pooled):

| label | arch | n | blocks median | edges median | mean block len | back edges median | unresolved targets median |
|---|---|---|---|---|---|---|---|
| 0 | x64 | 1488 | 5000.0000 | 6378.5000 | 4.7700 | 455.5000 | 838.5000 |
| 0 | x86 | 995 | 2741.0000 | 4057.0000 | 4.2400 | 344.0000 | 516.0000 |
| 1 | x64 | 114 | 5000.0000 | 6517.0000 | 4.3200 | 610.0000 | 2002.0000 |
| 1 | x86 | 1152 | 5000.0000 | 6260.0000 | 6.2700 | 612.0000 | 627.0000 |


Median share of branch/call sites whose target could not be resolved: **20.6%** (goodware 22.2%, ransomware 15.6%). 51% of files hit the 5,000-block cap and 27% hit the instruction cap, so for about half the corpus these are graphs of a prefix of the code, not of the program.


**Does size alone separate the classes?** Partly, and the direction depends on which goodware is used. Median blocks per file: Mendeley goodware 3,477, Mendeley ransomware 5,000, Goodware_Balanced 5,000; median edges 4,840 / 6,295 / 6,263. So against Mendeley goodware the ransomware graphs are the larger ones, and against Goodware_Balanced they are the same size. The `size_only` rows in section 3 put a number on exactly that, and it is the number to quote: the ten scalars alone reach **ROC-AUC 0.919** on `mendeley` and **0.702** on `balanced`. Two cautions on those medians: `n_blocks` is **censored** -- half the corpus sits exactly at the 5,000-block cap, so a median of 5,000 means 'at least 5,000', not 5,000 -- and the cap rate itself differs by class and by tree (see the `at block cap` column), which makes 'hit the cap' a feature in its own right.

## 3. Results

### mendeley

train 1799 (1003 good / 796 ransomware), val 219, test 491 (129 good / 362 ransomware). Majority-class accuracy on test: 0.7373.

| representation | model | decision | acc | bal_acc | macro_f1 | auc | recall_ran | fpr | n_feat |
|---|---|---|---|---|---|---|---|---|---|
| wl_tfidf | LR | default | 0.7556 | 0.8243 | 0.7398 | 0.9766 | 0.6796 | 0.0310 | 323076 |
| wl_tfidf | LR | calibrated | 0.8554 | 0.8895 | 0.8352 | 0.9766 | 0.8177 | 0.0388 | 323076 |
| wl_svd | RF | default | 0.8167 | 0.8707 | 0.7987 | 0.9678 | 0.7569 | 0.0155 | 128 |
| wl_svd | RF | calibrated | 0.8310 | 0.8804 | 0.8125 | 0.9678 | 0.7762 | 0.0155 | 128 |
| wl_svd | SVM-RBF | default | 0.5682 | 0.6922 | 0.5663 | 0.9319 | 0.4309 | 0.0465 | 128 |
| wl_svd | SVM-RBF | calibrated | 0.5356 | 0.6751 | 0.5353 | 0.9319 | 0.3812 | 0.0310 | 128 |
| wl_svd | MLP | default | 0.8330 | 0.8643 | 0.8105 | 0.9489 | 0.7983 | 0.0698 | 128 |
| wl_svd | MLP | calibrated | 0.8615 | 0.8811 | 0.8386 | 0.9489 | 0.8398 | 0.0775 | 128 |
| graph2vec | RF | default | 0.5275 | 0.6746 | 0.5274 | 0.9550 | 0.3646 | 0.0155 | 128 |
| graph2vec | RF | calibrated | 0.6436 | 0.7533 | 0.6378 | 0.9550 | 0.5221 | 0.0155 | 128 |
| graph2vec | SVM-RBF | default | 0.7128 | 0.7903 | 0.6993 | 0.9547 | 0.6271 | 0.0465 | 128 |
| graph2vec | SVM-RBF | calibrated | 0.9145 | 0.9170 | 0.8951 | 0.9547 | 0.9116 | 0.0775 | 128 |
| graph2vec | MLP | default | 0.8167 | 0.8607 | 0.7964 | 0.9357 | 0.7680 | 0.0465 | 128 |
| graph2vec | MLP | calibrated | 0.8167 | 0.8607 | 0.7964 | 0.9357 | 0.7680 | 0.0465 | 128 |
| size_only | RF | default | 0.5499 | 0.6873 | 0.5493 | 0.9193 | 0.3978 | 0.0233 | 10 |
| size_only | RF | calibrated | 0.4358 | 0.6124 | 0.4321 | 0.9193 | 0.2403 | 0.0155 | 10 |
| size_only | LR | default | 0.6558 | 0.7217 | 0.6409 | 0.7579 | 0.5829 | 0.1395 | 10 |
| size_only | LR | calibrated | 0.6395 | 0.6707 | 0.6150 | 0.7579 | 0.6050 | 0.2636 | 10 |


`decision = default` is argmax at 0.5. `calibrated` reuses the same fitted model and the same scores, moving only the cut, chosen on TRAIN out-of-fold predictions (`cross_val_predict`, same folds) -- test is never consulted. Train is 45% ransomware and test is 74%, so the default cut is at the wrong prior.


Goodware recall by architecture (1 - FPR within arch), calibrated rows:

| representation | model | x64 (n=12) | x86 (n=117) |
|---|---|---|---|
| wl_tfidf | LR | 1.0000 | 0.9573 |
| wl_svd | RF | 1.0000 | 0.9829 |
| wl_svd | SVM-RBF | 1.0000 | 0.9658 |
| wl_svd | MLP | 1.0000 | 0.9145 |
| graph2vec | RF | 1.0000 | 0.9829 |
| graph2vec | SVM-RBF | 1.0000 | 0.9145 |
| graph2vec | MLP | 1.0000 | 0.9487 |
| size_only | RF | 1.0000 | 0.9829 |
| size_only | LR | 0.5000 | 0.7607 |


Per-family ransomware recall, best structural row (`graph2vec/SVM-RBF`, macro-F1 0.8951). All 14 families are unseen in training:

| family | recall | correct | support |
|---|---|---|---|
| blackbyte | 0.5714 | 4 | 7 |
| hive | 0.6400 | 32 | 50 |
| quantum | 0.8333 | 5 | 6 |
| blackcat | 0.8800 | 44 | 50 |
| blackbasta | 0.9000 | 27 | 30 |
| clop | 0.9778 | 44 | 45 |
| avoslocker | 1.0000 | 50 | 50 |
| bianlian | 1.0000 | 11 | 11 |
| bluesky | 1.0000 | 34 | 34 |
| holyghost | 1.0000 | 4 | 4 |
| karma | 1.0000 | 13 | 13 |
| lorenz | 1.0000 | 16 | 16 |
| maui | 1.0000 | 3 | 3 |
| playcrypt | 1.0000 | 43 | 43 |


Weakest three families: blackbyte (0.57, n=7), hive (0.64, n=50), quantum (0.83, n=6). Family recall spreads across the full 0-1 range on a single fixed split, so a difference of a few points of macro-F1 between two rows here is within the noise of which 14 families happened to land in test.

### balanced

train 1798 (1002 good / 796 ransomware), val 219, test 489 (127 good / 362 ransomware). Majority-class accuracy on test: 0.7403.

| representation | model | decision | acc | bal_acc | macro_f1 | auc | recall_ran | fpr | n_feat |
|---|---|---|---|---|---|---|---|---|---|
| wl_tfidf | LR | default | 0.4888 | 0.6266 | 0.4886 | 0.8396 | 0.3398 | 0.0866 | 343769 |
| wl_tfidf | LR | calibrated | 0.5031 | 0.6260 | 0.5021 | 0.8396 | 0.3702 | 0.1181 | 343769 |
| wl_svd | RF | default | 0.5317 | 0.6658 | 0.5309 | 0.8373 | 0.3867 | 0.0551 | 128 |
| wl_svd | RF | calibrated | 0.5317 | 0.6633 | 0.5307 | 0.8373 | 0.3895 | 0.0630 | 128 |
| wl_svd | SVM-RBF | default | 0.4703 | 0.6218 | 0.4702 | 0.8314 | 0.3066 | 0.0630 | 128 |
| wl_svd | SVM-RBF | calibrated | 0.5133 | 0.6406 | 0.5124 | 0.8314 | 0.3757 | 0.0945 | 128 |
| wl_svd | MLP | default | 0.4928 | 0.6242 | 0.4925 | 0.8416 | 0.3508 | 0.1024 | 128 |
| wl_svd | MLP | calibrated | 0.5583 | 0.6633 | 0.5538 | 0.8416 | 0.4448 | 0.1181 | 128 |
| graph2vec | RF | default | 0.4213 | 0.6014 | 0.4170 | 0.8306 | 0.2265 | 0.0236 | 128 |
| graph2vec | RF | calibrated | 0.5297 | 0.6619 | 0.5288 | 0.8306 | 0.3867 | 0.0630 | 128 |
| graph2vec | SVM-RBF | default | 0.5133 | 0.6406 | 0.5124 | 0.8519 | 0.3757 | 0.0945 | 128 |
| graph2vec | SVM-RBF | calibrated | 0.5133 | 0.6432 | 0.5126 | 0.8519 | 0.3729 | 0.0866 | 128 |
| graph2vec | MLP | default | 0.5460 | 0.6627 | 0.5434 | 0.8514 | 0.4199 | 0.0945 | 128 |
| graph2vec | MLP | calibrated | 0.5419 | 0.6599 | 0.5396 | 0.8514 | 0.4144 | 0.0945 | 128 |
| size_only | RF | default | 0.4131 | 0.5857 | 0.4095 | 0.7016 | 0.2265 | 0.0551 | 10 |
| size_only | RF | calibrated | 0.4131 | 0.5883 | 0.4091 | 0.7016 | 0.2238 | 0.0472 | 10 |
| size_only | LR | default | 0.6135 | 0.6802 | 0.5993 | 0.6252 | 0.5414 | 0.1811 | 10 |
| size_only | LR | calibrated | 0.6237 | 0.6641 | 0.6017 | 0.6252 | 0.5801 | 0.2520 | 10 |


`decision = default` is argmax at 0.5. `calibrated` reuses the same fitted model and the same scores, moving only the cut, chosen on TRAIN out-of-fold predictions (`cross_val_predict`, same folds) -- test is never consulted. Train is 45% ransomware and test is 74%, so the default cut is at the wrong prior.


Goodware recall by architecture (1 - FPR within arch), calibrated rows:

| representation | model | x64 (n=81) | x86 (n=46) |
|---|---|---|---|
| wl_tfidf | LR | 0.9877 | 0.6957 |
| wl_svd | RF | 1.0000 | 0.8261 |
| wl_svd | SVM-RBF | 0.9877 | 0.7609 |
| wl_svd | MLP | 0.9753 | 0.7174 |
| graph2vec | RF | 1.0000 | 0.8261 |
| graph2vec | SVM-RBF | 0.9753 | 0.8043 |
| graph2vec | MLP | 0.9753 | 0.7826 |
| size_only | RF | 0.9753 | 0.9130 |
| size_only | LR | 0.6914 | 0.8478 |


> On `balanced` the single highest macro-F1 row in the whole table is the **shortcut control** (`size_only/LR`, 0.6017), ahead of every structural representation. That is a finding about this dataset, not a result for graph embeddings: on Goodware_Balanced the CFG features stop working before the ten size scalars do. The per-family table below is for the best *structural* row.


Per-family ransomware recall, best structural row (`wl_svd/MLP`, macro-F1 0.5538). All 14 families are unseen in training:

| family | recall | correct | support |
|---|---|---|---|
| avoslocker | 0.0000 | 0 | 50 |
| bianlian | 0.0000 | 0 | 11 |
| blackbyte | 0.0000 | 0 | 7 |
| blackcat | 0.0000 | 0 | 50 |
| hive | 0.0000 | 0 | 50 |
| holyghost | 0.0000 | 0 | 4 |
| blackbasta | 0.2333 | 7 | 30 |
| playcrypt | 0.9302 | 40 | 43 |
| clop | 0.9333 | 42 | 45 |
| bluesky | 1.0000 | 34 | 34 |
| karma | 1.0000 | 13 | 13 |
| lorenz | 1.0000 | 16 | 16 |
| maui | 1.0000 | 3 | 3 |
| quantum | 1.0000 | 6 | 6 |


Weakest three families: avoslocker (0.00, n=50), bianlian (0.00, n=11), blackbyte (0.00, n=7); **6 families are missed entirely** (avoslocker, bianlian, blackbyte, blackcat, hive, holyghost). Family recall spreads across the full 0-1 range on a single fixed split, so a difference of a few points of macro-F1 between two rows here is within the noise of which 14 families happened to land in test.


## 4. Reading the numbers

- **mendeley**: best row is `graph2vec/SVM-RBF` at macro-F1 0.895 (AUC 0.955); the graph-size-only control reaches macro-F1 0.641 and **AUC 0.919**, against 0.977 for the best structural representation.
- **balanced**: best row is `size_only/LR` at macro-F1 0.602 (AUC 0.625); the graph-size-only control reaches macro-F1 0.602 and **AUC 0.702**, against 0.852 for the best structural representation.


Two things to read carefully.

**The `size_only` control.** Ten scalars -- block count, edge count, instructions used, mean block length, back edges, self loops, call edges, resolved and unresolved targets, section count -- and nothing else. In macro-F1 it is clearly behind the learned representations, and the medians in section 2 agree that graph size does not split the classes outright: about half of *each* class sits at the block cap and the class medians overlap. **In ranking terms it is not behind at all**: its ROC-AUC is close to the structural rows'. So most of what the CFG representation buys over file size is a better *operating point*, not a better ordering. Any claim that the graph structure is carrying the detection has to be made against the size control's AUC, not against a majority baseline.

**Threshold, not representation, is the larger effect here.** Several rows reach AUC above 0.95 while scoring near 0.55 macro-F1 at the default cut. The split is 45% ransomware in train and 74% in test; a model left at 0.5 is answering a different question from the one the test set asks. The calibrated rows move only the cut and nothing else.


**Which representation actually wins.** `wl_tfidf/LR` is the plain WL-subtree-kernel baseline -- a linear model on the WL histogram is a WL kernel machine -- so it is the reference the learned embeddings have to beat. On `mendeley` it has the **highest ROC-AUC of any row (0.977)** while `graph2vec/SVM-RBF` has the highest macro-F1 (0.895) at AUC 0.955. Ordering and operating point disagree, on 491 test rows, between two representations built from the same WL documents. The honest reading is that the 128-dimensional embeddings do not improve on the histogram they are compressed from; they only move where the threshold lands. Nothing in these tables supports a claim that PV-DBOW learns graph structure the WL kernel misses.


## 5. The architecture confound

| dataset | fold | goodware x86 | goodware x64 | all arch |
|---|---|---|---|---|
| mendeley | train | 566 | 437 | {'x86': 1341, 'x64': 458} |
| mendeley | test | 117 | 12 | {'x86': 407, 'x64': 84} |
| balanced | train | 164 | 838 | {'x86': 939, 'x64': 859} |
| balanced | test | 46 | 81 | {'x86': 336, 'x64': 153} |


Ransomware is 95% x86 in the train split (862 x86 / 42 x64, val fold included) and 80% x86 in test (290 / 72) in **both** datasets -- the ransomware half of the split is identical by construction. The goodware half is what changes, and it changes a lot: Mendeley goodware train is ~57% x86, Goodware_Balanced train is ~17% x86. So on `balanced` a classifier can reach a good training score with the rule *x64 implies goodware*, and that rule then meets a test set that is 64% x64 goodware but still 80% x86 ransomware. The `goodware recall by architecture` tables above show the damage directly: on `balanced` every structural row keeps goodware recall near 1.00 inside x64 and drops to 0.70-0.83 inside x86. Per-architecture modelling is the obvious next control. It has been run for the mnemonic TF-IDF baseline -- see `results/rules/summary.md` §5.6, which reports a model trained and tested inside one architecture -- but not yet for these representations.


## 6. Where this sits against the other tracks

Same cohort, same family-disjoint split, same metrics module. Best macro-F1 per track, read live from each track's `metrics.json` where one exists:

| track | best macro-F1 (mendeley) | best macro-F1 (balanced) | where |
|---|---|---|---|
| mnemonic 1-3-gram TF-IDF + LogReg / LinearSVC | **0.968** | 0.802 | `results/rules/summary.md` |
| tokenizer + word2vec + RF/SVM/MLP (expC / expD) | 0.926 | 0.844 | `results/summary.md` |
| mined n-gram + hand-written behaviour rules | 0.797 | 0.670 | `results/rules/summary.md` |
| graph2vec / WL on CFGs, **untuned** prototype, structural rows only | 0.895 | 0.554 | §3 here |
| graph2vec / WL on CFGs, **untuned**, incl. the `size_only` control | 0.895 | 0.602 | §3 here |
| **graph2vec / WL on CFGs, tuned, structural rows only** | **0.947** | **0.675** | §7 here |
| graph2vec / WL on CFGs, tuned, incl. the `size_only` control | 0.947 | 0.675 | §7 here |
| CNN-ViT on instruction images (single run; 5-seed mean 0.628 / 0.502) | 0.575 | 0.542 | `results/cnn_vit/summary.md` |

The two graph2vec pairs are the same code on the same split; what separates them is that the tuned rows chose their construction, node labelling, WL depth, normalisation and classifier by grouped cross-validation on TRAIN (§7) while the untuned rows are one hand-picked configuration. Read the untuned pair as what the prototype did, not as what the representation can do.

Two readings of the untuned pair. (i) On Mendeley goodware the CFG track was competitive with the tokenization baseline and behind a plain TF-IDF over the same mnemonics. (ii) On Goodware_Balanced it collapsed to roughly the size control, which looked like the more informative half of the comparison: whatever the CFG features separated on Mendeley looked like a property of *that* goodware corpus. §7 puts a searched configuration next to both; its before/after and `size_only` tables are what the second reading has to be checked against.


## 7. Tuned run

Section 3 is the *untuned* prototype and is kept as written. This section is a second pass whose only rule was that the test half of the split is opened once, at the end, for the configurations a cross-validated search on TRAIN had already committed to.

**The search protocol, and why the old CV numbers were not usable.** Section 3's `cv_best_f1_macro` column came from `StratifiedKFold(n_splits=2)` with no groups: ransomware families straddled the fold boundary, so a model could recognise a family it had already seen. That is why the untuned `balanced` rows report CV macro-F1 near 0.98 against a test macro-F1 of 0.53. Here every selection decision uses `StratifiedGroupKFold(n_splits=5)` over `family_or_group`, so a whole family (and on `balanced` a whole source project) is held out; the reported CV number is the pooled out-of-fold macro-F1. 103 configurations were evaluated on `mendeley` and the same protocol on `balanced`; all of them are in `results/graph2vec/tuned/<dataset>/cv_search.csv` with their CV macro-F1, balanced accuracy and AUC.

**What the search was allowed to change.** Graph construction (block cap 5,000 vs 20,000; a single 80k-instruction prefix vs four 20k-instruction windows spread over the file; call edges on/off; cross-section branch targets on/off; import thunks as shared extern nodes on/off; edge kinds salted into the WL relabelling or not), node labelling (`class` = dominant mnemonic class + terminator + length bucket + crypto/string/SIMD flags, `class_noflag`, `dom_term`, `seq_term` = hashed 8-mnemonic-class sequence, `deg_dom` = (in-degree, out-degree, class, terminator), `lenbucket` = length bucket + terminator only), WL depth h=1..4, histogram normalisation (TF-IDF / binary / log) and min_df, whether the size scalars are concatenated onto the histogram, SVD and PV-DBOW dimensions, the classifier grids, and architecture-aware sample reweighting.


### 7.1 mendeley

test 491 (129 good / 362 ransomware), train 2018.

**Before / after.** `cv` is the group-CV macro-F1 the configuration was selected on; `gap` is CV minus test.

| row | config | cv | acc | bal_acc | macro_f1 | auc | fpr | gap |
|---|---|---|---|---|---|---|---|---|
| before (best row of the untuned run) | graph2vec/SVM-RBF | - | 0.9145 | 0.9170 | 0.8951 | 0.9547 | 0.0775 | - |
| before (best *structural* row) | graph2vec/SVM-RBF | - | 0.9145 | 0.9170 | 0.8951 | 0.9547 | 0.0775 | - |
| after: baseline_h2_cap5000 [argmax] | wl_tfidf/LR | 0.9200 | 0.7393 | 0.8107 | 0.7242 | 0.9665 | 0.0388 | +0.1958 |
| after: baseline_h2_cap5000 [oof-thr] | wl_tfidf/LR | 0.9200 | 0.9145 | 0.8721 | 0.8855 | 0.9665 | 0.2171 | +0.0345 |
| after: tuned_best [argmax] | wl_svd/RF | 0.9784 | 0.7230 | 0.7722 | 0.7025 | 0.9390 | 0.1240 | +0.2759 |
| after: tuned_best [oof-thr] | wl_svd/RF | 0.9784 | 0.9124 | 0.8857 | 0.8867 | 0.9390 | 0.1705 | +0.0917 |
| after: tuned_best_sparse_histogram [argmax] | wl_tfidf/LR | 0.9739 | 0.7902 | 0.8478 | 0.7722 | 0.9824 | 0.0310 | +0.2016 |
| after: tuned_best_sparse_histogram [oof-thr] | wl_tfidf/LR | 0.9739 | 0.9593 | 0.9474 | 0.9474 | 0.9824 | 0.0775 | +0.0264 |
| after: tuned_best_graph2vec [argmax] | graph2vec/RF | 0.9567 | 0.7515 | 0.5296 | 0.4857 | 0.8455 | 0.9380 | +0.4710 |
| after: tuned_best_graph2vec [oof-thr] | graph2vec/RF | 0.9567 | 0.7373 | 0.5000 | 0.4244 | 0.8455 | 1.0000 | +0.5323 |
| after: size_only_control [argmax] | size_only/LR | 0.8099 | 0.8248 | 0.8463 | 0.7994 | 0.9394 | 0.1085 | +0.0106 |
| after: size_only_control [oof-thr] | size_only/LR | 0.8099 | 0.8350 | 0.8682 | 0.8132 | 0.9394 | 0.0620 | -0.0032 |
| floor: majority | - | - | 0.7373 | 0.5000 | 0.4244 | - | 1.0000 | - |
| floor: x86_rule | - | - | 0.6151 | 0.4471 | 0.4335 | - | 0.9070 | - |

**Reading that table.** Best committed row: `tuned_best_sparse_histogram` (`wl_tfidf/LR`) at macro-F1 **0.9474**, AUC 0.9824, FPR 0.0775. Untuned best was 0.8951 (`graph2vec/SVM-RBF`), so tuning moves macro-F1 by +0.0523. The same untuned *configuration* re-scored here under the group-CV threshold gives 0.8855. It clears the `size_only` shortcut control (0.8132 macro-F1, AUC 0.9394) by +0.1343. Against the floors -- majority 0.4244, x86-rule 0.4335 -- it sits above the architecture shortcut (+0.5139). CV->test gap on that row: +0.0265 (CV 0.9739).

Calibration, same cohort and split:

| track | mendeley macro-F1 | balanced macro-F1 | where |
|---|---|---|---|
| mnemonic 1-3-gram TF-IDF + LogReg | 0.955 | 0.800 | results/rules/summary.md |
| tokenizer + word2vec (expC) | 0.926 | - | results/summary.md |
| tokenizer + word2vec (expD) | 0.844 | - | results/summary.md |


**What helped, and what did not.** The search is a coordinate ascent, so each trial below is scored against the configuration in force when it ran, not against a global best; a change is kept only if it buys at least 0.005 CV macro-F1. Where a stage is a grid rather than a sequence of trials only its winner is shown. The last three stages carry `kept = -` because they do not feed the ascent: the configuration sent to test is the global CV argmax over all 103 rows, whatever the marginal gain.

| stage | change | cv_macro_f1 | delta | kept |
|---|---|---|---|---|
| start | the baseline construction and node labelling, under group CV, min_df 3 | 0.8857 | - | - |
| graph construction | max_blocks: 5000 -> 20000 | 0.8268 | -0.0590 | no |
| graph construction | cross_section: False -> True | 0.8863 | +0.0005 | no |
| graph construction | extern: False -> True | 0.8982 | +0.0124 | yes |
| graph construction | calls: True -> False | 0.8518 | -0.0463 | no |
| graph construction | windows: 1 -> 4 | 0.8962 | -0.0020 | no |
| graph construction | edge_labels: False -> True | 0.8793 | -0.0189 | no |
| node labelling x WL depth (24 cells) | best cell: labelling lenbucket, h=4 | 0.9627 | +0.0645 | yes |
| normalisation / min_df / size scalars (6 trials) | norm: tfidf -> binary | 0.8024 | -0.1603 | no |
| normalisation / min_df / size scalars (6 trials) | norm: tfidf -> log | 0.8264 | -0.1363 | no |
| normalisation / min_df / size scalars (6 trials) | min_df: 3 -> 2 | 0.9256 | -0.0371 | no |
| normalisation / min_df / size scalars (6 trials) | min_df: 3 -> 5 | 0.9597 | -0.0030 | no |
| normalisation / min_df / size scalars (6 trials) | min_df: 3 -> 10 | 0.9505 | -0.0122 | no |
| normalisation / min_df / size scalars (6 trials) | with_size: False -> True | 0.8546 | -0.1081 | no |
| dense embedding vs the histogram it compresses | best graph2vec: d=256, RF(n_estimators=400) | 0.9567 | -0.0060 | no |
| dense embedding vs the histogram it compresses | best wl_svd: d=300, RF(n_estimators=400) | 0.9774 | +0.0147 | yes |
| classifier grid on the histogram | best model: LR(C=10.0,class_weight=balanced) on wl_tfidf | 0.9739 | -0.0035 | - |
| classifier grid on the embedding | best model: RF(n_estimators=1000,max_depth=None,min_samples_leaf=3) on wl_svd | 0.9784 | +0.0010 | - |
| architecture-aware sample reweighting | best reweighted model: RF(n_estimators=300,max_depth=None,min_samples_leaf=3)+archw | 0.9774 | -0.0000 | - |

Node labelling against WL depth, the one full grid in the search (CV macro-F1):

| labelling | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|
| class | 0.8882 | 0.8982 | 0.9198 | 0.9240 |
| class_noflag | 0.8941 | 0.8992 | 0.9204 | 0.9251 |
| deg_dom | 0.8526 | 0.8934 | 0.9157 | 0.9370 |
| dom_term | 0.9113 | 0.9449 | 0.9495 | 0.9546 |
| lenbucket | 0.9153 | 0.9366 | 0.9479 | 0.9627 |
| seq_term | 0.9168 | 0.9241 | 0.9230 | 0.9267 |

**Does the structural model beat graph size?** Best CV reached by each representation anywhere in the search:

| representation | n_configs | best_cv_macro_f1 | best_cv_auc | model |
|---|---|---|---|---|
| size_only (the shortcut control) | 4 | 0.8099 | 0.9381 | LR(C=1.0) |
| wl_tfidf histogram | 51 | 0.9739 | 0.9929 | LR(C=10.0,class_weight=balanced) |
| wl_svd | 39 | 0.9784 | 0.9970 | RF(n_estimators=1000,max_depth=None,min_samples_leaf=3) |
| graph2vec | 9 | 0.9567 | 0.9909 | RF(n_estimators=400) |

**Top 5 configurations by CV macro-F1**, and what they score on test. Only the `committed = yes` row was selected before the test set was opened; the rest were scored afterwards purely to size the CV->test gap, and nothing is chosen from this table.

| construction | representation | model | cv_macro_f1 | cv_bal_acc | cv_auc | test_macro_f1 | test_auc | gap | committed |
|---|---|---|---|---|---|---|---|---|---|
| b5000/w1/calls1/xs0/ext1 | wl_svd/lenbucket/h4/el0/tfidf/df3/size0/d300 | RF(n_estimators=1000,max_depth=20,min_samples_leaf=3) | 0.9784 | 0.9775 | 0.9970 | 0.7025 | 0.9390 | 0.2759 | post hoc |
| b5000/w1/calls1/xs0/ext1 | wl_svd/lenbucket/h4/el0/tfidf/df3/size0/d300 | RF(n_estimators=1000,max_depth=None,min_samples_leaf=3) | 0.9784 | 0.9775 | 0.9970 | 0.7025 | 0.9390 | 0.2759 | yes |
| b5000/w1/calls1/xs0/ext1 | wl_svd/lenbucket/h4/el0/tfidf/df3/size0/d300 | RF(n_estimators=300,max_depth=None,min_samples_leaf=3) | 0.9779 | 0.9769 | 0.9970 | 0.7550 | 0.9373 | 0.2229 | post hoc |
| b5000/w1/calls1/xs0/ext1 | wl_svd/lenbucket/h4/el0/tfidf/df3/size0/d300 | RF(n_estimators=300,max_depth=20,min_samples_leaf=3) | 0.9779 | 0.9769 | 0.9970 | 0.7550 | 0.9373 | 0.2229 | post hoc |
| b5000/w1/calls1/xs0/ext1 | wl_svd/lenbucket/h4/el0/tfidf/df3/size0/d300 | RF(n_estimators=1000,max_depth=None,min_samples_leaf=1) | 0.9779 | 0.9769 | 0.9969 | 0.6603 | 0.9349 | 0.3176 | post hoc |

**Per architecture**, out-of-fold-threshold rows:

| config | model | x86 n | x86 macro_f1 | x86 rec_ran | x86 rec_good | x64 n | x64 macro_f1 | x64 rec_ran | x64 rec_good |
|---|---|---|---|---|---|---|---|---|---|
| baseline_h2_cap5000 | wl_tfidf/LR | 407 | 0.9090 | 1.0000 | 0.7607 | 84 | 0.7619 | 0.8056 | 1.0000 |
| tuned_best | wl_svd/RF | 407 | 0.8876 | 0.9448 | 0.8205 | 84 | 0.8714 | 0.9306 | 0.9167 |
| tuned_best_sparse_histogram | wl_tfidf/LR | 407 | 0.9576 | 0.9828 | 0.9231 | 84 | 0.8714 | 0.9306 | 0.9167 |
| tuned_best_graph2vec | graph2vec/RF | 407 | 0.4161 | 1.0000 | 0.0000 | 84 | 0.4615 | 1.0000 | 0.0000 |
| size_only_control | size_only/LR | 407 | 0.9134 | 0.9207 | 0.9402 | 84 | 0.3823 | 0.3056 | 0.9167 |

**Per-family ransomware recall**, tuned_best_sparse_histogram (`wl_tfidf/LR`, macro-F1 0.9474). All test families are unseen in training:

| family | recall | correct | support |
|---|---|---|---|
| hive | 0.8600 | 43 | 50 |
| blackbasta | 0.9000 | 27 | 30 |
| bianlian | 1.0000 | 11 | 11 |
| avoslocker | 1.0000 | 50 | 50 |
| blackbyte | 1.0000 | 7 | 7 |
| blackcat | 1.0000 | 50 | 50 |
| bluesky | 1.0000 | 34 | 34 |
| clop | 1.0000 | 45 | 45 |
| holyghost | 1.0000 | 4 | 4 |
| karma | 1.0000 | 13 | 13 |
| lorenz | 1.0000 | 16 | 16 |
| maui | 1.0000 | 3 | 3 |
| playcrypt | 1.0000 | 43 | 43 |
| quantum | 1.0000 | 6 | 6 |


### 7.2 balanced

test 489 (127 good / 362 ransomware), train 2017.

**Before / after.** `cv` is the group-CV macro-F1 the configuration was selected on; `gap` is CV minus test.

| row | config | cv | acc | bal_acc | macro_f1 | auc | fpr | gap |
|---|---|---|---|---|---|---|---|---|
| before (best row of the untuned run) | size_only/LR | - | 0.6237 | 0.6641 | 0.6017 | 0.6252 | 0.2520 | - |
| before (best *structural* row) | wl_svd/MLP | - | 0.5583 | 0.6633 | 0.5538 | 0.8416 | 0.1181 | - |
| after: baseline_h2_cap5000 [argmax] | wl_tfidf/LR | 0.6699 | 0.4724 | 0.6079 | 0.4723 | 0.8228 | 0.1102 | +0.1976 |
| after: baseline_h2_cap5000 [oof-thr] | wl_tfidf/LR | 0.6699 | 0.5808 | 0.6478 | 0.5680 | 0.8228 | 0.2126 | +0.1019 |
| after: tuned_best [argmax] | wl_tfidf/LR | 0.9323 | 0.5685 | 0.6804 | 0.5647 | 0.8265 | 0.0866 | +0.3677 |
| after: tuned_best [oof-thr] | wl_tfidf/LR | 0.9323 | 0.5869 | 0.6699 | 0.5775 | 0.8265 | 0.1575 | +0.3548 |
| after: tuned_best_wl_svd [argmax] | wl_svd/MLP | 0.9311 | 0.6012 | 0.6872 | 0.5919 | 0.8548 | 0.1339 | +0.3393 |
| after: tuned_best_wl_svd [oof-thr] | wl_svd/MLP | 0.9311 | 0.6973 | 0.7445 | 0.6755 | 0.8548 | 0.1575 | +0.2557 |
| after: tuned_best_graph2vec [argmax] | graph2vec/MLP | 0.8432 | 0.5869 | 0.6903 | 0.5812 | 0.8824 | 0.0945 | +0.2620 |
| after: tuned_best_graph2vec [oof-thr] | graph2vec/MLP | 0.8432 | 0.6033 | 0.6912 | 0.5942 | 0.8824 | 0.1260 | +0.2490 |
| after: size_only_control [argmax] | size_only/LR | 0.7735 | 0.6053 | 0.6721 | 0.5915 | 0.6749 | 0.1890 | +0.1820 |
| after: size_only_control [oof-thr] | size_only/LR | 0.7735 | 0.6115 | 0.6737 | 0.5962 | 0.6749 | 0.1969 | +0.1773 |
| floor: majority | - | - | 0.7403 | 0.5000 | 0.4254 | - | 1.0000 | - |
| floor: x86_rule | - | - | 0.7587 | 0.7195 | 0.7048 | - | 0.3622 | - |

**Reading that table.** Best committed row: `tuned_best_wl_svd` (`wl_svd/MLP`) at macro-F1 **0.6755**, AUC 0.8548, FPR 0.1575. Untuned best was 0.6017 (`size_only/LR`), so tuning moves macro-F1 by +0.0738. The same untuned *configuration* re-scored here under the group-CV threshold gives 0.5680. It clears the `size_only` shortcut control (0.5962 macro-F1, AUC 0.6749) by +0.0792. Against the floors -- majority 0.4254, x86-rule 0.7048 -- it sits **below** the architecture shortcut (-0.0293). That is the number to quote for this dataset: a rule that reads nothing but the ELF/PE machine field scores higher than the best searched CFG model, so nothing here demonstrates that the control-flow structure is carrying the detection. CV->test gap on that row: +0.2556 (CV 0.9311).

Calibration, same cohort and split:

| track | mendeley macro-F1 | balanced macro-F1 | where |
|---|---|---|---|
| mnemonic 1-3-gram TF-IDF + LogReg | 0.955 | 0.800 | results/rules/summary.md |
| tokenizer + word2vec (expC) | 0.926 | - | results/summary.md |
| tokenizer + word2vec (expD) | 0.844 | - | results/summary.md |


**What helped, and what did not.** The search is a coordinate ascent, so each trial below is scored against the configuration in force when it ran, not against a global best; a change is kept only if it buys at least 0.005 CV macro-F1. Where a stage is a grid rather than a sequence of trials only its winner is shown. The last three stages carry `kept = -` because they do not feed the ascent: the configuration sent to test is the global CV argmax over all 103 rows, whatever the marginal gain.

| stage | change | cv_macro_f1 | delta | kept |
|---|---|---|---|---|
| start | the baseline construction and node labelling, under group CV, min_df 3 | 0.5834 | - | - |
| graph construction | max_blocks: 5000 -> 20000 | 0.6259 | +0.0425 | yes |
| graph construction | cross_section: False -> True | 0.6259 | +0.0000 | no |
| graph construction | extern: False -> True | 0.6265 | +0.0007 | no |
| graph construction | calls: True -> False | 0.6375 | +0.0116 | yes |
| graph construction | windows: 1 -> 4 | 0.7325 | +0.0951 | yes |
| graph construction | edge_labels: False -> True | 0.7142 | -0.0183 | no |
| node labelling x WL depth (24 cells) | best cell: labelling dom_term, h=1 | 0.9065 | +0.1740 | yes |
| normalisation / min_df / size scalars (6 trials) | norm: tfidf -> binary | 0.9302 | +0.0237 | yes |
| normalisation / min_df / size scalars (6 trials) | norm: tfidf -> log | 0.9038 | -0.0027 | no |
| normalisation / min_df / size scalars (6 trials) | min_df: 3 -> 2 | 0.9060 | -0.0005 | no |
| normalisation / min_df / size scalars (6 trials) | min_df: 3 -> 5 | 0.9076 | +0.0011 | no |
| normalisation / min_df / size scalars (6 trials) | min_df: 3 -> 10 | 0.9081 | +0.0016 | no |
| normalisation / min_df / size scalars (6 trials) | with_size: False -> True | 0.8947 | -0.0118 | no |
| dense embedding vs the histogram it compresses | best graph2vec: d=64, MLP(hidden_layer_sizes=(256,)) | 0.8432 | -0.0870 | no |
| dense embedding vs the histogram it compresses | best wl_svd: d=100, SVM-RBF(C=10.0,gamma=scale) | 0.8957 | -0.0345 | no |
| classifier grid on the histogram | best model: LR(C=0.1) on wl_tfidf | 0.9323 | +0.0021 | - |
| classifier grid on the embedding | best model: MLP(hidden_layer_sizes=(256,),alpha=0.0001) on wl_svd | 0.9311 | +0.0009 | - |
| architecture-aware sample reweighting | best reweighted model: LR(C=1.0)+archw | 0.9239 | -0.0063 | - |

Node labelling against WL depth, the one full grid in the search (CV macro-F1):

| labelling | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|
| class | 0.8216 | 0.7325 | 0.6573 | 0.6506 |
| class_noflag | 0.8216 | 0.7347 | 0.6523 | 0.6443 |
| deg_dom | 0.8772 | 0.7832 | 0.7477 | 0.6818 |
| dom_term | 0.9065 | 0.8307 | 0.7736 | 0.7375 |
| lenbucket | 0.8570 | 0.6301 | 0.6039 | 0.5858 |
| seq_term | 0.8148 | 0.6998 | 0.6781 | 0.6390 |

**Does the structural model beat graph size?** Best CV reached by each representation anywhere in the search:

| representation | n_configs | best_cv_macro_f1 | best_cv_auc | model |
|---|---|---|---|---|
| size_only (the shortcut control) | 4 | 0.7735 | 0.8406 | LR(C=1.0) |
| wl_tfidf histogram | 55 | 0.9323 | 0.9755 | LR(C=0.1) |
| wl_svd | 35 | 0.9311 | 0.9804 | MLP(hidden_layer_sizes=(256,),alpha=0.0001) |
| graph2vec | 9 | 0.8432 | 0.9557 | MLP(hidden_layer_sizes=(256,)) |

**Top 5 configurations by CV macro-F1**, and what they score on test. Only the `committed = yes` row was selected before the test set was opened; the rest were scored afterwards purely to size the CV->test gap, and nothing is chosen from this table.

| construction | representation | model | cv_macro_f1 | cv_bal_acc | cv_auc | test_macro_f1 | test_auc | gap | committed |
|---|---|---|---|---|---|---|---|---|---|
| b20000/w4/calls0/xs0/ext0 | wl_tfidf/dom_term/h1/el0/binary/df3/size0 | LR(C=0.1) | 0.9323 | 0.9296 | 0.9735 | 0.5647 | 0.8265 | 0.3677 | yes |
| b20000/w4/calls0/xs0/ext0 | wl_tfidf/dom_term/h1/el0/binary/df3/size0 | LR(C=0.1,class_weight=balanced) | 0.9314 | 0.9287 | 0.9737 | 0.5722 | 0.8247 | 0.3591 | post hoc |
| b20000/w4/calls0/xs0/ext0 | wl_svd/dom_term/h1/el0/binary/df3/size0/d100 | MLP(hidden_layer_sizes=(256,),alpha=0.01) | 0.9311 | 0.9276 | 0.9785 | 0.5919 | 0.8557 | 0.3393 | post hoc |
| b20000/w4/calls0/xs0/ext0 | wl_svd/dom_term/h1/el0/binary/df3/size0/d100 | MLP(hidden_layer_sizes=(256,),alpha=0.0001) | 0.9311 | 0.9276 | 0.9785 | 0.5919 | 0.8548 | 0.3393 | yes |
| b20000/w4/calls0/xs0/ext0 | wl_tfidf/dom_term/h1/el0/binary/df3/size0 | LR(C=1.0) | 0.9302 | 0.9271 | 0.9744 | 0.5624 | 0.8379 | 0.3678 | post hoc |

**Per architecture**, out-of-fold-threshold rows:

| config | model | x86 n | x86 macro_f1 | x86 rec_ran | x86 rec_good | x64 n | x64 macro_f1 | x64 rec_ran | x64 rec_good |
|---|---|---|---|---|---|---|---|---|---|
| baseline_h2_cap5000 | wl_tfidf/LR | 336 | 0.4784 | 0.6345 | 0.4130 | 153 | 0.3462 | 0.0000 | 1.0000 |
| tuned_best | wl_tfidf/LR | 336 | 0.5105 | 0.6034 | 0.6087 | 153 | 0.4113 | 0.0694 | 0.9753 |
| tuned_best_wl_svd | wl_svd/MLP | 336 | 0.5019 | 0.6069 | 0.5652 | 153 | 0.9064 | 0.8056 | 1.0000 |
| tuned_best_graph2vec | graph2vec/MLP | 336 | 0.5347 | 0.6172 | 0.6739 | 153 | 0.4150 | 0.0694 | 0.9877 |
| size_only_control | size_only/LR | 336 | 0.5997 | 0.6759 | 0.7826 | 153 | 0.3141 | 0.0139 | 0.8148 |

**Per-family ransomware recall**, tuned_best_wl_svd (`wl_svd/MLP`, macro-F1 0.6755). All test families are unseen in training:

| family | recall | correct | support |
|---|---|---|---|
| blackcat | 0.0000 | 0 | 50 |
| avoslocker | 0.1800 | 9 | 50 |
| blackbasta | 0.3667 | 11 | 30 |
| hive | 0.6400 | 32 | 50 |
| blackbyte | 1.0000 | 7 | 7 |
| bianlian | 1.0000 | 11 | 11 |
| bluesky | 1.0000 | 34 | 34 |
| clop | 1.0000 | 45 | 45 |
| holyghost | 1.0000 | 4 | 4 |
| karma | 1.0000 | 13 | 13 |
| lorenz | 1.0000 | 16 | 16 |
| maui | 1.0000 | 3 | 3 |
| playcrypt | 1.0000 | 43 | 43 |
| quantum | 1.0000 | 6 | 6 |

**Hard negatives.** Which goodware the best structural model calls ransomware, and whether graph size explains it:

Best structural row: `tuned_best_wl_svd`.

| goodware bucket | n | false_positives | fp_rate | median_blocks |
|---|---|---|---|---|
| everyday | 71 | 17 | 0.2394 | 15759.0000 |
| hard_negative | 29 | 3 | 0.1034 | 17157.0000 |
| system | 27 | 0 | 0.0000 | 2883.0000 |

Projects contributing false positives:

| project | n | false_positives | median_blocks |
|---|---|---|---|
| steam | 9 | 8 | 4830.0000 |
| irfanview | 17 | 3 | 17613.0000 |
| bitwarden | 3 | 3 | 5563.0000 |
| zotero | 27 | 2 | 12750.0000 |
| eraser | 2 | 2 | 17665.5000 |
| 7zip | 6 | 1 | 20000.0000 |
| gitextensions | 11 | 1 | 17695.0000 |

Median blocks: false positives 5563 (n=20), correctly kept goodware 14283 (n=107); at the block cap 0.00 vs 0.18. Rank separation of the false positives from the rest of the test goodware by size alone (AUC, 0.5 = size says nothing): n_blocks 0.329, n_edges 0.322, insns 0.315. So size does not explain the false positives, and not in the direction the block cap would suggest either: an AUC below 0.5 means the misclassified goodware is *smaller* than the goodware the model keeps, the opposite of 'big binaries look like ransomware'. The false positives concentrate by project instead -- see the table above -- which points at what the code is, not how much of it there is.


### 7.3 Reproducing

```
python graph2vec_pipeline/graph_cache.py --max-blocks 20000
python graph2vec_pipeline/graph_cache.py --max-blocks 20000 --windows 4
python graph2vec_pipeline/tune.py --dataset both
python graph2vec_pipeline/final_eval.py --dataset both
python graph2vec_pipeline/make_summary.py
```
Seed 42 throughout; the WL relabelling uses a fixed 64-bit mixer and no RNG, the graph cache is a pure function of the disassembly, and PV-DBOW seeds its SGD from `np.random.default_rng(42)`. `config_used.yaml` in each tuned directory records exactly what was fitted.


**Reproducibility.** The selected configuration was refitted from scratch in a separate process (`final_eval.py --repro-check`) -- graph assembly, WL relabelling, the min_df vocabulary, IDF, SVD, scaler and classifier -- and compared against `predictions.csv` sample by sample:

| dataset | test samples | max abs score difference | identical predictions | macro-F1 run 1 | macro-F1 run 2 |
|---|---|---|---|---|---|
| mendeley | 491 | 1.11e-16 | yes | 0.7025 | 0.7025 |
| balanced | 489 | 1.11e-16 | yes | 0.5647 | 0.5647 |


- PV-DBOW (the graph2vec rows) is fitted once on the whole TRAIN split rather than per CV fold. It never sees a label, but a held-out fold's graphs did contribute to its label vectors, so its CV score is mildly optimistic relative to the sparse rows.

- The WL vocabulary (min_df) is taken over the whole TRAIN split; IDF, SVD, scaling and the classifier are refitted per fold.

- WL relabelling here is the hashed variant (order-independent sum of mixed neighbour hashes) rather than the sorted-tuple hash used by graph2vec_pipeline/cfg.wl_labels; same feature map up to 64-bit collisions, ~100x faster on 20M nodes.

- `posthoc_top5` scores the five highest-CV configurations of the search on test. Only the ones that also appear in `results` were committed to before the test set was opened; the rest exist to size the CV->test gap and must not be read as a result.

- The `graph2vec` (PV-DBOW) rows are the one part of this table that does **not** reproduce exactly. Its SGD stops on a wall-clock budget (`max_seconds`, checked every 200 steps), so a busier machine takes fewer steps and lands on different document vectors from the same seed. Two runs of the `mendeley` PV-DBOW pick a few hours apart gave test macro-F1 0.617 and 0.486 at argmax. Everything else here -- the graph build, the WL relabelling, the vocabulary, IDF, SVD and every classifier -- is seed-determined and did reproduce to the last bit (see the table above). Read the PV-DBOW row as a range, not a number, and do not rank it against the others on a single run.
