# CNN-ViT (HierarchicalMalwareNet) on the shared cohort

Produced by `cnn_vit_pipeline/train_eval.py` on 2026-09-13 from images rendered
by the unmodified `asm_parser.py`. Two datasets, the same model, the same
protocol, the same seeds - only the goodware corpus differs.

| | dataset 1 `mendeley` | dataset 2 `balanced` |
|---|---|---|
| ransomware | Mendeley, `mal_train` / `mal_test`, family-disjoint | identical to dataset 1 |
| goodware | Mendeley, `good_train` / `good_test` | Goodware_Balanced, split by source project (expB's own groups) |
| test fold | 129 goodware + 362 ransomware | 127 goodware + 362 ransomware |

Everything below is the **test fold**, scored once, with the best-validation-loss
checkpoint restored.

Sections 1-7 are the **untuned** run at seed 1337; section 8 is its 5-seed
sweep; **section 9 is the tuned run** (112 configurations selected by group CV
on the train split only). Do not quote a single-seed number from section 1
without section 8's spread or section 9's before/after.

---

## 1. Headline

| metric | `mendeley` | `balanced` |
|---|---:|---:|
| accuracy | 0.5764 | 0.5726 |
| **majority-class accuracy** | **0.7373** | **0.7403** |
| balanced accuracy | 0.7052 | 0.5835 |
| macro F1 | 0.5747 | 0.5422 |
| macro precision | 0.6810 | 0.5643 |
| macro recall | 0.7052 | 0.5835 |
| ROC AUC | 0.8550 | 0.6221 |
| recall goodware | 0.9767 | 0.6063 |
| precision goodware | 0.3807 | 0.3263 |
| F1 goodware | 0.5478 | 0.4242 |
| recall ransomware | 0.4337 | 0.5608 |
| precision ransomware | 0.9812 | 0.8024 |
| F1 ransomware | 0.6015 | 0.6602 |
| false positive rate | 0.0233 | 0.3937 |

| confusion matrix | `mendeley` | `balanced` |
|---|---:|---:|
| TN (goodware called goodware) | 126 | 77 |
| FP | 3 | 50 |
| FN | 205 | 159 |
| TP (ransomware called ransomware) | 157 | 203 |

**Read the second row first.** On both datasets the model's raw accuracy is
*below* the majority-class baseline - always answering "ransomware" would score
0.7373 and 0.7403. The model is not useless (ROC AUC 0.855 on `mendeley` says
its score ranks samples far better than chance), but at its own decision
threshold it is worse than a constant. Balanced accuracy and macro F1, not
accuracy, are the numbers to quote.

The two datasets fail in opposite directions, which is informative:

* `mendeley` is **conservative**: it almost never calls goodware ransomware
  (FPR 0.023, ransomware precision 0.981) and pays for it by missing 205 of
  362 ransomware samples. Its training goodware and its test goodware come from
  the same corpus and look alike, so "not-goodware" is a call it makes
  reluctantly.
* `balanced` is **trigger-happy**: FPR 0.394. Its goodware comes from a
  different corpus with a different architecture mix (train 859 x64 / 939 x86
  against `mendeley`'s 458 / 1341), and its test goodware includes the
  `hard_negative` bucket - packers, crypto tools, archivers - deliberately
  chosen to look like malware. It does.

---

## 2. Per architecture

| dataset | arch | n | accuracy | bal. acc | recall goodware | recall ransomware | FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| mendeley | x86 | 407 | 0.6437 | 0.7424 | 0.9744 | 0.5103 | 0.0256 |
| mendeley | x64 | 84 | 0.2500 | 0.5625 | 1.0000 | 0.1250 | 0.0000 |
| balanced | x86 | 336 | 0.6250 | 0.4444 | 0.1957 | 0.6931 | 0.8043 |
| balanced | x64 | 153 | 0.4575 | 0.4336 | 0.8395 | 0.0278 | 0.1605 |

Architecture is the largest single effect in these results, and it points in
opposite directions per dataset:

* **`mendeley`**: x64 ransomware recall collapses to 0.125 (9 of 72) against
  0.510 on x86. The training ransomware is overwhelmingly x86, so x64
  ransomware in the test fold is close to out of distribution.
* **`balanced`**: the model has effectively learned "x86 -> ransomware, x64 ->
  goodware". x86 goodware recall is 0.196 (9 of 46) while x64 goodware recall
  is 0.840 (68 of 81); x64 ransomware recall is 0.028 (2 of 72). Balanced
  accuracy is about 0.43 *within each architecture* - at or below chance once
  architecture is held fixed. Its headline 0.583 balanced accuracy is carried
  by the architecture correlation, not by code structure.

That is worth keeping as a measurement: the balanced goodware corpus is far
more x64-heavy than the Mendeley ransomware, so architecture is a shortcut
feature, and this run shows the CNN-ViT taking it.

---

## 3. Per ransomware test family

All 14 test families are disjoint from the 24 training families. Recall only -
these rows contain no goodware.

| family | support | `mendeley` | `balanced` |
|---|---:|---:|---:|
| blackcat | 50 | **1.000** (50/50) | **0.980** (49/50) |
| bluesky | 34 | **0.971** (33/34) | **1.000** (34/34) |
| playcrypt | 43 | 0.744 (32/43) | 0.628 (27/43) |
| blackbasta | 30 | 0.533 (16/30) | 0.467 (14/30) |
| hive | 50 | 0.300 (15/50) | 0.180 (9/50) |
| lorenz | 16 | 0.250 (4/16) | **1.000** (16/16) |
| clop | 45 | 0.156 (7/45) | **0.867** (39/45) |
| maui | 3 | 0.000 (0/3) | 1.000 (3/3) |
| holyghost | 4 | 0.000 (0/4) | 0.250 (1/4) |
| karma | 13 | 0.000 (0/13) | 0.231 (3/13) |
| avoslocker | 50 | 0.000 (0/50) | 0.140 (7/50) |
| quantum | 6 | 0.000 (0/6) | 0.167 (1/6) |
| bianlian | 11 | 0.000 (0/11) | 0.000 (0/11) |
| blackbyte | 7 | 0.000 (0/7) | 0.000 (0/7) |

Highlights:

* **Two families carry both runs.** blackcat and bluesky are 84 of the 362 test
  samples (23%) and are detected almost perfectly by both. Remove them and
  `mendeley`'s ransomware recall drops from 0.434 to 0.266 (74 of 278).
* **Seven families are never detected at all on `mendeley`** - maui, holyghost,
  karma, avoslocker, quantum, bianlian, blackbyte: 0 of 94 samples. Generalising
  to an unseen family is where this representation breaks, which is exactly what
  the family-disjoint split was built to expose and what a random split hides.
* **clop and lorenz flip.** `balanced` gets 39/45 clop and 16/16 lorenz where
  `mendeley` gets 7/45 and 4/16. Both are x86-dominated families, so this is the
  x86 -> ransomware bias of section 2 collecting points, not better family
  generalisation. Read the `balanced` family column as the architecture prior,
  not as evidence the model learned clop.
* Night Sky and Thanos are absent: the cohort filter removes them, leaving 24
  train / 14 test families out of the raw corpus's 25 / 15.

---

## 4. What the cohort and split changed versus the original yanping protocol

| | original (`stratified_split.py` + `model_train.py`) | this run |
|---|---|---|
| split | random shuffle inside each class folder | Mendeley's own `good_train`/`mal_train` against `good_test`/`mal_test` |
| ransomware families | the same family on both sides | **disjoint**: 24 train, 14 test, asserted in `tests/test_cohort_split.py` |
| duplicates | the same bytes under two filenames, on both sides | one row per sha256 |
| .NET assemblies | included | excluded (`tag:dotnet`) |
| packed / UPX | included while packed | excluded, or the UPX-unpacked bytes used |
| Thanos, Night Sky | included | excluded |
| validation fold | a random slice of train | fixed seed 1337, 10% of train, group-aware: no family and no goodware project in both train and val |
| goodware (dataset 2) | not attempted | expB's project-grouped balanced-goodware split, intersected with the cohort |

Group overlap between every pair of folds is empty in both datasets
(`metrics.json` -> `samples.group_overlap`), so nothing seen in training
reappears in val or test under another name.

**What this costs, and why it is the right cost.** A random per-class split lets
the model train on one conti sample and be scored on a near-identical conti
sample; that measures memorisation, and it scores very well. Under the
family-disjoint cohort split the same model, the same code and the same
hyper-parameters land below the majority-class baseline. The distance between
those two numbers *is* the finding: the original protocol was answering a much
easier question than "will this flag a ransomware family nobody has seen
before".

The cohort filter also removes the two shortcuts that made the old numbers
unsafe to compare - packed samples, whose 256x256 image is largely a picture of
the packer, and .NET assemblies, whose image is largely IL metadata. The
shortcut that remains, the architecture correlation in dataset 2, is visible in
section 2 rather than hidden.

No cohort-faithful CNN-ViT number existed before this run, so the comparison
above is between protocols, not between two measured accuracies.

---

## 5. Caveat: these images come from a different extractor

**The images scored here were rendered from the revised, skip-data extractor's
disassembly, not from the `asm_parse.py` linear sweep the original CNN-ViT
pipeline used.** [`asm_tool/README.md` section 6](../../asm_tool/README.md) has
the measured difference: over the 1,342 comparable goodware binaries the linear
sweep yields 37.8 M instructions against the skip-data extractor's 65.9 M
(57.3%), and 41% of the files that did not hit the instruction cap produce under
half the skip-data count.

The substitution was necessary, not preferred:

* the `asm_parse.py`-format ransomware trees do not exist on this host - those
  binaries are VM-only; and
* the host Mendeley-goodware `asm_parse.py` tree was built from a copy in which
  67 UPX files are still packed, so its sha256s do not match the cohort CSVs and
  only 1,023 of 1,114 in-cohort goodware-train rows could be joined at all.

The revised extractor has output for every class and every set, keyed by
sha256, so `asm_tool/unified_to_asm.py` re-shapes it and the *same, unmodified*
`asm_parser.py` renders it. Both classes and both corpora pass through the same
extractor, so the comparisons inside this document are internally consistent.

What that means for comparisons:

* **Safe**: `mendeley` against `balanced` here; the per-arch and per-family
  blocks here; these numbers against the tokenization and EMBER pipelines,
  which score the same cohort and the same split.
* **Not safe**: any of these against an older CNN-ViT figure produced from
  `asm_parse.py` output. The input representation differs on top of the split.
* Longer streams are not automatically an advantage here: `asm_parser.py`'s
  canvas holds 65,536 tokens, about 21,845 instructions, so past that length the
  extra instructions are truncated and the only effect is *which* first ~21,845
  instructions get drawn.

---

## 6. Run cost and reproduction

Device: **NVIDIA GeForce RTX 5080 (sm_120), torch 2.11.0+cu128**, in a dedicated
Python 3.12 venv at `C:/Users/chaoa/Downloads/cnn_vit_venv`. The system Python
3.14 has only a CPU build of torch; the 5080 is Blackwell and needs a cu128 or
newer wheel, which exists for 3.12 but not for 3.14.

| | `mendeley` | `balanced` |
|---|---:|---:|
| epochs requested / run | 80 / **13** | 80 / **13** |
| stopped by | early stopping on val loss, patience 8 | early stopping on val loss, patience 8 |
| seconds per epoch (median) | 3.79 | 4.14 |
| seconds per epoch (mean) | 3.81 | 4.34 |
| training seconds total | 49.5 | 56.4 |
| wall clock incl. final eval | 52.8 s | 59.6 s |
| train / val / test | 1799 / 219 / 491 | 1798 / 219 / 489 |

No epoch cap was needed. Both runs early-stopped at epoch 13 of a requested 80,
in under a minute each. The CPU fallback was never exercised; at the ~62
ms/sample/epoch measured earlier on this host it would have been roughly 2
minutes per epoch, so about 30 minutes per run - still inside budget, just
slower.

Conversion and image counts - all exact, nothing lost:

| stage | mendeley | balanced goodware |
|---|---:|---:|
| manifest rows with `disassembled == 1` | 2,598 | 1,343 |
| `.asm` written by `unified_to_asm.py` | 2,598 | 1,343 |
| PNG + ViT-mask pairs rendered | 2,598 | 1,343 |
| cohort rows with an image | 2,509 / 2,509 | 2,506 / 2,506 |

`.skip` markers removed: 313,768 (mendeley) and 56,143 (balanced). Files hitting
the 100,000-instruction cap: 544 and 420. Zero unrecognised lines, zero PNGs
that failed to map to a sha256, zero cohort rows without an image, zero
tree-label-versus-cohort-label disagreements.

Reproduction, in order:

    # 1. revised extractor -> asm_parse-shaped .asm trees
    python asm_tool/unified_to_asm.py --extract ".../Shared/Extract" \
        --out "C:/Users/chaoa/Downloads/asm_output/unified_mendeley"
    python asm_tool/unified_to_asm.py --extract ".../Shared/Extract_Goodware_Balanced" \
        --out "C:/Users/chaoa/Downloads/asm_output/unified_goodware_balanced"

    # 2. images, with the UNMODIFIED asm_parser.py, one run per set so the
    #    --default-class assigns the right class
    python asm_parser.py --asm-dir ".../unified_mendeley/good_train" \
        --out-dir ".../cnn_vit_images/unified_mendeley" \
        --labels-csv <scratch>/labels_good_train.csv --default-class 0
    #   likewise good_test (0), mal_train (1), mal_test (1),
    #   and the whole balanced tree (0)

    # 3. split + hard-linked train/val/test tree
    python cnn_vit_pipeline/build_dataset.py --dataset mendeley \
        --out "C:/Users/chaoa/Downloads/cnn_vit_data/mendeley" --clean
    python cnn_vit_pipeline/build_dataset.py --dataset balanced \
        --out "C:/Users/chaoa/Downloads/cnn_vit_data/balanced" --clean

    # 4. train and score
    python cnn_vit_pipeline/train_eval.py \
        --data "C:/Users/chaoa/Downloads/cnn_vit_data/mendeley" --dataset mendeley \
        --epochs 80 --batch-size 16 --seed 1337 --device cuda \
        --ckpt-dir "C:/Users/chaoa/Downloads/cnn_vit_models/mendeley"

## 7. Files

    results/cnn_vit/<dataset>/metrics.json          expA schema + per_architecture + per_family_recall + history
    results/cnn_vit/<dataset>/splits.csv            sha256, file, source, label, group, fold, split, arch, family
    results/cnn_vit/<dataset>/training_log.csv      epoch, lr, train_loss, val_loss, val_acc, seconds
    results/cnn_vit/<dataset>/test_predictions.csv  per-sample y_true / y_pred / score
    results/cnn_vit/<dataset>/config_used.yaml      the configuration this run used

Model weights are deliberately **not** under `results/` - they are binary. They
live in `C:/Users/chaoa/Downloads/cnn_vit_models/<dataset>/best_model.pth`, and
the intermediate `.asm`, image and hard-linked dataset trees live under
`C:/Users/chaoa/Downloads/{asm_output,cnn_vit_images,cnn_vit_data}/`.

## 8. Seed sweep (5 seeds, GPU, same protocol)

Single-seed numbers above are not stable. Five additional seeds (1-5) per dataset, identical protocol, `results/cnn_vit/seeds/<dataset>/seed<N>/`:

| dataset | metric | mean | sd | min | max |
|---|---|---|---|---|---|
| mendeley | accuracy | 0.679 | 0.055 | 0.631 | 0.766 |
| mendeley | balanced accuracy | 0.686 | 0.049 | 0.597 | 0.727 |
| mendeley | macro-F1 | 0.628 | 0.018 | 0.604 | 0.659 |
| mendeley | ROC-AUC | 0.802 | 0.074 | 0.710 | 0.881 |
| mendeley | ransomware recall | 0.671 | 0.170 | 0.536 | 0.953 |
| mendeley | goodware recall | 0.701 | 0.268 | 0.240 | 0.915 |
| mendeley | majority-class accuracy | 0.737 | | | |
| balanced | accuracy | 0.516 | 0.132 | 0.356 | 0.751 |
| balanced | balanced accuracy | 0.593 | 0.067 | 0.509 | 0.709 |
| balanced | macro-F1 | 0.502 | 0.113 | 0.352 | 0.695 |
| balanced | ROC-AUC | 0.645 | 0.024 | 0.612 | 0.678 |
| balanced | ransomware recall | 0.433 | 0.202 | 0.191 | 0.796 |
| balanced | goodware recall | 0.754 | 0.070 | 0.622 | 0.827 |
| balanced | majority-class accuracy | 0.740 | | | |

Reading: on Mendeley the ranking is real (AUC 0.71-0.88) but the operating point is unstable and macro-F1 stays at 0.60-0.66. On Balanced the model is at chance within each architecture; seed-to-seed spread of macro-F1 is 0.35-0.69, so no single Balanced number should be quoted. Any comparison against the tokenization pipeline should use the seed means with their spread, not the seed-1337 run in section 3.

---

## 9. Tuned

`cnn_vit_pipeline/tuned_train.py` re-runs the same model on the same split with
every knob exposed - input encoding, optimisation, loss, sampling, capacity,
regularisation, decision threshold. **Nothing here chose anything on the test
fold.** Selection is 3-fold `StratifiedGroupKFold` over the **train split
only** (whole ransomware families and whole goodware projects held out
together), with a *further* group-aware inner val fold carved out of each
CV-train part for early stopping, so the fold being scored never picked the
checkpoint that scores it. Every configuration tried is in
`results/cnn_vit/tuned/<dataset>/cv_search.csv`. Only after the search closed
did `final` train on the whole train split and score the test fold, once per
seed.

**112 configurations** (56 per dataset, 3.4 h of GPU cross-validation in
total). The first 35 per dataset were a one-factor-at-a-time sweep around the
untuned recipe; a `combine` stage (16 per dataset) then crossed the per-axis
winners, because the one-factor sweep had left the encoding winner un-combined
with anything - on `balanced` its best encoding was never paired with a single
optimisation setting. A final `cvseed` stage re-cross-validated the base and
the top three at a **second CV fold-split seed**, and one `ampoff` run prices
the harness's bf16 against the published run's fp32.

### 9.1 What the search found


**`mendeley`** - 56 configurations, 3-fold group CV on the train split (2018 samples), CV seed 1337. Base configuration CV macro-F1 **0.7274**.

| # | config | stage | CV macro-F1 | CV seeds | vs base | CV bal.acc | CV AUC | differs from base by |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `combine_005` | combine | 0.7374 ± 0.0480 | 2 | +0.0100 | 0.7355 | 0.8021 | `depth=3` `dropout=0.3` `weight_decay=0.1` `width=16` |
| 2 | `optim_013` | optim | 0.7373 | 1 | +0.0099 | 0.7376 | 0.8072 | `epochs=150` `patience=20` |
| 3 | `optim_012` | optim | 0.7373 | 1 | +0.0099 | 0.7376 | 0.8072 | `schedule=plateau` |
| 4 | `encoding_001` | encoding | 0.7337 ± 0.0090 | 2 | +0.0064 | 0.7358 | 0.8005 | _(base)_ |
| 5 | `combine_003` | combine | 0.7332 | 1 | +0.0058 | 0.7311 | 0.7707 | `dropout=0.3` `weight_decay=0.1` |
| 6 | `combine_001` | combine | 0.7328 | 1 | +0.0055 | 0.7307 | 0.8103 | `label_smoothing=0` `schedule=plateau` |
| 7 | `optim_015` | optim | 0.7328 | 1 | +0.0055 | 0.7307 | 0.8103 | `label_smoothing=0` |
| 8 | `combine_008` | combine | 0.7286 | 1 | +0.0012 | 0.7287 | 0.8011 | `depth=3` `dropout=0.3` `schedule=plateau` `weight_decay=0.1` `width=16` |
| 9 | `optim_010` | optim | 0.7284 ± 0.0139 | 2 | +0.0010 | 0.7278 | 0.7704 | `weight_decay=0.1` |
| 10 | `capacity_025` | capacity | 0.7279 | 1 | +0.0006 | 0.7263 | 0.7734 | `dropout=0.3` |

Per-axis effect at CV seed 1337 - every row that changed only knobs belonging to that axis, against the base:

| axis | n | best delta | worst delta | mean delta |
|---|---:|---:|---:|---:|
| encoding | 5 | -0.0019 | -0.1013 | -0.0521 |
| optimisation | 24 | +0.0109 | -0.0724 | -0.0255 |
| loss / sampling | 5 | -0.0054 | -0.1072 | -0.0715 |
| regularisation / capacity | 8 | +0.0006 | -0.0726 | -0.0364 |
| decision rule | 2 | -0.0257 | -0.0257 | -0.0257 |
| combinations of two or more axes | 7 | +0.0439 | -0.0921 | -0.0192 |

**`balanced`** - 56 configurations, 3-fold group CV on the train split (2017 samples), CV seed 1337. Base configuration CV macro-F1 **0.7523**.

| # | config | stage | CV macro-F1 | CV seeds | vs base | CV bal.acc | CV AUC | differs from base by |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `encoding_002` | encoding | 0.8100 | 1 | +0.0577 | 0.8108 | 0.8641 | `variant=stride3` |
| 2 | `combine_012` | combine | 0.8097 | 1 | +0.0575 | 0.8086 | 0.8707 | `depth=3` `variant=stride3` |
| 3 | `encoding_004` | encoding | 0.8064 | 1 | +0.0541 | 0.8050 | 0.8490 | `variant=mnem1s` |
| 4 | `combine_005` | combine | 0.8064 | 1 | +0.0541 | 0.8027 | 0.8379 | `lr=0.0001` `variant=stride3` |
| 5 | `combine_004` | combine | 0.8063 ± 0.0073 | 2 | +0.0540 | 0.8046 | 0.8306 | `lr=0.001` `variant=stride3` |
| 6 | `combine_010` | combine | 0.8049 | 1 | +0.0526 | 0.8022 | 0.8494 | `depth=3` `dropout=0.3` `variant=stride3` `weight_decay=0.001` `width=16` |
| 7 | `optim_006` | optim | 0.8034 | 1 | +0.0511 | 0.7995 | 0.8115 | `lr=0.003` |
| 8 | `combine_003` | combine | 0.8007 | 1 | +0.0484 | 0.7971 | 0.8446 | `depth=3` `dropout=0.3` `variant=stride3` `width=16` |
| 9 | `combine_013` | combine | 0.7987 | 1 | +0.0465 | 0.7972 | 0.8598 | `variant=stride3` `width=16` |
| 10 | `capacity_030` | capacity | 0.7980 | 1 | +0.0457 | 0.7940 | 0.8518 | `depth=3` `dropout=0.3` `width=16` |

Per-axis effect at CV seed 1337 - every row that changed only knobs belonging to that axis, against the base:

| axis | n | best delta | worst delta | mean delta |
|---|---:|---:|---:|---:|
| encoding | 5 | +0.0577 | +0.0094 | +0.0404 |
| optimisation | 15 | +0.0511 | -0.0016 | +0.0231 |
| loss / sampling | 5 | +0.0310 | -0.1166 | -0.0098 |
| regularisation / capacity | 8 | +0.0457 | -0.0229 | +0.0043 |
| decision rule | 2 | -0.0618 | -0.0639 | -0.0629 |
| combinations of two or more axes | 16 | +0.0898 | -0.1232 | +0.0205 |

### 9.2 Read the second CV seed first

The `cvseed` rows are the most important result in this section. On `mendeley`
the configuration that led the search at fold-split seed 1337 (`combine_005`,
CV 0.7713) scored 0.7035 at seed 7 - **0.7374 +/- 0.0480** over the two. The
base configuration itself is **0.7337 +/- 0.0090**. The whole top ten spans
+0.001 to +0.010 over the base, while a single configuration's own spread
across two fold splits is 0.009-0.048. *On `mendeley`, 56 configurations found
nothing that survives changing the fold split.*

`balanced` is better but not clean either: its base scores 0.7523 at seed 1337
and **0.8014** at seed 7, so the base's own two-seed mean (0.777) is most of
the way to the `stride3` winner's 0.810. The one configuration measured at both
seeds, `stride3` + `lr=1e-3`, is stable (0.8063 +/- 0.0073), which is the main
reason to believe the encoding effect at all.

The bf16 control: `amp=off` scores 0.6770 on `mendeley` against the base's
0.7274, and 0.7529 on `balanced` against 0.7523. On `balanced` the numerics are
neutral; on `mendeley` the swing is as large as anything the search found -
another way of saying the same thing. At three folds and one fold-split seed
the CV estimate does not resolve differences of a point.

### 9.3 CV -> test gap (post hoc)

The top-3 CV configurations were also scored on the test fold, one seed each,
**after** selection was finished. These rows chose nothing; they exist to
measure how far the CV number over-states test.


**`mendeley`**

| CV rank | config | CV macro-F1 | test macro-F1 (seed 1) | test - CV | test AUC | differs from base by |
|---:|---|---:|---:|---:|---:|---|
| 1 | `combine_005` | 0.7374 (mean of 2 CV seeds) | 0.6794 | -0.0580 | 0.8890 | `depth=3` `dropout=0.3` `weight_decay=0.1` `width=16` |
| 2 | `optim_013` | 0.7373 | 0.6651 | -0.0721 | 0.8543 | `epochs=150` `patience=20` |
| 3 | `optim_012` | 0.7373 | 0.6651 | -0.0721 | 0.8543 | `schedule=plateau` |

**`balanced`**

| CV rank | config | CV macro-F1 | test macro-F1 (seed 1) | test - CV | test AUC | differs from base by |
|---:|---|---:|---:|---:|---:|---|
| 1 | `encoding_002` | 0.8100 | 0.6631 | -0.1469 | 0.6560 | `variant=stride3` |
| 2 | `combine_012` | 0.8097 | 0.5770 | -0.2328 | 0.6843 | `depth=3` `variant=stride3` |
| 3 | `encoding_004` | 0.8064 | 0.3487 | -0.4577 | 0.6621 | `variant=mnem1s` |

The gap is large, always negative, and **not rank-preserving**. On `balanced`
the third-ranked CV configuration (`mnem1s`, CV 0.806) is the *worst* on test
(0.349); the CV ordering does not transfer. That is the expected shape given
what the split does - the test fold holds 14 ransomware families none of which
appear in training, and on `balanced` the test goodware also carries the
`hard_negative` bucket - but it means CV over the train split is a weak
selection signal here, not merely a pessimistic one.

### 9.4 Before / after

Test fold, mean ± sd over 5 seeds each. `untuned` is the published `train_eval.py` sweep in `results/cnn_vit/seeds/`; `base (harness)` is the *same recipe* re-run through the tuning harness (bf16, GPU-resident batching) and is the configuration every search row was varied from.

| metric | `mendeley` untuned | `mendeley` base (harness) | `mendeley` tuned | `balanced` untuned | `balanced` base (harness) | `balanced` tuned |
|---|---:|---:|---:|---:|---:|---:|
| **macro F1** | 0.628 ± 0.020 | 0.660 ± 0.096 | 0.655 ± 0.049 | 0.502 ± 0.126 | 0.440 ± 0.091 | 0.518 ± 0.106 |
| balanced accuracy | 0.686 ± 0.055 | 0.738 ± 0.069 | 0.744 ± 0.024 | 0.593 ± 0.075 | 0.557 ± 0.058 | 0.620 ± 0.060 |
| accuracy | 0.679 ± 0.062 | 0.675 ± 0.105 | 0.670 ± 0.061 | 0.516 ± 0.147 | 0.445 ± 0.095 | 0.530 ± 0.121 |
| ROC AUC | 0.802 ± 0.082 | 0.858 ± 0.031 | 0.865 ± 0.021 | 0.645 ± 0.027 | 0.575 ± 0.115 | 0.665 ± 0.025 |
| recall goodware | 0.701 ± 0.300 | 0.871 ± 0.053 | 0.899 ± 0.090 | 0.754 ± 0.078 | 0.792 ± 0.036 | 0.808 ± 0.069 |
| recall ransomware | 0.671 ± 0.190 | 0.606 ± 0.148 | 0.588 ± 0.111 | 0.433 ± 0.225 | 0.323 ± 0.136 | 0.432 ± 0.187 |
| false positive rate | 0.299 ± 0.300 | 0.129 ± 0.053 | 0.101 ± 0.090 | 0.246 ± 0.078 | 0.208 ± 0.036 | 0.192 ± 0.069 |
| **majority-class accuracy (floor)** | **0.737** | **0.737** | **0.737** | **0.740** | **0.740** | **0.740** |

Per architecture (test fold, mean ± sd over 5 seeds). `balanced accuracy` here is *within* the architecture, so 0.5 is chance no matter how the architecture mix is skewed.

| dataset | arch | n | untuned bal.acc | base (harness) bal.acc | tuned bal.acc | untuned macro-F1 | base (harness) macro-F1 | tuned macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `mendeley` | x64 | 84 | 0.460 ± 0.198 | 0.617 ± 0.170 | 0.589 ± 0.109 | 0.295 ± 0.097 | 0.383 ± 0.256 | 0.297 ± 0.206 |
| `mendeley` | x86 | 407 | 0.733 ± 0.067 | 0.772 ± 0.053 | 0.787 ± 0.029 | 0.697 ± 0.056 | 0.717 ± 0.070 | 0.727 ± 0.034 |
| `balanced` | x64 | 153 | 0.498 ± 0.016 | 0.488 ± 0.004 | 0.516 ± 0.022 | 0.393 ± 0.054 | 0.349 ± 0.013 | 0.401 ± 0.059 |
| `balanced` | x86 | 336 | 0.489 ± 0.085 | 0.443 ± 0.065 | 0.527 ± 0.053 | 0.422 ± 0.130 | 0.356 ± 0.092 | 0.438 ± 0.105 |

Per ransomware test family, recall (mean ± sd over 5 seeds). All 14 families are disjoint from the 24 training families.

| family | support | `mendeley` untuned | `mendeley` tuned | `balanced` untuned | `balanced` tuned |
|---|---:|---:|---:|---:|---:|
| avoslocker | 50 | 0.436 ± 0.517 | 0.144 ± 0.131 | 0.244 ± 0.370 | 0.464 ± 0.404 |
| blackcat | 50 | 1.000 ± 0.000 | 0.980 ± 0.045 | 0.640 ± 0.410 | 0.340 ± 0.444 |
| hive | 50 | 0.476 ± 0.295 | 0.368 ± 0.314 | 0.216 ± 0.153 | 0.136 ± 0.159 |
| clop | 45 | 0.622 ± 0.321 | 0.587 ± 0.141 | 0.760 ± 0.138 | 0.893 ± 0.081 |
| playcrypt | 43 | 0.926 ± 0.069 | 0.916 ± 0.058 | 0.423 ± 0.379 | 0.605 ± 0.244 |
| bluesky | 34 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.565 ± 0.425 | 0.465 ± 0.457 |
| blackbasta | 30 | 0.740 ± 0.205 | 0.687 ± 0.183 | 0.480 ± 0.240 | 0.607 ± 0.293 |
| lorenz | 16 | 0.762 ± 0.218 | 0.637 ± 0.252 | 0.475 ± 0.311 | 0.025 ± 0.056 |
| karma | 13 | 0.154 ± 0.109 | 0.154 ± 0.094 | 0.292 ± 0.126 | 0.431 ± 0.259 |
| bianlian | 11 | 0.291 ± 0.398 | 0.145 ± 0.177 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| blackbyte | 7 | 0.286 ± 0.416 | 0.200 ± 0.313 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| quantum | 6 | 0.033 ± 0.075 | 0.000 ± 0.000 | 0.167 ± 0.000 | 0.233 ± 0.149 |
| holyghost | 4 | 0.300 ± 0.274 | 0.100 ± 0.224 | 0.050 ± 0.112 | 0.150 ± 0.137 |
| maui | 3 | 0.800 ± 0.447 | 0.800 ± 0.447 | 1.000 ± 0.000 | 0.400 ± 0.548 |

Chosen for `mendeley`: encoding `head3`, `lr=0.0003`, `weight_decay=0.1`, `batch_size=16`, `schedule=cosine`, `label_smoothing=0.15`, `loss=ce`, `sampler=class`, width 16 depth 3 dropout 0.3, decision `argmax`.

Chosen for `balanced`: encoding `stride3`, `lr=0.0003`, `weight_decay=0.01`, `batch_size=16`, `schedule=cosine`, `label_smoothing=0.15`, `loss=ce`, `sampler=class`, width 32 depth 6 dropout 0.15, decision `argmax`.

### 9.5 What actually moved the needle

* **Encoding, on `balanced` only.** It is the one axis whose every level beat
  the base (+0.009 to +0.058 CV), and the chosen `stride3` - sampling
  instructions on an even stride over the whole file instead of taking the
  first 21,846 - is the single largest effect in the table. 61% of the balanced
  goodware files overflow the canvas, so `head3` was only ever shown their
  first fifth. On `mendeley` every encoding variant *lost* (-0.002 to -0.101):
  its files are shorter, and the head of the file is apparently the informative
  part.
* **Optimisation** produced the best single-knob deltas on both datasets
  (`weight_decay=0.1` on `mendeley`, `lr=3e-3` on `balanced`), but both sit
  inside the fold-split noise measured in 9.2.
* **Loss and sampling made things worse.** Focal loss, natural sampling and the
  architecture-balanced `cell` sampler all lost on `mendeley` (mean -0.072),
  and `cell` lost 0.117 on `balanced`. The inverse-frequency
  `WeightedRandomSampler` the original pipeline already used is the right
  choice.
* **The fitted decision threshold lost on both** (-0.026 `mendeley`, -0.063
  `balanced`) once it was scored *nested* - each fold's threshold fitted on the
  other folds only. Both final models therefore use plain `argmax` at 0.5. A
  threshold refitted on all out-of-fold scores looks better only because the
  fold it scores helped choose it.
* **Nothing cleared the floor.** Majority-class accuracy is 0.737 / 0.740.
  Tuned accuracy is 0.670 +/- 0.061 and 0.530 +/- 0.121. Both models are still
  worse than answering "ransomware" every time, exactly as in section 1.

The honest summary of the before/after table: on `mendeley` **tuning bought
nothing** - 0.655 +/- 0.049 macro-F1 against the same recipe's 0.660 +/- 0.096
through the same harness. What moved (0.628 -> 0.660) was the harness itself
(bf16, and a different RNG consumption order), not a hyper-parameter. On
`balanced` the tuned model gains +0.078 macro-F1 over the harness base and
+0.016 over the published untuned sweep, against a seed spread of +/- 0.106 -
real in the CV sense, not resolvable in the test sense. ROC-AUC is the one
metric that moves consistently: 0.802 -> 0.865 on `mendeley` and 0.645 ->
0.665 on `balanced`. The tuned models *rank* a little better even though their
operating point does not improve.

### 9.6 Does the gain survive inside each architecture?

**No, on `balanced`.** Within-architecture balanced accuracy is 0.516 +/- 0.022
(x64) and 0.527 +/- 0.053 (x86) after tuning, against 0.498 and 0.489 before.
Chance is 0.5. The tuned `balanced` model is still reading architecture, not
code structure, and section 2's verdict stands unchanged: its headline balanced
accuracy is carried by the x64-heavy goodware corpus. The `cell` sampler -
which balances the (label, architecture) cells precisely to remove that
shortcut - was the worst configuration in the entire `balanced` search
(-0.117), which says the shortcut is most of what the model has.

**Partly, on `mendeley`.** x86 balanced accuracy goes 0.733 +/- 0.067 ->
0.787 +/- 0.029 and x64 0.460 +/- 0.198 -> 0.589 +/- 0.109, both measured
inside the architecture, and the tuned run's seed-to-seed spread is roughly
half the untuned one. x64 macro-F1 stays at 0.297 +/- 0.206 on 84 samples: x64
ransomware remains close to out of distribution.

### 9.7 Reproducibility

Seed 1 of each chosen configuration was re-run from scratch into a scratch
directory and diffed against the recorded run: **identical on both datasets** -
same 491 / 489 test rows, identical `y_pred`, max score delta 0.0, same
`epochs_run` (22 and 13), same macro-F1 to full precision. The harness seeds
`random`, `numpy` and `torch` per run and takes its batches by index from a
GPU-resident tensor, so there is no DataLoader worker ordering to drift.

### 9.8 Files

    results/cnn_vit/tuned/<dataset>/cv_search.csv        every configuration tried, its CV metrics and its cost
    results/cnn_vit/tuned/<dataset>/metrics.json         chosen config, 5 seeds; shared schema + macro_f1_sd + n_seeds
    results/cnn_vit/tuned/<dataset>/predictions.csv      all 5 seeds stacked, `seed` column
    results/cnn_vit/tuned/<dataset>/config_used.yaml     the configuration and the paths it used
    results/cnn_vit/tuned/<dataset>/seeds/seed<N>/       per-seed metrics.json, predictions.csv, training_log.csv
    results/cnn_vit/tuned/<dataset>/baseline/            the BASE configuration, 5 seeds, same layout
    results/cnn_vit/tuned/<dataset>/posthoc/rank<N>/     top-3 CV configs, 1 seed, test-scored after selection

`baseline/` and `posthoc/` are named that way on purpose: `baseline/` is the
untuned recipe re-measured through the tuning harness, and `posthoc/` was run
after selection closed and chose nothing. Model weights stay out of `results/`,
under `C:/Users/chaoa/Downloads/cnn_vit_models/tuned/<dataset>/seed<N>/`.

Regenerate every table above with
`python cnn_vit_pipeline/tuned_report.py all` - it reads these files and trains
nothing.
