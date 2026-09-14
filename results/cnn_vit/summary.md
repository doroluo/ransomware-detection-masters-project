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

## 7. Seed sweep (5 seeds, GPU, same protocol)

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
