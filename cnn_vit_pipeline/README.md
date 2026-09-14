# CNN-ViT pipeline (HierarchicalMalwareNet on the shared cohort)

Re-runs the yanping CNN-ViT model on the **same samples and the same split**
as the tokenization and EMBER pipelines, so the three sets of numbers describe
the same thing. The model, the dataset class and the training protocol are the
originals from `model_train.py` — imported, not copied. Only the data layout
changes.

## Files

| file | what |
|---|---|
| `cohort.py` | the one shared place where the cohort and the split are decided, plus the shared `metrics.json` writer. `ember_pipeline/train_eval.py` imports it too. |
| `build_dataset.py` | maps PNGs back to sha256, applies the split, materialises `train/val/test` |
| `train_eval.py` | trains, evaluates, writes `results/cnn_vit/<dataset>/{metrics.json,splits.csv,training_log.csv,test_predictions.csv,config_used.yaml}` |
| `encode.py` | **tuning only** — parses `.asm` once into a token cache, then renders it under one of five input encodings (`head3` is the current pipeline, byte for byte) |
| `tuned_model.py` | **tuning only** — `model_train.HierarchicalMalwareNet` with its hard-coded sizes turned into arguments; asserted shape-identical at its defaults |
| `tuned_train.py` | **tuning only** — group-CV hyper-parameter search on train, then one final 5-seed test run, into `results/cnn_vit/tuned/<dataset>/` |
| `tuned_report.py` | **tuning only** — reads those artefacts and prints the markdown tables that go in `summary.md` (top-N CV, per-axis effect, CV→test gap, before/after). Trains nothing. |
| [`../asm_tool/unified_to_asm.py`](../asm_tool/unified_to_asm.py) | turns the revised extractor's disassembly into an `asm_parse.py`-shaped `.asm` tree that `asm_parser.py` can render unmodified |

Tested by `tests/test_cohort_split.py` and `tests/test_cnn_vit_encoding.py`.

`CNN-ViT/model_train.py` and `CNN-ViT/asm_parser.py` are **not** modified by any
of this: `train_eval.py` still imports the original model, and the tuning
harness carries its own parameterised copy whose default configuration is
checked against the original at runtime
(`python -m cnn_vit_pipeline.tuned_model`).

Results and the narrative that goes with them:
[`results/cnn_vit/summary.md`](../results/cnn_vit/summary.md).

## Which Python

Training used a dedicated venv at `C:/Users/chaoa/Downloads/cnn_vit_venv`
(Python 3.12 + `torch 2.11.0+cu128`), because the RTX 5080 is Blackwell
(`sm_120`) and needs a cu128-or-newer wheel, and the system Python 3.14 only
has a CPU build. Everything except training — conversion, rendering,
`build_dataset.py`, the tests — runs fine on the system Python. `--device cpu`
remains the fallback and is the default.

## What differs from the original yanping protocol

| | original | here |
|---|---|---|
| split | `stratified_split.py` — a random shuffle inside each class folder | the cohort split: Mendeley's own `good_train`/`mal_train` vs `good_test`/`mal_test` |
| ransomware families | the same family lands on both sides | **disjoint**: 24 train families, 14 test families, asserted in the tests |
| .NET assemblies | included | excluded (`tag:dotnet`) |
| packed / UPX | included | excluded unless UPX-unpackable (`tag:packed_other`, `tag:upx`) |
| duplicates | the same bytes under two filenames land on both sides | one row per sha256 |
| Thanos | included | excluded |
| val | a random slice | fixed seed 1337, 10% of train, stratified by label, **group-aware** — no family and no goodware source project appears in both train and val |

Why it matters: a random per-class split lets the model see a family in
training and then be scored on the same family — often on a near-identical
binary. That measures memorisation. A packed sample's 256x256 image is mostly
a picture of the packer, and a .NET assembly's is mostly IL metadata, so
neither tells you much about ransomware behaviour. The original numbers are
not wrong, they answer a different and much easier question.

## Two image sources

`build_dataset.py --source` picks where the PNGs come from. `auto` (the
default) uses `unified` when both of its trees are on disk and falls back to
`asm_parse` otherwise.

### `unified` — what the results in `results/cnn_vit/` were produced from

The revised extractor (`Shared/extract_unified.py`) has full disassembly for
**every** class and set on this host, keyed by sha256.
[`asm_tool/unified_to_asm.py`](../asm_tool/unified_to_asm.py) re-shapes it
into an `asm_parse.py`-shaped tree:

```
Shared/Extract/asm/<sha256>.asm          ->  OUT/<set>/<family>/<sha256>.asm
Shared/Extract_Goodware_Balanced/asm/..  ->  OUT/goodware_balanced/<bucket>/<sha256>.asm
```

Only `;` section headers and `.skip<TAB>N bytes` markers are removed — the
latter because `asm_parser.parse_asm_line` would otherwise turn each one into
a bogus `UNKNOWN_OPCODE` triplet. Everything else is copied through byte for
byte, and the stream is capped at 100,000 instructions to match
`asm_parse.py`'s own cap. The same unmodified `asm_parser.py` then renders the
images, one run per set so `--default-class` assigns the right class.

Because the `.asm` basename is the sha256, the PNG -> sha256 join is exact.
`build_dataset.py` still checks every stem against `asm_manifest.csv` (all
three roots `asm_parser.py` could have been pointed at are accepted) rather
than parsing hex out of a filename, and it reads each image's class from the
`Class_<n>_*` folder, so the tree-label-vs-cohort-label cross-check keeps an
independent source that could disagree.

> **Caveat.** These images come from the *skip-data* extractor, while the
> original yanping pipeline used `asm_parse.py`'s linear sweep, which stops at
> the first undecodable byte. The two produce materially different stream
> lengths — see [`asm_tool/README.md` §6](../asm_tool/README.md). Both classes
> and both corpora go through the same extractor here, so comparisons *within*
> these results are sound; comparing one of these numbers against an old
> `asm_parse`-sourced number is not.

### `asm_parse` — the original, kept working

**This source cannot produce a cohort-faithful run on this host.** The
ransomware `.asm` trees do not exist here (the binaries are VM-only), and the
host Mendeley-goodware tree was built from a copy in which 67 UPX files are
still packed, so their on-disk sha256s do not match the cohort CSVs. It is
kept working because it is the right source once the VM delivers.

`asm_parse.py` writes `<filename>.asm` mirroring the source tree and an
`asm_manifest.csv` carrying `sha256` next to each `asm_path`. `asm_parser.py`
names each image with `asm_id()`, which is the .asm path relative to the tree
with `/` replaced by `__` (so `everyday/node.exe.asm` -> `everyday__node.exe`,
keeping the 24 same-basename pairs in the balanced goodware distinct).
`build_dataset.py` recomputes `asm_id()` over the manifest — importing the
real function, not re-implementing it — and joins on the resulting stem. The
join is by hash, never by filename. Anything that fails to map is written to
`unmapped.csv`.

## Commands

```bash
# 1. lay out the tree (hard links by default, so no extra disk)
#    --source defaults to auto: unified when its trees exist, else asm_parse
python cnn_vit_pipeline/build_dataset.py --dataset mendeley --out DIR --clean
python cnn_vit_pipeline/build_dataset.py --dataset balanced --out DIR --clean

# 2. train and score (--device cuda if a CUDA torch is available)
python cnn_vit_pipeline/train_eval.py --data DIR --dataset mendeley \
    --epochs 80 --batch-size 16 --seed 1337 --device cuda \
    --ckpt-dir ../cnn_vit_models/mendeley

# plumbing + timing check: goodware images only, with a clearly FAKE second
# class, capped sample count. Reports seconds/epoch. Never writes to results/.
python cnn_vit_pipeline/train_eval.py --smoke \
    --images ../cnn_vit_images/mendeley_goodware \
    --scratch /tmp/cnnvit_smoke --epochs 3 --max-samples 512 --device cpu
```

`build_dataset.py` writes `manifest.csv` (sha256, asm_id, label, fold, family,
arch, source, paths), `build_report.json` (counts, coverage, what is missing)
and, when relevant, `unmapped.csv` and `duplicate_sha_pngs.csv`.

`train_eval.py` refuses to run when the test fold has fewer than two classes,
names the image trees that are still missing, points at
[`vm_package/README.md`](../vm_package/README.md) step 3 and exits non-zero.

## Training protocol (unchanged from `model_train.py`)

AdamW, base LR 3e-4, weight decay 1e-2, 5 warm-up epochs then cosine annealing
to 1e-6, batch 16, `CrossEntropyLoss(label_smoothing=0.15)`, dropout 0.15,
`WeightedRandomSampler` for class balance, gradient clipping at 1.0, early
stopping on validation loss with `PATIENCE = 8` and `MIN_DELTA = 1e-4`, best
checkpoint restored before the final test pass. Seeds are pinned
(`--seed`, default 1337).

## Importing `model_train.py`

`model_train.py` has no module-level side effects — everything that touches
disk or starts training is under `if __name__ == "__main__"` — so importing it
is safe and its `__main__` block never runs. Its one import-time obstacle is
`from torchvision import transforms`, and **torchvision is not installed here**
(torch 2.11 CPU is). `model_train` uses exactly one thing from it,
`transforms.ToTensor()`, on a mode-'L' PIL image, which is `uint8 HxW ->
float32 1xHxW / 255`. `train_eval.py` registers a minimal stand-in for that
single call *only when torchvision is genuinely absent*, prints a notice when
it does, and records `torchvision_shim: true` in `metrics.json`. This keeps
the real `HierarchicalMalwareNet`, `MalwareMaskedDataset`,
`MaskAwareStructuralShift` and `evaluate_model` in play instead of a copy that
would go stale. Installing torchvision makes the stand-in disappear on its own.

## metrics.json

Same schema as `results/expA/metrics.json`, plus `per_architecture` (the full
metric block for x86 and for x64) and `per_family_recall` (recall, correct and
support per ransomware test family). Also recorded: `epochs_run`,
`epochs_requested`, `epoch_cap_reason`, `seconds_per_epoch_mean` / `_median` /
`_min` / `_max`, `train_seconds_total`, `device`, `device_name`, `seed`, the
resolved `config`, and the full per-epoch `history`.

`train_eval.py` also writes, next to it: `splits.csv` (the five columns
`results/exp*/splits.csv` uses, plus sha256, arch and family),
`training_log.csv` (epoch, lr, train loss, val loss, val acc, seconds),
`test_predictions.csv` (per-sample `y_true` / `y_pred` / `score_ransomware`)
and `config_used.yaml`. Model weights go to `--ckpt-dir`, never into
`results/` — they are binary.

## Inputs

Images live under `C:/Users/chaoa/Downloads/cnn_vit_images/<tree>/Class_*/`
(override with `--images-root` / `RANSOM_CNN_VIT_IMAGES`), built from
`C:/Users/chaoa/Downloads/asm_output/<tree>/` (`--asm-root` /
`RANSOM_ASM_OUTPUT`).

| image tree | source | dataset | status |
|---|---|---|---|
| `unified_mendeley` | unified | mendeley (both classes), balanced (class 1) | present, 2,598 PNGs |
| `unified_goodware_balanced` | unified | balanced (class 0) | present, 1,343 PNGs |
| `mendeley_goodware` | asm_parse | mendeley | present, 1,032 PNGs |
| `goodware_balanced` | asm_parse | balanced | present, 1,343 PNGs |
| `mendeley_goodware_test` | asm_parse | mendeley | **awaiting the VM** |
| `mendeley_ransomware_train` | asm_parse | both | **awaiting the VM** |
| `mendeley_ransomware_test` | asm_parse | both | **awaiting the VM** |

Building the unified trees (this is what the current results used):

```bash
python asm_tool/unified_to_asm.py \
    --extract "C:/Users/chaoa/Downloads/asm and mm/Shared/Extract" \
    --out     "C:/Users/chaoa/Downloads/asm_output/unified_mendeley"
python asm_tool/unified_to_asm.py \
    --extract "C:/Users/chaoa/Downloads/asm and mm/Shared/Extract_Goodware_Balanced" \
    --out     "C:/Users/chaoa/Downloads/asm_output/unified_goodware_balanced"

# one asm_parser.py run per set, so --default-class assigns the right class
for s in good_train:0 good_test:0 mal_train:1 mal_test:1; do
  python asm_parser.py \
      --asm-dir "…/asm_output/unified_mendeley/${s%%:*}" \
      --out-dir "…/cnn_vit_images/unified_mendeley" \
      --labels-csv "<scratch>/labels_${s%%:*}.csv" --default-class "${s##*:}"
done
python asm_parser.py --asm-dir "…/asm_output/unified_goodware_balanced" \
    --out-dir "…/cnn_vit_images/unified_goodware_balanced" \
    --labels-csv "<scratch>/labels_gb.csv" --default-class 0
```

Coverage is complete: 2,509/2,509 mendeley and 2,506/2,506 balanced cohort rows
have an image, nothing unmapped, nothing dropped.

Once the VM delivers the `asm_parse.py` trees, generate them per
`vm_package/README.md` step 3, render with `--default-class 1`, and re-run
`build_dataset.py --source asm_parse`.

### Known join gap on the `asm_parse` source (Mendeley goodware)

67 Mendeley goodware-train binaries are UPX-packed on disk while the cohort
CSV records their **unpacked** sha256, so they cannot be joined by hash; 19
more cohort filenames are absent from the host tree. Only 1,023 of 1,114
in-cohort goodware-train rows therefore have an `asm_parse` image.
`build_dataset.py` prints the breakdown, stores it in `build_report.json`, and
names every missing row in `missing_from_images.csv`. This is precisely why the
results were produced from the `unified` source, where the join is by sha256 on
both sides and nothing is lost. The fix belongs upstream, in extraction.

## Tuning: `encode.py` + `tuned_train.py`

The three files above reproduce the *original* protocol. The tuning harness
answers a different question — how much of the CNN-ViT's weak showing is the
protocol and how much is the configuration — and it is kept strictly separate
so the baseline cannot drift.

### Protocol

Selection never touches the test fold. `tuned_train.py search` runs
**3-fold group cross-validation over the train split only**
(`StratifiedGroupKFold` on `family_or_group`, so whole ransomware families and
whole goodware projects are held out together), and inside each CV-train part
it carves a *further* group-aware inner val fold for early stopping — so the
fold being scored never chose the checkpoint that scores it. Every
configuration tried is appended to `results/cnn_vit/tuned/<dataset>/cv_search.csv`.
Only after that does `tuned_train.py final` train on the full train split
(early stopping on the fixed `cohort.add_val_fold` val fold, the same one
`train_eval.py` uses) and score the test fold once per seed.

Where the decision threshold is fitted rather than fixed at 0.5, the reported
CV number is **nested**: each fold's threshold comes from the out-of-fold
scores of the other folds only. `threshold_refit_all_oof` — the one the final
model deploys — is refitted on all out-of-fold train scores and still never
sees test.

### Input encodings

`encode.py` parses each `.asm` exactly once into a token cache
(`cnn_vit_cache/<tree>/{tokens.u8,index.csv}`) and renders canvases from it.
`--variant` picks what lands on the 256x256 canvas:

| variant | what the canvas holds |
|---|---|
| `head3` | **the current pipeline, byte for byte.** Instruction triplets (opcode/API id, operand-1 class, operand-2 class) in file order, truncated at 65,536 tokens — the first 21,845 whole instructions plus the opcode of the 21,846th |
| `stride3` | the same triplets, but for files longer than that window the instructions kept are sampled on an even stride across the **whole** file instead of taking the first ones (50% of Mendeley and 61% of balanced-goodware files exceed the window) |
| `mnem1` | mnemonic-only: one token per instruction, operand classes dropped, so three times as much code fits — the representation the mnemonic TF-IDF baseline wins with |
| `mnem1s` | mnemonic-only plus the even-stride subsampling |
| `crop3` | K=4 fixed windows of 21,846 consecutive instructions with evenly spaced starts (crop 0 == `head3`); one is drawn at random per sample per epoch and `--tta` averages all K at eval time |

`--row-align` additionally pads each canvas row out to a whole number of
instructions (85 instructions = 255 tokens + one `END_PAD`), so every row
starts on an opcode instead of drifting through three phases.

`tests/test_cnn_vit_encoding.py` pins all of this against a hand-written eight
-instruction `.asm` and asserts that `head3` equals the unmodified
`asm_parser.py` output byte for byte.

### Commands

```bash
PY="C:/Users/chaoa/Downloads/cnn_vit_venv/Scripts/python.exe"   # 3.12 + cu128

# 0. the model copy really is the original
$PY -m cnn_vit_pipeline.tuned_model
$PY cnn_vit_pipeline/encode.py selftest
python -m pytest tests/test_cnn_vit_encoding.py -q                  # system Python is fine

# 1. parse every .asm once (~4 min on 16 cores; variant-agnostic)
$PY cnn_vit_pipeline/encode.py cache --asm-tree unified_mendeley
$PY cnn_vit_pipeline/encode.py cache --asm-tree unified_goodware_balanced

# 2. search: group CV on TRAIN only, appends to cv_search.csv.
#    --resume skips configurations already in the file, so an interrupted
#    search is picked up exactly where it stopped.
for D in mendeley balanced; do
  $PY cnn_vit_pipeline/tuned_train.py search --dataset $D --resume \
      --grid encoding,optim,loss,decision,capacity
done
#   --grid file:<path.json> takes an explicit list of override dicts; the
#   stage column in cv_search.csv is the file's stem. That is how the
#   `combine` stage (per-axis winners crossed) was run:
#       $PY cnn_vit_pipeline/tuned_train.py search --dataset balanced \
#           --grid file:grids/balanced/combine.json --resume
#   and how the top-3 were re-cross-validated at a second fold-split seed:
#       $PY cnn_vit_pipeline/tuned_train.py search --dataset balanced \
#           --grid file:grids/balanced/cvseed.json --seed 7 --resume

# 3. the BASE configuration, 5 seeds - the "before" the tuned run is compared
#    against, measured through this harness rather than train_eval.py
for D in mendeley balanced; do
  $PY cnn_vit_pipeline/tuned_train.py final --dataset $D --seeds 1,2,3,4,5 \
      --ckpt-tag base --out results/cnn_vit/tuned/$D/baseline
done

# 4. post hoc, AFTER selection closed: the top-3 CV configurations on test,
#    one seed each, for the CV->test gap table. --ckpt-tag keeps their weights
#    out of the chosen configuration's checkpoint directory.
$PY cnn_vit_pipeline/tuned_train.py final --dataset balanced --seeds 1 \
    --variant stride3 --ckpt-tag ph1 --out results/cnn_vit/tuned/balanced/posthoc/rank1

# 5. final: the chosen configuration, 5 seeds -> results/cnn_vit/tuned/<dataset>/
$PY cnn_vit_pipeline/tuned_train.py final --dataset mendeley --seeds 1,2,3,4,5 \
    --weight-decay 0.1 --width 16 --depth 3 --dropout 0.3
$PY cnn_vit_pipeline/tuned_train.py final --dataset balanced --seeds 1,2,3,4,5 \
    --variant stride3

# 6. the tables in summary.md section 9 (reads the files above, trains nothing)
python cnn_vit_pipeline/tuned_report.py all

# 7. (reproducibility) re-run one seed of the chosen configuration into a
#    scratch directory and diff it. Both datasets come back bit-identical:
#    same y_pred, max score delta 0.0, same epochs_run.
$PY cnn_vit_pipeline/tuned_train.py final --dataset balanced --seeds 1 \
    --variant stride3 --ckpt-tag repro --out /tmp/repro_balanced

# 8. (optional) render the chosen encoding to PNG trees and rebuild the
#    hard-linked dataset, so the tuned images can be inspected like any other
$PY cnn_vit_pipeline/encode.py render --asm-tree unified_mendeley \
    --variant <v> --out-root "C:/Users/chaoa/Downloads/cnn_vit_images/tuned_<v>"
$PY cnn_vit_pipeline/encode.py render --asm-tree unified_goodware_balanced \
    --variant <v> --out-root "C:/Users/chaoa/Downloads/cnn_vit_images/tuned_<v>"
$PY cnn_vit_pipeline/build_dataset.py --dataset mendeley --source unified \
    --images-root "C:/Users/chaoa/Downloads/cnn_vit_images/tuned_<v>" \
    --out "C:/Users/chaoa/Downloads/cnn_vit_data/tuned_<v>_mendeley" --clean
```

**Chosen by CV, and what it is worth.** `mendeley` takes `head3` with
`weight_decay=0.1`, width 16, depth 3, dropout 0.3; `balanced` takes `stride3`
at otherwise stock settings. Both decide at plain `argmax`. Re-cross-validating
the top three at a *second* fold-split seed showed that on `mendeley` nothing
in 56 configurations survives changing the fold split, and the tuned run scores
0.655 ± 0.049 macro-F1 against the same recipe's 0.660 ± 0.096 — tuning bought
nothing there. `balanced` gains +0.078 over that baseline on the strength of
the encoding change alone, but is still at chance *within* each architecture.
Read [`results/cnn_vit/summary.md` §9](../results/cnn_vit/summary.md) before
quoting any of it.

Model weights go to `C:/Users/chaoa/Downloads/cnn_vit_models/tuned/<dataset>/seed<N>/`
and never into `results/`. The token cache and the per-variant canvas arrays
live under `C:/Users/chaoa/Downloads/cnn_vit_cache/` (override with
`RANSOM_CNN_VIT_CACHE`).

Results: [`results/cnn_vit/summary.md`](../results/cnn_vit/summary.md), section
"Tuned".
