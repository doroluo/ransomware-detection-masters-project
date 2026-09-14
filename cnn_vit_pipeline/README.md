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
| `train_eval.py` | trains, evaluates, writes `results/cnn_vit/<dataset>/metrics.json` |

Tested by `tests/test_cohort_split.py`.

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

## How a PNG finds its sha256

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
python cnn_vit_pipeline/build_dataset.py --dataset mendeley --out DIR --clean
python cnn_vit_pipeline/build_dataset.py --dataset balanced --out DIR --clean

# 2. train and score
python cnn_vit_pipeline/train_eval.py --data DIR --epochs 80 --device cpu

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
`seconds_per_epoch_mean` / `_median`, `device`, `seed`, and the full per-epoch
`history`.

## Inputs and what is still missing

Images live under `C:/Users/chaoa/Downloads/cnn_vit_images/<tree>/Class_*/`
(override with `--images-root` / `RANSOM_CNN_VIT_IMAGES`), built from
`C:/Users/chaoa/Downloads/asm_output/<tree>/` (`--asm-root` /
`RANSOM_ASM_OUTPUT`).

| image tree | dataset | status |
|---|---|---|
| `mendeley_goodware` | mendeley | present, 1,032 PNGs |
| `goodware_balanced` | balanced | present, 1,343 PNGs |
| `mendeley_goodware_test` | mendeley | **awaiting the VM** |
| `mendeley_ransomware_train` | both | **awaiting the VM** |
| `mendeley_ransomware_test` | both | **awaiting the VM** |

Generate the missing `.asm` trees per `vm_package/README.md` step 3, then turn
them into images with the same settings the goodware used:

```bash
python asm_parser.py --asm-dir  ~/asm_output/mendeley_ransomware_train \
                     --out-dir  cnn_vit_images/mendeley_ransomware_train \
                     --labels-csv cnn_vit_images/mendeley_ransomware_train_labels.csv \
                     --default-class 1
```

then re-run `build_dataset.py`.

### Known join gap (Mendeley goodware)

67 Mendeley goodware-train binaries are UPX-packed on disk while the cohort
CSV records their **unpacked** sha256, so they cannot be joined by hash; 19
more cohort filenames are absent from the host tree. 1,023 of 1,114 in-cohort
goodware-train rows therefore have an image today. `build_dataset.py` prints
the breakdown and stores it in `build_report.json`. The fix belongs upstream,
in extraction, not here.
