# EMBER pipeline (LightGBM on 537 static features)

This directory re-runs the EMBER branch's model on the **shared cohort** so its
numbers can be put next to the tokenization and CNN-ViT numbers. The features
and the model are the branch's, unchanged. Only the sample selection and the
split change.

## The features

`ember_extractor.py` and `sanitizer.py` are verbatim from the EMBER branch.
`extract_features_single()` returns **537 float32 dimensions**:

| slice | dims | what |
|---|---|---|
| `0:256` | 256 | normalised byte-value histogram |
| `256:512` | 256 | 16x16 byte-entropy histogram (2048-byte window, 1024 step), flattened |
| `512:533` | 21 | structural: file size, `sizeof_image`, entry point, machine, characteristics, subsystem, DLL characteristics, has-imports/exports/resources/signature/debug/relocations/TLS, import-entry count, imported-DLL count, export count, section count, mean/max/min section entropy |
| `533:537` | 4 | printable-string count, mean length, max length, chars per byte |

`prune_brittle_features()` is the branch's own ablation: it drops `0:512` and
the last 4, leaving the **21 structural dims**. Both are reported, as
`feature_set: "full"` and `feature_set: "pruned"`.

`extract_features.py` is the per-file extractor. It reads bytes and parses with
LIEF; it never executes a sample. Every input gets a manifest row (ok or
error), and every vector is keyed by **sha256**, which is what makes the cohort
filter and the family split applicable after the fact.

## The two variants

| `--variant` | samples | split |
|---|---|---|
| `cohort` | cohort rows only (`in_cohort == 1`: tag `plain` or `upx_unpacked`, Thanos dropped; .NET, entropy-flagged packed, non-unpackable UPX, no-code, odd-arch, broken and duplicate rows excluded) | the shared family-disjoint split from `cnn_vit_pipeline/cohort.py` — 24 train families, 14 test families, no family on both sides |
| `branch` | everything that parsed, no filter | `train_test_split(test_size=0.2, random_state=42, stratify=y)` |

Run both. The gap between them is the point: it is the size of the correction.

> **Caveat on the branch's headline numbers.** They came from `branch`-style
> conditions — a random 80/20 split with no exclusions. A random split puts the
> same ransomware family, and often the same bytes under a second filename, on
> both sides, so the test set measures memorisation as much as generalisation.
> The corpus also contains .NET assemblies whose byte histogram is dominated by
> IL metadata and packed samples whose histogram is dominated by the packer.
> Treat `branch` as a reproduction for comparison, not as a result.

## Commands

```bash
# what the split looks like and what is still missing - trains nothing
python ember_pipeline/train_eval.py --dataset mendeley --dry-run
python ember_pipeline/train_eval.py --dataset balanced --dry-run

# the real runs (need the VM's ransomware npz - see below)
python ember_pipeline/train_eval.py --dataset mendeley --variant cohort
python ember_pipeline/train_eval.py --dataset mendeley --variant branch
python ember_pipeline/train_eval.py --dataset balanced --variant cohort
python ember_pipeline/train_eval.py --dataset balanced --variant branch

# prove the LightGBM + metrics path with no ransomware present
# (pseudo-labels x86/x64 on goodware; never writes under results/)
python ember_pipeline/train_eval.py --dataset mendeley --smoke --out /tmp/ember_smoke
```

Output goes to `results/ember/<dataset>/<variant>/`:
`metrics.json`, `split_used.csv`, and `duplicate_sha_rows.csv` when relevant.

LightGBM is trained exactly as the branch does: `objective=binary`,
`learning_rate=0.05`, `num_leaves=31`, 100 boosting rounds, all seeds pinned to
42 and `deterministic=True` so re-runs reproduce.

## Inputs

Feature files live in `C:/Users/chaoa/Downloads/ember_features/`
(override with `--features-dir` or `RANSOM_EMBER_FEATURES`):

| npz | dataset | produced on |
|---|---|---|
| `mendeley_good_train` | mendeley | host — present |
| `goodware_balanced` | balanced | host — present |
| `mendeley_good_test` | mendeley | **VM** |
| `mendeley_mal_train` | both | **VM** |
| `mendeley_mal_test` | both | **VM** |

The ransomware never leaves the VM, so its vectors are extracted there; see
[`vm_package/README.md`](../vm_package/README.md) step 4. If any required npz
is absent the harness names the missing files and the runbook and exits
non-zero. It does not fabricate, impute or fall back to a smaller test set.

## metrics.json

Same schema as `results/expA/metrics.json` — `experiment`, `description`,
`task`, `samples`, `results`, `elapsed_seconds`, with each result carrying
accuracy, balanced accuracy, macro-F1, ROC-AUC, per-class precision/recall/F1,
FPR and the confusion matrix — plus two additions:

* `per_architecture` — the whole metric block again for `x86` and for `x64`
  (architecture comes from the cohort CSVs).
* `per_family_recall` — recall, correct count and support for each ransomware
  test family, so a single family collapsing is visible instead of averaged
  away.

## Known join gap (Mendeley goodware)

67 of the Mendeley goodware-train binaries are UPX-packed on disk, and the
cohort CSV records their **unpacked** sha256. The host extractor ran on the
packed originals, so those rows cannot be joined by hash, and 19 further
cohort filenames are not in the host tree at all. That leaves 1,028 of 1,114
in-cohort goodware-train rows with features today. `--dry-run` prints this
breakdown. Resolving it means unpacking those 67 before extraction, the same
way the cohort builder did — it is a data-preparation fix, not a harness fix.
