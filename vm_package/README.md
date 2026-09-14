# VM runbook: the ransomware side of every pipeline

The ransomware binaries never leave the VM. Everything below runs there,
offline, and produces **text and feature files only**, which are then copied
back to the host. Nothing executes a sample; every tool parses bytes.

Status (13 September 2026): the Tokenization and CNN-ViT pipelines no longer
need anything from the VM — both classes come from the revised extractor's
`Shared/Extract` output (CNN-ViT via `asm_tool/unified_to_asm.py`). Only the
EMBER rerun still needs VM-side extraction, and that is being handled by a
teammate. Steps 3 and 5's `.asm` items are kept for a future
`asm_parse.py`-sourced CNN-ViT comparison; they are optional.

| pipeline | needs from the VM | produced by |
|---|---|---|
| Tokenization, revised features | `Shared/Extract/{mn,manifest.csv}` | already done (`extract_unified.py`) |
| Tokenization, traditional features | `LLM_Features` | already done (`extract.py`) |
| CNN-ViT (Yanping), unified source | `Shared/Extract/asm` | already done (`extract_unified.py`) |
| CNN-ViT, `asm_parse.py` source (optional) | `.asm` trees in `asm_parse.py` format | step 3 |
| EMBER (LightGBM) | 537-dim feature vectors keyed by SHA-256 | step 4 (teammate) |

`Goodware_Test` is VM-only, and `Goodware_Training` on the VM is the complete,
already-`upx -d`'d copy (1,134 files; the host copy has 1,115 with 67 still
packed, so their hashes do not match the cohort). Both goodware sets therefore
go through steps 3 and 4 on the VM as well; the host-side goodware outputs are
a fallback only.

## 1. Build the package on the host

```bash
python vm_package/make_package.py --out ~/Downloads/vm_package.zip
```

That zips: `asm_parse.py`, `check_arch.py`, `asm_tool/` (with `wheels/`),
`ember_pipeline/` (with `wheels/`), and this README. About 66 MB, almost all wheels.

## 2. Install on the VM (offline)

```bash
unzip vm_package.zip -d ~/vm_package && cd ~/vm_package
python3 --version                      # pick the matching cpXY folder below
PYV=cp310                              # cp38 / cp310 / cp312
python3 -m pip install --no-index --find-links asm_tool/wheels/$PYV      -r asm_tool/requirements-vm.txt
python3 -m pip install --no-index --find-links ember_pipeline/wheels/$PYV -r ember_pipeline/requirements-vm.txt
```

## 3. CNN-ViT `.asm` trees

Same extractor and same settings as the goodware trees already on the host
(`asm_output/mendeley_goodware`, `asm_output/goodware_balanced`): linear
sweep, `MEM_EXECUTE` sections, 100,000-instruction cap, `<filename>.asm`,
one manifest row per input.

```bash
mkdir -p ~/asm_output
python3 asm_parse.py --in-dir ~/Downloads/Ransomware_Training/rans     --out-dir ~/asm_output/mendeley_ransomware_train
python3 asm_parse.py --in-dir ~/Downloads/Ransomware_Test/rans_test    --out-dir ~/asm_output/mendeley_ransomware_test
python3 asm_parse.py --in-dir ~/Downloads/Goodware_Test/goodware_test  --out-dir ~/asm_output/mendeley_goodware_test
python3 asm_parse.py --in-dir ~/Downloads/Goodware_Training/goodware   --out-dir ~/asm_output/mendeley_goodware_train
```

Family folders are mirrored, so `mendeley_ransomware_train/avaddon/<sha>.asm`.
Each folder gets an `asm_manifest.csv` with `sha256`, `status`, `arch`,
`packed_flag`, `is_dotnet`, `instructions`.

## 4. EMBER feature vectors

The EMBER branch's own extractor, unchanged, keyed by SHA-256 so the host can
apply the shared cohort filter and the family split afterwards.

```bash
mkdir -p ~/ember_features
python3 ember_pipeline/extract_features.py --in ~/Downloads/Ransomware_Training/rans    --out ~/ember_features/mendeley_mal_train  --label 1 --set mal_train
python3 ember_pipeline/extract_features.py --in ~/Downloads/Ransomware_Test/rans_test   --out ~/ember_features/mendeley_mal_test   --label 1 --set mal_test
python3 ember_pipeline/extract_features.py --in ~/Downloads/Goodware_Test/goodware_test --out ~/ember_features/mendeley_good_test  --label 0 --set good_test
python3 ember_pipeline/extract_features.py --in ~/Downloads/Goodware_Training/goodware  --out ~/ember_features/mendeley_good_train --label 0 --set good_train
```

Each produces `<name>.npz` (X, sha256, rel_path, label) and
`<name>.manifest.csv` (one row per input file, including errors). The
byte-entropy histogram is the slow part; expect roughly a minute per 100 MB
of binaries.

## 5. Copy back to the host

Only these leave the VM:

```
~/asm_output/mendeley_ransomware_train/   ->  C:/Users/chaoa/Downloads/asm_output/mendeley_ransomware_train/
~/asm_output/mendeley_ransomware_test/    ->  C:/Users/chaoa/Downloads/asm_output/mendeley_ransomware_test/
~/asm_output/mendeley_goodware_test/      ->  C:/Users/chaoa/Downloads/asm_output/mendeley_goodware_test/
~/asm_output/mendeley_goodware_train/     ->  C:/Users/chaoa/Downloads/asm_output/mendeley_goodware_train/   (replaces the host-built mendeley_goodware/)
~/ember_features/*.npz, *.manifest.csv    ->  C:/Users/chaoa/Downloads/ember_features/
```

The `.asm` trees are a faithful transcript of malware code. Treat the archive
the way you would treat the samples.

## 6. Then, on the host

Every run below is one command and applies the same cohort
(`Shared/cohort_mendeley.csv`, `Shared/cohort_balanced.csv`: tag `plain` or
`upx_unpacked`, Thanos excluded) and the same family split.

```bash
# Tokenization: expA/expB traditional, expC/expD revised
<venv python> llm_features_pipeline/run_pipeline.py --experiment expC
<venv python> llm_features_pipeline/run_pipeline.py --experiment expD

# CNN-ViT and EMBER: see cnn_vit_pipeline/README.md and ember_pipeline/README.md
```
