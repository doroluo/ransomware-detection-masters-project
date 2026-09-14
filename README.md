# LLM_Classification: PE -> .asm -> opcode text -> classifier

Branch of the ransomware-detection master's project that carries the
disassembly tooling, the tokenization / embedding / classifier pipeline, the
apples-to-apples rerun harnesses for the CNN-ViT (`yanping`) and EMBER
branches, and the recorded experiment results. The CNN-ViT model code itself
lives on `yanping` under `CNN-ViT/`; the EMBER extractor is vendored here
verbatim under `ember_pipeline/`.

## Dataset
https://data.mendeley.com/datasets/yzhcvn7sj5/3 (Moreira et al., v3, with executables)

Corpus profile, cohort definition and the reasoning behind every exclusion:
`extract-opcode/WRITEUP.md` in the dataset folder. The cohort used by every
harness here is `manifests/cohort/cohort_{mendeley,balanced}.csv`: tag `plain`
or `upx_unpacked` only (no .NET, no entropy-packed, no unrecoverable UPX, no
code-less / odd-arch / broken / duplicate rows), Thanos excluded as a family.

## asm_parse.py
Converts a tree of PE executables into `.asm` files, mirroring the input directory
structure so class folders stay separated.

### Usage
```
python3 asm_parse.py --in-dir ../Goodware_Balanced --out-dir ../asm_output/goodware_balanced
python3 asm_parse.py --in-dir X --out-dir Y --limit 50        # smoke test
python3 asm_parse.py --in-dir X --out-dir Y --refresh-manifest
```
| flag | meaning |
|---|---|
| `--in-dir` / `--out-dir` | required; paths are no longer hardcoded |
| `--limit N` | stop after N files (0 = all, the default) |
| `--max-instructions N` | per-file cap, default 100,000 (0 = uncapped) |
| `--jobs N` | worker processes |
| `--overwrite` | redo files that already have a `.asm`; otherwise runs resume |
| `--refresh-manifest` | rewrite the manifest only — re-read every binary, recompute every column, verify the existing `.asm` still matches, write no `.asm` |

Output goes under one shareable, gitignored root, `asm_output/<dataset>/`:
`asm_output/goodware_balanced/` (formerly `../Balanced_Goodware_ASM`),
`asm_output/mendeley_goodware/`, and `asm_output/ransomware/` from the VM. See
[asm_tool/README.md](asm_tool/README.md) §0 for what each contains.

Output line format, which is what `asm_parser.py`'s `parse_asm_line` expects:
```
0x00401000:  mov	eax, ebx
```

### Naming
Output is `<original filename>.asm`, **not** `<stem>.asm` — `7z.exe` and `7z.dll`
live in the same class folder and would otherwise both write `7z.asm`. Keeping the
extension also makes the `.asm` basename equal the source PE filename, which is what
`asm_parser.py` matches on when building labels.

### What gets skipped, and why
Written to `asm_manifest.csv` alongside the output, one row per input file, so every
exclusion is accounted for. The directory walk skips only `_upx_packed/`, `.tools/`,
`flagged/` and `quarantined/` by name and `.csv`/`.md`/`.json` by suffix; everything
else gets a row, and manifest row count equals input file count.

| status | meaning |
|---|---|
| `ok` / `truncated` | disassembled; `truncated` hit `--max-instructions` |
| `dotnet_ilonly` | IL-only managed assembly — see below |
| `upx_packed` | UPX-packed; run `upx -d` first, or it disassembles the decompressor stub |
| `no_exec_section` | no `MEM_EXECUTE` section (e.g. `api-ms-win-*` API-set forwarders) |
| `empty_disassembly` | executable section present, nothing decodable in it |
| `arch_unsupported` | not x86/x64 |
| `not_pe` / `pe_error:*` / `read_error:*` / `crash:*` | malformed or unreadable input, recorded rather than raised |

Columns: `rel_path`, `asm_path`, `status`, `arch`, `instructions` (the original five,
unchanged), plus `sha256`, `machine`, `bitness`, `is_dotnet`, `packed_flag`,
`packer_guess`, `sections_disassembled`, `n_instructions`, `error`, `source`.
`packed_flag` / `packer_guess` are advisory metadata from packer section names,
executable-section entropy > 7.2 and an entry point outside every executable
section — **they never change which files get disassembled**.

**Resume never degrades the manifest.** A resumed run reproduces the row the
original run wrote, including the `ok` / `truncated` distinction, and repairs a
manifest left behind by an older version. Use `--refresh-manifest` to rebuild a
manifest for an existing `.asm` tree without writing to that tree.

**Known limitation, deliberately kept:** the capstone sweep runs with `skipdata`
off, so a section ends at the first undecodable byte — exactly what `extract.py`
did when `LLM_Features` was built. On `Goodware_Balanced`, 485 of the 1,171
uncapped files (41.4%) yield under half the instructions an uncapped skip-data
sweep gets, and some Go/Rust binaries yield fewer than ten. Measured in full in
[asm_tool/README.md](asm_tool/README.md) §6, with the reason changing it would
create a class shortcut rather than remove one.

**On .NET:** the check is the `COMIMAGE_FLAGS_ILONLY` bit in the CLR header, not the
mere presence of a COM descriptor. IL-only assemblies hold CIL bytecode with a ~6-byte
native stub — capstone would decode CIL as x86 and emit thousands of confident-looking
fake instructions. Mixed-mode (C++/CLI) assemblies carry real native code and *are*
disassembled. In the 1,500-sample goodware corpus this was 69 IL-only vs 1 mixed-mode.

**On the instruction cap:** `asm_parser.py` keeps only the first 65,536 tokens
(~21,800 instructions), so 100,000 is over 4x headroom. It matters because uncapped,
that corpus disassembles to ~31 GB — one 185 MB binary alone produces millions of
lines. Capped, it is 1.23 GB.

**Section selection** uses `IMAGE_SCN_MEM_EXECUTE` (`0x20000000`), matching
`extract.py`. An earlier version tested `0x20` (`IMAGE_SCN_CNT_CODE`), which selects a
different set of sections.

## Pipelines, one binary task

Four routes from a PE file to a prediction share one disassembler and one
cohort. The CNN-ViT scripts (`asm_parser.py`, `model_train.py`,
`stratified_split.py`) are on the `yanping` branch under `CNN-ViT/`; this
branch drives them through `cnn_vit_pipeline/`.

```
                         asm_parse.py
        PE tree  ------------------------------>  .asm tree
                                                    |
                  asm_parser.py                     |   asm_tool/asm_to_opcodes.py
        PNG + ViT mask  <-----------------------    +-------------->  opcode .txt
                |                                                         |
        model_train.py (CNN-ViT, yanping)                 llm_features_pipeline (BPE/WPC + w2v/BERT)
                |                                                         |
        cnn_vit_pipeline/  (cohort split, metrics)           results/exp{A,B,A_cohort,B_cohort,C,D}

        PE tree  --ember_pipeline/extract_features.py-->  537-dim vectors --ember_pipeline/train_eval.py--> results/ember/
```

Every harness applies the same cohort and the same family-disjoint split
(`cnn_vit_pipeline/cohort.py` is the single implementation), so the four
pipelines are scored on identical samples. The original EMBER and CNN-ViT
runs did not exclude .NET or packed files and used random splits; the
`--variant branch` mode of the EMBER harness reproduces that protocol so the
two can be compared side by side.

**Both are binary: 0 = goodware / benign, 1 = ransomware.** Both upstream
projects were multi-class and had to be converted:

| upstream | was | now |
|---|---|---|
| `Adversarial_Evaluation_CNN-ViT_Malware_Classifier` | BIG 2015, 9 families | `num_classes = 2` |
| `Tokenization-Testing-for-Malware-Data` | multi-class framing, `New/` on BIG 2015 | binary (its current loader already maps `mal_* -> 1`, `good_* -> 0`) |

The ransomware filenames carry a family prefix (`avaddon_<sha256>`). It is used
**only as a train/test split group**, never as a label — see
[docs/tokenization_audit.md](docs/tokenization_audit.md) §0.

## asm_tool/

`asm_to_opcodes.py`, `consistency_check.py`, `requirements.txt`,
`requirements-vm.txt`, the vendored manylinux wheels and the offline VM
instructions. Full detail in [asm_tool/README.md](asm_tool/README.md).

The one thing to know: `asm_parse.py` emits `0x00401000:  mov	eax, ebx` and
`extract.py` emits `mov eax, ebx`. The tokenizer folds `0x...` into `<HEX>`, so
mixing a raw `.asm` tree with `LLM_Features` puts a `<HEX>:` prefix on every
line of one class and none of the other — a perfect, content-free shortcut.
**Run `asm_to_opcodes.py` before combining them.**

`consistency_check.py` proves the two disassemblers agree: on 10 goodware
binaries (5 x86, 5 x64) the mnemonic sequence and the normalized instruction
sequence match exactly.

The VM wheels are **manylinux2014_x86_64**, not host wheels — the VM is Ubuntu.
`asm_tool/wheels/cp38`, `cp310` and `cp312` each hold `pefile` and `capstone`
and nothing else; `wheels/` is gitignored and travels by file copy.

## llm_features_pipeline/

One config, one command, both experiments.

```bash
python llm_features_pipeline/run_pipeline.py --dry-run        # resolve splits only
python llm_features_pipeline/run_pipeline.py                  # full run
python llm_features_pipeline/run_pipeline.py --experiment expB
python llm_features_pipeline/run_pipeline.py --results-dir /tmp/repro   # reproduce a run
                                                     # without overwriting results/
python llm_features_pipeline/check_schema.py         # feature-directory schema equality
```

Needs gensim/transformers, i.e. Python 3.12 — the repo's own scripts run on
3.14 but gensim has no 3.14 wheel. Point it at a 3.12 interpreter:

```bash
"../Tokenization-Testing-for-Malware-Data/.venv/Scripts/python.exe"     llm_features_pipeline/run_pipeline.py
```

`Tokenization-Testing-for-Malware-Data` is **imported, not copied**
(`normalize_instruction`, `train_tokenizer`, `build_word2vec_embeddings`,
`build_bert_embeddings`), so there is one implementation of each. Only its data
loader is replaced, because that loader discards filenames, walks directories
unsorted while the label vector is matched to the embedding matrix by row
position, and cannot keep related samples out of opposite splits.

| experiment | ransomware | goodware |
|---|---|---|
| `expA` | Mendeley `mal_train`/`mal_test` | Mendeley `good_train`/`good_test` |
| `expB` | **the same files, the same split** | `Goodware_Balanced`, group-split by source project |

Class sizes are matched exactly (1,116 / 131 goodware, 975 / 382 ransomware) and
the ransomware rows of `results/expA/splits.csv` and `results/expB/splits.csv`
are verified identical, so A vs B varies the goodware *source* and nothing else
about the sample counts. It does **not** hold the goodware architecture mix
constant — see below.

Output per experiment in `results/<name>/`: `metrics.json` (accuracy, balanced
accuracy, per-class and macro precision/recall/F1, ROC-AUC, FPR, confusion
matrix, per-class support, per-architecture goodware recall, the sample counts
and the config that produced them), `sample_counts.json` (including
exact-duplicate leak rates and the cross-source dedup report), `splits.csv`,
`config_used.yaml`, plus `results/summary.md` comparing the two.

### Splits are not random, on purpose

`mal_train` and `mal_test` are **family-disjoint** in the Mendeley release: 25
families in train, 15 different ones in test, zero overlap. A stratified 80/20
re-split would undo that — `dharma` alone contributes 45 byte-identical samples —
and turn the task into near-duplicate retrieval. `config.yaml` sets
`split.ransomware: preserve_mendeley` and reuses the identical ransomware split
in both experiments.

`Goodware_Balanced` has no such boundary, so one is built: whole source projects
(`entry_id`) go to one side only, apportioned per bucket so `everyday`,
`hard_negative` and `system` keep their 54/25/21 proportions in both splits.
Verified on the actual split: pool 53.9 / 24.9 / 21.2, train 53.9 / 24.8 / 21.2,
test 54.2 / 24.4 / 21.4, with zero `entry_id` straddling the two sides.

**Two deviations from the plan's "same stratified 80/20 split procedure", both
deliberate and both disclosed in `results/summary.md`:**

1. *The ransomware split is the release's family-disjoint one*, as above. It is
   reused **verbatim and identically in both experiments**, so it contributes
   nothing to the A-vs-B difference. Ransomware test leakage on the shipped
   split is 1/382; a random re-split would be far higher.
2. *Exp A's goodware split is also the release's own*, kept as shipped so A
   stays the published baseline. It has no group discipline, and that is exactly
   what makes 64/131 of `good_test` a verbatim copy of a training file.

Exp B's goodware side **is** split the way the plan intends: grouped by
`entry_id`, stratified by bucket, fixed seed, counts matched to Exp A exactly.

Cross-source dedup is **enforced at load time**, not assumed: any
`Goodware_Balanced` file whose source-binary sha256 appears in
`mendeley_goodware_sha256.json`, or whose opcode stream is byte-identical to a
Mendeley goodware feature file, is dropped and named in the log. Today it
removes 0 of 1,343 — the guard is there so a corpus refresh cannot silently
reintroduce an overlap.

### Reproducibility

Two things had to be fixed before a re-run reproduced the committed numbers, and
both were found by actually re-running:

* **`embedding.w2v.workers` must stay 1.** gensim's default of 4 races its worker
  threads over the corpus, so a fixed `seed` does *not* give a fixed model:
  measured, two runs at `workers=4` differ by up to 3.8e-2 per component, while
  `workers=1` is bit-identical.
* **The WordPiece trainer has no seed and picks a different vocabulary in every
  process** (Rust `HashMap` order — `PYTHONHASHSEED` and `RAYON_NUM_THREADS=1`
  both make no difference). Two runs of the identical command gave RF/WPC
  macro-F1 0.6044 and 0.6629. Mitigated by caching the trained tokenizer under
  `results/tokenizers/` with a fingerprint of its training corpus and reusing it
  whenever that corpus is unchanged. Delete that directory to force a retrain —
  and expect the WPC row to move when you do.

With those in place, two runs of `--experiment expB` are bit-identical on every
model. Everything else is seeded; full inventory in
[docs/tokenization_audit.md](docs/tokenization_audit.md) §1.7–§1.10. A run takes
a few minutes per experiment; verify with `--results-dir` and diff the
`metrics.json` files.

### Architecture is a confound, and it is not controlled

The ransomware side is 96% x86 in train and 76% in test; the Mendeley goodware
is 57% x86; Exp B's goodware is 20% x86. Bitness survives normalization through
the operands (`rbp`, `r8`–`r15`, rip-relative addressing), so **the A-vs-B gap
varies goodware source and goodware architecture mix together**.
`results/summary.md` carries the full table plus a per-architecture goodware
recall breakdown for Exp B, whose architecture comes from `opcode_manifest.csv`.
**It is not a hypothetical: Exp B recalls x64 goodware at 0.99–1.00 and x86
goodware at 0.43–0.70 across all three models.** The models are partly reading
bitness, so the A-vs-B delta cannot yet be read as a statement about goodware
quality.

The breakdown cannot be produced for Exp A's goodware test set — those 131
binaries are not present locally and the feature files carry no architecture —
nor for either ransomware side, whose binaries are VM-only. Producing the
ransomware half needs `check_arch.py` run inside the VM.

## tests/

```bash
python tests/make_fixtures.py --corpus ../Goodware_Balanced
python -m pytest tests/ -q
```

90 tests. `tests/test_asm_parse.py` (31) runs over a benign fixture corpus
built from the VirusTotal-clean `Goodware_Balanced` set plus four synthesised
cases, including `highentropy.bin`, a copy of `pe64.bin` with its code section
overwritten by `os.urandom` — the positive case for the entropy packing flag,
never packed or executed. `tests/test_tokenization_pipeline.py` (37) covers
`normalize_instruction` on hand-written lines and the pipeline's loader.
`tests/test_cohort_split.py` (22) pins the shared cohort split: family
disjointness, group disjointness, counts, determinism, and the `metrics.json`
schema shared by all harnesses. Fixtures are gitignored and rebuilt on demand,
so no binaries enter git.

## docs/tokenization_audit.md

The audit of the tokenization repo and the `LLM_Features` corpus. The findings
that change how results should be read:

* **48.9% of `good_test` is a byte-identical copy of a `good_train` file.**
  `extract.py` disassembles an installer's *stub*, so PortableApps `*.paf.exe`
  files collapse to a handful of streams (the largest duplicate group is 60
  files, 44 train / 16 test).
* **Two files carry one identical stream under both labels** — two
  installer-wrapped `makop` samples and two goodware installers.
* `mal_train`'s 975 files are **543 distinct streams**; `dharma` contributes 45
  identical ones.
* `Goodware_Balanced` duplication is **5.4%** against `LLM_Features`' 39.6%, and
  the two goodware sources share **zero** SHA-256.

## check_arch.py
Reports the architecture mix of a PE directory. **Read-only** — headers are parsed,
nothing is executed or modified. Use it on the ransomware set: the Mendeley
`p3v94dft2y` description does not state whether its samples are 32- or 64-bit, and if
one class is predominantly x86 while the other is x64, `machine` becomes a shortcut
feature and the model can separate the classes without learning anything about behaviour.

### Usage
```
python3 check_arch.py --dir <ransomware dir> --compare ../Goodware_Balanced/corpus_index.csv
```
Prints the x86/x64 split, .NET and UPX counts, and grades the gap against the compared
corpus (>25 points = warning). `--csv` writes a per-file report.

The walk excludes the same paths `asm_parse.py` excludes (`_upx_packed/`, `.tools/`,
`flagged/`, `quarantined/`, index files), so both tools describe the same corpus — on
`Goodware_Balanced` that is 1,500 PEs, 75.0% x64, 70 .NET, 0 UPX. Counting everything
instead gave 1,525 PEs and 25 "UPX packed", which are the packed originals in
`_upx_packed/` plus `.tools/upx.exe`. `--no-skip` restores that older behaviour.

## cnn_vit_pipeline/

Rerun harness for the `yanping` CNN-ViT model on the shared cohort. Read-only
wrappers: it imports `asm_parser.py` / `model_train.py` from the `yanping`
checkout (path in its README) and never edits them.

```bash
python cnn_vit_pipeline/build_dataset.py --dataset mendeley --out <dir>   # PNG+mask -> train/val/test by cohort split
python cnn_vit_pipeline/train_eval.py    --data <dir> --epochs 80 --device cpu
```

`build_dataset.py` maps every PNG back to a SHA-256 through the
`asm_output/<tree>/asm_manifest.csv` files, applies the cohort split, and
materialises the folders `model_train.py` expects (hard links). `val` is a
fixed-seed, group-aware 10% of train. It refuses to run the test fold until
the VM-side trees exist (`vm_package/README.md`). Measured on this host's CPU:
~62 ms per sample per epoch, i.e. ~2 minutes per epoch on the full cohort,
1–3 h per dataset with early stopping. Details and what differs from the
original `stratified_split.py` protocol: [cnn_vit_pipeline/README.md](cnn_vit_pipeline/README.md).

## ember_pipeline/

The EMBER branch's feature extractor (`ember_extractor.py`, `sanitizer.py`,
verbatim apart from None-guards for LIEF >= 1.0) plus a per-file driver that
keys every 537-dim vector by SHA-256 and writes a manifest row for every input,
and a LightGBM harness with two variants:

```bash
python ember_pipeline/extract_features.py --in <PE dir> --out ember_features/<name> --label 0|1 --set <set>
python ember_pipeline/train_eval.py --dataset mendeley --variant cohort    # shared cohort + family split
python ember_pipeline/train_eval.py --dataset mendeley --variant branch    # the EMBER branch's own protocol
```

`branch` reproduces what the EMBER branch did (every parsable file, random
stratified 80/20) so the effect of the cohort and the family split is
measurable. Ransomware and test-goodware vectors come from the VM.
[ember_pipeline/README.md](ember_pipeline/README.md).

## vm_package/

The offline runbook for everything that has to happen inside the VM
(ransomware and test-goodware `.asm` trees for CNN-ViT, EMBER vectors, and the
full `upx -d`'d goodware-train set), what comes back, and the one-command host
runs that follow. `make_package.py` zips the scripts and vendored wheels.

## manifests/

Text manifests for the shared, gitignored corpora, so every number in
`results/` can be traced to a SHA-256 without the binaries:
`cohort/` (the cohort definition), `goodware_balanced/` and `mendeley_goodware/`
(`asm_parse.py` manifests), `mendeley_unified/` (the revised extractor's manifest
of the full Mendeley corpus from the VM), `llm_features_revised*/` (which rows
became revised feature files and why), `ember_features/` (extraction manifests).
