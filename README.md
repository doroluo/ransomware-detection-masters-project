## Dataset
https://data.mendeley.com/datasets/p3v94dft2y/3

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

## Two pipelines, one binary task

This repo now carries two routes from a PE file to a prediction, sharing a
disassembler:

```
                         asm_parse.py
        PE tree  ------------------------------>  .asm tree
                                                    |
                  asm_parser.py                     |   asm_tool/asm_to_opcodes.py
        PNG + ViT mask  <-----------------------    +-------------->  opcode .txt
                |                                                         |
        model_train.py (CNN-ViT)                          llm_features_pipeline (BPE/WPC + w2v/BERT)
```

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

One config, one command, six experiments.

```bash
python llm_features_pipeline/run_pipeline.py --dry-run        # resolve splits only
python llm_features_pipeline/run_pipeline.py                  # all six
python llm_features_pipeline/run_pipeline.py --experiment expA         # baseline
python llm_features_pipeline/run_pipeline.py --experiment expB         # + Goodware_Balanced
python llm_features_pipeline/run_pipeline.py --experiment expA_cohort  # A, cohort-filtered
python llm_features_pipeline/run_pipeline.py --experiment expB_cohort  # B, cohort-filtered
python llm_features_pipeline/run_pipeline.py --experiment expC         # revised features, Mendeley goodware
python llm_features_pipeline/run_pipeline.py --experiment expD         # revised features, Goodware_Balanced
python llm_features_pipeline/run_pipeline.py --summary-only   # rebuild summary.md from metrics.json
python llm_features_pipeline/run_pipeline.py --results-dir /tmp/repro   # reproduce a run
                                                     # without overwriting results/

# feature-directory schema equality (candidate vs reference)
python llm_features_pipeline/check_schema.py
python llm_features_pipeline/check_schema.py \
  --candidate "C:/Users/chaoa/Downloads/LLM_Features_Revised/Features_Extraction/mal_train" \
  --reference "C:/Users/chaoa/Downloads/LLM_Features/Features_Extraction/good_train"
python llm_features_pipeline/check_schema.py \
  --candidate "C:/Users/chaoa/Downloads/LLM_Features_Revised_Balanced/good_all" \
  --reference "C:/Users/chaoa/Downloads/LLM_Features_Revised/Features_Extraction/good_train"
```

`expB_cohort` and `expD` read `results/expB/splits.csv` for their goodware
membership, so `expB` has to have run first — the config order guarantees it in
a full sweep. A single-experiment run leaves `summary.md` alone rather than
replacing a six-experiment file with a sixth of one; rebuild it with
`--summary-only`.

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

| experiment | ransomware | goodware | sample set | feature form |
|---|---|---|---|---|
| `expA` | Mendeley `mal_train`/`mal_test` | Mendeley `good_train`/`good_test` | as shipped | full instruction per line |
| `expA_cohort` | the same files, cohort-filtered | the same files, cohort-filtered | cohort | full instruction per line |
| `expC` | `LLM_Features_Revised` | `LLM_Features_Revised` | cohort | **mnemonic only** |
| `expB` | **identical to `expA`** | `Goodware_Balanced`, group-split by source project | as shipped | full instruction per line |
| `expB_cohort` | cohort-filtered | `expB`'s membership ∩ cohort | cohort | full instruction per line |
| `expD` | `LLM_Features_Revised` | `LLM_Features_Revised_Balanced`, `expB`'s membership | cohort | **mnemonic only** |

Two variables, separated. The **cohort** (`cohort_mendeley.csv` /
`cohort_balanced.csv`) drops .NET, entropy-packed, UPX-unrecoverable, no-code,
odd-architecture, broken and duplicate samples and the `thanos` family. The
**revised** feature form comes from `extract_unified.py` (capstone skip-data
sweep, uncapped) via `asm_tool/mn_to_features.py`, which applies the cohort
filter as it writes — so `expC`/`expD` are cohort-only by construction and
differ from `expA_cohort`/`expB_cohort` in the feature form alone. `expA` → `C`
and `expB` → `D` move both at once and cannot be interpreted on their own; the
four one-variable contrasts are the point.

Test-set **macro-F1**, from `results/summary.md`. The last two columns are
floors: two rules that read no opcode at all, scored on the same test set —
`majority` calls every file ransomware, `x86 → ran` reads only the architecture
recorded in the cohort CSV. A model cell is evidence about opcodes only where it
clears both.

| experiment | test n | RF / WPC | MLP / WP | SVM-RBF / SW | *floor* majority | *floor* x86 → ran |
|---|---|---|---|---|---|---|
| `expA` | 513 | 0.7496 | 0.8098 | 0.7398 | 0.4268 | 0.4266 |
| `expA_cohort` | 491 | 0.7166 | 0.7848 | 0.7595 | 0.4244 | 0.4335 |
| `expC` | 491 | 0.7660 | **0.9260** | 0.8779 | 0.4244 | 0.4335 |
| `expB` | 513 | 0.6629 | 0.6756 | 0.5862 | 0.4268 | *0.6799* |
| `expB_cohort` | 489 | 0.6482 | 0.5901 | 0.5396 | 0.4254 | *0.7048* |
| `expD` | 485 | 0.5530 | 0.5972 | **0.8439** | 0.4274 | *0.7113* |

**Reading it.** The **cohort filter** moves macro-F1 by −0.0128 on average
across the three matched rows of `expA` → `expA_cohort` (range −0.0330 to
+0.0197) and by −0.0489 across `expB` → `expB_cohort` (−0.0854 to −0.0147). It
removes 104 and 102 files respectively, overwhelmingly entropy-packed ransomware
whose opcode stream is the packer's stub rather than the payload's — so the
small negative move is the price of deleting a trivially separable group of
positives, not a degradation. The **traditional → revised** change moves it by
+0.1030 across `expA_cohort` → `expC` (+0.0494 to +0.1412) and +0.0720 across
`expB_cohort` → `expD` (−0.0952 to +0.3043). Two things move inside that one
contrast and should not be conflated: operands are gone (the distinct-line
vocabulary collapses to 466, taking the most direct architecture tell with it),
and the sweep itself is different (`extract_unified.py` skips undecodable bytes
instead of stopping at the first one, so the 5,000 instructions come from
further into the binary). `expC`'s jump is not leakage: its `mal_test` shares
**no** opcode stream and no source-binary sha256 with training, and its
goodware-side leak (62/129) is slightly *smaller* than `expA`'s (64/131) — see
[docs/tokenization_audit.md](docs/tokenization_audit.md) §2.7.

**The caveat that outranks all of it.** In `expB`, `expB_cohort` and `expD` the
architecture-only floor is worth 0.68–0.71 macro-F1, and **eight of those nine
model cells do not beat it**; only `expD`'s SVM-RBF/SW row (0.8439) clears it.
In `expA`/`expA_cohort`/`expC` the same rule scores only 0.43, level with the
majority-class floor and with a balanced accuracy of 0.43–0.45 (worse than
chance) — but
that is not an all-clear: Mendeley `good_test` is ~91% x86 against
`good_train`'s ~57%, so the bitness shortcut is learned in training and
*misfires* on the test set. What it costs is visible per slice: `expA`'s x64
macro-F1 is 0.2865–0.3066 against 0.8724–0.9634 on x86, with x64 ransomware
recall of 0.19–0.22. Until the architecture mixes are matched, no number in the
table is a statement about opcodes alone.

Class sizes are matched exactly between `expA` and `expB` (1,116 / 131 goodware,
975 / 382 ransomware) and the ransomware rows of `results/expA/splits.csv` and
`results/expB/splits.csv` are verified identical, so A vs B varies the goodware
*source* and nothing else about the sample counts. It does **not** hold the
goodware architecture mix constant — see below.

Cohort membership is joined to feature files on `<family>_<filename>.txt`, never
on sha256: 25 of the 1,408 Mendeley ransomware rows record a sha256 that differs
from the sha256 in their own filename. Measured over the 1,357 ransomware
feature files, a sha256-first join leaves **25 unmatched** and is **ambiguous
for 9 more** (each matching 2–3 rows, the extras all `tag:dup, in_cohort=0`
against a `tag:plain, in_cohort=1` row) — so it would silently drop files the
cohort keeps. The filename join matches all 1,357 and claims no row twice.
`data.Cohort` and `tests/test_tokenization_cohort.py` pin this.

Output per experiment in `results/<name>/`: `metrics.json` (accuracy, balanced
accuracy, per-class and macro precision/recall/F1, ROC-AUC, FPR, confusion
matrix, per-class support, **per-architecture metrics for both classes**,
**per-family ransomware recall**, the sample counts and the config that produced
them), `sample_counts.json` (exact-duplicate leak rates, the cross-source dedup
report, the cohort annotation/filter reports, and architecture and family counts
for both classes and both splits), `splits.csv` (now carrying `arch`, `family`,
`cohort_tag`, `in_cohort`), **`predictions.csv`** (one row per model × tokenizer
× embedding × mask × test file, so a new question about an old run does not need
a re-run), `config_used.yaml`, plus `results/summary.md` comparing all six.

### Splits are not random, on purpose

`mal_train` and `mal_test` are **family-disjoint** in the Mendeley release: 25
families in train, 15 different ones in test, zero overlap. A stratified 80/20
re-split would undo that — `dharma` alone contributes 45 byte-identical samples —
and turn the task into near-duplicate retrieval. `config.yaml` sets
`split.ransomware: preserve_mendeley` and reuses the identical ransomware split
in all six experiments.

`Goodware_Balanced` has no such boundary, so one is built: whole source projects
(`entry_id`) go to one side only, apportioned per bucket so `everyday`,
`hard_negative` and `system` keep their 54/25/21 proportions in both splits.
Verified on the actual split: pool 53.9 / 24.9 / 21.2, train 53.9 / 24.8 / 21.2,
test 54.2 / 24.4 / 21.4, with zero `entry_id` straddling the two sides.

**Three deviations from the plan's "same stratified 80/20 split procedure", all
deliberate and all disclosed in `results/summary.md`:**

1. *The ransomware split is the release's family-disjoint one*, as above. It is
   reused **verbatim and identically in all six experiments**, so it
   contributes nothing to any difference between them. Ransomware test
   leakage on the shipped split is 1/382 on the traditional features and
   **0/362** on the revised ones; a random re-split would be far higher.
2. *Exp A's goodware split is also the release's own*, kept as shipped so A
   stays the published baseline. It has no group discipline, and that is exactly
   what makes 64/131 of `good_test` a verbatim copy of a training file.
3. *`expB_cohort` and `expD` do not re-split at all.* They read `expB`'s
   committed goodware membership out of `results/expB/splits.csv` and intersect
   it with the pool in front of them (`data.reuse_split`). Re-running
   `group_split` on a pool the cohort filter has changed would move whole source
   projects across the train/test line, and the result would then differ from
   `expB` for two reasons at once — which is the one thing these four
   experiments exist to avoid.

Exp B's goodware side **is** split the way the plan intends: grouped by
`entry_id`, stratified by bucket, fixed seed, counts matched to Exp A exactly.

Cross-source dedup is **enforced at load time**, not assumed: any
`Goodware_Balanced` file whose source-binary sha256 appears in
`mendeley_goodware_sha256.json`, or whose opcode stream is byte-identical to a
Mendeley goodware feature file, is dropped and named in the log. On the
traditional features it removes 0 of 1,343. On the **revised** (mnemonic-only)
features it removes **13 of 1,337**: dropping the operands makes six Inno
Setup / NSIS installer stubs byte-identical to a Mendeley goodware file that the
full-instruction form kept distinct. That is the §2.1 installer-stub problem
getting *more* visible, not less, and it is why `expD`'s goodware counts sit a
little below `expB_cohort`'s (1,112 / 123 against 1,113 / 127).

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

With those in place, **all six experiments were re-run end to end** into a
scratch tree (14 September 2026, warm tokenizer cache — every run logged
`reusing cached WPC tokenizer`). `splits.csv`, `predictions.csv`,
`sample_counts.json` and `config_used.yaml` came back byte-identical in all six,
and `metrics.json` differed only in `elapsed_seconds` — plus, in `expB_cohort`,
the recorded path of the `splits.csv` it reuses, which pointed at the scratch
copy of `expB`. Everything else is seeded; full inventory in
[docs/tokenization_audit.md](docs/tokenization_audit.md) §1.7–§1.10. A run takes
2–5 minutes per experiment; verify with `--results-dir` and diff the
`metrics.json` files.

The WPC nondeterminism comes from the tie-break at the vocabulary size cutoff.
On the **revised** corpora that cutoff is never reached — the distinct-mnemonic
vocabulary is a few hundred against `vocab_size: 1000`. Tested, not assumed:
`expC` was re-run twice with the tokenizer cache pointed at an empty directory
that was deleted between the runs, so each trained its own vocabulary in its own
process. The vocabularies still differ slightly (12 of 941 entries), but every
difference is an unused fragment of a mnemonic that is a whole word anyway, and
both runs produced a `predictions.csv` **byte-identical** to the committed one.
The cache is still used, and is still required for the four traditional-feature
experiments, where the cutoff *is* reached.

### Architecture is a confound, and it is now measured on every side

The ransomware side is 96% x86 in train and 76–80% in test; Mendeley goodware is
57% x86 in train but **91% in test**; Goodware_Balanced is 18–23% x86. Bitness
survives normalization through the operands (`rbp`, `r8`–`r15`, rip-relative
addressing) in the traditional feature form, and even the mnemonic-only form
leaves the x64-only instruction set (`vpxor`, `rorx`, `cmpxchg16b`, the AVX-512
mnemonics) visible as a bare mnemonic.

The gap the earlier version of this section had to leave open is closed. The
cohort CSVs carry an architecture for **every input binary of every set**,
`good_test` and both ransomware splits included, so `results/summary.md` now
prints, for all six experiments: the full x86/x64 table per split and class,
test metrics for **both classes inside each architecture slice**, and the
`x86 → ransomware` floor. Three things it shows:

* In `expB`, `expB_cohort` and `expD` the architecture-only rule scores
  0.68–0.71 macro-F1 and **eight of nine model rows do not beat it**.
* In `expA`, `expA_cohort` and `expC` that rule scores only 0.43, with a
  balanced accuracy of 0.43–0.45 — *worse than chance* — because `good_test` is
  91% x86, like the ransomware. The shortcut is still
  learned from training (57% x86 goodware against 95% x86 ransomware); it simply
  misfires on test, which is what the x64 slices show: `expA` scores 0.2865–
  0.3066 macro-F1 on x64 against 0.8724–0.9634 on x86, recalling x64 ransomware
  at 0.19–0.22.
* Splitting the test set by bitness costs up to 0.5032 macro-F1 against the
  pooled score across the 18 model/experiment pairs where both slices hold both
  classes. That is the size of the shortcut the pooled number was buying.

So the A-vs-B delta still cannot be read as a statement about goodware quality —
but it is no longer an untested worry, and the per-slice numbers are the honest
version of every headline score in this section. Matching the architecture mixes
(or sampling a bitness-balanced test set) is the remaining work; the counts
needed to do it are in `results/*/splits.csv`.

## tests/

```bash
python tests/make_fixtures.py --corpus ../Goodware_Balanced
python -m pytest tests/ -q
```

`tests/test_asm_parse.py` is 31 tests over a benign fixture corpus built from
the VirusTotal-clean `Goodware_Balanced` set, plus four synthesised cases — including
`highentropy.bin`, a copy of `pe64.bin` with its code section overwritten by
`os.urandom`, which is the positive case for the entropy packing flag and was
never packed or executed. Fixtures are gitignored and rebuilt on demand, so no
binaries enter git.

## docs/tokenization_audit.md

The audit of the tokenization repo and the `LLM_Features` corpus. The findings
that change how results should be read:

* **48.9% of `good_test` is a byte-identical copy of a `good_train` file.**
  `extract.py` disassembles an installer's *stub*, so every PortableApps
  `*.paf.exe` yields the same opcode stream.
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

## asm_parser.py
Takes existing .asm files and convert it into 256x256 greyscale image with ViT attention mask.
1. Parse each instruction into (opcode/operand/API) using token map
2. Flatten to 65536 tokens, truncate and padding to fit
3. Save .png and vitmask.npy under Goodware & Ransomware folder
4. Sort by labels

### Structure & Definition
.png - 256x256 grayscale image. Each pixel is a token ID from the asm file(opcode/operand/API), padded or truncated to fit.
_vit_mask.py - 16x16 array read code(1) and padding(0)

### Reserved token IDs — read this before changing the tokenizer

| id | name | meaning |
|---|---|---|
| 0 | `PADDING` | an operand slot that does not exist, e.g. `ret` has no operands |
| 1 | `UNKNOWN_OPCODE` | mnemonic not in `TOKEN_MAP` |
| 2 | `NOP_SLED` | |
| 3 | `END_PAD` | canvas space past the end of the file's instruction stream |

**0 and 3 are deliberately different.** Both mean "nothing here", but 0 is a property
of a single instruction and 3 is a property of file length. If they shared an id, the
CNN branch could read file size as instruction shape — the ViT mask is the only thing
that separates them, and the CNN branch never sees it.

### Padding is constant, not noise (changed)

Short files used to be padded with Gaussian noise shaped to each file's own token
statistics. That was replaced with constant `END_PAD` for two reasons:

1. **It leaked.** The padded fraction varies enormously and tracks things the model
   must not key on. Measured over the 1,343-file goodware ASM set, median canvas fill
   is **8.9% for x86 vs 73.2% for x64**, and **93.2% for hard negatives vs 47.8% for
   system binaries**. The noise region encoded architecture and class almost perfectly
   while carrying no information about the code.
2. **It was non-deterministic.** `np.random` was unseeded and ran inside a
   `ProcessPoolExecutor`, so the same `.asm` produced a different image on every run —
   results were unreproducible and the noise acted as uncontrolled augmentation.

Nothing downstream depended on the old behaviour: `model_train.py` masks the
transformer from `_vit_mask.npy`, not from pixel values.

> Note for a future pass: the shift augmentation in `model_train.py` (~line 66) fills
> with `torch.zeros`, i.e. pixel value 0 = `PADDING`. For consistency it should fill
> with 3. The mask is shifted alongside it so the ViT branch is unaffected either way.

    training_dataset_sorted/

        ├── Class_0_Goodware/     # 42 samples right now

            ├── <name>.png

            └── <name>_vit_mask.npy

        └── Class_1_Ransomware/   # 50 samples
    
            ├── <hash>.png

            └── <hash>_vit_mask.npy

### Usage
```python3 asm_parser.py```

## Adversarial evaluation (ported)

`generate_adversarial_train.py`, `generate_adversarial_test.py`,
`vulnerability_test.py`, `full_model_retrain.py` and
`vulnerability_test_retrained_model.py` are ported from
`Adversarial_Evaluation_CNN-ViT_Malware_Classifier`. They were brought into this
repo rather than run in place, because that repo ships **its own copy of
`asm_parser.py`** and the two had diverged.

> **Do not add a second tokenizer to this repo.** The adversarial scripts
> `from asm_parser import ...`, which must resolve to the one canonical
> `asm_parser.py` here. If clean images are built with one tokenizer and
> adversarial images with another, the difference shows up as "the model
> degrades under attack" when it is really just a different padding scheme —
> which corrupts the exact number this evaluation exists to measure.

### What changed from upstream

| change | why |
|---|---|
| `num_classes` 9 → 2 | upstream targets BIG 2015's 9 malware families; this task is binary |
| flat `os.listdir` → recursive `rglob` | `asm_parse.py` mirrors class folders, so a flat listing finds nothing and the run exits reporting 0 files |
| basename → `asm_id()` | 24 samples share a basename across buckets (`node.exe`, `liblzma.dll`, …); mapping on basename silently drops one of each pair |
| Gaussian noise padding → `END_PAD` | keeps adversarial images tokenized identically to clean ones |
| emoji removed from `print()` | Windows consoles use cp1252 and raise `UnicodeEncodeError` mid-run |

Attack tactics are unchanged: NOP sleds, dead-code injection, patch obfuscation,
instruction substitution and API obfuscation, each at three intensities — 15
attack variants per sample.

### Order of operations
```
asm_parse.py            PE tree      -> .asm tree
asm_parser.py           .asm tree    -> PNG + _vit_mask.npy, trainLabels.csv
stratified_split.py     PNG          -> evaluation_dataset_split/{train,val,test}
generate_adversarial_train.py        -> evaluation_dataset_split/train_attack_*
generate_adversarial_test.py         -> evaluation_dataset_split/test_attack_*
model_train.py                       -> best_model.pth   (clean training)
vulnerability_test.py                -> evasion_metrics_summary.{csv,md}
full_model_retrain.py                -> full_retrained_model.pth  (adversarial training)
vulnerability_test_retrained_model.py-> full_retraining_evasion_metrics_summary.*
```

### A bug fixed in `parse_asm_line`
The obfuscation engine appends `    ; instruction substitution` to every line it
rewrites. `parse_asm_line` used to reject any line where `';' in line.split()`,
so whitespace-separated comments made `;` a standalone token and the whole line
was discarded — **every substituted instruction was dropped**, and the
comment-stripping line below it was unreachable. The substitution attack was
measuring deletion, not substitution. Comments are now stripped before the
rejection test.

## stratified_split.py
Spliting the training_dataset_sorted from asm_parser.py into train/val/test while keeping the same ratio. Copies each .png and vit_mask.npy. 

### Structure
    evaluation_dataset_split/

        ├── test/

            ├── Class_0_Goodware

                └── <name>.png

                └── <name>_vit_mask.npy

            ├── Class_1_Ransomware
                
                └── <hash>.png

                └── <hash>_vit_mask.npy

        ├── train/

            ├── Class_0_Goodware

                └── <name>.png

                └── <name>_vit_mask.npy

            ├── Class_1_Ransomware
                
                └── <hash>.png

                └── <hash>_vit_mask.npy        
        
        ├── val/

            ├── Class_0_Goodware

                └── <name>.png

                └── <name>_vit_mask.npy

            ├── Class_1_Ransomware
                
                └── <hash>.png

                └── <hash>_vit_mask.npy
    
### Usage
```python3 stratified_split.py```

## model_train.py
Train the CNN-ViT malware classifier:
1. Loads .png + _vit_mask.npy from train / val / test
2. Trains HierarchicalMalwareNet (CNN stem → transformer → classifier)
3. Uses early stopping on validation loss
4. Evaluates on the test set (accuracy, report, confusion matrix)
5. Saves best_model.pth and clean_model_early_stop.pth

### Dependency
```pip install scikit-learn```
```python3 -m pip install torch torchvision```

### Usage
```python3 model_train.py```