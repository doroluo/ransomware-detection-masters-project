# Tokenization audit

Audit of `Tokenization-Testing-for-Malware-Data` (`Tokenization/tokenization.py`,
`Embedding/build_masked_embeddings.py`, `Classification/GridSearch.py`) as used
for the binary ransomware-vs-goodware task, plus the `LLM_Features` corpus those
scripts consume.

Everything below was measured, not inferred. Reproduce with:

```bash
python -m pytest tests/ -q
python asm_tool/consistency_check.py --corpus ../Goodware_Balanced -n 10
python llm_features_pipeline/check_schema.py
python llm_features_pipeline/run_pipeline.py --dry-run
```

Every number in §1.3–§1.5 and §2.1–§2.6 was independently re-measured from the
raw files on 13 September 2026 (file contents hashed with SHA-256, counted, and
joined on sha256) as part of a second-pass audit. **Every corpus count
reproduced exactly** — file counts, unique-stream counts, leak rates, family
counts, packed/.NET counts, sha256 overlap. What did not survive that pass were
one miscount (§1.3), four *characterisations* attached to correct counts, and
one conclusion; all six are marked **[corrected]** inline (§1.1, §1.3, §2.1,
§2.4, §2.5). The second pass also found three reproducibility defects the first
pass missed — §1.7, §1.8 and §1.10 — which is why the numbers in `results/` were
regenerated. §1.10 was found only by re-running the pipeline and diffing, not by
reading code.

A third pass on 14 September 2026 added §2.7 — the cohort CSVs and the `REVISED`
mnemonic-only corpus, and the four experiments built on them — and re-measured
what those made measurable for the first time. It closed the architecture gap
§4.2 and §5 had to leave open, and it found that eight of the nine
Goodware_Balanced model rows score below a rule that reads only the
architecture. Every `results/*/metrics.json` was re-derived from its own
`predictions.csv` and `splits.csv` and reproduced exactly, and **all six
experiments were re-run end to end** into a scratch directory with
`--results-dir`: `splits.csv`, `predictions.csv`, `sample_counts.json` and
`config_used.yaml` came back byte-identical and `metrics.json` differed in
`elapsed_seconds` alone (plus, in `expB_cohort`, the recorded path of the
`splits.csv` it reused, which pointed at the scratch copy).

A fourth pass, also on 14 September 2026, added §6 — a 787-configuration
grouped-cross-validation search over `expC` and `expD`, with the test set held
out until one final evaluation each. It found that the committed
`max_instructions = 5000` truncation is the largest CV-measurable limitation of
the representation, that subword tokenization buys nothing on mnemonic-only
input (§2.7 reached from the other end), and that **neither tuned run beats the
fixed configuration on the test set**: `expD`'s does not even clear the
architecture-only floor, and its x64 ransomware recall of 0.1667 is §5's open
architecture item reproducing itself under a better search.

---

## 0. Binary, not multi-class

Both upstream repos were built for multi-class malware-family work:

| repo | original target | here |
|---|---|---|
| `Adversarial_Evaluation_CNN-ViT_Malware_Classifier` | BIG 2015, 9 families (`num_classes = 9`) | `num_classes = 2` |
| `Tokenization-Testing-for-Malware-Data` (`New/` tree) | same BIG 2015 heritage | binary |

The *current* `Tokenization/tokenization.py` is already binary — it maps
`mal_* -> 1`, `good_* -> 0` — so no label surgery was needed there. What did
need care is the **family prefix** in the ransomware filenames
(`avaddon_<sha256>.txt`, 25 families in train, 15 in test). It is tempting to
use it as a target. It is used here **only as a split group**, never as a label.
`llm_features_pipeline/data.py` enforces this: `Sample.label` is `0|1`, and the
family goes into `Sample.group`. `tests/test_tokenization_pipeline.py` pins it
(`test_family_is_a_group_never_a_label`, `test_labels_are_binary_and_correct`),
so a future refactor cannot quietly turn the prefix back into a target.

---

## 1. Findings in the tokenization code

### 1.1 Row order is not deterministic — and row order *is* the label mapping

`_iter_labeled_instruction_lists` walks `folder_path.iterdir()`, which is
filesystem order, not sorted order. That would be a cosmetic complaint except
for how the pieces fit together downstream:

* `tokenization.py` writes `SW_train.pkl`, `WP_train.pkl`, `BPE_train.pkl` and
  `WPC_train.pkl` from **three separate** `iterdir()` walks of the train
  folders — `load_raw_with_operands` (which BPE and WPC share),
  `load_sw_dataframe` and `load_wp_dataframe` — and three more for test.
  **[corrected]**: the original text said four walks; BPE and WPC come from the
  same one. Three is still three, and the hazard is unchanged.
* `build_masked_embeddings.py` produces `X` in the row order of the PKL it read.
* `GridSearch.py` takes `X` from the `.npy` and `y` from the PKL, and pairs them
  **by position**.

So if the directory listing ever changes between two of those walks — a file
added, a corpus rebuilt, a different filesystem, a different OS — `X[i]` and
`y[i]` refer to different samples and nothing raises. The failure is silent and
looks like "the model got worse."

**Fix applied:** the pipeline's loader uses `sorted(dir.glob("*.txt"))`, carries
the filename in the frame, and asserts `len(X_train) == len(y_train)` at every
embed/fit boundary.

### 1.2 Filenames are discarded

The DataFrame holds only `Instructions` / `Opcodes` / `Malware` / `Label`. With
no identifier it is impossible to deduplicate by hash, to group-split, to trace
a false positive back to a binary, or to tell which files were dropped.

**Fix applied:** `file`, `source`, `group` and `split` columns travel with every
row through tokenization and embedding.

### 1.3 `normalize_instruction` is called twice per line

```python
instructions = [normalize_instruction(l) for l in lines if normalize_instruction(l)]
```

Correct, just 2x the regex work. **[corrected]** The original text put the
volume at "~13M lines". Re-counted, `LLM_Features` holds **78,725,187**
non-blank lines (good_train 24,681,771; good_test 1,832,807; mal_train
18,286,820; mal_test 33,923,789). 13M was the count *after* the 5,000-per-file
cap — but this list comprehension runs **before** the cap, so the doubled work
is on all 78.7M. (The capped figure is 9,407,052, not 13M, since most files are
shorter than 5,000 instructions.)

Not changed upstream; the pipeline's own reader calls `normalize_instruction`
once per line and stops reading at the cap, so it touches 9.4M lines once
instead of 78.7M twice.

### 1.4 Empty files are dropped silently

`if not instructions: continue` — with no filename and no counter, a file that
yields nothing simply disappears from the sample count. 33 files in the corpus
have fewer than 10 instructions and 130 have fewer than 50 (re-counted and
confirmed; no file in `LLM_Features` is actually empty, so nothing is dropped
today — but nothing would say so if it were).

**Fix applied:** dropped files are counted and named in the run log.

### 1.5 `GridSearch.py` reports accuracy only, on a 74%-malware test set

`mal_test` is 382 files and `good_test` is 131, so the test split is 74.5%
ransomware. A classifier that answers "ransomware" every time scores **0.745
accuracy**. The script prints accuracy and nothing else, and `GridSearchCV`
optimises the default scorer, which is also accuracy.

**Fix applied:** `scoring="f1_macro"` for model selection, and every result row
carries accuracy, balanced accuracy, the majority-class baseline, per-class
precision/recall/F1 with support, macro precision/recall/F1, ROC-AUC, FPR, the
confusion matrix, and per-architecture goodware recall where the architecture is
known. Verified against `GridSearch.py`: it imports only `accuracy_score` and
calls `GridSearchCV(clf, grid, cv=2, verbose=1)` with no `scoring=`, so the
default accuracy scorer is what selects its models.

### 1.6 What is *not* wrong

Worth stating, because these were the checks that could have invalidated the
whole pipeline:

* **No vocabulary leakage.** `train_tokenizer` is called with `train` only;
  Word2Vec is fit on `train_m` only; DistilBERT is frozen and pretrained.
  Re-checked line by line: `build_sequence_frame` builds `fit_df` from
  `text[df["split"] == "train"]`, and `build_word2vec_embeddings` passes only
  `train_m` as `sentences`. Test rows are *encoded* with the fitted tokenizer
  and *projected* through the fitted Word2Vec, which is inference, not fitting.
* **No cross-source duplication, and now it is enforced rather than assumed.**
  §2.5.
* **No test contamination in model selection.** `GridSearchCV` folds only the
  training matrix; the test set is touched once, to score the selected
  estimator. (The folds themselves were not seeded — see §1.8.)
* **Opcode regex coverage is a non-issue here.** The scripts never extract a
  mnemonic with a regex — they normalise the whole instruction line — so
  prefixed forms (`rep movsb`, `lock cmpxchg`) survive intact.
* **`db` / `dd` / `align` data lines cannot leak in.** That hazard belongs to
  IDA-style BIG 2015 `.asm`. Both `extract.py` and `asm_parse.py` emit capstone
  output only, so every line is a decoded instruction.

### 1.7 The results were not reproducible: Word2Vec ran on four threads

Found in the second-pass audit, and the most serious defect in the pipeline as
first written.

`Embedding/build_masked_embeddings.build_word2vec_embeddings` takes
`workers: int = 4`. `run_pipeline.py` did not pass `workers`, so it inherited
that default. gensim's `seed` fixes the initialisation and the negative-sampling
draw; it does **not** serialise the worker threads, which consume the corpus in
whatever order they win the queue. Two runs of the identical command therefore
produce different vectors. This was not inferred from the source — it was
reproduced directly.

Measured directly, 400 sentences over a 300-token vocabulary, `seed=42`:

| workers | two runs identical? | max abs difference per component |
|---|---|---|
| 4 (the inherited default) | **no** | 3.8e-2 |
| 1 | yes | 0 |

Every number in `results/` produced before this fix was therefore
unreproducible: a re-run would not land on the same macro-F1.

**Fix applied:** `config.yaml` gains `embedding.w2v.workers: 1` and
`run_pipeline.py` passes it. The tokenization repo is left alone — the default
is changed at the call site, not upstream.

### 1.8 GridSearchCV folds were cut in family order

`cv=2` as a bare integer gives `StratifiedKFold(n_splits=2, shuffle=False)`,
which slices the training matrix in row order. Row order here is
ransomware-then-goodware, sorted by filename — that is, sorted by *family*. So
fold 1 was roughly `avaddon`…`makop` and fold 2 `maze`…`zeppelin`: an
arbitrary, undocumented, and unseeded grouping of the model-selection folds.

Not a leak (the test matrix is still untouched), but not reproducible as a
*procedure* either, since it silently depends on filename ordering.

**Fix applied:** an explicit `StratifiedKFold(n_splits=cv, shuffle=True,
random_state=seed)`.

### 1.9 Seed inventory

Every stochastic component, after the fixes above:

| component | seed | deterministic? |
|---|---|---|
| WPC/BPE tokenizer training | **no seed exists**; mitigated by caching the trained tokenizer | **no** — see §1.10 |
| Word2Vec | `embedding.w2v.seed: 42`, `workers: 1` | yes (§1.7) |
| token masking | `seed` per row (`seed + i`), `seed + 10_000` for test | yes; and a no-op at `mask_rates: [0.0]` |
| GridSearchCV folds | `StratifiedKFold(shuffle=True, random_state=classification.seed)` | yes (§1.8) |
| RandomForest | `random_state=classification.seed` | yes |
| MLP | `random_state=classification.seed` | yes |
| SVC | `random_state=classification.seed` (unused: `probability=False`) | yes |
| goodware group split | `split.seed: 42`, `random.Random(seed)` over a sorted group list | yes |
| DistilBERT | frozen pretrained weights, `model.eval()`, no dropout | yes |

`n_jobs=-1` on `GridSearchCV` parallelises across *candidates*, each fit
independently and seeded, so it does not affect the outcome — unlike gensim's
`workers`, which parallelises *inside* one fit.

### 1.10 The WordPiece trainer picks a different vocabulary in every process

Found by actually re-running, which is the only way this one shows up.

After fixing §1.7 and §1.8, Exp B was run twice from the identical command with
`PYTHONHASHSEED=0`. Two of the three model/tokenizer pairs came back
**bit-identical**. The third did not:

| model / tok | uses a trained tokenizer? | run 1 macro-F1 | run 2 macro-F1 |
|---|---|---|---|
| MLP / WP (bigrams) | no | 0.6756 | 0.6756 |
| SVM-RBF / SW (raw instructions) | no | 0.5862 | 0.5862 |
| **RF / WPC (WordPiece)** | **yes** | **0.6044** | **0.6629** |

That isolates it exactly: everything downstream of the tokenizer is
reproducible, and the tokenizer is not. Confirmed directly — training WordPiece
on one fixed corpus in eight separate processes produced **two different
vocabularies**, alternating unpredictably.

It is not fixable from Python:

* `PYTHONHASHSEED` has no effect. That controls Python's `hash()`; the
  `tokenizers` package is a Rust extension and its `HashMap` iteration order is
  seeded from the OS per process.
* `RAYON_NUM_THREADS=1` and `TOKENIZERS_PARALLELISM=false` have no effect
  either — tested over eight runs, the vocabulary still flipped. It is not a
  thread-count race; it is tie-breaking at the vocabulary size cutoff.
* The trainer exposes no seed parameter.

A 0.06 swing in macro-F1 is larger than several of the differences these
experiments exist to measure, so this is not a rounding concern.

**Fix applied:** the trained tokenizer is saved to
`results/tokenizers/<experiment>_<alg>.json` with a fingerprint of the exact
training corpus, and reused whenever that corpus is byte-for-byte unchanged
(`run_pipeline.get_tokenizer`). The committed numbers are therefore exactly
reproducible from the committed artifacts. The residual nondeterminism is
confined to the first run on a genuinely new corpus, and the log says plainly
when that is happening. Delete `results/tokenizers/` to force a retrain.

This is a mitigation, not a cure. Anyone reporting WPC results on a *new*
corpus should train the tokenizer several times and report the spread.

Observed so far on Exp B, three independent trainings of the same corpus:
macro-F1 **0.6044, 0.6629, 0.6629**. The A-vs-B RF/WPC delta is about -0.13, so
the tokenizer draw is roughly half the size of the effect being measured. The
SW and WP rows, which use no trained tokenizer, are stable to the last decimal
across all three runs.

---

## 2. Findings in the `LLM_Features` corpus

These matter more than the code defects.

### 2.1 Half of the goodware test set is a verbatim copy of training data

Hashing the content of all 2,604 `.txt` files:

| | files | unique opcode streams |
|---|---|---|
| `good_train` | 1,116 | 970 |
| `good_test` | 131 | **61** |
| `mal_train` | 975 | **543** |
| `mal_test` | 382 | 208 |

Test files whose exact stream also appears in train:

| | leaked | of | rate |
|---|---|---|---|
| `good_test` | **64** | 131 | **48.9%** |
| `mal_test` | 1 | 382 | 0.3% |

Of the 64, **62** duplicate a `good_train` file and **2** duplicate a
`mal_train` file (§2.2).

**Cause.** `extract.py` disassembles the executable section of whatever file it
is given. For an installer, that is the *stub* — the NSIS or Inno Setup engine —
not the payload. PortableApps `*.paf.exe` files therefore collapse onto a handful
of streams, one per launcher build. **[corrected]** The original text said "every
`*.paf.exe` produces the same opcode stream"; re-measuring shows several distinct
streams, not one. The four largest duplicate groups in the goodware half are:

| group size | in `good_train` | in `good_test` | `*.paf.exe`-named |
|---|---|---|---|
| 60 | 44 | 16 | 59 |
| 42 | 30 | 12 | 42 |
| 35 | 31 | 4 | 35 |
| 16 | 4 | 12 | 0 |

The largest group alone puts 44 copies in train and 16 in test. The goodware half
of the task is, to a large extent, "recognise the launcher stub."

The ransomware half does not have this problem, for the reason in §2.3.

### 2.2 Two files carry one identical stream under both labels

| stream | label 0 | label 1 |
|---|---|---|
| 151,163 bytes | `good_test/root_CairoSetup_64bit.exe.txt` | `mal_train/makop_a617fdbf…` |
| 146,839 bytes | `good_test/root_CloudTierSetup.exe.txt` | `mal_train/makop_c311e0cd…` |

Byte-identical, verified. The makop samples are installer-wrapped, so
`extract.py` measured the installer in all four cases. The model is trained on
that stream as ransomware and tested on it as goodware; those two test files are
unanswerable by construction.

More importantly it means **installer-wrapped ransomware is represented in this
corpus by its installer, not by its ransomware**.

### 2.3 `mal_train` / `mal_test` is family-disjoint, and must stay that way

25 families in train (netwalker, gandcrab, phobos, darkside, avaddon, …), 15
entirely different ones in test (hive, blackcat, avoslocker, clop, …), **zero
overlap**. That is a deliberate, and good, generalisation test — and it is why
ransomware leakage is 0.3% while goodware leakage is 48.9%.

> A random stratified 80/20 re-split, which is what the worker brief asked for,
> would destroy this. `dharma` alone contributes **45 byte-identical samples**;
> `phobos` 37; `lockbit` 35. Re-splitting at random puts copies of the same
> stream on both sides and turns the task into near-duplicate retrieval.
> `config.yaml` therefore sets `split.ransomware: preserve_mendeley`, and the
> identical ransomware split is reused in **both** experiments so A and B differ
> only in their goodware.

Within `mal_train`, 110 duplicate groups exist; 109 are inside a single family
and exactly one crosses families. So 975 ransomware training files are 543
distinct code streams. All re-measured and confirmed, including the full family
lists: train is avaddon, babuk, blackmatter, conti, darkside, dharma,
doppelpaymer, exorcist, gandcrab, lockbit, makop, maze, mountlocker, nefilim,
netwalker, phobos, pysa, ragnarok, ransomexx, revil, ryuk, stop, thanos,
wastedlocker, zeppelin; test is avoslocker, bianlian, blackbasta, blackbyte,
blackcat, bluesky, clop, hive, holyghost, karma, lorenz, maui, nightsky,
playcrypt, quantum.

### 2.4 `Goodware_Balanced` does not have this problem

Same measurement on the 1,343 converted files: **1,298 unique streams from 1,343
files**, 28 duplicate groups, 73 files involved — **5.4%** versus 193 groups /
1,032 files / **39.6%** for `LLM_Features`.

**[corrected]** The original text attributed the remainder entirely to
"installers the downloader could not unpack." Re-measuring, that is most of it
but not all. The largest groups are:

| size | what they are |
|---|---|
| 8 | NSIS stubs (Bitwarden, draw.io, Obsidian, Signal …) |
| 6 | the shared `distlib` console-script launcher (`f2py`, `idna`, `numpy-config`, `pip3.11` …) — **not** an installer |
| 5, 4 | Inno Setup stubs (Audacity, GIMP, ShareX, Stellarium …) |
| 3 | WiX/Burn bootstrappers |
| 2 × many | NSIS plugin DLLs (`InstallOptions.dll`, `LangDLL.dll`) present in two buckets |

Same mechanism as §2.1 — a shared stub disassembled in place of the payload —
plus one variant of it (per-entry-point launcher executables that differ only in
an embedded path). Far rarer here because that corpus extracts installers and
indexes their payloads instead of disassembling the stub.

### 2.5 The two goodware sources do not overlap at all

SHA-256 of all 1,115 binaries in `Goodware_Training/goodware` (1,112 distinct —
three binaries are byte-identical duplicates under different names) against the
1,500 in `Goodware_Balanced/corpus_index.csv` (1,500 distinct): **0 files in
common**. The same result holds on the 1,343 rows of
`LLM_Features_Balanced/opcode_manifest.csv`, whose `sha256` column intersects
`mendeley_goodware_sha256.json` (1,112 hashes) in **0** entries.

**[corrected]** The original text concluded "no cross-source dedup is required."
A measurement taken once is not a guarantee; a corpus refresh can silently
reintroduce an overlap. It is now **enforced at load time** by
`data.dedup_goodware_sources()`, on two independent identities:

| identity | what it catches | removed today |
|---|---|---|
| source-binary sha256 (`opcode_manifest.csv` vs `mendeley_goodware_sha256.json`) | the same executable under two filenames | 0 of 1,343 |
| opcode-stream sha256 of the feature file itself | *different* binaries whose disassembly is byte-identical — the §2.1 installer-stub case, which the binary hash cannot see | 0 of 1,343 |

Anything removed is named in the run log and recorded under
`samples.cross_source_dedup` in `sample_counts.json` and `metrics.json`.

Scope, stated precisely, because the two channels have different reach:

* The **sha256 channel** covers the training goodware only. Those 1,115 binaries
  account for 1,097 of the 1,116 `good_train` feature files (matched by
  filename; 19 `good_train` files have no local binary and 18 local binaries
  have no feature file). The **`good_test` binaries are not present locally** —
  0 of 131 match — so nothing can be hashed for them.
* The **content-hash channel closes that hole.** It hashes the opcode text of
  all 1,247 Mendeley goodware feature files, `good_test` included, so a
  `Goodware_Balanced` file that produced the same stream as a `good_test` file
  would still be caught. It found none.

What remains unverified is only the narrow case of a `good_test` *binary* that
is byte-identical to a `Goodware_Balanced` binary yet disassembles differently —
which cannot happen, the disassembler being deterministic — or one under a
different filename whose stream also differs. Since `Goodware_Balanced` was
sourced from 2026 vendor releases and `good_test` is the same PortableApps-era
collection as `good_train`, an overlap there is in any case unlikely.

### 2.6 `extract.py` silently drops UPX-packed goodware unless it is pre-unpacked

`extract.py`'s own header says to run `upx -d` over the tree first. Running
`asm_parse.py` — which applies the same UPX guard — over
`Goodware_Training/goodware` flags **67 of 1,115 (6.0%) as still packed**, plus
11 IL-only .NET and 5 that disassemble to nothing — 1,032 (92.6%) survive.
Re-measured and confirmed exactly. (WRITEUP.md §3.1 counts 13 .NET and 4
entropy-packed for the same folder; it counts mixed-mode C++/CLI as .NET and has
an entropy rule, neither of which `asm_parse.py` applies. The two are consistent,
not contradictory.)

So the packed goodware in that snapshot would contribute nothing, and whoever
built `LLM_Features` must have unpacked first (only 18 binaries lack a feature
file). The point for the VM run: **if the ransomware tree is not `upx -d`'d
first, packed ransomware is silently absent from `mal_train`/`mal_test`** — and
packed samples are exactly the ones a detector most needs to see. The manifest
makes this visible; the original pipeline does not.

### 2.7 The cohort and the `REVISED` corpus

Everything in §2.7 was measured on 14 September 2026 against the trees on disk —
file contents hashed with SHA-256 and counted, the cohort CSVs and
`revised_manifest.csv` read directly, the cached tokenizer loaded from
`results/tokenizers/`. It covers the two artefacts that arrived after the first
pass and the four experiments built on them (`expA_cohort`, `expB_cohort`,
`expC`, `expD`). The counts it quotes out of `results/` are reproducible with:

```bash
python llm_features_pipeline/run_pipeline.py --experiment expC --results-dir /tmp/repro
python llm_features_pipeline/run_pipeline.py --summary-only
python -m pytest tests/test_tokenization_cohort.py -q
```

**What the two artefacts are.**

* `../asm and mm/Shared/cohort_mendeley.csv` (2,675 rows) and
  `cohort_balanced.csv` (1,500 rows): one row per *input binary*, kept or not,
  carrying `arch`, `tag`, `in_cohort` and `exclude_reason`. This is the first
  source in the project with an architecture for `good_test` and for the
  ransomware side, so §4.2 and the last three bullets of §5 are answerable for
  the first time.
* `LLM_Features_Revised/Features_Extraction` and
  `LLM_Features_Revised_Balanced/good_all`: `asm_tool/mn_to_features.py` output
  from `extract_unified.py` (capstone skip-data sweep, uncapped). Lines are
  **mnemonics only** — `mov`, not `mov eax, ebx` — and the cohort filter is
  already applied as they are written.

**What changed in the loader** (`llm_features_pipeline/data.py`,
`run_pipeline.py`):

| change | why |
|---|---|
| `data.Cohort` joins on `<family>_<filename>.txt`, never on sha256 | measured on the 1,357 ransomware feature files: a sha256-first join leaves **25 unmatched** (their filename stem appears nowhere in the `sha256` column) and is **ambiguous for 9 more**, each matching 2-3 rows. The extra rows in all 9 are `tag:dup, in_cohort=0` against a `tag:plain, in_cohort=1` row, so a sha256-first join would drop files the cohort keeps. The filename join matches all 1,357 and claims no row twice (`tests/test_tokenization_cohort.py::test_filename_join_separates_what_a_sha256_join_collides`, `::test_real_filename_join_claims_no_cohort_row_twice`) |
| annotate first, filter second; an unmatched file is **dropped**, not kept | an unmatched file is one whose packing status is unknown, and the whole point of the cohort is to know it |
| `cohort_filter: true` on the traditional corpora only | the revised trees are pre-filtered; the run asserts it rather than assuming it — all 2,509 files of `expC` and all 2,501 of `expD` carry `in_cohort == 1` |
| `reuse_goodware_split: expB` for `expB_cohort` and `expD` | re-splitting a pool the filter has changed would move whole source projects across the train/test line, so those runs would differ from `expB` for two reasons at once |
| `predictions.csv` written per run | a new question about an old run (per architecture, per family, per source) is answerable without a re-run, and a re-run is only trustworthy with a warm tokenizer cache |

**Leakage in the revised Mendeley tree**, measured exactly as §2.1 measured the
traditional one — SHA-256 of the file contents, no normalization:

| | files | unique opcode streams | traditional, files / unique (§2.1) |
|---|---|---|---|
| `good_train` | 1,114 | 972 | 1,116 / 970 |
| `good_test` | 129 | **65** | 131 / 61 |
| `mal_train` | 904 | **422** | 975 / 543 |
| `mal_test` | 362 | 211 | 382 / 208 |

Test files whose exact stream also appears in train:

| | leaked | of | rate | traditional |
|---|---|---|---|---|
| `good_test` | 62 | 129 | **48.1%** | 64 / 131 = 48.9% |
| `mal_test` | **0** | 362 | **0.0%** | 1 / 382 = 0.3% |

Of the 62, **60** duplicate a `good_train` file and **2** duplicate a
`mal_train` file — the same two makop installer streams as §2.2, which survive
the new extractor unchanged. The four largest goodware duplicate groups are
60 (44 train / 16 test), 47 (31/16), 35 (31/4) and 13 (8/5).

So the revised extractor changes the goodware leak by −0.8 points and takes the
ransomware leak to zero. **§2.1 is inherited, not repaired**: half of the
goodware test set is still a verbatim copy of training data, in `expC` exactly
as in `expA`, which is why that pair can be compared to each other but neither
can be read as a field estimate.

**Train/test disjointness, on three identities.** The traditional audit could
only check two, because `good_test`'s binaries are not on this machine (§2.5).
`revised_manifest.csv` records the sha256 of every input binary of every set, so
the third is now checkable:

| identity | overlap between train and test |
|---|---|
| feature filename | **0** |
| source-binary sha256 (2,509 kept manifest rows, every row carries one, all distinct inside each set) | **0** |
| opcode-stream sha256 | `good_test` 62/129, `mal_test` **0/362** |

Nothing fitted sees the test rows either, which is a separate question from the
corpus and was re-checked in the code:

| step | fitted on |
|---|---|
| WordPiece vocabulary (`build_sequence_frame`) | `text[df["split"] == "train"]` only |
| Word2Vec (`build_word2vec_embeddings`) | `sentences=train_m[seq_col]`; test rows are only *projected* through the fitted model |
| hyper-parameter selection (`fit_and_score`) | `StratifiedKFold` over `X_train` only; no scaler, nothing else fitted |

That closes the leakage question for `expC` on the ransomware side completely:
its test families are unseen (§2.3), no test binary is a training binary, no
test file's opcode stream appears in training, and no fitted object has seen a
test row. `expC`'s MLP/WP macro-F1 of 0.9260 against `expA`'s 0.8098 is
therefore **not** a leakage artefact — and the goodware-side leak it does carry
is *smaller* than `expA`'s (48.1% against 48.9%), so it cannot account for the
gap either. Where the gain actually sits is per-family: `hive` keeps all 50 test
samples through the cohort filter, and `expC`'s MLP/WP recalls it at **1.0000**
where `expA`'s manages **0.1600**. That one family is 50 of `expC`'s 362
ransomware test files, and it is the single largest contributor to the
difference.

**The revised `Goodware_Balanced` tree, and a new cross-source collision.**

| | files | unique streams | files in duplicate groups |
|---|---|---|---|
| revised `good_all` | 1,337 | 1,283 | 89 (6.7%), 35 groups |
| traditional `good_all` (§2.4) | 1,343 | 1,298 | 73 (5.4%), 28 groups |

Byte-identical, as an opcode stream, to a Mendeley goodware file:
**13 of 1,337** revised against **0 of 1,343** traditional. They are Inno Setup
and NSIS installer stubs — **six** distinct streams covering all 13, each shared
with between one and five Mendeley goodware files, and one of the six is the
`root_CairoSetup_64bit.exe` stream that §2.2 already found under both labels.
**[corrected]** An earlier draft of `results/summary.md` said five distinct
stubs; re-measured it is six, and the summary now says so. Dropping the operands
makes the §2.1 installer-stub problem strictly *more* visible, which is the
honest reading: the collisions were always there, the full-instruction form
merely hid them behind a differing register allocation.

`data.dedup_goodware_sources()` removes all 13 at load time (§2.5), and that,
plus the cohort, is the whole of `expD`'s goodware arithmetic:

```
131  expB test goodware
 -4  no revised feature file: hard_negative_ffmpeg / ffplay / ffprobe.exe,
     system_sppsvc.exe - exactly the four the cohort filter drops in expB_cohort
 -4  cross-source dedup: everyday_SteamSetup.exe, everyday_helper.exe,
     everyday_setup.exe, everyday_uninstall.exe
123  expD test goodware
```

The intermediate 127 is `expB_cohort`'s test goodware count, which is why the
two lines of the subtraction can be read off two different experiments.

**What the tokenizers do on mnemonic-only input.** Measured on the `expC`
training corpus under the pipeline's own 5,000-instruction cap, and on the
cached tokenizer in `results/tokenizers/expC_WPC.json` that produced the
committed `expC` numbers:

* The distinct-normalized-line vocabulary collapses from **36,802** (Exp A's
  training rows, same cap, same `normalize_instruction`) to **466**. SW's token
  is now a bare mnemonic; WP's adjacent-line bigram (`mov_push`) is the only
  view left that carries order. **[corrected]** An earlier draft of
  `results/summary.md` put the traditional figure at 22,451; re-measured over
  the committed `expA` training split it is 36,802 capped (88,277 uncapped), and
  the summary now says so.
* WordPiece trained at `vocab_size: 1000` stops **below the cap** — the cached
  `expC_WPC.json` holds 941 entries, 667 word-initial and 274 `##` — so the
  tie-break that makes the trainer nondeterministic (§1.10) is never reached.
* The pieces it does learn are almost never used. The Exp C corpus holds **478**
  distinct mnemonics and **466 encode as a single whole-word token**. The 12
  that fragment (`cvtpd2ps` → `cvt ##pd ##2ps`, `cvtsd2ss` → 4 pieces,
  `xacquire` → 6) are *exactly* the 12 that occur only in the test split — the
  tokenizer is fit on train rows only, as it must be, so those are the only
  words it has never seen. Between them they account for **176 of 11,952,897**
  mnemonic occurrences — 0.0015%. There are no `<UNK>` tokens at all: the `##`
  machinery earns its place only by keeping those 12 out of `<UNK>`.
  **On mnemonic-only input WPC is whole-word
  tokenization to within a rounding error**, so an RF/WPC row and an SW row in
  the revised columns are reading all but the same stream. A WPC number there
  is evidence about whole-word mnemonic tokenization wearing a WordPiece label.
* **§1.10 does not bite here, and that was tested rather than argued.** `expC`
  was re-run twice with `tokenization.tokenizer_cache` pointed at an empty
  directory that was deleted between the runs, so each trained its own
  vocabulary from scratch in its own process. The vocabularies are *not*
  identical — 12 of 941 entries differ, and 102 shared entries get different
  ids — but every difference is an unused fragment of a mnemonic that is a
  whole word anyway, and both encode all 478 distinct mnemonics to the same
  token strings. Both runs produced a `predictions.csv` **byte-identical** to
  the committed one, and a `metrics.json` differing only in `elapsed_seconds`
  and in the cache path embedded in the config. The cache is still required for
  the four traditional-feature experiments, where the cutoff *is* reached and
  the tie-break is worth 0.06 macro-F1 (§1.10).

**Floors.** `data.baselines()` scores two rules that read no opcode at all on
each test set, and `results/summary.md` prints them under every table:

| experiment | majority-class macro-F1 | `x86 → ransomware` macro-F1 | model rows that clear the second |
|---|---|---|---|
| expA | 0.4268 | 0.4266 | 3 of 3 |
| expA_cohort | 0.4244 | 0.4335 | 3 of 3 |
| expC | 0.4244 | 0.4335 | 3 of 3 |
| expB | 0.4268 | **0.6799** | **0 of 3** |
| expB_cohort | 0.4254 | **0.7048** | **0 of 3** |
| expD | 0.4274 | **0.7113** | **1 of 3** (SVM-RBF/SW, 0.8439) |

This is the sharpest single result of the third pass, and it cuts two ways.

1. In the three Goodware_Balanced experiments a rule that reads only the
   architecture column scores 0.68–0.71 macro-F1, and **eight of those nine
   model rows do not beat it**. Nothing in `expB`, `expB_cohort` or `expD` has
   been shown to use the opcode stream except `expD`'s SVM-RBF/SW row.
2. In the three Mendeley-goodware experiments the same rule scores only 0.43 —
   level with the majority-class floor (0.4266 against 0.4268 in `expA`, 0.4335
   against 0.4244 in the other two) and with a balanced accuracy of 0.433–0.447,
   i.e. *worse than chance*. That is **not** evidence that those experiments are
   free of the confound. Mendeley `good_test` is ~91% x86 while `good_train` is ~57%,
   so the bitness shortcut is present in training and **misfires on the test
   set**. What that looks like is in the per-architecture table of
   `results/summary.md`: `expA`'s x64 slice scores 0.2865–0.3066 macro-F1
   against 0.8724–0.9634 on x86, with x64 ransomware recall of 0.19–0.22.

The `x86 → ransomware` rule counts an `unknown` architecture as not-x86. Six
test files in `expA` and four in `expB` are `unknown`; the cohort-filtered and
revised experiments have none.

---

## 3. Format compatibility

`asm_parse.py` emits `0x00401000:  mov\teax, ebx`; `extract.py` emits
`mov eax, ebx`. `normalize_instruction` rewrites `0x...` to `<HEX>`, so an
unconverted `.asm` line tokenizes to `<HEX>: mov eax ebx`.

If ransomware features come from `LLM_Features` (no prefix) and goodware from a
raw `.asm` tree (prefix on every line), that prefix is present in exactly one
class. It is a content-free feature that separates the classes perfectly.
`asm_tool/asm_to_opcodes.py` strips it, and `tests/test_asm_parse.py` asserts
the output never starts with `0x`.

`asm_tool/consistency_check.py` compares the two disassemblers on the same
binaries. On 10 files (5 x86, 5 x64), the mnemonic sequence and the normalized
instruction sequence **match exactly** in all 10. The only difference is the
printed branch target — `extract.py` disassembles from `section.VirtualAddress`
and `asm_parse.py` from `ImageBase + section.VirtualAddress` — which normalizes
to `<HEX>` on both sides.

### 3.1 Schema equality, checked over the whole corpus

`consistency_check.py` proves the two disassemblers agree on 10 files.
`llm_features_pipeline/check_schema.py` proves the two *feature directories*
are the same shape, over every file in both. It exits non-zero on any hard
violation.

```bash
python llm_features_pipeline/check_schema.py
#   --candidate LLM_Features_Balanced/good_all
#   --reference LLM_Features/Features_Extraction/good_train
```

Result (13 September 2026):

| check | reference (`good_train`) | candidate (`good_all`) |
|---|---|---|
| files | 1,116 | 1,343 |
| instruction lines | 24,681,771 | 37,798,569 |
| filename `<prefix>_<name>.txt` | 0 violations | 0 |
| UTF-8 decodable | 0 | 0 |
| `0x…:` address prefix on a line | 0 | 0 |
| tab character | 0 | 0 |
| `;` comment line | 0 | 0 |
| `.skip` / other directive line | 0 | 0 |
| line not starting with a bare mnemonic (= not one instruction per line) | 0 | 0 |
| empty file | 0 | 0 |

**PASS, exit 0.** The two directories can be mixed in one experiment without a
format artefact separating the classes — which is the failure mode described at
the top of this section, and the one that would look like a perfect model.

The mnemonic-vocabulary comparison is reported but never fatal, because two
corpora legitimately use different instructions. It is worth reading anyway:

* reference 435 distinct mnemonics, candidate 941, shared 415, Jaccard **0.432**
* **526 candidate-only** mnemonics, led by `vmovaps` (22,948), `vpxor` (10,872),
  `vaesdec` (10,848), `vshufps` (10,453), `vpaddd` (9,419), `vbroadcasti32x4`
  (3,469) — AVX / AVX-512 / AES-NI, i.e. 2026 vendor builds compiled for modern
  ISA extensions.
* **20 reference-only** mnemonics, led by `swapgs` (46), `wrmsr` (38), `iretq`
  (34), `invlpg` (5) — privileged instructions from Windows system binaries.

That asymmetry is a **real confound, not a schema defect**: the vector
instruction set is far better represented in Exp B's goodware than anywhere in
the Mendeley ransomware, so "uses AVX-512" is available to the model as a
goodware cue in Exp B and essentially absent in Exp A. It compounds the
architecture skew in §5 and belongs with it in any reading of the A-vs-B gap.

---

## 4. What was changed, and where

| change | file |
|---|---|
| sorted, filename-carrying, group-aware loader | `llm_features_pipeline/data.py` |
| per-stratum group split with exact apportionment | `llm_features_pipeline/data.py` |
| exact-duplicate leak measurement | `llm_features_pipeline/data.py` |
| cross-source sha256 dedup, enforced at load time (§2.5) | `llm_features_pipeline/data.py` |
| cached, fingerprinted tokenizer for reproducibility (§1.10) | `llm_features_pipeline/run_pipeline.py`, `config.yaml` |
| full binary metric set, `f1_macro` model selection | `llm_features_pipeline/run_pipeline.py` |
| one tokenizer method in memory at a time | `llm_features_pipeline/run_pipeline.py` |
| `workers=1` for Word2Vec — reproducibility (§1.7) | `llm_features_pipeline/run_pipeline.py`, `config.yaml` |
| seeded, shuffled GridSearchCV folds (§1.8) | `llm_features_pipeline/run_pipeline.py` |
| macro precision/recall, embedded config, `--results-dir` | `llm_features_pipeline/run_pipeline.py` |
| per-architecture goodware recall in the summary | `llm_features_pipeline/run_pipeline.py` |
| corpus-wide schema equality check (§3.1) | `llm_features_pipeline/check_schema.py` |
| hand-written normalizer + loader unit tests | `tests/test_tokenization_pipeline.py` |
| `.asm` -> `LLM_Features` conversion | `asm_tool/asm_to_opcodes.py` |
| disassembler equivalence proof | `asm_tool/consistency_check.py` |
| cohort join, annotate-then-filter, split reuse (§2.7) | `llm_features_pipeline/data.py`, `config.yaml` |
| per-architecture metrics for BOTH classes, per-family ransomware recall (§2.7) | `llm_features_pipeline/run_pipeline.py` |
| majority-class and architecture-only floors beside every score (§2.7) | `llm_features_pipeline/data.py` (`baselines`), `run_pipeline.py` |
| `predictions.csv` per run, so an old run can answer a new question | `llm_features_pipeline/run_pipeline.py` |
| cohort, mnemonic-only and floor unit tests | `tests/test_tokenization_cohort.py` |

`Tokenization-Testing-for-Malware-Data` itself is **unmodified**. Its
`normalize_instruction`, `train_tokenizer`, `build_word2vec_embeddings` and
`build_bert_embeddings` are imported by the pipeline, so there is one
implementation of each and a fix there reaches this project without a second
code path. Only the data loader — the part with the defects in §1.1, §1.2 and
§1.4 — is replaced. The two defects that live *inside* the imported code
(§1.7's `workers=4` default, §1.10's unseedable trainer) are worked around at
the call site rather than patched upstream, so the import stays clean: the
pipeline passes `workers=1` and caches the tokenizer it was handed.

---

### 4.1 There is a third copy of `tokenization.py`, and it has drifted

`LLM_Classification` carries its own `Tokenization/tokenization.py` (156 lines)
alongside the tokenization repo's (260 lines).

`normalize_instruction` is **identical** in both — same five regexes, same order
— so nothing already produced is invalidated. What differs is everything around
it: the repo copy has no `Instructions`/`Opcodes` column aliasing, no
path-relative dataset resolution, and an older `load_*` family.

This is the same hazard as the two `asm_parser.py` copies: two implementations
of one step, drifting apart, with results attributed to the model rather than to
which file happened to be imported. `llm_features_pipeline` imports from the
**tokenization repo** path named in `config.yaml`, never from the in-repo copy.
Either delete `LLM_Classification:Tokenization/` or make it the single source
and repoint the config — but do not keep both.

### 4.2 The A-vs-B gap is not yet attributable to the goodware source

Recorded here because it is the conclusion the whole exercise was supposed to
support, and it does not hold up on the evidence available.

With `arch` from `opcode_manifest.csv`, Exp B's goodware test recall splits like
this (from `results/summary.md`, all three model/tokenizer pairs):

| model / tok | recall, x64 goodware (n=85) | recall, x86 goodware (n=46) |
|---|---|---|
| RF / WPC | 0.9882 | 0.5870 |
| MLP / WP | 1.0000 | 0.4348 |
| SVM-RBF / SW | 0.9882 | 0.6957 |

Nearly every x64 benign file is recognised; between 30% and 57% of the x86
benign files are called ransomware. Given that the ransomware class is 96% x86
in train and 76% in test, that is what a model keying on bitness looks like.

Exp B's goodware is 20% x86 where Exp A's is ~57%, so the two experiments differ
in goodware *architecture mix* as well as goodware *source*, and the models are
demonstrably using the former. Until the ransomware side can be split by
architecture too — which needs a per-sample index out of the VM — the A-vs-B
delta cannot be read as "the Mendeley goodware is easier."

**[updated 14 September 2026]** The per-sample index arrived, in the cohort
CSVs, and it did not need the VM: `cohort_mendeley.csv` and
`cohort_balanced.csv` carry an architecture for every input binary of every set,
`good_test` and both ransomware splits included. The question this section had
to leave open is now answered in §2.7, and the answer is worse than the one
feared here:

* The conclusion above stands and is now quantified. An `x86 → ransomware` rule
  that reads no opcode scores macro-F1 0.6799 on Exp B's test set. **No Exp B
  model row beats it** (0.5862, 0.6629, 0.6756), and none does in `expB_cohort`
  either. One of `expD`'s three does.
* It is not symmetric. The same rule scores 0.4266 on Exp A — below the
  majority-class floor — because Mendeley `good_test` is ~91% x86 and so looks
  like the ransomware to a bitness rule, while `good_train` is ~57%. Exp A's
  models still learn the shortcut and it still costs them: their x64 test slice
  scores 0.2865–0.3066 macro-F1 against 0.8724–0.9634 on x86.

So the A-vs-B gap remains unattributable to the goodware source, for the reason
given here, and the per-architecture slices in `results/summary.md` are now the
form in which any of these scores should be read.

---

## 5. Open items

* **§2.1 is reported, not repaired.** Deduplicating `good_test` would change the
  corpus the published baseline was measured on. The leak rate is printed in
  every run and written into `results/*/sample_counts.json`, so it can be read
  alongside the score. A deduplicated variant is a one-line addition to
  `config.yaml` when someone wants it.
* **Installer-wrapped samples measure the installer.** Affects both classes.
  Fixing it means unpacking installers before disassembly, which
  `Goodware_Balanced` does for goodware but nobody has done for the Mendeley
  ransomware.
* **Architecture skew is large, and as of 14 September 2026 it is measured on
  every side.** First measured 13 September 2026 with the gaps noted below the
  table; those gaps are now closed:

  | side | x86 | x64 | source |
  |---|---|---|---|
  | Mendeley ransomware, train (1,023) | 978 (95.6%) | 42 (4.1%) | WRITEUP.md §3.2 |
  | Mendeley ransomware, test (385) | 292 (75.8%) | 92 (23.9%) | WRITEUP.md §3.3 |
  | Mendeley goodware, train (1,115 on disk) | 630 (56.5%) | 485 (43.5%) | WRITEUP.md §3.1 |
  | Mendeley goodware, test (131) | — | — | binaries not present locally |
  | `Goodware_Balanced` index (1,500) | 375 (25.0%) | 1,125 (75.0%) | `corpus_index.csv` |
  | `LLM_Features_Balanced` converted (1,343) | 308 (22.9%) | 1,035 (77.1%) | `opcode_manifest.csv` |
  | Exp B goodware actually used (1,247) | 247 (19.8%) | 1,000 (80.2%) | `sample_counts.json` |

  The ransomware side is 96%/76% x86 and Exp B's goodware is 20% x86 — a gap of
  well over 25 points, so `machine` **is** a shortcut feature on this corpus and
  both experiments need a per-architecture breakdown. Bitness is not hidden: it
  leaks through the operands (`rbp`, `r8`–`r15`, rip-relative addressing) and
  survives normalization.

  **[closed 14 September 2026]** Both gaps named below are filled. The cohort
  CSVs (§2.7) carry an architecture for every input binary of every set, so
  `results/summary.md` now prints the x86/x64 split for `good_test` and for both
  ransomware splits in all six experiments, test metrics for **both classes
  inside each architecture slice**, and an `x86 → ransomware` floor. The
  `check_arch.py`-in-the-VM route is no longer needed for this. What was
  missing, and is not any more:
  * ~~**Exp A's goodware test set**~~ — 12 x64 / 117 x86 / 2 unknown; 91% x86,
    against 57% in `good_train`.
  * ~~**The ransomware side of either experiment**~~ — train 42 x64 / 927 x86 /
    6 unknown, test 88 x64 / 290 x86 / 4 unknown (the cohort-filtered variants
    drop the unknowns and 16 x64 test samples).

  What the filled-in numbers show is in §2.7: eight of the nine
  Goodware_Balanced model rows score **below** a rule that reads only the
  architecture column. The skew is not merely present; on that half of the
  experiment grid it is most of the measured signal. What remains open is the
  *remedy* — matching the architecture mixes, or scoring on a bitness-balanced
  test set — not the measurement.

* **The 5,000-instruction cap is a modelling limitation, and now a measured
  one.** §6.2: lifting it to 20,000 tokens is worth +0.031 CV macro-F1 on
  `expC`, more than any other single axis in the search — the median `expC`
  ransomware test file is 144,236 mnemonics, so the committed runs read the
  first 3.5% of it. What is *not* settled is whether that CV gain is real
  signal: the two 50,000-token configurations in `expC`'s top five lose 0.09 to
  0.11 macro-F1 on test relative to the 20,000-token ones (§6.3). Reading more
  of the file helps in cross-validation and hurts on the held-out split, and
  which of those two the deployment resembles is not answerable from this
  corpus.
* **WPC results are not reproducible on a new corpus.** §1.10. The cached
  tokenizer makes the committed numbers reproducible, but the first run on any
  new corpus draws one vocabulary out of several the trainer might have picked,
  and the spread is worth 0.06 macro-F1. Before publishing a WPC number on new
  data, train the tokenizer n times and report mean and spread rather than one
  draw. SW and WP are unaffected — they use no trained tokenizer.
* **The vector-instruction vocabulary is a second source-correlated cue.**
  §3.1: Exp B's goodware uses 526 mnemonics the Mendeley goodware never does,
  dominated by AVX/AVX-512/AES-NI. Same remedy as the architecture skew — it has
  to be controlled for before the A-vs-B gap is read as a statement about
  goodware quality. The revised feature form does **not** remove it and may
  widen it: `extract_unified.py` sweeps past undecodable bytes instead of
  stopping at the first one, so it reaches vector code the linear sweep never
  got to (§2.7).
* **`expB`, `expB_cohort` and `expD` have not been shown to use opcodes at
  all.** §2.7: eight of those nine model rows score below the architecture-only
  floor. This is the one open item that should be resolved before any of those
  three experiments is quoted. Either the goodware architecture mix gets matched
  to the ransomware's, or those rows are reported with the floor printed beside
  them — which `results/summary.md` now does, in every table.
  **[reinforced 14 September 2026]** §6 searched 787 configurations on `expD`
  under grouped cross-validation and the selected one scores **0.6830** on test,
  *below* the 0.7113 architecture floor, with x64 ransomware recall of 0.1667
  against 0.8034 on x86. Tuning does not get around this; it walks further into
  it, because reading the bitness is what the train split rewards.
* **Half the goodware test set is still a copy of training data in the revised
  corpus too.** §2.7: 62 of 129, against 64 of 131 traditional. The revised
  extractor was never meant to fix this and did not. `expA` ↔ `expC` is a fair
  comparison *to each other* because both carry it at the same rate; neither is
  a field estimate.

---

## 6. The tuned runs (`tune.py`, `results/exp*_tuned/`)

Everything in §6 was measured on 14 September 2026 with
`llm_features_pipeline/tune.py` against the two revised-feature experiments.
The tables it refers to are generated, not typed: the **Tuned** section of
`results/summary.md` is rebuilt from `results/exp*_tuned/metrics.json` and
`cv_search.csv` by `run_pipeline.py --summary-only`, so the prose here and the
numbers there cannot drift apart.

```bash
python llm_features_pipeline/tune.py --experiment expC --stage cache
python llm_features_pipeline/tune.py --experiment expC --stage search
python llm_features_pipeline/tune.py --experiment expC --stage final
python llm_features_pipeline/tune.py --experiment expC --stage check-pooling
python llm_features_pipeline/run_pipeline.py --summary-only
python -m pytest tests/test_tokenization_tuning.py -q
```

### 6.1 What the search was, and what it was allowed to see

**787 configurations per experiment**, cross-validated in 8,457 s (`expC`) and
8,331 s (`expD`). Three tracks, and the split of effort between them is a
deliberate one — a Word2Vec fit per fold at `workers=1` (§1.7) is the single
most expensive thing in the search, so the compute went where the CV score was
already higher:

| track | configurations | what varies |
|---|---|---|
| `tfidf` | 252 | sequence budget (5,000 / 20,000 / 50,000), sampler (`head` / `strided`), n-gram order (1-1 / 1-2 / 1-3), 14 linear classifier settings |
| `w2v` | 507 | 13 Word2Vec settings (dim, window, epochs, min_count) × 3 poolings × 13 dense classifier settings |
| `tokenizer` | 28 | SW / WP / WPC / BPE, WPC vocabulary 500–4,000, BPE 1,000 and 4,000, at 5,000 and 20,000 tokens |

**The protocol, and the evidence it held.** `run_search()` reads the `split`
column exactly once, to take `split == "train"`; the strings `test` and
`te_rows` do not occur anywhere else in it, and it computes no test metric of
any kind. Inside that train split the folds are
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` over the
`group` column of `splits.csv` — ransomware family for a positive, source
project (or the file itself) for a negative. What that buys is recorded per
fold in `search_records.json`:

| | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| `expC` rows held out | 396 | 401 | 407 | 407 | 407 |
| `expC` families held out | 4 | 5 | 5 | 5 | 5 |
| `expD` rows held out | 401 | 396 | 407 | 406 | 406 |
| `expD` groups held out | 59 | 57 | 60 | 59 | 59 |

Whole families go out together — fold 0 of `expC` holds out all of `dharma`,
`netwalker`, `ragnarok` and `stop` — which is the shape of the real Mendeley
test split (§2.3). A random-fold search would have been selecting for a task
the test set does not ask.

Everything fitted is fitted inside the fold:

| object | fitted on |
|---|---|
| TF-IDF vocabulary, document frequencies and IDF | `fold_tfidf(counts, train_rows)` — the fold's training rows only |
| Word2Vec | `_w2v_matrix([seqs[i] for i in tr], ...)` — the fold's training documents only |
| the IDF used by `tfidf_mean` pooling | `_unigram_idf(counts, tr)` — the fold's training rows only |
| WordPiece / BPE vocabulary | the train SPLIT (`text.iloc[train_rows]`), not the fold; recorded as a control rather than a contender, because §2.7 already measured WPC as whole-word tokenization to within 0.0015% of token occurrences |
| decision threshold | out-of-fold scores of the OTHER folds (`nested_threshold_scores`) |

**The one operation that sees a test row, and why it is not a leak.** In
`--stage final`, `_features_for` counts n-grams over train and test documents
together and prunes at global document frequency 5 before the vectoriser is
fitted, because the unpruned trigram space at a 50,000-token budget does not fit
in memory. A column's *training* document frequency can never exceed its global
one, so every column a train-only fit would have kept survives the global prune,
and the kept set, the IDF and the `max_features` cut come out identical either
way. That is asserted rather than argued:
`tests/test_tokenization_tuning.py::test_global_min_df_prune_cannot_change_what_the_train_fit_keeps`
puts the same documents through both routes for three n-gram orders and two
`min_df` values and requires the same columns and the same numbers.

**The identity the whole search rests on** is that mean pooling over 50,000
token vectors is one sparse product against a ~1,300-column count vector.
Checked against the repo's own `build_word2vec_embeddings` on a real corpus
slice rather than asserted:

```
max |matmul mean - float64 token-by-token mean| = 0.000e+00
max |matmul mean - repo float32 mean|: train 4.504e-05 test 4.127e-05
  (values up to 2.317; float32 eps x that is 2.8e-07 per term)
POOLING EQUIVALENCE: PASS
```

The first line is exact — it is the same arithmetic. The second is float32
rounding in the repo's accumulator, which is what it can be and no better.

### 6.2 What the search found

Per-axis best and median CV macro-F1 are tabulated in `results/summary.md`
("what moved the number, axis by axis"). The short version, best CV macro-F1 at
each setting:

| axis | `expC` | `expD` |
|---|---|---|
| sequence budget 5,000 → 20,000 → 50,000 | 0.9168 → 0.9436 → 0.9474 | 0.9232 → 0.9174 → 0.9237 |
| n-gram 1-1 → 1-3 | 0.9271 → 0.9474 | 0.9237 → 0.9206 |
| TF-IDF over tokens vs Word2Vec | 0.9474 vs 0.9271 | 0.9237 vs 0.9232 |
| sampler `head` vs `strided` | 0.9474 vs 0.9293 | 0.9206 vs 0.9237 |
| tokenizer SW / WP / WPC / BPE | 0.9474 / 0.9267 / 0.9057 / 0.9052 | 0.9237 / 0.9081 / 0.8949 / 0.8934 |
| pooling mean / tfidf_mean / mean_max | 0.9102 / 0.9119 / 0.9271 | 0.9152 / 0.9008 / 0.9232 |
| weighting `none` vs `class_arch` | 0.9474 vs 0.9297 | 0.9237 vs 0.8600 |
| best of each classifier | LogReg 0.9474, LinearSVC 0.9420, MLP 0.9271, SVM-RBF 0.9097, RF 0.8671 | LinearSVC 0.9237, SVM-RBF 0.9232, MLP 0.9222, LogReg 0.9202, RF 0.8566 |

Read as a marginal, not an ablation — the axes are unevenly sampled on purpose,
so the BEST column is the load-bearing one, and the medians in `summary.md` say
whether a winner is a lone spike or the whole band moving.

**What helped, in cross-validation.**

* **The sequence budget, on `expC`, by more than anything else.** +0.031 CV
  macro-F1 from 5,000 to 20,000 tokens. The committed pipeline's
  `max_instructions = 5000` is a severe truncation on this corpus: the median
  `expC` ransomware TEST file holds 144,236 mnemonics, so 5,000 lines is the
  first 3.5% of it, and 66% of those files are over 50,000 lines. What lives in
  that first 3.5% is CRT startup and compiler prologue — the part most alike
  across everything built with the same toolchain.
* **n-gram order.** 1-3 over 1-1 is +0.020 on `expC`. With one mnemonic per
  line the unigram bag is a 728-symbol histogram; bigrams and trigrams are the
  only thing left in the representation that carries order.
* **TF-IDF over the token stream, over Word2Vec + pooling.** +0.020 CV on
  `expC`, +0.0005 on `expD`. It is also two orders of magnitude cheaper.
* **`mean_max` pooling, if Word2Vec is used at all.** +0.017 (`expC`) and
  +0.008 (`expD`) over the plain mean the committed pipeline uses. The max half
  is a presence feature over token TYPES, which is what the mean averages away.
* **Tuning the committed representation's own hyper-parameters.** The committed
  setting (head / 5,000 / dim 100 / window 30 / 5 epochs / mean pooling) is in
  the table by construction: its best row is CV **0.8760** on `expC` (rank 222
  of 787) and **0.8501** on `expD` (rank 406), against a best Word2Vec row
  anywhere in the search of 0.9271 / 0.9232. So roughly half the search's CV
  gain is Word2Vec settings and pooling and half is the budget and the n-gram
  order.

**What did not help.**

* **Subword tokenization, at any vocabulary size.** WPC and BPE are *below* SW
  and WP on both experiments, and the vocabulary size does not matter — 500,
  1,000, 2,000 and 4,000 land within 0.0005 of each other on `expC`. This is
  §2.7 measured a second way and from the other end: on mnemonic-only input
  there is nothing for a subword model to do but reproduce the whole word, and
  what it does learn costs a little by splitting the 12 mnemonics it never saw
  in training.
* **The architecture reweighting.** `class_arch` — making the 42 x64 training
  positives weigh as much in aggregate as the 862 x86 ones — is worth −0.018 CV
  on `expC` and **−0.064** on `expD`. It was put in the search precisely because
  §2.7 says bitness is most of the measured signal on the Goodware_Balanced
  side; cross-validation rejected it.
* **Fitting the decision threshold.** Both fitted rules were scored nested for
  all 787 configurations. On the best rows they are worth +0.001 to +0.010 in CV
  and give it back on test. The reported rows therefore use the untuned cut,
  which is also what the committed runs use, so the before/after comparison
  moves the representation and the classifier and nothing else.
* **Random forests.** The worst classifier family on both experiments, at the
  top of the table and at the median.

### 6.3 The final evaluation, and the result

One configuration per experiment was selected **by cross-validation alone** —
highest out-of-fold macro-F1, `cv_rank == 1` — then fitted on the whole train
split and scored on the test set once. Both selected configurations are
deterministic end to end (a linear model on a TF-IDF matrix; no Word2Vec, no
MLP, no bagging), so one run is the whole story and a seed sweep would produce
five identical rows. The four runners-up were scored afterwards **for the
CV-to-test gap table only**; they are marked `"selected": false` in
`metrics.json` and are not the result.

| | `expC` | `expD` |
|---|---|---|
| chosen configuration | SW TF-IDF 1-3, head, 50,000 tokens, 87,269 features, LogReg C=0.1 balanced | SW TF-IDF 1-1, strided, 50,000 tokens, 859 features, LinearSVC C=0.1 balanced |
| CV macro-F1 | 0.9474 | 0.9237 |
| **test macro-F1** | **0.8576** | **0.6830** |
| CV − test | +0.0898 | +0.2407 |
| best committed row | 0.9260 (MLP/WP) | 0.8439 (SVM-RBF/SW) |
| majority-class floor | 0.4244 | 0.4274 |
| `x86 → ransomware` floor | 0.4335 | **0.7113** |
| mnemonic TF-IDF calibration | 0.9680 (0.955 dedup-clean) | 0.8023 |

**The search did not beat the fixed configuration on either experiment** — by
−0.068 on `expC` and −0.161 on `expD` — and the `expD` result does not clear the
architecture-only floor. That is the measurement; three things follow from it,
and all three are checkable in the files.

1. **The CV ranking does not transfer, and the top five show it.** On `expC` the
   five differ by 0.0079 in cross-validation and by **0.1104** on test, and the
   gap runs from −0.029 to +0.090. The two 50,000-token rows (CV ranks 1 and 4)
   both land on 0.8576 and the three 20,000-token rows on 0.9540–0.9680: the
   axis that won the search is the axis that lost the test set. Selecting the
   best of those five *after* seeing the test scores would report 0.9680 —
   +0.11 over the honest number, and exactly the error the protocol exists to
   prevent.
2. **Where `expC` lost it is goodware, not ransomware.** The tuned row raises
   ransomware recall to 0.9917 (committed: 0.9641) and drops x86 goodware recall
   to 0.6154 (committed: 0.8803). Reading more of each file makes the model more
   willing to call a file ransomware, and 62 of the 129 test goodware files are
   verbatim copies of training files (§2.7), so in cross-validation that
   willingness is not punished the way the test split punishes it.
3. **`expD`'s tuned row is an architecture detector.** Its x64 ransomware recall
   is **0.1667** against 0.8034 on x86, and its x64 goodware recall is 1.0000;
   the committed SVM-RBF/SW row scores 0.9011 macro-F1 on the x64 slice where
   the tuned row scores 0.5077. Per family, `blackcat` goes 1.0000 → 0.0000,
   `hive` 0.8400 → 0.0200 and `blackbyte` 1.0000 → 0.0000 — and `hive`,
   `bianlian` and `blackbyte` are 43-of-50, 11-of-11 and 7-of-7 x64. This is the
   §5 open item reproducing itself under a better search: with train goodware
   82% x64 and train ransomware 95% x86, about the best thing a model can do in
   grouped cross-validation on that train split is read the bitness, and the
   test split does not reward it.

The honest summary is that on these two experiments the ceiling is set by the
corpus, not by the configuration. `expC`'s audited mnemonic TF-IDF calibration
row (0.9680 raw, 0.955 with the duplicated test goodware removed) is still the
number to beat, and nothing in 787 configurations beat it; `expD` cannot be read
as a statement about opcodes at all until its architecture mixes are matched.

### 6.4 Reproducibility

`--stage final --results-dir <scratch>` refits the recorded configurations
somewhere else, reading the committed `search_records.json` so it checks the
final fit rather than demanding the 2.3-hour search be repeated. Both
experiments were re-run that way at `--top-k 5`, so the check covers the
stochastic runners-up as well as the deterministic selected rows:
`predictions.csv`, `splits.csv`, `sample_counts.json` and `config_used.yaml`
came back **byte-identical** and `metrics.json` differed in `elapsed_seconds`
alone — the same standard §2.7 held the six committed experiments to. The
Word2Vec rows reproduce because `workers=1` (§1.7) and the seed is fixed; the
`expD` MLP row (CV rank 3) reproduces for the same reason.

Two operational notes, because both cost time here:

* The token cache is a scratch artifact in the system temp directory and is
  *not* preserved between sessions. `--stage cache` rebuilds it from the corpus
  in 252 s (`expC`, 2,509 files) and 350 s (`expD`, 2,501 files); `--stage
  search` and `--stage final` both refuse to run without it.
* `--stage final` on `expD` at `--top-k 5` was killed once, with no traceback,
  while two other pipelines held ~6 GB on the same machine. That is a memory
  ceiling rather than a defect — the trigram count matrix at a 50,000-token
  budget over 2,501 documents is the largest object the pipeline ever builds —
  and re-run alone it completed and produced the numbers above.
