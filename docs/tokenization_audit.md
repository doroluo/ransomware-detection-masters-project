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
* **Architecture skew is measured everywhere it can be, and it is large.**
  Re-measured 13 September 2026:

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

  What exists now: `results/summary.md` carries the table above and a
  per-architecture goodware-recall breakdown for Exp B, computed from the
  manifest. What is still missing, and why:
  * **Exp A's goodware test set** — the 131 `good_test` binaries are not on this
    machine (0 of 131 match by filename) and the feature files carry no
    architecture, so there is nothing to join on.
  * **The ransomware side of either experiment** — VM-only binaries; only the
    aggregate counts above exist. Run
    `python check_arch.py --dir <ransomware dir> --compare ../Goodware_Balanced/corpus_index.csv`
    inside the VM to produce a per-sample index, then split ransomware recall by
    bitness.

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
  goodware quality.
