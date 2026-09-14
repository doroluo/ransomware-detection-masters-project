# Research directions

What could be tried next on this corpus, what each thing needs, what it would
cost, and what result would make it worth continuing. Written against measured
numbers, not against the literature.

Every direction below is scored on the same four axes:

| axis | what it means |
|---|---|
| **inputs** | *available now* (the `asm/` + `mn/` trees and the cohort CSVs on this host), *needs binaries* (the original PE files -- present on disk, never executed), *needs VM* (dynamic execution in an isolated guest), or *needs re-extraction* (a new pass of `extract_unified.py` to emit something the current trees do not carry) |
| **compute** | order of magnitude on this machine (CPU-only, Python 3.14) or on a rented GPU |
| **risk** | the single thing most likely to make the result meaningless |
| **worth pursuing if** | the concrete measurement that would justify more time |

---

## 0. Where we actually are

Everything below has to be read against these numbers. All are on the **same**
cohort and the **same** family-disjoint split: 24 ransomware families in train,
14 entirely different families in test, 2,509 files (Mendeley) / 2,506
(Goodware_Balanced). Macro-F1 on the held-out test set:

| approach | mendeley | balanced | file |
|---|---|---|---|
| **mnemonic 1-3-gram TF-IDF + LogReg / LinearSVC** | **0.968** (0.955 audited) | 0.802 | `results/rules/summary.md` |
| tokenizer + word2vec + RF/SVM/MLP (expC / expD) | 0.926 | 0.844 | `results/summary.md` |
| graph2vec / WL subtree on linear-sweep CFGs | 0.895 | 0.554 | `results/graph2vec/summary.md` |
| mined n-gram + hand-written behaviour rules | 0.797 | 0.670 | `results/rules/summary.md` |
| CNN-ViT over instruction images (5-seed mean) | 0.628 | 0.502 | `results/cnn_vit/summary.md` |

**The simplest method on the list is currently the best one, and it survived
its leakage audit.** A bag of 1-, 2- and 3-grams over mnemonics, IDF-weighted,
fed to a logistic regression -- roughly forty lines of scikit-learn, about
three minutes end to end on a laptop CPU -- beats a word2vec pipeline, a graph
embedding of the control flow those same instructions form, and a vision
transformer, on the same rows. The audit (`results/rules/summary.md` §5,
`results/rules/baseline_audit.json`) found: vocabulary and IDF fitted on train
rows only, no sha256 / family / group overlap, the same cohort rows as every
other pipeline, no length or architecture feature, and a collapse to the
always-predict-ransomware floor (0.386 vs a 0.424 floor) when the training
labels are shuffled. It also found what the number is *partly* worth: **48.1%
of the Mendeley goodware test rows are verbatim copies of a training row**
(0.0% for ransomware), so the honest Mendeley figure is **0.955** on the
non-duplicated test rows and **0.926** if the training set is deduplicated too.
Quote **0.95 / 0.80**, not 0.968.

The strong part of the result is the ransomware side, and duplication cannot
explain it: 13 of the 14 held-out families are detected at recall 1.00, none of
them has any representative in training, and not one ransomware test file is a
byte-copy of a training file. The architecture shortcut is not being taken
either -- a model trained on x64 rows *only* (42 positive examples) still
reaches 0.762. What the model has actually learned is compiler idiom: the
positive weights are wide-integer arithmetic (`adc mov`, `sbb mov mov`,
`add adc mov`, `rol`, `idiv` -- bignum and block-cipher inner loops without
SIMD) and the negative weights are MSVC CRT padding and epilogues
(`int3 int3 int3`, `call leave ret`). Real, explainable, and a property of
toolchains rather than of encryption -- which is why a different goodware
corpus costs it 15 points and makes it miss `blackcat` entirely (0/50).

Three facts about the corpus constrain everything downstream and are repeated
here because each of the directions below can be sunk by one of them:

1. **Architecture is the most available shortcut.** Ransomware is 95% x86 in
   train and 80% x86 in test; Mendeley goodware train is 57% x86 and
   Goodware_Balanced train is ~17% x86. "x64 implies goodware" is a free 80%+ on
   `balanced` training data and then meets a test set that does not honour it.
   Whether a given model takes the shortcut has to be *measured*, not assumed:
   the TF-IDF baseline demonstrably does not (it scores 0.934 inside x64 on
   Mendeley), while every graph2vec representation on `balanced` does --
   goodware recall ~1.00 inside x64, 0.70-0.83 inside x86.
2. **The Mendeley goodware half contains massive near-duplication.** Measured
   on the `mn/` streams this pipeline reads: **62 of 129 Mendeley goodware test
   files (48.1%) have a byte-identical mnemonic stream in train**, and two
   streams appear under both labels. The cause is that `extract.py`
   disassembles installer *stubs* rather than payloads
   (`docs/tokenization_audit.md` §2.1) -- the largest single duplicate group is
   60 PortableApps launchers, 44 in train and 16 in test. Goodware_Balanced
   does not have this problem: 5 of 127 (3.9%), and deduplicating there makes
   the score go *up*. Ransomware test leakage is 0.0% on both datasets.
3. **There is exactly one split.** 14 test families, some with 3 samples. Family
   recall on every track spreads across the full 0-1 range, so differences of a
   few points of macro-F1 between two methods are not measurements.

---

## 1. LLM approaches over `asm` / `mn` text

Three genuinely different things get called "using an LLM" here. They have
different costs, different risks and different defensibility.

### 1a. Embedding models over instruction windows

Take a pretrained text-embedding model, slide it over the mnemonic (or full
instruction) stream in fixed windows, mean- or max-pool the window vectors into
one document vector, classify that with the same LR/RF/SVM grid.

- **inputs**: available now (`mn/`, `asm/`).
- **compute**: the corpus is ~19,500 mnemonics per file at the current 30k cap
  (~49M mnemonic tokens over 2,509 files). At 512-token windows that is roughly
  **50 windows per file, ~125,000 window embeddings** for one dataset. A small
  local sentence-encoder on CPU does that overnight; on a single GPU, under an
  hour. If done through a hosted embedding API instead, it is ~65M input tokens
  per pass.
- **risk**: a model pretrained on natural language has no useful prior over a
  639-symbol mnemonic vocabulary. `mov push mov call` is not English and the
  subword tokenizer will shred it. Expect the embedding to encode little beyond
  token frequency -- which TF-IDF already encodes, better and for free.
- **worth pursuing if**: pooled window embeddings beat `wl_tfidf` (0.835) on
  the Mendeley split *and* hold above 0.80 on `balanced`. If they land near
  TF-IDF, they are a more expensive TF-IDF and should be dropped.

### 1b. Fine-tuning a small encoder on mnemonic sequences

Train a small transformer encoder (6 layers, 256 dims, vocabulary = the 639
mnemonics seen in Mendeley / 1,004 across both trees) from scratch or from a
generic checkpoint: MLM pretraining on all available streams, then a
classification head fine-tuned on the family-disjoint train split.

- **inputs**: available now. Optionally re-extract to lift the 30k cap.
- **compute**: ~49M mnemonic tokens is a small pretraining corpus but not
  absurd for a 10-20M-parameter model; a few GPU-hours. Fine-tuning on 2,018
  labelled documents is minutes. **The labelled set is the binding constraint,
  not the compute**: 904 ransomware documents drawn from 24 families is, in
  effect, 24 examples of the thing being learned.
- **risk**: overfitting to families. With 24 training families and 14 test
  families, a high-capacity model will memorise family-specific instruction
  idiom and the test score will be a lottery over which 14 families were held
  out. This is not hypothetical -- it is exactly the pattern in
  `results/graph2vec/summary.md` §3, where per-family recall runs from 0.00 to
  1.00 within a single row.
- **worth pursuing if**: it beats the TF-IDF baseline under **family-holdout
  cross-validation** (§8), not on the single fixed split. On the fixed split
  alone the result is uninterpretable.
- **do this before the fancy version**: MLM-pretrain on `mn/`, freeze, and use
  the encoder purely as a feature extractor into logistic regression. If the
  frozen features do not beat TF-IDF, fine-tuning will not rescue them.

### 1c. Zero- / few-shot with a frontier LLM on summarised disassembly

Send a summary of each file to a frontier model and ask for a verdict, or for a
behavioural description.

**The token budget is the whole story.** Files in this corpus run to **3.16
million instructions**; the mean is ~19,500 mnemonics under the current 30k cap.
A mnemonic plus separator is roughly 1.3 tokens, so:

| what is sent | tokens per file | 2,509 files |
|---|---|---|
| full uncapped disassembly of the largest file | ~4M | does not fit in any context window |
| the 30k-mnemonic prefix currently used | ~40k | ~100M input tokens |
| a 5,000-mnemonic prefix (the tokenization baseline's window) | ~6.5k | ~16M input tokens |
| a compact summary (histograms, top n-grams, section table, ~40 lines) | ~600 | ~1.5M input tokens |

Against the current first-party price list (input $/MTok: `claude-fable-5-1`
$10, `claude-opus-5` $5, `claude-sonnet-5` $2, `claude-haiku-4-5` $1 -- the
brief's `claude-haiku-4-5-20251001` is the same model; the SDK takes the
undated id), **one pass over the corpus at the 30k-prefix setting costs roughly
$100 on Haiku and $500 on Opus 5**, before output tokens and before the
several passes that prompt iteration always needs. At the summary setting it is
$1.50-$15 a pass, which is affordable. The Batch API halves that. Prompt
caching does **not** help: every request has a different long document, so
there is no shared prefix to cache beyond the system prompt.

Context limits: `claude-fable-5-1`, `claude-opus-5` and `claude-sonnet-5` are
1M-token context; `claude-haiku-4-5` is 200K. So even a 1M-context model cannot
take the largest file whole, and the "summarised disassembly" framing is not a
cost optimisation -- it is a requirement.

Three caveats, all disqualifying for a headline number:

- **It is not a defensible detector.** A hosted model behind a network call,
  with an unknown and changing training set, is not a malware detector you can
  ship, audit, or reproduce in a thesis. Treat this track as *analysis* -- "what
  does this sample look like it does" -- not as a row in the results table.
- **Leakage is unbounded and unmeasurable.** These families are public, named,
  and extensively written about. A model asked "is this ransomware" about a file
  whose family name appears anywhere in the prompt, or whose distinctive strings
  are in the summary, may be recalling a blog post rather than reading the
  disassembly. There is no way to prove otherwise. If this track is run at all,
  the prompt must never contain the family name, the file name, or the sha256,
  and a control condition (the same summary with mnemonics shuffled) must be
  run alongside.
- **Prompt injection from strings.** `asm/` and any string dump are attacker-
  controlled bytes. A ransom note is *text written by an adversary* and can
  contain instructions aimed at whatever reads it. Anything extracted from a
  sample must be passed as data inside a delimiter, never concatenated into the
  instruction part of a prompt, and the model's output must be parsed as a
  constrained label, never executed or trusted as a command.

- **worth pursuing if**: on a 200-file stratified subsample, a summary-only
  prompt reaches macro-F1 above 0.85 *and* the shuffled-mnemonic control
  collapses to chance. Anything less means it is reading the summary's obvious
  giveaways, not the code.

### 1d. LLM-generated behavioural summaries as features

The more defensible use of 1c: don't ask for a verdict, ask for a structured
description ("which crypto primitives appear", "is there a file-enumeration
loop", "what does the import surface suggest"), then feed the *structured
fields* into a classical classifier as extra columns alongside TF-IDF.

- **inputs**: available now, but much better with import tables (§4).
- **compute**: one pass at the summary setting, ~$2-15 per dataset, plus a
  second pass for a second seed to measure how stable the fields are.
- **risk**: the labels are generated by a model that has seen the family names
  in pretraining; and the same field for the same file can change between runs.
  Both are testable: run the extraction twice and report field-level agreement
  before using the fields at all.
- **worth pursuing if**: the generated fields add **more than 2 points of
  macro-F1** over TF-IDF alone on `balanced` (the hard split), and field
  agreement between two runs is above 0.9. On `mendeley` any gain is suspect
  because of the duplication in §0.
- **relation to existing work**: `rules_pipeline/behaviour_rules.py` already
  computes 23 hand-written versions of exactly these fields, deterministically
  and for free. The honest framing of 1d is "can a model write better rules than
  we did", and the comparison is against that file, not against nothing.

---

## 2. Function- and block-level embeddings (asm2vec, PalmTree, jTrans)

The instruction-embedding literature (asm2vec, PalmTree, jTrans, SAFE) works at
**function** granularity: a function is a document, its control-flow paths are
sentences, and the embedding is trained to place semantically equivalent
functions near each other.

**The blocker is function boundaries, and it is a hard one.** The `asm/` trees
come from a linear sweep. There is no function recovery, no call graph with
real callees, and no symbol table. `graph2vec_pipeline/cfg.py` cuts basic
blocks out of the sweep, but a basic block is not a function, and the sweep's
own measurements say how bad it is: **20.6% of branch and call sites have an
unresolvable target** and 51% of files hit the 5,000-block cap
(`results/graph2vec/summary.md` §2). asm2vec on blocks-pretending-to-be-
functions is not asm2vec.

- **inputs**: needs re-extraction with a recursive-descent disassembler that
  does function recovery -- realistically Ghidra headless or IDA. On 2,509 PE
  files Ghidra headless is roughly 30 s-5 min per file: **1-3 days of
  wall-clock**, parallelisable.
- **compute**: after extraction, training asm2vec-style embeddings on the
  recovered functions is a few CPU-hours; PalmTree/jTrans want a GPU and a
  pretrained checkpoint.
- **risk**: two. (i) The published checkpoints for these models were pretrained
  on specific compilers and toolchains; a 2021-2022 ransomware corpus compiled
  with unknown toolchains may be out of distribution. (ii) Function-level
  embeddings answer "are these two functions the same", which is a
  *similarity/search* task -- turning it into a binary classifier requires a
  pooling step that throws away most of what the embedding gives you.
- **worth pursuing if**: Ghidra recovery on a 100-file pilot yields a median of
  >200 functions per file with <20% of the code unattributed. If function
  recovery on this corpus is poor, the whole family of methods is unavailable
  and should be written off explicitly rather than left as an open item.

---

## 3. Graph methods beyond graph2vec

### 3a. GNNs on the CFGs we already have

Message-passing over the basic-block graph, with the existing node labels
(mnemonic class, terminator kind, length bucket, crypto/string/SIMD flags) as
node features, trained end to end.

- **inputs**: available now -- the cached WL documents and the underlying block
  graphs are already built by `graph2vec_pipeline/build_graphs.py`.
- **compute**: 2,509 graphs of up to 5,000 nodes. A 3-layer GIN or GraphSAGE
  trains in minutes on a GPU, tens of minutes on CPU. This is the *cheapest*
  unexplored direction in this document.
- **risk**: the same one that already sank graph2vec here. `size_only` -- ten
  scalars, no structure at all -- reaches ROC-AUC 0.919 on `mendeley` and is
  the **highest macro-F1 row of any representation** on `balanced`. A GNN that
  learns "big graph, many back edges" is learning file size with extra steps.
  Any GNN result must be reported next to `size_only` and next to a degree-only
  ablation.
- **worth pursuing if**: a GNN beats `wl_tfidf/LR` (0.835 mendeley / 0.502
  balanced) by more than 3 points on **both** datasets, with node features
  ablated to check that structure, not size, is carrying it.

### 3b. The teammate's transformer-over-graphs direction (upstream `yanping`)

There is a parallel line of work on the upstream branch applying a
graph-attention transformer to adversarial malware detection. Described
generically: nodes are program units, attention replaces fixed neighbourhood
aggregation, and the training objective includes adversarially perturbed
samples so the model does not collapse onto one fragile feature.

- **inputs**: available now for the CFG variant; the adversarial-perturbation
  half needs binaries (to rewrite and re-disassemble) or a perturbation model
  applied at the instruction-stream level. This repo already has
  `generate_adversarial_train.py` / `generate_adversarial_test.py` and
  `vulnerability_test.py` from the earlier phase, which is the natural place to
  join the two lines.
- **compute**: GPU, hours to a day.
- **risk**: coordination, not method. Two people fitting different models to the
  same 14 test families will both report a number and neither will be
  comparable unless both go through `cnn_vit_pipeline/cohort.py`. **The
  integration requirement is one line: import the shared split.**
- **worth pursuing if**: it is already being pursued. The useful contribution
  from this side is the *harness* -- same cohort, same metrics schema, same
  per-family and per-arch breakdowns -- so the two results can be put in one
  table.

---

## 4. API-call graphs

Ransomware is defined behaviourally (enumerate, read, encrypt, overwrite,
delete shadow copies, drop note), and those behaviours are API calls:
`CryptEncrypt`, `FindFirstFileW`, `SetFileAttributes`, `DeleteFile`,
`WNetEnumResource`. An import-table-derived call graph is closer to the actual
concept than any opcode statistic.

- **inputs**: **needs re-extraction.** The `asm/` and `mn/` trees carry no
  import table and no symbol names -- `call dword ptr [0x405128]` is as much as
  the current data has, and the IAT address is not resolved to a name. The PE
  files are on disk, so a `pefile`-based pass over the imports directory is
  cheap and needs no VM and no execution. A *static* call graph (which imported
  function is called from which block) also needs the IAT-to-name mapping
  applied to the sweep's indirect call targets, which `extract_unified.py` does
  not currently emit.
- **compute**: import extraction is minutes for the whole corpus. The graph
  construction is the work.
- **risk**: **packing destroys imports.** ~7% of each class is packed and the
  cohort filter already drops entropy-flagged samples; but installer-wrapped
  samples (the same ones that cause the duplication in §0) will show the
  *installer's* imports, not the payload's. Import-based features on this
  corpus may measure "which installer was used".
- **worth pursuing if**: an imports-only bag-of-names baseline (a one-hot over
  imported function names, logistic regression -- an afternoon's work) beats
  0.80 macro-F1 on `balanced`. That is the cheap probe; do it before building
  any graph.

---

## 5. Classical baselines, for calibration

**Lead with this section.** The measured result is that the classical baselines
are not the floor, they are the ceiling.

### 5a. Mnemonic n-gram TF-IDF -- already done, currently the best result

macro-F1 0.968 (Mendeley) / 0.802 (Goodware_Balanced) as reported; **0.955 /
0.802 after removing test rows that are verbatim copies of a training row**;
0.926 / 0.809 with the training set deduplicated as well. Fully audited in
`results/rules/summary.md` §5-6: the fit is clean, the split is clean, the
label-permutation control collapses to chance, and the remaining inflation is
the Mendeley goodware duplication and nothing else. Remaining work is not "make
it better", it is the family-holdout version (§8), which is what would turn it
into a real claim rather than one draw from a 14-family lottery.

### 5b. Markov transition matrices over mnemonics

A |V|x|V| transition-probability matrix per file, flattened into a feature
vector. With the Mendeley vocabulary of 639 mnemonics that is 408,321 features
per file -- so in practice restrict to the top 100-150 mnemonics (10,000-22,500
features), which covers essentially all mass.

- **inputs**: available now.
- **compute**: trivial -- one pass over `mn/`, minutes.
- **risk**: this is mathematically very close to 2-gram TF-IDF (a row-normalised
  bigram count matrix). Expect it to land within noise of the existing 0.968,
  and expect that to be *uninformative* rather than confirmatory.
- **worth pursuing if**: as a cheap robustness check on 5a, yes -- an afternoon.
  As a new direction, no.

### 5c. HMM features (hmm2vec)

`Embedding/hmm2vec.py` in the tokenization repo
(`Tokenization-Testing-for-Malware-Data`) fits a `CategoricalHMM` per sample
over its opcode sequence, then takes the emission (B) matrix, reorders its rows
so the state that most strongly emits `mov` comes first, and flattens it into a
fixed-length feature vector.

- **inputs**: available now; the code exists and works on the same opcode
  representation.
- **compute**: **this is the expensive one.** One Baum-Welch fit per sample,
  `n_iter=100`, over sequences of ~20,000 symbols x 2,509 samples. Expect
  hours to a day on CPU, and expect a long tail on the largest files. Truncate
  the sequence (the existing pipeline truncates at 5,000) and it becomes
  practical.
- **risk**: the row-reordering-by-`mov` trick is what makes per-sample HMMs
  comparable at all, and it is fragile -- if two samples' `mov` state is
  genuinely different in character, the features are not aligned. Also, an HMM
  over a symbol stream captures roughly what a bigram model captures, so 5b's
  warning applies.
- **worth pursuing if**: it is already implemented in a repo this project
  compares against, so running it on **our** cohort and split is a cheap way to
  add a row to `results/summary.md` that reviewers will expect. Worth a day for
  completeness; not worth optimising.

### 5d. MalConv (and byte-level CNNs generally)

- **inputs**: **needs binaries.** MalConv eats the raw byte stream of the PE,
  not disassembly. The files are on disk and reading their bytes is safe (no
  execution), but nothing in the current pipeline produces them, and 2 MB of
  bytes per file at the standard MalConv input length is a different data
  loader entirely.
- **compute**: GPU; MalConv's embedding-plus-gated-convolution over 2M-byte
  inputs is slow but standard. A day of GPU time to train, plus the extraction.
- **risk**: MalConv is famous for latching onto PE header fields and section
  padding rather than code. On a corpus where the two classes come from
  different *collection processes* (a ransomware corpus and a goodware corpus
  assembled separately), that is the most likely outcome, and it would look like
  a great score. Guard with a header-only ablation.
- **worth pursuing if**: the CNN-ViT track (`results/cnn_vit/`) is the closest
  existing analogue and it reached 0.628 / 0.502. A byte-level model has to
  clear that convincingly on `balanced` before it is more than a literature box
  to tick.

---

## 6. Structural / header features, and combining tracks

The one thing that has *not* been tried is the obvious one: concatenate the
tracks. TF-IDF (0.968 / 0.802), WL graph features (0.895 / 0.554), behaviour
rules (0.797 / 0.670) and PE-header structural features are four fairly
different views of the same file.

- **inputs**: available now for three of the four; PE headers need a `pefile`
  pass (minutes, no VM, no execution). `ember_pipeline/` in this repo already
  covers part of this ground.
- **compute**: negligible. Stack the feature blocks, or average calibrated
  probabilities from the per-track models.
- **risk**: header fields are the purest shortcut in the corpus -- timestamps,
  linker version, section names and the machine field all encode *which corpus
  a file came from*. The WRITEUP's own position is that there must be no
  "packed" indicator feature for exactly this reason, and a machine-field-only
  classifier is reported as the floor. Any header feature has to be justified
  individually, and the machine field excluded outright.
- **worth pursuing if**: a simple probability-average of TF-IDF + behaviour
  rules + WL beats TF-IDF alone on `balanced`. That is a half-day of work and it
  is the highest-value-per-hour item in this document.

---

## 7. Per-architecture modelling / arch-balanced sampling

The confound is measured, not suspected: on `balanced`, goodware recall inside
x64 is ~0.99 and inside x86 is 0.70-0.83 for every structural representation
(`results/graph2vec/summary.md` §5). Two fixes:

- **train and evaluate separately within x86 and within x64.** The x86 slice is
  large enough on both datasets (train: 1,341 / 939 rows). The x64 slice is
  the interesting one and is thin on ransomware (42 train / 72 test), so an
  x64-only model is a genuinely harder and more honest task.
- **arch-balanced sampling**: resample or reweight so P(arch | class) matches
  across classes in train, and report the drop. The drop *is* the size of the
  shortcut.

- **inputs**: available now. `cnn_vit_pipeline/cohort.py` already carries the
  `arch` column on every row and `build_result` already emits per-architecture
  metrics, so the plumbing exists.
- **compute**: a rerun of whatever model, so minutes to an hour.
- **risk**: none, methodologically. The risk is to the headline numbers -- this
  will make them smaller, which is the point.
- **worth pursuing if**: unconditionally. This is a correctness fix, not a
  direction. The per-architecture TF-IDF models are already in
  `results/rules/baseline_audit.json` and should be the template.

---

## 8. Family-holdout cross-validation across all 40 families

Every number in this project comes from **one** split: 24 train families, 14
test families, fixed. `results/graph2vec/summary.md` shows per-family recall
running from 0.00 to 1.00 inside a single model's results. That means the
headline macro-F1 is, in part, a statement about which families happened to be
held out.

Replace it with leave-k-families-out cross-validation over all 38-40 families:
repeated stratified group splits on `family_or_group`, ~10 folds, reporting mean
and spread.

- **inputs**: available now.
- **compute**: 10x the cost of one run. For TF-IDF that is 30 minutes; for
  graph2vec, a few hours; for anything GPU-trained, the real cost of the idea.
  Run it on the cheap models first -- the point is to measure the *variance*,
  and that transfers.
- **risk**: the original Mendeley split is the paper's split, and abandoning it
  makes this work incomparable to the published baseline. The answer is to
  report both: the fixed split for comparability, the CV for honesty.
- **worth pursuing if**: unconditionally, and early. If the spread across folds
  turns out to be +/-0.10 macro-F1, then most of the comparisons in §0 are not
  comparisons, and that finding is worth more than any individual model.

---

## 9. Calibration and threshold choice for a 74%-ransomware test set

Train is 45% ransomware; test is 74%. Several graph2vec rows reach ROC-AUC
above 0.95 while sitting at macro-F1 0.53 at the default 0.5 cut -- the model
ranks correctly and decides wrongly. `graph2vec_pipeline/train_eval.py` already
picks a threshold from train out-of-fold scores and reports both operating
points, and on `graph2vec/SVM-RBF` that single change moves macro-F1 from 0.699
to 0.895.

Remaining work: proper probability calibration (Platt / isotonic on the val
fold), prior correction for the known train/test prevalence shift, and -- more
important than either -- **deciding what operating point the thesis is actually
claiming**. A detector that must not block benign software is a different
question from one that must not miss ransomware, and macro-F1 answers neither.

- **inputs**: available now. `cohort.add_val_fold` already carves a group-aware
  val fold.
- **compute**: negligible.
- **risk**: none. The risk is *not* doing it and reporting numbers at an
  arbitrary cut.
- **worth pursuing if**: unconditionally, and it is nearly free. Report FPR at
  fixed recall (e.g. TPR=0.95) alongside macro-F1 everywhere; that single column
  would make every table in this project more useful.

---

## 10. YARA-style signatures and association rules

Prototyped: `rules_pipeline/ngram_rules.py` mines mnemonic n-grams level-wise
(Apriori downward closure), scores them by lift and odds under a Jeffreys
prior, and reduces them with greedy set cover under a train-precision floor.
The current Mendeley rule set is 14 ransomware-leaning and 22
goodware-leaning rules covering 791/796 training ransomware, and individual
rules transfer to unseen families -- e.g. `xorps movlpd` fires on 134 test files
at 98.5% precision, and `mov adc mov` at 95.8% precision / 62.4% recall
(`results/rules/summary.md` §2).

- **inputs**: available now; for real YARA the rules would have to be expressed
  over bytes or strings rather than mnemonics, which needs binaries.
- **compute**: the existing miner is ~4 minutes per dataset.
- **risk**: the rules are mnemonic sequences, so they are not directly
  deployable as YARA (which matches bytes/strings); and a mnemonic n-gram is
  trivially evasable by instruction substitution or padding. The honest claim is
  *interpretability*, not robustness.
- **worth pursuing if**: as the interpretable companion to the TF-IDF result,
  it already earns its place -- a reviewer who asks "what is the model looking
  at" gets a 36-rule answer. As a standalone detector it is 17 points of
  macro-F1 behind the TF-IDF baseline on Mendeley and should not be pitched as
  one.
- **note for anyone extending it**: the miner and the scorer must agree on the
  n-gram encoding. They silently did not until `rules_pipeline/ngram_rules.py`
  was fixed (int16 overflow in `_codes`), and the symptom was a table of
  high-lift rules that appeared to fire on nothing. `tests/test_graph2vec_rules.py`
  now pins the invariant.

---

## Recommended order

1. **Fix the measurement before adding methods** -- family-holdout CV over all
   40 families (§8), per-architecture and arch-balanced runs (§7), and FPR at
   fixed recall reported everywhere (§9). Three items, all cheap, and until
   they are done no comparison in this project is load-bearing.
2. **Bank the classical result honestly** -- the mnemonic TF-IDF baseline is
   currently the best method on this corpus and it passed its leakage audit.
   Write it up as the **primary result, not as a baseline**, and quote the
   audited figures (0.955 Mendeley on non-duplicated test rows, 0.802
   Goodware_Balanced) rather than the headline 0.968, with the finding that
   what it has learned is compiler idiom stated in the same breath.
3. **Take the two cheap wins** -- stack the existing tracks (§6, half a day) and
   run an imports-only bag-of-names probe (§4, an afternoon). Both are hours of
   work against methods that took weeks.
4. **Then, and only then, one deep model** -- the GNN on the existing CFGs (§3a)
   is the cheapest and the only one whose inputs already exist; it must be
   reported against `size_only` and a degree-only ablation or it means nothing.
5. **Treat the LLM track as analysis, not detection** -- summary-level prompting
   (§1c-d) to explain samples and to generate candidate behaviour rules, always
   with a shuffled-mnemonic control, never as a reported detector.

**What this implies for the deep models.** A bag of mnemonic 1-3-grams with a
logistic regression on top beats the word2vec pipeline, the graph embedding and
the vision transformer on the same rows, at a few minutes of CPU. That is not
an argument that deep models cannot work on binaries -- it is an argument that
*this corpus, at this size, with this split*, does not have enough signal
beyond token statistics to distinguish them. 904 ransomware files from 24
training families is 24 effective examples; every model above 100k parameters
is fitting family idiom. The productive response is not a bigger model, it is
more families, a harder goodware set, and a cross-validated estimate of how
much of any reported gain is real -- all of which are items 1-3 above. Any
deep-learning result on this data has to clear 0.955 / 0.802 **under
family-holdout CV** to be worth the GPU, and nothing measured so far is close.
