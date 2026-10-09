# CFG graph transformer on the IDA Mendeley export

## Goal

Classify Mendeley binaries as goodware (0) or ransomware (1) from the IDA Pro
control-flow graphs, with a transformer whose attention is shaped by the graph,
and measure whether the graph structure adds anything over the same model with
the structure removed.

## Data

Source: `Mendeley_GNN_v1` (IDA 9.4, schema `ida-cfg-research/2.1`), built from
the VM extraction by `build_gnn_bundle.py` and kept at
`ransomware-detection-masters-project/reports/google_drive_upload/2026-10-07/Mendeley_GNN_v1`.
Each sample is one gzip JSON-lines file: a header (PE metadata, import table),
one record per IDA function (library and thunk flags, basic blocks, intra-function
edges, instructions with mnemonic, operand categories, code references and
resolved import ids), string records and a completion footer.

The ASM listings (`Mendeley_ASM_v1`, `Downloads/idapro_asm/`) are not used: the
CFG records already carry every instruction the listings print, plus the block
boundaries and edges the listings only imply.

### Filters

| Rule | Removed | Why |
|---|---|---|
| `recommended_split` empty | 236 | managed .NET, unsupported machine, no recovered function instructions, cohort conflicts (the bundle's own policy) |
| `representation == upx_packed` | 23 | IDA saw the UPX stub, not the program: 1 to 5 functions of identical decompressor code in either class |
| `architecture == x64` (primary runs only) | 603 | the architecture confound, below |

Usable after the first two rules: 2,411 binaries (train 1,932, test 479).

### The architecture confound

| split | ransomware x86 | ransomware x64 | goodware x86 | goodware x64 |
|---|---:|---:|---:|---:|
| train | 854 | 42 | 573 | 481 |
| test | 289 | 68 | 115 | 12 |

(counts before the UPX filter). In train, x64 almost always means goodware; in
test, x64 is mostly ransomware. A model that learns "x64 means goodware" scores
well on validation and badly on test. The primary experiment is therefore
x86-only. The all-architecture run is kept as a secondary result with per-arch
recall. Register names and operand widths are never model inputs, because both
encode the architecture; operands enter only as categories (register,
immediate, phrase, displacement, memory, code target).

### Evaluation protocols

1. **Bundle split.** The bundle's train/test split is family-disjoint (23
   ransomware families in train, 15 different ones in test), so it is a family
   holdout. A validation set for early stopping is carved from train by whole
   families (ransomware) and by `identity_group` (goodware). The test set is
   scored once per trained model.
2. **Project 5-fold family holdout.** Every eligible sample joins
   `results/family_holdout/folds_mendeley.csv` (directly or through
   `provenance/identity_aliases.csv`), so the same folds as the seq transformer,
   graph2vec and TF-IDF results in `results/family_holdout/summary.md` apply.
   Pooled out-of-fold predictions give numbers comparable to that table
   (restricted to the eligible subset, which the table notes).

Reported per run: accuracy, precision, recall and F1 for the ransomware class,
macro-F1, ROC AUC, recall per class per architecture, recall per held-out
family, and the floors (majority class, x86 rule) on the same manifest. Three
seeds; mean and sample standard deviation.

## Cache

`build_cache.py` streams each export once (process pool, at most four workers,
one function at a time) and writes one `.npz` per sample plus `index.csv` and
`vocab.json` to a cache directory outside the repository. Per sample:

| array | dtype, shape | meaning |
|---|---|---|
| `mn` | int32 [I] | mnemonic id per instruction |
| `sig` | int32 [I] | id of mnemonic + operand-category signature (`xor r,r`, `mov r,m`) |
| `beh` | int8 [I] | behavior id from Yanping's `token_mapping.BEHAVIOR_MAP` (crypto opcode, file API, ...) |
| `api` | int32 [A] | import ids referenced by instructions, flattened |
| `blk_insn_ptr` | int32 [B+1] | instruction offsets per block |
| `blk_api_ptr` | int32 [B+1] | api offsets per block |
| `blk_feat` | float16 [B, 6] | the bundle's six structural block features |
| `fn_blk_ptr` | int32 [F+1] | block offsets per function (address order) |
| `fn_flags` | int8 [F] | bit 0 library (FLIRT), bit 1 thunk |
| `cfg_edges` | int32 [E, 2] | intra-function edges, global block indices |
| `call_edges` | int32 [C, 2] | caller, callee function indices (call instructions whose code reference is a function start) |

Vocabularies are global string tables; the model keeps a token only if it
occurs in at least two training binaries of the current run and maps the rest
to UNK, so nothing is fitted on validation or test data.

## Model

Hierarchical: instructions form a block, blocks form a function, functions form
the binary. Graph structure enters at both upper levels as attention bias.

1. **Block encoder.** Mean of embeddings of `mn`, `sig`, `beh` and the block's
   API ids, plus a linear map of the six structural features. Width 128.
2. **Function encoder (CFG graph transformer).** Functions longer than 128
   blocks are cut into address-ordered segments of at most 128 blocks; every
   block is kept. Per segment, two pre-norm transformer layers with 4 heads.
   Attention logits get a learned per-head bias indexed by the directed
   shortest-path distance between the two blocks, forward and backward, each
   clipped to {0, 1, 2, 3, 4+, unreachable} (Graphormer). In- and out-degree
   embeddings are added to block inputs. Attention-pooled to one vector per
   segment.
3. **Binary encoder (call-graph transformer).** Up to 512 segments per binary.
   Two transformer layers over segment vectors and a CLS token, with a learned
   per-head bias for caller to callee, callee to caller, and same-function
   segments. Segment features add log block count and the library flag. CLS
   goes to a linear classifier.

Segment selection when a binary has more than 512: non-library first, then by
number of API references, then by size; training draws a random subset weighted
the same way each epoch, evaluation uses the deterministic order.

**Ablation (`--no-graph`).** The same network with every structural input
removed: no shortest-path or call bias, no degree embeddings, the in/out degree
columns of the structural features zeroed. What remains is a set transformer
over bags of instructions. If macro-F1 does not move, the graph is not helping,
and the report says so.

**Training.** AdamW (lr 3e-4, weight decay 0.01), bf16 autocast, batch of 4
binaries, class-balanced loss, up to 40 epochs, early stopping on validation
macro-F1 with patience 8, threshold 0.5.

## Baselines on the identical manifests

- TF-IDF over `sig` tokens and API names, logistic regression.
- Mean and max of the six structural block features plus log counts, logistic
  regression.
- Floors: majority class, x86 rule.

## Files

| file | role |
|---|---|
| `build_cache.py` | export to cache |
| `data.py` | manifests, filters, folds, vocab cut, per-binary tensors |
| `model.py` | the hierarchical graph transformer and the ablation switch |
| `train.py` | one run: protocol, arch filter, seed, ablation; writes predictions and metrics |
| `baselines.py` | TF-IDF and structural logistic regressions on the same manifests |
| `report.py` | aggregates runs into `RESULTS.md` |

## Results (2026-10-08)

Full tables: `RESULTS.md` (`report.py`, threshold 0.5) and `val_threshold.py`
(threshold chosen on validation, per-fold mean ± sd). Test-set macro-F1; bundle rows
are mean ± sd over seeds 0 to 2, kfold rows are per-fold mean ± sd (seed 0) and pooled.

| setting | graph transformer | no-graph ablation | TF-IDF sig + API, LogReg | structural LogReg |
|---|---|---|---|---|
| bundle split, x86 | 0.857 ± 0.071 (AUC 0.972 ± 0.003) | 0.862 ± 0.068 (AUC 0.903 ± 0.034) | 0.956 ± 0.000 | 0.733 ± 0.044 |
| 5-fold family holdout, x86 | 0.887 ± 0.076, pooled 0.886 (AUC 0.938) | 0.913 ± 0.076, pooled 0.911 (AUC 0.972) | pooled 0.939 | pooled 0.816 |
| bundle split, all arch | 0.932 ± 0.007 | 0.803 ± 0.122 | 0.885 ± 0.039 | 0.721 ± 0.079 |

Floors on the x86 test manifest: 0.417 (majority and x86 rule coincide).

1. On the headline setting (x86, family-disjoint) the graph bias does not help.
   Graph and no-graph differ by less than one seed or fold standard deviation, and a
   logistic regression over the same CFG tokens beats both by 0.03 to 0.10 macro-F1.
2. On the bundle split the graph model ranks better (AUC 0.972 ± 0.003 against
   0.903 ± 0.034) but its 0.5 operating point is unstable: goodware recall is 0.80,
   0.50 and 0.87 for seeds 0 to 2. Choosing the threshold on validation does not fix
   it (chosen thresholds 0.67 ± 0.36). The validation carve is 243 binaries with five
   ransomware families; validation macro-F1 swings 0.70 to 0.84 between epochs and the
   best epoch ranges from 2 to 15. This is the likely source of the seed spread.
3. With all architectures the graph model is the most stable and the best of the four
   (0.932 ± 0.007; x64 recall R 0.88 / G 0.97), while the ablation collapses on x64
   goodware (G x64 0.42) and TF-IDF loses x86 goodware (G x86 0.70). The test set has
   only 12 x64 goodware, so this is a robustness observation, not a headline.
4. The first full run capped binaries at 512 segments and truncated 776 of 2,411;
   test macro-F1 was 0.700, and recall on truncated binaries was 0.55. The cap was
   raised to 4,096 (53 truncated, 6.3 GB peak GPU) before the matrix; nothing else
   changed.

Comparability: these kfold numbers cover x86 bundle-eligible binaries only (1,810).
`results/family_holdout/summary.md` scores all architectures over the 2,509-row
cohort, so its seq-transformer (0.923) and graph2vec (0.942) rows are not the same
manifest. The same-manifest anchor is the TF-IDF row above.

Open: kfold has one seed; two more seeds (about 2.5 h on the RTX 5080) would firm up
point 1. A larger, family-stratified validation carve, or model selection by
validation AUC, is the first thing to try against point 2.
