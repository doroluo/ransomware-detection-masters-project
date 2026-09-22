# Code audit (2026-09-21)

Six read-only audits over every model pipeline in this branch and on the two
teammate branches (`dorothy-work`, `yanping`), consolidated. Line references are
to `LLM_Classification` at `26f5f7b` unless a branch is named. The baseline
test suite (402 tests) passes in 38 s before any change below.

Sizes: 30,944 lines of Python in this branch. About 13,000 of them are the
model pipelines; roughly 5,000 are Markdown report writers with frozen prose,
and roughly 2,500 are dead, duplicated or superseded.

## 1. Verdict per pipeline

| pipeline | lines | status | minimal keep | headline problems |
|---|---|---|---|---|
| `family_holdout/` (folds, common, runners, aggregate) | 3,500 | the evaluation layer; healthy core, drifting edges | ~1,900 | goodware label hard-coded in `folds.py:104-108`; no cross-corpus sha dedup; no family-name normalisation; three sd definitions; `run_cnn_vit.py` reimplements the whole contract (883 lines) |
| `rules_pipeline/` + `family_holdout/run_tfidf.py` | 2,130 + 228 | TF-IDF is the best model and the cleanest code; `rules_pipeline/` is superseded by `run_tfidf.py` | 228 | LinearSVC writes an unbounded margin into the `score` column that the ensemble treats as a probability |
| `seq_model/` | 3,050 | sound; best deep model | ~1,850 | fingerprint ignores the imports file content; env var named `RANSOM_SHARED` while everything else uses `RANSOM_SHARED_DIR`; pretrain resume replays the same windows |
| `graph2vec_pipeline/` | 4,400 | working but two generations coexist | ~2,550 | `final_eval.py:195` computes arch sample weights over test rows (latent leak); OOF threshold fitted on the WL block but applied to WL+imports; PV-DBOW stops on wall-clock time (load-dependent embeddings); legacy `train_eval.py` still uses ungrouped 2-fold CV |
| `imports/` | 490 | correct extractor | ~350 | parse failures and genuine empty import tables both become an all-zero vector |
| `llm_features_pipeline/` + `Tokenization/` | 4,330 + 156 | works; "LLM" is a misnomer (subword tokenizer + word2vec + RF/SVM/MLP) | ~1,800 | vendored `Tokenization/tokenization.py` is a stale copy that shadows the real module; 5,000-line head cap on 144,000-line files; 1,130 lines of report prose with hard-coded numbers |
| `cnn_vit_pipeline/` + `CNN-ViT/` | 3,800 + 1,080 | evaluated properly (0.68 / 0.74 macro-F1), weakest model | ~1,400 | augmentation pads with a token the encoder never emits; image and mask shifts desynchronise by up to one patch; per-arch macro-F1 halved on single-class buckets; token cache has no fingerprint; `stratified_split.py` dead; 158-line dead `__main__` |
| `ember_pipeline/` | 906 | never produced a result; no callers; no tests | ~250 or delete | features are a hand-rolled approximation, not EMBER; goodware extracted on the host and ransomware on the VM with LIEF unpinned (environment confounded with label); booster never saved |
| Dorothy `dorothy-work` GIN | 1,990 | real contribution is `gin_model.py` (163) + graph construction (~150) | ~500 | 875 lines re-implement `extract_unified.py` + `folds.py`; `arch` dropped before metrics so the x86 shortcut cannot be measured; val-tuned threshold degenerates to 0.0 under family-disjoint (test precision 0.83); report leads with best-of-5 seed; its own baseline was never run |
| Yanping `yanping` CT_GAT + CNN-ViT | 2,320 | CNN-ViT files are byte-duplicates of ours; CT_GAT transformer exists, GAT is two empty files | ~300 | noise padding encodes the sample's own mean token id and length into the canvas, unseeded; `getprocaddress` lines dropped by a substring filter; two tokenizers diverge; random label-stratified split, not family-disjoint; no results committed |

## 2. Bugs by severity

Severity: **A** changes a reported number or leaks labels; **B** silent wrong
output on a plausible input; **C** crash, dead code or drift.

### A. Numbers and leakage

| # | where | what | fix |
|---|---|---|---|
| A1 | `family_holdout/folds.py:104-108` | every row read from the balanced cohort gets `label = 0`; ransomware appended there is relabelled goodware and every assertion still passes | filter `label == "0"`, carry the label through |
| A2 | `family_holdout/folds.py:115`, `common.py:70-81` | no sha256 dedup across corpora; a shared sample lands in two folds and in `predictions.csv` twice | dedup with Mendeley priority; assert unique sha in `Folds` |
| A3 | `family_holdout/folds.py:84`, `common.py:104-111` | family names are used verbatim (`Conti` and `conti` are two families in two folds); `counts()` selects by name ignoring label | normalise + alias table; select by `(name, label == 1)` |
| A4 | `common.py:185` (ddof=1) vs `aggregate.py:46`, `run_ensemble.py:241`, `run_cnn_vit.py:366` (population) vs `seq_model/make_summary.py:62` | three fold-sd definitions, 12% apart, printed under one label; the README's decision rule depends on which file you read | one helper, sample sd |
| A5 | `common.py:121`, `run_cnn_vit.py:410`, `tools/compare_pipelines.py:45` | majority floor uses the test fold's labels (an oracle) | derive from the training folds |
| A6 | `cnn_vit_pipeline/cohort.py:276, 311-313` | `_safe_div` maps 0/0 to 0.0, so per-arch macro-F1 is halved whenever a bucket has one class; `aggregate.py` prints it | return NaN and print blank |
| A7 | `graph2vec_pipeline/final_eval.py:102, 195` | `arch_sample_weight` computed over all rows including test | restrict to train rows (as `tune.py:397` does) |
| A8 | `family_holdout/run_graph2vec.py:249-253` | OOF threshold fitted on the WL block, applied to a model fitted on WL + imports | compute on the same matrix |
| A9 | `family_holdout/run_stacker.py:100-103` | LOFO combiner fitted on 3-seed K-fold scores, applied to 1-seed LOFO scores | document, or fit on LOFO-scale scores |
| A10 | `family_holdout/run_tfidf.py:109-110`, `run_ensemble.py:70-83`, `run_stacker.py:62-64` | LinearSVC `decision_function` written as `score`; ensemble averages it with a probability, stacker clips it to two constants | assert scores in [0,1] when combining; sigmoid-calibrate or refuse |
| A11 | `CNN-ViT/model_train.py:66, 70-71` | augmentation pads with pixel 0 (`PADDING`) while the encoder pads with `END_PAD`=3, on 40% of training samples and never at eval; image shifts by pixel rows, mask by `ceil(shift/16)` patch rows inserted at `floor(row/16)` | pad with `END_PAD/255`; shift image by whole patch rows so the mask matches |
| A12 | `ember_pipeline` (whole) | goodware features on the host, ransomware on the VM, LIEF unpinned; the 21 "pruned" structural dims are exactly the ones LIEF 0.x and 1.0 report differently | extract both classes on the VM, pin LIEF, record its version in the npz |
| A13 | `graph2vec_pipeline/embed.py:162-166` | PV-DBOW stops on `max_seconds`; embeddings depend on machine load; `repro_check` compares them bit-for-bit | stop on a step budget |
| A14 | `asm_tool/cap_api_vocab.py:11-24`, `graph2vec_pipeline/tune.py:242-260`, `llm_features_pipeline/tune.py:222-252`, `seq_model/pretrain.py` | corpus-wide (test-inclusive) vocabulary or embedding decisions baked into inputs | documented transductive steps; keep, but say so in every summary that uses them |
| A15 | `dorothy-work/build_graphs.py:241-249` | `arch` dropped from the graph payload; no per-arch metric possible | carry it |
| A16 | `dorothy-work/train_gin.py:105-113` | threshold search returns the minimum probability when val recall saturates; all five family-disjoint seeds report threshold 0.0 | pick on out-of-fold train scores, or fix 0.5 |
| A17 | `yanping/CNN-ViT/asm_parser.py:262-272` | empty canvas filled with `normal(mean(tokens), 0.15*std)`, unseeded; the conv stem is unmasked so file length and mean token id reach the classifier | pad with a constant (`END_PAD`), seed |

### B. Silent wrong output

| # | where | what |
|---|---|---|
| B1 | `llm_features_pipeline/run_pipeline.py:132-133, 561-599` | files that normalise to nothing are dropped after counts and floors were computed over the full set |
| B2 | `llm_features_pipeline/data.py:24, 41-43` | any ransomware filename not matching `<family>_<32+ hex>.txt` becomes family `unknown`; all positives then share one group |
| B3 | `llm_features_pipeline/data.py:76-80, 134-137` | a mistyped manifest path degrades to no grouping and no dedup instead of failing |
| B4 | `llm_features_pipeline/tune.py:321-329` | `TypeError` on `sample_weight` is swallowed; MLP "class_arch" rows are unweighted duplicates |
| B5 | `imports/extract_imports.py:193-195`, `merge_imports.py:48` | parse errors and empty import tables both become `[]` and then an all-zero vector |
| B6 | `imports/merge_imports.py:71-72` | missing fold file skips the coverage check silently |
| B7 | `graph2vec_pipeline/build_graphs.py:85-87`, `graph_cache.py:149-151` | unparseable `.asm` becomes an empty graph, a legitimate zero row |
| B8 | `cnn_vit_pipeline/encode.py:161, 197-203` | token cache keyed by tree name only; editing `TOKEN_MAP` or regenerating `.asm` leaves a stale 300 MB cache in use |
| B9 | `cnn_vit_pipeline/tuned_train.py:184` | cohort rows with no image dropped without a count |
| B10 | `seq_model/run_family_holdout.py:119-129` | fingerprint ignores the imports file; a regenerated `imports_flat.json` reuses old runs |
| B11 | `ember_pipeline/extract_features.py:49-54, 81`; `train_eval.py:174` | skipped files get no manifest row; empty input raises after writing an empty npz; `rsplit` keeps the full parent path as the family |
| B12 | `ember_pipeline/ember_extractor.py` (9 sites) | bare `except Exception` turns LIEF failures into all-zero structural blocks tagged `ok` |
| B13 | `family_holdout/run_ensemble.py:163-165, 301-305` | accepts any number of members, uses exactly two |
| B14 | `yanping/CT_GAT/preprocessing/token_mapping.py:236` | `"proc"` substring filter over the whole line drops every `GetProcAddress` call |
| B15 | `yanping/CT_GAT/preprocessing/prepare_dataset.py:92, 123-125` | flat `.asm` dir keyed by stem; a basename present in both classes gets the ransomware label and one disassembly overwrites the other |
| B16 | `dorothy-work/extract_opcodes.py:258-285` | its `--family-disjoint` copy of the splitter sends all goodware to train |

### C. Dead, duplicated, brittle

- Three byte-identical copies of `extract.py` (root here, `dorothy-work/`, `yanping/CNN-ViT/`), all superseded by `Shared/extract_unified.py`; the root copy runs its pipeline on import.
- `Tokenization/tokenization.py` (156) + `__pycache__`: stale vendored copy that shadows the external module; `llm_features_pipeline/tune.py:201-219` exists only to defend against it.
- `CNN-ViT/stratified_split.py` (76) dead; `CNN-ViT/model_train.py:393-550` dead `__main__` with `/home/yl/...` paths; `cls_token` allocated and unused.
- `ember_extractor.extract_features_from_folder` (29) unused; `sanitize` path never invoked; `ember_pipeline/wheels/` referenced but absent.
- `graph2vec_pipeline/{build_graphs,train_eval}.py` (449) superseded by `graph_cache/tune/final_eval`; `rules_pipeline/{train_eval,baseline_audit,baseline_ablation,make_summary}.py` (1,611) superseded by `run_tfidf.py`.
- `llm_features_pipeline/tune.py:337-370 cv_evaluate` never called; `check_schema.py` orphaned.
- `family_holdout/make_summary.py` only knows three pipelines and crashes on a `cnn_vit` directory (`floors` vs `floors_pooled`).
- Seven copies of the shared-folder roots with two env-var names; four copies of the floors; four of `best_threshold`; two of `hash_imports` (pinned equal by a test); five of "mean ± sd"; three YAML emitters and one YAML parser.
- Hard-coded `C:/Users/chaoa/...` with no env override: `folds.py:56-59`, `run_tokenization.py:75-76,350`, `llm_features_pipeline/config.yaml:13-39`, `seq_model/config.yaml:64,116-117`.
- `vm_package/make_package.py` ships stale contents; `run_imports_vm.sh` points at a wheels directory that does not exist.
- Report writers: `graph2vec_pipeline/make_summary.py` (1,004; one 342-line function), `rules_pipeline/make_summary.py` (619), `seq_model/make_summary.py` (587), `run_pipeline.py:711-1844` (1,130) with ~280 lines of prose quoting hard-coded numbers.

## 3. Minimal-implementation plan

Order chosen so each step is testable on its own and the headline numbers can
be regenerated after it.

1. **Evaluation layer first** (`family_holdout/`): fix A1 to A6, add the
   corpus column, cross-corpus dedup, family normalisation, arch-aware fold
   balancing, one sd helper, one floors helper, train-derived majority floor.
   Rewrite `run_cnn_vit.py` on `common.py` like the other six runners. Delete
   `make_summary.py` into `aggregate.py`.
2. **Shared paths module** (`repo_paths.py`): the shared roots, the
   `corpus -> tree` map, one capped stream reader, one env-var name. Replace the
   seven copies.
3. **Delete dead code**: root `extract.py`, `Tokenization/`, the CNN-ViT
   `__main__` and `stratified_split.py`, the legacy graph2vec and rules
   runners, `cv_evaluate`, `extract_features_from_folder`, the vm_package
   staleness.
4. **Per-pipeline correctness**: A7 to A13, B1 to B12.
5. **Report writers**: derive every number from the payload; drop the prose
   blocks; one writer per pipeline at most, ideally one for all.
6. **Teammate branches**: propose the `mn/` + fold-file adapter for the GIN
   (replaces 875 lines) and the `mn_api/` adapter for CT_GAT, plus the arch
   breakdown, in a PR to each branch rather than editing them here.

Steps 1 to 3 change no model; steps 4 and 5 change some reported numbers and
every affected summary must be regenerated and the write-up updated.
