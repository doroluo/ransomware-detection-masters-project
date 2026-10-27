# Review of the `gnn` and `yanping` branch pipelines (2026-10-19)

This page reviews the two GAT pipelines on `origin/gnn` (Dorothy, 331ce0b, which also carries Yanping's
`preprocess_full.py` from `origin/yanping` 3fd99e1 and Janice's AutoTokenizer work), explains what had to change to
compare them with our models, and lists fixes the owners may want to make on their branches. Nothing on those branches
was modified: our adapters in `experiments/team_pipelines/` import the code pinned by sha256 and only feed it sample
lists and collect its outputs.

| Pipeline | Files | Nodes | Model |
|---|---|---|---|
| A, opcode mode | `construct_blocks.py --gnn`, `train_cfg.py` | 57 numbers per block: mnemonic mix, API-group counts, size and degree | 2 GATConv layers, mean pool |
| A, CodeBERT mode | `construct_blocks.py`, `train_cfg.py` | 768-d frozen `microsoft/codebert-base` vector of each block's text | same |
| B | `CT_GAT/preprocessing/preprocess_full.py` (+ `tokenize_asm_gz.py`), `graph_full_loader.py`, `gat_full_model.py`, `train_full_gat.py` | 141 numbers per block: a 64-d pooled five-channel token embedding and 77 counts and rates | 2 GATConv layers with typed edges (edge_dim 8), mean + max pool |

## 1. Where `train.csv` comes from (Janice's error)

`tokenize_asm_gz.py` (line 425) looks for `CT_GAT/data/Mendeley_Ransomware/` and `CT_GAT/data/Mendeley_Goodware/` (the
`gnn` branch moved this to `<repo>/data`) and reads `<folder>/splits/{train,test}.csv` in each. Those folders are the
per-collection zips of the 2026-10-08 Drive upload, `Collections/Mendeley_Ransomware.zip` and
`Collections/Mendeley_Goodware.zip`: each holds `samples.csv`, `splits/{train,test,unassigned}.csv` (with the
`collection` and `asm_path` columns the code reads) and `samples/{train,test}/<family>/<sha256>.{jsonl.gz,asm.gz}`.
`Mendeley_CFG_v1.zip` is the older single bundle (`Mendeley_GNN_v1/`, one shared `splits/` with `excluded.csv`, CFGs only,
no `.asm.gz`), so the path does not exist and the tokenizer would have no listings to read. Unzip both collection zips
into the data folder, or pass both folders with `--root` (repeatable). The sample hashes and CFG files are identical in
both packagings.

## 2. What makes the published numbers incomparable

These apply to both pipelines unless marked.

1. **One fixed split.** Both train and test on Mendeley's original family-disjoint split (24 train / 14 test ransomware
   families). One split with 14 test families has a wide spread; our tables use 5 family-holdout folds and paired
   cluster-bootstrap intervals.
2. **That split is inverted on x64.** Train x64 is 481 goodware / 42 ransomware; test x64 is 12 goodware / 68 ransomware.
   Any feature that tells x64 from x86 learns "x64 means goodware" in training and is wrong on test. Our headline is
   x86-only for this reason.
3. **Validation shares families with training.** `train_cfg.stratified_holdout` (line 41) and
   `graph_full_loader.split_graphs` (line 172) take a random 15% per label. Early stopping and the decision threshold are
   then tuned on families the model has already seen, which is optimistic. Our `carve_validation` holds out whole
   families (and goodware identity groups).
4. **No per-sample predictions.** Neither trainer saves test probabilities by sample, so paired comparisons and AUC
   intervals are impossible. `train_cfg.py` also reports no AUC.

## 3. Pipeline A findings

- **Register names in the CodeBERT text.** `_operand_text` (line 135) writes raw register names (`rax`, `r8`, `eax`),
  decimal immediates and hex addresses into the block text. Under the inverted split (item 2) register names are an
  x64 detector, so the CodeBERT arm can score by architecture. Text is also cut at 128 subword tokens, about the first
  15-25 instructions of a block.
- **`--limit` returns one class.** `load_split_rows` (line 80) sorts by label then id, and `--limit` takes the prefix, so
  the README's `--limit 4` run is goodware only.
- **Stale graphs are reused.** Existing `.pt` files are skipped unless `--overwrite` (lines 657, 733), so a new
  `--max-blocks` silently keeps old graphs; `train_cfg.load_split` (line 27) reads every `.pt` in the folder with no check
  against the split.
- **Failures vanish.** Per-sample exceptions are caught and skipped (lines 705, 753), so the test set can shrink without
  notice.
- Smaller: in-degree and out-degree are undercounted in truncated functions; `je`/`jne` bins are always zero because IDA
  writes `jz`/`jnz`; the `urlmon` key never matches; `inspect_graph.py` defaults to Janice's old `graph_0.pt` naming.
- Janice's `origin/AutoTokenizer` version is fully superseded by this one (merged in 55e709b, rewritten in 3445c0c). Its
  ~30 MB of `.pt` files were removed from the tree in b2b066f but remain in git history.

## 4. Pipeline B findings

- **API vocabulary fitted on train and test.** `--split all` is the default (line 1784) and the vocabulary grows on every
  sample unless `--freeze-vocab` is set (lines 1093-1106). Test-only API names then get their own ids.
- **The "learned" embedding is random and unseeded.** The five-channel embedding is pooled under `no_grad` and never
  trained, and no `torch.manual_seed` is set, so the 64-d half of every node changes between preprocessing runs.
- **Size caps drop mostly ransomware.** The default caps (asm over 200 MB, over 80k blocks or over 450k instructions,
  lines 72-74) skip 89 of 2,434 bundle samples, 82 of them ransomware.
- **Listing-to-CFG alignment fails mostly on ransomware.** Each CFG instruction is matched to a line of the `.asm.gz`
  listing. In the first 511 binaries built for our folds, 68 of 340 ransomware (20%) but only 2 of 171 goodware (1%)
  have under half of their selected instructions matched; the worst families are stop, pysa, playcrypt and gandcrab.
  Unmatched instructions lose their opcode and operand channels, so "missing features" is itself a ransomware signal.
  The likely cause is that the Mendeley listings (`ida_asm_mendeley_20261006`) and CFG exports (2026-10-07) come from
  different IDA runs, which disagree most on packed or obfuscated code; this is a hypothesis, not yet checked.
  (Corrected in section 12: the listings collapse library functions; fixed in main PR #10, section 13.1.)
- **Operand classes depend on architecture.** `_REG_DATA` (line 241) covers `r8`-`r15` and `eax`-`dl` but not `r8d`,
  `xmm*`, `ah`-`dh`, `sil`/`dil` and others, which fall through to `MEM_GLOBAL_REF` (lines 267, 274), so x64 code has
  inflated `MEM_GLOBAL_REF` rates. On x86-only folds this matters less.
- **"signal" sampling is not one connected subgraph.** It ranks whole functions by imports and calls and returns their
  union; the module docstring says connected, and `connected_keys` is unused. `walk` is connected.
- Bugs: `_selected_rows` reads `row["collection"]` (line 1201), which the bundle CSVs lack (KeyError); the CFG path is
  derived from the listing's folder, so the bundle layout (CFGs and listings apart) fails; the vocabulary is saved after
  every sample, including `--limit` runs; a rerun into the same folder with a different `--sample`, `--seed` or budget
  silently mixes old and new graphs (`can_resume`); `train_full_gat.roc_auc` (line 47) returns 0.0 instead of NaN for a
  one-class set.

## 5. How we ran them on our folds

`experiments/team_pipelines/README.md` has the commands. In short, for kfold x86 (1,810 binaries, 5 family-holdout
folds, seeds 0 1 2):

- every arm uses our fold manifests, family-held-out validation, our threshold rule, and writes our predictions format,
  so `stats.py` and `gates.py --stage team` pair it with our models on identical test ids;
- no sample is skipped (size caps off; a failure stops the build);
- A runs as published (opcode mode, 500-block budget, their recipe); for CodeBERT we used the owner's choice of block
  text built from our export's register-free strings (`xor r,r ; kernel32!createfilew`), fed to Dorothy's GAT (`cb500`)
  and to our GNN (paths 1 and 2), instead of the raw-operand text;
- B is preprocessed once with `signal` sampling at 500 blocks and a fixed seed; per fold, API ids unseen in that fold's
  training binaries become UNK and the embedding is re-pooled, so nothing is fitted on test;
- our own GNN also runs at the same 500-block cap (`b500`) so budget is not the difference.

## 6. Results (final, 2026-10-20)

kfold x86, 1,810 test binaries (1,808 for sig500, which excludes the two 341 MB listings), AUC checkpoint at its validation
threshold. The table is seed 0 pooled over the five folds; the paired differences use seeds 0 1 2 where both arms have them
(TF-IDF, the header-imports LR and our b500 arms: seed 0 only, a direction). Source: `RUNS/gates_team.md` (written by
`gates.py --stage team`); the same values come out of `python -m evaluation.stats`. Ransomware is the positive class.
The rows of the fixed build (`sig500v3`, Models 2 and 3), the uncapped arms and the five-channel Model 1 were added later: section 13.

| model | AUC | PR-AUC | macro-F1 | accuracy | precision | recall | TP | FN | FP | TN |
|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF | 0.982 | 0.990 | 0.935 | 0.939 | 0.951 | 0.951 | 1079 | 55 | 55 | 621 |
| header-imports LR | 0.985 | 0.991 | 0.954 | 0.957 | 0.971 | 0.960 | 1089 | 45 | 33 | 643 |
| our typed GNN (R7), whole graph | 0.986 | 0.990 | 0.935 | 0.938 | 0.965 | 0.935 | 1060 | 74 | 38 | 638 |
| our no-edge GNN (R2), whole graph | 0.977 | 0.983 | 0.941 | 0.945 | 0.964 | 0.947 | 1074 | 60 | 40 | 636 |
| ours b500 typed (500 blocks) | 0.923 | 0.954 | 0.843 | 0.848 | 0.932 | 0.817 | 926 | 208 | 68 | 608 |
| ours b500 no-edges (500 blocks) | 0.955 | 0.964 | 0.915 | 0.919 | 0.959 | 0.910 | 1032 | 102 | 44 | 632 |
| A, `gnn500` (opcode mode, 500 blocks) | 0.956 | 0.975 | 0.850 | 0.854 | 0.939 | 0.820 | 930 | 204 | 60 | 616 |
| A on CodeBERT vectors, `cb500` | 0.791 | 0.872 | 0.723 | 0.724 | 0.925 | 0.608 | 690 | 444 | 56 | 620 |
| B, `sig500` (first build, 331ce0b) | 0.953 | 0.974 | 0.894 | 0.898 | 0.962 | 0.871 | 986 | 146 | 39 | 637 |
| B, `sig500v2` (streaming build, c7899f0) | 0.948 | 0.972 | 0.874 | 0.886 | 0.881 | 0.945 | 1072 | 62 | 145 | 531 |
| our GNN on the CodeBERT vectors (path 2) | 0.961 | 0.979 | 0.937 | 0.941 | 0.958 | 0.947 | 1074 | 60 | 47 | 629 |

Against R7 (3 seeds, second minus first): `gnn500` dAUC -0.025 [-0.043, -0.009], dmacro-F1 -0.072 [-0.136, -0.008]; `cb500` -0.159
[-0.221, -0.104], -0.184 [-0.242, -0.129]; `sig500` -0.029 [-0.054, -0.010], -0.050 [-0.118, +0.014]; `sig500v2` -0.031 [-0.058, -0.009],
-0.059 [-0.105, -0.007]. 3-seed means (AUC / macro-F1): `gnn500` 0.954 / 0.863, `cb500` 0.820 / 0.751, `sig500` 0.949 / 0.886,
`sig500v2` 0.948 / 0.876, against R7 0.979 / 0.935. No team arm beats R7 under the pre-registered rule. A and `cb500` lose on ransomware
recall, not on false alarms; `sig500v2` has the best recall of the team arms and the most false alarms (145).

**The cap.** The teammates' pipelines select at most 500 blocks per binary; our models read the whole graph (up to a 500,000-block
memory guard). To separate the cap from the model, our own GNN runs at the same budget (`ours b500`, `--subgraph cap --budget 500`:
whole functions ranked non-library first, then API references, then size). The cap costs it: typed 0.986 -> 0.923, dAUC -0.063
[-0.105, -0.029], dmacro-F1 -0.092 [-0.182, -0.003] (not shown non-inferior); no-edge 0.977 -> 0.955, -0.021 [-0.041, -0.004]. At
equal budget (seed 0, a direction): `gnn500` and `sig500` are nominally above our b500 typed GNN (`gnn500` dAUC +0.032 [-0.002, +0.071];
`sig500` +0.030 [-0.003, +0.068], dmacro-F1 +0.051 [+0.013, +0.104], the one interval that excludes zero) and level with our b500
no-edge GNN (`gnn500` +0.001 [-0.018, +0.018]; `sig500` -0.002 [-0.025, +0.020], 1,808 binaries); our b500 no-edge GNN is above our
b500 typed GNN (+0.032 [+0.003, +0.068]). No team arm passes the pre-registered rule against either b500 arm. The cap costs our own
models 0.021 (no edges) to 0.063 (typed) AUC, which is of the size of the 0.025 to 0.031 AUC gap between `gnn500` / `sig500` /
`sig500v2` and R7, so the gap cannot be assigned to the architecture. `cb500` is far lower than the rest (0.791) and the same GAT cannot be compared
with our GNN on the vectors at equal budget: our GNN on the same CodeBERT vectors with the whole graph (path 2, seed 0 only) reaches
0.961, dAUC -0.169 [-0.255, -0.095] against `cb500`, of which the cap is part. Path 2 is 0.025 below R7 (-0.025 [-0.059, -0.005]),
macro-F1 level (+0.003 [-0.042, +0.066]), so the frozen vectors do not beat token counts.

**The v2 build and the label leak.** `sig500v2` is a rebuild of pipeline B with Yanping's streaming `preprocess_full` (main c7899f0),
which includes the two listings `sig500` had to leave out (peak 20.2 GB for each of them in the first build). It is not better:
`sig500` -> `sig500v2` dAUC -0.001 [-0.011, +0.009], dmacro-F1 -0.009 [-0.039, +0.029] on the 1,808 binaries both have (3 seeds). Its
listing-to-CFG match (section 12; `match_acceptance --skip-rebuild` on both finished builds, 2026-10-20) is also label-correlated, and
more so than the first build's. The AUC of `1 - match rate` as a ransomware score is 0.543 [0.461, 0.635] for the first build
(1,808 ids) and 0.747 [0.650, 0.827] for the second (all 1,810 ids), over 712 clusters each; the share of binaries with under 90% of
their selected instructions matched is 21.0% of ransomware (238 of 1,132) and 3.4% of goodware (23 of 676) in the first build, and 32.9%
(373 of 1,134) against 7.0% (47 of 676) in the second. The `endp` reset of section 11 is the added leak (section 12.4). Whether a trained
model uses this is not measured here: the AUC of `sig500v2` is not higher (0.948 against 0.953), but its operating point moves toward
recall and false alarms (precision 0.881 against 0.962, FP 145 against 39).

These runs pin the teammate code at `origin/gnn` 331ce0b (A and the first B build) and main c7899f0 (B v2). Since then `origin/gnn`
22d1230 lets `train_cfg.py` load `preprocess_full.py` subgraphs with edge attributes, `origin/yanping` 38c38e5 adds single-instruction
tokenization to `tokenize_asm_gz.py`, and the new `origin/TokenSeqModel` (Janice) adds Model 1 (`src/models/token_sequence.py`) in a
`src/` layout. That branch also deletes `CNN-ViT/`, the old `CT_GAT/` files and most of `meetings/`; check before it is merged
anywhere. Model 1 itself is run through `token_sequence/` on our folds (`docs/PLAN_GNN.md`, "Final results", section 7).

## 7. Suggested fixes, in order of impact

1. Hold out validation by family (or reuse a fixed list of validation families) in both trainers.
2. Save per-sample test probabilities (`sample_id, label, prob`) and report AUC.
3. A: build CodeBERT text from operand categories, not raw operands.
4. B: freeze the API vocabulary on training samples (`--split train` first, then `--freeze-vocab` for test); seed or
   train the embedding; turn the size caps off or apply the same exclusions to every model.
5. B: investigate the alignment gaps before trusting B's numbers; log the per-sample match rate.
   Also stream the listing: `read_text_lines` (`preprocess_full.py` line 302) does `handle.read().splitlines()`, which
   holds the whole decompressed listing in memory. Two Mendeley listings are 341 MB compressed but 23 GB and 890 million
   lines uncompressed, 99.9% of them `db`/`dd` data lines for a payload of a few hundred MB, while their code is only
   3.2k-3.7k blocks; they need more than 20 GB of RAM and were excluded from B with that reason. Iterating the gzip handle
   line by line and keeping only instruction lines fixes it; reading `jsonl.gz` alone (section 9) avoids it.
6. A: make `--limit` take a balanced prefix; fail or list per-sample errors; check graphs against the split.

(Item 4 and the alignment failure of item 5 were fixed in main PR #10, section 13.1; the listing is streamed since main c7899f0.)

## 8. The team diagram (2026-10-19) against what exists

| Diagram box | Team code | Ours (`3-block-level-counting`) | Evidence and suggestion |
|---|---|---|---|
| `jsonl.gz` → `load_cfg()` | `preprocess_full.py` | `pipeline/s02_build_cfg_cache` | Same input. Keep. |
| `asm.gz` → `parse_asm_instructions()` → match RVA and mnemonic | `tokenize_asm_gz.py`, `preprocess_full.py` | none: every token comes from `jsonl.gz` | **Drop.** The match fails on 20% of ransomware but 1% of goodware (section 4), which makes missing features a label signal. The team's own `ida_feature_readers.py` produces the same operand-role triplet from `jsonl.gz` alone (built for 6,517 of 6,523 exports), and its side arrays aligned with our cache with zero failures. |
| Full CFG | built per sample | the cache (`cfg_edges`, edge kinds, back edges, call edges) | Keep. |
| Randomly select subgraphs | `walk` / `signal` at a block budget | `s04_select_subgraph` (`cap`, `lcc`; whole graph by default) | **Use the whole graph, deterministic.** Every budget we tried lost AUC against the whole graph (cap 20k -0.011, cap 5k -0.016, lcc 20k -0.017, lcc 5k -0.011; none non-inferior), and at 500 blocks both A arms lost more. Random selection also makes a test prediction depend on a seed. Keep a deterministic cap only as a memory guard. |
| `tokenize_selected_blocks()` | five channels: opcode, two operand roles, API id, behaviour | `s03_tokenizer`: `mn`, `sig`, `beh`, `api` | Settled: under the pre-registered 3-seed test the team's operand-role vocabulary tied ours (section "Vocabulary decision", `PLAN_GNN.md`). Either works; fit the API vocabulary on training binaries only. |
| Model 1, token sequence | `origin/TokenSeqModel` `src/models/token_sequence.py` (new) | no sequence model on this branch; TF-IDF over the same tokens is the bag-of-tokens reference | Add to the same folds and report it next to TF-IDF: on this data the earlier graph-transformer sequence arms (mF1 0.906-0.916) stayed below TF-IDF (0.939). |
| Model 2, block features + CFG | `FullBlockGAT` / `BlockGAT` | `s05_block_counting` + `s06`-`s08` GNN (R7) | Both run (section 6). |
| Model 3, block embeddings + CFG | 64-d pooled random embedding (B); CodeBERT (A) | `transformer_paths` path 2: any frozen block encoder, now CodeBERT on register-free text | Running (paths 1 and 2). A trained block encoder is the open slot. |
| (missing) baselines | none | TF-IDF, header-imports LR, majority and x86-rule floors | **Add** to the diagram: every model is read against them. |
| (missing) evaluation | one fixed split, random validation | 5-fold family holdout, family-held-out validation, paired bootstrap | **Add** as the box all three models feed. |

(Later results, section 13: after main PR #10 the listing match no longer leaks the label (AUC of `1 - match rate` 0.502); the team's
Models 2 and 3 and Janice's five-channel Model 1 have run on our folds.)

## 9. Target architecture (adopted 2026-10-19)

```mermaid
flowchart TD
    A["IDA export jsonl.gz (no asm.gz)"] --> B["CFG cache: blocks, typed edges, call edges, tokens (s02)"]
    B --> C["Whole graph, deterministic (s04; 500k-block cap only as a memory guard)"]
    C --> D["Tokenizer, vocab 'ours', fitted on training binaries (s03)"]
    D --> M1["Model 1: token sequence (Janice's model, through an adapter)"]
    D --> M2["Model 2: block counts + CFG (HierGNN)"]
    D --> ENC["Frozen block encoder (transformer_paths): CodeBERT now, our own pretrained encoder later"]
    ENC --> M3["Model 3: block embeddings + CFG (same HierGNN, --block-input vec)"]
    BASE["Baselines: header-imports LR, TF-IDF, majority and x86-rule floors"] --> EV
    M1 --> EV
    M2 --> EV
    M3 --> EV
    EV["Evaluation: x86 5-fold family holdout and v3 all-arch; AUC, PR-AUC, accuracy, precision, recall, macro-F1, confusion matrix; paired cluster bootstrap over 3 seeds x 5 folds"]
```

Removed from the team diagram, with the evidence in section 8: the `asm.gz` input and the RVA/mnemonic match (a
label-correlated alignment failure, and one listing needs more than 20 GB of RAM), and random subgraph selection (every
budget lost to the whole graph). Added: the baselines and evaluation boxes.

Why the baselines box matters: the header-imports LR, which reads only the PE import table, ties our typed GNN on x86
(0.985 vs 0.986 AUC) and on v3 (0.966-0.970 vs 0.965 AUC, paired difference inside the noise), and on v3 it catches more
x64 ransomware (0.91-0.94 recall vs 0.83). The open question is whether any code model adds anything beyond imports; the
`noapi` ablation (block tokens without `dll!function` names, `gates.py --stage noapi`) tests it.

## 10. Model 1 adapter contract (for `origin/TokenSeqModel`)

To put Model 1 in the same tables, it needs to read our folds and write our prediction format; nothing else in
`src/models/token_sequence.py` has to change.

- Input: the block export `RUNS/export/ours_cap0` (`transformer_paths/EXPORT_SCHEMA.md`): per fold, `fold{k}/manifest.json`
  gives train / val / test sample ids (validation already held out by family), and the store gives each binary's blocks
  as token ids in a per-fold vocabulary fitted on that fold's training binaries. Do not build a vocabulary from test
  binaries.
- Training: train on `train`, choose the epoch and the decision threshold on `val` only, never look at `test`.
- Output, per fold and seed, a run directory written with `experiments/team_pipelines/common.write_run`
  (`predictions.csv`, `predictions_auc.csv` with `sample_id, split, label, prob, pred, arch, family, fold`, and
  `metrics.json`), so `stats.py` and `gates.py` pair it with every other model on identical test ids.
- Report seeds 0 1 2 over folds 0-4. The earlier sequence transformers on this data reached macro-F1 0.906-0.916 against
  TF-IDF's 0.939, so compare Model 1 with TF-IDF first.

## 11. Streaming version (main c7899f0)

Note for Yanping on `CT_GAT/pipeline/s02_preprocess/preprocess_full.py` and `tokenize_asm_gz.py` at main c7899f0. We pinned both files by sha256 as a second source next to 331ce0b, built 60 samples again with the same vocabulary and embedder, and rebuilt pipeline B on all 1,810 binaries with it (`build_b.py --source main-c7899f0`). Nothing in her files was edited.

**What the change fixed.** The two listings that needed more than 20 GB of RAM and stopped the first build (`bbbf38de4f40` and `e8a0e80dfc52`, 341 MB compressed, 23 GB and about 890 million lines uncompressed each) now build. Peak resident memory, polled over the process tree, was 0.62 GB for the first (3,999.7 s wall) and 0.60 GB for the second (4,028.6 s wall), against 20.2 GB and a `MemoryError` before. The time is two passes over 890 million lines; ordinary listings are no slower than before (an 18 MB listing took 209 s against 231 s).

**Does it give the same samples?** Not quite. We took 60 binaries of the first build (30 ransomware, 30 goodware, including the 10 lowest-match ones, which all have a match rate of exactly 0), rebuilt them with her version with the first build's vocabulary and embedder frozen, and compared the saved tensors. In all 60 the selected blocks, the edges and the edge attributes are identical (the sampler is untouched). 25 samples are identical in every array; 35 differ, and in each of the 35 her version matches fewer instructions and in none more: 4,268 of their 137,857 instructions (3.1%), up to 12 points of match rate in one sample (`4708750c9a6f`, 1.0000 to 0.8799). The differing arrays are `instruction_matched`, the opcode, operand and behaviour columns of `instruction_ids` (an unmatched instruction gets the unknown-opcode tokens), `block_features` (on average 9 of 500 blocks), and for 15 samples the api column (46 instructions). Mean match rate over the 60: ransomware 0.630 to 0.604, goodware 0.956 to 0.950.

The cause is one thing. `tokenize_asm_gz._note_location` (lines 547-560) takes the label address from the first token of any listing line, so the line `sub_401E80      endp` sets the address back to the start of the function. `align_sampled_blocks` closes the open span at that moment with `end_rva` equal to the function start, which is smaller than the span's own start; `bisect_left` (line 859) then returns an empty slice and the instructions of that span are never paired. Every function that has an inner label loses the code after its last label, which is usually the epilogue. Two examples, both in the listing text and both matched by the first version:

- `4708750c9a6f`, block (26, 14): `loc_4020F5:` followed by `mov eax, [ebp+var_14]`, `mov esp, ebp`, `pop ebp`, `retn`, then `sub_401E80 endp`. First version: 4 instructions matched. Her version: 4 unmatched.
- `1c209b731ecb`, block (13, 429): `loc_4030EE:` followed by `pop edi`, `pop esi`, `pop ebx`, `leave`, `retn 4`, then `sub_4015A0 endp`. First version: 5 matched. Her version: 5 unmatched.

Her version is less correct here: those lines are in the listing at the right address, and the first version pairs them. To check the cause we ran her module with `_note_location` wrapped to ignore `endp` lines (a wrapper in a scratch script, her files untouched): the 35 differing samples then come out identical to the first version in `instruction_matched`, `instruction_ids` and `block_features`. A fix is to leave the address alone on `endp` and `ends` lines, or to ignore a span whose end is not above its start, plus a test with a function that has an inner label before its `endp`. On the first 550 binaries the full rebuild had finished when this was written, the share with less than 90% of the selected instructions matched is 33.8% for ransomware (23.2% in the first build) and 7.2% for goodware (3.3%); the share below 50% is unchanged (19.7% and 1.7%). `experiments/team_pipelines/match_rates.py` prints these tables for any build.

**What is still in her `main()` and what the change added.** Line numbers are in `preprocess_full.py` unless a file is named.

- Vocabulary on train and test: `--split` defaults to `all` (1918) and `--freeze-vocab` is off by default (1956-1959), so the vocabulary grows over every sample (`full_run`, 1970).
- Embedder unseeded: `FiveChannelEmbedder` is created at 1972 and nothing in either file calls `torch.manual_seed`; `--seed` reaches only the sampler (2014), and `expand_api` draws new rows when the vocabulary grows.
- Size caps: the defaults are at 80-82 and the check at 1995-2004, but the caps no longer fire. `tokenize_asm_gz._selected_rows` (1243-1250) stopped copying `asm_bytes`, `num_blocks` and `num_instructions` into the row, so lines 1987-1989 read 0 and `--max-asm-bytes`, `--max-cfg-blocks` and `--max-cfg-instructions` do nothing; they were active in 331ce0b. The two big listings are therefore no longer skipped, by accident.
- CFG path from the listing path: `_cfg_path_for_asm` (1906) is used at 2006; the Mendeley bundle keeps CFGs and listings in different folders, so a missing CFG raises at 2009.
- Failures skipped: `except Exception` (2032-2035) logs, counts `failed` and goes on, and the process exits normally, so the output folder can lack samples without a failing status.
- Vocabulary and embedder saved after every sample: 2027-2028, also for `--limit` and `--sample-id` runs; line 2038 only decides whether to log it.
- `can_resume` (1974) is true when `channel_embedder.pt` exists; an existing `.pt` is then skipped (1990) whatever `--sample`, `--seed`, `--num-blocks` or vocabulary produced it, and the embedder is loaded from the folder.
- `signal` is not connected: `sample_signal_subgraph` (1044) returns the union of the highest-ranked functions and `connected_keys` (1118) is never called, while the module docstring (line 1) says one connected subgraph.
- `_selected_rows` still reads `row["collection"]` (`tokenize_asm_gz.py` 1247), which the bundle CSVs lack.
- New, two full passes over the listing for every sample: `align_sampled_blocks` (1809-1814) and `tokenize_selected_blocks` (1817-1823), whose `collect_ida_imports` (`tokenize_asm_gz.py` 690-720) scans every line again with a regular expression. The memory is fixed but the decompress-and-scan time is doubled; reading the `extrn` lines inside the first pass would halve it.
- New, the `endp` regression above (`_note_location`, and 858-859 and 887-895 in `align_sampled_blocks`). It also makes `mismatch_report` unreliable: the span that follows `endp` is re-anchored at the function start with no code lines, so every selected instruction of the function is reported as `no_asm_line_in_address_span` (884), matched or not. For `4708750c9a6f` `missing_asm_count` is 5,763 of 6,079 selected instructions while 730 were really lost, and for `1c209b731ecb` it is 2,861 of 2,861 while 39 were lost; the lost ones carry the same reason as the matched ones.
- New, `bisect` and unsorted data: `pairable_rvas` (834-835) and `selected_rvas` (836) are sorted, so the bisect calls are valid; what is not monotone is the sequence of label addresses in the listing, and the code does not check it. An end address below the start gives an empty slice without a message (859-860); the same address seen twice processes its function twice (`placed`, 865, only feeds the final report).
- New, the `selected_rvas` filter (896) drops a line only when no selected instruction lies at or above the span's address, so lines of all spans before the last selected instruction are buffered until the next label and discarded at 861-863. The memory is bounded by the largest code span, not by the selected ones; a function with millions of code lines and no inner label would still be held whole.
- New, a selected block whose span is never opened (no label at or below its address, or an address the listing does not have) is left unmatched and reported as `rva_before_any_asm_label` (901-905), the only case that gets that reason. The report keeps 32 examples per kind, so only its counts say how many instructions are affected, and those are inflated by the `endp` case above.
- New, line splitting: `splitlines()` is replaced by iterating the text handle, so form feeds and the other Unicode line boundaries no longer split a line; the 60-sample comparison shows no effect from this.

(The `endp` reset and most of the issues above were addressed in main PR #10, section 13.1.)

## 12. Why the match fails, and how to check a fix

Note for Yanping on the listing-to-CFG match in pipeline B. We measured it on the first build (`RUNS/team/b_sig500`, `preprocess_full` of 331ce0b, 1,808 of 1,810 binaries) and replayed 80 single samples through her pinned functions (`load_cfg`, `parse_asm_instructions`, `align_cfg_to_asm`, and `align_sampled_blocks` for the second source), with the example limit of her mismatch report lifted in memory. None of her files was edited and no new listing or CFG parser was written. The tools are `experiments/team_pipelines/match_diagnosis.py` and `match_acceptance.py` (README, "Match diagnosis and acceptance"). All 80 replays reproduce the build's own matched count exactly.

**The finding in one line.** The listings do not contain the code of the functions IDA marked as library functions: each is one line `; [000002BA BYTES: COLLAPSED FUNCTION initialize_pointers(void)]`, and the CFG has all of them. Of the instructions lost in the 80 replayed samples, 99.8% (ransomware) and 96.7% (goodware) belong to such functions; mnemonic disagreements and the other causes below are 0.2% of the ransomware loss and 3.3% of the goodware loss. The label gap exists because ransomware samples are mostly statically linked library code (C runtime, Go runtime, MFC) and the `signal` sampler ranks whole functions by imports and calls, which, in the worst samples, selects functions that are all library functions (this is read off the replayed samples, not measured over the whole build). The earlier hypothesis in section 4, two IDA runs that disagree on packed code, is not what the data show.

Evidence that the listing is the missing side, from the CFG records themselves. Every function record in the `.jsonl.gz` carries `library` and `thunk` flags. For `05e06709523f` (ryuk, 533 functions) 438 functions are `library` and none of them has a `proc` line in the listing, while 94 of the 95 non-library functions do (the 95th is a thunk with a COLLAPSED marker); 371 of the 438 library functions have a COLLAPSED marker under the same name, and the other 67 are neither listed nor matched to a marker by name. Names fail to match when the marker carries the demangled form, as in `48976d7bf38c`, where the CFG's `?initialize_pointers@@YAHXZ` is `initialize_pointers(void)` in the listing; we did not check these 67 one by one. Over the 80 replayed samples the CFG holds 75,015 functions; 29,090 have a label at their entry, the listing has 44,035 collapsed markers, and 1,845 procedures are listed under another spelling of their name (IDA turns `.` into `_`: the Go function `unicode.init` is `unicode_init`). 29,090 + 44,035 + 1,845 = 74,970 of 75,015.

### 12.1 Match rate by label, family and size (1,808 binaries of the first build)

| label | binaries | mean match | below 50% | below 90% | exactly 0 |
|---|---|---|---|---|---|
| ransomware | 1132 | 0.855 | 187 (16.5%) | 238 (21.0%) | 1.7% |
| goodware | 676 | 0.978 | 7 (1.0%) | 23 (3.4%) | 0.0% |

The AUC of `1 - match rate` as a ransomware score is 0.543, with a 95% interval of [0.461, 0.635] from 1,000 resamples of the 712 clusters (one per ransomware family, one per goodware binary, as `evaluation/stats.py` draws them). The point estimate is low because most binaries of both labels sit at or near 1.0 (63% of the ransomware and 67% of the goodware binaries match completely), so the rank statistic is decided in the tails and by ties; the interval is wide and its upper end is above 0.6. The label signal sits in the tail (the 21% against 3.4%), which a rank statistic over all binaries understates. The ten worst families:

| family | binaries | mean match | below 50% | below 90% |
|---|---|---|---|---|
| lorenz | 15 | 0.295 | 100.0% | 100.0% |
| stop | 35 | 0.297 | 82.9% | 82.9% |
| pysa | 38 | 0.314 | 100.0% | 100.0% |
| playcrypt | 43 | 0.574 | 44.2% | 76.7% |
| blackbasta | 26 | 0.586 | 53.8% | 57.7% |
| gandcrab | 49 | 0.599 | 42.9% | 46.9% |
| nefilim | 25 | 0.658 | 36.0% | 44.0% |
| ryuk | 33 | 0.666 | 27.3% | 54.5% |
| mountlocker | 5 | 0.821 | 20.0% | 20.0% |
| karma | 13 | 0.828 | 23.1% | 23.1% |

Fourteen of the 36 ransomware families match at 99% or better on average (netwalker, blackcat, blackmatter, bluesky, darkside, hive, lockbit and others at 1.000), so the effect is a property of the binaries, not of ransomware as such. The goodware pool `goodware` (562) has 3.7% below 90%, the held-out `goodware_test` (114) 1.8%; the full per-family table is in the JSON of `match_diagnosis`.

Lost share by function length (selected instructions of one function, from the saved tensors, all 1,808 binaries):

| label | length | functions | selected | lost | lost share | fully lost functions |
|---|---|---|---|---|---|---|
| ransomware | 1-9 | 5927 | 9542 | 3773 | 39.5% | 3597 |
| ransomware | 10-49 | 2669 | 73160 | 3564 | 4.9% | 137 |
| ransomware | 50-199 | 8053 | 947354 | 118465 | 12.5% | 996 |
| ransomware | 200-999 | 6111 | 2324397 | 235015 | 10.1% | 523 |
| ransomware | 1000+ | 842 | 2458802 | 126542 | 5.1% | 76 |
| goodware | 1-9 | 1513 | 3727 | 1389 | 37.3% | 923 |
| goodware | 10-49 | 2394 | 61152 | 4000 | 6.5% | 210 |
| goodware | 50-199 | 3934 | 424878 | 13950 | 3.3% | 84 |
| goodware | 200-999 | 1921 | 732057 | 9015 | 1.2% | 17 |
| goodware | 1000+ | 321 | 646549 | 2232 | 0.3% | 2 |

Functions of 1 to 9 instructions are lost most often in both labels (37-40%); the label gap is in the long functions, where ransomware loses 5-12% against 0.3-3%. In the replay, where the label spans (CFG instructions between two listing labels) are known:

| span length | spans | selected | lost | lost share |
|---|---|---|---|---|
| 1-9 | 5123 | 19872 | 139 | 0.7% |
| 10-49 | 2605 | 50731 | 955 | 1.9% |
| 50-199 | 313 | 24231 | 3100 | 12.8% |
| 200-999 | 137 | 33298 | 17625 | 52.9% |
| 1000+ | 143 | 96439 | 91553 | 94.9% |

A span of a few instructions is a function that has a label and a listing; a span of 200 or more is a label followed by collapsed functions that have none, so the CFG rows of all of them are bucketed under the last label before them and nothing is left to pair with.

### 12.2 What the unmatched instructions are

The gap reports in the manifests (counts only; the report keeps 32 examples per kind) summed over the first build. Scope: for 331ce0b the alignment covers the whole CFG, so the columns count every CFG instruction, not only the selected ones; for the second source they count selected instructions and are inflated by the `endp` reset (section 11).

| label | manifests | selected instr. | CFG instr. without listing line | listing lines without CFG instr. | duplicate RVAs | address problems |
|---|---|---|---|---|---|---|
| ransomware | 1132 | 5813255 | 18876058 | 3784738 | 185915 | 1 |
| goodware | 676 | 1868363 | 1188764 | 592594 | 21441 | 1 |

Reasons of the retained examples: ransomware `collapsed_function` 16,785, `no_asm_line_in_address_span` 7,424, `rva_before_any_asm_label` 804; goodware `collapsed_function` 12,976, `no_asm_line_in_address_span` 2,981. Her `collapsed_function` test is a substring match of the CFG name in the marker, so it misses every function whose marker is demangled; those fall under `no_asm_line_in_address_span`.

The replay sorts every unmatched selected instruction of 80 samples into one cause. The samples are the 30 lowest-match samples, the 10 lowest-match goodware samples, and 20 random samples per label that lose some but not all (seed 0). Counts are instructions.

| cause | ransomware | goodware |
|---|---|---|
| (i) listing label missing below the RVA | 0 | 1 |
| (ii) lost after a mnemonic disagreement | 0 | 23 |
| (iii) function without listing lines | 100024 | 13289 |
| (iv) duplicate RVA (skipped on purpose by the pairing) | 217 | 408 |
| (v) a label kind her address rule does not read | 21 | 15 |
| (vi) `endp` reset | 0 | 0 |
| (vii) other | 0 | 0 |
| selected instructions replayed | 157795 | 67402 |
| unmatched | 100262 | 13736 |

Cause (iii) splits into the function being a COLLAPSED FUNCTION under the CFG's own name (ransomware 69,190, goodware 11,874) and the function having no label at its entry while its marker could not be matched by name (ransomware 30,834, goodware 1,415); the second group is the demangled and sanitised names above, counted by the label census (the unlabelled functions equal the collapsed markers to within 0.1%). Code labels without an address (`def_`, `nullsub_`, `j_` and other names) number 20,093 `named`, 141 `def_` and 86 `j_` over the 80 listings; only 36 instructions (cause v) were lost after a disagreement in a span that holds one. No listing line without a label address occurred (`lines_without_label` 0 in all 80). (vi) is zero because these are first-source builds; section 11 and 12.4 cover the second source.

Mnemonic disagreements: the first disagreement of a label span, as (listing, CFG) pair, with the instructions lost after it. There are five in the 80 samples, and none is a prefix, a `retn`/`ret` spelling, a filler or data printed as code.

| pair (listing, CFG) | spans | lost |
|---|---|---|
| mov, retn | 3 | 21 |
| lea, push | 1 | 16 |
| mov, jmp | 1 | 14 |
| call, push | 1 | 7 |
| jmp, call | 1 | 1 |

### 12.3 The ten worst samples

Ten lowest-match samples of distinct (family, selected instructions) pairs; several others are copies of the same binary (seven ryuk binaries have the same 2,499 selected instructions and all lose them). `selected`, `matched` and the listing census are from the replay. Each excerpt is at most 8 lines of her listing file for the first function with an unmatched selected instruction that has a marker.

| sample | family | match | selected | CFG functions | with a label | collapsed markers |
|---|---|---|---|---|---|---|
| `9e6be0a3bf10` | nefilim | 0.000 | 4587 | 2584 | 1 | 2318 |
| `148845af1fc1` | nefilim | 0.000 | 4135 | 2539 | 1 | 2245 |
| `f0aec57001a1` | phobos | 0.000 | 3065 | 3639 | 413 | 3225 |
| `05e06709523f` | ryuk | 0.000 | 2839 | 533 | 94 | 439 |
| `2ec5256a7edb` | ryuk | 0.000 | 2499 | 421 | 69 | 352 |
| `d236d64b7bf9` | conti | 0.000 | 2402 | 1438 | 303 | 1136 |
| `48976d7bf38c` | blackbasta | 0.000 | 2271 | 2151 | 866 | 1281 |
| `1fe9b489c25b` | revil | 0.049 | 2537 | 387 | 54 | 333 |
| `dc6b0e8c1e9c` | revil | 0.050 | 2463 | 391 | 54 | 337 |
| `54ac8cbce477` | goodware | 0.058 | 2213 | 173 | 13 | 160 |

`9e6be0a3bf10` (Go; the function is `unicode.init`, 4,587 selected instructions):

```
                align 10h
; [000000E1 BYTES: COLLAPSED FUNCTION unicode_isExcludingLatin]
                align 10h
; [0000551A BYTES: COLLAPSED FUNCTION unicode_init]
                align 10h
; [000001E5 BYTES: COLLAPSED FUNCTION reflect_makeMethodValue]
                align 10h
; [00000062 BYTES: COLLAPSED FUNCTION reflect_name_tagLen]
```

`148845af1fc1` (Go; the same function, 4,135 selected instructions):

```
                align 10h
; [000000CF BYTES: COLLAPSED FUNCTION unicode_isExcludingLatin]
                align 10h
; [00004EE8 BYTES: COLLAPSED FUNCTION unicode_init]
                align 10h
; [000001BC BYTES: COLLAPSED FUNCTION reflect_makeMethodValue]
                align 10h
; [00000065 BYTES: COLLAPSED FUNCTION reflect_name_tagLen]
```

`f0aec57001a1` (`@SysUtils@_16771` and its neighbours):

```
; [000000BC BYTES: COLLAPSED FUNCTION unknown_libname_421]
; [00000049 BYTES: COLLAPSED FUNCTION unknown_libname_422]
                align 4
; [00000661 BYTES: COLLAPSED FUNCTION unknown_libname_423]
                align 4
; [00000061 BYTES: COLLAPSED FUNCTION Graphics::CopyPalette(uint)]
                align 4
; [0000020C BYTES: COLLAPSED FUNCTION unknown_libname_424]
```

`05e06709523f` (`__input_l`, 5,786 bytes of C runtime):

```
                align 10h
; [0000005F BYTES: COLLAPSED FUNCTION _ValidateLocalCookies]
                align 10h
; [0000169A BYTES: COLLAPSED FUNCTION __input_l]
                align 4
jpt_35007A2C    dd offset loc_35007BA2  ; DATA XREF: __input_l+42C↑r
                dd offset loc_35007A33  ; jump table for switch statement
                dd offset loc_35007A54
```

`2ec5256a7edb` (`___security_init_cookie`):

```
; [00000028 BYTES: COLLAPSED FUNCTION ___scrt_uninitialize_crt]
; [0000003B BYTES: COLLAPSED FUNCTION __onexit]
; [00000015 BYTES: COLLAPSED FUNCTION _atexit]
; [0000009C BYTES: COLLAPSED FUNCTION ___security_init_cookie]

; =============== S U B R O U T I N E =======================================

```

`d236d64b7bf9` (MFC; `unknown_libname_56`, after the last listed function `sub_408CAA`):

```
sub_408CAA      endp

; [0000007E BYTES: COLLAPSED FUNCTION CWnd::PrepareForHelp(void)]
; [000005C2 BYTES: COLLAPSED FUNCTION unknown_libname_56]
                db 8Bh, 0FFh
jpt_408FBA      dd offset loc_40903D, offset loc_40904F, offset loc_40904F
                                        ; DATA XREF: unknown_libname_56+1B3↑r
                dd offset loc_40904F, offset loc_409069, offset loc_409085 ; jump table for switch statement
```

`48976d7bf38c` (the CFG name is the mangled `?initialize_pointers@@YAHXZ`; the marker carries the demangled name):

```
; [00000027 BYTES: COLLAPSED FUNCTION __Xtime_get_ticks]
; [00000055 BYTES: COLLAPSED FUNCTION _xtime_get]
; [0000001F BYTES: COLLAPSED FUNCTION __Cnd_broadcast]
; [000002BA BYTES: COLLAPSED FUNCTION initialize_pointers(void)]
; [00000026 BYTES: COLLAPSED FUNCTION ___crtGetSystemTimePreciseAsFileTime]
; [0000001A BYTES: COLLAPSED FUNCTION unknown_libname_34]
; [00000063 BYTES: COLLAPSED FUNCTION _Init_atexit::~_Init_atexit(void)]
                db 5 dup(0CCh)
```

`1fe9b489c25b` (`___scrt_fastfail`):

```
                retn
sub_4017E5      endp

; [0000011B BYTES: COLLAPSED FUNCTION ___scrt_fastfail]
; [00000033 BYTES: COLLAPSED FUNCTION ___scrt_get_show_window_mode]
; [00000044 BYTES: COLLAPSED FUNCTION ___scrt_is_managed_app]

; =============== S U B R O U T I N E =======================================
```

`dc6b0e8c1e9c` (revil, `___scrt_fastfail` again, another build of the same program):

```
                retn
sub_401846      endp

; [0000011A BYTES: COLLAPSED FUNCTION ___scrt_fastfail]
; [00000031 BYTES: COLLAPSED FUNCTION ___scrt_get_show_window_mode]
; [00000005 BYTES: COLLAPSED FUNCTION j_UserMathErrorFunction]
; [00000043 BYTES: COLLAPSED FUNCTION ___scrt_is_managed_app]
```

`54ac8cbce477` (goodware, `___tmainCRTStartup`; 13 of 173 functions have a listing):

```
                pop     ecx
                pop     ebp
                retn
; [0000017E BYTES: COLLAPSED FUNCTION ___tmainCRTStartup]
; [0000000A BYTES: COLLAPSED FUNCTION start]
; [00000042 BYTES: COLLAPSED FUNCTION __CxxUnhandledExceptionFilter(_EXCEPTION_POINTERS *)]

; =============== S U B R O U T I N E =======================================
```

### 12.4 The second source (main c7899f0) on the partial rebuild

The rebuild with her streaming `preprocess_full` is not finished: the checkpoint of `RUNS/team/b_sig500_v2` held 1,350 of 1,810 ids when this was measured and there is no `build_summary.json` yet, so criterion 1 cannot pass and everything below is for those 1,350 ids, next to the same 1,350 ids of the first build. To redo it on the finished build, run the two commands of 12.6 on `RUNS/team/b_sig500_v2` once `build_summary.json` reports 1,810 processed.

| on the same 1,350 ids | first source (331ce0b) | second source (c7899f0) |
|---|---|---|
| ransomware mean match | 0.850 | 0.809 |
| ransomware below 90% | 186 of 853 (21.8%) | 289 of 853 (33.9%) |
| ransomware below 50% | 145 of 853 (17.0%) | 147 of 853 (17.2%) |
| goodware mean match | 0.977 | 0.964 |
| goodware below 90% | 18 of 497 (3.6%) | 36 of 497 (7.2%) |
| goodware below 50% | 6 of 497 (1.2%) | 6 of 497 (1.2%) |
| AUC of 1 - match rate | 0.547, interval [0.462, 0.634] | 0.754, interval [0.666, 0.832] |

The `endp` reset of section 11 shows up as a leak of its own: it moves instructions from "matched" to "unmatched" mostly in ransomware, which raises the AUC from 0.547 to 0.754 and the ransomware below-90% share by 12 points, while the below-50% tail is unchanged (the collapsed functions, which it does not touch). The replay of 50 samples of the second build (15 lowest-match, 5 lowest-match goodware, 15 random lossy samples per label; all 50 reproduce the build's matched count) classifies cause (vi) as instructions that the first source's alignment pairs and the second source's does not:

| cause | ransomware | goodware |
|---|---|---|
| (i) listing label missing below the RVA | 0 | 0 |
| (ii) lost after a mnemonic disagreement | 0 | 44 |
| (iii) function without listing lines | 46672 | 9207 |
| (iv) duplicate RVA | 152 | 448 |
| (v) a label kind her address rule does not read | 0 | 0 |
| (vi) `endp` reset | 7428 | 1037 |
| (vii) other | 0 | 1 |
| selected instructions replayed | 126147 | 52538 |
| unmatched | 54252 | 10737 |

So in the second build the collapsed functions still dominate the loss (86% of the ransomware and 86% of the goodware instructions lost), and the `endp` reset adds 14% and 10% on top. Both have to go: option 1 of 12.5 removes them together, option 3(a) removes only the second.

### 12.5 Fix options, ranked

1. **Take the per-instruction fields from the `.jsonl.gz`** and make the listing optional. Each instruction record has `rva`, `size`, `mnemonic`, the structured `operands` (`category` register / phrase / immediate / code_target, `register`, `immediate`, `target_rva`), `is_call`, `import_ids`, `call_resolution`, `code_refs` and `data_refs`; each function record has `library` and `thunk`. The opcode and operand-class channels, the call and import channels and the API channel (through `import_ids` and `call_resolution`) can be built from these without the listing. This removes causes (i) to (vii) for every instruction in one step, makes the match rate 100% by construction, needs no 23 GB listing in memory, and takes away the ransomware-only tail that makes `1 - match rate` a label feature. What it costs: the operand classes come from the structured operand (`MEM_STACK_REF` needs the base register, `MEM_GLOBAL_REF` the absence of one), so the token ids differ slightly from the ones built from listing text; sig500 must be rerun and compared with sig500v3. Where the listing is kept (comments with API names for `call eax`), pairing becomes optional enrichment, and its match rate should be logged and not used as a gate.
2. **Re-export the listings so that library functions are expanded and every line carries its address.** With the bodies of the collapsed functions in the file, causes (iii) and (iv) disappear and pairing could use addresses instead of order. It depends on a new export of the listings, and the two largest listings (23 GB each) get larger when expanded; the CFG already has everything that is missing, so this is a longer route to the same result as 1.
3. **Harden the pairing**, which is worth doing whichever of 1 or 2 is chosen, but fixes little alone. (a) The `endp` reset: leave the label address alone on `endp` and `ends` lines, or ignore a span whose end is not above its start (section 11). (b) Treat a function without listing lines as a known case: do not pair it, give its instructions the CFG-derived tokens of option 1 and not the unknown-opcode tokens, and do not count it as unmatched. (c) Read `library` from the function record to decide this, not the substring test on the marker name (which misses demangled and sanitised names). (d) Match procedure names after the same sanitising IDA applies (`.` and other non-identifier characters become `_`), so `unicode.init` finds `unicode_init proc near`. (e) After a mnemonic disagreement, re-anchor at the next listing label and keep pairing, instead of dropping the rest of the span. (f) Make the build fail a sample, or at least print it, when the match rate is under 90%, and count duplicate RVAs separately from losses (they are skipped on purpose, 408 goodware instructions in the replay).

### 12.6 How to check a fix

Run `match_acceptance.py` on her rebuilt output (a directory with the `<id>.pt`, `<id>.manifest.json`, `checkpoint.json`, `settings.json` and `build_summary.json` that `build_b.py` writes, or the same layout from her own script):

```
python -m experiments.team_pipelines.match_acceptance --build <her new dir>
python -m experiments.team_pipelines.match_diagnosis  --build <her new dir> --cache gaps.json --json diag.json
```

It prints PASS or FAIL for five criteria and exits with status 0 only if all pass: (1) all 1,810 kfold-x86 ids built, none excluded; (2) the share of binaries with under 90% of their selected instructions matched within 2 points between ransomware and goodware; (3) the share under 50% below 1% for both labels; (4) the AUC of `1 - match rate` below 0.55; (5) a deterministic rebuild of 5 sampled ids, tensor for tensor (`torch.equal`), starting from the build's own vocabulary and embedder (`build_b --frozen-from`), because a fresh vocabulary numbers API names differently. The peak-RSS and sig500v3 criteria of the plan are not tested by the script; `measure_rss.py` and `gates.py --stage team` do those. With option 1 criteria 2 to 4 pass by construction and the script then checks mostly criteria 1 and 5. A partial build is evaluated on the ids it holds and marked partial.

Current verdicts: first build, FAIL (criteria 1, 2 and 3 fail; criterion 4 passes at 0.543, interval [0.461, 0.635]; criterion 5 passes, 5 of 5 sampled ids rebuild identically with the frozen vocabulary). Second build, partial (1,350 of 1,810 ids): FAIL (1 of 5 pass): criterion 1 fails (460 ids missing), criterion 2 fails (33.9% against 7.2%, a gap of 26.6 points), criterion 3 fails (17.2% and 1.2%), criterion 4 fails (0.754, interval [0.666, 0.832]), criterion 5 passes (5 of 5 sampled ids rebuild identically with the frozen vocabulary). Both are expected to fail until the fix lands.

Update 2026-10-20, second build finished (1,810 of 1,810 ids, nothing excluded; `match_acceptance --skip-rebuild`, so criterion 5 is not run):
criterion 1 passes; criterion 2 fails (under 90%: ransomware 32.9%, 373 of 1,134, goodware 7.0%, 47 of 676, a gap of 25.9 points);
criterion 3 fails (under 50%: ransomware 16.8%, 190 of 1,134, goodware 1.0%, 7 of 676); criterion 4 fails (AUC of `1 - match rate`
0.747, interval [0.650, 0.827] over 712 clusters). The first build, re-run the same way, gives 0.543 [0.461, 0.635]. Section 6 uses
both numbers.

(The fix landed in main PR #10; the rebuilt output passes all five criteria, section 13.1.)

## 13. Fixed preprocessing, Models 2 and 3, and the cap (2026-10-21)

Setup as in section 6: kfold x86, 1,810 test binaries in 712 clusters, every team arm trained on our folds and family-held-out
validation split, AUC checkpoint at its validation threshold. A table value is the mean over seeds 0 1 2 of the five-fold pooled
metric; a difference is second minus first with the 95% paired cluster bootstrap interval (2,000 resamples). "Beats" is the
pre-registered rule: dAUC >= 0.01 and dmacro-F1 >= 0.02, both intervals above zero. Rows marked seed 0 are a direction, not the test.
Sources: `RUNS/gates_team.md` and `RUNS/gates_model1_variants.md` (written by `gates.py`), the acceptance files
`RUNS/team/{v3_full,whole,v3_subset,v3b_subset}_acceptance.json` (`match_acceptance.py`), `match_rates.py` on each build, and main's
PRs #10, #12 and #13. Every paired difference below was rerun with `python -m evaluation.stats --selection auc` and agrees with the gates
files; the `-thoof` pair was rerun with main's `s04_evaluation.stats`.

### 13.1 What PR #10 changed, and the acceptance results

Main PR #10 (`s02-endp-library-fix`, merged 2026-10-21 as 8a971b7) changes `s02_preprocess/preprocess_full.py` and
`tokenize_asm_gz.py`:

- **`endp`.** A line `<name> endp` or `<name> ends` no longer moves the label address (option 3(a) of 12.5), so the epilogue after a
  function's last inner label is paired again. With `--include-library` (the old block selection) the 60 samples of section 11 come
  out identical to the first build in every compared array.
- **Library functions.** `load_cfg` keeps each function's `library` flag and both samplers skip library blocks. A binary with fewer
  non-library blocks than the budget keeps just those and is not topped up with library blocks; only a binary made entirely of library
  functions is sampled from them. `--include-library` restores the old selection.
- **The rest of section 11.** The API vocabulary grows on training rows only and is frozen for the other splits; the embedder is seeded
  (`torch.manual_seed(--seed)`); per-sample failures are listed in `failures.json` and make `main` return 1; the size caps work again and
  default to off; the CFG path comes from a `cfg_path` column or `--cfg-root`; `settings.json` stops a resume with other settings;
  `extrn` rows are read in the aligning pass; the `signal` docstring no longer says connected; the vocabulary and embedder are saved
  every 50 samples instead of after every sample.

This is option 3(a) plus a change of block selection, not option 1: the opcode and operand channels still come from the listing, and the
match rate is high because the sampler no longer picks functions that the listing collapses. A side effect is that a binary made mostly
of library code keeps fewer blocks. On the full 500-block build 136 of 1,134 ransomware (12.0%) and 88 of 676 goodware (13.0%) keep
fewer than 500 blocks (minimum 40 and 16; median 500 for both), counted from the build's manifests.

Match rates of the four full builds (`match_rates.py`; share of each binary's selected instructions paired with a listing line):

| build | label | binaries | mean match | below 50% | below 90% | exactly 0 |
|---|---|---|---|---|---|---|
| v1 `b_sig500` (331ce0b) | ransomware | 1132 | 0.855 | 187 (16.5%) | 238 (21.0%) | 1.7% |
| v1 `b_sig500` (331ce0b) | goodware | 676 | 0.978 | 7 (1.0%) | 23 (3.4%) | 0.0% |
| v2 `b_sig500_v2` (c7899f0) | ransomware | 1134 | 0.813 | 190 (16.8%) | 373 (32.9%) | 1.8% |
| v2 `b_sig500_v2` (c7899f0) | goodware | 676 | 0.965 | 7 (1.0%) | 47 (7.0%) | 0.0% |
| v3 `b_sig500_v3` (PR #10) | ransomware | 1134 | 0.997 | 0 (0.0%) | 7 (0.6%) | 0.0% |
| v3 `b_sig500_v3` (PR #10) | goodware | 676 | 0.995 | 0 (0.0%) | 2 (0.3%) | 0.0% |
| uncapped `b_sigwhole_v3` (PR #10) | ransomware | 1134 | 0.991 | 0 (0.0%) | 6 (0.5%) | 0.0% |
| uncapped `b_sigwhole_v3` (PR #10) | goodware | 676 | 0.994 | 0 (0.0%) | 2 (0.3%) | 0.0% |

The AUC of `1 - match rate` as a ransomware score, over 712 clusters: v1 0.543 [0.461, 0.635] (1,808 ids), v2 0.747 [0.650, 0.827],
v3 0.502 [0.440, 0.574], uncapped 0.447 [0.339, 0.562]. The label leak of the match is gone. On the 300-id subset that PR #10 used
(150 ransomware including the 40 lowest-match of v2, 150 goodware, 182 clusters) the same statistic is v1 0.634, v2 0.790, 0.596 for
the first version of the fix that topped short samples up with library blocks (`v3_subset_acceptance.json`), and 0.573 [0.480, 0.663]
for the merged version (`v3b_subset_acceptance.json`); the subset over-represents the worst ransomware, so its values are higher than
the full builds' and are not comparable with them.

`match_acceptance` on the full v3 build (`v3_full_acceptance.json`): PASS on all five criteria. (1) 1,810 of 1,810 ids built, none
excluded; (2) under 90% matched: ransomware 0.6% (7 of 1,134), goodware 0.3% (2 of 676), a gap of 0.3 points (limit 2); (3) under 50%:
0 in both labels (limit 1%); (4) AUC of `1 - match rate` 0.502, interval [0.440, 0.574] (limit 0.55); (5) 5 of 5 sampled ids rebuild
identically in every field. On the uncapped build (`whole_acceptance.json`) criteria 1 to 4 pass (1,810 of 1,810; 0.5%, 6 of 1,134,
against 0.3%, 2 of 676, a gap of 0.2 points; 0 under 50%; AUC 0.447 [0.339, 0.562]); criterion 5 was not run, so the script's
overall verdict reads FAIL.

The residue under 90% is Go: in PR #10's subset the ransomware binaries still under 90% are Go binaries whose non-library functions
the listing also collapses (`unicode.init`), which the CFG does not mark as library.

### 13.2 The 500-block arms on the fixed build (Models 2 and 3)

`sig500v3` is the trainer of section 5 on the v3 build with all 141 inputs per block; `sig500v3-feat` feeds the GAT only the 77
handcrafted block features (Model 2: block features + CFG); `sig500v3-emb` only the 64-d pooled five-channel embedding (Model 3: block
embeddings + CFG). The embedding is the frozen embedder of pipeline B, now seeded, pooled per block; it is not trained.

| arm | AUC | PR-AUC | macro-F1 |
|---|---|---|---|
| R7 typed, whole graph | 0.979 | 0.987 | 0.935 |
| team sig500v2 | 0.948 | 0.972 | 0.876 |
| team sig500v3 | 0.948 | 0.973 | 0.893 |
| team sig500v3 feat (Model 2) | 0.954 | 0.976 | 0.879 |
| team sig500v3 emb (Model 3) | 0.966 | 0.982 | 0.909 |

| first | second | dAUC [95% CI] | dmacro-F1 [95% CI] | rule |
|---|---|---|---|---|
| team sig500v2 | team sig500v3 | -0.000 [-0.022, +0.019] | +0.017 [-0.014, +0.046] | does not beat |
| team sig500 | team sig500v3 (1,808 binaries) | -0.001 [-0.023, +0.018] | +0.008 [-0.020, +0.035] | does not beat |
| R7 | team sig500v3 | -0.031 [-0.064, -0.006] | -0.042 [-0.095, +0.012] | does not beat |
| R7 | sig500v3 feat (Model 2) | -0.024 [-0.040, -0.010] | -0.057 [-0.121, +0.003] | does not beat |
| R7 | sig500v3 emb (Model 3) | -0.012 [-0.026, -0.000] | -0.026 [-0.075, +0.028] | does not beat |
| sig500v3 feat (Model 2) | sig500v3 emb (Model 3) | +0.012 [+0.000, +0.025] | +0.031 [+0.002, +0.069] | **beats** |
| team sig500v3 | sig500v3 emb (Model 3) | +0.019 [+0.001, +0.042] | +0.016 [-0.008, +0.044] | does not beat |
| ours b500 no edges (seed 0) | team sig500v3 | +0.003 [-0.020, +0.024] | -0.026 [-0.101, +0.050] | direction |
| ours b500 no edges (seed 0) | sig500v3 feat (Model 2) | -0.020 [-0.049, +0.007] | -0.073 [-0.151, +0.002] | direction |
| ours b500 no edges (seed 0) | sig500v3 emb (Model 3) | +0.016 [+0.000, +0.031] | +0.020 [-0.030, +0.077] | direction |

The fix removes the leak but does not change the AUC: v2 to v3 is -0.000 [-0.022, +0.019]. It does move the operating point back. On
seed 0, v2 had 145 false alarms at precision 0.881; v3 has 38 at precision 0.963 (recall 0.945 to 0.862), close to the first build's
39 and 0.962. So the v2 shift toward recall and false alarms of section 6 went with the `endp` leak.

Model 3 beats Model 2 under the pre-registered rule: +0.012 AUC and +0.031 macro-F1. The lower AUC bound prints as +0.000; the rule
requires it to be above zero and it is before rounding, so the pass is at the margin. Removing the 77 features from the full input
does not hurt (all to emb: +0.019 [+0.001, +0.042] AUC, macro-F1 interval includes zero): at 500 blocks the handcrafted counts add
nothing to the pooled embedding. No 500-block arm beats R7; Model 3 is the closest (-0.012 AUC).

### 13.3 Removing the cap

`b_sigwhole_v3` is the same PR #10 source with no block budget: `build_b --source main-fix --sample signal --num-blocks 1000000`,
which keeps every non-library block (it was built before main's PR #13 added `--num-blocks 0` / `--whole`; PR #13's tests show on
hand-built CFGs that a budget of 10**9 and no budget select the same blocks). The 500-block cap had cut 998 of 1,134 ransomware (88.0%) and 588 of 676 goodware
(87.0%); uncapped, the median binary has 1,938 (ransomware) and 1,732.5 (goodware) blocks, the largest 189,895. The trainer is
unchanged except that a batch above 300,000 nodes is split into parts whose gradients are summed (`train_b.py`, `NODE_BUDGET`).

| arm | AUC | PR-AUC | macro-F1 |
|---|---|---|---|
| R7 typed, whole graph | 0.979 | 0.987 | 0.935 |
| team sigwhole | 0.970 | 0.982 | 0.899 |
| team sigwhole feat (Model 2) | 0.970 | 0.983 | 0.929 |
| team sigwhole emb (Model 3) | 0.981 | 0.989 | 0.901 |

| first | second | dAUC [95% CI] | dmacro-F1 [95% CI] | rule |
|---|---|---|---|---|
| team sig500v3 | team sigwhole | +0.022 [+0.002, +0.052] | +0.006 [-0.039, +0.056] | does not beat |
| sig500v3 feat (Model 2) | sigwhole feat (Model 2) | +0.015 [+0.003, +0.029] | +0.050 [+0.006, +0.108] | **beats** |
| sig500v3 emb (Model 3) | sigwhole emb (Model 3) | +0.014 [+0.004, +0.025] | -0.009 [-0.034, +0.015] | does not beat |
| R7 | team sigwhole | -0.009 [-0.023, +0.002] | -0.036 [-0.074, -0.002] | does not beat |
| R7 | sigwhole feat (Model 2) | -0.009 [-0.021, +0.001] | -0.007 [-0.046, +0.038] | does not beat |
| R7 | sigwhole emb (Model 3) | +0.002 [-0.005, +0.009] | -0.035 [-0.091, +0.019] | does not beat |
| sigwhole feat (Model 2) | sigwhole emb (Model 3) | +0.011 [+0.002, +0.022] | -0.028 [-0.069, +0.001] | does not beat |
| team sigwhole | sigwhole emb (Model 3) | +0.010 [+0.001, +0.023] | +0.001 [-0.042, +0.036] | does not beat |

Removing the cap raises AUC in all three arms, by 0.014 to 0.022 with every interval above zero, and Model 2 also passes the rule
(+0.015 AUC, +0.050 macro-F1). This is the size of the cost the cap had on our own GNN (section 6: 0.021 no edges, 0.063 typed), and
it closes most of the gap to R7: the 500-block arms were 0.012 to 0.031 AUC below R7 with no interval reaching above zero (upper ends
-0.006, -0.010 and -0.000), the uncapped arms are -0.009 to +0.002 with every interval including zero. Uncapped Model 3 is level with R7 on AUC (+0.002 [-0.005, +0.009]) and
PR-AUC (+0.002 [-0.003, +0.008]) and lower on macro-F1 (-0.035), with an interval that includes zero; by the non-inferiority reading of
`gates_team.md` (lower bounds above -0.01 AUC and -0.02 macro-F1) it is non-inferior in AUC and not shown non-inferior in macro-F1.
Uncapped, Model 3 is above Model 2 on AUC (+0.011) but not on macro-F1 (-0.028), so the 500-block ordering of 13.2 holds for the ranking
only. None of these arms has a no-edge control, so they do not test whether the CFG edges matter.

### 13.4 Janice's five-channel Model 1

`fivechannel` is Janice's `Model1TokenSequence` from main's PR #12 (`s03a_transformer/five_channel/`), unchanged, on the five-channel
instruction tuples (opcode, two operand roles, API, behaviour) of a pipeline-B build, trained on our folds with the API vocabulary made
a training-set one per fold. It reads the first `--max-instructions` instructions of each binary: 128 as she wrote it, or 512. L128 and
L512 read the 500-block v3 build, `L512 whole` the uncapped build.

| arm | AUC | PR-AUC | macro-F1 |
|---|---|---|---|
| Model 1 tokenseq (ours, `token_sequence/`) | 0.970 | 0.982 | 0.902 |
| five-channel L128, 500-block build | 0.919 | 0.954 | 0.863 |
| five-channel L512, 500-block build | 0.926 | 0.958 | 0.878 |
| five-channel L512, uncapped build | 0.919 | 0.955 | 0.863 |

| first | second | dAUC [95% CI] | dmacro-F1 [95% CI] | rule |
|---|---|---|---|---|
| tokenseq | five-channel L128 | -0.051 [-0.083, -0.024] | -0.039 [-0.097, +0.026] | does not beat |
| tokenseq | five-channel L512 | -0.043 [-0.071, -0.020] | -0.024 [-0.063, +0.022] | does not beat |
| five-channel L512 | five-channel L512 uncapped | -0.007 [-0.059, +0.038] | -0.015 [-0.072, +0.038] | does not beat |
| R7 | five-channel L512 uncapped | -0.060 [-0.108, -0.020] | -0.072 [-0.134, -0.022] | does not beat |
| TF-IDF (seed 0) | five-channel L128 | -0.072 [-0.117, -0.034] | -0.099 [-0.147, -0.058] | direction |
| TF-IDF (seed 0) | five-channel L512 | -0.063 [-0.100, -0.032] | -0.057 [-0.120, -0.006] | direction |

Why it does not gain: the model sees a prefix of almost every binary. In the seed-0 runs 1,809 of the 1,810 test binaries have more
than 128 instructions and 1,806 more than 512 (`truncated_instructions` in `metrics.json`, summed over the five folds), in both builds.
The uncapped build gives it more instructions per binary, but it still reads only the first 512, so it cannot use them, and it does not
(-0.007, interval across zero). On the same 500-block build L512 is nominally 0.007 AUC above L128 (means; no paired test). Our
`token_sequence/` Model 1 cuts each binary's whole token stream into windows of 512, encodes each window and averages the window vectors
per binary; at test time it reads every window, in training a random choice of at most 64 per binary and epoch. It is 0.043 AUC above
the five-channel L512. The suggestion for her model is the same windowing over all instructions; the five-channel tuples and her
encoder can stay as they are.

### 13.5 Model 1 with the out-of-fold threshold

Main PR #12 adds `--threshold-from oof`: the decision threshold is chosen on out-of-fold predictions of the training binaries (3 refits on
folds that keep a ransomware family or a goodware identity group whole) instead of on the validation families, on which the model
saturates within a few epochs so that a wide band of thresholds ties and the first is taken. Only the threshold of the AUC checkpoint
changes; the two runs train the same model with the same seed.

| arm | AUC | PR-AUC | macro-F1 |
|---|---|---|---|
| Model 1 tokenseq, validation threshold | 0.970 | 0.982 | 0.902 |
| Model 1 tokenseq, out-of-fold threshold (`-thoof`) | 0.968 | 0.981 | 0.918 |

tokenseq to `-thoof`: dAUC -0.002 [-0.007, +0.002] (level; the 0.002 is run-to-run noise of a retrained model), dPR-AUC -0.001
[-0.005, +0.003], dmacro-F1 +0.016 [-0.014, +0.054]. Macro-F1 moves from 0.902 to 0.918 but the interval includes zero, so the rule is
not met. Against TF-IDF (seed 0) `-thoof` is -0.011 [-0.023, -0.002] AUC and -0.014 [-0.063, +0.021] macro-F1.

### 13.6 What this means

Once the 500-block cap is removed, the team pipeline is level with our best model on AUC: uncapped Model 3 (the edge-typed GAT on
pooled five-channel embeddings) reads 0.981 against R7's 0.979, +0.002 [-0.005, +0.009], with macro-F1 lower by 0.035 and an interval
that includes zero. Most of the 0.025 to 0.031 AUC gap of section 6 was the cap, which section 6 could not rule out, and the fixed
preprocessing removed the label leak of the listing match without changing AUC. The earlier conclusions are unchanged. Graph structure:
these arms have no no-edge control, and on our GNN typed edges are 0.004 AUC above no edges with an interval including zero
(`PLAN_GNN.md`, "Final results", 1). Imports: the header-imports LR (0.985 AUC, seed 0) is at the level of every arm here (the highest
seed-0 team AUC is 0.986, uncapped Model 3); no pair against it was run for these arms. Tokens: the best team arm is the one fed only a
pooled embedding of the instruction tokens, and the 77 handcrafted counts add nothing to it at 500 blocks, which agrees with the
bag-of-tokens reading of `PLAN_GNN.md`.

Next steps:

- **Yanping (s02).** Run the rebuild check (criterion 5) on the uncapped build, `match_acceptance --build RUNS/team/b_sigwhole_v3`
  without `--skip-rebuild`. Option 1 of 12.5 (opcode and operand channels from the `.jsonl.gz`) is still open; it would remove the
  dependence on the listing and the Go residue of 13.1. Use `--whole` (PR #13) for model inputs: every budget tested, on our GNN
  (section 8) and on these arms, lost AUC.
- **Dorothy (Models 2 and 3, the GAT).** Train on uncapped builds with a per-batch node budget, as `train_b.py` does; report the `feat`
  and `emb` arms with family-held-out validation and per-sample predictions (section 7, items 1 and 2). Pair Model 3 against a no-edge
  version of the same GAT to test the CFG edges. A trained block encoder remains the open slot for Model 3 (section 8).
- **Janice (Model 1, five-channel).** Window over all instructions instead of the first 128 or 512, as `token_sequence/` does, and use
  `--threshold-from oof`; then compare with tokenseq (0.970 AUC) and TF-IDF (0.982, seed 0).
