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

## 6. Results (partial, 2026-10-19)

kfold x86, 1,810 test binaries, AUC checkpoint at its validation threshold; pooled table seed 0, paired differences over
seeds 0 1 2 (TF-IDF: seed 0 only).

| model | AUC | PR-AUC | macro-F1 | accuracy | precision | recall | TP | FN | FP | TN |
|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF | 0.982 | 0.990 | 0.935 | 0.939 | 0.951 | 0.951 | 1079 | 55 | 55 | 621 |
| header-imports LR | 0.985 | 0.991 | 0.954 | 0.957 | 0.971 | 0.960 | 1089 | 45 | 33 | 643 |
| our typed GNN (R7), whole graph | 0.986 | 0.990 | 0.935 | 0.938 | 0.965 | 0.935 | 1060 | 74 | 38 | 638 |
| A, opcode mode, 500 blocks | 0.956 | 0.975 | 0.850 | 0.854 | 0.939 | 0.820 | 930 | 204 | 60 | 616 |
| A on CodeBERT vectors, 500 blocks | 0.791 | 0.872 | 0.723 | 0.724 | 0.925 | 0.608 | 690 | 444 | 56 | 620 |

Against R7 (3 seeds): A opcode dAUC -0.025 [-0.043, -0.009], dmacro-F1 -0.072 [-0.136, -0.008]; A on CodeBERT vectors
dAUC -0.159 [-0.221, -0.104], dmacro-F1 -0.184 [-0.242, -0.129]. Both lose on ransomware recall, not on false alarms.
Still running: B (`sig500`), our GNN at the same 500-block cap (separates the cap from the model), and our GNN on the
same CodeBERT vectors (paths 1 and 2, separates the vectors from Dorothy's GAT). B will also be reported on the subset
whose selected instructions are at least 90% matched (section 4). Final tables: `RUNS/gates_team.md`,
`RUNS/gates_paths_codebert.md`.

These runs pin the teammate code at `origin/gnn` 331ce0b. Since then `origin/gnn` 22d1230 lets `train_cfg.py` load
`preprocess_full.py` subgraphs with edge attributes, `origin/yanping` 38c38e5 adds single-instruction tokenization to
`tokenize_asm_gz.py`, and the new `origin/TokenSeqModel` (Janice) adds Model 1 (`src/models/token_sequence.py`) in a
`src/` layout. That branch also deletes `CNN-ViT/`, the old `CT_GAT/` files and most of `meetings/`; check before it
is merged anywhere.

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

## 8. The team diagram (2026-10-19) against what exists

| Diagram box | Team code | Ours (`3-block-level-counting`) | Evidence and suggestion |
|---|---|---|---|
| `jsonl.gz` → `load_cfg()` | `preprocess_full.py` | `pipeline/s02_build_cfg_cache` | Same input. Keep. |
| `asm.gz` → `parse_asm_instructions()` → match RVA and mnemonic | `tokenize_asm_gz.py`, `preprocess_full.py` | none: every token comes from `jsonl.gz` | **Drop.** The match fails on 20% of ransomware but 1% of goodware (section 4), which makes missing features a label signal. The team's own `ida_feature_readers.py` produces the same operand-role triplet from `jsonl.gz` alone (built for 6,517 of 6,523 exports), and its side arrays aligned with our cache with zero failures. |
| Full CFG | built per sample | the cache (`cfg_edges`, edge kinds, back edges, call edges) | Keep. |
| Randomly select subgraphs | `walk` / `signal` at a block budget | `s04_select_subgraph` (`cap`, `lcc`; whole graph by default) | **Use the whole graph, deterministic.** Every budget we tried lost AUC against the whole graph (cap 20k -0.011, cap 5k -0.016, lcc 20k -0.017, lcc 5k -0.011; none non-inferior), and at 500 blocks both A arms lost more. Random selection also makes a test prediction depend on a seed. Keep a deterministic cap only as a memory guard. |
| `tokenize_selected_blocks()` | five channels: opcode, two operand roles, API id, behaviour | `s03_tokenizer`: `mn`, `sig`, `beh`, `api` | Settled: under the pre-registered 3-seed test the team's operand-role vocabulary tied ours (section "Vocabulary decision", `PLAN_GNN.md`). Either works; fit the API vocabulary on training binaries only. |
| Model 1, token sequence | `origin/TokenSeqModel` `src/models/token_sequence.py` (new) | no sequence model on this branch; TF-IDF over the same tokens is the bag-of-tokens reference | Add to the same folds and report it next to TF-IDF: on this data the earlier graph-transformer sequence arms (mF1 0.906-0.916) stayed below TF-IDF (0.939). |
| Model 2, block features + CFG | `FullBlockGAT` / `BlockGAT` | `s05_block_counting` + `s06`-`s08` GNN (R7) | Running both (section 6). |
| Model 3, block embeddings + CFG | 64-d pooled random embedding (B); CodeBERT (A) | `transformer_paths` path 2: any frozen block encoder, now CodeBERT on register-free text | Running (paths 1 and 2). A trained block encoder is the open slot. |
| (missing) baselines | none | TF-IDF, header-imports LR, majority and x86-rule floors | **Add** to the diagram: every model is read against them. |
| (missing) evaluation | one fixed split, random validation | 5-fold family holdout, family-held-out validation, paired bootstrap | **Add** as the box all three models feed. |

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
