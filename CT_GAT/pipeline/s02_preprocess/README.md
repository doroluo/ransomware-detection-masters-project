## Dataset

https://data.mendeley.com/datasets/p3v94dft2y/3

Binary goodware vs ransomware. Labels: `0` = benign, `1` = ransomware.

Each sample is a pair: an IDA listing `*.asm.gz` and a control-flow graph `*.jsonl.gz` (schema `ida-cfg-research/2.1`). Membership comes from each collection's `splits/train.csv` and `splits/test.csv`.

## Layout

```
CT_GAT/
  data/
    Mendeley_Ransomware/
    Mendeley_Goodware/
  pipeline/
    s01_ida_export/
      extract_cfg_with_ida.py   (CFG export, jsonl.gz)
      run_mendeley_asm.py       (listing export, asm.gz)
    s02_preprocess/
      tokenize_asm_gz.py
      preprocess_full.py
    s03a_transformer/
    s03b_cfg_feature/
    s03c_cfg_transformer/
  processed/
    api_vocab.json
    cfg_subgraphs/
  README.md
```

PyTorch is required. `pefile` is used only when `tokenize_asm_gz.py` is given `--pe-root`.

## Pipeline

```
data/Mendeley_*/samples/<split>/<family>/<sha>.asm.gz
data/Mendeley_*/samples/<split>/<family>/<sha>.jsonl.gz
        │
        ├─ tokenize_asm_gz.py
        │     processed/mendeley_sequences/<sha>.pt
        │     processed/mendeley_splits/{train,test}.csv
        │     processed/api_vocab.json
        │
        └─ preprocess_full.py
              processed/cfg_subgraphs/<sha>.manifest.json
              processed/cfg_subgraphs/<sha>.pt
              processed/cfg_subgraphs/channel_embedder.pt
              processed/api_vocab.json
```

`preprocess_full.py` loads the whole CFG, keeps one subgraph, aligns those instructions to the IDA listing by label address, and tokenizes only the selected lines. The same blocks are written for the sequence model, the handcrafted-feature graph, and the embedding graph.

## tokenize_asm_gz.py

Turns each IDA listing into one sequence. Each row is `[opcode_id, operand1_id, operand2_id, api_id, behavior_id]`.

| api_id | meaning |
|--------|---------|
| `0` | the instruction does not name an API |
| `75` | a call or jump through an import slot whose name cannot be read |
| `76 + index` | a name in `processed/api_vocab.json` |

`extrn` lines are imports, not sequence rows. `import_inventory` stores imported, called, and unused APIs. Behavior tags are `NONE`, `CRYPTO_OPCODE`, `DECRYPT`, `ENCRYPT`, `KEY_SETUP`, `FILE_ENUM`, `FILE_READ`, `FILE_WRITE`, and `FILE_DELETE`.

A full run loads `processed/api_vocab.json`, appends new names, and writes it back. `--limit` and `--sample-id` start from an empty vocabulary and do not write that file.

```bash
python tokenize_asm_gz.py
python tokenize_asm_gz.py --limit 2
python tokenize_asm_gz.py --split train --architecture x86
python tokenize_asm_gz.py --sample-id <sha256>
python tokenize_asm_gz.py --pe-root path/to/pe/files
```

`--pe-root` is optional. Files there must be named `<sha256>.exe`, `.dll`, or `.pe`. The Mendeley folders have no original binaries, so a normal run takes imports from the IDA `extrn` list.

## preprocess_full.py

Set `--num-blocks` or `--max-instructions` (or both). The saved piece can be smaller than the request when the connected region runs out of blocks.

**Uncapped mode.** `--num-blocks 0` (or `--whole`, the same thing) sets no block budget: with `--sample signal` the sample keeps every non-library block of the binary (library functions only when it has no other code, as below), and `--max-instructions` may still cap instructions. `--sample walk` is still one connected region, which is always one function (CFG edges stay inside a function), so use `signal` for a whole binary. The cost is larger `.pt` files and longer runs, since the block embeddings, the alignment with the listing and the saved edges grow with the binary (some have millions of blocks). `settings.json` records `num_blocks: 0`, so the settings guard treats it as a different setting.

`--sample walk` (the default) grows one connected subgraph from a random block. `--sample signal` fills the same budget with the blocks of whole functions that call imports, highest signal first; they come from several functions, so the result is not one connected subgraph. Either way, the transformer and the graph models receive that same set of blocks.

**Library functions are skipped.** The listing holds one `COLLAPSED FUNCTION` line for each function IDA marks as library, so its blocks can never be paired with assembly. `load_cfg` keeps the `library` flag of each function record and both samplers leave those blocks out. A binary with fewer than `--num-blocks` non-library blocks therefore samples just those blocks, with no top-up from library code, which would add blocks that can never be matched (and, since ransomware is linked mostly from library code, would tell the label). Only a binary whose functions are all library functions is sampled from them, so it still gets a sample. `--include-library` restores the old sampling.

**Vocabulary and embedder.** The API vocabulary grows on `train` rows only: a run processes the train rows of every `--root` first, then every other split with the vocabulary frozen (an unseen name maps to id 75). `--freeze-vocab` freezes it for the train rows too. The embedder is seeded with `--seed`, and rows it adds when the vocabulary grows depend only on the seed and the API id. Goodware and ransomware, and train and test, share one `processed/cfg_subgraphs/channel_embedder.pt`; the embedding tables are a random starting point, this script does not train them.

**Saving and resuming.** The embedder and `processed/api_vocab.json` are saved every `--checkpoint-every` samples (default 50) and at the end; `--limit` and `--sample-id` runs never write the shared vocabulary. `checkpoint.json` in the output folder lists the samples that the saved vocabulary and embedder cover, and a rerun skips only those. The folder also keeps `settings.json` (sample mode, seed, budgets, `--include-library`, `--freeze-vocab`, split order, sha256 of the two source files). A rerun with other settings, or into a folder of samples without that file, stops; `--overwrite` rebuilds the folder with the new settings.

**Where the CFG is.** The `cfg_path` column of the splits CSV (relative to the collection root) names it; without that column it is the listing's name with `.jsonl.gz`. `--cfg-root` points at a separate CFG tree: the column is read relative to it, or the listing's path under the collection root is repeated under it. A splits CSV without a `collection` column uses the folder name.

**Failures and caps.** A sample that raises is recorded in `failures.json` (id and error) and the run goes on; the exit status is 1 when any sample failed, and `--fail-fast` stops at the first. `--max-asm-bytes`, `--max-cfg-blocks` and `--max-cfg-instructions` read `asm_bytes`, `num_blocks` and `num_instructions` of the splits CSV; they are off (0) by default, and every sample a cap skips is logged and listed in `skipped.json` with its reason. The listing is decompressed and scanned once per sample.

The manifest stores mismatch counts and up to 32 examples of each gap. `missing_asm_count` counts the selected instructions with no listing line, not the instructions that share an RVA with another one (those are never paired and are in `duplicate_count`).

```bash
python preprocess_full.py --num-blocks 8 --seed 0
python preprocess_full.py --root ../../data/Mendeley_Goodware --limit 1 --num-blocks 8 --seed 0
python preprocess_full.py --sample-id <sha256> --num-blocks 8 --seed 0
python preprocess_full.py --split train --sample signal --num-blocks 64 --seed 0
python preprocess_full.py --include-library --cfg-root <cfg tree> --fail-fast --num-blocks 8 --seed 0
python preprocess_full.py --overwrite --num-blocks 8 --seed 0
```

Each `<sha>.pt` holds the shared block ids, instruction token rows, block features, 64-dimensional block embeddings, and the directed edges among the selected blocks.

## jsonl-only path (cache, tokenizer, subgraph)

A second path through this stage reads only the CFG export (`*.jsonl.gz`) and never the assembly listing. It is a package (`cache/`, `tokenizer/`, `subgraph.py`, imported as `s02_preprocess.<module>` from `CT_GAT/pipeline`), so it does not touch `preprocess_full.py` or `tokenize_asm_gz.py`, which stay flat scripts (`preprocess_full.py` puts this folder on `sys.path` itself).

| Module | What it does | Run from `CT_GAT/pipeline` |
|---|---|---|
| `cache/build_cache.py` | each export becomes `<cache>/samples/<sample_id>.npz` (token ids, block and function pointers, CFG and call edges, edge kinds, back edges), with `index.csv` and `vocab.json` beside it | `python -m s02_preprocess.cache.build_cache [--cache NAME] [--corpora ...] [--base-cache DIR] [--limit N] [--dry]` |
| `cache/check_cache.py` | checks every invariant of every built sample and compares a few against the reader (`gnn_dataset.py`) that ships in the collection or the bundle | `python -m s02_preprocess.cache.check_cache [--cache NAME]` |
| `cache/folds.py` | the 5-fold family-holdout fold CSV of a dataset; families and goodware identity groups are never split | `python -m s02_preprocess.cache.folds [--corpus NAME] [--dataset NAME] [--out <dir>]` |
| `tokenizer/` | `TokenVocab` maps the cache's global token ids to run ids by training document frequency; `token_table` lays out the tokens of a block selection | imported |
| `subgraph.py` | `select_subgraph`: whole functions that fit a block budget (policy `cap`) | imported |

Nothing is configured for the default path: with `Mendeley_Ransomware.zip` and `Mendeley_Goodware.zip` unzipped into `CT_GAT/data/`, `build_cache` reads the collections there and the committed folds in `s02_preprocess/folds/`, and writes the cache to `CT_GAT/cache/mendeley_x86/`. `python -m common.catalog` shows which collections, caches and fold files a machine has. A different location is set with `GT_COLLECTIONS`, `GT_CACHE` and the other variables or with `datasets.json` ([Overrides](../README.md#overrides) in the pipeline README); `--cache <name>` picks a dataset and `common/paths.py` describes the order.

`build_cache --cache <name>` builds the cache of a dataset of `common/catalog.py` (`mendeley_x86`, the default, or `combined_v3`) or of `datasets.json`. `--corpora` defaults to the dataset's corpora and may not name others, `--base-cache` to the cache of the dataset's base (`combined_v3` starts from `mendeley_x86`, which must be built first: `python -m s02_preprocess.cache.build_cache`). A collection folder that is not unzipped under `CT_GAT/data` stops the build with the folder and the zip that holds it. `--dry` prints the cache, corpora, base cache and input paths the command resolves and stops. `folds.py` writes to `CT_GAT/cache/<dataset>/` unless `--out` says otherwise: the folds it generates are not the published ones in `s02_preprocess/folds/`.

**Which models read which path, and why.** The scripts above (`tokenize_asm_gz.py`, `preprocess_full.py`) feed the transformer in `s03a_transformer`. The models that read the cache are the reference models of `reference/` (the graph model, the frozen block-encoder path and the token-sequence model) and the baselines of `s04_evaluation`; [`../README.md`](../README.md) draws how the stages connect. The reasons are in the team-pipeline review of the 3-block-level-counting branch, [`TEAM_PIPELINES_REVIEW.md`](../docs/history/TEAM_PIPELINES_REVIEW.md):

- Section 8 compares the team diagram with what exists and recommends dropping the box that pairs the listing with the CFG: the pairing fails on about 20% of ransomware but 1% of goodware, so the missing features themselves tell the label, and every token the cache needs is already in the `jsonl.gz`.
- Section 11 reviews the streaming version of the two scripts here (main c7899f0): it fixes the memory use on the two 23 GB listings, but pairs fewer instructions on 35 of 60 rebuilt samples because a label line ending a function resets the address.
- Section 12 shows why the pairing fails (the listing collapses the library functions that the CFG holds), measures it by label, family and size, and ranks the fixes; the first one is to take the per-instruction fields from the `jsonl.gz`, which is what the cache does.

`token_mapping.py` in this folder is Yanping's module, copied unchanged from `origin/yanping` (`CT_GAT/preprocessing/token_mapping.py`, sha256 `8da1001501cf88228f6bf34b2cfe0829f18cd71730cc4bf43631eb6930be56e3`; `tests/test_token_mapping.py` fails if it changes). The cache builder takes the behaviour id of every instruction from it (`categorize_behavior`, `BEHAVIOR_MAP`), and the existing caches were built with exactly this file, so it is moved here rather than rewritten and nobody edits it for this package. `tokenize_asm_gz.py` is not a substitute: its API matcher gives a different id to 44 of the 12,327 API names of the first cache's vocabulary.

`cache/folds.py` is the fold algorithm of `family_holdout/folds.py` (LLM_Classification history), restated for one corpus. It reproduces the committed fold assignment: fed the family and group columns of `folds_mendeley.csv` it returns that file's `fold` column on all 2,509 rows (`tests/test_folds.py`, passing). It cannot regenerate that file from the bundle, because the original goodware groups were hashes of each file's mnemonic stream (1,024 groups), taken from files that are not in the bundle, whose `identity_group` is one group per file (1,169); the original cohort of 2,509 files is also larger than the 2,411 eligible rows (2,411 files join, 2,085 of them change fold). The module docstring has the details.

Tests: from `CT_GAT/pipeline`, `python -m pytest -m "not real_data"` runs without any data; `python -m pytest` also runs the tests marked `real_data` when the cache of the run exists (`CT_GAT/cache/mendeley_x86` after a `build_cache`, or the folder `GT_CACHE` names).
