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

`--sample walk` (the default) grows one connected subgraph from a random block. `--sample signal` fills the same budget from functions that call imports. Either way, the transformer and the graph models receive that same set of blocks.

Goodware and ransomware, and train and test, share one `processed/cfg_subgraphs/channel_embedder.pt`. A later run reloads it and skips a sample that already has a `.pt`. `--overwrite` builds a new embedder and rebuilds those samples. The embedding tables are a random starting point; this script does not train them.

The manifest stores mismatch counts and up to 32 examples of each gap. Each saved sample also rewrites `processed/api_vocab.json`. `--freeze-vocab` keeps that file unchanged and maps an unseen API name to id 75.

```bash
python preprocess_full.py --num-blocks 8 --seed 0
python preprocess_full.py --root ../../data/Mendeley_Goodware --limit 1 --num-blocks 8 --seed 0
python preprocess_full.py --sample-id <sha256> --num-blocks 8 --seed 0
python preprocess_full.py --split train --sample signal --num-blocks 64 --seed 0
python preprocess_full.py --overwrite --num-blocks 8 --seed 0
```

`--max-asm-bytes`, `--max-cfg-blocks`, and `--max-cfg-instructions` are accepted. The row passed into the run does not include assembly size or the original block and instruction counts, so those three limits do not skip samples.

Each `<sha>.pt` holds the shared block ids, instruction token rows, block features, 64-dimensional block embeddings, and the directed edges among the selected blocks.

## jsonl-only path (cache, tokenizer, subgraph)

A second path through this stage reads only the CFG export (`*.jsonl.gz`) and never the assembly listing. It is a package (`cache/`, `tokenizer/`, `subgraph.py`, imported as `s02_preprocess.<module>` from `CT_GAT/pipeline`), so it does not touch `preprocess_full.py` or `tokenize_asm_gz.py`, which stay flat scripts (`preprocess_full.py` puts this folder on `sys.path` itself).

| Module | What it does | Run from `CT_GAT/pipeline` |
|---|---|---|
| `cache/build_cache.py` | each export becomes `<cache>/samples/<sample_id>.npz` (token ids, block and function pointers, CFG and call edges, edge kinds, back edges), with `index.csv` and `vocab.json` beside it | `python -m s02_preprocess.cache.build_cache` |
| `cache/check_cache.py` | checks every invariant of every built sample and compares a few against the bundle's own reader | `python -m s02_preprocess.cache.check_cache` |
| `cache/folds.py` | the 5-fold family-holdout fold CSV of a dataset; families and goodware identity groups are never split | `python -m s02_preprocess.cache.folds --out <dir>` |
| `tokenizer/` | `TokenVocab` maps the cache's global token ids to run ids by training document frequency; `token_table` lays out the tokens of a block selection | imported |
| `subgraph.py` | `select_subgraph`: whole functions that fit a block budget (policy `cap`) | imported |

Where the data lives is set per machine, not in the code: copy `datasets.json.example` to `datasets.json` (gitignored) and give each dataset its cache folder, the protocols that read it and the raw inputs it is built from; `--cache <name>` picks a dataset and `GT_CACHE` (a folder) overrides it. `common/paths.py` describes the order.

**Which models read which path, and why.** The scripts above (`tokenize_asm_gz.py`, `preprocess_full.py`) feed the transformer in `s03a_transformer`. The models that read the cache are the graph models of `s03b_cfg_feature`, the frozen block-encoder path of `s03c_cfg_transformer`, the token-sequence model of `s03a_transformer` and the baselines of `s04_evaluation`; [`../README.md`](../README.md) draws how the stages connect. The reasons are in the team-pipeline review of the 3-block-level-counting branch, [`TEAM_PIPELINES_REVIEW.md`](../docs/history/TEAM_PIPELINES_REVIEW.md):

- Section 8 compares the team diagram with what exists and recommends dropping the box that pairs the listing with the CFG: the pairing fails on about 20% of ransomware but 1% of goodware, so the missing features themselves tell the label, and every token the cache needs is already in the `jsonl.gz`.
- Section 11 reviews the streaming version of the two scripts here (main c7899f0): it fixes the memory use on the two 23 GB listings, but pairs fewer instructions on 35 of 60 rebuilt samples because a label line ending a function resets the address.
- Section 12 shows why the pairing fails (the listing collapses the library functions that the CFG holds), measures it by label, family and size, and ranks the fixes; the first one is to take the per-instruction fields from the `jsonl.gz`, which is what the cache does.

`token_mapping.py` in this folder is Yanping's module, copied unchanged from `origin/yanping` (`CT_GAT/preprocessing/token_mapping.py`, sha256 `8da1001501cf88228f6bf34b2cfe0829f18cd71730cc4bf43631eb6930be56e3`; `tests/test_token_mapping.py` fails if it changes). The cache builder takes the behaviour id of every instruction from it (`categorize_behavior`, `BEHAVIOR_MAP`), and the existing caches were built with exactly this file, so it is moved here rather than rewritten and nobody edits it for this package. `tokenize_asm_gz.py` is not a substitute: its API matcher gives a different id to 44 of the 12,327 API names of the first cache's vocabulary.

`cache/folds.py` is the fold algorithm of `family_holdout/folds.py` (LLM_Classification history), restated for one corpus. It reproduces the committed fold assignment: fed the family and group columns of `folds_mendeley.csv` it returns that file's `fold` column on all 2,509 rows (`tests/test_folds.py`, passing). It cannot regenerate that file from the bundle, because the original goodware groups were hashes of each file's mnemonic stream (1,024 groups), taken from files that are not in the bundle, whose `identity_group` is one group per file (1,169); the original cohort of 2,509 files is also larger than the 2,411 eligible rows (2,411 files join, 2,085 of them change fold). The module docstring has the details.

Tests: from `CT_GAT/pipeline`, `python -m pytest -m "not real_data"` runs without any data; `python -m pytest` also runs the tests marked `real_data` when `datasets.json` (or `GT_CACHE`) names a cache.
