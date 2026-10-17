# How graphs are built and tokenized

One page per question the team asked: which file builds the graph, which file tokenizes, and where each
choice lives. Paths are relative to `CT_GAT/block_level_counting/`.

## 1. From an IDA export to cached arrays (once per dataset)

`pipeline/s02_build_cfg_cache/build_cache.py` streams every `.jsonl.gz` export once, one function record at a
time (`encode_sample`). For each binary it writes one `.npz` to the cache:

| What | How | Array(s) |
|---|---|---|
| Functions | Export order; library (FLIRT) and thunk flags kept | `fn_blk_ptr`, `fn_flags` |
| Blocks | Each function's blocks sorted by start address; global block ids are contiguous per function | `blk_insn_ptr`, `blk_api_ptr`, `blk_feat` (6 structural features) |
| Block edges (CFG) | The export's per-function edge list, remapped to global block ids | `cfg_edges` |
| Edge type | `classify_edge_kind` (fall-through, conditional, jmp, indirect, extern, other) and `dfs_back_edges` (loop back edge) | `edge_kind`, `edge_back` |
| Call edges | Call instructions whose code reference is a function start, as (caller, callee) function ids, deduplicated | `call_edges` |
| Tokens per instruction | `mn` = mnemonic; `sig` = mnemonic + operand categories (`xor r,r`, `mov r,m`); `beh` = Yanping's behaviour id (`CT_GAT/preprocessing/token_mapping.py`); `api` = `dll!name` of the imports an instruction references | `mn`, `sig`, `beh`, `api` (global string ids, `vocab.json`) |

Register names and operand widths are never stored as tokens: `sig` keeps only the operand **category**
(`OPERAND_ABBREV`: register, immediate, memory, ...). `check_cache.py` verifies every array.

## 2. Tokenizing a block for a run (every training run)

| Step | File | What it does |
|---|---|---|
| Vocabulary | `pipeline/s03_tokenizer/vocab.py` (`TokenVocab`) | Keeps a token only if it occurs in at least 2 **training** binaries of the fold; the rest map to UNK. Computes IDF on training binaries; adds `[MASK]` ids. Nothing is fitted on validation or test |
| Block token list | `pipeline/s03_tokenizer/block_tokens.py` (`token_table`) | For each kept block: its `mn`, `sig`, `beh` tokens per instruction plus its `api` tokens, as one flat list with per-block offsets |
| Block vector | `pipeline/s05_block_counting/weights.py` + `encoder.py` | Per-block counting with TF-IDF, mean, raw, presence or log weighting, then a learned projection to 128 dimensions, plus the 6 structural features. Or precomputed vectors from a frozen encoder (`block_vecs.py`, paths 1 and 2) |

## 3. Selecting the graph and building the batch (every training run)

| Step | File | What it does |
|---|---|---|
| Subgraph selection | `pipeline/s04_select_subgraph/subgraph.py` (`select_subgraph`) | Whole binary by default (500k-block cap); optional `cap` / `lcc` policies at a block budget. Whole functions only, never uses labels, deterministic in evaluation |
| Block edges for the GNN | `pipeline/s06_gnn_block_cfg/edges.py` (`block_edges`) | Keeps edges inside kept functions; type id = `edge_kind * 2 + edge_back` (12 types); optional controls (shuffle, chain, no edges) |
| Call edges for the GNN | `pipeline/s08_gnn_call_graph/calls.py` (`call_pairs`) | Call edges between kept functions, no self-calls |
| Batch | `pipeline/batching.py` (`encode_binary_gnn`, `collate`) | Puts it together, shifts ids into one batch, applies training-time masking/dropping |

## 4. The data export for the transformer team

`transformer_paths/block_export.py` writes the same selected blocks, tokens, edges and folds to
`RUNS/export/<name>/` (schema in `transformer_paths/EXPORT_SCHEMA.md`). A teammate's encoder trained on that
export (rules in `transformer_paths/TRANSFORMER_CONTRACT.md`) produces block vectors that plug into step 5,
so all paths see identical blocks and folds.

To use a text model (CodeBERT or similar), build each block's text from the export's vocabulary strings
(for example `xor r,r ; mov r,m ; call kernel32!createfilew`) rather than from raw operands. That keeps
register names, immediates and addresses out, and it matches the blocks our counting path sees.
