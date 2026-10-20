# Block dataset export

An export is the set of basic blocks of the binaries in one cache, with the fold structure of the GNN experiments, so
that a per-block encoder can be trained and its vectors plugged back into the GNN on the very same blocks and folds.
It is made by

```bash
python -m s03c_cfg_transformer.block_export --protocol kfold --arch x86 --out RUNS/export/ours_cap0
```

(run from `CT_GAT/pipeline`; `RUNS` is the run directory, `GT_RUNS` or `common/paths.py`, and exports live under `RUNS/export/` only, or under `GT_EXPORT`). The reference
export `ours_cap0` has 1810 x86 binaries, 17,603,969 blocks, 282,998,296 tokens and 2.7 GiB. Read it with
`common/reader.py`; the rules for the encoder you hand back are in `TRANSFORMER_CONTRACT.md`.

## What a block is

A block is one basic block of a binary's CFG. The blocks are the ones the GNN sees in evaluation:
`select_subgraph(policy 'cap', budget 0, rng None)` over the non-empty functions, which is every block of every
binary here (the largest binary has 201,944 blocks, under the cap of 500,000; `n_binaries_block_capped` in
`store.json` is 0). A block is a sequence of tokens in global vocabulary ids, laid out as `block_tokens.token_table` lays
it out: all the mnemonic tokens of its instructions, then all their signature tokens, then all their behaviour tags,
then its API references: `[mn_1 .. mn_n | sig_1 .. sig_n | beh_1 .. beh_n | api_1 .. api_m]`. `tok_kind` gives the kind
of every token (0 mn, 1 sig, 2 beh, 3 api), so an encoder can regroup them (the i-th mnemonic, signature and tag belong
to instruction i).

**Truncation.** At most 192 tokens per block. A block with `3 n + m > 192` (n instructions, m API references) keeps its
first instructions with at most 160 tokens (3 per instruction) and its last units with at most 32 tokens, where the
units of a block are its instructions in order followed by its API references (1 token each); a head that holds every
instruction continues into the API references. Cuts fall between units, never inside an instruction. The kept tokens
keep the layout above. About 2% of the blocks (346,459 in the reference export) are empty: no instruction and no API reference, hence
no token (`blk_tok_ptr[k] == blk_tok_ptr[k+1]`); they all share one block key, and an encoder has to give them a finite
vector. Truncated blocks are flagged in `blk_truncated`; the reference export cuts 46,452 of 17,603,969
blocks. The rule is also written in `store.json` (`truncation_rule`).

## Layout

```
<export>/
  store/                     fold-independent, global vocabulary ids
    store.json
    binaries.json
    <arrays>.npy
  fold0/ .. fold4/           one directory per fold
    manifest.json  lut.npy  vocab.json  block_key.npy
```

### store/

`B` binaries, `N` blocks, `F` functions, `T` tokens, `E` CFG edges, `C` call edges (reference export: B 1810,
N 17,603,969, F 1,311,743, T 282,998,296, E 23,969,543, C 3,545,862).

| file | dtype | shape | content |
|---|---|---|---|
| `store.json` | json | | format, protocol, arch, cache path, `vocab_sha` (vocabulary of the CFG cache), policy and budget, `max_tokens_per_block` 192, `head_tokens` 160, `tail_tokens` 32, `truncation_rule`, `token_layout`, global vocabulary sizes and offsets, counts, `code_sha` (sha256 of the exporting code), `git_head`, `export_sha` (sha256 over every array and fold file, the id that checkpoints carry) |
| `binaries.json` | json | list of B | per binary: `index`, `sample_id`, `label` (1 ransomware), `arch`, `family`, `identity_group`, `corpus`, `fold`, `n_blocks_total`, `n_blocks_kept` |
| `bin_blk_ptr` | int64 | [B+1] | blocks of binary b are `bin_blk_ptr[b] : bin_blk_ptr[b+1]` |
| `bin_fn_ptr` | int64 | [B+1] | functions of binary b, same way |
| `bin_cfg_ptr` | int64 | [B+1] | CFG edges of binary b, same way |
| `bin_call_ptr` | int64 | [B+1] | call edges of binary b, same way |
| `blk_tok_ptr` | int64 | [N+1] | tokens of block k are `tokens[blk_tok_ptr[k] : blk_tok_ptr[k+1]]` |
| `tokens` | int32 | [T] | global vocabulary ids, layout above |
| `tok_kind` | int8 | [T] | kind of each token: 0 mn, 1 sig, 2 beh, 3 api |
| `blk_fn` | int32 | [N] | store function id of each block (an index into `fn_flags`, `fn_cache_id`) |
| `blk_cache_id` | int32 | [N] | the block's index inside its sample in the CFG cache (ascending within a binary) |
| `blk_feat` | float16 | [N, 6] | structural block features of the cache (log counts, in/out degree) |
| `blk_truncated` | int8 | [N] | 1 if the block was cut at 192 tokens |
| `fn_flags` | int8 | [F] | bit 0 library (FLIRT), bit 1 thunk |
| `fn_cache_id` | int32 | [F] | the function's index inside its sample in the CFG cache |
| `cfg_edges` | int32 | [E, 2] | (source, target) store block ids; only edges inside one function |
| `edge_type` | int8 | [E] | `edge_kind * 2 + back` as the GNN uses it, in 0..11 |
| `call_edges` | int32 | [C, 2] | (caller, callee) store function ids; unique pairs, no self calls |
| `bin_feat` | float32 | [B, 5] | log1p of the binary's counts: blocks, functions, instructions, CFG edges, call edges |

The global vocabulary has 16,464 ids in four consecutive ranges: mn 0..608, sig 609..1945, beh 1946..1954, api
1955..16463 (`global_offsets` in `store.json` and `vocab.json`). It is the vocabulary of the CFG cache and does not
depend on the fold.

### fold{k}/

| file | dtype | shape | content |
|---|---|---|---|
| `manifest.json` | json | | `fold`, `seed` 0, `val_seed` 0, `train`, `val`, `test` (lists of store binary indices), `val_families`, `manifest_sha`, `n_blocks`, `n_distinct_blocks` |
| `lut.npy` | int32 | [16464] | global id -> fold id |
| `vocab.json` | json | | `specials`, `n_ids`, `min_df`, the `offset`, `size` and `unk` id of each kind, `lut_sha` |
| `block_key.npy` | int64 | [N] | blake2b-64 (little-endian, signed) of the block's fold-id sequence after the cut and the truncation, as int32 bytes |

The manifest is `make_manifest(seed 0, val_seed 0)` of the GNN experiments: the test fold is `fold`, the training pool is
the other four folds, and about 15% of its whole ransomware families and goodware identity groups are held out as
`val`. `train`, `val` and `test` partition the store.

The **fold vocabulary** is `TokenVocab(manifest.train)` shifted by 4. Ids 0..3 are the specials PAD 0, CLS 1, SEP 2,
MASK 3 (the store contains none of them; they are for your use). Then come the kinds in the order mn, sig, beh, api,
each as its UNK id followed by the tokens that occur in at least 2 training binaries (`min_df`); every other token of a
kind maps to that kind's UNK. The behaviour tags are all kept and have no UNK. Fold 0 has 5,610 ids. A token's fold id
is `lut[global id]`.

`block_key` identifies a block by its fold-id sequence: two blocks with the same key have the same sequence (the export
checks that no two blocks of different length share a key). Fold 0 has 620,320 distinct keys over 17,603,969 blocks.
The block vectors of a frozen encoder are stored once per key (`precompute_blockvec.py`).

## Reading it

`reader.py`: `BlockStore(root)` (the store arrays as memory-mapped attributes, `binaries`, `fold(k)`),
`FoldView(store, fold, split)` with split `train`, `val`, `test` or `all` (`binaries()`, `blocks()`,
`distinct_blocks()`, `sequence(block)`, `padded(blocks)`) and `collate_blocks(seqs, pad=0, max_len=None)` returning
`(LongTensor [B, L], BoolTensor [B, L])`, the mask being True on tokens.

Iterating the training blocks of fold 0, one block per distinct token sequence:

```python
import torch
from common.reader import BlockStore, FoldView, collate_blocks

store = BlockStore('<runs>/export/ours_cap0')      # GT_RUNS, or GT_EXPORT
view = FoldView(store, fold=0, split='train')
block_ids, keys = view.distinct_blocks()                    # one store block id per distinct block_key
embed = torch.nn.Embedding(view.data['vocab']['n_ids'], 32, padding_idx=0)   # stand-in for your encoder
for lo in range(0, len(block_ids), 256):
    seqs = [view.sequence(b) for b in block_ids[lo:lo + 256]]       # fold-vocabulary ids, int32
    tokens, mask = collate_blocks(seqs, pad=0, max_len=192)         # [B, L] long, [B, L] bool (True = token)
    h = (embed(tokens) * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)   # [B, 32]
    loss = h.pow(2).mean()                                          # your loss goes here
    loss.backward()
```

Train on `split='train'`, select on `split='val'`, and do not read `test` (TRANSFORMER_CONTRACT.md).
