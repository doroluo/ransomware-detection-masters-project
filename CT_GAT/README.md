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
