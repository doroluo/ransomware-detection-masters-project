## Dataset
https://data.mendeley.com/datasets/p3v94dft2y/3

Binary goodware vs ransomware. Labels: `0` = Benign, `1` = Ransomware.

The binary label stays benign / ransomware. Behavior tags describe **what a stretch of code is doing** (encrypt, file read, …) so models can point at suspicious regions.

### Pipeline overview
```
goodware/ + ransomware/
        ↓
prepare_dataset.py   (1) exes → .asm  (resolve call/jmp to API names)
        ↓              (2) .asm + labels → metadata.csv
data/asm/ + data/metadata.csv
        ↓
token_mapping.py     [opcode_or_api, op1, op2, behavior]
        ↓
   ┌────┴───────────────────┐
   ↓                        ↓
build_sequences.py          build_pyg_graphs.py
LongTensor[N, 4]            CFG + loop/call edges + block counts
   ↓                        ↓
create_split.py ──────────── shared train / val / test ids
   ↓                        ↓
OpcodeTransformer           OpcodeGAT                 ← baseline
BehaviorOpcodeTransformer   BehaviorOpcodeGAT         ← behavior-aware
   ↓
compare_models.py           ransomware F1 + sample explanation
```

Run scripts from `CT_GAT/preprocessing/`.

### Quick start (e.g. 100 per class)
```bash
cd CT_GAT/preprocessing
python3 prepare_dataset.py --mode 1 --limit 100
python3 prepare_dataset.py --mode 2 --limit 100
python3 build_sequences.py --limit 200
python3 build_pyg_graphs.py --limit 200 --overwrite
python3 create_split.py          # uses whatever is in processed/sequences/
python3 compare_models.py --epochs 10
```

`--limit` on prepare is **per class**. Use ~2× that on the builders so both classes are covered. Keep `processed/sequences/` limited to the samples you want in the split (old leftover `.pt` files are included by `create_split.py`).

---

## prepare_dataset.py
Shared front-end for Transformer and GAT.
1. Mode **1** — Capstone-disassemble `goodware/` + `ransomware/` into `data/asm/*.asm`  
   Resolves IAT / MSVC thunks so lines look like `0x00401234:  call ReadFile` (no `;` comment).
2. Mode **2** — Match `.asm` stems to exe folders → `data/metadata.csv`
3. Optional samples-per-class limit

### Structure
```
CT_GAT/data/asm/<file_id>.asm
CT_GAT/data/metadata.csv    # file_id, exe_path, asm_path, label
```

### Dependency
```bash
pip install pefile capstone
```

### Usage
```bash
python3 prepare_dataset.py
python3 prepare_dataset.py --mode 1 --limit 50
python3 prepare_dataset.py --mode 2 --limit 50
```

---

## token_mapping.py
Each instruction → `[opcode_or_api, operand1, operand2, behavior]`.

1. Opcode / known API (`TOKEN_MAP`, `API_MAP`) — `call ReadFile` remaps channel 0 to the API id  
2. Operand categories (`REG_DATA`, `MEM_*`, `IMM_VALUE`, …)  
3. **Behavior** (`BEHAVIOR_MAP`): `NONE`, `CRYPTO_OPCODE`, `DECRYPT`, `ENCRYPT`, `KEY_SETUP`, `FILE_ENUM`, `FILE_READ`, `FILE_WRITE`, `FILE_DELETE`  
4. `parse_capstone_insn` / `parse_asm_line` / `parse_asm_file`

```bash
python3 token_mapping.py path/to/file.asm
```

---

## build_sequences.py
`metadata.csv` → `processed/sequences/<file_id>.pt`  
`{file_id, sequence: LongTensor[N, 4], label, asm_path}`

```bash
python3 build_sequences.py
python3 build_sequences.py --limit 200
```

## build_pyg_graphs.py
`.asm` → CFG graph `.pt` with:
- `token_ids [N, 4]`
- `insn_to_block`, `edge_index`
- `edge_type` — `0=cfg`, `1=loop`, `2=call`
- `block_crypto_count`, `block_file_count`

```bash
python3 build_pyg_graphs.py --overwrite
python3 build_pyg_graphs.py --limit 200 --overwrite
```

## check_sequences.py / create_split.py
Sanity-check sequences; stratified 70/15/15 split → `processed/splits/{train,validation,test}.csv`.

---

## data_loader.py
Both paths use a **2048-instruction budget** as `4 × 512` windows over the **full** ASM (not only the file head).

| Loader | How windows are chosen | Shape |
|--------|-------------------------|--------|
| **Baseline** | Evenly spaced contiguous windows (adaptive stride) | `[4, 512, 3]` |
| **Behavior** | Densest crypto/file regions (full-file scan) | `[4, 512, 4]` |

```bash
python3 data_loader.py
```

---

## transformer_model.py
- **`OpcodeTransformer`** — baseline; embeds 3 channels; mean-pools across stride windows → 2 logits  
- **`BehaviorOpcodeTransformer`** — embeds behavior; scores each window; takes the best window for the binary label; behavior head for encrypt / decrypt / file_read / file_write  

## gat_model.py
- **`OpcodeGAT`** — baseline; mean-pool instructions → blocks → graph  
- **`BehaviorOpcodeGAT`** — behavior embeddings, typed edges, crypto/file counts; **mean-pool** for the graph label; per-block malware + behavior heads for explanation; `find_read_to_write_paths` via CFG  

Training notes for GAT behavior: binary CE is primary; behavior BCE (and light block CE) only on tagged blocks, weight `0.1`, with early stopping.

---

## train.py / evaluate.py
Baseline transformer only → `processed/checkpoints/best_transformer.pt`

## train_gat.py / evaluate_gat.py
Baseline GAT only → `processed/checkpoints/best_gat.pt`

## compare_models.py
Trains all four models on the same split, prints **ransomware F1**, then explains one test ransomware sample (top windows, top blocks, read→write paths).

```bash
python3 compare_models.py --epochs 10
python3 compare_models.py --skip-train
python3 compare_models.py --skip-train --sample-id <file_id>
```

Checkpoints:
```
processed/checkpoints/best_transformer.pt
processed/checkpoints/best_transformer_behavior.pt
processed/checkpoints/best_gat.pt
processed/checkpoints/best_gat_behavior.pt
```

### Dependencies
```bash
pip install numpy pandas scikit-learn
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric pefile capstone
```

---
