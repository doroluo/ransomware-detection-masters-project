## Dataset
https://data.mendeley.com/datasets/p3v94dft2y/3

Binary goodware vs ransomware. Labels: `0` = Benign, `1` = Ransomware.

### Pipeline overview
```
goodware/ + ransomware/
        ↓
prepare_dataset.py   (1) exes -> .asm
        ↓              (2) .asm + labels -> metadata.csv
data/asm/ + data/metadata.csv        ← shared by Transformer & GAT
        ↓
build_sequences.py (+ token_mapping.py) ← Transformer path
        ↓
processed/sequences/*.pt
        ↓
create_split.py → data_loader.py → transformer_model.py → train.py → evaluate.py
```

Run scripts from `CT_GAT/preprocessing/`.

---

## prepare_dataset.py
Shared front-end for Transformer and GAT. Prompts for mode and sample limit (or use CLI flags).
1. Mode **1** — Capstone-disassemble `goodware/` + `ransomware/` into `CT_GAT/data/asm/*.asm`
2. Mode **2** — Match `.asm` stems to exe folders and write `data/metadata.csv` with labels
3. Optional **samples per class** limit for both modes

### Structure
    CT_GAT/data/asm/

        └── <file_id>.asm

    CT_GAT/data/metadata.csv

        columns: file_id, exe_path, asm_path, label

### Dependency
```pip install pefile capstone```

### Usage
```python3 prepare_dataset.py```
```python3 prepare_dataset.py --mode 1 --limit 50```
```python3 prepare_dataset.py --mode 2 --limit 50```

## token_mapping.py
Token maps and instruction encoding for the transformer. Each instruction becomes `[opcode_or_api, operand1, operand2]`.
1. Map opcode (or known API under CALL) with `TOKEN_MAP` / `API_MAP`
2. Categorize operands (`REG_DATA`, `REG_STACK`, `MEM_*`, `IMM_VALUE`, etc.)
3. `parse_capstone_insn(mnemonic, op_str)` for Capstone output
4. `parse_asm_file` / `parse_asm_line` for existing `.asm` text

### Usage
```python3 token_mapping.py path/to/file.asm```

## build_sequences.py
Batch-parse ASM files from `metadata.csv` into transformer-ready `.pt` sequences.
1. Read `CT_GAT/data/metadata.csv`
2. Call `token_mapping.parse_asm_file` on each `asm_path`
3. Save `{file_id, sequence: LongTensor[N, 3], label, asm_path}`

### Structure
    CT_GAT/processed/sequences/

        └── <file_id>.pt

### Usage
```python3 build_sequences.py```

## check_sequences.py
Sanity-check saved `.pt` sequences: count, shape, dtype, and first / last instructions of one sample.

### Usage
```python3 check_sequences.py```

## create_split.py
Stratified train / validation / test split over `processed/sequences/*.pt`. Writes CSVs with `file_id`, `file_path`, `label` while keeping class balance.

### Structure
    CT_GAT/processed/splits/

        ├── train.csv

        ├── validation.csv

        └── test.csv

### Usage
```python3 create_split.py```

## data_loader.py
PyTorch `Dataset` / `DataLoader` for opcode sequences.
1. Load split CSVs and each sample’s `.pt` sequence
2. Truncate / pad every sequence to length 2048 → shape `[2048, 3]`
3. Batch size 8 → `[8, 2048, 3]` with `(sequence, label, length)`

### Usage
```python3 data_loader.py```

## transformer_model.py
Defines `OpcodeTransformer`: encoder-only transformer for binary malware classification.
1. Embed opcode/API + two operand channels; add position embeddings
2. `TransformerEncoder` (2 layers, 4 heads, `d_model=64`) with padding mask
3. Mean-pool real tokens, then linear head → 2 logits (Benign / Ransomware)

### Usage
```python3 transformer_model.py```

## train.py
Train the opcode transformer classifier:
1. Load train / val loaders from `data_loader.py`
2. Train `OpcodeTransformer` with AdamW + CrossEntropyLoss
3. Track validation loss / accuracy each epoch
4. Save best weights to `processed/checkpoints/best_transformer.pt`

### Dependency
```pip install numpy pandas scikit-learn```
```pip install torch --index-url https://download.pytorch.org/whl/cpu```

### Usage
```python3 train.py```

## evaluate.py
Evaluate the best checkpoint on the held-out test set:
1. Load `processed/checkpoints/best_transformer.pt`
2. Run inference on `test_loader`
3. Print accuracy, classification report, and confusion matrix

### Usage
```python3 evaluate.py```

---

