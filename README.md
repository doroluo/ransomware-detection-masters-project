# THIS NEEDS TO BE UPDATED!!!

## Dataset
https://data.mendeley.com/datasets/p3v94dft2y/3

Binary goodware vs ransomware. Labels: `0` = Benign, `1` = Ransomware.

The binary label stays benign / ransomware. Behavior tags describe **what a stretch of code is doing** (encrypt, file read, …) so models can point at suspicious regions.

### Pipeline overview
```
```

---

### Structure
```
```

### Dependencies
```bash
pip install pefile capstone
pip install numpy pandas scikit-learn
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric pefile capstone
```

---

### Preprocessing

Running 1 sample:
```
python preprocess_full.py \
    --num-blocks 32 \
    --max-instructions 128 \
    --limit 1 \
    --seed 0
```

Running 20 samples: 10 ransomware, 10 goodware
```
python preprocess_full.py \
  --num-blocks 32 \
  --max-instructions 128 \
  --limit 20 \
  --seed 42
```
---

### Model 1: Token Sequencing - CT_GAT/pipeline/s03a_transformer/token_sequence.py

How to run:
```
cd ~/project/ransomware-detection-masters-project/CT_GAT
python -m pipeline.s03a_transformer.token_sequence
```

How to run with customizable batch sizes (if no arg is provided, default is 4):
```
python -m pipeline.s03a_transformer.token_sequence \
  --data-dir processed/ \
  --batch-size 1 \
  --device cpu
```

Output:
- Logits Shape: [a, 2]
    - a: The batch size/Number of samples evaluated
    - 2: Number of target classes (0 = Goodware, 1 = Ransomware)

- Predictions: [n, n, n, n...] where n is 1 or 0

- Target Label: 0 or 1, determines if the input is goodware or ransomware

