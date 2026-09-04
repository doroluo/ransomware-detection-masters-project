## Dataset
https://data.mendeley.com/datasets/p3v94dft2y/3

## asm_parse.py
Converts executable files in a directory into `.asm` files. A limit can be set for the number of files processed.
### Usage
```python3 asm_parse.py```

## asm_parser.py
Takes existing .asm files and convert it into 256x256 greyscale image with ViT attention mask.
1. Parse each instruction into (opcode/operand/API) using token map
2. Flatten to 65536 tokens, truncate and padding to fit
3. Save .png and vitmask.npy under Goodware & Ransomware folder
4. Sort by labels

### Structure & Definition
.png - 256x256 grayscale image. Each pixel is a token ID from the asm file(opcode/operand/API), padded or truncated to fit.
_vit_mask.py - 16x16 array read code(1) and padding(0)

    training_dataset_sorted/

        ├── Class_0_Goodware/     # 42 samples right now

            ├── <name>.png

            └── <name>_vit_mask.npy

        └── Class_1_Ransomware/   # 50 samples
    
            ├── <hash>.png

            └── <hash>_vit_mask.npy

### Usage
```python3 asm_parser.py```

## stratified_split.py
Spliting the training_dataset_sorted from asm_parser.py into train/val/test while keeping the same ratio. Copies each .png and vit_mask.npy. 

### Structure
    evaluation_dataset_split/

        ├── test/

            ├── Class_0_Goodware

                └── <name>.png

                └── <name>_vit_mask.npy

            ├── Class_1_Ransomware
                
                └── <hash>.png

                └── <hash>_vit_mask.npy

        ├── train/

            ├── Class_0_Goodware

                └── <name>.png

                └── <name>_vit_mask.npy

            ├── Class_1_Ransomware
                
                └── <hash>.png

                └── <hash>_vit_mask.npy        
        
        ├── val/

            ├── Class_0_Goodware

                └── <name>.png

                └── <name>_vit_mask.npy

            ├── Class_1_Ransomware
                
                └── <hash>.png

                └── <hash>_vit_mask.npy
    
### Usage
```python3 stratified_split.py```

## model_train.py
Train the CNN-ViT malware classifier:
1. Loads .png + _vit_mask.npy from train / val / test
2. Trains HierarchicalMalwareNet (CNN stem → transformer → classifier)
3. Uses early stopping on validation loss
4. Evaluates on the test set (accuracy, report, confusion matrix)
5. Saves best_model.pth and clean_model_early_stop.pth

### Dependency
```pip install scikit-learn```
```python3 -m pip install torch torchvision```

### Usage
```python3 model_train.py```