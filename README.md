# masters-project

## EMBER Model + LightGBM Decision Tree
In this branch, we are training, testing, and comparing the EMBER model to our CNN + Transformer & Natural Language Processing Models.
You can find more information about EMBER [here](https://github.com/elastic/ember). This branch takes in raw .exes as input.

This repository evaluates the adversarial robustness of static Portable Executable (PE) classifiers against feature-space benign mimicry attacks. The pipeline extracts a custom d = 537 feature vector directly from PE binaries.

### Directory Set Up
```
./goodware/*.exes
./ransomware/rans/* - Nested Ransomware Family directories with executables
```

### Dataset
- [Download the Goodware Dataset](https://data.mendeley.com/datasets/p3v94dft2y/3)
- [Download the Ransomware PE Header Dataset](https://data.mendeley.com/datasets/p3v94dft2y/3)


## Objectives
1. Realistic Threat Modeling: Simulate Targeted Feature-Space Mimicry by blending malleable feature regions (histograms, string stats, import padding) of ransomware vectors with paired goodware targets while strictly preserving rigid PE header invariants.

2. Defense Strategy Evaluation: Quantify the trade-offs between clean classification accuracy (Precision, Recall) and adversarial vulnerability (Evasion Rate) across Baseline LightGBM, Feature Pruning, and Adversarial Retraining.


### Dependencies
```bash
pip install lief lightgbm scikit-learn numpy
```

### Running Basic Extraction & Training
```python
python3 extract_and_train.py
```

### Running the Model Robustness & Defense Framework
A suite designed to extract EMBER static features from PE binaries and evaluate against adversarial evasion techniques.
The extractor and sanitizer files are run along with the train_and_defend file.
```python
python3 train_and_defend.py
```


### Model Comparison
| Metric / Dimension | Baseline LightGBM | Feature Pruning Defense | Adversarial Retraining |
| :--- | :--- | :--- | :--- |
| **Core Mechanism** | Standard gradient-boosted decision trees trained strictly on unperturbed clean data (`X_clean`). | Proactively strips all brittle byte/entropy/string features prior to training and testing. | Augments the training set with mild synthetic mimicry attacks (`X_adv_train`) before training. |
| **Feature Space Used** | All 537 features (Byte/Entropy Histograms + PE Headers + String Stats). | Reduced 21 features (Structural Headers & Section Metrics only; `0:512` and string features removed). | All 537 features, but decision trees dynamically adjust split weights. |
| **Clean Accuracy** | **Precision:** 96.97% <br> **Recall:** 97.96% | **Precision:** 98.95% <br> **Recall:** 95.92% | **Precision:** 97.00% <br> **Recall:** 98.98% |
| **Evasion Rate (&alpha; = 0.60)** | 46.94% (High Vulnerability) | 14.29% (Moderate Robustness) | 5.10% (High Robustness) |
| **Primary Trade-off** | Achieves high clean accuracy, but heavily over-indexes on easily manipulated byte distributions. | Dramatically improves robustness without complex data generation, but slightly lowers clean recall. | Achieves the highest overall resilience and recall, but requires generating realistic synthetic training attacks. |
