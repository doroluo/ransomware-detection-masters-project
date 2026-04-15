# masters-project

## EMBER Model + LightGBM Decision Tree
In this branch, we are training, testing, and comparing the EMBER model to our CNN + Transformer & Natural Language Processing Models.
You can find more information about EMBER [here](https://github.com/elastic/ember). This branch takes in raw .exes as input.

### Directory Set Up
```
./goodware/*.exes
./ransomware/rans/* - Nested Ransomware Family directories with executables
```

### Dataset
[Download the Goodware Dataset](https://data.mendeley.com/datasets/p3v94dft2y/3)
[Download the Ransomware PE Header Dataset](https://data.mendeley.com/datasets/p3v94dft2y/3)

### Dependencies
```python
pip install lief lightgbm scikit-learn numpy
```

### Running
```python
python3 extract_and_train.py
```

