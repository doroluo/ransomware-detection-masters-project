# masters-project

## CFG Autotokenizer
Tokenizes the CFG blocks generated IDA Pro. The CFG zip file is located [here](https://drive.google.com/drive/folders/19bufYY6xL3eUWbv0abOYgvC_wi5Ak1eO?usp=drive_link).

The project uses the Hugging Face Transformers (microsoft/codebert-base) to extract contextual embeddings from basic blocks and into feature vectors.

### Install Dependencies
```bash
pip install torch torch-geometric transformers
```

### Run Autotokenizer & Graph Processing
Make sure the CFG Files are in your project directory. They should be formatted as:
```
root_dir / cfg / {goodware, ransomware} / {train, test} / *.jsonl
```

Then run:
```bash
python construct_blocks.py
```

### Inspect a Processed Graph
```bash
python inspect_graph.py
```
