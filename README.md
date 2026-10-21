# Ransomware detection from IDA Pro control-flow graphs

A master's project that classifies Windows PE binaries as goodware (label `0`) or ransomware (label `1`) from what IDA Pro
recovers in them: the control-flow graph (CFG) of each function, its instructions and the imported APIs. The binaries are
never executed. IDA runs on an offline VM, and every later stage reads only the text and feature files exported there.

## Data

The binaries are the Mendeley ransomware dataset: https://data.mendeley.com/datasets/p3v94dft2y/3. The code reads the files
derived from it, which are on the team Google Drive:

| File | Contents | Read by |
|---|---|---|
| `Collections/Mendeley_Ransomware.zip`, `Collections/Mendeley_Goodware.zip` | per binary, the IDA CFG export `<sha256>.jsonl.gz` and the IDA listing `<sha256>.asm.gz`, in `samples/<split>/<family>/`; unzip into `CT_GAT/data/` | `preprocess_full.py`, `tokenize_asm_gz.py` |
| `Mendeley_GNN_v1.zip` | the same CFG exports as one bundle (`samples.csv`, `splits/`, `cfg/`) | the CFG cache (`s02_preprocess/cache`) |
| `Collections/Ransomware_Balanced.zip`, `Collections/Balanced_Goodware.zip` | two more collections in the same layout, for the all-architecture evaluation | the CFG cache with `--corpora` |
| `family_holdout_folds.zip` | the 5-fold family-holdout fold assignments of every published result | the CFG cache, and through it every `kfold` run |
| `Mendeley_x86_cfg_subgraphs_500blocks_v3.zip` | the output of `preprocess_full.py` for the 1,810 x86 binaries of the folds, 500 blocks per binary; unzip at `CT_GAT/` | Janice's five-channel model, Models 2 and 3 of the team diagram |
| `Mendeley_x86_cfg_subgraphs_uncapped_v3.zip` | the same with every non-library block of each binary | as above |

Each zip has a `.sha256` file beside it and a README inside it.

## Pipeline

<img width="741" height="750" alt="Team pipeline diagram" src="https://github.com/user-attachments/assets/dbb96697-ee9c-457b-a7c7-5f1022c492e9" />

The code is in `CT_GAT/pipeline`, one folder per stage of the diagram:

| Stage | Folder | What it does |
|---|---|---|
| 1. IDA export | `s01_ida_export/` | the IDAPython extractor that writes the `.jsonl.gz` CFG and the `.asm.gz` listing of a binary (runs on the VM) |
| 2. Preprocessing | `s02_preprocess/` | two routes. `preprocess_full.py` selects blocks from the CFG, aligns their instructions with the listing and writes one `.pt` sample per binary (five-channel token rows, 77 block features, 64-d block embeddings, edges). The CFG cache (`cache/`) reads only the `.jsonl.gz` exports and keeps the whole graph. |
| 3a. Model 1 | `s03a_transformer/` | transformers over the instruction sequence: `token_sequence/` (on the CFG cache) and Janice's `five_channel/` (on the `.pt` samples) |
| 3b. Model 2 | `s03b_cfg_feature/` | a hierarchical edge-typed GNN over the block CFG and the call graph, on block token counts |
| 3c. Model 3 | `s03c_cfg_transformer/` | frozen block encoders (CodeBERT) whose block vectors replace the token counts as the GNN's input |
| 4. Evaluation | `s04_evaluation/` | baselines, metrics, validation thresholds, paired bootstrap comparisons and result tables |

`experiments/` holds the one training command (`experiments.train`), the experiment plans and their runner
(`experiments.queue`), and adapters that train the teammates' GAT pipelines on the same folds. Labels are binary; the
behaviour tags of the five-channel tokens (encrypt, file read, ...) describe what an instruction does, and models can use
them to point at suspicious regions.

## Repository layout

```
README.md                         this file
meetings/                         team and advisor meeting notes
.github/workflows/                CI: the pipeline tests on every push and pull request
CT_GAT/
    data/                         the unzipped collections (gitignored)
    processed/                    the output of preprocess_full.py (gitignored)
    pipeline/                     all the code; see CT_GAT/pipeline/README.md
        s01_ida_export/  s02_preprocess/  s03a_transformer/  s03b_cfg_feature/  s03c_cfg_transformer/  s04_evaluation/
        common/  experiments/  docs/  tests/
        requirements.txt  datasets.json.example
```

## Setup

Python 3.12:

```bash
cd CT_GAT/pipeline
pip install -r requirements.txt
```

This installs the CPU build of PyTorch. To train on a GPU, install the CUDA build from https://pytorch.org first.
`requirements.txt` lists the optional packages (CodeBERT, the teammates' GAT pipelines) and what needs them.

## Running

Every command runs from `CT_GAT/pipeline`. There are two starting points.

**From the preprocessed zips** (no IDA, no CFG cache). Unzip one of the `Mendeley_x86_cfg_subgraphs_*` archives at
`CT_GAT/`. If you take the files from the Drive one by one instead, put them here:

| File | Goes to |
|---|---|
| `api_vocab.json` | `CT_GAT/processed/api_vocab.json` (the API vocabulary; the five-channel demo and `preprocess_full.py` read it there) |
| `<sha256>.pt` (one per binary) | `CT_GAT/processed/cfg_subgraphs/` |
| `channel_embedder.pt` | `CT_GAT/processed/cfg_subgraphs/` |
| `<sha256>.manifest.json` (500-block zip only) | `CT_GAT/processed/cfg_subgraphs/` (alignment report per binary; no model reads it) |
| `build_info/` | `CT_GAT/processed/cfg_subgraphs/build_info/` (per-binary index and build settings; no model reads it) |

Use one build at a time. Each zip's `.pt` files carry API ids of its own `api_vocab.json` (12,406 names in the 500-block
build, 12,788 uncapped), so the files of the two zips must not be mixed. Then:

```bash
cd CT_GAT/pipeline
python -m s03a_transformer.five_channel.model --batch-size 4
```

This runs Janice's five-channel Model 1 (untrained) over every sample and prints its predictions. `--data-dir` reads
another folder of `.pt` files; the API vocabulary is always read from `CT_GAT/processed/api_vocab.json`.

To build the samples yourself, unzip `Collections/Mendeley_Ransomware.zip` and `Collections/Mendeley_Goodware.zip` into
`CT_GAT/data/` and run `preprocess_full.py`. It processes every binary of the collections in `CT_GAT/data/`
(`--architecture x86` keeps the x86 ones; the zips hold the 1,810 x86 binaries of the folds, built with
`experiments/team_pipelines/build_b.py`) and writes to `CT_GAT/processed/` by default (`--output` and `--vocab` change
that):

```bash
python s02_preprocess/preprocess_full.py --sample signal --num-blocks 500 --seed 0 --architecture x86   # 500 blocks
python s02_preprocess/preprocess_full.py --sample signal --num-blocks 0 --seed 0 --architecture x86     # uncapped
python s02_preprocess/preprocess_full.py --limit 2 --num-blocks 32 --seed 0                            # a quick try
```

The options (sampling, library functions, caps, resuming) are in
[`CT_GAT/pipeline/s02_preprocess/README.md`](CT_GAT/pipeline/s02_preprocess/README.md).

**The whole pipeline** (training and evaluating every model on the folds). It reads the CFG bundle and the fold files,
not the `CT_GAT/data/Mendeley_*` folders (those are the input of `preprocess_full.py` only). Unzip `Mendeley_GNN_v1.zip`
and `family_holdout_folds.zip` anywhere and write `CT_GAT/pipeline/datasets.json` (gitignored) with your paths:

```json
{"x86": {"cache": "D:/gt/cache_x86", "protocols": ["bundle", "kfold"],
         "sources": {"bundle": "D:/gt/Mendeley_GNN_v1",
                     "folds": "D:/gt/folds/family_holdout/folds_mendeley.csv",
                     "folds_v3": "D:/gt/folds/family_holdout_v3",
                     "runs": "D:/gt/runs"}}}
```

`bundle` is the folder that holds `samples.csv`; the cache build reads both fold sources, also for x86 alone. Then build
the cache once, and train and evaluate:

```bash
python -m s02_preprocess.cache.build_cache                       # the CFG cache (CPU, once)
python -m s04_evaluation.baselines --protocol kfold --fold 0 1 2 3 4 --arch x86
python -m experiments.train --protocol kfold --fold 0 --model gnn --mask-blocks 0.15     # Model 2, one fold
python -m experiments.queue --plan experiments/plans/day1.txt    # a whole experiment: baselines and five GNN arms
python -m s04_evaluation.report                                  # RESULTS.md over all runs
```

The plans `v3` and the other `kfold_v3` experiments need the all-architecture cache as well (the `collections` source
and a second build; see the pipeline README); the x86 plans (`day1`, `model1`, `noapi`, ...) need only this one.
[`CT_GAT/pipeline/README.md`](CT_GAT/pipeline/README.md) has every step: the data settings, Models 1 to 3, the
teammates' pipelines, the experiment plans, the comparison statistics and where each output goes.

The tests run on any machine without the data:

```bash
cd CT_GAT/pipeline
python -m pytest -m "not real_data and not gpu"
```

## Results

5-fold family holdout on the 1,810 Mendeley x86 binaries, 3 seeds; area under the ROC curve (AUC):

* Our edge-typed GNN over the whole CFG (R7) reaches 0.979. A logistic regression on the PE import table (0.985), TF-IDF over
  the instruction tokens (0.982) and the same GNN with no edges and no API names (0.981) are level with it, so the CFG
  structure adds nothing measurable over the tokens and imports.
* Pipeline B (`preprocess_full.py`) with the uncapped samples: Model 3 (block embeddings and CFG) 0.981, level with R7;
  Model 2 (block features and CFG) 0.970. At 500 blocks per binary both are lower (0.966 and 0.954).
* Model 1: the token-sequence transformer 0.970; Janice's five-channel model 0.919 to 0.926, since it reads only the first
  128 or 512 instructions of each binary.

The tables, their confidence intervals and the decision rule are in
[`CT_GAT/pipeline/docs/history/`](CT_GAT/pipeline/docs/history/README.md).
