# Ransomware detection from IDA Pro control-flow graphs

A master's project that classifies Windows PE binaries as goodware (label `0`) or ransomware (label `1`) from what IDA Pro
recovers in them: the control-flow graph (CFG) of each function, its instructions and the imported APIs. The binaries are never
executed. IDA runs on an offline VM, and every later stage reads only the files exported there. This page says how to run the
code; the stages, the settings and the evaluation are in [`CT_GAT/pipeline/README.md`](CT_GAT/pipeline/README.md).

## Setup

Python 3.12. From the repository root:

```bash
pip install -r CT_GAT/pipeline/requirements.txt
```

This installs the CPU build of PyTorch and `torch_geometric`. For training on a GPU, install the CUDA build of torch 2.11.0 from
https://pytorch.org first (`requirements.txt` pins that version and would replace another). Models 2 and 3 and full Model 1 runs
want a CUDA GPU; the smoke runs below use `--device cpu`.

## Data

The binaries are the Mendeley ransomware dataset (https://data.mendeley.com/datasets/p3v94dft2y/3). The code reads files derived from
it, which are on the team Google Drive, each with a `.sha256` file beside it. The first two are required; each preprocessed zip is
optional and replaces a build on the CPU: about 4 h for the 500-block samples and 5.6 h for the uncapped ones.

| File | Size | sha256 | Contents |
|---|---|---|---|
| `Collections/Mendeley_Ransomware.zip` | 5.0 GB | `c3785afff453792b007eee3f736f725df926f90c34638af88d4feb19772ada9c` | the IDA exports of the ransomware binaries. Unzip into `CT_GAT/data/` |
| `Collections/Mendeley_Goodware.zip` | 1.5 GB | `33afd5dc007b1e551cd657209961a68aebbd975cecba9db266e7f6af48cf9cc1` | the same for the goodware binaries. Unzip into `CT_GAT/data/` |
| `Mendeley_x86_cfg_subgraphs_500blocks_v3.zip` | 1.4 GB | `fe8a1c4fc93aaf0913c9992ab0f832738cd8c32095524d8cc10ee390237771b3` | the preprocessed samples of Models 1 to 3, `signal` sampler, 500 blocks per binary. Unzip at `CT_GAT/` |
| `Mendeley_x86_cfg_subgraphs_signal_250blocks_v3.zip` | pending | pending | the same, `signal`, 250 blocks. Unzip at `CT_GAT/` |
| `Mendeley_x86_cfg_subgraphs_callwalk_500blocks_v3.zip` | pending | pending | the same, `callwalk` (a walk across calls), 500 blocks |
| `Mendeley_x86_cfg_subgraphs_callwalk_250blocks_v3.zip` | pending | pending | the same, `callwalk`, 250 blocks |
| `Mendeley_x86_cfg_subgraphs_uncapped_v3.zip` | 2.6 GB | `4b51e3238398967ad891647364fbe815fcce5c8f90f5d33249fcae29c9564e8d` | every non-library block of each binary, which `signal` and `callwalk` both give. Unzip at `CT_GAT/` |

The sizes and checksums are those of the files on the Drive; "pending" zips are packaged with `package_build.py` once their build ends.
Unpacked, the collection zips take about their zip size, the 500-block zip 9.9 GB and the uncapped one 13.3 GB; the CFG cache below
takes 2.6 GB. Unzip a preprocessed zip over an empty `CT_GAT/processed/`: the folder holds one build at a time, and the zips have
different API vocabularies. The balanced collections of the all-architecture cache are in the pipeline README.

Each collection zip holds one top-level folder named like the zip, so the destination is `CT_GAT/data` itself; a preprocessed zip
holds `processed/` and a README, so its destination is `CT_GAT`. Run these from the repository root, with `<path to>` the folder of
the downloaded zips:

```bash
# Windows, PowerShell or cmd (tar is the one that comes with Windows and reads zips)
mkdir CT_GAT\data
tar -xf <path to>\Mendeley_Ransomware.zip -C CT_GAT\data
tar -xf <path to>\Mendeley_Goodware.zip -C CT_GAT\data
tar -xf <path to>\Mendeley_x86_cfg_subgraphs_500blocks_v3.zip -C CT_GAT

# macOS and Linux (on macOS, tar -xf <zip> -C CT_GAT/data also works)
mkdir -p CT_GAT/data
unzip <path to>/Mendeley_Ransomware.zip -d CT_GAT/data
unzip <path to>/Mendeley_Goodware.zip -d CT_GAT/data
unzip <path to>/Mendeley_x86_cfg_subgraphs_500blocks_v3.zip -d CT_GAT
```

The result is `CT_GAT/data/Mendeley_Ransomware/samples.csv`, `CT_GAT/data/Mendeley_Goodware/samples.csv`,
`CT_GAT/processed/api_vocab.json` and `CT_GAT/processed/cfg_subgraphs/`. Git Bash on Windows has `unzip` but its `tar` cannot read
zips. In Windows Explorer, "Extract All" must be given the destination above, not the folder it proposes, or the files end up one
folder too deep. Nothing else is configured: the fold assignments are in the repository, and caches and runs go inside the clone.

## Run

Every command below runs from `CT_GAT/pipeline` (`cd CT_GAT/pipeline`). Run outputs go to `CT_GAT/pipeline/runs/` (`GT_RUNS` changes it).

Check the data (the first command lists the collections, the cache and the fold files it finds, and exits 0), then build the CFG
cache from the two collections (CPU, 4 workers; 244 s on the development machine; a rerun skips built samples) and check it. The
cache gives Models 1 to 3 their folds and labels:

```bash
python -m common.catalog
python -m s02_preprocess.cache.build_cache
python -m s02_preprocess.cache.check_cache
```

The earlier jsonl-only models and the TF-IDF and import-table baselines are on the branch `jsonl-reference-models`; their published
numbers are in `CT_GAT/pipeline/docs/history/`.

Models 1 to 3 read `CT_GAT/processed/cfg_subgraphs`. Each has a smoke run (CPU) and the full run, here on fold 0; `--fold` picks one
of the five folds and `--seed` the seed. A smoke run needs the `.pt` files of 24 binaries (8 each for training, validation and test),
so unpack the 500-block zip first. The smoke runs of Models 2 and 3 leave a `team_smoke_*` folder in the system temp folder.

```bash
# Model 1, the five-channel transformer over the instruction sequence
python -m experiments.train --model fivechannel --protocol kfold --fold 0 --seed 0 --device cpu --smoke 8
python -m experiments.train --model fivechannel --protocol kfold --fold 0 --seed 0

# Model 2, the team GAT on the 77 block features and the CFG
python -m experiments.team_pipelines.train_b --arm sig500v3-feat --fold 0 --seed 0 --smoke 8 --device cpu
python -m experiments.team_pipelines.train_b --arm sig500v3-feat --fold 0 --seed 0

# Model 3, the block transformer trained end to end under the GAT
python -m experiments.team_pipelines.train_b --arm sig500v3-e2e --fold 0 --seed 0 --smoke 8 --device cpu
python -m experiments.team_pipelines.train_b --arm sig500v3-e2e --fold 0 --seed 0
```

`--arm sig500v3-emb` is the fixed-embedding variant of Model 3: the stored 64-d block embeddings and the CFG. The arm families
(each with `-feat`, `-emb`, `-e2e` unless noted): `sig500v3-*`, `sig250-*`, `callwalk500-*`, `callwalk250-*` and `sigwhole-*` (`-feat`,
`-emb` only, on the uncapped zip). `--build DIR` points a model at a build in another folder. `build_b --source repo` (about 4 h)
replaces the 500-block zip; its command is in the pipeline README.

A plan in `experiments/plans/` is one experiment for `experiments.queue`; `--dry` prints its runs without starting them:

```bash
python -m experiments.queue --plan experiments/plans/team_b.txt --dry          # team_builds.txt: the 135 runs of the other builds
```

When the runs exist, the tables:

```bash
python -m s04_evaluation.report
python -m s04_evaluation.gates --stage team
```

`report` writes `RESULTS.md` in the run directory over every run found. `gates` prints the arms of a stage that it finds and names
the missing folds and seeds; `--stage team` also compares with published arms that a new run directory does not have, so those rows
print as missing unless the published runs are in `GT_RUNS`. GPU runs are not repeatable to the last digit (CUDA scatter is
nondeterministic), so compare models through several seeds and the intervals in the tables, not through one AUC.

The tests need no data: `python -m pytest -m "not real_data and not gpu"`, from `CT_GAT/pipeline`.

## Results

5-fold family holdout on the 1,810 Mendeley x86 binaries; area under the ROC curve (AUC), mean over 3 seeds unless marked. Sources:
[`docs/history/`](CT_GAT/pipeline/docs/history/README.md).

* Model 3 with fixed block embeddings on the uncapped samples (`sigwhole-emb`): 0.981. At 500 blocks per binary (`sig500v3-emb`): 0.966.
  Trained end to end (`sig500v3-e2e`) it has no 3-seed result yet; its plan is `team_b`.
* Model 2 on the uncapped samples (`sigwhole-feat`): 0.970. At 500 blocks per binary (`sig500v3-feat`): 0.954.
* Model 1: 0.919 to 0.926, since it reads only the first 128 or 512 instructions of each binary.
* Published comparisons, from runs of the earlier code (see Run): a logistic regression on the PE import table 0.985 (seed 0),
  TF-IDF over the instruction tokens 0.982 (seed 0), the typed GNN (R7) 0.979. Model 3 on the uncapped samples is level with R7
  (+0.002, -0.005 to +0.009).

## More

[`CT_GAT/pipeline/README.md`](CT_GAT/pipeline/README.md) holds the stage folders, the on-disk layout, the overrides, the all-architecture
cache, `preprocess_full.py` and the IDA export; the plan, the result tables and the review are in `docs/history/`.
