# Setup

Python 3.12. From the repository root:

```bash
pip install -r CT_GAT/pipeline/requirements.txt
```

This installs the CPU build of PyTorch and `torch_geometric`. For a GPU, first install the CUDA build of torch 2.11.0 from
https://pytorch.org; `requirements.txt` pins that version and would replace another. Model 1 runs on a CPU or a GPU (`--device`); the
full runs of Models 2 and 3 need a CUDA GPU, and their smoke runs use `--device cpu`.

# Data

The binaries are the Mendeley ransomware dataset (https://data.mendeley.com/datasets/p3v94dft2y/3). The Mendeley collections and the
preprocessed zips derive from it; they are on the project Google Drive with a `.sha256` file beside each zip. The two Mendeley
collections build the `mendeley_x86` cache, whose folds cover the x86 binaries (1,810) or the whole Mendeley dataset (2,411, with 601
x64). The balanced collections (MalwareBazaar ransomware and balanced goodware) add the all-architecture cache `combined_v3` (5,652
binaries with a fold, 4,850 architecture-matched). The three models train on each of these three datasets, with the collections and the
preprocessed zip of that dataset; a zip replaces a CPU build of 4 to 5 h (500 blocks) or 5.6 h (uncapped) for the x86 dataset, longer
for the others.

| Zip | Size | sha256 | Unzip into | Enables |
|---|---|---|---|---|
| `Collections/Mendeley_Ransomware.zip` | 5.0 GB | `c3785afff453792b007eee3f736f725df926f90c34638af88d4feb19772ada9c` | `CT_GAT/data/` | the `mendeley_x86` cache: the labels and folds of Models 1 to 3, `build_b`, `preprocess_full.py` |
| `Collections/Mendeley_Goodware.zip` | 1.5 GB | `33afd5dc007b1e551cd657209961a68aebbd975cecba9db266e7f6af48cf9cc1` | `CT_GAT/data/` | the same, for the goodware binaries |
| `Collections/Ransomware_Balanced.zip` | 8.9 GB | `12e379ac1afbd352d58740fe8d51af124c01dfd957d857453f58dda7895a5f67` | `CT_GAT/data/` | with `Balanced_Goodware.zip` and the two Mendeley zips, the `combined_v3` cache (`build_cache --cache combined_v3`) |
| `Collections/Balanced_Goodware.zip` | 15.7 GB | `c82042b6dd482181e6f977b184cfc09d84a94305c46fa9728322ce399857111c` | `CT_GAT/data/` | the same |
| `Collections/VirusShare Ransomware.zip` | 14.1 GB | `0a946d1e70d81c616e8e4f744b407b878ff83a2d23178fb1779e4f7088e9794a` | none | nothing: no fold file lists it |
| `Mendeley_x86_cfg_subgraphs_signal_500blocks_v3.zip` | 1.41 GB (9.87 GB unpacked) | `d7b984e4cadc95d2bd143c688bfcdfb99136370da129d48fbe490e2f17cd3421` | `CT_GAT/` | Models 1 to 3 on the x86 folds read any build; the `sig500v3-*` arms need this one |
| `Mendeley_x86_cfg_subgraphs_signal_250blocks_v3.zip` | 0.81 GB (5.95 GB unpacked) | `61eb2847d8c24842477272f41ddf501f4312b15b96d05902115730a9f0487116` | `CT_GAT/` | the `sig250-*` arms |
| `Mendeley_x86_cfg_subgraphs_callwalk_500blocks_v3.zip` | 1.43 GB (10.11 GB unpacked) | `3f153392fb87fc67b4d968c701a8ea1c828cb654a66393eb25827951b0edae4e` | `CT_GAT/` | the `callwalk500-*` arms |
| `Mendeley_x86_cfg_subgraphs_callwalk_250blocks_v3.zip` | 0.78 GB (5.53 GB unpacked) | `f52ac00cee289d536aa328a47617f94cd011a29ab3b8d799e66224cded7c1ccd` | `CT_GAT/` | the `callwalk250-*` arms |
| `Mendeley_x86_cfg_subgraphs_uncapped_v3.zip` | 2.59 GB (13.29 GB unpacked) | `678beede56597afd1e369248a8e520cce0666a4b6d4af936a960bf5b7d1ccb0e` | `CT_GAT/` | the `sigwhole-*` arms |
| `Mendeley_all_cfg_subgraphs_signal_500blocks_v3.zip` | pending | pending | `CT_GAT/` | the whole Mendeley dataset (`--protocol kfold --arch all`); the `sig500v3-*` arms |
| `Mendeley_all_cfg_subgraphs_signal_250blocks_v3.zip` | pending | pending | `CT_GAT/` | the same dataset; the `sig250-*` arms |
| `Mendeley_all_cfg_subgraphs_callwalk_500blocks_v3.zip` | pending | pending | `CT_GAT/` | the same dataset; the `callwalk500-*` arms |
| `Mendeley_all_cfg_subgraphs_callwalk_250blocks_v3.zip` | pending | pending | `CT_GAT/` | the same dataset; the `callwalk250-*` arms |
| `Mendeley_all_cfg_subgraphs_uncapped_v3.zip` | pending | pending | `CT_GAT/` | the same dataset; the `sigwhole-*` arms |
| `combined_v3_cfg_subgraphs_signal_500blocks_v3.zip` | pending | pending | `CT_GAT/` | the combined dataset (`--protocol kfold_v3 --arch all` and `matched`); the `sig500v3-*` arms |
| `combined_v3_cfg_subgraphs_callwalk_500blocks_v3.zip` | pending | pending | `CT_GAT/` | the same dataset; the `callwalk500-*` arms |

`CT_GAT/processed/` holds one build, and the zips have different API vocabularies: empty it before unzipping another preprocessed zip.
`train_b` stops when the build's dataset, sampler or block count differs from the arm's. The plans look for each build in
`CT_GAT/pipeline/runs/team/` (folder names in the pipeline README): move `CT_GAT/processed/cfg_subgraphs` there, or pass `--build DIR`.

From the repository root, with the zips in it, on Windows (PowerShell or cmd); the other zips unzip like the last one:

```bash
mkdir CT_GAT\data
tar -xf Mendeley_Ransomware.zip -C CT_GAT\data
tar -xf Mendeley_Goodware.zip -C CT_GAT\data
tar -xf Mendeley_x86_cfg_subgraphs_signal_500blocks_v3.zip -C CT_GAT
```

On macOS and Linux, and in Git Bash on Windows, whose `tar` cannot read zips:

```bash
unzip Mendeley_Ransomware.zip -d CT_GAT/data
unzip Mendeley_Goodware.zip -d CT_GAT/data
unzip Mendeley_x86_cfg_subgraphs_signal_500blocks_v3.zip -d CT_GAT
```

A collection zip holds one folder and a preprocessed zip holds `processed/`. In Windows Explorer, set the destination of "Extract All"
to the folder in the table; the proposed folder is one level too deep. The fold assignments are in the repository. The result is
`CT_GAT/data/<collection>/samples.csv` for each of the four collections, and `CT_GAT/processed/api_vocab.json` and
`CT_GAT/processed/cfg_subgraphs/` for a preprocessed zip.

# Run

Commands run from `CT_GAT/pipeline`. Run outputs go to `CT_GAT/pipeline/runs/`; `GT_RUNS` changes it. `common.catalog` lists the
collections, caches and fold files it finds and exits 0. `build_cache` reads the two Mendeley collections, `check_cache` verifies the
result, and `--cache combined_v3` builds the all-architecture cache (5,838 samples) from the balanced zips and the cache above:

```bash
python -m common.catalog
python -m s02_preprocess.cache.build_cache
python -m s02_preprocess.cache.check_cache
python -m s02_preprocess.cache.build_cache --cache combined_v3
```

## Models 1 to 3

Model 1 is the five-channel transformer over the instruction sequence (`experiments.train --model fivechannel`). Model 2 is a graph
attention network (GAT) on the 77 block features and the control-flow graph (`--arm sig500v3-feat`). Model 3 is the block transformer
trained end to end under the GAT (`--arm sig500v3-e2e`); `--arm sig500v3-emb` is Model 3 with the stored 64-d block embeddings. The
models read the build in `CT_GAT/processed/cfg_subgraphs`, or `--build DIR`. A Model 1 smoke run trains 8 binaries and scores 8
validation and 8 test binaries, so it needs the `.pt` files of those 24; Models 2 and 3 train 8 and score every validation binary.

The x86 folds are the default dataset. With `--protocol kfold`, `--arch all` trains on the whole Mendeley dataset; with
`--protocol kfold_v3`, `--arch all` or `--arch matched` trains on the combined one. A run reads the build of its zip. The other arm
families (`sig250-*`, `callwalk500-*`, `callwalk250-*` with `-feat`, `-emb` and `-e2e`, and `sigwhole-*`) are in the table;
`--arch-balance` weights the training binaries by class and architecture. On the whole Mendeley dataset the architecture is a proxy for
the label (x86: 1,134 ransomware and 676 goodware, x64: 108 and 493; the x86 rule alone scores 0.650 macro-F1, and 55% of the x64
ransomware are in fold 2), so a model gains AUC there by scoring x64 binaries as goodware. Read the recall of ransomware x64 and the
cell-balanced metrics, and take `combined_v3` (x86-rule floor 0.690 on all, 0.606 on matched, 741 x64 ransomware) as the
all-architecture test. The smoke runs on fold 0 come first, then runs on the other datasets; a full run omits `--smoke 8 --device cpu`:

```bash
python -m experiments.train --model fivechannel --protocol kfold --fold 0 --seed 0 --device cpu --smoke 8
python -m experiments.team_pipelines.train_b --arm sig500v3-e2e --fold 0 --seed 0 --smoke 8 --device cpu
python -m experiments.train --model fivechannel --protocol kfold --arch all --fold 0 --seed 0 --build runs/team/b_sig500_mendeley_all
python -m experiments.team_pipelines.train_b --arm sig500v3-feat --protocol kfold --arch all --fold 0 --seed 0
python -m experiments.team_pipelines.train_b --arm sig500v3-e2e --protocol kfold_v3 --arch all --fold 0 --seed 0
python -m experiments.train --model fivechannel --protocol kfold_v3 --arch matched --arch-balance --fold 0 --seed 0 --build runs/team/b_sig500_combined_v3
```

## Plans and tables

A plan in `experiments/plans/` is one experiment for `experiments.queue`: `model1_variants`, `team_b`, `team_builds` (135 runs),
`team_mendeley_all` (345 runs, whole Mendeley dataset) and `team_v3` (390 runs, combined dataset). `report` writes `RESULTS.md` in the
run directory. `gates` prints the arms of a stage and names the missing folds and seeds; `--stage team` also compares with published
arms, which print as missing unless the published runs are in `GT_RUNS`, and `team_mendeley_all`, `team_v3_all` and `team_v3_matched`
add the recall of each class and architecture. GPU runs differ in the last digits, so compare models through several seeds. The block
prints a plan's runs (`--dry`), writes the tables, checks the arms and runs the tests, with no dataset:

```bash
python -m experiments.queue --plan experiments/plans/team_b.txt --dry
python -m s04_evaluation.report
python -m s04_evaluation.gates --stage team
python -m pytest -m "not real_data and not gpu"
```

# Results

5-fold family holdout on the 1,810 Mendeley x86 binaries; area under the ROC curve (AUC), mean over 3 seeds unless marked. Sources:
[`docs/history/`](CT_GAT/pipeline/docs/history/README.md).

* Model 3 with fixed block embeddings on the uncapped samples (`sigwhole-emb`): 0.981. At 500 blocks per binary (`sig500v3-emb`): 0.966.
  Trained end to end (`sig500v3-e2e`) it has no 3-seed result yet; its plan is `team_b`.
* Model 2 on the uncapped samples (`sigwhole-feat`): 0.970. At 500 blocks per binary (`sig500v3-feat`): 0.954.
* Model 1: 0.919 to 0.926; it reads the first 128 or 512 instructions of each binary.
* Published comparisons, from runs of the earlier jsonl-only code on the branch `jsonl-reference-models`: a logistic regression on the
  PE import table 0.985 (seed 0), TF-IDF over the instruction tokens 0.982 (seed 0), the typed GNN (R7) 0.979; Model 3 on the uncapped
  samples is level with R7 (+0.002 AUC, 95% interval -0.005 to +0.009).

# Further documentation

[`CT_GAT/pipeline/README.md`](CT_GAT/pipeline/README.md) describes the stage folders, the on-disk layout, the overrides,
`build_b --source repo` (a 500-block build in 4 to 5 h), `preprocess_full.py` and the IDA export. The plan, the result tables and the review are in `docs/history/`.
