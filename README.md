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
preprocessed zips derive from it; they are on the project Google Drive with a `.sha256` file beside each zip. The three models need
the two Mendeley collections and one preprocessed zip, and train on the Mendeley x86 folds (1,810 binaries). The balanced collections
are other corpora (MalwareBazaar ransomware and balanced goodware), the input of the all-architecture cache `combined_v3`, which the
models do not yet train on. A preprocessed zip replaces a CPU build of 4 to 5 h (500 blocks) or 5.6 h (uncapped).

| Zip | Size | sha256 | Unzip into | Enables |
|---|---|---|---|---|
| `Collections/Mendeley_Ransomware.zip` | 5.0 GB | `c3785afff453792b007eee3f736f725df926f90c34638af88d4feb19772ada9c` | `CT_GAT/data/` | the `mendeley_x86` cache: the labels and folds of Models 1 to 3, `build_b`, `preprocess_full.py` |
| `Collections/Mendeley_Goodware.zip` | 1.5 GB | `33afd5dc007b1e551cd657209961a68aebbd975cecba9db266e7f6af48cf9cc1` | `CT_GAT/data/` | the same, for the goodware binaries |
| `Collections/Ransomware_Balanced.zip` | 8.9 GB | `12e379ac1afbd352d58740fe8d51af124c01dfd957d857453f58dda7895a5f67` | `CT_GAT/data/` | with `Balanced_Goodware.zip` and the two Mendeley zips, the `combined_v3` cache (`build_cache --cache combined_v3`) |
| `Collections/Balanced_Goodware.zip` | 15.7 GB | `c82042b6dd482181e6f977b184cfc09d84a94305c46fa9728322ce399857111c` | `CT_GAT/data/` | the same |
| `Collections/VirusShare Ransomware.zip` | 14.1 GB | `0a946d1e70d81c616e8e4f744b407b878ff83a2d23178fb1779e4f7088e9794a` | none | nothing: no fold file lists it |
| `Mendeley_x86_cfg_subgraphs_signal_500blocks_v3.zip` | 1.41 GB (9.87 GB unpacked) | `d7b984e4cadc95d2bd143c688bfcdfb99136370da129d48fbe490e2f17cd3421` | `CT_GAT/` | Models 1 to 3 read any build; the `sig500v3-*` arms need this one |
| `Mendeley_x86_cfg_subgraphs_signal_250blocks_v3.zip` | 0.81 GB (5.95 GB unpacked) | `61eb2847d8c24842477272f41ddf501f4312b15b96d05902115730a9f0487116` | `CT_GAT/` | the `sig250-*` arms |
| `Mendeley_x86_cfg_subgraphs_callwalk_500blocks_v3.zip` | 1.43 GB (10.11 GB unpacked) | `3f153392fb87fc67b4d968c701a8ea1c828cb654a66393eb25827951b0edae4e` | `CT_GAT/` | the `callwalk500-*` arms |
| `Mendeley_x86_cfg_subgraphs_callwalk_250blocks_v3.zip` | 0.78 GB (5.53 GB unpacked) | `f52ac00cee289d536aa328a47617f94cd011a29ab3b8d799e66224cded7c1ccd` | `CT_GAT/` | the `callwalk250-*` arms |
| `Mendeley_x86_cfg_subgraphs_uncapped_v3.zip` | 2.59 GB (13.29 GB unpacked) | `678beede56597afd1e369248a8e520cce0666a4b6d4af936a960bf5b7d1ccb0e` | `CT_GAT/` | the `sigwhole-*` arms |

`CT_GAT/processed/` holds one build, and the zips have different API vocabularies: empty it before unzipping another preprocessed
zip. `train_b` stops when the build's sampler or block count differs from the arm's. For the `team_builds` plan, each build must be
where `train_b` looks for its arm: move the contents of `CT_GAT/processed/cfg_subgraphs` to `CT_GAT/pipeline/runs/team/b_sig250_main`,
`b_callwalk500_main` or `b_callwalk250_main`, or pass `--build DIR`.

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
to the folder in the table; the proposed folder is one level too deep. The fold assignments are in the repository. The result, with
`<collection>` each of the four collection names:

```
CT_GAT/data/<collection>/samples.csv
CT_GAT/processed/api_vocab.json
CT_GAT/processed/cfg_subgraphs/
```

# Run

Commands run from `CT_GAT/pipeline`. Run outputs go to `CT_GAT/pipeline/runs/`; `GT_RUNS` changes it.

## Cache

`common.catalog` lists the collections, caches and fold files it finds and exits 0. `build_cache` reads the two Mendeley collections
(CPU, 4 workers, 244 s; a rerun skips built samples) and `check_cache` verifies the result:

```bash
python -m common.catalog
python -m s02_preprocess.cache.build_cache
python -m s02_preprocess.cache.check_cache
```

The all-architecture cache (5,838 samples) needs the balanced zips and the cache above, and names the missing folder and zip otherwise:

```bash
python -m s02_preprocess.cache.build_cache --cache combined_v3
```

## Models 1 to 3

Model 1 is the five-channel transformer over the instruction sequence (`experiments.train --model fivechannel`). Model 2 is a graph
attention network (GAT) on the 77 block features and the control-flow graph (`--arm sig500v3-feat`). Model 3 is the block
transformer trained end to end under the GAT (`--arm sig500v3-e2e`). The models read the build in `CT_GAT/processed/cfg_subgraphs`.
A Model 1 smoke run trains 8 binaries and scores 8 validation and 8 test binaries, so it needs the `.pt` files of those 24; a Model 2
or 3 smoke run trains 8 and scores every validation binary, with no test set. The smoke runs, then the full runs, on fold 0:

```bash
python -m experiments.train --model fivechannel --protocol kfold --fold 0 --seed 0 --device cpu --smoke 8
python -m experiments.team_pipelines.train_b --arm sig500v3-feat --fold 0 --seed 0 --smoke 8 --device cpu
python -m experiments.team_pipelines.train_b --arm sig500v3-e2e --fold 0 --seed 0 --smoke 8 --device cpu
python -m experiments.train --model fivechannel --protocol kfold --fold 0 --seed 0
python -m experiments.team_pipelines.train_b --arm sig500v3-feat --fold 0 --seed 0
python -m experiments.team_pipelines.train_b --arm sig500v3-e2e --fold 0 --seed 0
```

`--arm sig500v3-emb` is Model 3 with the stored 64-d block embeddings. The arm families are `sig500v3-*`, `sig250-*`, `callwalk500-*`
and `callwalk250-*`, each with `-feat`, `-emb` and `-e2e`, and `sigwhole-feat` and `sigwhole-emb` on the uncapped zip. `--build DIR`
points a model at another build folder.

## Plans and tables

A plan in `experiments/plans/` is one experiment for `experiments.queue`: `model1_variants`, `team_b` and `team_builds` (the 135 runs
on the `sig250`, `callwalk500` and `callwalk250` builds). `report` writes `RESULTS.md` in the run directory over the runs found.
`gates` prints the arms of a stage and names the missing folds and seeds; `--stage team` also compares with published arms, which
print as missing unless the published runs are in `GT_RUNS`. GPU runs differ in the last digits (CUDA scatter is nondeterministic),
so compare models through several seeds and the table intervals. The block prints a plan's runs without starting them (`--dry`),
writes the tables, checks the arms and runs the tests; none of it needs the datasets:

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
* Published comparisons, from runs of the earlier jsonl-only code on the branch `jsonl-reference-models`: a logistic regression on
  the PE import table 0.985 (seed 0), TF-IDF over the instruction tokens 0.982 (seed 0), the typed GNN (R7) 0.979. Model 3 on the
  uncapped samples is level with R7 (difference +0.002 AUC, 95% interval -0.005 to +0.009).

# Further documentation

[`CT_GAT/pipeline/README.md`](CT_GAT/pipeline/README.md) describes the stage folders, the on-disk layout, the overrides,
`build_b --source repo` (a 500-block build in 4 to 5 h), `preprocess_full.py` and the IDA export. The plan, the result tables and the review are in `docs/history/`.
