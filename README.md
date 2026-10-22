# Ransomware detection from IDA Pro control-flow graphs

A master's project that classifies Windows PE binaries as goodware (label `0`) or ransomware (label `1`) from what IDA Pro
recovers in them: the control-flow graph (CFG) of each function, its instructions and the imported APIs. The binaries are
never executed. IDA runs on an offline VM, and every later stage reads only the text and feature files exported there.

## Data

The binaries are the Mendeley ransomware dataset: https://data.mendeley.com/datasets/p3v94dft2y/3. The code reads files derived
from it, which are on the team Google Drive. Two downloads are required; everything else in the table is optional.

| File | Contents | Needed for |
|---|---|---|
| `Collections/Mendeley_Ransomware.zip` (5.0 GB) | required. Per binary, the IDA CFG export `<sha256>.jsonl.gz` and the IDA listing `<sha256>.asm.gz`, in `samples/<split>/<family>/`, with the collection's `samples.csv` and `summary.json`. Unzip into `CT_GAT/data/` | the CFG cache and every command that reads it, `build_b`, `preprocess_full.py` |
| `Collections/Mendeley_Goodware.zip` (1.5 GB) | required. The same for the goodware binaries. Unzip into `CT_GAT/data/` | as above |
| `Mendeley_x86_cfg_subgraphs_500blocks_v3.zip` | the preprocessed samples of Janice's model: one `.pt` file per binary for the 1,810 x86 binaries of the folds, 500 blocks per binary. Unzip at `CT_GAT/`. It saves the 4 h build of `build_b --source repo` | Janice's five-channel model, Models 2 and 3 of the team diagram |
| `Mendeley_x86_cfg_subgraphs_uncapped_v3.zip` | the same with every non-library block of each binary | as above |
| `Collections/Ransomware_Balanced.zip` (8.9 GB), `Collections/Balanced_Goodware.zip` (15.7 GB) | two more collections in the same layout. Unzip into `CT_GAT/data/` | the all-architecture evaluation (`build_cache --cache combined_v3`) |
| `Collections/VirusShare Ransomware.zip` | unzips to `VirusShare_Malware/`. It is in no dataset: no fold file lists it and all its rows are in `splits/unassigned.csv`, so no command of the pipeline reads it | nothing |
| `Mendeley_GNN_v1.zip` | the Mendeley CFG exports as one bundle (`samples.csv`, `splits/`, `cfg/`), an alternative to the two Mendeley collections | the CFG cache, when the bundle is configured or unzipped into `CT_GAT/data/` |

The sizes are those of the zips on the Drive. Each zip has a `.sha256` file beside it and a README inside it. The commands for
every unzip are in Running, Quick start (the two Mendeley zips), Janice's model (the preprocessed zips) and All-architecture
evaluation (the two balanced zips). The fold
assignments of every published result are in the repository, in `CT_GAT/pipeline/s02_preprocess/folds/`, so
`family_holdout_folds.zip` on the Drive is not needed; it holds the same fold assignments. With the two Mendeley zips unzipped into
`CT_GAT/data/` nothing else is downloaded or configured for the commands under "Running".

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
    cache/                        the CFG caches the pipeline builds, one folder per dataset: CT_GAT/cache/<dataset> (gitignored)
    processed/                    the preprocessed .pt samples: the unzipped zips, or the output of preprocess_full.py or build_b (gitignored)
    pipeline/                     all the code; see CT_GAT/pipeline/README.md
        s01_ida_export/  s02_preprocess/  s03a_transformer/  s03b_cfg_feature/  s03c_cfg_transformer/  s04_evaluation/
        s02_preprocess/folds/     the committed fold assignments of every published result
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

Every command runs from `CT_GAT/pipeline`. Run outputs go to `CT_GAT/pipeline/runs/` (`RUNS` below), which is gitignored.

### Quick start

Download `Mendeley_Ransomware.zip` and `Mendeley_Goodware.zip` from the Drive and unzip them into `CT_GAT/data/`. Each zip holds
one top-level folder named like the zip, so the unzip creates `CT_GAT/data/Mendeley_Ransomware/` and
`CT_GAT/data/Mendeley_Goodware/`, and the destination is `CT_GAT/data` itself. The repository does not ship `CT_GAT/data/`
(it is gitignored), so create it first. From the repository root, with `<path to>` the folder that holds the downloaded zips:

```bash
# Windows, PowerShell or cmd (tar is the one that comes with Windows and reads zips)
mkdir CT_GAT\data
tar -xf <path to>\Mendeley_Ransomware.zip -C CT_GAT\data
tar -xf <path to>\Mendeley_Goodware.zip -C CT_GAT\data

# macOS and Linux (on macOS, tar -xf <zip> -C CT_GAT/data also works)
mkdir -p CT_GAT/data
unzip <path to>/Mendeley_Ransomware.zip -d CT_GAT/data
unzip <path to>/Mendeley_Goodware.zip -d CT_GAT/data
```

Git Bash on Windows has `unzip`, so the macOS and Linux form works there; its `tar` is GNU tar and cannot read zips. In Windows
Explorer, "Extract All" works only if the destination is set to `CT_GAT\data`; the destination it proposes is a folder named
after the zip, which leaves `CT_GAT\data\Mendeley_Ransomware\Mendeley_Ransomware\samples.csv`, one folder too deep. On macOS,
double-clicking a zip extracts it next to the zip (in Downloads); move the extracted folder into `CT_GAT/data/`. Either way
the result is

```
CT_GAT/data/
    Mendeley_Ransomware/
        samples.csv  summary.json  START_HERE.md  SHA256SUMS.txt  (and three Python scripts)
        splits/  provenance/
        samples/<split>/<family>/<sha256>.jsonl.gz, <sha256>.asm.gz
    Mendeley_Goodware/        the same
```

A collection counts as present when `CT_GAT/data/<name>/samples.csv` exists. No `datasets.json` and no environment variable is
needed: the code finds the collections there, reads the fold files from the repository and writes the cache to
`CT_GAT/cache/mendeley_x86/`.

```bash
cd CT_GAT/pipeline
python -m common.catalog
python -m s02_preprocess.cache.build_cache
python -m s02_preprocess.cache.check_cache
```

`common.catalog` prints, per dataset, which collection folders are present under `CT_GAT/data`, whether a bundle is
configured, whether the cache is built and whether the fold files are found. It always exits 0. After the two unzips it
shows `mendeley_x86` with both Mendeley folders present and the cache not built, and `combined_v3` with `Ransomware_Balanced`
and `Balanced_Goodware` missing:

```
mendeley_x86: protocols bundle, kfold; corpora mendeley
  collections under CT_GAT/data:
    Mendeley_Ransomware: present (snapshot 20261008T184757Z)
    Mendeley_Goodware: present (snapshot 20261008T184719Z)
```

A folder that holds the collection one level too deep is listed as `nested`, with the folder to move up, and a zip that
is in `CT_GAT/data` but not unzipped is listed as `zip`. `build_cache` stops with the same message.

`build_cache` reads the CFG exports of the two collections and writes `samples/<sample_id>.npz`, `index.csv` and `vocab.json` to
`CT_GAT/cache/mendeley_x86/`. It prints `2411 eligible, 0 copied from base, 0 already built, 2411 to build`, and ends with a
summary in which `ok` is 2411 and `vocab_sha` starts with `4ded77e0`; 236 binaries without a recommended split and 23 UPX stubs
are left out. On the development machine `index.csv`, `vocab.json` and all 2,411 `samples/*.npz` are byte-identical to the cache
the published runs read. A rerun skips the samples already built. `check_cache` checks every invariant of every sample.

Baselines, including the header-imports baseline. The import arrays come from the CFG exports, so the team's reader output
(`GT_READER_ROOT`, not on the Drive) is not needed:

```bash
python -m s04_evaluation.baselines --protocol kfold --fold 0 1 2 3 4 --arch x86 --seed 0
python -m s03b_cfg_feature.gnn.s10_unused_import_features.import_side --from-cfg
python -m s04_evaluation.baselines --protocol kfold --fold 0 1 2 3 4 --imports
```

`import_side --from-cfg` reads the header record of each `.jsonl.gz` and writes `CT_GAT/cache/mendeley_x86/side/imports/`. The
header-imports baseline on these arrays reproduces the published predictions on all five folds (fold 0 AUC 0.9962).

Model 2, the GNN on block token counts (one run per fold and seed), and Model 1, the transformer over the instruction
sequence. Model 1 reads a block export, made once:

```bash
python -m experiments.train --protocol kfold --fold 0 --seed 0 --model gnn --mask-blocks 0.15
python -m s03c_cfg_transformer.block_export --protocol kfold --arch x86 --out RUNS/export/ours_cap0
python -m experiments.train --model tokenseq --protocol kfold --fold 0 --seed 0 --device cpu --smoke 8
python -m experiments.train --model tokenseq --protocol kfold --fold 0 --seed 0
```

The export records `export_sha` `e09971a0...` in the tokenseq run, the value of the published runs. `--smoke 8 --device cpu`
trains 8 binaries for 2 epochs in a temporary folder, as a check of the setup. Janice's five-channel model (`--model
fivechannel`) reads the pipeline-B samples in `CT_GAT/processed/cfg_subgraphs`; its commands are in the next section, after
the zip or the build that fills that folder. Model 3 (frozen CodeBERT block vectors) and the
teammates' GAT pipelines are in [`CT_GAT/pipeline/README.md`](CT_GAT/pipeline/README.md).

When the runs you want exist, the tables:

```bash
python -m s04_evaluation.report
python -m s04_evaluation.val_threshold
python -m s04_evaluation.gates --stage model1_variants
python -m s04_evaluation.gates --stage day1
```

`report` writes `RUNS/RESULTS.md` over every run found. `val_threshold` re-scores the saved predictions of the runs found, and with no run folder stops and says so. `gates`
prints the arms of a stage that it finds and names the missing folds and seeds. `--stage v3` needs the all-architecture runs below.

Time and hardware per command, measured on the development machine:

| Command | Hardware | Time |
|---|---|---|
| `build_cache` | CPU, 4 workers | 200 s of building for 2,411 samples (244 s in all, from `build_summary.json`) |
| `baselines` | CPU | minutes |
| `check_cache` | CPU | not measured here |
| `import_side --from-cfg` | CPU | 33 s for 2,411 samples |
| `experiments.train --model gnn` | GPU only: it moves the model to CUDA | not measured here |
| `block_export` | CPU | minutes |
| `experiments.train --model tokenseq` | GPU; `--device cpu --smoke 8` runs on the CPU | not measured here |
| `build_cache --cache combined_v3` | CPU, 4 workers | 1,307 s of building for 5,838 samples (1,494 s in all) |
| `build_b --source repo` | CPU | 7.6 s per sample, about 4 h for 1,810 binaries at 500 blocks (a measured rate; the zip replaces it) |
| `experiments.train --model fivechannel` | CPU or GPU (`--device`) | not measured here |

### Janice's model

Janice's five-channel Model 1 trains on a folder of pipeline-B `.pt` files, one per binary, and takes its folds from the CFG
cache, so `build_cache` above comes first. `--model fivechannel` reads `CT_GAT/processed/cfg_subgraphs` unless `--build` names
another folder. The folder is filled in one of two ways, and holds one build at a time.

**From the zip** (fast). Unzip `Mendeley_x86_cfg_subgraphs_500blocks_v3.zip` (or the
uncapped one) at `CT_GAT/`. The zip has no top-level folder: it holds `processed/` and a README, so the destination is `CT_GAT`
itself and not `CT_GAT/processed`. From the repository root:

```bash
# Windows, PowerShell or cmd
tar -xf <path to>\Mendeley_x86_cfg_subgraphs_500blocks_v3.zip -C CT_GAT
# macOS and Linux
unzip <path to>/Mendeley_x86_cfg_subgraphs_500blocks_v3.zip -d CT_GAT
```

This creates `CT_GAT/processed/api_vocab.json`, `CT_GAT/processed/cfg_subgraphs/` and the README at `CT_GAT/`. In Windows
Explorer, set the destination of "Extract All" to `CT_GAT`; the proposed folder named after the zip puts everything one folder
too deep, where nothing reads it. If you take the files from the Drive one by one instead, put them here:

| File | Goes to |
|---|---|
| `api_vocab.json` | `CT_GAT/processed/api_vocab.json` (the API vocabulary; the five-channel demo and `preprocess_full.py` read it there) |
| `<sha256>.pt` (one per binary) | `CT_GAT/processed/cfg_subgraphs/` |
| `channel_embedder.pt` | `CT_GAT/processed/cfg_subgraphs/` |
| `<sha256>.manifest.json` (500-block zip only) | `CT_GAT/processed/cfg_subgraphs/` (alignment report per binary; no model reads it) |
| `build_info/` | `CT_GAT/processed/cfg_subgraphs/build_info/` (per-binary index and build settings; no model reads it) |
| `README_Mendeley_x86_cfg_subgraphs_<budget>_v3.md` | stays at `CT_GAT/`, where the zip unpacks it (git ignores it; no command reads it) |

Each zip's `.pt` files carry API ids of its own `api_vocab.json` (12,406 names in the 500-block build, 12,788 uncapped), so the
files of the two zips must not be mixed.

**From the collections.** `build_b --source repo` runs `preprocess_full.py` and `tokenize_asm_gz.py` of this repository over
the CFGs and listings of the unzipped Mendeley collections. It needs the cache for the fold ids. A quick check builds two
binaries, about 20 s, into a scratch folder; take two `sample_id`s of x86 rows that have a `fold` from
`CT_GAT/cache/mendeley_x86/index.csv`:

```bash
python -m experiments.team_pipelines.build_b --source repo --sample signal --num-blocks 500 --seed 0 --ids <sample_id> <sample_id> --out RUNS/team/b_check
```

The full form takes about 4 h and writes into an empty `CT_GAT/processed/cfg_subgraphs/`:

```bash
python -m experiments.team_pipelines.build_b --source repo --sample signal --num-blocks 500 --seed 0
```

This is the published 500-block build: a rebuild of 60 binaries was identical to it in every stored field. The full build takes
about 4 h on the CPU for the 1,810 binaries (7.6 s per binary, measured on the development machine) and resumes from its
checkpoint. It writes its `api_vocab.json` inside its `--out` folder (`CT_GAT/processed/cfg_subgraphs/api_vocab.json`), not to
`CT_GAT/processed/api_vocab.json`, which the untrained demo reads.

With either build in place, the first command checks the setup (8 training binaries, 2 epochs, a temporary folder) and the
second trains and evaluates the model on fold 0 (CPU or GPU, `--device`). `--smoke 8` also scores 8 validation and 8 test
binaries, so the folder must hold the `.pt` files of those 24 binaries; a folder with fewer, such as the quick check above,
stops with the ids that have no file:

```bash
python -m experiments.train --model fivechannel --protocol kfold --fold 0 --seed 0 --device cpu --smoke 8
python -m experiments.train --model fivechannel --protocol kfold --fold 0 --seed 0
```

The untrained Model 1 over every sample of the folder, with its predictions printed:

```bash
python -m s03a_transformer.five_channel.model --batch-size 4
```

(`--data-dir` reads another folder of `.pt` files). The demo always reads the API vocabulary from
`CT_GAT/processed/api_vocab.json`, which the preprocessed zip provides and `build_b` does not: after a `build_b` run, copy
the build folder's file there, as in `cp ../processed/cfg_subgraphs/api_vocab.json ../processed/api_vocab.json`. Without the file
the demo still runs, but the vocabulary is empty and every API id is clamped to one embedding row.

`s02_preprocess/preprocess_full.py` is Yanping's own script and a different path from `build_b`: it processes the train rows
first, freezes the test split and includes 21 UPX-stub x86 rows, so its output is not the published `.pt` set. It reads the
collections in `CT_GAT/data/` and writes to `CT_GAT/processed/` (`--output` and `--vocab` change that):

```bash
python s02_preprocess/preprocess_full.py --sample signal --num-blocks 500 --seed 0 --architecture x86
python s02_preprocess/preprocess_full.py --sample signal --num-blocks 0 --seed 0 --architecture x86
python s02_preprocess/preprocess_full.py --limit 2 --num-blocks 32 --seed 0
```

The options (sampling, library functions, caps, resuming) are in
[`CT_GAT/pipeline/s02_preprocess/README.md`](CT_GAT/pipeline/s02_preprocess/README.md).

### All-architecture evaluation

The all-architecture protocol (`kfold_v3`) adds two collections. Unzip `Ransomware_Balanced.zip` and `Balanced_Goodware.zip` into
`CT_GAT/data/` as in the Quick start, from the repository root:

```bash
# Windows, PowerShell or cmd
tar -xf <path to>\Ransomware_Balanced.zip -C CT_GAT\data
tar -xf <path to>\Balanced_Goodware.zip -C CT_GAT\data
# macOS and Linux
unzip <path to>/Ransomware_Balanced.zip -d CT_GAT/data
unzip <path to>/Balanced_Goodware.zip -d CT_GAT/data
```

`python -m common.catalog` (from `CT_GAT/pipeline`) then lists the four folders of `combined_v3` as present. The combined cache starts
from the vocabulary and the samples of the `mendeley_x86` cache, which must be built first.

```bash
python -m s02_preprocess.cache.build_cache --cache combined_v3
python -m s02_preprocess.cache.check_cache --cache combined_v3
python -m s04_evaluation.baselines --protocol kfold_v3 --fold 0 1 2 3 4 --arch all --seed 0
python -m s03b_cfg_feature.gnn.s10_unused_import_features.import_side --from-cfg --cache ../cache/combined_v3
python -m s04_evaluation.baselines --protocol kfold_v3 --fold 0 1 2 3 4 --arch all --imports
python -m experiments.queue --plan experiments/plans/v3.txt --budget-min 600
python -m s04_evaluation.gates --stage v3
```

The combined cache holds 5,838 samples; one CIL binary fails to build, as in the published cache. A `kfold_v3` run reads
`CT_GAT/cache/combined_v3/` by default. `import_side --from-cfg` writes the import arrays the `--imports` baseline reads, and
`baselines` takes `--arch x86` unless told otherwise, so the v3 baselines pass `--arch all`. `--arch matched` keeps the
architecture-matched subset (the rows with `arch_matched` 1 in `index.csv`), and `--arch-balance` weighs each class and
architecture cell equally in place of the class weights. The queue runs the baselines and the GNN arms of the v3 plan (GPU).

### Overrides

Every default above is a fallback. The order is environment variable, then `datasets.json`, then the default of
`CT_GAT/pipeline/common/catalog.py`. To keep data or caches elsewhere, copy `CT_GAT/pipeline/datasets.json.example` to
`datasets.json` (gitignored) and edit the paths, or set a variable: `GT_COLLECTIONS` (the folder that holds the unzipped
collections), `GT_CACHE` (a cache folder), `GT_RUNS` (the run directories), `GT_FOLDS` and `GT_FOLDS_V3` (other fold files),
`GT_TEAM_B_BUILD` (the build folder of `--model fivechannel`) and `GT_READER_ROOT` (the team's reader output).

The bundle `Mendeley_GNN_v1.zip` replaces the two Mendeley collections as the source of the Mendeley CFG exports. Unzip it
into `CT_GAT/data/` (the folder `Mendeley_GNN_v1/` with its `summary.json` is found there), or anywhere and set `GT_BUNDLE`
to that folder; the cache that `build_cache` writes is the same.
[`CT_GAT/pipeline/README.md`](CT_GAT/pipeline/README.md) lists every setting and what reads it.

### Tests

The tests that need no data run on any machine:

```bash
cd CT_GAT/pipeline
python -m pytest -m "not real_data and not gpu"
```

`-m real_data` runs the tests that read the built caches and expects the full-size ones (2,411 samples in `mendeley_x86`, 5,838
in `combined_v3`); a cache built with `--limit` fails some of them.

## Results

5-fold family holdout on the 1,810 Mendeley x86 binaries, 3 seeds; area under the ROC curve (AUC):

* Our edge-typed GNN over the whole CFG (R7) reaches 0.979. A logistic regression on the PE import table (0.985), TF-IDF over
  the instruction tokens (0.982) and the same GNN with no edges and no API names (0.981) are level with it, so the CFG
  structure adds nothing measurable over the tokens and imports.
* Pipeline B (`preprocess_full.py`) with the uncapped samples: Model 3 (block embeddings and CFG) 0.981, level with R7;
  Model 2 (block features and CFG) 0.970. At 500 blocks per binary both are lower (0.966 and 0.954).
* Model 1: the token-sequence transformer 0.970; Janice's five-channel model 0.919 to 0.926, since it reads only the first
  128 or 512 instructions of each binary.

These are means over 3 seeds. Runs on a GPU are not repeatable to the last digit (CUDA scatter is nondeterministic), so a rerun
on the same data does not return the same AUC: two runs of the token-sequence transformer on fold 0, seed 0, with the same data and the same
`export_sha` gave test AUC 0.9716 and 0.9899. Compare runs through several seeds and the intervals in the tables linked below, not through one AUC.

The tables, their confidence intervals and the decision rule are in
[`CT_GAT/pipeline/docs/history/`](CT_GAT/pipeline/docs/history/README.md).
