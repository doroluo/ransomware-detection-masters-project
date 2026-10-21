# Team pipelines under our folds

Adapters that run the teammates' GAT pipelines (A: construct_blocks + train_cfg; B: preprocess_full + graph_full_loader +
gat_full_model) under our 5-fold x86 family holdout, so their results sit in the same tables as ours
(`s04_evaluation/stats.py`, `report.py`, `gates.py`). The review of the two pipelines and the numbers are in
[TEAM_PIPELINES_REVIEW.md](../../docs/history/TEAM_PIPELINES_REVIEW.md).

Run every script as a module from `CT_GAT/pipeline` (`python -m experiments.team_pipelines.<name>`), not as a file: as a file
this directory is first on `sys.path` and its `common.py` shadows our `common` package. In the commands below `<runs>` is the run
directory of the machine (`GT_RUNS`, see [the pipeline README](../../README.md)); a leading `RUNS/` in an `--out` / `--build`
argument stands for it.

## What this needs

Everything here is optional for the rest of the pipeline: importing any module of this package works without the teammates'
code, without `torch_geometric` and without `psutil`, and the tests of these modules skip with a reason when an input is absent.
What the commands need:

| need | for | set with |
|---|---|---|
| the teammates' gnn worktree at commit 331ce0b | `build_a`, `build_b` (default source), `train_a`, `train_b`, `fit`, match tools | `GT_TEAM_GNN_ROOT` |
| the teammates' main worktree at commit c7899f0 | `build_b --source main-c7899f0`, `train_b --arm sig500v2` builds | `GT_TEAM_MAIN_ROOT` |
| the Mendeley bundle (`samples.csv`, `cfg/`) | `paths_for`, `team_row`, both builds, when the listings are configured too | `GT_BUNDLE` (or the `bundle` source of `datasets.json`) |
| the IDA listings `<sample_id>.asm.gz` | `build_b`, the match tools' replay, when the bundle is configured too | `GT_TEAM_ASM_DIR` |
| the two Mendeley collections, unzipped (`CT_GAT/data/Mendeley_{Ransomware,Goodware}`) | `paths_for`, `team_row`, `build_b` when neither the bundle nor `GT_TEAM_ASM_DIR` is set (their CFGs and listings, from `samples.csv`) | `GT_COLLECTIONS` (default `CT_GAT/data`) |
| `torch_geometric` | `build_a`, `train_a`, `train_b` (graphs and the GAT) | `pip install torch_geometric` |
| `psutil` | `measure_rss` | `pip install psutil` |

The variables are read when a function needs them, never at import; an unset one raises `TeamSourceNotConfigured` (a
`FileNotFoundError`) whose message names the variable. With neither `GT_BUNDLE` nor `GT_TEAM_ASM_DIR` set, `paths_for` reads the
Mendeley collections instead (the mendeley row of `build_cache.corpus_samples('mendeley')`, its `cfg_path` and `asm_path` relative to
its collection); with only one of the two set it asks for the other, as before. The worktrees are checked against the sha256 pins below before any of
their code runs, and nothing is ever written into them.

## Team pipelines: how to run

Each run is one command from `CT_GAT/pipeline`. A run is finished when `<runs>/<name>/metrics.json` exists; the trainers do not
check, so a loop over folds and seeds should skip those, and a failed run is simply started again (the trainers are deterministic
given the seed). The runs below were made on the
CPU (`--device cpu`, `CUDA_VISIBLE_DEVICES=-1`) so that the GPU queue of `experiments/queue.py` is not touched. `TEAM_THREADS`
caps the torch threads of a trainer (3 when run next to a GPU queue and a build).

```
export TEAM_THREADS=3 CUDA_VISIBLE_DEVICES=-1
# builds (once; resumable)
python -m experiments.team_pipelines.build_a --mode gnn --budget 500 --out RUNS/team/a_gnn500
python -m experiments.team_pipelines.build_b --sample signal --num-blocks 500 --seed 0 --out RUNS/team/b_sig500

# for each seed S in 0 1 2 and fold F in 0 1 2 3 4: the run <runs>/kfold{F}_x86_gnn_s{S}_team-<arm>
python -m experiments.team_pipelines.train_a --fold F --seed S --arm gnn500 --device cpu
python -m experiments.team_pipelines.train_a --fold F --seed S --arm cb500 --block-vec ours_cap0_codebert --device cpu
python -m experiments.team_pipelines.train_b --fold F --seed S --arm sig500 --device cpu
```

- `gnn500` and `cb500` (`train_a`) need no build but `gnn500`'s (`build_a`); `cb500` reads the CodeBERT block vectors
  `ours_cap0_codebert` of the cache (`s03c_cfg_transformer/precompute_blockvec.py`). 15 runs per arm (3 seeds x 5 folds), names
  `kfold{F}_x86_gnn_s{S}_team-gnn500` and `..._team-cb500`.
- `sig500` (`train_b`) starts when `<runs>/team/b_sig500/build_summary.json` accounts for all 1810 binaries: `total` is 1810 and
  `processed` plus the number of ids in `excluded` is 1810 (the ids `build_b --exclude` left out on purpose count; `train_b`
  leaves them out of every split). A summary that says less means the build stopped early; do not train on it.
- `sig500v2` is the same trainer on the rebuild with Yanping's streaming preprocess: `build_b --source main-c7899f0 --out
  RUNS/team/b_sig500_v2` (below), then `train_b --fold F --seed S --arm sig500v2 --device cpu`, 15 runs
  `kfold{F}_x86_gnn_s{S}_team-sig500v2`. Its build must report all 1810 processed and none excluded.
- Our own matched arms (`--subgraph cap --budget 500 --tag b500`, typed and `--control no_edges`, seeds 0 1 2, folds 0-4: 30 runs)
  are the plan `experiments/plans/team.txt`, run with `python -m experiments.queue --plan experiments/plans/team.txt`; the
  CodeBERT arms of path 2 are `experiments/plans/paths_codebert.txt`.
- The table: `python -m s04_evaluation.gates --stage team` reads whatever is on disk: TF-IDF, the header-imports LR, R7, R2, our
  b500 arms, the team arms and path 2 on CodeBERT vectors; AUC, PR-AUC, macro-F1, accuracy, precision, recall and the confusion
  matrix per arm (seed 0 pooled), the paired differences of each team arm against TF-IDF, R7 and our b500 typed arm (and cb500
  against path 2, the b500 typed arm against R7), with the pre-registered rule text. A missing arm prints as missing.

`RunConfig` accepts `--budget` from 100 blocks (0 is the binary cap). Below 1000 the variant suffix spells the budget out
(`sg-cap500`, `sg-cap128`), so two budgets cannot share a directory name; the matched arms still pass `--tag b500`. Every
existing run name is unchanged (`tests/test_run_names.py`).

CPU time per epoch (3 threads, Ryzen 7 9800X3D shared with the GPU queue and the build, fold 0): gnn500 about 6 ms per training
graph (57-d features), cb500 about 6 ms (768-d float16 vectors cast per batch), sig500 about 8 ms (141-d features, edge
attributes). A fold has about 1,200 training and 230 validation binaries, so an epoch takes roughly 8 s (gnn500, cb500) and
11 s (sig500); the recipes stop after at most 20 (A) and 40 (B) epochs, which bounds a fold at about 3 and 7 minutes of training
plus under a minute of loading and prediction. The 15 runs of an arm therefore take at most about 1 h (A) and 2 h (B).

## What is here

- `common.py`
  - `import_team(module, source='gnn-331ce0b')`: imports a teammate module from the review worktree of `source`
    (`root_of(source)`: `GT_TEAM_GNN_ROOT` or `GT_TEAM_MAIN_ROOT`, see "Teammate sources") after checking the sha256 of every teammate file the import
    runs against that source's pins. A mismatch raises with the file name. Nothing is written into the worktree (bytecode
    caching is off during the import).
  - `fold_manifest(fold, seed)`: our `make_manifest` for protocol `kfold`, arch `x86`, `val_seed=None`, as the
    `kfold_x86_gnn-mblk15_s{seed}` runs use. `all_ids()`: the 1,810 test ids of the five folds, sorted.
  - `paths_for(sample_id)`: the CFG (`<GT_BUNDLE>/samples.csv` `cfg_path`) and the IDA listing
    (`<GT_TEAM_ASM_DIR>/<sha>.asm.gz`); with neither configured, both from the Mendeley collections (`samples.csv` `cfg_path` and
    `asm_path`). A missing row or file raises.
  - `team_config(fold, seed, arm)` and `write_run(...)`: a run directory `kfold{f}_x86_gnn_s{seed}_team-<arm>` with
    `config.json`, `manifest.json`, `predictions.csv` (F1 state, `pred = prob >= 0.5`, as `experiments/train.py`),
    `predictions_auc.csv` (AUC state, `pred = prob >= threshold_val`) and `metrics.json`. The columns and the metrics are
    `experiments/train.py`'s (`write_predictions`, `evaluate`) over `s04_evaluation/metrics.py`; the rows must be exactly the
    manifest's validation and test ids. `manifest.json` is read by `report.py` for the floors.
  - `team_row(sample_id)`: the row `construct_blocks` expects (bundle `samples.csv` row, int `label`, `cfg_file`).
- `match_diagnosis.py`, `match_acceptance.py`: why the listing-to-CFG match fails and the PASS/FAIL checks for a fix (step 24, below).
- `build_b.py`, `build_a.py`: the two preprocessing builds; `train_a.py`, `train_b.py`: the trainers; `fit.py`: the
  teammates' training loop on our manifests (shared by both trainers). See below.
- `compare_builds.py`, `match_rates.py`, `measure_rss.py`: small tools on builds (step 21, below).

## build_b.py

```
python -m experiments.team_pipelines.build_b --sample signal --num-blocks 500 --seed 0 [--out RUNS/team/b_sig500]
       [--source gnn-331ce0b|main-c7899f0|repo] [--limit N] [--ids <sample_id> ...] [--frozen-from <build dir>]
```

Zero config, the exact build for Janice's model. `--out` defaults to the `team_b_build` source, `CT_GAT/processed/cfg_subgraphs`
(`GT_TEAM_B_BUILD`, `datasets.json` or the catalog default), which is where `python -m experiments.train --model fivechannel` reads.
`--source repo` runs `preprocess_full.py` and `tokenize_asm_gz.py` of this repository (`CT_GAT/pipeline/s02_preprocess`, unchanged)
over the CFGs and listings of the unzipped Mendeley collections:

```
python -m s02_preprocess.cache.build_cache --cache mendeley_x86        # once: the fold ids come from this cache's index.csv
python -m experiments.team_pipelines.build_b --source repo --sample signal --num-blocks 500 --seed 0
```

With no `--ids` it builds all 1,810 x86 ids in sorted order, with one vocabulary pass and seed 0: the published pipeline-B build
(`b_sig500_v3`), about 4 h on the CPU (7.6 s per sample, measured on the development machine). It resumes from its checkpoint. The faster way is the preprocessed zip
`Mendeley_x86_cfg_subgraphs_500blocks_v3.zip`, unzipped at `CT_GAT/`; the build does not write `CT_GAT/processed/api_vocab.json` that
the zip carries, which only Janice's untrained demo in `model.py` reads. Without the cache `build_b` stops and names the
`build_cache` command. A folder that holds sample `.pt` files but neither `checkpoint.json` nor `settings.json` (the unzipped zip) is
refused, not overwritten: give `--out` an empty folder.

`--source` picks which pinned teammate code runs (see "Teammate sources" below); the default `gnn-331ce0b` is what built
`RUNS/team/b_sig500`, so the first build stays reproducible. `settings.json` records `source` and the sha256 `pins` of the files that
`preprocess_full` runs; the checkpoint's settings carry `source` too, so resuming a directory with another source stops (a checkpoint
without `source`, as the first build wrote, reads as `gnn-331ce0b`). The checkpoint of a `repo` build carries the `pins` as well: they
are the sha256 of the two files as they were when the build started (there is no fixed pin, the repository is the reference; a
Windows checkout with CRLF line endings hashes differently from an LF one), so a resume after either file changed stops and
names the file. `--frozen-from DIR` starts from DIR's `api_vocab.json` and
`channel_embedder.pt` and calls `process_sample(freeze_vocab=True)`: ids and embeddings of two sources are then comparable sample by
sample (the equivalence check).

`RUNS/` at the start of `--out` is the run directory `<runs>`. For each of the 1,810 ids (sorted) it calls the
teammates' `preprocess_full.process_sample(asm_path, cfg_path, ...)` with the explicit cfg path and `save_sample`, so no
file of theirs is parsed by new code. The size caps (`--max-asm-bytes`, `--max-cfg-blocks`, `--max-cfg-instructions`)
live in their `main`, which is not called: nothing is skipped. One shared `ApiVocab` and one `FiveChannelEmbedder`
(created after `torch.manual_seed(seed)`) are used for the whole build.

Output in `<out>`: `<id>.pt` and `<id>.manifest.json` per sample (keys of the `.pt`: `instruction_ids [N,5]`,
`insn_to_block`, `instruction_matched`, `instruction_rvas`, `block_features [B,77]`, `block_embeddings [B,64]`,
`edge_index`, `edge_type`, `edge_attr`, `edge_back`, `block_types`, `block_external`, `block_keys`, `label`,
`dataset_metadata`, ...), `api_vocab.json`, `channel_embedder.pt`, `checkpoint.json`, `build_log.csv`
(`sample_id, status, n_blocks_total, n_blocks_kept, seconds`) and `build_summary.json`. Vocabulary, embedder and
checkpoint are saved every 50 samples and at the end.

Resume: rerun the same command. The vocabulary and embedder are loaded and the ids in `checkpoint.json` are kept; any
other `.pt` (finished after the last checkpoint, with API ids the saved vocabulary may lack) is deleted and rebuilt, so
`build_log.csv` can hold a second row for such an id. Rows added to the embedder after a resume are initialised from a
different RNG state than in an uninterrupted run. A different `--sample / --num-blocks / --seed` on an existing `<out>`
stops. A sample that raises stops the build with its id and a traceback; nothing is skipped.

Exclusion: `--exclude <id> ... --exclude-reason TEXT` leaves ids out on purpose. They are not attempted, count as neither built nor
failed, and are written with the reason to `<out>/settings.json` (the checkpoint's own settings are unchanged, so an earlier build still
resumes) and to `build_summary.json` as `excluded`; a build with `processed` + the number excluded = `total` is finished, which is what
the `sig500` runs wait for. Pass the same `--exclude` on every resume. The one exclusion so far is
`bbbf38de4f40754f235441a8e6a4c8bdb9365dab7f5cfcdac77dbb4d6236360b`: `process_sample` raises `MemoryError` reading its 341 MB `asm.gz`
(`read_text_lines`), also when that sample is built alone (peak 20.2 GB resident on a 32 GB machine). The failing code is the teammates'
pinned `preprocess_full.py`, which is not edited. `train_b` then runs on the manifest ids minus the excluded one (training, validation and
test of every fold; the other 1,809 are unchanged), `common.write_run(excluded=, excluded_reason=)` records it in `metrics.json`, and
`stats.py` / `gates.py` print "N excluded from both arms: <ids> (<reason>)" under every paired difference that involves a sig500 run.

Each manifest json is 3-6 MB (about 8 GB for the full build); the `.pt` files are about 0.5 MB.

## build_a.py (pipeline A, opcode mode)

```
python -m experiments.team_pipelines.build_a --mode gnn --budget 500 --out RUNS/team/a_gnn500 [--limit N] [--ids ...] [--workers W]
```

For each of the 1,810 ids (sorted) the row `construct_blocks` expects (the bundle `samples.csv` row, int `label`, `cfg_file`;
`common.team_row`) goes to `gnn_graph_from_export(row, max_blocks=500)`. Output `<out>/<id>.pt` (PyG `Data`: `x` [n, 57],
`edge_index`, `func_batch`, `y`, plus `sample_id`, `label`, `num_blocks_total`), written through a temp file and renamed,
`settings.json` (mode, budget; a different request on an existing dir stops), `build_log.csv` (`sample_id, status,
n_blocks_total, n_blocks_kept, seconds`) and `build_summary.json`. Resumable (ids with a `.pt` are skipped). A sample that
raises stops the build with its id; nothing is skipped. `--workers W` parses W exports at a time (spawned processes).
`--mode codebert` does not exist: the CodeBERT variant is train_a's `cb500` arm.

## train_a.py (arms gnn500, cb500)

```
python -m experiments.team_pipelines.train_a --fold k --seed s --arm gnn500|cb500 [--block-vec ours_cap0_codebert] [--smoke N] [--device cuda|cpu]
```

- Inputs. `gnn500`: `RUNS/team/a_gnn500/<id>.pt` of build_a (57 opcode / API-group / structure features per block; the
  teammates' ranking keeps 500 blocks of whole functions). `cb500`: no raw export is read. The graph of a binary comes from
  our cache: its blocks are `select_subgraph(z, 'cap', 500, rng=None)` (the blocks the matched `--subgraph cap --budget 500`
  arm evaluates), its edges the cache's directed CFG edges with both ends kept and in one function, its node features the
  block vectors of `transformer_paths.precompute_blockvec` (`BlockVecSource`: kept cache block -> export `block_key` ->
  vector; every key must exist and the vectors must belong to this protocol, arch and export). Default
  `--block-vec ours_cap0_codebert` (768 d); `ours_cap0_ref` (64 d, fold 0 only) for smoke runs. The graphs are built in memory
  per fold (x held as float16 and cast per batch).
- What is fitted where. Nothing before training: build_a's features are fixed tables and the cb500 vectors come from a frozen
  encoder (the vectors of fold k come from the encoder checkpoint of fold k). `BlockGAT(hidden 128, heads 4, dropout 0.2)` is
  trained per fold and seed on our train ids after `torch.manual_seed(seed)`.
- Recipe (train_cfg): AdamW lr 1e-3, weight decay 1e-4, cross-entropy weighted N / n_class of the training graphs, batch 8
  (`_batches`, `random.Random(seed)`), at most 20 epochs, stop when the validation ransomware F1 at 0.5 (`metrics_at`, their
  rule) has not improved for 5 epochs. Our `fold_manifest(fold, seed)` replaces their `stratified_holdout`.
- Two states are kept: best validation F1 (theirs; `predictions.csv` at 0.5) and best validation AUC (`predictions_auc.csv` at
  our `val_threshold_of`). Stopping follows the F1 rule only, so the AUC state is the best AUC among the epochs the F1 rule
  trained (the harness's own gnn stops on the later of the two bests).
- Output. `RUNS/kfold{f}_x86_gnn_s{s}_team-gnn500|cb500/` from `common.write_run`, plus `history.csv` (epoch, loss, seconds,
  val F1 / macro-F1 / AUC, peak GPU MB), `best.pt`, `best_auc.pt`. `metrics.json` also holds `n_blocks_kept` and
  `n_blocks_total` (test sum / mean / max), `in_dim`, `peak_gpu_mb`, `epoch_secs_mean`, `device` and `torch_threads`, the recipe and, for cb500, the vectors
  used. In `predictions*.csv` `n_blocks_kept` and `n_blocks_total` are exact; `n_segments_total` is the number of kept
  functions (their export parser discards the total).
- `--smoke N`: N available training ids (both classes), the available validation ids (else a few training ids), 2 epochs,
  written only to a `tempfile.mkdtemp` directory; it never calls `write_run` and creates nothing under `RUNS`.

The matched `--subgraph cap --budget 500` arm starts with `python -m experiments.train ... --tag b500` (`RunConfig` accepts budgets from 100 blocks).

## train_b.py (arms sig500, sig500v2)

```
python -m experiments.team_pipelines.train_b --fold k --seed s --arm sig500|sig500v2 [--build DIR] [--smoke N] [--device cuda|cpu]
```

- Arms. `sig500` reads `RUNS/team/b_sig500` (source `gnn-331ce0b`), `sig500v2` reads `RUNS/team/b_sig500_v2` (source `main-c7899f0`).
  `--build DIR` overrides the directory of either arm; the build's `settings.json` names its source and the arm must match it
  (a mismatch stops). The build's own `preprocess_full` is imported for the embedder and the pooling (the constants and the
  `FiveChannelEmbedder` are the same in both; `tests/test_team_pipelines.py` compares them); `graph_full_loader` and `gat_full_model`
  exist only in the gnn source and always come from there. Everything else is identical for the two arms, and `metrics.json` records
  `build_dir`, `build_source` and `build_pins`.
- Inputs. The build directory, read only: the `.pt` of the ids in `checkpoint.json` (the embedder state saved
  there), `api_vocab.json`, `channel_embedder.pt`, `build_log.csv`. A full run needs every id of the fold that the build did not exclude, and stops otherwise.
- What is fitted where. The build's vocabulary covers all 1,810 binaries; per fold it is cut back to the training binaries:
  API ids (>= `FIRST_API_ID`) that no training binary uses become `UNK_API` in `instruction_ids[:, 3]`; the instruction
  vectors are re-embedded with `FiveChannelEmbedder` (the saved state, never trained) and re-pooled with
  `pool_block_embeddings`; the api-none / unresolved / known columns of `block_features` are recounted from the ids the way
  `build_model_inputs` counts them (known = id >= 76, unresolved = 75, none = 0 on a matched instruction). The opcode,
  operand and behaviour channels stay as built: a name that a real freeze would give `UNK_API` keeps its behaviour tag.
  Before the swap one sample is re-pooled unchanged and must reproduce its stored `block_embeddings` (1e-5) and api-kind
  columns (checked at every start, and by the tests).
- Graph and model. `graph_full_loader._node_features(.., 'both', scale=True)` (unit-length embedding plus scaled features,
  141 d) and `_edge_attr` (one-hot relation plus back-edge bit, edge_dim 8); `FullBlockGAT` with its defaults.
- Recipe (train_full_gat): AdamW lr 1e-3, weight decay 1e-4, class-weighted cross-entropy, batch 16, at most 40 epochs,
  patience 8 on the validation ransomware F1 at 0.5. Both selection rules, outputs and `--smoke` as in train_a.
  `n_segments_total` is empty (the build does not record the function count); `n_blocks_total` comes from `build_log.csv`.

## Teammate sources

`common.SOURCES` holds two sha256-pinned sources and the repository's own copy of Yanping's files; `import_team(module, source=...)`
imports from one of them and `check_pins` checks the files first. A mismatch raises with the file name.

| source | worktree (environment variable) | commit | files (under the worktree) |
|---|---|---|---|
| `gnn-331ce0b` (default) | `GT_TEAM_GNN_ROOT` | 331ce0b | `construct_blocks.py`, `train_cfg.py`, `CT_GAT/preprocessing/{preprocess_full, tokenize_asm_gz, graph_full_loader, gat_full_model, train_full_gat, transformer_model, token_mapping}.py` |
| `main-c7899f0` | `GT_TEAM_MAIN_ROOT` | c7899f0 | `CT_GAT/pipeline/s02_preprocess/{preprocess_full, tokenize_asm_gz}.py` |
| `repo` | none (this repository, `common.REPO_ROOT`) | the checkout | the same two files; sha256 computed from them when asked (`common.FilePins`), no fixed pin |

`preprocess_full` and `tokenize_asm_gz` exist in both, under the same module names. Importing one from a source first evicts the
other source's modules of that import (`preprocess_full`, `tokenize_asm_gz`, and the gnn source's `transformer_model` and
`token_mapping`) from `sys.modules` and puts that source's directories first on `sys.path`; a module that is already the pinned file
of the requested source stays loaded. Main's `preprocess_full` imports only `tokenize_asm_gz` (the opcode and operand vocabulary
sizes are literals), and main has no graph loader or model: `graph_full_loader` and `gat_full_model` always come from the gnn source.
Never modify, format or write into either worktree; the import switches bytecode caching off.

The runs reported in the review used torch 2.11 (CUDA 12.8 wheels), torch_geometric 2.8.1 and transformers 5.20.2. CodeBERT
(`microsoft/codebert-base`) is loaded from `HF_HOME` for anything that embeds with it. `build_b.py` needs neither transformers
nor a GPU.

## Yanping's streaming preprocess (step 21, main c7899f0)

Her change: `iter_text_lines` streams the listing, `align_sampled_blocks` keeps only the label spans that hold sampled
instructions, `process_sample` keeps its signature. What we found is in [TEAM_PIPELINES_REVIEW.md](../../docs/history/TEAM_PIPELINES_REVIEW.md) section 11. The tools:

```
# rebuild pipeline B with it (resumable, no exclusions); the first build is untouched
python -m experiments.team_pipelines.build_b --source main-c7899f0 --sample signal --num-blocks 500 --seed 0 --out RUNS/team/b_sig500_v2
# then train_b --arm sig500v2 on seeds 0 1 2 x folds 0-4 (CPU), as above

# equivalence on built samples: 60 ids of the first build (30 + 30, the 10 lowest-match included), same vocabulary and embedder
python -m experiments.team_pipelines.compare_builds pick --build RUNS/team/b_sig500 --out ids60.json
python -m experiments.team_pipelines.build_b --source main-c7899f0 --frozen-from RUNS/team/b_sig500 --out <tmp> --ids <the 60 ids>
python -m experiments.team_pipelines.compare_builds compare --a RUNS/team/b_sig500 --b <tmp> --ids-file ids60.json

# peak memory and wall time of one command (psutil poller over the process tree)
python -m experiments.team_pipelines.measure_rss --out m.json -- <python> -m experiments.team_pipelines.build_b ... --ids <id>

# share of binaries with < 50% and < 90% of the selected instructions matched, by label
# (--among DIR: only the ids DIR also holds, to read a finished build against a partial one)
python -m experiments.team_pipelines.match_rates --build RUNS/team/b_sig500_v2
python -m experiments.team_pipelines.match_rates --build RUNS/team/b_sig500 --among RUNS/team/b_sig500_v2
```

`compare_builds compare` checks, per sample, `block_keys`, `instruction_matched`, `instruction_ids` columns 0, 1, 2 and 4, column 3
(api), `block_features`, `edge_index` and `edge_attr`, and says which arrays differ and by how much. `python -m s04_evaluation.gates --stage team` has the
row `team sig500v2` and pairs it with `team sig500`, R7 and TF-IDF.

## Match diagnosis and acceptance (step 24)

Why the listing-to-CFG match fails and how to tell a fix from a non-fix: [TEAM_PIPELINES_REVIEW.md](../../docs/history/TEAM_PIPELINES_REVIEW.md) section 12. We deliver the
diagnosis and the acceptance checks; the fix is Yanping's. Both scripts read a build's saved tensors (and, for the gap reports,
its manifests) and call the teammates' pinned functions only through `import_team`; nothing is written under `RUNS/team/`.
Listing lines are parsed and aligned only by the teammates' functions; `match_diagnosis` itself reads raw listing text in two places, to count
label, `proc` and `endp` lines and to quote COLLAPSED FUNCTION markers. They run on the CPU (`CUDA_VISIBLE_DEVICES=-1`) from `CT_GAT/pipeline`.

```
# the diagnosis of the first build. The replay covers 80 samples (30 lowest-match, 10 lowest-match goodware, 20 random lossy ones
# per label), about 35 minutes on a busy machine; --replay-cache keeps one JSON per replayed sample so an interrupted run resumes,
# --cache keeps the manifest gap reports (8.8 GB of JSON, 70 s to read); --replay-worst 0 --replay-goodware 0 --replay-partial 0
# skips the replay
python -m experiments.team_pipelines.match_diagnosis --build RUNS/team/b_sig500 --cache <gaps.json> --replay-cache <dir> --json v1.json --markdown v1.md --csv v1.csv

# the same on the rebuild with her streaming preprocess, once RUNS/team/b_sig500_v2/build_summary.json reports processed 1810
# (on a partial build the tables cover the ids it holds)
python -m experiments.team_pipelines.match_diagnosis --build RUNS/team/b_sig500_v2 --cache <file_v2.json> --json v2.json --markdown v2.md --csv v2.csv

# PASS or FAIL of a build against the five acceptance criteria of the plan; exit status 0 only if all pass
python -m experiments.team_pipelines.match_acceptance --build RUNS/team/b_sig500
python -m experiments.team_pipelines.match_acceptance --build RUNS/team/b_sig500_v2          # her rebuilt output: run this
python -m experiments.team_pipelines.match_acceptance --build <her new dir> --skip-rebuild      # quick: criteria 1-4, never PASS
```

`match_diagnosis` prints, per label and per family, the share of binaries with under 50% and under 90% of their selected
instructions matched and the mean; the 10 worst families; the AUC of `1 - match rate` as a ransomware score with a 95% interval over
clusters (`fam:<family>` for ransomware, `id:<sample_id>` for goodware, as `s04_evaluation/stats.py`); the lost share by function
length; the gap reports of the manifests; and, for the lowest-match samples, a replay with the teammates' own alignment that sorts
every unmatched selected instruction into one cause (the list is in the module docstring). The replay reads the whole listing
and CFG of a sample, so a listing above `--max-listing-mb` (60 MB compressed) is skipped and listed. `--source` is read from the
build's `settings.json`.

`match_acceptance` checks (1) all 1,810 ids built and none excluded, (2) the under-90% share of the two labels within 2 points,
(3) the under-50% share below 1% for both, (4) the AUC of `1 - match rate` below 0.55, (5) a deterministic rebuild of 5 ids drawn
with `random.Random(0)`: `build_b --frozen-from <build>` into a temporary directory, every tensor `torch.equal` to the build's.
The rebuild keeps the build's own vocabulary and embedder, because a fresh vocabulary numbers API names differently and its tensors
could not be compared; it takes a minute or two per ordinary listing and several minutes for the largest ones. On a build that
is not complete, criteria 2-4 are computed on the ids it holds and marked partial. Tests: `tests/test_match_diagnosis.py`
(synthetic builds and a fake pinned module; one real small binary built and replayed with both sources).

## Tests

`tests/test_team_pipelines.py`, `test_team_exclusion.py`, `test_team_v2.py`, `test_match_diagnosis.py` (CPU). They run everywhere
on the tiny cache of `tests/tiny_cache.py` and on synthetic builds: every module imports without `torch_geometric`, `psutil`
and the worktrees (a subprocess with fake packages that raise `ImportError`); the environment variables name the missing input;
the pin check and `import_team` on a toy source; `paths_for` and `team_row` on a synthetic collection root (neither the bundle nor
`GT_TEAM_ASM_DIR` set) and on every half-configured pair; the pins of the `repo` source from a temp copy of its two files, a resume
that refuses a changed file, `import_team` of the `repo` files; `build_b`'s default `--out`, its stop without a cache index and its
refusal to overwrite a folder of sample files that is no build; the five test sets partition the ids and no family or identity group is shared by
train, val and test; a synthetic five-fold `write_run` arm is read by `s04_evaluation/stats.py` against `tfidf`, by `report.py`
and by the `team` stage of the gates, and its directory name equals `RunConfig.name`; explicit exclusions of pipeline B;
`compare_builds`, `match_rates` and `match_diagnosis` / `match_acceptance` on synthetic builds (a fake pinned module).

They skip with a reason without their input: both pin sets pass on the worktrees and a tampered copy of either raises, importing
one source never leaves the other's modules in `sys.modules` (`GT_TEAM_GNN_ROOT`, `GT_TEAM_MAIN_ROOT`); the trainers on a
synthetic `run_arm` (`torch_geometric`); `measure_rss` (`psutil`); the checks over all 1,810 binaries, `paths_for` on every id, the
builds, re-pooling, the UNK swap, `cb500` against `encode_binary_gnn` of the matched arm and the three `--smoke` paths (marked
`real_data`: the cache, `GT_BUNDLE`, `GT_TEAM_ASM_DIR`, and the build or vector directories where they read them).
