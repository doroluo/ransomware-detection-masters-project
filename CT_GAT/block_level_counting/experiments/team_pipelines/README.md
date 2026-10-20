# Team pipelines under our folds

Adapters that run the teammates' GAT pipelines (A: construct_blocks + train_cfg; B: preprocess_full + graph_full_loader +
gat_full_model) under our 5-fold x86 family holdout, so their results sit in the same tables as ours
(`stats.py`, `report.py`, `gates.py`). Plan: `~/.claude/orchestrate/rdmp-gt/rubrics/team_pipelines_plan.md`.

Run every script as a module from `CT_GAT/block_level_counting` (`python -m experiments.team_pipelines.<name>`), not as
a file: as a file this directory is first on `sys.path` and its `common.py` shadows our `common` package.

## Team pipelines: how to run

Two scripts, both resumable (a run whose `RUNS/<name>/metrics.json` exists is skipped), both with `--dry` (prints every run in
order and writes nothing). Run them from `CT_GAT/block_level_counting`.

```
nohup bash scripts/team_runs.sh     > RUNS/logs/team_runs.log 2>&1 &    # CPU, now, next to the GPU queue
nohup bash scripts/team_gpu_runs.sh > RUNS/logs/team_gpu_runs.log 2>&1 & # GPU, waits for the GPU queue to finish
```

- `scripts/team_runs.sh` (CPU only: `--device cpu` and `CUDA_VISIBLE_DEVICES=-1`). Three lanes side by side, each over seeds 0 1 2
  and folds 0-4 (45 runs): `train_a --arm gnn500`, `train_a --arm cb500` (CodeBERT vectors `ours_cap0_codebert`), and `train_b --arm
  sig500`, which waits until `RUNS/team/b_sig500/build_summary.json` accounts for all 1810 binaries (processed + excluded == 1810). Each trainer uses `TEAM_THREADS`
  torch threads (default 3, 6 of the 16 logical cores for the two A lanes while the build and the GPU queue's loaders run).
  Stop it with `touch RUNS/STOP_TEAM` (a file of its own; `RUNS/STOP` stops the GPU queue and not this); the run in flight finishes.
  One retry per failed run; logs in `RUNS/logs/<name>.log`.
- `scripts/team_runs.sh --v2` runs the one lane `sig500v2` instead: `train_b --arm sig500v2` on `RUNS/team/b_sig500_v2` (the rebuild with
  Yanping's streaming preprocess, below), seeds 0 1 2 x folds 0-4, run names `kfold{f}_x86_gnn_s{seed}_team-sig500v2`, CPU only. It
  waits until that build's summary reports all 1810 binaries processed and none excluded. Same stop file, retry and skip rules;
  `--dry` and `--check-build` work with it.
- `scripts/team_gpu_runs.sh` (GPU; started by the owner once the queue is done). Waits for `master4 done` in
  `RUNS/logs/master4.log`, then runs our matched arms (`train.py --protocol kfold --arch x86 --model gnn --mask-blocks 0.15 --subgraph
  cap --budget 500 --tag b500`, typed and `--control no_edges`, seeds 0 1 2, folds 0-4: 30 runs, directories
  `kfold<f>_x86_gnn[-noedges]-mblk15-sg-cap500_s<seed>_b500`), then `queue_gnn.sh --stage paths_codebert`, then `report.py`, `gates.py
  --stage team > RUNS/gates_team.md` and `gates.py --stage paths_codebert > RUNS/gates_paths_codebert.md`. It stops on `RUNS/STOP`.
- `python gates.py --stage team` reads whatever is on disk: TF-IDF, the header-imports LR, R7, R2, our b500 arms, the three team arms
  and path 2 on CodeBERT vectors; AUC, PR-AUC, macro-F1, accuracy, precision, recall and the confusion matrix per arm (seed 0
  pooled), the paired differences of each team arm against TF-IDF, R7 and our b500 typed arm (and cb500 against path 2, the b500
  typed arm against R7), with the pre-registered rule text. A missing arm prints as missing.

`RunConfig` accepts `--budget` from 100 blocks (0 is the binary cap). Below 1000 the variant suffix spells the budget out
(`sg-cap500`, `sg-cap128`), so two budgets cannot share a directory name; the matched arms still pass `--tag b500`. Every
existing run name is unchanged (`tests/test_gnn_config.py::test_existing_run_names_unchanged`).

CPU time per epoch (3 threads, Ryzen 7 9800X3D shared with the GPU queue and the build, fold 0): gnn500 about 6 ms per training
graph (57-d features), cb500 about 6 ms (768-d float16 vectors cast per batch), sig500 about 8 ms (141-d features, edge
attributes). A fold has about 1,200 training and 230 validation binaries, so an epoch takes roughly 8 s (gnn500, cb500) and
11 s (sig500); the recipes stop after at most 20 (A) and 40 (B) epochs, which bounds a fold at about 3 and 7 minutes of training
plus under a minute of loading and prediction. The 15 runs of an arm therefore take at most about 1 h (A) and 2 h (B).

## What is here

- `common.py`
  - `import_team(module, source='gnn-331ce0b')`: imports a teammate module from the review worktree of `source`
    (`rdmp-team-gnn` or `rdmp-team-main`, see "Two teammate sources") after checking the sha256 of every teammate file the import
    runs against that source's pins. A mismatch raises with the file name. Nothing is written into the worktree (bytecode
    caching is off during the import).
  - `fold_manifest(fold, seed)`: our `make_manifest` for protocol `kfold`, arch `x86`, `val_seed=None`, as the
    `kfold_x86_gnn-mblk15_s{seed}` runs use. `all_ids()`: the 1,810 test ids of the five folds, sorted.
  - `paths_for(sample_id)`: the CFG (`Mendeley_GNN_v1/samples.csv` `cfg_path`) and the IDA listing
    (`ida_asm_mendeley_20261006/asm/<sha>.asm.gz`). A missing row or file raises.
  - `team_config(fold, seed, arm)` and `write_run(...)`: a run directory `kfold{f}_x86_gnn_s{seed}_team-<arm>` with
    `config.json`, `manifest.json`, `predictions.csv` (F1 state, `pred = prob >= 0.5`, as `training/train.py`),
    `predictions_auc.csv` (AUC state, `pred = prob >= threshold_val`) and `metrics.json`. The columns and the metrics are
    `training/train.py`'s (`write_predictions`, `evaluate`) over `evaluation/metrics.py`; the rows must be exactly the
    manifest's validation and test ids. `manifest.json` is not in the rubric but `report.py` reads it for the floors.
  - `team_row(sample_id)`: the row `construct_blocks` expects (bundle `samples.csv` row, int `label`, `cfg_file`).
- `match_diagnosis.py`, `match_acceptance.py`: why the listing-to-CFG match fails and the PASS/FAIL checks for a fix (step 24, below).
- `build_b.py`, `build_a.py`: the two preprocessing builds; `train_a.py`, `train_b.py`: the trainers; `fit.py`: the
  teammates' training loop on our manifests (shared by both trainers). See below.

## build_b.py

```
python -m experiments.team_pipelines.build_b --sample signal --num-blocks 500 --seed 0 --out RUNS/team/b_sig500
       [--source gnn-331ce0b|main-c7899f0] [--limit N] [--ids <sample_id> ...] [--frozen-from <build dir>]
```

`--source` picks which pinned teammate code runs (see "Two teammate sources" below); the default `gnn-331ce0b` is what built
`RUNS/team/b_sig500`, so the first build stays reproducible. `settings.json` records `source` and the sha256 `pins` of the files that
`preprocess_full` runs; the checkpoint's settings carry `source` too, so resuming a directory with another source stops (a checkpoint
without `source`, as the first build wrote, reads as `gnn-331ce0b`). `--frozen-from DIR` starts from DIR's `api_vocab.json` and
`channel_embedder.pt` and calls `process_sample(freeze_vocab=True)`: ids and embeddings of two sources are then comparable sample by
sample (the equivalence check).

`RUNS/` at the start of `--out` is `graph_transformer_runs/`. For each of the 1,810 ids (sorted) it calls the
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
`team_runs.sh` waits for. Pass the same `--exclude` on every resume. The one exclusion so far is
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

The matched `--subgraph cap --budget 500` arm starts with `train.py --tag b500` (`RunConfig` accepts budgets from 100 blocks).

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

## Three teammate sources

`common.SOURCES` holds three sha256-pinned sources; `import_team(module, source=...)` imports from one of them and `check_pins` checks
the files first. A mismatch raises with the file name.

| source | review worktree | commit | files (under the worktree) |
|---|---|---|---|
| `gnn-331ce0b` (default) | `C:/Users/chaoa/Downloads/rdmp-team-gnn` | 331ce0b | `construct_blocks.py`, `train_cfg.py`, `CT_GAT/preprocessing/{preprocess_full, tokenize_asm_gz, graph_full_loader, gat_full_model, train_full_gat, transformer_model, token_mapping}.py` |
| `main-c7899f0` | `C:/Users/chaoa/Downloads/rdmp-team-main` | c7899f0 | `CT_GAT/pipeline/s02_preprocess/{preprocess_full, tokenize_asm_gz}.py` |
| `main-fix` | `C:/Users/chaoa/Downloads/rdmp-team-yfix` (detached) | head of branch `s02-endp-library-fix` (the PR into main) | the same two files with the `endp` fix and the library-function skip (`build_b --include-library` restores the old sampling) |

`preprocess_full` and `tokenize_asm_gz` exist in both, under the same module names. Importing one from a source first evicts the
other source's modules of that import (`preprocess_full`, `tokenize_asm_gz`, and the gnn source's `transformer_model` and
`token_mapping`) from `sys.modules` and puts that source's directories first on `sys.path`; a module that is already the pinned file
of the requested source stays loaded. Main's `preprocess_full` imports only `tokenize_asm_gz` (the opcode and operand vocabulary
sizes are literals), and main has no graph loader or model: `graph_full_loader` and `gat_full_model` always come from the gnn source.
Never modify, format or write into either worktree; the scripts set `PYTHONDONTWRITEBYTECODE=1`.

Python `C:/Users/chaoa/Downloads/gt_venv/Scripts/python.exe`: torch 2.11.0+cu128, torch_geometric 2.8.1,
transformers 5.20.2. CodeBERT (`microsoft/codebert-base`) is cached under `C:/Users/chaoa/Downloads/hf_home`; set
`HF_HOME=C:/Users/chaoa/Downloads/hf_home` for anything that loads it. `build_b.py` needs neither transformers nor a GPU.

## Yanping's streaming preprocess (step 21, main c7899f0)

Her change: `iter_text_lines` streams the listing, `align_sampled_blocks` keeps only the label spans that hold sampled
instructions, `process_sample` keeps its signature. What we found is in `docs/TEAM_PIPELINES_REVIEW.md` section 11. The tools:

```
# rebuild pipeline B with it (resumable, no exclusions); the first build is untouched
python -m experiments.team_pipelines.build_b --source main-c7899f0 --sample signal --num-blocks 500 --seed 0 --out RUNS/team/b_sig500_v2
bash scripts/team_runs.sh --v2          # waits for that build, then trains sig500v2 on seeds 0 1 2 x folds 0-4 (CPU)

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
(api), `block_features`, `edge_index` and `edge_attr`, and says which arrays differ and by how much. `gates.py --stage team` has the
row `team sig500v2` and pairs it with `team sig500`, R7 and TF-IDF.

## Match diagnosis and acceptance (step 24)

Why the listing-to-CFG match fails and how to tell a fix from a non-fix: `docs/TEAM_PIPELINES_REVIEW.md` section 12. We deliver the
diagnosis and the acceptance checks; the fix is Yanping's. Both scripts read a build's saved tensors (and, for the gap reports,
its manifests) and call the teammates' pinned functions only through `import_team`; nothing is written under `RUNS/team/`.
Listing lines are parsed and aligned only by the teammates' functions; `match_diagnosis` itself reads raw listing text in two places, to count
label, `proc` and `endp` lines and to quote COLLAPSED FUNCTION markers. Python is `C:/Users/chaoa/Downloads/gt_venv/Scripts/python.exe`, CPU only (`CUDA_VISIBLE_DEVICES=-1`), run from the
`block_level_counting` root.

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
clusters (`fam:<family>` for ransomware, `id:<sample_id>` for goodware, as `evaluation/stats.py`); the lost share by function
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

`tests/test_team_v2.py` (step 21): `build_b --source main-c7899f0` records source and pins and stops on another source; `--frozen-from`
keeps the vocabulary and embedder; `train_b` arms read the build of their source and stop on a mismatch; `compare_builds`,
`match_rates` and `measure_rss` on synthetic builds; the `--v2` lane's dry names equal `RunConfig` names; the gates pairs. In
`tests/test_team_pipelines.py` both pin sets pass on their worktrees, a tampered copy of either raises, and importing one source never
leaves the other's modules in `sys.modules`.

`tests/test_team_pipelines.py` (CPU, about 35 s): the five test sets partition the 1,810 ids and no family or identity
group is shared by train, val and test; `paths_for` resolves all 1,810 ids; a synthetic five-fold `write_run` arm is
read by `evaluation/stats.py` against `tfidf` and by `report.py`, and its directory name equals `RunConfig.name`; the
pin check raises on a tampered temp copy (never on the review worktree).

Step 18 adds: the UNK remap touches only ids unseen in train; re-pooling reproduces the stored embeddings and api-kind
columns (and fails on a perturbed payload); an unswapped graph equals `graph_full_loader.load_full_graph`; the cb500 blocks,
vectors and edges equal `encode_binary_gnn` of the matched arm for three binaries (`ours_cap0_ref`); a synthetic `run_arm`
over all 1,810 ids writes a run directory; the three `--smoke` paths leave a patched `RUNS` empty and never call `write_run`;
`build_a` builds, resumes and stops on a bad id. Tests that need build output or vectors skip when absent.
