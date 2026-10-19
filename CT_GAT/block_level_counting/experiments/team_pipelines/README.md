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
  sig500`, which waits until `RUNS/team/b_sig500/build_summary.json` reports 1810 of 1810 binaries. Each trainer uses `TEAM_THREADS`
  torch threads (default 3, 6 of the 16 logical cores for the two A lanes while the build and the GPU queue's loaders run).
  Stop it with `touch RUNS/STOP_TEAM` (a file of its own; `RUNS/STOP` stops the GPU queue and not this); the run in flight finishes.
  One retry per failed run; logs in `RUNS/logs/<name>.log`.
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
  - `import_team(module)`: imports a teammate module from the review worktree `C:/Users/chaoa/Downloads/rdmp-team-gnn`
    after checking the sha256 of every teammate file the import runs against `PINNED_SHA256`. A mismatch raises with
    the file name. Nothing is written into that worktree (bytecode caching is off during the import).
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
- `build_b.py`, `build_a.py`: the two preprocessing builds; `train_a.py`, `train_b.py`: the trainers; `fit.py`: the
  teammates' training loop on our manifests (shared by both trainers). See below.

## build_b.py

```
python -m experiments.team_pipelines.build_b --sample signal --num-blocks 500 --seed 0 --out RUNS/team/b_sig500
       [--limit N] [--ids <sample_id> ...]
```

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

## train_b.py (arm sig500)

```
python -m experiments.team_pipelines.train_b --fold k --seed s --arm sig500 [--smoke N] [--device cuda|cpu]
```

- Inputs. `RUNS/team/b_sig500` of build_b, read only: the `.pt` of the ids in `checkpoint.json` (the embedder state saved
  there), `api_vocab.json`, `channel_embedder.pt`, `build_log.csv`. A full run needs all 1,810 ids and stops otherwise.
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

## Pins and environment

- Teammate code: review worktree `C:/Users/chaoa/Downloads/rdmp-team-gnn`, detached at `331ce0b`. Pinned files:
  `construct_blocks.py`, `train_cfg.py`, `CT_GAT/preprocessing/{preprocess_full, tokenize_asm_gz, graph_full_loader,
  gat_full_model, train_full_gat, transformer_model, token_mapping}.py` (the last two are imported by `preprocess_full`).
  Never modify, format or write into that worktree.
- Python `C:/Users/chaoa/Downloads/gt_venv/Scripts/python.exe`: torch 2.11.0+cu128, torch_geometric 2.8.1,
  transformers 5.20.2. CodeBERT (`microsoft/codebert-base`) is cached under `C:/Users/chaoa/Downloads/hf_home`; set
  `HF_HOME=C:/Users/chaoa/Downloads/hf_home` for anything that loads it. `build_b.py` needs neither transformers nor a GPU.

## Tests

`tests/test_team_pipelines.py` (CPU, about 35 s): the five test sets partition the 1,810 ids and no family or identity
group is shared by train, val and test; `paths_for` resolves all 1,810 ids; a synthetic five-fold `write_run` arm is
read by `evaluation/stats.py` against `tfidf` and by `report.py`, and its directory name equals `RunConfig.name`; the
pin check raises on a tampered temp copy (never on the review worktree).

Step 18 adds: the UNK remap touches only ids unseen in train; re-pooling reproduces the stored embeddings and api-kind
columns (and fails on a perturbed payload); an unswapped graph equals `graph_full_loader.load_full_graph`; the cb500 blocks,
vectors and edges equal `encode_binary_gnn` of the matched arm for three binaries (`ours_cap0_ref`); a synthetic `run_arm`
over all 1,810 ids writes a run directory; the three `--smoke` paths leave a patched `RUNS` empty and never call `write_run`;
`build_a` builds, resumes and stops on a bad id. Tests that need build output or vectors skip when absent.
