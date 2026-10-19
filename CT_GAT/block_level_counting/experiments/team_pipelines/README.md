# Team pipelines under our folds

Adapters that run the teammates' GAT pipelines (A: construct_blocks + train_cfg; B: preprocess_full + graph_full_loader +
gat_full_model) under our 5-fold x86 family holdout, so their results sit in the same tables as ours
(`stats.py`, `report.py`, `gates.py`). Plan: `~/.claude/orchestrate/rdmp-gt/rubrics/team_pipelines_plan.md`.

Run every script as a module from `CT_GAT/block_level_counting` (`python -m experiments.team_pipelines.<name>`), not as
a file: as a file this directory is first on `sys.path` and its `common.py` shadows our `common` package.

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
- `build_b.py`: pipeline B preprocessing, see below.

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

## Pins and environment

- Teammate code: review worktree `C:/Users/chaoa/Downloads/rdmp-team-gnn`, detached at `331ce0b`. Pinned files:
  `construct_blocks.py`, `train_cfg.py`, `CT_GAT/preprocessing/{preprocess_full, tokenize_asm_gz, graph_full_loader,
  gat_full_model, train_full_gat, transformer_model, token_mapping}.py` (the last two are imported by `preprocess_full`).
  Never modify, format or write into that worktree.
- Python `C:/Users/chaoa/Downloads/gt_venv/Scripts/python.exe`: torch 2.11.0+cu128, torch_geometric 2.8.1,
  transformers 5.20.2. CodeBERT (`microsoft/codebert-base`) is cached under `C:/Users/chaoa/Downloads/hf_home`; set
  `HF_HOME=C:/Users/chaoa/Downloads/hf_home` for anything that loads it. `build_b.py` needs neither transformers nor a GPU.

## Tests

`tests/test_team_pipelines.py` (CPU, about 25 s): the five test sets partition the 1,810 ids and no family or identity
group is shared by train, val and test; `paths_for` resolves all 1,810 ids; a synthetic five-fold `write_run` arm is
read by `evaluation/stats.py` against `tfidf` and by `report.py`, and its directory name equals `RunConfig.name`; the
pin check raises on a tampered temp copy (never on the review worktree).
