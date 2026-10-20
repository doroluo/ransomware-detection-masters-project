# History: how the pipeline got its numbers

These are the working documents of the `3-block-level-counting` branch, copied as they were when the experiments finished
(2026-10-20), plus the final gate tables. They are the record behind the headline in [`../../README.md`](../../README.md): read them
for why an option exists and what was measured, not for the commands, which have moved (table below).

| file | what it is |
|---|---|
| [`PLAN_GNN.md`](PLAN_GNN.md) | the plan of the edge-typed hierarchical GNN, its control ladder (R0 TF-IDF, R2 no edges, R4 shuffled edges, R5 untyped, R7 typed), the pre-registered rule, every decision, and the "Final results" section (2026-10-20) |
| [`RESULTS.md`](RESULTS.md) | `report.py` over every run directory: one row per configuration, mean +- sd over seeds |
| [`TEAM_PIPELINES_REVIEW.md`](TEAM_PIPELINES_REVIEW.md) | the review of the teammates' two GAT pipelines (`gnn`, `yanping`), how we ran them on our folds, the adopted target architecture (section 9, drawn in the root README), the Model 1 adapter contract, and why the listing-to-CFG match fails (section 12) |
| [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md) | the source and the departure of every layer-level choice |
| [`GRAPHS_AND_TOKENS.md`](GRAPHS_AND_TOKENS.md) | which file builds the graph, which tokenizes, and where each is described |

The final gate summaries, written by `python -m s04_evaluation.gates --stage <stage>` on the runs of the author's machine (the
numbers of the "Final results" section of `PLAN_GNN.md` are read from them). All are seed 0 pooled tables with the paired
differences over the seeds complete in both arms:

| file | stage | the plan that produced its runs |
|---|---|---|
| [`gates_noapi.md`](gates_noapi.md) | `noapi`: the api-token ablation | `noapi`, `r2_noapi` |
| [`gates_team.md`](gates_team.md) | `team`: the teammates' pipelines under our folds, our 500-block arms | `team`, `paths_codebert`, and the trainers of `experiments/team_pipelines` |
| [`gates_v3.md`](gates_v3.md) | `v3`: the combined cache, all architectures, arch-balanced arms | `v3` |
| [`gates_paths_codebert.md`](gates_paths_codebert.md) | `paths_codebert`: Model 3, the GNN on CodeBERT block vectors | `paths_codebert` |
| [`gates_model1.md`](gates_model1.md) | `model1`: the token-sequence transformer | `model1` |
| [`gates_imports_header.md`](gates_imports_header.md) | `imports`: the header-imports baseline and R7 with header imports (the queue named the file `gates_imports_header.md`) | `imports`, `imports_header` |
| [`gates_subgraph.md`](gates_subgraph.md) | `subgraph`: the subgraph budgets and the block weightings | `subgraph` |

## Where the old names went

The documents name files as they were on the branch. In `CT_GAT/pipeline`:

| on the branch | now |
|---|---|
| `train.py`, `training/train.py` | `experiments/train.py` (`python -m experiments.train`) |
| `build_cache.py`, `check_cache.py`, `pipeline/s02_build_cfg_cache/` | `s02_preprocess/cache/` (new: `folds.py`, the fold CSV builder) |
| `pipeline/s03_tokenizer/`, `pipeline/s04_select_subgraph/` | `s02_preprocess/tokenizer/`, `s02_preprocess/subgraph.py` |
| `pipeline/s05_*` ... `pipeline/s11_*`, `pipeline/model.py`, `pipeline/batching.py` | `s03b_cfg_feature/gnn/` (same submodule names) |
| `baselines.py`, `report.py`, `stats.py`, `val_threshold.py`, `gates.py`, `evaluation/` | `s04_evaluation/`; the `--stage` definitions are `experiments/stages/<name>.py` |
| `evaluation/run_io.py`, `transformer_paths/reader.py` | `common/run_io.py`, `common/reader.py` |
| `transformer_paths/` | `s03c_cfg_transformer/` (`codebert_encoder.py` and `reference_encoder.py` are `encoders/codebert.py` and `encoders/reference.py`) |
| `token_sequence/` | `s03a_transformer/token_sequence/` |
| `queue_gnn.sh --stage <s>`, `scripts/*_runs.sh`, `stop_queue.sh` | `python -m experiments.queue --plan experiments/plans/<s>.txt`, `--stop` |
| `subgraph_chain.py` | `python -m experiments.subgraph_chain` |
| `experiments/team_pipelines/` | the same path (needs `GT_TEAM_GNN_ROOT`, `GT_TEAM_MAIN_ROOT`, `torch_geometric`) |
| `experiments/side_arrays/` | `experiments/side_arrays/import_reader_side.py` only (the operand-role side arrays that `--vocab yanping|hybrid` reads) |
| `RUNS`, `CACHE` in the text | `GT_RUNS`, `GT_CACHE` / `datasets.json` (`common/paths.py`) |
| `docs/<file>.md` | `docs/history/<file>.md` |

Not carried over: `smoke_gnn_memory.py`, the def-use and API-tag side arrays (`build_side.py`) and the WL baseline
(`wl_baseline.py`), the shell queues themselves (the plans replace them), and `docs/PLAN.md` (the first plan, of the graph transformer
that was removed). The documents are otherwise unchanged except for machine paths, which are written as `GT_RUNS` or `<cache>`.
