# Expanded grids for the rough draft

Run from `CT_GAT/pipeline` using the CUDA environment described in [the original grid guide](../model_grid/README.md).
Model 1 and Model 3 learn their position-aware instruction Transformers end to end. **There is no frozen-Transformer Model 3 arm in this search.** Model 2 remains the 77-feature handcrafted GAT baseline. Mean and learned CLS pooling are paired experiments for Model 3.

## Teammate commands

Use the same prepared 500-block build and grouped index on both machines. Replace the example paths with your local copies. Do not place two GPU jobs on the same GPU at once.

RTX 5080, Models 1 and 3, **30 hours total across both models**:

```powershell
python grid_search.py --wide --models 1 3 --dataset combined --build C:/data/b_sig500_combined_v3 --index C:/data/graph_transformer_cache_combined/index.csv --build-250 C:/data/b_sig250_combined_v3 --build-uncapped C:/data/b_siguncapped_combined_v3 --gpu-hours 30 --reuse-from runs/draft_combined_20261028 --out runs/wide_combined
```

RTX 5070 Ti, Model 2, its own independent budget:

```powershell
python grid_search.py --wide --models 2 --dataset combined --build C:/data/b_sig500_combined_v3 --index C:/data/graph_transformer_cache_combined/index.csv --build-250 C:/data/b_sig250_combined_v3 --build-uncapped C:/data/b_siguncapped_combined_v3 --gpu-hours 30 --out runs/wide_model2
```

`--build-250` and `--build-uncapped` are optional: omitted builds are recorded as pending coverage, while the 500-block search proceeds. To prepare them, see below. `--plan-only` prints case counts without training. `--smoke 8 --device cpu --wait-build-hours 0 --out runs/smoke_wide` exercises small real-data subsets without touching test tensors. For a single model use `--models 1`, `--models 2`, or `--models 3`.

To run all three models locally, use `--models 1 2 3` with one shared `--gpu-hours 30` budget. The runner trains them sequentially. A short RTX 5080 overlap test showed substantial per-model slowdown; see [the concurrency measurement](CONCURRENCY.md). Existing active studies require a new plan/output when adding a model; preserve completed trials, partial checkpoints and the cumulative budget when migrating.

The existing `python grid_search.py --models ...` small grid remains available; **add `--wide` for this expanded search.** Repeat exactly the same wide command and output directory to resume. Use a new output directory when changing the grid, source, cohort, budget, or training protocol.

## What is searched

These are staged Cartesian grids. The best validation settings from one stage are fixed for the next; this is not one enormous Cartesian product of all axes.

| Stage / model | Values | Planned settings |
|---|---|---:|
| Optimizer, Models 1/2 | LR: 0.00003, 0.0001, 0.0003, 0.001, 0.003; dropout: 0, 0.1, 0.3, 0.5; weight decay: 0, 0.0001, 0.01 | 60 each |
| Optimizer, Model 3 | Same optimizer grid × mean/CLS | 120 |
| Architecture, Model 1 | embedding width 32/64/128 × Transformer depth 1/2/4 × sequence limit 128/512/1024 instructions | 27 |
| Architecture, Model 2 | GAT hidden width 64/128/256 × depth 1/2/3/4 × heads 2/4/8 | 36 |
| Architecture, Model 3 | embedding width 32/64/128 × Transformer depth 1/2/3 × GAT hidden width 64/128/256 × GAT depth 1/2/3 × mean/CLS | 162 |
| Instruction limit, Model 3 | 32/64/128/256 real instructions per block × mean/CLS, using the selected architecture | 8 |
| Coverage, all models | 250 / 500 / uncapped blocks, using the selected optimizer and the common reference architecture | 3 each, or 6 for paired Model 3 |
| Confirmation | selected 500-block model, seeds 0/1/2; both matched Model 3 pooling arms | 3 each, or 6 for Model 3 |

Transformer heads stay at 4; Model 3 GAT heads stay at 4. Reference architecture: embedding width 64, Transformer depth 2, GAT hidden width 128/depth 2. Batch sizes are 32/16/8 for Models 1/2/3 in the primary search. Coverage uses batch size 1 and CPU activation offload **for every cap** so a cap comparison does not also change those settings.

Primary screening uses 12 epochs per configuration and seed 0. Coverage uses **2 epochs per cap** as a controlled feasibility/coverage pilot; do not compare those scores directly with 12-epoch screening or call them final full-graph results. `--coverage-epochs` can increase the matched coverage budget in a new final-paper study. Confirmation uses up to 40 epochs, patience 8. Only validation AUC and then macro-F1 at threshold 0.5 select configurations. For Model 3 selection averages the two pooling arms under otherwise identical settings; incomplete pairs cannot win. Thresholds are fitted on validation before test evaluation.

The default 30-hour budget reserves 8 hours for optimizer screening, 8 for architecture screening, 6 for instruction-limit and coverage work, and 8 for repeat-seed confirmation. This is shared across the selected local models; it is not 30 hours per model or a combined budget across two computers. Data preparation, waiting and report writing are excluded. Active model/training/validation/checkpoint/test wall time is counted conservatively, not hardware utilization time. It is checkpointed every two seconds. The runner stops at an epoch boundary, so a phase/total limit can overrun by its final epoch. Unused phase allocations are not automatically spent elsewhere. Architecture grids visit single-axis changes before interactions; budget-limited grids are explicitly marked partial. Finishing the queue does not mean every planned setting completed.

## Scheduling ready work

The queue completes optimizer, architecture and Model 3 instruction-limit selection on validation, then freezes `primary_selection.json` before any new outer-test evaluation. It next runs the three-seed confirmations on the ready 500-block build. Coverage pilots do not select the primary models, so they need not block these results while an uncapped build is being prepared.

After confirmation, the queue runs every ready 250/500/uncapped coverage case across all models before waiting for missing builds. Each build's wait timeout starts when the queue actually reaches its deferred wait. Validation-only coverage receives fresh prepared inputs so it does not inherit test tensors loaded for confirmation. A resume refuses to overwrite a different frozen primary selection.

The instruction-limit search now has priority within the same six-hour instruction/coverage allowance; the remaining allowance serves coverage pilots. This scheduling priority can change which optional pilots fit into a time-limited study, but changes no per-trial model, optimizer, data, split, epoch count, seed, pooling pairing or numerical implementation. Existing study plans need an explicit audited scheduler-only source amendment before resuming under this version; ordinary source/plan checks remain strict.

## Coverage inputs

250 blocks can be derived **exactly** from the existing signal selector's ranked prefix. This uses the manifest's selection order and then preserves canonical tensor/instruction order; it does not take the first 250 rows of a canonically ordered graph. Original edges are induced on the retained blocks, and instruction ownership/API tokens remain attached to their original instructions.

```powershell
python -m experiments.grid_study.coverage --from-build C:/data/b_sig500_combined_v3 --out C:/data/b_sig250_combined_v3
```

An uncapped build must be regenerated from the `.asm.gz` and structured CFG `.jsonl.gz` inputs. Configure `GT_COLLECTIONS` and `GT_CACHE` or `datasets.json` as in the repository's data guide. For the current combined cohort:

```powershell
python -m experiments.team_pipelines.build_b --source repo --protocol kfold_v3 --arch all --sample signal --num-blocks 0 --seed 0 --seeded-rows --max-instructions 4000000 --out C:/data/b_siguncapped_combined_v3
```

`--num-blocks 0` is genuinely uncapped **within the signal selector's eligible blocks**. Library functions are normally excluded (all-library samples have a documented fallback). The same 34 samples above four million raw instructions remain excluded in the current combined cohort. This is not every instruction in every original executable. Models 1/3 still have their declared instruction limits and pooling; no representation here is lossless. Existing one-million-block legacy builds are not silently relabeled uncapped.

The runner verifies identical split IDs across caps and rejects silent sample removal. Uncapped processed tensors are staged on disk to avoid holding every full graph in RAM. It waits up to `--wait-build-hours 12` for a requested build's completion summary without charging GPU time. OOM or nonfinite trials receive `failure.json`, never fabricated scores; incomplete cases remain visible.

## Files for the paper, available as results arrive

- Root `results.csv`: all completed settings, complete hyperparameters, seeds, validation/test stage, detection rate, FPR, AUC, macro-F1, timing, and original result paths.
- Root `REPORT.md`: rolling stage counts and test mean ± sample SD across seeds; these are one outer fold, not five-fold CV or confidence intervals.
- Root `status.json`, `budget.json`, `plan.json`, `primary_selection.json`: current work, cumulative time, exact sources/protocol and frozen primary choices. Budget accounting is durably saved first in checksummed `budget.journal/*.json` snapshots (the latest eight are retained). A Windows viewer may leave `budget.json` stale; resume recovers the newest valid cumulative record, including legacy `budget.json.tmp`, without resetting consumed time. A ledger write failure requests a checkpoint stop. Rolling report locks retain the newest report at `REPORT.md.tmp` or `results.csv.tmp`.
- Per trial: `config.json`, `history.json`, `best.pt`, `last.pt`, `metrics.json`, and prediction CSVs. Each epoch is checkpointed. Interrupted cases have history/checkpoints but no completed `metrics.json`.
- Per stage: `planned_cases.json` and `selection.json`, including planned/completed counts and whether the grid is partial.
- Per input: `dataset.json` with split IDs/payload hashes/exclusions and `coverage.json` with retained instruction/API/block counts.
- Reused trials retain their original config/fingerprint and add `reused_from.json`. Reuse requires identical tensors, split, model, training code, seed, epoch budget, optimizer and metrics. Only a CLI-dispatcher change is ignored. Old frozen/zero-position or otherwise incompatible models cannot be reused as these runs.

Export scientific plots at any point:

```powershell
python -m experiments.grid_study.plot --out runs/wide_combined
```

This writes learning-rate/detection/FPR plots under `figures/`, separated by weight decay and pooling. They show completed validation settings; one-seed screens have no seed error bars. Hyperparameter sensitivity is not a causal measure of feature or ransomware-behavior importance. Block/API attribution still requires a separately reviewed behavior evaluation.

To stop gracefully, create `STOP` in `runs/wide_combined/model1` (or `model2` / `model3`) for the active model; remove your STOP file before resuming. Existing results are never discarded. Do not edit source files underneath an active queue; use a separate checkout for further changes.
