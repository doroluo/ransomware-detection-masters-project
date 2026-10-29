# Validation of the grid runners

Validated on 2026-10-28 against main base `00543afa`.

**13 automated tests passed**, covering order-sensitive summaries, padding and empty blocks, trainable CLS, matching initial weights, instruction/API alignment, test isolation, grouped splits, exact CPU checkpoint resume, and the screen/confirm/resume workflow for all three models.

CUDA smoke training completed for all three models, including both Model 3 pooling variants. A two-epoch full train/validation benchmark then completed on the existing combined build. These are feasibility/timing checks, not final accuracy claims.

| Model | Pooling | Seconds/epoch (2-epoch mean) | Peak allocated GPU MiB |
|---|---|---:|---:|
| 1 | mean | 0.91 | 132 |
| 2 | graph_mean_max | 1.73 | 313 |
| 3 | cls | 19.62 | 1710 |
| 3 | mean | 17.88 | 1685 |

GPU: NVIDIA GeForce RTX 5080. PyTorch 2.11.0+cu128 / PyG 2.8.1. Preparation ranged from 9 to 28 seconds on this machine; cache state affects it.

Fold 0 after explicit build exclusions: 3,870 training, 711 validation, 1,037 reserved test binaries. The build excluded 31 training, 1 validation, and 2 test samples; reasons and IDs are recorded in each dataset.json. No test tensors were loaded by the smoke/benchmark commands.

Model 1 reads up to 128 prepared instructions per binary. Models 2/3 use the same prepared 500-block build; Model 3 reads up to 64 instructions per block. Mean and CLS use the same real instructions. These timings do not apply to whole-graph or uncapped-instruction training.

CPU test command: `python -m pytest tests/test_model_grid.py -q`. On restricted Windows environments, supply an existing writable parent for `--basetemp`.

Original runtime artifacts remain under `runs/grid_validation/` in the implementation checkout and are not committed. The committed source tests and commands in README.md reproduce the checks on another machine.
