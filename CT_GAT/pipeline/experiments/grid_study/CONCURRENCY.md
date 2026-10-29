# Local concurrency check, 2026-10-28

RTX 5080, combined dataset, fold 0, the same 500-block build. Model 2 used batch 16, GAT hidden width 128/depth 2/heads 4, LR 0.0001, dropout 0.1 and weight decay 0.0001. Two 12-epoch screens used the same seed and training protocol.

| Measurement | Training alone | Overlapping Model 3 |
|---|---:|---:|
| Model 2 mean epoch time | 1.671 s | 4.004 s |
| Model 2 total trial invocation | 20.34 s | 48.58 s |
| Model 3 observed CLS epoch time | about 21.7 s, subsequent steady epochs | 30.16 s, first CLS epoch during overlap |

Model 2 was 2.40 times slower under overlap. Model 3 also slowed, but its comparison crosses a trial boundary and includes the first epoch: this is not a controlled estimate of total throughput. A modest combined gain remains possible; individual slowdowns alone do not establish that concurrent execution is slower overall.

Sampled GPU utilization ranged from 57% to 80% and memory peaked at 3,073 MiB during the short observation. Memory fits for these small configurations. Previously tested larger Model 3 batches reached 11,585 MiB, and uncapped graph memory is more variable. Do not infer general concurrency capacity from the default setting.

Decision for the complete 30-hour search: run Models 1, 2 and 3 sequentially on one GPU, while assembly/CFG preprocessing continues on CPU. Reuse the standalone Model 2 screen deterministically; retain the concurrent duplicate as a performance diagnostic, not an additional independent seed or a second candidate selected by its accuracy. Carry forward all prior training time, conservatively charge both benchmark invocations, and keep the total budget at 30 hours rather than resetting it when adding Model 2.

## Model 3 runtime follow-up, 2026-10-29

After Models 1 and 2 completed their main grids, Model 3 already had exclusive GPU access. The queue was paused at a saved epoch boundary for training-only runtime diagnostics. CPU assembly/CFG preprocessing remained running. The diagnostic runtime was charged to the existing architecture budget; no scores were imported as scientific results.

Hardware: RTX 5080, Ryzen 7 9800X3D (8 cores / 16 threads). Input: the first 112 batches of eight samples from the fixed seed-0 first-epoch training order of the combined 500-block build. Sixteen batches warm up each configuration; 96 are timed. No held-out samples were loaded. These are minibatch throughput measurements, not complete epoch/validation/checkpoint benchmarks.

| Configuration | 1 CPU thread | 2 threads | 4 threads | 8 threads |
|---|---:|---:|---:|---:|
| default | 34.56 ms | 34.12 ms | 34.60 ms | 34.85 ms |
| large | 72.62 ms | 72.71 ms | 72.96 ms | 73.26 ms |

Default: Transformer width 64 / depth 2, GAT hidden 128 / depth 2, mean pooling, 64 instructions per block. Large: Transformer width 128 / depth 3, GAT hidden 256 / depth 3. One, two and four CPU threads were each measured twice; eight threads once. Thread-count differences were approximately 1%, insufficient to justify changing the active run.

Concurrency used the default architecture, one CPU thread per worker, independent seeded models and CUDA contexts. Each worker ran 288 measured optimizer steps after warmup. The order was one worker, two workers, two workers, one worker.

| Workers | Aggregate steps/s, run 1 | Aggregate steps/s, run 2 |
|---|---:|---:|
| 1 | 29.35 | 29.05 |
| 2 | 30.53 | 29.94 |

The two-worker aggregate throughput was 1.035 times the single-worker throughput (3.5% improvement), with approximately doubled latency per worker. This does not establish a useful full-trial speedup, especially for larger architectures or uncapped graphs. Keep the current sequential scheduler and CPU-thread setting. Splitting disjoint trials across a second physical GPU is the preferred substantial acceleration; it requires explicit assignment and compatible result contracts to avoid duplicate work.

Raw local evidence: `runs/model3_speed_check/thread_results.json`, `concurrency_summary.json`, and the diagnostic scripts/logs. Existing model, optimizer, data, epoch count, source contracts and checkpoint RNG state were not changed.
