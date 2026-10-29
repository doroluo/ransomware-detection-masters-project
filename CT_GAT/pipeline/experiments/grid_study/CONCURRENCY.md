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
