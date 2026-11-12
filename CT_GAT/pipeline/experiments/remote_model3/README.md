# Remote Model 3 throughput diagnostic

The small original handoff pilot checked that the model path completed on a few selected examples. It did not establish full-epoch throughput on the Radeon. This diagnostic times real training operations before another full job starts.

Run from CT_GAT/pipeline in the GPU-enabled Python environment, with the existing handoff and signal-250 dataset extracted:

```powershell
python -u -m experiments.remote_model3.benchmark --handoff-root "$HOME/model3_handoff" --variant signal250 --timeout 120
```

The original recipe uses one graph per microbatch, activation checkpointing, math attention, and CPU activation offload. The optimizer still updates once per logical batch, typically eight graphs for fold 3.

The diagnostic compares three explicit execution profiles:

| Profile | Graphs per microbatch | CPU activation offload | Activation checkpointing |
| --- | --- | --- | --- |
| original | 1 | Enabled | Enabled |
| resident | 1 | Disabled | Disabled |
| batched | Frozen logical batch size | Disabled | Disabled |

All three retain the same architecture, learning rate, logical batch size, instruction limit, and math attention backend. They use the frozen fold 3 mean-pooling recipe and identical selected training examples.

Each profile runs in a fresh process and has a 120-second timeout including initialization. On timeout, the supervisor terminates that diagnostic worker and its descendants. It does not attach to other training jobs.

Outputs are written to a new throughput_check subfolder inside the handoff directory. Return summary.json, environment.json, the profile JSON reports, progress JSON files, and logs. These record the PyTorch/HIP versions, actual model and gradient devices, input hashes, graph sizes, and phase timings. A timeout report preserves the last operation reached.

The selected examples come from the middle and large end of each class's training set. Source tensors are opened with weights_only=True. Packed graph caches are created locally for the benchmark and read through the original disk loader to include the cost of graph deserialization.

The estimate of training epoch time uses these selected examples and explicit synchronization between phases. Validation and checkpoint time are excluded. The benchmark produces no held-out performance result and does not update the paper queue. A revised full-run profile needs a measured successful diagnostic and a completed full-size epoch before assigning a multi-day workload.

For a CPU-only developer check:

```powershell
python -m experiments.remote_model3.benchmark --handoff-root C:/path/handoff --build C:/path/prepared_signal250 --device cpu --profiles original resident batched --samples 8 --steps 1 --threads 1 --timeout 120
```
