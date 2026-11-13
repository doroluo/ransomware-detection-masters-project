# Remote Model 3 throughput diagnostic

## Revised Radeon launcher

The Windows RX 9070 XT diagnostic completed training backward with GPU-resident
activations while the original checkpoint/offload profile timed out in backward.
The revised launcher disables both features, retains microbatch size 1, the frozen
logical optimizer batch size, math attention, and the original architecture,
instruction limits, split, vocabulary, optimizer, pooling arms, seeds, and patience.
This is an explicitly recorded execution profile. Cross-device numerical results
and runtime need separate reporting; identical scores are not guaranteed.

Use the GPU environment from the successful diagnostic and update only the
separate diagnostics checkout. Copy `run_model3.ps1` from this directory into the
existing extracted handoff root (beside index.csv, fold3, and data). Then:

```powershell
.\run_model3.ps1 qualify signal250
.\run_model3.ps1 3 signal250
```

Run the first command first. It trains one full training epoch and validates,
using mean pooling and seed 0, without evaluating held-out test data. It must
complete successfully before the second command can start six trials: mean/CLS
pooling times seeds 0/1/2, at most 40 epochs and patience 8. Qualification and full
trials have separate contracts and checkpoints. The full trial starts fresh.

After reviewing the first qualification and full run, `run_model3.ps1 all` handles
the original eight comparisons and four signal500 ablations. Each recipe must
pass its own full-epoch qualification first. All datasets must already be extracted.
The script downloads nothing and does not change security settings or stop other
training processes. Use `-HandoffRoot`, `-RepoPath`, and `-Python` if needed.

Results are under `results_gpu_resident_v1/qualification/<variant>_<ablation>` and
`results_gpu_resident_v1/jobs/<variant>_<ablation>`. Cache files are under
`cache_gpu_resident_v1`. Existing results/cache folders remain intact.
Progress is printed and saved in each trial's progress.json every approximately
30 seconds between completed logical batches, with phase markers around the
first forward/backward and validation/checkpoint phases. This is progress
reporting, not a watchdog: a single stalled GPU call cannot emit a heartbeat.
Epoch history and last.pt permit resuming at the last completed epoch, not at an
unfinished batch. Don't run multiple copies against the same output folder.

The runner records execution settings in plan/config/metrics JSON, source hashes,
PyTorch version, GPU name, epoch timings, and peak allocated GPU memory. A previous
tiny pilot does not satisfy the full-epoch qualification gate. Qualification is
invalidated by changed source, selection, vocabulary fingerprint, or build path.

## Diagnostic commands

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

Outputs are written to a new throughput_check subfolder inside the handoff directory. The terminal prints a compact summary that is also saved as summary.txt. Return that text plus summary.json, environment.json, the profile JSON reports, progress JSON files, and logs. These record the PyTorch/HIP versions, actual model and gradient devices, input hashes, graph sizes, and phase timings. A timeout report preserves the last operation reached.

The selected examples come from the middle and large end of each class's training set. Source tensors are opened with weights_only=True. Packed graph caches are created locally for the benchmark and read through the original disk loader to include the cost of graph deserialization.

The estimate of training epoch time uses these selected examples and explicit synchronization between phases. Validation and checkpoint time are excluded. The benchmark produces no held-out performance result and does not update the paper queue. A revised full-run profile needs a measured successful diagnostic and a completed full-size epoch before assigning a multi-day workload.

For a CPU-only developer check:

```powershell
python -m experiments.remote_model3.benchmark --handoff-root C:/path/handoff --build C:/path/prepared_signal250 --device cpu --profiles original resident batched --samples 8 --steps 1 --threads 1 --timeout 120
```
