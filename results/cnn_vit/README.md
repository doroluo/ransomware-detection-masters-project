# results/cnn_vit

Written by `cnn_vit_pipeline/train_eval.py`:

```
results/cnn_vit/<dataset>/metrics.json    # expA schema + per-arch + per-family + epoch history
```

`expB_splits_snapshot.csv` is not a result. It is a frozen copy of
`results/expB/splits.csv`, taken once at the start of this work because the
tokenization worker was editing that file concurrently. The balanced dataset's
goodware membership is read from this snapshot, so the split stays stable even
if the live file moves. `cnn_vit_pipeline/cohort.py` prefers the snapshot and
falls back to the live file.

No metrics yet: the ransomware and test-goodware images are built from `.asm`
trees the VM has not delivered. See `vm_package/README.md` step 3 and
`cnn_vit_pipeline/README.md`.
