# results/ember

Written by `ember_pipeline/train_eval.py`. One directory per run:

```
results/ember/<dataset>/<variant>/metrics.json      # expA schema + per-arch + per-family
results/ember/<dataset>/<variant>/split_used.csv    # exactly which sha256 went where
results/ember/<dataset>/<variant>/duplicate_sha_rows.csv   # when any were collapsed
```

`<dataset>` is `mendeley` or `balanced`; `<variant>` is `cohort` (the shared
family-disjoint split) or `branch` (the EMBER branch's own random 80/20, for
comparison only).

Empty today: the runs need the ransomware feature vectors that are extracted
inside the VM (`mendeley_mal_train.npz`, `mendeley_mal_test.npz`, and for the
mendeley dataset `mendeley_good_test.npz`). See `vm_package/README.md` step 4
and `ember_pipeline/README.md`.
