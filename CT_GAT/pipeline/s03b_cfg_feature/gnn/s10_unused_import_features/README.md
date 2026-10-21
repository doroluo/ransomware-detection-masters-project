# Step 10: import features

Option `RunConfig.imports`: `'header'` (`--imports header`, run-name suffix `-imph`, step 15); off by default (`''`). The earlier value
`'unreferenced'` (suffix `-imp`) was removed in step 25.

| file | content |
|---|---|
| `import_side.py` | converter: the team's `imports.json` (CT_GAT/IDA_FEATURE_READERS.md) -> `<cache>/side/imports/<sample_id>.npz` with the header, referenced and unreferenced import identities as the reader gives them; checks the export is complete, the counts match the cache, the three sets are consistent and every import the cache references is referenced for the reader; `load_imports` |
| `imports_feat.py` | `ImportVocab(train_rows, min_df)` (every header import identity present in at least `min_df` = 2 training binaries, with idf; `item(sample_id)` is one binary's bag and two log counts), `ImportBag` (`EmbeddingBag`, 32 dimensions) |

The bag (weights: idf, L2-normalised over the binary's kept names) and log1p(#unreferenced imports), log1p(#header imports) are
concatenated to the binary vector and `bin_feat` at the head (`s11_final_classifier/head.py`); `HierGNN.imports` exists only
when the option is on, so default checkpoints keep their keys. Validation and test binaries never enter the vocabulary or the idf.
The two log counts are log1p(#unreferenced imports) and log1p(#header imports); the unreferenced count stays so that the finished
`-imph` runs reproduce (`tests/test_imports.py` checks the vocabulary and the item against the definition).
"Unreferenced" is a static fact of the export, not evidence that the binary never uses the import at run time.

```bash
# from CT_GAT/pipeline
python -m s03b_cfg_feature.gnn.s10_unused_import_features.import_side --from-cfg      # from the CFG exports, into the default cache; no reader output needed
python -m s03b_cfg_feature.gnn.s10_unused_import_features.import_side --cache <cache dir> [more caches] --reader-root <reader output>
python -m s04_evaluation.baselines --protocol kfold --fold 0 1 2 3 4 --arch x86 --seed 0 --imports   # imports_header_lr
python -m experiments.train --protocol kfold --fold 0 --arch x86 --seed 0 --mask-blocks 0.15 --imports header   # R7 + imports
```
