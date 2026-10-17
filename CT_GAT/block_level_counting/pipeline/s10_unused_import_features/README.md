# Step 10: import features

Option `RunConfig.imports`: `'unreferenced'` (`--imports unreferenced`, run-name suffix `-imp`) or `'header'` (`--imports header`, suffix
`-imph`, step 15); off by default (`''`).

| file | content |
|---|---|
| `import_side.py` | converter: the team's `imports.json` (CT_GAT/IDA_FEATURE_READERS.md) -> `<cache>/side/imports/<sample_id>.npz` with the header, referenced and unreferenced import identities as the reader gives them; checks the export is complete, the counts match the cache, the three sets are consistent and every import the cache references is referenced for the reader; `load_imports` |
| `imports_feat.py` | `ImportVocab(train_rows, min_df, which)` (`which='unreferenced'`: identities unreferenced in at least `min_df` = 2 training binaries; `which='header'`: every header import identity present in at least `min_df` training binaries; idf in both; `item(sample_id)` is one binary's bag and two log counts), `ImportBag` (`EmbeddingBag`, 32 dimensions) |

The bag (weights: idf, L2-normalised over the binary's kept names) and log1p(#unreferenced imports), log1p(#header imports) are
concatenated to the binary vector and `bin_feat` at the head (`pipeline/s11_final_classifier/head.py`); `HierGNN.imports` exists only
when the option is on, so default checkpoints keep their keys. Validation and test binaries never enter the vocabulary or the idf.
`--imports header` changes only which names go into the vocabulary and the bag: the two log counts, the head input size and the other
checkpoint keys are the same, and an unreferenced run is bit for bit what it was (`tests/test_reader_wiring.py`).
"Unreferenced" is a static fact of the export, not evidence that the binary never uses the import at run time.

```bash
python -m pipeline.s10_unused_import_features.import_side --cache C:/Users/chaoa/Downloads/graph_transformer_cache_v2 [more caches]
python baselines.py --protocol kfold --fold 0 1 2 3 4 --arch x86 --seed 0 --imports   # imports_unref_lr, imports_header_lr, tfidf_sig_api_imp_lr
bash queue_gnn.sh --stage imports                                                        # baselines, gate, then R7 + --imports unreferenced
bash queue_gnn.sh --stage imports_header                                                 # R7 + --imports header, five folds, no gate
```
