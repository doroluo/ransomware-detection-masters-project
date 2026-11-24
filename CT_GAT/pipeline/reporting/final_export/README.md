# Final paper Excel export

The upload-ready workbook is [`reports/final_paper/Final_Paper_Results_With_Hyperparameters_2026-11-24.xlsx`](../../../../reports/final_paper/Final_Paper_Results_With_Hyperparameters_2026-11-24.xlsx).
Download the file with GitHub's **Download raw file** button, then upload the `.xlsx` to Google Drive.

This export includes 744 paper-included held-out trials: 678 original RTX 5080 trials, 60 completed attention-study confirmations, and six returned RX 9070 XT signal250 Model 3 fold-3 trials. Three historical uncapped trials remain explicitly excluded. Validation tables contain 2,545 original stage entries and 594 successful attention screens. Six attention screens failed with CUDA OOM; all selected final confirmations finished.

## Reading settings and scores

- **Summary** shows completed attention-study pooled test scores; the original primary results remain below the chart.
- **Held-out runs** places the recipe ID and readable exact settings next to the score. Original recipe IDs start `M`; attention-study recipe IDs start `A-M`.
- **Frozen recipes** contains 20 original and 20 attention selections. Settings are selected independently within each outer fold. A five-fold pooled score combines five recipes, with three training seeds per recipe.
- **Validation grid** contains tuning scores, separately labeled from held-out performance. The original grid was staged, not a full Cartesian product.
- **Hyperparameter guide** explains learning rate, weight decay, batch size, width, layers, heads, dropout, instruction limits, block counts, pooling, epochs, patience, folds and seeds.

Attention dropout (tested at 0, 0.1, 0.2) drops attention probabilities during training; feature dropout drops learned feature activations. Heads (2, 4, 8, 12) learn parallel attention patterns. The attention study fixes Transformer width at 96 and applies the selected GAT head count in every GAT layer. Those controls differ from the original architecture settings, so a cross-study change in score cannot be attributed solely to heads or dropout. The complete study does **not** meet 90% across all seven pooled metrics for any model/pooling group.

Radeon hardware/execution labels remain attached to its results. Its RX 9070 XT / Ryzen 7 7800X3D timings should not be converted to RTX 5080 times or interpreted as a GPU speed ratio.

## Rebuild the archived workbook

`export.mjs` uses `@oai/artifact-tool` from the Codex bundled spreadsheet runtime. It is a reporting operation and does not start training. Bind the bundled `node_modules` directory beside this script as described by the spreadsheet skill, then run with the bundled Node executable:

```powershell
node CT_GAT/pipeline/reporting/final_export/export.mjs --output "$env:TEMP/Final_Paper_Results_With_Hyperparameters_2026-11-24.xlsx"
python CT_GAT/pipeline/reporting/final_export/verify.py "$env:TEMP/Final_Paper_Results_With_Hyperparameters_2026-11-24.xlsx" --layout "$env:TEMP/layout.json"
```

The default inputs in `assets/` include the preserved original/Radeon workbook, its column mapping, and a gzip-compressed **JSON** snapshot of the verified attention records. No executable samples, datasets, checkpoints or pickle payloads are included. `--template`, `--layout`, `--data`, `--output`, and `--preview-dir` override the defaults. Choose an output folder where the runtime can write. Rendered previews and the verification layout are saved beside the output.

To refresh the prepared attention snapshot from the original experiment files, use Python with NumPy and scikit-learn:

```powershell
python CT_GAT/pipeline/reporting/final_export/prepare.py --study-root "D:/runs/attention_followup_20261112" --campaign-root "D:/runs/final_paper_20261103" --output "$env:TEMP/attention_records.json"
node CT_GAT/pipeline/reporting/final_export/export.mjs --data "$env:TEMP/attention_records.json" --output "$env:TEMP/refreshed.xlsx"
```

If the campaign's saved index or signal500 build paths differ on this machine, supply `--index` and `--signal500-build`. The preparer verifies saved configuration fingerprints, confirms complete study selections, checks every test prediction ID/label against the original cohort, and reconstructs 12 five-fold pooled seed scores. It does not deserialize model checkpoints. This export is intentionally fixed to the completed 2026 attention study and its original baseline; it rejects incomplete results instead of fabricating rows.

`verify.py` checks the actual saved Excel archive: historical values/formulas, every new raw record, pooled metrics, recipe counts, table styles, frozen panes and formula errors. `reports/final_paper/manifest.json` records file hashes and the final validation result.
