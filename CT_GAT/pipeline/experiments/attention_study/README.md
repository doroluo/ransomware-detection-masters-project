# Attention heads and attention dropout

This follow-up uses the completed paper campaign's fold-specific recipes and training-only API vocabularies. It runs after the RTX machine's existing local queue finishes. Jobs assigned to the Radeon remain independent.

The current model files and the original paper results are preserved. Each new result records its complete specification, effective head counts, data fingerprint, code fingerprint, metrics, and timing.

## Design

| Setting | Follow-up values |
| --- | --- |
| Transformer heads | 2, 4, 8, 12 |
| GAT heads | 2, 4, 8, 12 |
| Attention dropout | 0.0, 0.1, 0.2 |
| Transformer width | 96 for all head counts |
| Feature and residual dropout | Each fold's frozen primary value |
| GAT head policy | Requested head count in every GAT layer |
| Input | Signal 500 with the frozen common API vocabulary |
| Screening | 12 epochs, seed 0, validation only |
| Confirmation | Seeds 0, 1, 2, up to 40 epochs, patience 8 |
| Model 3 pooling | Matched mean and CLS settings |

Model 1 has 12 screening settings per fold. Model 2 has 12. Model 3 crosses Transformer heads, GAT heads, attention dropout, and pooling for 96. The five folds therefore have 600 planned screening entries and 60 confirmation runs.

The existing grid tested shared dropout values 0.0, 0.1, 0.3, and 0.5, with heads 2, 4, and 8. Its GAT output layer always uses one head. Consequently the reported GAT head setting does not change a one-layer GAT's architecture. The exported records expose both the requested head setting and the effective heads by layer.

The new attention setting controls MultiheadAttention's attention probabilities and GATConv's normalized attention coefficients. Feature, feedforward, residual, and classifier dropout retain the frozen primary value. A fixed Transformer width of 96 allows every requested head count to divide the width exactly. The new GAT output layer also uses the requested head count, so this is an explicit architecture variant.

The study selects settings using validation ROC-AUC, then macro-F1. Model 3 settings require a complete mean/CLS pair and use the pair's average validation ranking. Outer test tensors are loaded for confirmations after that model's selection is frozen. The existing folds have historical results, so these remain follow-up experiments on the established evaluation protocol.

Performance above 90% on every metric is an empirical goal. False-positive rate is reported with lower values indicating better performance.

## Export existing results

Run from CT_GAT/pipeline. Set the paths to the local campaign and output folder.

```powershell
python -m experiments.attention_study.report --campaign C:/path/final_paper_20261103 --out C:/path/attention_records.json
```

This writes a snapshot JSON plus validation_parameters.csv and held_out_parameters.csv. Every completed row is checked against its saved training contract before export. Uncapped runs follow the campaign's existing reporting exclusions.

The Excel builder uses the Codex-provided @oai/artifact-tool package. Put workbook.mjs in a directory whose node_modules resolves that package, then run it with the provided Node runtime:

```powershell
node C:/path/workbook.mjs C:/path/attention_records.json C:/path/attention_results.xlsx
```

To refresh Excel automatically after each model search, pass --xlsx-node and --xlsx-builder to the supervisor. Each model completion regenerates the snapshot, CSVs, and workbook, including the original campaign and completed follow-up results. The workbook includes full parameter columns and explicit effective attention settings.

## Queue the follow-up

```powershell
python -u -m experiments.attention_study.watch --campaign C:/path/final_paper_20261103 --out C:/path/attention_followup_20261112
```

The supervisor reads the current campaign state every 20 seconds. It waits until all jobs assigned to this machine reach a terminal state. It checks again before each new model search. Original campaign status, results, data, and workers are read only.

The follow-up uses the original campaign's total wall-clock deadline and free-space reserve. It stores its own plan and status in its output directory. The current default grid can stop at the deadline with unfinished settings still visible. Failures retain their configuration and receive no substitute scores.

The same supervisor command resumes completed settings and saved checkpoints. A source or plan mismatch requires a new output directory. A STOP file in the follow-up root requests an epoch-boundary checkpoint for its active worker. Remove a reviewed STOP file before resuming.

## Run a single model or a CPU smoke check

```powershell
python -m experiments.attention_study.study --campaign C:/path/final_paper_20261103 --out C:/path/attention_followup_20261112/fold0/model3 --fold 0 --model 3
python -m experiments.attention_study.study --campaign C:/path/final_paper_20261103 --out C:/path/attention_cpu_smoke/fold0/model3 --fold 0 --model 3 --heads 12 --attention-dropouts 0.2 --width 96 --screen-epochs 2 --confirm-epochs 2 --patience 2 --seeds 0 --threads 1 --device cpu --smoke 8
```

Smoke checks use small balanced subsets and exclude outer-test evaluation. The supervisor supplies the original campaign deadline automatically. A manual production study can set --deadline-epoch explicitly.

## Dependencies and returned records

The Python code uses the existing pipeline requirements. Prepared signal-500 tensors, the original combined index, each fold's selection.json, and each fold's common_api.json are required. It reuses the paper trainer's microbatch accumulation, checkpointing, math attention backend, and activation offload.

Return complete model output directories, including plan.json, dataset.json, selection.json, and all screen/confirm result folders. The Excel records include the requested and effective attention parameters. The existing dataset loaders and checkpoint handling retain the paper campaign's serialization contract.
