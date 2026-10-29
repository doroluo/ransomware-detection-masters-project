# Grid searches for Models 1, 2 and 3

Run from `CT_GAT/pipeline`. The default grid is **3 learning rates x 2 dropout values**.
Model 3 repeats the grid for **mean and CLS pooling**, giving 12 configurations.
These runners are additive; historical trainers, checkpoints and reports are unchanged.

## Environment

Use Python 3.12. The tested Windows environment is PyTorch 2.11.0+cu128 and PyTorch Geometric 2.8.1.
Install the CUDA wheel first so installing the repository requirements does not select the CPU wheel:

```powershell
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install matplotlib
```

The last line is optional; CSV/Markdown output does not require matplotlib. Verify CUDA with the command below.

## Quick start for the team

1. Pull the new `main` and use a Python environment with this pipeline's requirements,
   a CUDA-enabled PyTorch installation, and PyTorch Geometric. `matplotlib` is optional for PNG charts.
2. Each machine needs the **same prepared pipeline-B build** (the folder of SHA-named `.pt` files,
   `settings.json`/`build_info`, and vocabulary metadata) and the **same CFG-cache `index.csv`**.
   The grid uses the stored instruction-to-block assignments and CFG edges. It does not run malware,
   invoke IDA, or rebuild assembly/CFG exports.
3. Substitute your own local paths below. Use a different output root for a different dataset or experiment.

```powershell
cd CT_GAT/pipeline
python -c "import torch, torch_geometric; print(torch.__version__, torch.cuda.is_available())"

# Inspect the plan without loading tensor samples or training.
python grid_search.py --models 1 2 3 --dataset combined --build "PATH/TO/b_sig500_combined_v3" --index "PATH/TO/graph_transformer_cache_combined/index.csv" --out runs/draft_combined --plan-only

# RTX 5080: Models 1 and 3 sequentially.
python grid_search.py --models 1 3 --dataset combined --build "PATH/TO/b_sig500_combined_v3" --index "PATH/TO/graph_transformer_cache_combined/index.csv" --out runs/draft_combined

# RTX 5070 Ti: Model 2. Use identical data/index files, wherever they are on this machine.
python grid_search.py --models 2 --dataset combined --build "PATH/TO/b_sig500_combined_v3" --index "PATH/TO/graph_transformer_cache_combined/index.csv" --out runs/draft_combined
```

To assign Model 3 to the 5070 Ti instead, use `--models 1 2` on the 5080 and `--models 3` on the 5070 Ti.
The commands also work in Linux shells with appropriate paths. No machine-specific paths are embedded in code.

For Mendeley x86, use `--dataset mendeley-x86`, its `b_sig500_v3` build and `graph_transformer_cache_v2/index.csv`.
For Mendeley x86+x64, use `--dataset mendeley-all` and `b_sig500_mendeley_all` with the Mendeley index.
For combined architecture-matched analysis use `--dataset combined-matched` with the combined build/index.
An index without the requested fold/architecture metadata is rejected.

Individual entry points (same options):

```powershell
python -m experiments.model_grid.model1 --help
python -m experiments.model_grid.model2 --help
python -m experiments.model_grid.model3 --help
```

## What each model does

| Model | Representation | Summary |
|---|---|---|
| 1 | Original five-channel instruction tuples, learned positions, 2-layer/4-head Transformer, width 64 | Masked mean over the first 128 instructions of the prepared binary input; classifier |
| 2 | Original 77 handcrafted block features and typed CFG; existing FullBlockGAT | Existing mean+max graph pooling; no instruction Transformer |
| 3 | Five-channel tuples and learned positions within each block; same 2-layer/4-head Transformer, width 64; existing FullBlockGAT | Masked mean **or** learned CLS per block, then unchanged mean+max graph pooling |

The five channels are opcode, two operand categories, API identity and behavior tag. Learned positions are initialized
with normal standard deviation 0.02 in this experiment. Model 3's instruction embeddings, block Transformer and GAT
are trained together. Its mean and CLS arms share initial encoder/GAT weights for a seed (apart from the extra CLS token).
The extra CLS slot does **not** displace a real instruction: both arms retain up to `--block-length 64` real instructions.
The classifier in the reused Model 1 module is removed from Model 3 because that classifier is not used there.

Models 2/3 retain the team's GAT defaults: hidden size 128, 4 heads, 2 layers, edge width 8.
The dropout grid applies to the Transformer/head in Model 1, the GAT in Model 2, and **both** Transformer and GAT
in Model 3. This is a shared regularization sweep, not separate estimates of encoder versus GAT dropout effects.
Width is not instruction count. `--sequence-length` controls Model 1's real instruction limit;
`--block-length` controls Model 3's. Both are fixed within a grid.

Source instructions come from the prepared build. A 500-block build already omits some raw CFG blocks; increasing
sequence length cannot recover absent blocks. `coverage.json` records instruction/block counts in the raw CFG,
prepared input and model input, API-instruction counts and unmatched instructions. Handcrafted Model 2 features summarize
instructions but do not encode their sequence. Neither mean nor CLS pooling is lossless.

## Protocol and defaults

- **One outer fold**, fold 0, for the draft; grouped validation seed 0 stays fixed across all trials and training seeds.
  Families/benign identity groups and sample IDs must be disjoint across train/validation/test.
- **Screen:** learning rates `0.0001 0.0003 0.001` x dropout `0.1 0.3`; seed 0; exactly 12 epochs per trial.
  No test tensors are loaded and no test predictions are produced during screening.
- **Selection:** validation AUC, then validation macro-F1 at threshold 0.5. For Model 3, choose one shared LR/dropout
  by averaging these validation scores across mean/CLS, then confirm both at that same setting. This is a paired
  pooling experiment, not independent optimization of each pooling method.
- **Confirm:** train from scratch with seeds `0 1 2`, up to 40 epochs, patience 8 on the same validation selection rule.
  Restore the best validation checkpoint; fit its decision threshold on validation only; then score the outer test fold.
  Report mean +/- **sample SD across seeds** (not a confidence interval or fold uncertainty).
- AdamW, class weights fitted on training labels, gradient clipping 1.0. Weight decay is 0.01 for Model 1 and 0.0001
  for Models 2/3. Batch sizes are 32, 16 and 8 respectively. Model 3 encodes blocks in chunks of 1024 to reduce temporary
  attention memory; this does not truncate the selected instructions. CUDA Transformer operations use bf16; GAT uses fp32.
- API names absent from the training input are mapped to UNK. Existing behavior categories are inherited from the
  tokenizer, so this is **not** an unseen-behavior or API-free experiment.
- Exclusions are accepted only when the build explicitly records them, and are listed by split. Other missing files fail.
  Compare models using identical dataset fingerprints and review exclusions/coverage before interpreting scores.

Useful overrides:

```powershell
# Add these flags to a normal command:
--stage screen                         # validation-only grid, stop before confirmation
--stage confirm                        # requires all matching screen results; resumes confirmation
--learning-rates 0.0001 0.0003 0.001 --dropouts 0.1 0.3
--epochs 12 --confirm-epochs 40 --patience 8 --seeds 0 1 2
--folds 0 1 2 3 4                     # final paper: repeat selection separately INSIDE each outer fold
--batch-size 4 --block-chunk 256        # lower Model 3 memory; new output directory required
```

Never select settings on fold 0 and report the other four folds as untouched tests: their samples were available to
the fold-0 search. `--folds 0 1 2 3 4` instead runs a separate nested search for every outer fold.
These are the project's existing outer test folds and have historical results; this does not create a fresh external test set.

## Validate and estimate runtime first

```powershell
# Supply the same build/index/dataset arguments as above, but use separate output roots:
python grid_search.py --models 1 2 3 --dataset combined --build "PATH/TO/b_sig500_combined_v3" --index "PATH/TO/graph_transformer_cache_combined/index.csv" --out runs/grid_smoke --smoke 8
python grid_search.py --models 3 --dataset combined --build "PATH/TO/b_sig500_combined_v3" --index "PATH/TO/graph_transformer_cache_combined/index.csv" --out runs/grid_benchmark --benchmark
```

Smoke runs use small training/validation subsets and two epochs; they are not reportable model results.
Benchmark runs use the complete train/validation split, one setting (both pooling arms for Model 3), and two epochs.
Use measured preparation time plus epochs x measured seconds/epoch for an estimate; checkpointing/reporting add overhead.
Do not infer an RTX 5070 Ti time from the GPU name or from the old trainers' timings.

## Outputs, resuming and sharing

`--out ROOT` creates `ROOT/modelN/foldK/` containing:

- `REPORT.md`, `results.csv`, and optional `grid.png`: validation screen and separate test confirmations.
- `plan.json`, `dataset.json`, `payload_hashes.json`, `coverage.json`: effective settings, content hashes, exact splits and coverage.
- `selection.json`, `status.json`: validation-selected setting and queue status.
- `screen/<case>/` and `confirm/<case>_seedN/`: best/last checkpoints, epoch history, metrics and predictions.

Rerun the **same command** to skip completed matching trials and resume an interrupted trial from its last full epoch,
including optimizer and RNG state. A source/data/settings mismatch fails instead of mixing results; use a new output root.
The source hashes normalize line endings for cross-platform Git checkouts. Concurrent writers to the same model/fold are refused.
Create `ROOT/modelN/foldK/STOP` to stop that queue at its next epoch boundary; remove it before resuming.
Changing model code, sequence length, batch size, grid, split or training budget requires a new output root.
Exact restart equivalence is tested on CPU; CUDA scatter operations can be nondeterministic.

Share the completed `modelN/foldK` result folder with the team (checkpoints are optional for report aggregation,
but retain them locally). Compare dataset/source fingerprints before combining model results. Reusing an old result as a
new grid point is not supported: old trainers sometimes recorded generic config fields that were not their actual recipe.
The expensive prepared builds are reused; historical scores remain separately labeled baselines.
