# Pretrained M1, M2 and M3 demos

Run from `CT_GAT/pipeline` in the project's Python environment. The commands load frozen weights, prepare model inputs, and produce predictions. They do not retrain a model or execute an input executable.

## Which models these commands run

| Model | Selected completed standard family | Overall accuracy | Demo checkpoint |
|---|---|---:|---|
| M1 | Instruction Transformer, signal1000 | 83.23% | Family's fold 0, seed 0 |
| M2 | Handcrafted-feature GAT, signal750 | **85.80%** | Family's fold 0, seed 0 |
| M3 | Joint block Transformer/GAT, mean pooling, attention study, signal500 | 84.59% | Family's fold 0, seed 0 |

The family selection criterion is the highest observed complete five-fold **accuracy** for each model, excluding ablations. This is retrospective selection among experiments on established splits, not a new untouched test or a claim of statistical superiority. The demo member is a fixed reference checkpoint, not the best test-scoring seed or a model refitted on the entire corpus. The percentages summarize all five separately selected recipes and three seeds. They do not describe the accuracy of the single demo checkpoint or an ensemble.

M3 mean led the attention study; M2 signal750 had higher overall accuracy across the completed standard experiments. The earlier 94.73% M3 CLS result was for fold 0 only. M2's API-kind-count ablation scored 85.94%; it is excluded here because these demos preserve the standard model variants.

## Quick start

Use an isolated clone to leave other branches and experiments untouched:

```powershell
git clone --branch codex/final-results-export-20261124 https://github.com/doroluo/ransomware-detection-masters-project.git rdmp-demo
cd rdmp-demo\CT_GAT\pipeline
```

Activate the existing project PyTorch/PyTorch Geometric environment. For a fresh CPU environment, install the repository's `requirements.txt`. Download the three checkpoints once:

```powershell
python -m experiments.demo.download
```

This verifies the published archive SHA-256 and all member hashes. An already valid download is reused. To use a ZIP downloaded through GitHub's release page:

```powershell
python -m experiments.demo.download --archive "C:\Downloads\Best_Models_Demo_20261125.zip"
```

Checkpoint asset: [demo-models-20261125 release](https://github.com/doroluo/ransomware-detection-masters-project/releases/tag/demo-models-20261125). The archive is about 27 MB. It contains model weights, frozen API-name lists, corpus IDs for overlap checks, and training configurations. It contains no executable samples or training datasets.

## One command for a new IDA export pair

Provide the compatible structured CFG export and matching assembly listing from the same executable:

```powershell
python -m experiments.demo.predict --model all --cfg "C:\exports\sample.jsonl.gz" --asm "C:\exports\sample.asm.gz" --out ".\demo_results\sample.json"
```

For one model, change only `--model`:

```powershell
python -m experiments.demo.predict --model 1 --cfg "C:\exports\sample.jsonl.gz" --asm "C:\exports\sample.asm.gz" --out ".\demo_results\sample_m1.json"
python -m experiments.demo.predict --model 2 --cfg "C:\exports\sample.jsonl.gz" --asm "C:\exports\sample.asm.gz" --out ".\demo_results\sample_m2.json"
python -m experiments.demo.predict --model 3 --cfg "C:\exports\sample.jsonl.gz" --asm "C:\exports\sample.asm.gz" --out ".\demo_results\sample_m3.json"
```

CPU is the default. `--device cuda` selects an available PyTorch CUDA/HIP device. The runner automatically uses the appropriate block cap, feature construction, frozen training vocabulary, network settings, weights and saved classification threshold for each model. It reads the executable SHA-256 from the CFG header, so export filenames can be arbitrary.

The CFG must use the schema accepted by `s02_preprocess.preprocess_full` and include a complete footer. This command does not accept `.exe` directly and does not install or invoke IDA. Generate the compatible exports in the existing isolated IDA analysis environment first. A new export schema requires its own adapter; renaming a file is insufficient.

## Existing preprocessed samples

Each prepared directory must contain `<SHA256>.pt`, `settings.json` and the matching ordered `api_vocab.json`. It must use signal sampling, sampler seed 0, and the model's exact cap: M1=1000, M2=750, M3=500. Do not pass already-scaled features or a checkpoint as a sample.

```powershell
python -m experiments.demo.predict --model 3 --build "C:\new_sample\signal500" --sample-id "<64-character-lowercase-SHA256>" --out ".\demo_results\sample_m3.json"
```

The three caps require separate prepared samples. `--model all` therefore uses the IDA-export input path, which prepares each model separately. `--bundle DIR` overrides the default `demo_models` checkpoint folder.

## Exact demo recipes

| Setting | M1 | M2 | M3 mean |
|---|---|---|---|
| Sampler / block cap / seed | signal / 1000 / 0 | signal / 750 / 0 | signal / 500 / 0 |
| Transformer width / layers / heads | 64 / 1 / 4 | Unused | 96 / 1 / 4 |
| Instruction limit | First 1024 in the selected input | No Transformer truncation | First 64 per block |
| GAT hidden / layers | Unused | 128 / 2 | 128 / 2 |
| Actual GAT heads per layer | Unused | `[4, 1]` | `[12, 12]` |
| Pooling | Instruction mean | Graph mean + max | Instruction mean, then graph mean + max |
| Feature dropout during training | 0.0 | 0.0 | 0.1 |
| Independent attention dropout | Original shared setting | Original shared setting | 0.0 |
| Training learning rate | 0.001 | 0.003 | 0.001 |
| Training weight decay | 0 | 0 | 0 |
| Logical training batch | 32 | 32 | 8 |
| Training microbatch | 32 | 1 | 1 |
| Checkpoint epoch | 3 | 22 | 13 |
| Saved validation threshold | 0.18 | 0.20 | 0.10 |

Dropout is disabled by `eval()` during prediction. Learning rate, weight decay and batch size document training; they are not new inference settings. The classifier uses the exact threshold saved with its checkpoint, rather than assuming 0.5. Full saved specifications and source fingerprints are in `recipes.json` and the downloaded training configurations. Inherited fields for a component absent from a model are unused.

Inspect recipes without loading weights:

```powershell
python -m experiments.demo.predict --list
```

## Output and validation scope

The new JSON file records each model's ransomware score, predicted class, saved threshold, checkpoint identity, preprocessing coverage and evaluation scope. Existing output files are refused. A single file cannot establish accuracy, recall, FPR or AUC, and the softmax score is not a calibrated real-world infection probability.

Known corpus IDs are rejected by default. `--allow-known-sample` permits a replay check and labels its output as existing-corpus replay. Absence of a hash from the archived index does not prove an unseen family or an unrelated variant.

The runner loads project tensor dictionaries/checkpoints with `weights_only=True`, checks the checkpoint identity and model/preprocessing source hashes, and never falls back to unrestricted pickle loading.

CPU validation replays cover five saved validation predictions per model. The observed largest absolute score differences were about 0.00371 for M1, 0.000000067 for M2, and 0.000000107 for M3; all replay classifications agreed at the saved thresholds. CPU `auto` emulates Transformer bfloat16 training numerics with the math-attention path. `--cpu-precision fp32` is available for diagnostics and can produce different scores. CUDA/HIP uses the original training autocast policy.

Compatible IDA export preparation is tested with synthetic export/listing pairs. No previously unseen external executable was supplied for this handoff, so actual external-file results remain to be measured. GPU execution was not requalified during packaging. No training run was started or interrupted.

## Graphs and analysis

The [five-figure package](../../../../reports/hyperparameters_20261124/READ_ME_FIRST.txt) covers overall test metrics, optimizer settings, architecture/input limits/batch size, attention interactions, and block-budget sampling. PNGs are suitable for slides; SVGs retain vector graphics and editable text. CSV files preserve plotted numbers. Validation-stage curves and held-out metrics are explicitly distinguished.

The [verification report](../../../../reports/demo_models_20261125/replay_verification.json) records the local CPU replay checks. The [family recipe table](../../../../reports/demo_models_20261125/best_family_recipes.csv) records every fold-specific recipe for the selected model families; the small downloadable bundle supplies the fixed demo members only.
