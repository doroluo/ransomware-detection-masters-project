# Adding an experiment

Every command runs from `CT_GAT/pipeline`. An experiment is some runs and a table:

1. a run is one `RunConfig` (`common/config.py`): its fields are the command line of `python -m experiments.train`, its `name`
   is its directory under `<runs>` (`GT_RUNS`), and the options that differ from the defaults are in the name;
2. a *plan* (`experiments/plans/<name>.txt`) lists the runs, and `python -m experiments.queue` runs them, skipping those that have a
   `metrics.json`;
3. a *gates stage* (`experiments/stages/<name>.py`) names the arms (groups of runs, by their name) and the pairs to compare, and
   `python -m s04_evaluation.gates --stage <name>` writes the table.

The examples below are part of the test suite: `tests/test_docs.py` extracts every block tagged `example=<name>` and runs it on the
synthetic cache of `tests/tiny_cache.py`, so they work as written.

## A model

A model is one `ModelSpec` (`common/registry.py` documents the members) and one line in `MODELS`. The training loop owns the class
weights, the optimiser, patience, the two checkpoints, the validation thresholds and the run directory; the spec owns what is fitted
on the training manifest (`build`), how a list of binaries becomes batches (`loader`) and the logits of a batch (`forward`).
`GnnSpec` (`s03b_cfg_feature/gnn/spec.py`) is the full example; the smallest useful model reuses its data path and changes the network:

```python example=model
from dataclasses import replace

from s03b_cfg_feature.gnn.spec import GnnSpec


class ShallowSpec(GnnSpec):
    """The GNN with one message-passing layer per level: the same data path, the same loop, a smaller network."""
    name = 'shallow'

    def build(self, cfg, info):
        return super().build(replace(cfg, gnn_fn_layers=1, gnn_bin_layers=1), info)

    def variant_tokens(self, cfg):
        return ('shallow',) + tuple(cfg.variant.split('-')[1:])
```

The registry lines (the first three are the real ones; the module path is where the class can be imported from):

```python example=registry
MODELS = {
    'gnn': 's03b_cfg_feature.gnn.spec:GnnSpec',
    'tokenseq': 's03a_transformer.token_sequence.spec:TokenSeqSpec',
    'fivechannel': 's03a_transformer.five_channel.spec:FiveChannelSpec',
    'shallow': 'my_stage.spec:ShallowSpec',
}
```

Then `RunConfig` has to know the name, because it fixes the run name: add `'shallow'` to `MODELS` and to the `model: Literal[...]` field of
`common/config.py`, and make `RunConfig.variant` return a variant that starts with the model word (see its `tokenseq` branch). A
model that needs its own loop (Model 1 does) gives its spec a `run(cfg, out_name, **options)` and the registry hands the run over.
`requires` lists the paths under the cache that every run reads; a missing one is named before anything is written.
The model-specific flags that `RunConfig` has no field for are the `MODEL_OPTIONS` of `experiments/train.py`.

Training needs CUDA. To check a spec without a GPU, build it and run one batch, as `tests/test_docs.py` does with the example.

## An encoder (Model 3)

A frozen per-block encoder is any `torch.nn.Module` that maps the token ids of one block to a vector and depends on nothing else.
The contract is `s03c_cfg_transformer/encoder_contract.BlockEncoder` (the rules are in `s03c_cfg_transformer/TRANSFORMER_CONTRACT.md`):
attributes `d_out` and `max_len`, and `forward(tokens [B, L] long, mask [B, L] bool) -> [B, d_out]`; the class must be importable by name.
You train it on `FoldView(store, k, 'train')` of a block export, select on `'val'`, never read `'test'`, and save a checkpoint per fold:

```python example=encoder
import torch
from torch import nn

from s03c_cfg_transformer.encoder_contract import save_checkpoint


class BagEncoder(nn.Module):
    """Mean of token embeddings: the smallest encoder that satisfies the contract (the reference encoder has random weights)."""

    def __init__(self, n_ids, d_out=16, max_len=192):
        super().__init__()
        self.d_out, self.max_len = d_out, max_len
        self.embed = nn.Embedding(n_ids, d_out, padding_idx=0)

    def forward(self, tokens, mask):
        weight = mask.to(self.embed.weight.dtype)[..., None]
        return (self.embed(tokens) * weight).sum(1) / weight.sum(1).clamp(min=1)


def write_checkpoint(store, export_root, out_path, fold):
    manifest = store.fold(fold)['manifest']
    ids = lambda split: [store.binaries[i]['sample_id'] for i in manifest[split]]
    init = {'n_ids': store.fold(fold)['vocab']['n_ids']}
    save_checkpoint(out_path, BagEncoder(**init), init, export_root, fold, ids('train'), ids('val'))
```

`out_path` is `<runs>/encoders/<name>/fold<k>/best.pt`. Check it, compute the vectors once per distinct block, and train the GNN on them:

```bash
python -m s03c_cfg_transformer.block_export --protocol kfold --arch x86 --out RUNS/export/ours_cap0
python -m s03c_cfg_transformer.encoder_contract --ckpt <runs>/encoders/bag/fold0/best.pt --export <runs>/export/ours_cap0 --fold 0
python -m s03c_cfg_transformer.precompute_blockvec --encoder-name bag --export ours_cap0 --folds 0 1 2 3 4
python -m experiments.train --protocol kfold --fold 0 --model gnn --mask-blocks 0.15 --block-input vec --block-vec ours_cap0_bag
```

The vectors land in `<cache>/side/blockvec/ours_cap0_bag/fold<k>/`; `--control no_edges` with the last command is path 1 (no edges).
`experiments/plans/paths_codebert.txt` is this for CodeBERT, three seeds.

## A dataset

A dataset is a CFG cache plus the fold assignment of the protocols that read it.

1. **Cache.** `python -m s02_preprocess.cache.build_cache --corpora <names>` streams the IDA exports (`s01_ida_export`) of the
   sources into `<cache>/samples/*.npz`, `index.csv` and `vocab.json`; `python -m s02_preprocess.cache.check_cache` checks every
   invariant. The inputs are the `sources` of the dataset (below).
2. **Folds.** `index.csv` carries the fold of each binary. `python -m s02_preprocess.cache.folds --out <dir>` writes the 5-fold
   family-holdout CSV that `build_cache` reads: ransomware families and goodware identity groups go to a fold whole, largest first
   into the lightest fold. The same function on a table of your own:

```python example=folds
from s02_preprocess.cache.folds import assign_folds

pool = ([{'sha256': f'r{i}', 'label': 1, 'family': f'fam{i // 3}', 'group': '', 'arch': 'x86', 'orig_set': 'train', 'corpus': 'mendeley'}
         for i in range(30)]
        + [{'sha256': f'g{i}', 'label': 0, 'family': '', 'group': f'g{i}', 'arch': 'x86', 'orig_set': 'train', 'corpus': 'mendeley'}
           for i in range(30)])
folds = {row['sha256']: row['fold'] for row in assign_folds(pool)}
families = {}
for row in pool:
    if row['label']:
        families.setdefault(row['family'], set()).add(folds[row['sha256']])
assert all(len(f) == 1 for f in families.values())          # a family is never split
assert sorted(set(folds.values())) == [0, 1, 2, 3, 4]
```

3. **`datasets.json`** (gitignored; start from `datasets.json.example`) tells the code where it is. The `protocols` of a dataset are
   the `--protocol` values that read it by default (the first dataset that lists a protocol wins); `--cache <name>` picks a dataset on the
   command line, `GT_CACHE` overrides everything. `sources` are optional:

```json example=datasets
{
  "mine": {
    "cache": "caches/mine",
    "protocols": ["kfold"],
    "sources": {"runs": "runs_mine"}
  }
}
```

   A relative path is relative to the file. `GT_DATASETS` names another file, so a second machine or a test uses its own.

## A control or an ablation

Most ablations are options that exist: a flag of `experiments.train` is a field of `RunConfig`, and a field that differs from its
default is a token of the run name. The run name is what a gates stage finds the arm by, so an arm is a plan line:

```python example=control
from common.config import RunConfig

typed = RunConfig(model='gnn', protocol='kfold', fold=0, mask_blocks=0.15)
no_edges = RunConfig(model='gnn', protocol='kfold', fold=0, mask_blocks=0.15, control='no_edges')
assert typed.name == 'kfold0_x86_gnn-mblk15_s0'
assert no_edges.name == 'kfold0_x86_gnn-noedges-mblk15_s0'
assert typed.group == 'kfold_x86_gnn-mblk15'                       # what a gates stage names: the runs of all folds and seeds
assert RunConfig(model='gnn', protocol='kfold', fold=0, mask_blocks=0.15, tag='mine').name == 'kfold0_x86_gnn-mblk15_s0_mine'
```

`--tag <word>` labels a run without changing what it does, which is how the 500-block arms and the teammates' arms are told
apart. The controls of the edge structure are `no_edges`, `chain`, `shuffle` (degree-preserving shuffled edges) and `permute_types`
(`--control`); the others are `--untyped`, `--no-call-mp`, `--direction`, `--drop-edges`, `--flip-edges`, `--mask-tokens`,
`--mask-blocks`, `--drop-functions`, `--train-fraction`, `--vocab`, `--imports` and `--encoder`.

A new option is a `RunConfig` field with its default (so that no existing name changes), its token in `RunConfig.variant`, the code
that reads it, and a case in `tests/test_run_names.py`, which fails if any recorded name changes. A new structural control is also a branch of `control_edges` in
`s03b_cfg_feature/gnn/s06_gnn_block_cfg/edges.py` and a key of `CONTROLS` in `common/config.py`.

## A plan file

A plan is one command per line (`#` starts a comment): `train <experiments.train arguments>`, `baselines <arguments>`, `report`,
`val_threshold`, `gates <stage> [--out NAME]`.

```text example=plan
# R7 typed against R2 no edges, folds 0-4, one seed
baselines --protocol kfold --fold 0 1 2 3 4 --arch x86 --seed 0
train --protocol kfold --fold 0 --arch x86 --seed 0 --model gnn --mask-blocks 0.15
train --protocol kfold --fold 0 --arch x86 --seed 0 --model gnn --mask-blocks 0.15 --control no_edges

report
gates day1
```

```bash
python -m experiments.queue --plan experiments/plans/mine.txt --dry                    # what would run
python -m experiments.queue --plan experiments/plans/mine.txt --budget-min 300         # run it; stop after 300 GPU minutes
python -m experiments.queue --stop                                                     # stop before the next train line
```

The queue skips a `train` line whose run directory has a `metrics.json` and a `baselines` line whose directories hold a validation
threshold, so a plan can be run again after a crash or extended; a failed line is retried once (the first log is kept as
`<runs>/logs/<name>.log.try1`); `report`, `val_threshold` and `gates` still run after a stop or an exhausted budget. The output of
`gates` goes to `<runs>/gates_<stage>.md`. CPU steps the plan depends on (a block export, encoder vectors) are not plan lines: put the
commands in the plan's leading comment, as `experiments/plans/paths.txt` does. The whole plan is checked before anything starts, so a
misspelt flag is an error with its line number. See the docstring of `experiments/queue.py` for the log lines and the exit codes.

## A gates stage

A stage is data. `experiments/stages/<name>.py` defines `STAGE`: the arms `(label, group)` and the pairs `(first, second)` of labels to
compare. Build the group names from `variant_of` so they cannot drift from the run names:

```python example=stage
from s04_evaluation.stage import Stage, variant_of, x86_group

ARMS = (('R2 no edges', x86_group(variant_of(control='no_edges', mask_blocks=0.15))),
        ('R7 typed', x86_group(variant_of(mask_blocks=0.15))))
PAIRS = (('R2 no edges', 'R7 typed'),)
STAGE = Stage('mine', intro='R7 against R2, kfold x86.', arms=ARMS, pairs=PAIRS)
```

Add `'mine'` to `STAGES` in `experiments/stages/__init__.py` (and the file to `experiments/stages/`); the engine, the tables and the
command line need no change. `python -m s04_evaluation.gates --stage mine` prints the markdown and the plan's `gates mine` line
writes it to `<runs>/gates_mine.md`. `seeds=(0, 1, 2)` pairs over the seeds complete in both arms; `rule=` adds the verdict column
(`experiments/stages/noapi.py` and `team.py` are examples).

## Reading the tables

A gate summary starts with the pooled table of seed 0: one row per arm, the metrics of the five folds' test predictions pooled, with
the AUC checkpoint scored at its validation threshold. A missing arm, or one with missing folds, says so in its row.

```text excerpt=gates_noapi
## Pooled, AUC checkpoint

| arm | group | AUC | PR-AUC | macro-F1 |
|---|---|---|---|---|
| TF-IDF | tfidf | 0.982 | 0.990 | 0.935 |
| R7 typed | kfold_x86_gnn-mblk15 | 0.986 | 0.990 | 0.935 |
```

The second table is the comparison: "second minus first" with the 95% interval of a paired bootstrap that resamples *clusters* (a
ransomware family, or a single goodware binary), so that a family of near-identical samples counts once:

```text excerpt=gates_noapi
## Paired differences, second minus first

| first | second | dAUC [95% CI] | dmacro-F1 [95% CI] | rule |
|---|---|---|---|---|
| R7 typed | R7 noapi | -0.017 [-0.044, -0.001] | -0.025 [-0.048, -0.007] | api loss shown |
```

Three rules of thumb. The pooled table is one seed and the paired table uses the seeds complete in both arms, so the two can disagree
(R7 is 0.986 in the first and 0.979, the mean of three seeds, in the second); a row marked "seed 0 only" is a direction, not the test.
The pre-registered rule (`s04_evaluation/stage.py`): a second arm beats the first only when dAUC >= 0.01 and dmacro-F1 >= 0.02 and both
intervals exclude zero. And a result is read with its baselines: TF-IDF and the header-imports logistic regression are in every
table, because a model that does not beat them has not shown that it needs the code at all. `python -m s04_evaluation.stats --a <group>
--b <group>` prints one comparison, and `python -m s04_evaluation.report` writes `RESULTS.md`, one row per configuration. The tables of
the experiments that were run, and the verdicts drawn from them, are in [`history/`](history/README.md).
