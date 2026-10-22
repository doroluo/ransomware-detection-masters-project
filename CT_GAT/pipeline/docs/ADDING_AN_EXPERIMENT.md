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
The smallest useful model is a logistic regression on four counts of each binary, read from the cache arrays:

```python example=model
import numpy as np
import torch
from torch import nn

from common.manifests import load_sample


class CountsSpec:
    """Logistic regression on log counts of blocks, functions, instructions and API references of a binary."""
    name = 'counts'
    requires = ('index.csv', 'samples')

    def check(self, cfg):
        pass

    def build(self, cfg, info):
        return nn.Linear(4, 1)

    def loader(self, cfg, rows, train):
        for start in range(0, len(rows), cfg.batch_size):
            chunk = rows[start:start + cfg.batch_size]
            zs = [load_sample(r['sample_id']) for r in chunk]
            counts = [[len(z['blk_feat']), len(z['fn_flags']), len(z['mn']), len(z['api'])] for z in zs]
            yield {'sample_id': [r['sample_id'] for r in chunk], 'label': torch.tensor([r['label'] for r in chunk]).float(),
                   'x': torch.tensor(np.log1p(counts)).float(), 'n_segments_total': [1] * len(chunk)}

    def forward(self, model, batch):
        return model(batch['x']).squeeze(1)

    def variant_tokens(self, cfg):
        return ('counts',)
```

The registry lines (the first is the real one; the module path is where the class can be imported from):

```python example=registry
MODELS = {
    'fivechannel': 's03a_transformer.five_channel.spec:FiveChannelSpec',
    'counts': 'my_stage.spec:CountsSpec',
}
```

Then `RunConfig` has to know the name, because it fixes the run name: add `'counts'` to `MODELS` and to the `model: Literal[...]` field of
`common/config.py`, and make `RunConfig.variant` return a variant that starts with the model word (see its `fivechannel` branch). A
model that needs its own loop (Model 1 does) gives its spec a `run(cfg, out_name, **options)` and the registry hands the run over.
`requires` lists the paths under the cache that every run reads; a missing one is named before anything is written.
The model-specific flags that `RunConfig` has no field for are the `MODEL_OPTIONS` of `experiments/train.py`.

Training needs CUDA. To check a spec without a GPU, build it and run one batch, as `tests/test_docs.py` does with the example.

## A dataset

A dataset is a CFG cache plus the fold assignment of the protocols that read it. It is declared in one of two places: a row of
`common/catalog.py`, which is how the repository's own datasets (`mendeley_x86`, `combined_v3`) are declared and needs no
per-machine file, or an entry of `datasets.json`, which belongs to one machine.

1. **A catalog row.** `CORPORA` says what a corpus is: the collection folders under `CT_GAT/data` that hold its `samples.csv`,
   and the `rule` of `build_cache.exclusion` that leaves rows out (`split` for the Mendeley collections, `eligible` for the
   balanced ones). A new collection folder is also named in `COLLECTIONS`, as the Drive zip unpacks. `DATASETS` says what a cache
   holds: the `--protocol` values that read it by default (the first dataset that lists a protocol wins), its corpora, the fold
   files it reads under `s02_preprocess/folds/`, and the `base` dataset whose cache seeds it. The cache of a dataset is
   `CT_GAT/cache/<name>/`. `python -m common.catalog` lists the dataset with its collections, cache and fold files, and
   `tests/test_catalog.py` checks that every collection name is known, and that every corpus, base dataset and fold file of the table exists.
2. **Cache.** `python -m s02_preprocess.cache.build_cache --cache <name>` streams the IDA exports (`s01_ida_export`) of the
   corpora into `<cache>/samples/*.npz`, `index.csv` and `vocab.json`; `python -m s02_preprocess.cache.check_cache --cache <name>`
   checks every invariant. A folder that is not unzipped stops the build with its name and the zip that holds it.
3. **Folds.** `index.csv` carries the fold of each binary, which `build_cache` takes from the fold files by sha256:
   `s02_preprocess/folds/folds_mendeley.csv` for the Mendeley corpus and `s02_preprocess/folds/v3/folds_all.csv` for every corpus
   (columns in [`s02_preprocess/folds/README.md`](../s02_preprocess/folds/README.md)). Those are the files of the two datasets
   above; a dataset reads the fold files named by its own `folds` entry in `DATASETS` (paths under `s02_preprocess/folds/`, keys
   `folds` and `folds_v3`), so the rows of a new corpus go into the files that entry names, committed there, and `build_cache
   --cache <name>` joins them. `GT_FOLDS` and `GT_FOLDS_V3`, or `folds` and `folds_v3` under `sources` of an entry of
   `datasets.json`, replace them on one machine. `python -m s02_preprocess.cache.folds` writes the 5-fold family-holdout CSV to
   `CT_GAT/cache/<dataset>/` unless `--out` says otherwise: ransomware families and goodware identity groups go to a fold whole,
   largest first into the lightest fold. The same function on a table of your own:

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

4. **A `datasets.json` entry** declares the same thing on one machine, for a cache or raw inputs outside the repository
   (start from `datasets.json.example`). The `protocols` of a dataset are the `--protocol` values that read it by default;
   `--cache <name>` picks a dataset on the command line, `GT_CACHE` overrides everything. `sources` are optional:

```json example=datasets
{
  "mine": {
    "cache": "caches/mine",
    "protocols": ["kfold"],
    "sources": {"runs": "runs_mine"}
  }
}
```

   A relative path is relative to the file. A file entry named like a catalog dataset replaces it; any other name adds a dataset
   ahead of the catalog's. `GT_DATASETS` names another file, so a second machine or a test uses its own.

## A control or an ablation

Most ablations are options that exist: a flag of `experiments.train` is a field of `RunConfig`, and a field that differs from its
default is a token of the run name. The run name is what a gates stage finds the arm by, so an arm is a plan line:

```python example=control
from common.config import RunConfig

base = RunConfig(model='fivechannel', protocol='kfold', fold=0)
long = RunConfig(model='fivechannel', protocol='kfold', fold=0, max_instructions=512)
assert base.name == 'kfold0_x86_fivechannel_s0'
assert long.name == 'kfold0_x86_fivechannel-L512_s0'
assert long.group == 'kfold_x86_fivechannel-L512'                  # what a gates stage names: the runs of all folds and seeds
assert RunConfig(model='fivechannel', protocol='kfold', fold=0, tag='mine').name == 'kfold0_x86_fivechannel_s0_mine'
```

`--tag <word>` labels a run without changing what it does, which is how the uncapped arms and the teammates' arms are told apart.
`--max-instructions` (instructions per binary, `-L512`) and `--threshold-from oof` (the decision threshold chosen on out-of-fold
predictions of the training binaries instead of the validation families, run name suffix `-thoof`;
`experiments.train.choose_thresholds` is where a new model asks for it) are the options of `fivechannel`.

`RunConfig` also keeps the fields of the earlier jsonl-only models (the hierarchical GNN, the token-sequence transformer), because the
run directories they wrote are read by name; `tests/test_run_names.py` fails if any recorded name changes. A new option is a
`RunConfig` field with its default (so that no existing name changes), its token in `RunConfig.variant`, the code that reads it, and a
case in that test.

## A plan file

A plan is one command per line (`#` starts a comment): `train <experiments.train arguments>`, `train_b <arguments>`, `report`,
`val_threshold`, `gates <stage> [--out NAME]`.

```text example=plan
# five-channel Model 1 at 128 and 512 instructions, fold 0, one seed
train --protocol kfold --fold 0 --arch x86 --seed 0 --model fivechannel
train --protocol kfold --fold 0 --arch x86 --seed 0 --model fivechannel --max-instructions 512

report
gates model1_variants
```

```bash
python -m experiments.queue --plan experiments/plans/mine.txt --dry                    # what would run
python -m experiments.queue --plan experiments/plans/mine.txt --budget-min 300         # run it; stop after 300 GPU minutes
python -m experiments.queue --stop                                                     # stop before the next train line
```

The queue skips a `train` line whose run directory has a `metrics.json`, so a plan can be run again after a crash or extended; a
failed line is retried once (the first log is kept as `<runs>/logs/<name>.log.try1`); `report`, `val_threshold` and `gates` still run
after a stop or an exhausted budget. The output of `gates` goes to `<runs>/gates_<stage>.md`. CPU steps the plan depends on (a
pipeline-B build) are not plan lines: put the commands in the plan's leading comment, as `experiments/plans/model1_variants.txt`
does. The whole plan is checked before anything starts, so a misspelt flag is an error with its line number. See the docstring of `experiments/queue.py` for the log lines and the exit codes.

## A gates stage

A stage is data. `experiments/stages/<name>.py` defines `STAGE`: the arms `(label, group)` and the pairs `(first, second)` of labels to
compare. Build the group names from `RunConfig.group` so they cannot drift from the run names:

```python example=stage
from common.config import RunConfig
from s04_evaluation.stage import Stage

KFOLD = dict(protocol='kfold', fold=0, arch='x86', model='fivechannel')
ARMS = (('five-channel L128', RunConfig(**KFOLD).group),
        ('five-channel L512', RunConfig(**KFOLD, max_instructions=512).group))
PAIRS = (('five-channel L128', 'five-channel L512'),)
STAGE = Stage('mine', intro='L512 against L128, kfold x86.', arms=ARMS, pairs=PAIRS)
```

Add `'mine'` to `STAGES` in `experiments/stages/__init__.py` (and the file to `experiments/stages/`); the engine, the tables and the
command line need no change. `python -m s04_evaluation.gates --stage mine` prints the markdown and the plan's `gates mine` line
writes it to `<runs>/gates_mine.md`. `seeds=(0, 1, 2)` pairs over the seeds complete in both arms; `rule=` adds the verdict column
(`experiments/stages/team.py` is an example).

## Reading the tables

A gate summary starts with the pooled table of seed 0: one row per arm, the metrics of the five folds' test predictions pooled, with
the AUC checkpoint scored at its validation threshold. A missing arm, or one with missing folds, says so in its row.

```text excerpt=gates_team
## Pooled, AUC checkpoint

| arm | group | AUC | PR-AUC | macro-F1 |
|---|---|---|---|---|
| team sig500v3 emb (Model 3) | kfold_x86_gnn_team-sig500v3-emb | 0.971 | 0.985 | 0.935 |
| fivechannel-L512 whole | kfold_x86_fivechannel-L512_whole | 0.910 | 0.948 | 0.846 |
```

The second table is the comparison: "second minus first" with the 95% interval of a paired bootstrap that resamples *clusters* (a
ransomware family, or a single goodware binary), so that a family of near-identical samples counts once:

```text excerpt=gates_team
## Paired differences, second minus first

| first | second | dAUC [95% CI] | dmacro-F1 [95% CI] | rule |
|---|---|---|---|---|
| team sig500v3 feat (Model 2) | team sig500v3 emb (Model 3) | +0.012 [+0.000, +0.025] | +0.031 [+0.002, +0.069] | beats |
```

Three rules of thumb. The pooled table is one seed and the paired table uses the seeds complete in both arms, so the two can
disagree; a row marked "seed 0 only" is a direction, not the test. The pre-registered rule (`s04_evaluation/stage.py`): a second arm
beats the first only when dAUC >= 0.01 and dmacro-F1 >= 0.02 and both intervals exclude zero. And a result is read with its
baselines: the TF-IDF and header-imports logistic regressions are rows of the `team` table that read their published runs (their code
is on branch `jsonl-reference-models`), because a model that does not beat them has not shown that it needs the code at all.
`python -m s04_evaluation.stats --a <group> --b <group>` prints one comparison, and `python -m s04_evaluation.report` writes
`RESULTS.md`, one row per configuration. The tables of the experiments that were run, and the verdicts drawn from them, are in
[`history/`](history/README.md).
