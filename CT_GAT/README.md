## Script names

- Extract CFGs from binaries inside the VM: `preprocessing/extract_cfg_with_ida.py` (IDA required).
- Read existing downloaded CFGs for ML: `preprocessing/gnn_dataset.py` (no IDA required).
  Use `functions(sample)` for every function, or `sample_graph(sample)` to combine their CFGs as separate components.
- `preprocessing/ida_cfg_research.py` remains a compatibility alias for existing imports and commands.

The extractor code is unchanged by this rename. Running jobs use their existing frozen VM tools.

## Using the Mendeley dataset for GNNs

Start with [the portable dataset guide](GNN_DATASET_GUIDE.md). It explains the
sample index, explicit train/test lists, block/edge format, architecture filters,
and Python/PyG loading examples. `preprocessing/build_gnn_bundle.py` packages
the raw exports without re-extraction; `preprocessing/gnn_dataset.py` provides
the dependency-free reader. The default manifests include 1,950 training and
484 test binaries with recovered native CFGs and accepted cohort identities;
the full 2,670-sample inventory and original occurrence metadata are retained.

## Research dataset: IDAPython CFG extraction

The current VM pipeline uses `preprocessing/extract_cfg_with_ida.py` and
`preprocessing/run_research_cfg.py` (Python 3.8, IDA Pro 9.4). It preserves
uncapped recovered CFGs, structured operands, API references, symbols, strings,
and PE features, with isolated static analysis, resource monitoring, resumable
workers, and verified incremental transfer to the desktop.

Start with [the quality review and operating policy](IDA_CFG_QUALITY_REVIEW.md)
for the active schema, resource settings, output layout, validation evidence,
and limitations. `preprocessing/reconcile_cfg_dataset.py` produces separate
experiment manifests with verified UPX identity aliases and disjoint train/test
identity groups, preserving all original cohort and folder metadata.

The [initial plan](IDA_CFG_PLAN.md) and [original pilot runbook](IDA_CFG_RUNBOOK.md)
document the superseded minimal export. Samples, generated feature datasets,
logs and SSH keys are not stored in this repository.

Run the extraction/identity regression tests with:

```sh
python -m unittest discover -s CT_GAT/preprocessing -p 'test_*.py'
```

The [Mendeley assembly companion](IDA_ASM_RUNBOOK.md) exports full IDA text listings
from the same analyzed database as the CFG, with incremental verified desktop transfer. Earlier samples receive a separate assembly backfill.
