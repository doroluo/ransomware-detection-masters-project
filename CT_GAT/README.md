## Research dataset: IDAPython CFG extraction

The current VM pipeline uses `preprocessing/ida_cfg_research.py` and
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
