"""python -m pytest tests/test_docs.py: the documentation is checked against the code.

* every block tagged `example=<name>` in docs/ADDING_AN_EXPERIMENT.md runs here, on the tiny cache of tests/tiny_cache.py (the
  model on the CPU), and every tag of the document has a test;
* the excerpts of the gate tables in the document are lines of docs/history/gates_team.md;
* every relative link of every markdown file of the pipeline resolves, the stage READMEs say how to run their stage from
  `CT_GAT/pipeline`, and the history index lists every file of docs/history.
"""
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from common import paths
from common.config import RunConfig
from common.manifests import load_index, make_manifest
from common.registry import MODELS, ModelSpec, RunInfo, spec_for
from tiny_cache import tiny  # noqa: F401  (a fixture)

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'docs' / 'ADDING_AN_EXPERIMENT.md'
BLOCK = re.compile(r'^```(\w+) (example|excerpt)=([\w-]+)\n(.*?)^```', re.S | re.M)


def blocks(kind='example'):
    """{name: (language, code)} of the tagged blocks of the document."""
    found = {}
    for lang, tag, name, code in BLOCK.findall(DOC.read_text(encoding='utf-8')):
        if tag == kind and kind == 'example':
            assert name not in found, f'two examples named {name}'
            found[name] = (lang, code)
        elif tag == kind:
            found.setdefault(name, []).append(code)
    return found


EXAMPLES = blocks()
TESTED = {'model', 'registry', 'folds', 'datasets', 'control', 'plan', 'stage'}


def run_example(name, module_name=None):
    """Execute a python example as the module `module_name` (importable by name, as an encoder class must be)."""
    module = types.ModuleType(module_name or f'docs_example_{name}')
    sys.modules[module.__name__] = module
    try:
        exec(compile(EXAMPLES[name][1], f'ADDING_AN_EXPERIMENT.md:{name}', 'exec'), module.__dict__)
    except BaseException:
        sys.modules.pop(module.__name__, None)
        raise
    return module


def test_every_example_of_the_document_has_a_test_and_the_other_way_round():
    assert set(EXAMPLES) == TESTED


# a model ---------------------------------------------------------------------------------------------------------------

def test_the_model_example_builds_loads_and_runs_one_batch(tiny, tmp_path):  # noqa: F811
    mod = run_example('model')
    cfg = RunConfig(protocol='kfold', fold=0, batch_size=4, num_workers=0)
    manifest = make_manifest(cfg, load_index())
    spec = mod.CountsSpec()
    assert isinstance(spec, ModelSpec) and spec.name == 'counts' and spec.requires == ('index.csv', 'samples')
    spec.check(cfg)
    net = spec.build(cfg, RunInfo(manifest, tmp_path))
    batch = next(iter(spec.loader(cfg, manifest.val, False)))
    logits = spec.forward(net, batch)
    assert logits.shape == (len(batch['sample_id']),) == (min(4, len(manifest.val)),) and torch.isfinite(logits).all()
    assert batch['x'].shape == (len(batch['sample_id']), 4) and torch.isfinite(batch['x']).all()
    assert spec.variant_tokens(cfg) == ('counts',)


def test_the_registry_example_extends_the_real_registry_by_one_line(monkeypatch):
    table = {}
    exec(EXAMPLES['registry'][1], table)
    shown = table['MODELS']
    assert {k: v for k, v in shown.items() if k != 'counts'} == MODELS and list(shown)[-1] == 'counts'
    mod = run_example('model')
    monkeypatch.setitem(sys.modules, 'my_stage.spec', SimpleNamespace(CountsSpec=mod.CountsSpec))
    monkeypatch.setitem(MODELS, 'counts', shown['counts'])
    assert type(spec_for(SimpleNamespace(model='counts'))) is mod.CountsSpec


@pytest.mark.gpu
def test_the_model_example_trains_one_epoch_through_the_registry_loop(tiny, tmp_path, monkeypatch):  # noqa: F811
    """The registry loop on CUDA (bf16 autocast) for a spec of the document: the run directory has its metrics and predictions."""
    import json

    from common import config
    from experiments import train
    mod = run_example('model')
    monkeypatch.setattr(config, 'MODELS', (*config.MODELS, 'counts'))
    plain = RunConfig.variant
    monkeypatch.setattr(RunConfig, 'variant', property(lambda self: 'counts' if self.model == 'counts' else plain.fget(self)))
    monkeypatch.setitem(sys.modules, 'my_stage.spec', SimpleNamespace(CountsSpec=mod.CountsSpec))
    monkeypatch.setitem(MODELS, 'counts', 'my_stage.spec:CountsSpec')
    monkeypatch.setattr(train, 'CACHE', tiny.root)
    monkeypatch.setattr(train, 'RUNS', tmp_path / 'runs')
    cfg = RunConfig(model='counts', protocol='kfold', fold=0, batch_size=4, num_workers=0, fixed_epochs=1)
    metrics = train.run(cfg, 'counts_smoke')
    out = tmp_path / 'runs' / 'counts_smoke'
    assert metrics['epochs_run'] == 1 and json.loads((out / 'metrics.json').read_text(encoding='utf-8'))['run'] == cfg.name
    assert (out / 'predictions.csv').stat().st_size > 0


# a dataset -------------------------------------------------------------------------------------------------------------

def test_the_folds_example_never_splits_a_family():
    run_example('folds')


def test_the_datasets_example_is_read_by_the_registry(tmp_path):
    text = EXAMPLES['datasets'][1]
    path = tmp_path / 'datasets.json'
    path.write_text(text, encoding='utf-8')
    registry_ = paths.load_registry({'GT_DATASETS': str(path)})
    assert registry_['mine']['cache'] == tmp_path / 'caches' / 'mine' and registry_['mine']['protocols'] == ('kfold',)
    env = {'GT_DATASETS': str(path)}
    assert paths.cache_for('kfold', env=env) == tmp_path / 'caches' / 'mine'
    assert paths.cache_for('kfold_v3', env=env) == paths.CACHE_ROOT / 'combined_v3'   # no file dataset lists it: the catalog's does
    assert paths.cache_for('kfold_v3', env=env, name='mine') == tmp_path / 'caches' / 'mine'
    assert paths.runs_for(env=env) == tmp_path / 'runs_mine'
    assert paths.cache_for('kfold', env={**env, 'GT_CACHE': 'elsewhere'}) == Path('elsewhere')


# a control -------------------------------------------------------------------------------------------------------------

def test_the_control_example_names_are_the_run_names():
    run_example('control')


# a plan ----------------------------------------------------------------------------------------------------------------

def test_the_plan_example_is_a_valid_plan_with_the_runs_it_says():
    from experiments import queue
    steps = queue.parse_plan(EXAMPLES['plan'][1])
    assert [s.verb for s in steps] == ['train', 'train', 'report', 'gates']
    assert [s.label for s in steps[:2]] == ['kfold0_x86_fivechannel_s0', 'kfold0_x86_fivechannel-L512_s0']
    assert steps[0].done_files == ('kfold0_x86_fivechannel_s0/metrics.json',)


# a gates stage ---------------------------------------------------------------------------------------------------------

def test_the_stage_example_renders_against_runs_on_disk(tmp_path, monkeypatch):
    from eval_synth import build_fake
    from s04_evaluation import gate_tables, gates, stats
    mod = run_example('stage')
    stage = mod.STAGE
    assert [group for _, group in stage.arms] == ['kfold_x86_fivechannel', 'kfold_x86_fivechannel-L512']
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    gate_tables.pooled.cache_clear()
    try:
        build_fake(tmp_path, {(('model', 'fivechannel'),): (1.2, range(5)),
                              (('model', 'fivechannel'), ('max_instructions', 512)): (1.6, range(5))})
        text = '\n'.join(gates.render(stage, 30))
    finally:
        gate_tables.pooled.cache_clear()
    assert text.startswith('# Gate summary, stage mine') and 'L512 against L128, kfold x86.' in text
    assert '| five-channel L128 | kfold_x86_fivechannel | 0.' in text and 'missing' not in text.split('## Paired')[0]
    assert '| five-channel L128 | five-channel L512 |' in text


# the tables ------------------------------------------------------------------------------------------------------------

def test_the_excerpts_of_the_document_are_lines_of_the_final_gate_table():
    table = (ROOT / 'docs' / 'history' / 'gates_team.md').read_text(encoding='utf-8').splitlines()
    excerpts = blocks('excerpt')['gates_team']
    assert len(excerpts) == 2
    for code in excerpts:
        missing = [line for line in code.splitlines() if line and line not in table]
        assert not missing, missing


# the markdown ----------------------------------------------------------------------------------------------------------

def markdown_files():
    return sorted(p for p in ROOT.rglob('*.md') if not {'.pytest_cache', '.ruff_cache', 'runs'} & set(p.relative_to(ROOT).parts))


def test_every_relative_link_resolves():
    files = markdown_files()
    assert len(files) > 10
    bad = []
    for md in files:
        for target in re.findall(r'\]\(([^)\s]+)\)', md.read_text(encoding='utf-8')):
            if target.startswith(('http://', 'https://', 'mailto:', '#')):
                continue
            if not (md.parent / target.split('#')[0]).exists():
                bad.append(f'{md.relative_to(ROOT)} -> {target}')
    assert not bad, bad


@pytest.mark.parametrize('readme', ['s02_preprocess', 's03a_transformer', 's03b_cfg_feature', 's03c_cfg_transformer', 's04_evaluation'])
def test_a_stage_readme_says_how_to_run_the_stage_from_the_pipeline_root(readme):
    text = (ROOT / readme / 'README.md').read_text(encoding='utf-8')
    assert 'CT_GAT/pipeline' in text and 'python -m' in text


def test_the_root_readme_lists_every_stage_folder_and_every_plan():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    for folder in ('s01_ida_export', 's02_preprocess', 's03a_transformer', 's03b_cfg_feature', 's03c_cfg_transformer',
                   's04_evaluation', 'common', 'experiments', 'docs', 'tests'):
        assert f'`{folder}/`' in text, folder
    for plan in (ROOT / 'experiments' / 'plans').glob('*.txt'):
        assert f'`{plan.stem}`' in text, plan.name
    assert '```mermaid' in text and 'flowchart TD' in text


def test_the_history_index_lists_every_file_of_the_history():
    index = (ROOT / 'docs' / 'history' / 'README.md').read_text(encoding='utf-8')
    for path in (ROOT / 'docs' / 'history').glob('*.md'):
        if path.name != 'README.md':
            assert f'({path.name})' in index, path.name


def test_the_final_gate_tables_are_the_stages_that_exist():
    from s04_evaluation.gates import stage_names
    stages = {'gates_team': 'team', 'gates_model1_variants': 'model1_variants'}       # the other tables are those of stages that moved
    for file, stage in stages.items():
        head = (ROOT / 'docs' / 'history' / f'{file}.md').read_text(encoding='utf-8').splitlines()[0]
        assert stage in stage_names() and head.startswith(f'# Gate summary, stage {stage} '), file
