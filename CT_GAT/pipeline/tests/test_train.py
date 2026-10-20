"""The training loop of `experiments/train.py` and the model registry it runs through (`common/registry.py`).

Ported from the 3-block-level-counting branch (tests/test_harness.py: patience, the supported options, the validation threshold,
the loss weights, the selection rule; tests/test_harness_prep.py: `check_supported`). The runs that train a network need CUDA
(marked `gpu`) and use the tiny cache (tests/tiny_cache.py), so they do not need the real data.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score

from common import registry
from common.config import RunConfig
from common.manifests import load_index, make_manifest
from common.registry import MODELS, ModelSpec, RunInfo, spec_for
from experiments import train
from experiments.train import loss_weight_fn, should_stop, val_auc_of, val_threshold_of
from s03b_cfg_feature.gnn import spec as gnn_spec
from s03b_cfg_feature.gnn.spec import GnnSpec
from s04_evaluation.metrics import cell_balanced, cell_weights
from s04_evaluation.val_threshold import GRID
from tiny_cache import tiny  # noqa: F401  (a fixture)

ROOT = Path(__file__).resolve().parents[1]


def gnn(**kw):
    return RunConfig(model='gnn', **kw)


# ---- the registry ---------------------------------------------------------------------------------------------------

def test_the_registry_serves_the_gnn_and_returns_a_fresh_spec_per_run():
    assert MODELS == {'gnn': 's03b_cfg_feature.gnn.spec:GnnSpec'}
    a, b = spec_for(RunConfig()), spec_for(RunConfig())
    assert type(a) is GnnSpec and a is not b
    assert isinstance(a, ModelSpec) and a.name == 'gnn' and a.requires == ('index.csv', 'vocab.json', 'samples')
    for member in ('build', 'loader', 'forward', 'variant_tokens', 'check'):
        assert callable(getattr(a, member))


def test_a_model_without_a_spec_is_refused_by_name():
    with pytest.raises(NotImplementedError, match="tokenseq.*MODELS"):
        spec_for(RunConfig(model='tokenseq', protocol='kfold', fold=0))
    with pytest.raises(NotImplementedError):
        train.run(RunConfig(model='tokenseq', protocol='kfold', fold=0))


def test_a_model_is_one_spec_and_one_registry_line(monkeypatch):
    class Stub:
        name = 'stub'
        requires = ()

        def check(self, cfg): pass
        def build(self, cfg, info): return 'net'
        def loader(self, cfg, rows, train): return rows
        def forward(self, model, batch): return batch
        def variant_tokens(self, cfg): return ('stub',)

    monkeypatch.setitem(sys.modules, 'stub_model', SimpleNamespace(Stub=Stub))
    monkeypatch.setitem(MODELS, 'stub', 'stub_model:Stub')
    spec = spec_for(SimpleNamespace(model='stub'))
    assert isinstance(spec, ModelSpec) and spec.build(None, None) == 'net'
    monkeypatch.setitem(MODELS, 'other', 'stub_model:Stub')
    with pytest.raises(ValueError, match='registered as'):
        spec_for(SimpleNamespace(model='other'))


def test_importing_the_registry_imports_no_model_and_no_torch():
    code = 'import sys\nimport common.registry\nprint(sorted(m for m in sys.modules if m.split(".")[0] in ("torch", "s03b_cfg_feature")))'
    r = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == '[]', (r.stdout, r.stderr)


def test_variant_tokens_are_the_tokens_of_the_run_name_variant():
    spec = GnnSpec()
    assert spec.variant_tokens(gnn()) == ('gnn',)
    cfg = gnn(control='no_edges', mask_blocks=0.15, imports='header', arch_balance=True, select_on='cell', budget=20000)
    assert spec.variant_tokens(cfg) == ('gnn', 'noedges', 'mblk15', 'ab', 'imph', 'selcell', 'sg', 'cap20k')
    assert '-'.join(spec.variant_tokens(cfg)) == cfg.variant and cfg.name.split('_')[2] == cfg.variant


# ---- the options a spec refuses or needs ------------------------------------------------------------------------------

def test_unsupported_flags_raise():
    spec = GnnSpec()
    spec.check(gnn())
    spec.check(gnn(stack_tfidf=True))                                        # supported since step 12
    for kw in (dict(aux_recon=0.1), dict(aux_edge=0.1)):
        with pytest.raises(NotImplementedError):
            spec.check(gnn(**kw))
    cfg, _ = train.parse_config(['--model', 'gnn', '--aux-edge', '0.1'])
    with pytest.raises(NotImplementedError):
        spec.check(cfg)
    with pytest.raises(NotImplementedError):
        train.run(cfg)
    cfg, _ = train.parse_config(['--model', 'gnn', '--stack-tfidf'])
    spec.check(cfg)


def test_block_vectors_stop_with_a_message_when_the_vectors_package_is_absent(monkeypatch):
    monkeypatch.setattr(gnn_spec, 'vectors_package_present', lambda: False)
    cfg = gnn(protocol='kfold', fold=0, block_input='vec', block_vec='ours_cap0_codebert')
    with pytest.raises(ModuleNotFoundError, match='s03c_cfg_transformer.precompute_blockvec'):
        gnn_spec.open_block_vecs(cfg)
    with pytest.raises(ModuleNotFoundError, match='block-input vec'):
        GnnSpec().build(cfg, RunInfo(None, None))


def test_in_this_checkout_the_vectors_package_is_absent_and_the_option_says_so():
    if gnn_spec.vectors_package_present():
        pytest.skip('stage 3c is present; the option reads its vectors')
    cfg = gnn(protocol='kfold', fold=0, block_input='vec', block_vec='ours_cap0_codebert')
    with pytest.raises(ModuleNotFoundError, match='s03c_cfg_transformer'):
        gnn_spec.open_block_vecs(cfg)


def test_a_cache_without_the_files_of_the_model_is_named_in_the_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='index.csv.*GT_CACHE'):
        train.check_requires(GnnSpec(), tmp_path)
    (tmp_path / 'samples').mkdir()
    for name in ('index.csv', 'vocab.json'):
        (tmp_path / name).write_text('', encoding='utf-8')
    train.check_requires(GnnSpec(), tmp_path)


# ---- patience, thresholds, weights and the selection rule ------------------------------------------------------------

def test_should_stop_rules():
    g = gnn(patience=3)
    assert not should_stop(g, 6, 2, 4) and should_stop(g, 7, 2, 4)
    assert not should_stop(g, 6, 4, 2) and should_stop(g, 7, 4, 2)
    assert not should_stop(gnn(patience=3, fixed_epochs=2), 99, 1, 1)


def test_val_threshold_of_uses_grid():
    rows = [{'sample_id': str(i), 'label': int(i >= 6)} for i in range(10)]
    probs = np.array([0.1] * 6 + [0.9] * 4)
    t = val_threshold_of(rows, [r['sample_id'] for r in rows], probs)
    assert t in GRID and 0.1 < t <= 0.9


CELLS = [(1, 'x86'), (1, 'x64'), (0, 'x86'), (0, 'x64')]


def cell_sample(counts):
    """y, arch of a synthetic set with counts[i] samples in CELLS[i]."""
    y, arch = [], []
    for (label, a), n in zip(CELLS, counts):
        y += [label] * n
        arch += [a] * n
    return np.array(y), np.array(arch)


def test_loss_weights_arch_balance_and_default():
    cpu = torch.device('cpu')
    y, arch = cell_sample([60, 10, 20, 30])
    rows = [{'sample_id': f's{i}', 'label': int(l), 'arch': a} for i, (l, a) in enumerate(zip(y, arch))]
    batch = {'sample_id': [r['sample_id'] for r in rows], 'label': torch.tensor(y, dtype=torch.float32)}
    ab = loss_weight_fn(gnn(arch_balance=True), rows, cpu)(batch).numpy()
    assert ab == pytest.approx(cell_weights(y, arch)) and ab.mean() == pytest.approx(1.0)
    plain = loss_weight_fn(gnn(), rows, cpu)(batch).numpy()
    n_pos = int(y.sum())
    assert plain == pytest.approx(np.where(y == 1, 120 / (2 * n_pos), 120 / (2 * (120 - n_pos))), abs=0)
    assert plain.mean() == pytest.approx(1.0) and not np.allclose(plain, ab)
    flipped = {'sample_id': batch['sample_id'][::-1], 'label': batch['label'].flip(0)}
    assert loss_weight_fn(gnn(arch_balance=True), rows, cpu)(flipped).numpy() == pytest.approx(ab[::-1])
    x86 = [r for r in rows if r['arch'] == 'x86']
    sub = {'sample_id': [r['sample_id'] for r in x86], 'label': torch.tensor([r['label'] for r in x86]).float()}
    assert (loss_weight_fn(gnn(arch_balance=True), x86, cpu)(sub).numpy()
            == pytest.approx(loss_weight_fn(gnn(), x86, cpu)(sub).numpy()))


SELECT_CELLS = [(1, 'x86', 200), (1, 'x64', 20), (0, 'x86', 20), (0, 'x64', 200)]


def selection_fixture():
    """A validation set where architecture predicts the label (ransomware mostly x86, goodware mostly x64), and two
    epochs: epoch 1 scores by architecture only, epoch 2 by label only."""
    rows = [{'sample_id': f'{y}{a}{i}', 'label': y, 'arch': a} for y, a, n in SELECT_CELLS for i in range(n)]
    ids = [r['sample_id'] for r in rows]
    y = np.array([r['label'] for r in rows])
    arch = np.array([r['arch'] for r in rows])
    rng = np.random.default_rng(0)
    by_arch = np.clip(np.where(arch == 'x86', 0.8, 0.2) + rng.normal(0, 0.05, len(y)), 0.01, 0.99)
    by_label = np.clip(np.where(y == 1, 0.65, 0.35) + rng.normal(0, 0.2, len(y)), 0.01, 0.99)
    return rows, ids, y, arch, [by_arch, by_label]


def test_select_on_cell_picks_other_epochs_and_thresholds_than_overall():
    rows, ids, y, arch, epochs = selection_fixture()
    vals = [train.evaluate(rows, ids, p, 0.5) for p in epochs]
    pick = lambda select_on: int(np.argmax([val_auc_of(gnn(select_on=select_on), v) for v in vals]))
    assert [v['auc'] for v in vals] == sorted([v['auc'] for v in vals], reverse=True)
    assert vals[0]['cell_balanced']['auc'] < 0.6 < vals[1]['cell_balanced']['auc']
    assert pick('overall') == 0 and pick('cell') == 1
    assert val_auc_of(gnn(), vals[0]) == vals[0]['auc'] and val_auc_of(gnn(select_on='cell'), vals[0]) == (
        vals[0]['cell_balanced']['auc'])
    thr = {s: val_threshold_of(rows, ids, epochs[pick(s)], s) for s in ('overall', 'cell')}
    assert thr['overall'] != thr['cell'] and all(t in GRID for t in thr.values())
    # on the same probabilities the two rules also disagree
    assert val_threshold_of(rows, ids, epochs[0], 'overall') != val_threshold_of(rows, ids, epochs[0], 'cell')
    assert val_threshold_of(rows, ids, epochs[0]) == val_threshold_of(rows, ids, epochs[0], 'overall')


def test_cell_threshold_is_the_weighted_macro_f1_maximiser():
    rows, ids, y, arch, epochs = selection_fixture()
    w = cell_weights(y, arch)
    for p in epochs:
        want = max(GRID, key=lambda g: f1_score(y, (p >= g).astype(int), average='macro', sample_weight=w))
        assert val_threshold_of(rows, ids, p, 'cell') == want
        assert val_threshold_of(rows, ids, p, 'cell') == max(
            GRID, key=lambda g: cell_balanced(y, p, (p >= g).astype(int), arch)['macro_f1'])


# ---- the run directory ------------------------------------------------------------------------------------------------

def fake_state(tiny, tmp_path, monkeypatch, **kw):  # noqa: F811
    cfg = gnn(protocol='kfold', fold=0, **kw)
    manifest = make_manifest(cfg, load_index(tiny.root))
    rng = np.random.default_rng(0)

    def rows(split):
        ids = [r['sample_id'] for r in getattr(manifest, split)]
        return ids, rng.random(len(ids)), [(3, 5, 5)] * len(ids)

    state = {'val': rows('val'), 'test': rows('test')}
    monkeypatch.setattr(train, 'RUNS', tmp_path)
    return cfg, manifest, state


def test_write_run_as_names_the_directory_by_the_config_or_by_out(tiny, tmp_path, monkeypatch):  # noqa: F811
    cfg, manifest, state = fake_state(tiny, tmp_path, monkeypatch, mask_blocks=0.15)
    extra = {'best_epoch': 2, 'epochs_run': 3, 'val_macro_f1': 0.9, 'auc_best_epoch': 3, 'val_auc': 0.95, 'peak_gpu_mb': 7}
    default = train.write_run_as(None, cfg, state, state, 0.4, 0.6, extra, manifest=manifest)
    assert (tmp_path / cfg.name / 'metrics.json').is_file() and default['run'] == cfg.name
    custom = train.write_run_as('smoke_x', cfg, state, state, 0.4, 0.6, extra, manifest=manifest)
    out = tmp_path / 'smoke_x'
    assert sorted(p.name for p in out.iterdir()) == sorted(p.name for p in (tmp_path / cfg.name).iterdir())
    assert (out / 'metrics.json').read_bytes() == (tmp_path / cfg.name / 'metrics.json').read_bytes()       # NaN != NaN as values
    assert custom['run'] == default['run'] == cfg.name
    assert RunConfig.load(out / 'config.json') == cfg and not list(tmp_path.glob('.stage-*'))
    assert default['peak_gpu_mb'] == 7 and default['selection']['auc']['best_epoch'] == 3


def test_the_re_exported_helpers_are_the_ones_of_run_io():
    from common import run_io
    assert (train.AUC_RULES, train.evaluate, train.write_predictions, train.truncated_count) == (
        run_io.AUC_RULES, run_io.evaluate, run_io.write_predictions, run_io.truncated_count)
    assert train.write_run is run_io.write_run and registry.RunInfo is RunInfo


# ---- a run on the tiny cache (needs CUDA) ----------------------------------------------------------------------------

@pytest.fixture
def tiny_runs(tiny, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(train, 'RUNS', tmp_path)
    monkeypatch.setattr(train, 'CACHE', tiny.root)
    return tmp_path


@pytest.mark.gpu
@pytest.mark.parametrize('select_on', ['overall', 'cell'])
def test_run_records_and_applies_the_selection_rule(tiny_runs, select_on):
    """Three epochs of a d_model 16 GNN on the 40 tiny binaries (a unit test of train.run, not a training run)."""
    cfg = gnn(protocol='kfold', fold=0, d_model=16, fixed_epochs=3, batch_size=4, num_workers=0, select_on=select_on,
              mask_blocks=0.15)
    metrics = train.run(cfg)
    out = tiny_runs / cfg.name
    manifest = make_manifest(cfg, load_index())
    assert cfg.name.endswith('-selcell_s0') == (select_on == 'cell')
    assert metrics['select_on'] == select_on and metrics['selection']['auc']['rule'] == train.AUC_RULES[select_on]
    assert json.loads((out / 'metrics.json').read_text(encoding='utf-8'))['select_on'] == select_on
    epochs = {}
    with (out / 'val_probs.csv').open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            epochs.setdefault(int(r['epoch']), {})[r['sample_id']] = float(r['prob'])
    score = lambda probs: train.evaluate(manifest.val, list(probs), np.array(list(probs.values())), 0.5)
    vals = {e: score(p) for e, p in epochs.items()}
    want = max(vals, key=lambda e: (val_auc_of(cfg, vals[e]), -e))
    assert metrics['selection']['auc']['best_epoch'] == want
    assert metrics['selection']['auc']['val_auc'] == pytest.approx(val_auc_of(cfg, vals[want]))
    with (out / 'predictions_auc.csv').open(encoding='utf-8') as f:
        val_rows = [r for r in csv.DictReader(f) if r['split'] == 'val']
    ids, probs = [r['sample_id'] for r in val_rows], np.array([float(r['prob']) for r in val_rows])
    assert metrics['selection']['auc']['threshold_val'] == val_threshold_of(manifest.val, ids, probs, select_on)
    with (out / 'predictions.csv').open(encoding='utf-8') as f:
        f1_rows = [r for r in csv.DictReader(f) if r['split'] == 'val']
    assert metrics['selection']['f1']['threshold_val'] == val_threshold_of(
        manifest.val, [r['sample_id'] for r in f1_rows], np.array([float(r['prob']) for r in f1_rows]), 'overall')
    assert metrics['best_epoch'] == metrics['selection']['f1']['best_epoch']
    assert {p.name for p in out.iterdir()} == {'config.json', 'manifest.json', 'vocab.json', 'val_probs.csv', 'history.csv',
                                               'best.pt', 'best_auc.pt', 'predictions.csv', 'predictions_auc.csv',
                                               'metrics.json'}


@pytest.mark.gpu
def test_a_run_with_the_import_bag_and_an_out_name_writes_its_files_there(tiny_runs):
    cfg = gnn(protocol='kfold', fold=1, d_model=16, fixed_epochs=2, batch_size=4, num_workers=0, imports='header')
    metrics = train.run(cfg, 'smoke_imports')
    out = tiny_runs / 'smoke_imports'
    assert metrics['run'] == cfg.name == 'kfold1_x86_gnn-imph_s0' and not (tiny_runs / cfg.name).exists()
    assert (out / 'imports_vocab.json').is_file() and RunConfig.load(out / 'config.json').cache == str(train.CACHE)
    state = torch.load(out / 'best.pt', map_location='cpu')
    assert 'imports.bag.weight' in state
    assert json.loads((out / 'imports_vocab.json').read_text(encoding='utf-8'))['which'] == 'header'
