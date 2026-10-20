"""python -m pytest tests/test_model1_trainer.py: Model 1 (`s03a_transformer/token_sequence/train.py`), its registry entry
`tokenseq` and the gate stage `model1` on synthetic runs.

Ported from the 3-block-level-counting branch (tests/test_model1_trainer.py). CPU only. The block export is built from the small
synthetic cache of tests/tiny_cache.py (40 binaries, all five folds); no test needs the real caches, the real export or a GPU,
except the one that checks bit-identical evaluation on CUDA (`gpu`). The run script of the branch (scripts/model1_runs.sh) comes
with the experiment queue, with its tests.
"""
import csv
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tiny_cache import tiny  # noqa: F401  (a fixture)

import experiments.train as experiments_train
from common import run_io
from common.config import RunConfig
from common.manifests import load_index, make_manifest
from common.reader import BlockStore
from common.registry import MODELS, spec_for
from experiments.train import val_threshold_of
from s03a_transformer.token_sequence import BinaryTokenSequence, FoldSequences, Model1TokenSequence
from s03a_transformer.token_sequence import train as ts
from s03a_transformer.token_sequence import spec as spec_module
from s03a_transformer.token_sequence.spec import TokenSeqSpec
from s03c_cfg_transformer import block_export as be
from s04_evaluation import gate_tables, gates, report, stats

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
FOLDS = range(5)
GNN_KEYS = ('run', 'select_on', 'best_epoch', 'epochs_run', 'val_macro_f1', 'truncated_test', 'test', 'floors', 'family_recall',
            'mean_train_epoch_secs', 'epoch_secs_mean', 'peak_gpu_mb', 'selection')
RUN_FILES = ('config.json', 'manifest.json', 'history.csv', 'best.pt', 'best_auc.pt', 'predictions.csv', 'predictions_auc.csv',
             'metrics.json')


@pytest.fixture(scope='module')
def export(tiny, tmp_path_factory):  # noqa: F811
    """A block export of the 40 binaries of the tiny cache with the five folds (carved with val_seed 0, as the real export)."""
    for path in (tiny.root / 'samples').glob('*.npz'):                    # the tiny samples carry no vocabulary hash
        with np.load(path) as z:
            arrays = dict(z)
        np.savez_compressed(path, **arrays, vocab_sha=np.array('tiny'))
    index = load_index()
    base = RunConfig(model='gnn', protocol='kfold', fold=0, arch='x86', seed=0, val_seed=0)
    manifests = {k: make_manifest(replace(base, fold=k), index) for k in FOLDS}
    rows = sorted((r for r in index if r['arch'] == 'x86' and r['fold'] is not None), key=lambda r: r['sample_id'])
    out = tmp_path_factory.mktemp('export') / 'tiny'
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(be, 'CACHE', tiny.root)
        be.build_export(base, out, rows, manifests, log=lambda *_: None)
    return SimpleNamespace(root=out, store=BlockStore(out), manifests=manifests)


def tok(fold=0, **kw):
    """A small-window Model 1 config: windows of 32 positions, 8 per binary at most, 2 epochs."""
    return RunConfig(**{'protocol': 'kfold', 'fold': fold, 'arch': 'x86', 'model': 'tokenseq', 'seq_len': 32, 'max_windows': 8,
                        'epochs': 2, 'num_workers': 0, **kw})


def train(export, cfg, out, **kw):
    return ts.run(cfg, export.root, 'cpu', None, kw.pop('window_budget', 6), out, log=lambda *_: None, **kw)


# ---- the run directory ------------------------------------------------------------------------------------------------

def test_a_two_epoch_run_writes_the_run_directory_the_harness_reads(export, tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    for fold in FOLDS:
        train(export, tok(fold), runs)
    d = runs / 'kfold0_x86_tokenseq-L32-w8_s0'
    for name in RUN_FILES:
        assert (d / name).is_file(), name
    cfg = RunConfig.load(d / 'config.json')
    assert cfg.name == d.name and cfg.model == 'tokenseq' and cfg.val_seed == 0 and cfg.group == 'kfold_x86_tokenseq-L32-w8'
    metrics = json.loads((d / 'metrics.json').read_text(encoding='utf-8'))
    assert set(GNN_KEYS) <= set(metrics) and metrics['run'] == d.name and metrics['epochs_run'] == 2
    assert set(metrics['selection']) == {'f1', 'auc'}
    assert {'rule', 'best_epoch', 'val_auc', 'threshold_val', 'test', 'test_at_half', 'family_recall'} <= set(metrics['selection']['auc'])
    assert metrics['export_sha'] == export.store.export_sha and metrics['smoke'] is None and metrics['device'] == 'cpu'
    with (d / 'history.csv').open(encoding='utf-8') as f:
        history = list(csv.DictReader(f))
    assert [h['epoch'] for h in history] == ['1', '2']
    assert {'loss', 'train_secs', 'val_secs', 'epoch_secs', 'val_macro_f1', 'val_auc', 'peak_gpu_mb'} <= set(history[0])
    # the manifest is the export's, and the predictions cover its validation and test binaries
    manifest = json.loads((d / 'manifest.json').read_text(encoding='utf-8'))
    want = export.manifests[0]
    assert manifest['train'] == [r['sample_id'] for r in want.train] and manifest['test'] == [r['sample_id'] for r in want.test]
    with (d / 'predictions.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert {r['sample_id'] for r in rows if r['split'] == 'test'} == set(manifest['test'])
    assert {r['sample_id'] for r in rows if r['split'] == 'val'} == set(manifest['val'])
    assert all(int(r['pred']) == int(float(r['prob']) >= 0.5) for r in rows)
    thr = metrics['selection']['auc']['threshold_val']
    with (d / 'predictions_auc.csv').open(encoding='utf-8') as f:
        assert all(int(r['pred']) == int(float(r['prob']) >= thr) for r in csv.DictReader(f))
    # stats.py, report.py and gates.py read it (five folds of seed 0)
    monkeypatch.setattr(stats, 'RUNS', runs)
    monkeypatch.setattr(report, 'RUNS', runs)
    group = cfg.group
    for selection in ('f1', 'auc'):
        complete, partial = stats.load_group(group, selection)
        assert sorted(complete) == [0] and not partial and len(complete[0].y) == 40
        result, info = stats.compare(group, group, selection, resamples=20)
        assert info['seeds'] == [0] and info['n'] == 40 and result['auc']['diff'] == 0
    gate_tables.pooled.cache_clear()
    auc, pr_auc, macro_f1 = gate_tables.pooled(group, 'auc')
    gate_tables.pooled.cache_clear()
    assert 0 <= auc <= 1 and 0 <= macro_f1 <= 1
    report.main(['--out', str(tmp_path / 'RESULTS.md')])
    text = (tmp_path / 'RESULTS.md').read_text(encoding='utf-8')
    line = next(l for l in text.splitlines() if l.startswith('| Model 1 token-sequence L32 w8 '))
    assert '| kfold | x86 | 1 |' in line and 'val_seed' not in line and "'epochs': 2" in line


def test_both_checkpoints_are_written_loadable_and_chosen_by_their_own_epoch(export, tmp_path):
    runs = tmp_path / 'runs'
    metrics = train(export, tok(1, epochs=3), runs)
    d = runs / tok(1, epochs=3).name
    train_ds = FoldSequences(export.store, 1, 'train', L=32)
    net = BinaryTokenSequence(Model1TokenSequence(n_ids=train_ds.n_ids, max_seq_len=32))
    for ckpt in ('best.pt', 'best_auc.pt'):
        state = torch.load(d / ckpt, map_location='cpu')
        assert set(state) == set(net.state_dict())
        net.load_state_dict(state)
    with (d / 'history.csv').open(encoding='utf-8') as f:
        history = list(csv.DictReader(f))
    f1 = [float(h['val_macro_f1']) for h in history]
    auc = [float(h['val_auc']) for h in history]
    assert metrics['best_epoch'] == 1 + int(np.argmax(f1)) and metrics['selection']['auc']['best_epoch'] == 1 + int(np.argmax(auc))
    assert metrics['val_macro_f1'] == pytest.approx(max(f1)) and metrics['selection']['auc']['val_auc'] == pytest.approx(max(auc))


def test_patience_counts_validation_auc_epochs_without_improvement(export, tmp_path, monkeypatch):
    aucs = iter([0.9, 0.8, 0.7, 0.95, 0.5])
    real = ts.evaluate

    def scripted(rows, ids, probs, threshold):
        return {**real(rows, ids, probs, threshold), 'auc': next(aucs)}
    monkeypatch.setattr(ts, 'evaluate', scripted)
    metrics = train(export, tok(2, epochs=5, patience=2), tmp_path)
    assert metrics['epochs_run'] == 3 and metrics['selection']['auc']['best_epoch'] == 1       # epochs 2 and 3 did not beat 0.9


# ---- the loss, the batches and the windows -----------------------------------------------------------------------------

def test_pack_bounds_the_windows_of_a_batch_and_keeps_every_item_once():
    sizes = [3, 5, 1, 9, 2, 2, 4, 0, 6]
    order = [4, 0, 8, 3, 1, 7, 2, 6, 5]
    batches = ts.pack(sizes, order, 8)
    assert [i for b in batches for i in b] == order                                          # nothing lost, nothing repeated
    for b in batches:
        assert sum(sizes[i] for i in b) <= 8 or len(b) == 1                                  # a lone big item may exceed
    assert [b for b in batches if sum(sizes[i] for i in b) > 8] == [[3]]
    assert ts.pack(sizes, [], 8) == []
    assert ts.pack([2, 2, 2], [0, 1, 2], 100) == [[0, 1, 2]]


def test_window_counts_are_exact_in_eval_and_an_upper_bound_in_training(export):
    for split in ('val', 'test'):
        ds = FoldSequences(export.store, 0, split, L=16)
        for i in range(len(ds)):
            assert ds.n_windows(i) == len(ds[i]['ids']) == -(-len(ds.stream(i).ids) // 16)
        assert (ds.positions() == [len(ds.stream(i).ids) for i in range(len(ds))]).all()
    ds = FoldSequences(export.store, 0, 'train', L=16, max_windows=3, seed=4)
    for epoch in range(4):
        ds.set_epoch(epoch)
        assert all(len(ds[i]['ids']) <= ds.n_windows(i) <= 3 for i in range(len(ds)))


def test_subset_keeps_the_given_binaries_of_the_split_only(export):
    full = FoldSequences(export.store, 0, 'val', L=16)                  # (a train item's windows are drawn by its position)
    pick = [full.binary(3), full.binary(1)]
    ds = FoldSequences(export.store, 0, 'val', L=16, subset=pick)
    assert len(ds) == 2 and ds.sample_ids() == [full.sample_ids()[3], full.sample_ids()[1]]
    assert ds.labels() == [full.labels()[3], full.labels()[1]] and torch_equal(ds[0], full[3])
    other = FoldSequences(export.store, 0, 'test', L=16).binary(0)
    with pytest.raises(ValueError, match='not in the val split'):
        FoldSequences(export.store, 0, 'val', L=16, subset=[other])


def torch_equal(a, b):
    return all(torch.equal(a[k], b[k]) for k in ('ids', 'kind', 'pad_mask')) and a['label'] == b['label']


def test_prefetch_gives_the_same_batches_with_or_without_threads(export):
    ds = FoldSequences(export.store, 0, 'train', L=16, max_windows=4, seed=3)
    ds.set_epoch(2)
    plan = ts.pack([ds.n_windows(i) for i in range(len(ds))], np.random.default_rng(0).permutation(len(ds)), 5)
    inline = list(ts.prefetch(ds, plan, 0))
    for workers in (1, 3):
        threaded = list(ts.prefetch(ds, plan, workers))
        assert len(threaded) == len(inline) == len(plan)
        for a, b in zip(inline, threaded):
            assert all(torch.equal(a[k], b[k]) for k in ('ids', 'kind', 'pad_mask', 'win2bin', 'labels'))
            assert a['sample_ids'] == b['sample_ids']
    assert [i for batch in inline for i in batch['sample_ids']] == [ds.sample_ids()[i] for b in plan for i in b]


def test_the_loss_weights_the_classes_by_n_over_two_n_class(export, tmp_path, monkeypatch):
    """One step on a real batch: the loss equals mean(w[label] * CE) with w_c = N / (2 n_c) from the training labels."""
    seen = {}
    real = ts.F.cross_entropy

    def spy(logits, labels, reduction='mean', **kw):
        out = real(logits, labels, reduction=reduction, **kw)
        seen.setdefault('ce', (logits.detach().clone(), labels.clone(), out.detach().clone()))
        return out
    monkeypatch.setattr(ts.F, 'cross_entropy', spy)
    cfg = tok(0, epochs=1)
    train(export, cfg, tmp_path, window_budget=1000)
    labels = np.array(FoldSequences(export.store, 0, 'train', L=32).labels())
    w = torch.tensor([len(labels) / (2 * (labels == c).sum()) for c in (0, 1)], dtype=torch.float32)
    logits, y, ce = seen['ce']
    history = list(csv.DictReader((tmp_path / cfg.name / 'history.csv').open(encoding='utf-8')))
    assert float(history[0]['loss']) == pytest.approx(float((ce * w[y]).mean()), rel=1e-5)       # one batch holds every binary
    assert w[0] != w[1] or (labels == 0).sum() == (labels == 1).sum()


# ---- no leak, reproducibility ------------------------------------------------------------------------------------------

def test_no_test_or_validation_binary_is_in_any_training_batch(export, tmp_path, monkeypatch):
    events = []
    real_get, real_forward = FoldSequences.__getitem__, ts.forward

    def get(self, i):
        item = real_get(self, i)
        events.append(('item', self.split, item['sample_id']))
        return item

    def forward(net, batch):
        events.append(('train' if net.training else 'eval', None, list(batch['sample_ids'])))
        return real_forward(net, batch)
    monkeypatch.setattr(FoldSequences, '__getitem__', get)
    monkeypatch.setattr(ts, 'forward', forward)
    train(export, tok(3, epochs=3, num_workers=2), tmp_path)
    manifest = export.manifests[3]
    ids = {name: {r['sample_id'] for r in getattr(manifest, name)} for name in ('train', 'val', 'test')}
    assert not (ids['train'] & ids['val']) and not (ids['train'] & ids['test']) and not (ids['val'] & ids['test'])
    batches = [e[2] for e in events if e[0] == 'train']
    in_training = {i for b in batches for i in b}
    assert batches and in_training <= ids['train'] and not (in_training & (ids['val'] | ids['test']))
    assert in_training == ids['train']                                                          # all of them, every epoch
    assert sum(len(b) for b in batches) == 3 * len(ids['train'])
    for kind, split, sid in (e for e in events if e[0] == 'item'):
        assert sid in ids[split], (split, sid)                                                  # a split only yields its own ids
    last_train = max(n for n, e in enumerate(events) if e[0] == 'train')
    test_items = [n for n, e in enumerate(events) if e[0] == 'item' and e[1] == 'test']
    assert test_items and min(test_items) > last_train                                          # the test set is read after training


def first_loss(export, cfg, out):
    train(export, replace(cfg, epochs=1), out)
    return list(csv.DictReader((out / cfg.name / 'history.csv').open(encoding='utf-8')))[0]['loss']


def test_the_first_epoch_loss_is_reproducible_per_seed(export, tmp_path):
    a, b = first_loss(export, tok(0), tmp_path / 'a'), first_loss(export, tok(0), tmp_path / 'b')
    c = first_loss(export, tok(0, seed=1), tmp_path / 'c')
    assert a == b and a != c
    assert first_loss(export, tok(0, num_workers=2), tmp_path / 'd') == a                        # threads do not change the draws


def test_same_seed_gives_the_same_run(export, tmp_path):
    for out in ('a', 'b'):
        train(export, tok(4, seed=2, epochs=2), tmp_path / out)
    name = tok(4, seed=2).name
    for f in ('history.csv', 'predictions.csv', 'predictions_auc.csv'):
        ra, rb = (tmp_path / o / name / f for o in ('a', 'b'))
        if f == 'history.csv':
            col = lambda p: [r['loss'] for r in csv.DictReader(p.open(encoding='utf-8'))]
            assert col(ra) == col(rb)
        else:
            assert ra.read_bytes() == rb.read_bytes()
    sa, sb = (torch.load(tmp_path / o / name / 'best_auc.pt', map_location='cpu') for o in ('a', 'b'))
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA device')
def test_evaluation_is_bitwise_repeatable_on_cuda(export):
    torch.manual_seed(0)
    val = FoldSequences(export.store, 0, 'val', L=32)
    net = BinaryTokenSequence(Model1TokenSequence(n_ids=val.n_ids, max_seq_len=32)).to('cuda')
    first, second = (ts.predict(net, val, torch.device('cuda'), 5, 0) for _ in range(2))
    assert first[0] == second[0] and np.array_equal(first[1], second[1]) and first[2] == second[2]
    assert not torch.are_deterministic_algorithms_enabled()                           # the flag is restored


def test_predict_restores_the_deterministic_flag(export):
    val = FoldSequences(export.store, 0, 'val', L=32)
    net = BinaryTokenSequence(Model1TokenSequence(n_ids=val.n_ids, max_seq_len=32))
    for before in (False, True):
        torch.use_deterministic_algorithms(before)
        try:
            ts.predict(net, val, torch.device('cpu'), 5, 0)
            assert torch.are_deterministic_algorithms_enabled() is before
        finally:
            torch.use_deterministic_algorithms(False)


def test_evaluation_is_deterministic_and_independent_of_the_batch_budget(export):
    torch.manual_seed(0)
    val = FoldSequences(export.store, 0, 'val', L=32)
    net = BinaryTokenSequence(Model1TokenSequence(n_ids=val.n_ids, max_seq_len=32))
    runs = [ts.predict(net, val, torch.device('cpu'), budget, 0) for budget in (1, 5, 1000, 5)]
    ids0, p0, sizes0 = runs[0]
    assert ids0 == val.sample_ids() and len(p0) == len(val)
    for ids, p, sizes in runs[1:]:
        assert ids == ids0 and np.allclose(p, p0, atol=1e-6) and sizes == sizes0
    assert np.array_equal(runs[1][1], runs[3][1])                                                # same budget: bit-identical
    assert [s[0] for s in sizes0] == [val.n_windows(i) for i in range(len(val))]


# ---- errors, smoke, command line ---------------------------------------------------------------------------------------

def test_the_export_must_match_the_run(export, tmp_path):
    with pytest.raises(ValueError, match='fold 4|folds'):
        ts.check_export(export.store, replace(tok(0), fold=0, protocol='kfold_v3'))
    meta = dict(export.store.meta)
    store = SimpleNamespace(meta={**meta, 'arch': 'x64'}, root='x', fold=export.store.fold)
    with pytest.raises(ValueError, match='x64'):
        ts.check_export(store, tok(0))
    ts.check_export(export.store, tok(0))
    manifest = export.store.fold(0)['manifest']
    store = SimpleNamespace(meta=meta, root='x', fold=lambda k: {'manifest': {**manifest, 'val_seed': 3}})
    with pytest.raises(ValueError, match='val_seed 3'):
        ts.check_export(store, tok(0))
    with pytest.raises(ValueError, match="model='tokenseq'"):
        ts.run(RunConfig(protocol='kfold', fold=0, model='gnn'), export.root, 'cpu', None, 6, tmp_path)


def test_export_manifest_is_the_cache_manifest_with_val_seed_zero(export):
    index = load_index()
    for fold in FOLDS:
        got = ts.export_manifest(export.store, fold)
        want = make_manifest(RunConfig(model='gnn', protocol='kfold', fold=fold, arch='x86', seed=1, val_seed=0), index)
        for name in ('train', 'val', 'test'):
            assert [r['sample_id'] for r in getattr(got, name)] == sorted(r['sample_id'] for r in getattr(want, name)) or \
                sorted(r['sample_id'] for r in getattr(got, name)) == sorted(r['sample_id'] for r in getattr(want, name))
        assert sorted(got.val_families) == sorted(want.val_families)
        assert {r['label'] for r in got.test} == {r['label'] for r in want.test}


def test_smoke_trains_on_n_binaries_for_two_epochs_in_a_temporary_directory(export, tmp_path, monkeypatch):
    RUNS = tmp_path / 'real_runs'                                                                # stands for the run directory
    RUNS.mkdir()
    monkeypatch.setattr(ts, 'RUNS', RUNS)
    before = sorted(p.name for p in RUNS.iterdir())
    printed = []
    cfg = tok(2, epochs=40)
    metrics = ts.run(cfg, export.root, 'cpu', 6, 6, None, printed.append)
    assert metrics['epochs_run'] == 2 and metrics['smoke'] == 6 and metrics['run'] == cfg.name
    assert sorted(p.name for p in RUNS.iterdir()) == before                                     # nothing under RUNS
    assert any('smoke 6' in line and 'train 6 val 6 test 6' in line for line in printed)
    proj = metrics['projection']
    assert proj['train']['smoke_windows'] > 0 and proj['train']['full_windows'] >= proj['train']['smoke_windows']
    assert proj['max_epochs'] == 40 and len(proj['epoch_secs']) == 2 and proj['train']['smoke_secs'] > 0 and proj['epoch_secs'][0] <= proj['epoch_secs'][1]
    assert any(line.startswith('projection ') for line in printed)
    assert any('s/epoch over 2 epochs' in line and 'peak GPU' in line and 'peak RAM' in line for line in printed)
    runs_dir = tmp_path / 'runs'
    with pytest.raises(ValueError, match='never writes under'):
        ts.run(cfg, export.root, 'cpu', 6, 6, RUNS, lambda *_: None)
    metrics_again = ts.run(cfg, export.root, 'cpu', 6, 6, runs_dir, lambda *_: None)             # an explicit directory is kept
    assert (runs_dir / cfg.name / 'metrics.json').is_file() and metrics_again['epochs_run'] == 2


def test_smoke_subset_takes_half_of_each_class(export):
    rows = export.manifests[0].train
    picks = ts.smoke_subset(rows, export.store, 5)
    labels = [export.store.binaries[i]['label'] for i in picks]
    assert len(picks) == 5 and sorted(labels) == [0, 0, 1, 1, 1] and len(set(picks)) == 5
    assert len(ts.smoke_subset(rows, export.store, 1000)) == len(rows)


def test_command_line():
    cfg, opts = ts.parse_config(['--protocol', 'kfold', '--arch', 'x86', '--fold', '3', '--seed', '2'])
    assert cfg == RunConfig(protocol='kfold', fold=3, arch='x86', seed=2, model='tokenseq') and cfg.name == 'kfold3_x86_tokenseq_s2'
    assert opts == {'export': 'ours_cap0', 'device': None, 'smoke': None, 'window_budget': ts.WINDOW_BUDGET}
    cfg, opts = ts.parse_config(['--fold', '0', '--seq-len', '256', '--max-windows', '32', '--export', 'x', '--device', 'cpu',
                                 '--smoke', '40', '--epochs', '5', '--patience', '3', '--window-budget', '64'])
    assert (cfg.seq_len, cfg.max_windows, cfg.epochs, cfg.patience) == (256, 32, 5, 3) and cfg.name == 'kfold0_x86_tokenseq-L256-w32_s0'
    assert opts == {'export': 'x', 'device': 'cpu', 'smoke': 40, 'window_budget': 64}
    for bad in (['--fold', '7'], ['--fold', '0', '--smoke', '1'], ['--fold', '0', '--seq-len', '0'], ['--seed', '0'],
                ['--fold', '0', '--window-budget', '0'], ['--fold', '0', '--arch', 'all']):
        with pytest.raises(SystemExit):
            ts.parse_config(bad)
    assert ts.resolve_export('ours_cap0') == ts.EXPORT / 'ours_cap0'



# ---- the registry entry and `python -m experiments.train --model tokenseq` -------------------------------------------------

def test_tokenseq_is_registered_and_owns_its_loop():
    assert MODELS['tokenseq'] == 's03a_transformer.token_sequence.spec:TokenSeqSpec'
    spec = spec_for(tok())
    assert type(spec) is TokenSeqSpec and spec is not spec_for(tok()) and spec.name == 'tokenseq' and spec.requires == ()
    spec.check(tok())
    assert spec.variant_tokens(tok()) == ('tokenseq', 'L32', 'w8')
    for refused in (spec.build, spec.loader, spec.forward):
        with pytest.raises(NotImplementedError, match='own loop'):
            refused(tok(), None)
    with pytest.raises(TypeError, match='export, device, smoke, window_budget'):
        spec.run(tok(), bogus=1)


def test_the_command_line_takes_the_flags_of_model_1_only_with_model_tokenseq():
    cfg, out, options = experiments_train.parse_args(['--model', 'tokenseq', '--protocol', 'kfold', '--fold', '2', '--seq-len', '32',
                                                      '--export', 'x', '--device', 'cpu', '--smoke', '8', '--window-budget', '6'])
    assert cfg == RunConfig(protocol='kfold', fold=2, model='tokenseq', seq_len=32) and out is None
    assert options == {'export': 'x', 'device': 'cpu', 'smoke': 8, 'window_budget': 6}
    assert experiments_train.parse_args(['--model', 'tokenseq', '--protocol', 'kfold', '--fold', '0'])[2] == {}
    assert experiments_train.parse_config(['--model', 'tokenseq', '--protocol', 'kfold', '--fold', '0', '--out', 'x']) == (
        RunConfig(protocol='kfold', fold=0, model='tokenseq'), 'x')
    for argv in (['--model', 'gnn', '--smoke', '8'], ['--export', 'x'], ['--window-budget', '4', '--protocol', 'kfold', '--fold', '0'],
                 ['--model', 'tokenseq', '--control', 'no_edges', '--protocol', 'kfold', '--fold', '0'],
                 ['--model', 'tokenseq'], ['--model', 'tokenseq', '--protocol', 'kfold', '--fold', '0', '--device', 'tpu']):
        with pytest.raises(SystemExit):
            experiments_train.parse_args(argv)


def runs_in(tmp_path, monkeypatch):
    """Point every module that names the run directory at `tmp_path`."""
    monkeypatch.setattr(experiments_train, 'RUNS', tmp_path)
    monkeypatch.setattr(spec_module, 'RUNS', tmp_path)
    monkeypatch.setattr(ts, 'RUNS', tmp_path)


def test_experiments_train_runs_model_1_through_the_registry(export, tmp_path, monkeypatch):
    """The same run as `ts.run`, asked for through `experiments.train.run`: the loop of the registry is not used, the
    cache is not read, and --out names the directory."""
    runs_in(tmp_path, monkeypatch)
    monkeypatch.setattr(experiments_train, 'check_requires', lambda *_: pytest.fail('the registry loop ran'))
    cfg = tok(1)
    metrics = experiments_train.run(cfg, export=str(export.root), device='cpu', window_budget=6)
    assert metrics['run'] == cfg.name and (tmp_path / cfg.name / 'metrics.json').is_file()
    direct = train(export, cfg, tmp_path / 'direct')
    assert json.dumps(direct['test']) == json.dumps(metrics['test']) and direct['best_epoch'] == metrics['best_epoch']   # json: NaN == NaN
    named = experiments_train.run(cfg, 'smoke_named', export=str(export.root), device='cpu', window_budget=6)
    assert named['run'] == cfg.name and not (tmp_path / '.stage-smoke_named').exists()
    assert {p.name for p in (tmp_path / 'smoke_named').iterdir()} == set(RUN_FILES)
    assert RunConfig.load(tmp_path / 'smoke_named' / 'config.json') == cfg
    with pytest.raises(ValueError, match='takes none of'):
        experiments_train.run(RunConfig(protocol='kfold', fold=0, model='gnn'), smoke=4)


def test_a_smoke_run_through_experiments_train_writes_nothing_under_the_run_directory(export, tmp_path, monkeypatch):
    runs_in(tmp_path, monkeypatch)
    metrics = experiments_train.run(tok(0, epochs=40), 'ignored', export=str(export.root), device='cpu', smoke=6, window_budget=6)
    assert metrics['smoke'] == 6 and metrics['epochs_run'] == 2 and list(tmp_path.iterdir()) == []


def test_the_thin_entry_parses_to_the_config_the_registry_entry_gets():
    cfg, opts = ts.parse_config(['--fold', '0', '--seq-len', '32', '--max-windows', '8', '--epochs', '2'])
    assert cfg == tok(0, num_workers=2) and opts['export'] == 'ours_cap0'
    assert experiments_train.parse_config(['--model', 'tokenseq', '--protocol', 'kfold', '--fold', '0', '--seq-len', '32',
                                           '--max-windows', '8', '--epochs', '2'])[0] == cfg


# ---- the gate stage on synthetic runs ----------------------------------------------------------------------------------

def fake_arm(runs, cfg_of, signal, seeds, manifest_of, rng_seed):
    for seed in seeds:
        rng = np.random.default_rng(1000 * rng_seed + seed)
        for fold in FOLDS:
            cfg, manifest = cfg_of(fold, seed), manifest_of(fold, seed)
            sets = []
            for _ in range(2):
                out = {}
                for split in ('val', 'test'):
                    rows = getattr(manifest, split)
                    y = np.array([r['label'] for r in rows])
                    out[split] = ([r['sample_id'] for r in rows],
                                  np.clip(0.5 + signal * (y - 0.5) + rng.normal(0, 0.15, len(rows)), 0.01, 0.99),
                                  [(3, 5, 5)] * len(rows))
                sets.append(out)
            f1, auc = sets
            run_io.write_run(runs / cfg.name, cfg, f1, auc, val_threshold_of(manifest.val, *f1['val'][:2]),
                             val_threshold_of(manifest.val, *auc['val'][:2]),
                             {'best_epoch': 1, 'epochs_run': 3, 'auc_best_epoch': 2, 'val_auc': 0.9, 'val_macro_f1': 0.8},
                             manifest=manifest)


def fake_baseline(runs, method, suffix, signal, manifest_of):
    rng = np.random.default_rng(7)
    for fold in FOLDS:
        manifest = manifest_of(fold, 0)
        d = runs / 'baselines' / f'kfold{fold}_x86_s0{suffix}'
        d.mkdir(parents=True)
        (d / 'metrics.json').write_text(json.dumps({'methods': {method: {'threshold_val': 0.5}}}), encoding='utf-8')
        with (d / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['sample_id', 'split', 'method', 'label', 'prob', 'pred', 'arch', 'family', 'fold'])
            for r in manifest.test:
                p = float(np.clip(0.5 + signal * (r['label'] - 0.5) + rng.normal(0, 0.15), 0.01, 0.99))
                w.writerow([r['sample_id'], 'test', method, r['label'], f'{p:.6f}', int(p >= 0.5), r['arch'], r['family'],
                            r['fold']])


def cells(text, header_cols):
    return [[c.strip() for c in l.split('|')[1:-1]] for l in text.splitlines() if l.startswith('| ') and l.count('|') == header_cols + 1]


def test_model1_gates_from_synthetic_runs(export, tiny, tmp_path, monkeypatch):  # noqa: F811
    runs = tmp_path / 'runs'
    (runs / 'baselines').mkdir(parents=True)
    cache_manifest = lambda cfg: (lambda f, s: make_manifest(cfg(f, s), load_index()))
    r7 = lambda f, s: RunConfig(protocol='kfold', fold=f, arch='x86', seed=s, model='gnn', mask_blocks=0.15)
    r2 = lambda f, s: replace(r7(f, s), control='no_edges')
    m1 = lambda f, s: RunConfig(protocol='kfold', fold=f, arch='x86', seed=s, model='tokenseq')
    export_manifest = lambda f, s: ts.export_manifest(export.store, f)
    fake_arm(runs, r7, 0.9, (0, 1, 2), cache_manifest(r7), 1)
    fake_arm(runs, r2, 0.5, (0, 1, 2), cache_manifest(r2), 2)
    fake_arm(runs, m1, 0.1, (0, 1, 2), export_manifest, 3)                                       # Model 1 much weaker than R7
    fake_baseline(runs, 'tfidf_sig_api_lr', '', 0.7, export_manifest)                            # header-imports LR left out
    monkeypatch.setattr(stats, 'RUNS', runs)
    monkeypatch.setattr(report, 'RUNS', runs)
    gate_tables.pooled.cache_clear()
    stage = gates.load_stage('model1')
    arms = dict(stage.arms)
    text = '\n'.join(gates.render(stage, 60))
    gate_tables.pooled.cache_clear()
    assert arms['Model 1 tokenseq'] in text
    pooled = {r[0]: r[2:] for r in cells(text, 5) if r[0] in arms and r[1] == arms[r[0]]}
    assert set(pooled) == set(arms)
    assert pooled['header-imports LR'][0].startswith('missing')
    for arm in ('TF-IDF', 'R7 typed', 'R2 no-edges', 'Model 1 tokenseq'):
        assert 0 <= float(pooled[arm][0]) <= 1
    paired = {r[0]: r for r in cells(text, 5) if r[1] == 'Model 1 tokenseq' and r[0] in arms}
    assert set(paired) == {'TF-IDF', 'header-imports LR', 'R7 typed', 'R2 no-edges'}
    assert paired['R7 typed'][-1] == 'does not beat' and 'seed' not in paired['R7 typed'][-1]            # 3 seeds in both arms
    assert paired['R2 no-edges'][-1] in ('beats', 'does not beat')
    assert paired['TF-IDF'][-1] == 'does not beat (seed 0 only: a direction, not the test)'            # the baseline has seed 0 only
    assert paired['header-imports LR'][-1] == 'missing'
    assert 'seeds [0, 1, 2]' in text and '### kfold_x86_tokenseq minus kfold_x86_gnn-mblk15' in text
