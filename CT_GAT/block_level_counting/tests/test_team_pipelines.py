"""The team-pipeline adapter (experiments/team_pipelines/common.py): fold partition, bundle lookup, run directories the
harness reads, and the sha256 pin on the teammates' code. CPU only. Reads the real v2 cache index and the Mendeley bundle."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from common.config import DEFAULT_CACHE, RUNS, RunConfig
from evaluation import gates, report, stats
from experiments.team_pipelines import build_a, common as tp, fit, train_a, train_b
from training.train import parse_config, val_threshold_of

FOLDS = range(5)
ROOT = Path(__file__).resolve().parents[1]
BASH = next((b for b in ('C:/Program Files/Git/bin/bash.exe', shutil.which('bash')) if b and Path(b).exists()), None)
needs_bash = pytest.mark.skipif(BASH is None, reason='needs bash')
needs_real_data = pytest.mark.skipif(not (DEFAULT_CACHE / 'index.csv').exists() or not tp.BUNDLE.exists(),
                                     reason='needs the v2 cache and the Mendeley bundle')
needs_baselines = pytest.mark.skipif(not (RUNS / 'baselines' / 'kfold0_x86_s0' / 'predictions.csv').exists(),
                                     reason='needs the tfidf kfold baselines')


@needs_real_data
def test_folds_partition_all_ids():
    ids = tp.all_ids()
    assert len(ids) == 1810 == len(set(ids)) and ids == sorted(ids)
    tests = [{r['sample_id'] for r in tp.fold_manifest(f, 0).test} for f in FOLDS]
    assert sum(map(len, tests)) == 1810
    assert set().union(*tests) == set(ids)
    for a in FOLDS:
        for b in range(a + 1, 5):
            assert not tests[a] & tests[b]


@needs_real_data
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_no_group_leaks_between_train_val_test(seed):
    for fold in FOLDS:
        m = tp.fold_manifest(fold, seed)
        groups = {name: {(r['family'] if r['label'] else r['identity_group']) for r in rows}
                  for name, rows in (('train', m.train), ('val', m.val), ('test', m.test))}
        assert groups['train'] and groups['val'] and groups['test']
        assert not groups['train'] & groups['val'], f'fold {fold}: a group is in train and val'
        assert not groups['train'] & groups['test'], f'fold {fold}: a group is in train and test'
        assert not groups['val'] & groups['test'], f'fold {fold}: a group is in val and test'
        ids = [r['sample_id'] for rows in (m.train, m.val, m.test) for r in rows]
        assert len(ids) == len(set(ids))
        assert all(r['arch'] == 'x86' for rows in (m.train, m.val, m.test) for r in rows)


@needs_real_data
def test_paths_for_resolves_every_id():
    for sample_id in tp.all_ids():
        cfg_path, asm_path = tp.paths_for(sample_id)
        assert cfg_path.name == f'{sample_id}.jsonl.gz' and cfg_path.is_file()
        assert asm_path.name == f'{sample_id}.asm.gz' and asm_path.is_file()


def test_paths_for_unknown_id_is_an_error():
    with pytest.raises(KeyError):
        tp.paths_for('0' * 64)


def fake_rows(manifest, rng, signal):
    """{'val': (ids, probs, sizes), 'test': ...} with probabilities that follow the label by `signal`."""
    out = {}
    for split in ('val', 'test'):
        rows = getattr(manifest, split)
        probs = np.clip(0.5 + signal * (np.array([r['label'] for r in rows]) - 0.5) + rng.normal(0, 0.2, len(rows)),
                        0.01, 0.99)
        out[split] = ([r['sample_id'] for r in rows], probs, [(10, 500, 700)] * len(rows))
    return out


def write_fake_arm(runs, arm='fake', seed=0):
    rng = np.random.default_rng(seed)
    dirs = []
    for fold in FOLDS:
        cfg = tp.team_config(fold, seed, arm)
        manifest = tp.fold_manifest(fold, seed)
        f1, auc = fake_rows(manifest, rng, 0.8), fake_rows(manifest, rng, 0.9)
        thr_f1 = val_threshold_of(manifest.val, *f1['val'][:2])
        thr_auc = val_threshold_of(manifest.val, *auc['val'][:2])
        tp.write_run(runs / cfg.name, cfg, f1, auc, thr_f1, thr_auc,
                     {'best_epoch': 3, 'epochs_run': 11, 'auc_best_epoch': 5, 'val_auc': 0.9, 'val_macro_f1': 0.8})
        dirs.append(runs / cfg.name)
    return dirs


@needs_real_data
@needs_baselines
def test_synthetic_run_is_read_by_stats_and_report(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    (runs / 'baselines').mkdir(parents=True)
    for fold in FOLDS:
        shutil.copytree(RUNS / 'baselines' / f'kfold{fold}_x86_s0', runs / 'baselines' / f'kfold{fold}_x86_s0')
    dirs = write_fake_arm(runs)
    for d, fold in zip(dirs, FOLDS):
        cfg = RunConfig.load(d / 'config.json')
        assert d.name == cfg.name == f'kfold{fold}_x86_gnn_s0_team-fake'
        assert cfg.group == 'kfold_x86_gnn_team-fake'
        for f in ('config.json', 'manifest.json', 'metrics.json', 'predictions.csv', 'predictions_auc.csv'):
            assert (d / f).is_file()
        header = (d / 'predictions.csv').read_text(encoding='utf-8').splitlines()[0]
        assert header == ('sample_id,split,label,prob,pred,arch,family,fold,n_segments_total,'
                          'n_blocks_kept,n_blocks_total')
        metrics = json.loads((d / 'metrics.json').read_text(encoding='utf-8'))
        for key in ('best_epoch', 'epochs_run', 'test', 'floors', 'family_recall'):
            assert key in metrics
        assert {'threshold_val', 'test', 'family_recall'} <= set(metrics['selection']['auc'])
        thr = metrics['selection']['auc']['threshold_val']
        for line in (d / 'predictions_auc.csv').read_text(encoding='utf-8').splitlines()[1:]:
            cells = line.split(',')
            assert int(cells[4]) == int(float(cells[3]) >= thr)

    monkeypatch.setattr(stats, 'RUNS', runs)
    monkeypatch.setattr(report, 'RUNS', runs)
    monkeypatch.setattr(report, 'HERE', tmp_path)
    for selection in ('f1', 'auc'):
        result, info = stats.compare('tfidf', 'kfold_x86_gnn_team-fake', selection, resamples=40)
        assert info['n'] == 1810 and info['seeds'] == [0] and not info['skipped']
        assert set(result) == {'auc', 'pr_auc', 'macro_f1'}
        assert np.isfinite(result['auc']['diff'])
    report.main()
    text = (tmp_path / 'RESULTS.md').read_text(encoding='utf-8')
    assert "team-fake" in text and 'edge-typed GNN' in text


@needs_real_data
def test_write_run_checks_the_ids_and_the_name(tmp_path):
    cfg = tp.team_config(0, 0, 'fake')
    manifest = tp.fold_manifest(0, 0)
    rows = fake_rows(manifest, np.random.default_rng(0), 0.8)
    extra = {'best_epoch': 1, 'epochs_run': 1}
    with pytest.raises(ValueError):
        tp.write_run(tmp_path / 'wrong_name', cfg, rows, rows, 0.5, 0.5, extra)
    short = {**rows, 'test': tuple(part[:-1] for part in rows['test'])}
    with pytest.raises(AssertionError):
        tp.write_run(tmp_path / cfg.name, cfg, short, rows, 0.5, 0.5, extra)
    with pytest.raises(KeyError):
        tp.write_run(tmp_path / cfg.name, cfg, rows, rows, 0.5, 0.5, {'best_epoch': 1})


@pytest.mark.skipif(not tp.TEAM_ROOT.exists(), reason='needs the review worktree')
def test_pin_check_passes_on_the_review_worktree_and_raises_on_a_tampered_copy(tmp_path):
    for module in tp.CLOSURE:
        tp.check_pins(module)
    copy = tmp_path / 'rdmp-team-gnn'
    for rel in tp.PINNED_SHA256:
        (copy / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tp.TEAM_ROOT / rel, copy / rel)
    for module in tp.CLOSURE:
        tp.check_pins(module, copy)
    victim = copy / 'CT_GAT/preprocessing/tokenize_asm_gz.py'
    victim.write_bytes(victim.read_bytes() + b'\n# tampered\n')
    with pytest.raises(RuntimeError, match='tokenize_asm_gz.py'):
        tp.import_team('preprocess_full', root=copy)
    with pytest.raises(RuntimeError, match='tokenize_asm_gz.py'):
        tp.check_pins('tokenize_asm_gz', copy)
    tp.check_pins('train_cfg', copy)  # an untouched module of the same copy still passes
    with pytest.raises(KeyError):
        tp.check_pins('not_a_module', copy)
    with pytest.raises(FileNotFoundError):
        tp.check_pins('construct_blocks', tmp_path / 'empty')


def test_pins_cover_every_file_of_every_closure():
    assert all(rel in tp.PINNED_SHA256 for files in tp.CLOSURE.values() for rel in files)
    assert all(len(sha) == 64 for sha in tp.PINNED_SHA256.values())


# ---- step 18: Pipeline A / B trainers -------------------------------------------------------------------------------
A_DIR, B_BUILD = train_a.GRAPH_DIR, train_b.B_DIR
REF_VEC = DEFAULT_CACHE / 'side' / 'blockvec' / 'ours_cap0_ref' / 'fold0' / 'meta.json'
needs_a_graphs = pytest.mark.skipif(not A_DIR.exists() or len(list(A_DIR.glob('*.pt'))) < 20,
                                    reason='needs a few build_a graphs')
needs_ref_vecs = pytest.mark.skipif(not REF_VEC.exists() or not tp.BUNDLE.exists(),
                                    reason='needs the reference block vectors ours_cap0_ref fold 0')
needs_b_build = pytest.mark.skipif(not (B_BUILD / 'checkpoint.json').exists(), reason='needs the build_b output')


def b_done():
    return sorted(json.loads((B_BUILD / 'checkpoint.json').read_text(encoding='utf-8'))['done'])


@pytest.fixture(scope='module', autouse=True)
def pinned_team_modules():
    """Other test modules import our own CT_GAT/preprocessing modules under the teammates' names (token_mapping,
    tokenize_asm_gz, ...). For this module those are set aside, the teammates' pinned ones are imported afresh, and
    everything (sys.modules, sys.path) is put back afterwards."""
    stems = {Path(f).stem for files in tp.CLOSURE.values() for f in files}
    saved_path, aside = list(sys.path), {s: sys.modules.pop(s) for s in stems if s in sys.modules}
    yield
    for s in stems:
        sys.modules.pop(s, None)
    sys.modules.update(aside)
    sys.path[:] = saved_path


@pytest.fixture(scope='module')
def pp(pinned_team_modules):
    return train_b.import_team('preprocess_full')


def test_remap_api_touches_only_ids_unseen_in_train(pp):
    tok = pp.tok
    ids = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, tok.UNK_API, 4], [5, 6, 7, tok.FIRST_API_ID, 0],
                        [5, 6, 7, tok.FIRST_API_ID + 1, 0], [5, 6, 7, tok.FIRST_API_ID + 9, 2]])
    seen = torch.tensor([tok.FIRST_API_ID, tok.FIRST_API_ID + 9])
    out = train_b.remap_api(tok, ids, seen)
    assert out[:, [0, 1, 2, 4]].equal(ids[:, [0, 1, 2, 4]])  # other channels untouched
    assert out[:, 3].tolist() == [0, tok.UNK_API, tok.FIRST_API_ID, tok.UNK_API, tok.FIRST_API_ID + 9]
    assert ids[3, 3] == tok.FIRST_API_ID + 1  # the input is not modified
    assert train_b.remap_api(tok, ids, torch.tensor([tok.FIRST_API_ID + 1]))[2, 3] == tok.UNK_API


@needs_real_data
@needs_b_build
def test_repool_reproduces_the_stored_embeddings_and_api_columns(pp):
    _, embedder, done = train_b.build_state()
    for sample_id in b_done()[:3]:
        payload = train_b.load_slim(B_BUILD, sample_id)
        assert train_b.check_repool(pp, embedder, payload) <= 1e-5
    bad = dict(payload, block_embeddings=payload['block_embeddings'] + 1e-3)
    with pytest.raises(AssertionError, match='re-pooled'):
        train_b.check_repool(pp, embedder, bad)
    features = payload['block_features'].clone()
    features[0, train_b.kind_slice(pp).start] += 1
    with pytest.raises(AssertionError, match='api-kind'):
        train_b.check_repool(pp, embedder, dict(payload, block_features=features))


@needs_real_data
@needs_b_build
def test_unk_swap_moves_known_to_unresolved_and_changes_nothing_else(pp):
    _, embedder, _ = train_b.build_state()
    gl = train_b.import_team('graph_full_loader')
    payload = train_b.load_slim(B_BUILD, b_done()[0])
    api = payload['instruction_ids'][:, 3]
    real = torch.unique(api[api >= pp.tok.FIRST_API_ID])
    assert len(real) > 1
    seen = real[: len(real) // 2]
    moved = int((api >= pp.tok.FIRST_API_ID).sum() - torch.isin(api, seen).sum())
    assert moved > 0
    base = train_b.make_graph(pp, gl, embedder, payload, None)
    everything_seen = train_b.make_graph(pp, gl, embedder, payload, real)
    assert torch.allclose(everything_seen.x, base.x, atol=1e-6)  # nothing unseen: the stored graph
    swapped = train_b.make_graph(pp, gl, embedder, payload, seen)
    assert swapped.edge_index.equal(base.edge_index) and swapped.edge_attr.equal(base.edge_attr)
    assert (swapped.x[:, :64] != base.x[:, :64]).any()  # the pooled embeddings moved
    stored = payload['block_features'][:, train_b.kind_slice(pp)].sum(0)
    new = train_b.block_counts(pp.tok, train_b.remap_api(pp.tok, payload['instruction_ids'], seen),
                               payload['instruction_matched'], payload['insn_to_block'],
                               payload['block_features'].shape[0]).sum(0)
    assert new[0] == stored[0] and new[1] == stored[1] + moved and new[2] == stored[2] - moved


@needs_real_data
@needs_b_build
def test_unswapped_graph_equals_graph_full_loader(pp):
    _, embedder, _ = train_b.build_state()
    gl = train_b.import_team('graph_full_loader')
    sample_id = b_done()[0]
    ours = train_b.make_graph(pp, gl, embedder, train_b.load_slim(B_BUILD, sample_id), None)
    theirs = gl.load_full_graph(B_BUILD / f'{sample_id}.pt')
    assert ours.x.shape[1] == 141 and ours.edge_attr.shape[1] == 8
    assert torch.equal(ours.x, theirs.x) and torch.equal(ours.edge_attr, theirs.edge_attr)
    assert torch.equal(ours.edge_index, theirs.edge_index) and ours.y == theirs.y and ours.sample_id == sample_id


def edge_set_of(out):
    """(local source, local target) pairs of an encode_binary_gnn item, local = index among the kept blocks."""
    first = np.cumsum(out['seg_len']) - out['seg_len']
    return set(zip((first[out['edge_seg']] + out['edge_src']).tolist(),
                   (first[out['edge_seg']] + out['edge_dst']).tolist()))


@needs_real_data
@needs_ref_vecs
def test_cb500_blocks_and_edges_equal_the_matched_arm():
    from common.manifests import load_sample
    from pipeline.batching import encode_binary_gnn
    from training.train import open_block_vecs
    rows = [r for r in tp.sample_rows().values() if r['fold'] is not None]
    by_size = sorted(rows, key=lambda r: int(r['num_blocks']))
    picks = [by_size[100], by_size[len(by_size) // 2], by_size[-300]]
    assert int(picks[0]['num_blocks']) < 500 < int(picks[1]['num_blocks'])
    cfg = RunConfig(protocol='kfold', fold=0, arch='x86', model='gnn', block_input='vec', block_vec='ours_cap0_ref',
                    subgraph='cap', budget=0)
    object.__setattr__(cfg, 'budget', 500)  # RunConfig refuses 500 (< 1000); the matched arm's evaluation uses it
    vecs = open_block_vecs(cfg)
    for r in picks:
        sample_id = r['sample_id']
        out = encode_binary_gnn(load_sample(sample_id), None, cfg, None, sample_id, vecs)
        graph, (n_fn, kept, total) = train_a.cache_graph(sample_id, vecs)
        assert kept == out['n_blocks_kept'] == graph.x.shape[0] <= 500 and total == out['n_blocks_total']
        assert n_fn == out['n_fn']
        assert np.array_equal(graph.x.numpy(), out['blk_vec'])  # same blocks in the same order, every key found
        pairs = set(zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist()))
        assert pairs == edge_set_of(out) and len(pairs) > 0
        assert graph.y.item() == r['label'] and graph.x.dtype == torch.float16


def synthetic_graphs(ids, rng, d=4):
    from torch_geometric.data import Data
    graphs, sizes = {}, {}
    for i in ids:
        n = int(rng.integers(3, 8))
        label = tp.sample_rows()[i]['label']
        x = torch.tensor(rng.normal(label * 0.8, 1.0, (n, d)), dtype=torch.float32)
        graphs[i] = Data(x=x, edge_index=torch.tensor(np.stack([np.arange(n - 1), np.arange(1, n)])),
                         y=torch.tensor([label]))
        graphs[i].sample_id = i
        sizes[i] = (2, n, n + 5)
    return graphs, sizes


@needs_real_data
def test_run_arm_writes_a_run_the_harness_reads(tmp_path, monkeypatch):
    from torch import nn
    from torch_geometric.nn import global_mean_pool

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 2)

        def forward(self, data):
            return self.lin(global_mean_pool(data.x, data.batch))

    monkeypatch.setattr(fit, 'RUNS', tmp_path)
    graphs, sizes = synthetic_graphs(tp.all_ids(), np.random.default_rng(0))
    hp = {**train_a.HP, 'epochs': 3}
    threads = torch.get_num_threads()
    monkeypatch.setenv('TEAM_THREADS', '2')
    try:
        out = fit.run_arm(arm='fake18', fold=2, seed=1, graphs=graphs, sizes=sizes, build_model=Tiny, hp=hp,
                          device=torch.device('cpu'), extra={'in_dim': 4}, log=lambda *_: None)
    finally:
        torch.set_num_threads(threads)
    d = tmp_path / 'kfold2_x86_gnn_s1_team-fake18'
    assert out['dir'] == d
    for f in ('config.json', 'manifest.json', 'metrics.json', 'predictions.csv', 'predictions_auc.csv', 'history.csv',
              'best.pt', 'best_auc.pt'):
        assert (d / f).is_file()
    metrics = json.loads((d / 'metrics.json').read_text(encoding='utf-8'))
    assert metrics['epochs_run'] <= 3 and metrics['in_dim'] == 4 and metrics['n_blocks_total']['test_sum'] > 0
    assert metrics['device'] == 'cpu' and metrics['torch_threads'] == 2 and metrics['recipe']['device'] == 'cpu'
    assert metrics['test']['auc'] > 0.5 and 'threshold_val' in metrics['selection']['auc']
    assert len((d / 'history.csv').read_text(encoding='utf-8').splitlines()) == metrics['epochs_run'] + 1
    del graphs[tp.all_ids()[5]]
    with pytest.raises(SystemExit, match='no graph'):
        fit.run_arm(arm='fake18', fold=2, seed=1, graphs=graphs, sizes=sizes, build_model=Tiny, hp=hp,
                    device=torch.device('cpu'), log=lambda *_: None)


def assert_smoke_clean(out, fake_runs):
    assert Path(out['dir']).parent == Path(tempfile.gettempdir())
    assert RUNS not in Path(out['dir']).parents and not any(fake_runs.iterdir())
    assert len(out['history']) == 2 and (Path(out['dir']) / 'history.csv').is_file()
    assert out['device'] == 'cpu' and out['peak_gpu_mb'] == 0 and out['epoch_secs_mean'] > 0


@pytest.fixture
def no_write_run(tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError('a smoke run must not call write_run')
    monkeypatch.setattr(fit, 'write_run', refuse)
    monkeypatch.setattr(fit, 'RUNS', tmp_path)
    return tmp_path


@needs_real_data
@needs_a_graphs
def test_train_a_gnn500_smoke_creates_no_run_dir(no_write_run):
    out = train_a.run('gnn500', 0, 0, smoke=12, device='cpu', log=lambda *_: None)
    assert_smoke_clean(out, no_write_run)
    shutil.rmtree(out['dir'])


@needs_real_data
@needs_ref_vecs
def test_train_a_cb500_smoke_creates_no_run_dir(no_write_run):
    out = train_a.run('cb500', 0, 0, smoke=12, block_vec='ours_cap0_ref', device='cpu', log=lambda *_: None)
    assert_smoke_clean(out, no_write_run)
    shutil.rmtree(out['dir'])


@needs_real_data
@needs_b_build
def test_train_b_smoke_creates_no_run_dir(no_write_run):
    out = train_b.run('sig500', 0, 0, smoke=12, device='cpu', log=lambda *_: None)
    assert_smoke_clean(out, no_write_run)
    shutil.rmtree(out['dir'])


@needs_real_data
def test_build_a_builds_resumes_and_stops_on_a_bad_id(tmp_path):
    small = min(tp.sample_rows().values(), key=lambda r: int(r['num_blocks']))['sample_id']
    out = tmp_path / 'a_gnn500'
    summary = build_a.main(['--budget', '500', '--out', str(out), '--ids', small])
    assert summary['processed'] == 1 and summary['built_this_run'] == 1
    graph = torch.load(out / f'{small}.pt', weights_only=False)
    assert graph.sample_id == small and graph.label == graph.y.item() == tp.sample_rows()[small]['label']
    assert graph.x.shape[1] == 57 and graph.x.shape[0] <= 500
    log = (out / 'build_log.csv').read_text(encoding='utf-8').splitlines()
    assert log[0] == 'sample_id,status,n_blocks_total,n_blocks_kept,seconds' and log[1].startswith(small + ',ok,')
    assert build_a.main(['--budget', '500', '--out', str(out), '--ids', small])['built_this_run'] == 0
    with pytest.raises(SystemExit):  # a different budget on an existing out dir
        build_a.main(['--budget', '128', '--out', str(out), '--ids', small])
    with pytest.raises(SystemExit):  # an id that is not a kfold x86 binary
        build_a.main(['--out', str(out), '--ids', '0' * 64])


# the run scripts and the team gates stage ---------------------------------------------------------------------------


def script(name, *args):
    r = subprocess.run([BASH, f'scripts/{name}', *args], cwd=ROOT, capture_output=True, text=True, env=os.environ)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


@needs_bash
@needs_real_data
def test_team_runs_dry_names_equal_run_config_names():
    lines = script('team_runs.sh', '--dry')
    runs = [l.split() for l in lines if l.startswith('run ')]
    assert len(runs) == 45 and lines[0].startswith('team_runs start') and lines[-1].startswith('team_runs done')
    waits = [l for l in lines if l.startswith('wait ')]
    assert len(waits) == 1 and waits[0].endswith('/team/b_sig500/build_summary.json reports processed + excluded == 1810/1810')
    seen = []
    for _, name, module, *argv in runs:
        opt = dict(zip(argv[::2], argv[1::2]))
        arm, fold, seed = opt['--arm'], int(opt['--fold']), int(opt['--seed'])
        assert name == tp.team_config(fold, seed, arm).name, name
        assert module == ('experiments.team_pipelines.train_b' if arm == 'sig500' else 'experiments.team_pipelines.train_a')
        assert opt['--device'] == 'cpu'                                    # every CPU run says so, none can fall to the GPU
        assert ('--block-vec' in opt) == (arm == 'cb500')
        assert opt.get('--block-vec', 'ours_cap0_codebert') == 'ours_cap0_codebert'
        assert arm in (train_b if arm == 'sig500' else train_a).ARMS
        seen.append((arm, seed, fold))
    # lane order gnn500, cb500, sig500, each seed by seed over folds 0-4: 3 arms x 3 seeds x 5 folds, no repeats
    assert seen == [(a, s_, f) for a in ('gnn500', 'cb500', 'sig500') for s_ in (0, 1, 2) for f in range(5)]


@needs_bash
@needs_real_data
def test_team_gpu_runs_dry_names_equal_run_config_names():
    lines = script('team_gpu_runs.sh', '--dry')
    assert lines[0].startswith("wait for 'master4 done' in ") and lines[0].endswith('logs/master4.log')
    runs = [l.split()[1:] for l in lines if l.startswith('run ')]
    assert len(runs) == 30
    seen = []
    for name, *argv in runs:
        cfg, _ = parse_config(argv)
        assert cfg.name == name, (name, cfg.name)
        assert (cfg.protocol, cfg.arch, cfg.model, cfg.mask_blocks, cfg.subgraph, cfg.budget, cfg.tag) == (
            'kfold', 'x86', 'gnn', 0.15, 'cap', 500, 'b500')
        assert re.fullmatch(r'kfold\d_x86_gnn(-noedges)?-mblk15-sg-cap500_s\d_b500', name)
        seen.append((cfg.control, cfg.seed, cfg.fold))
    assert seen == [(c, s_, f) for c in ('', 'no_edges') for s_ in (0, 1, 2) for f in range(5)]
    assert lines[-5:-1] == ['queue bash scripts/queue_gnn.sh --stage paths_codebert', 'cpu report.py',
                            'cpu gates.py --stage team > RUNS/gates_team.md',
                            'cpu gates.py --stage paths_codebert > RUNS/gates_paths_codebert.md']
    groups = {parse_config(a)[0].group for _, *a in runs}
    assert groups == {dict(gates.TEAM_ARMS)['ours b500 typed'], dict(gates.TEAM_ARMS)['ours b500 no-edges']}


def test_team_scripts_use_separate_stop_files_and_cpu_only():
    cpu = (ROOT / 'scripts' / 'team_runs.sh').read_text(encoding='utf-8')
    gpu = (ROOT / 'scripts' / 'team_gpu_runs.sh').read_text(encoding='utf-8')
    assert 'STOP=$R/STOP_TEAM' in cpu and 'STOP=$R/STOP\n' in gpu
    assert 'CUDA_VISIBLE_DEVICES=-1 PYTHONUNBUFFERED=1' in cpu and '--device cpu' in cpu and 'train.py' not in cpu.replace('train_', '')
    assert 'master4 done' in gpu and 'stage paths_codebert' in gpu


def test_team_groups_are_the_names_the_trainers_write():
    assert gates.team_group('cb500') == tp.team_config(3, 1, 'cb500').group == 'kfold_x86_gnn_team-cb500'
    arms = dict(gates.TEAM_ARMS)
    assert arms['ours b500 typed'] == 'kfold_x86_gnn-mblk15-sg-cap500_b500'
    assert arms['ours b500 no-edges'] == 'kfold_x86_gnn-noedges-mblk15-sg-cap500_b500'
    assert arms['R7 typed, whole graph'] == 'kfold_x86_gnn-mblk15'
    assert arms['R2 no-edges, whole graph'] == 'kfold_x86_gnn-noedges-mblk15'
    assert arms['path 2: typed, CodeBERT vectors'] == 'kfold_x86_gnn-mblk15-bv-ours_cap0_codebert'
    assert arms['header-imports LR'] == 'imports-header' and arms['TF-IDF'] == 'tfidf'
    pairs = gates.TEAM_PAIRS
    for ref in ('TF-IDF', 'R7 typed, whole graph', 'ours b500 typed'):
        assert [b for a, b in pairs if a == ref and b.startswith('team ')] == ['team gnn500', 'team cb500', 'team sig500']
    assert ('path 2: typed, CodeBERT vectors', 'team cb500') in pairs and gates.TEAM_COST in pairs and len(pairs) == 11


def test_team_gates_print_missing_arms(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    gates.pooled.cache_clear()
    text = '\n'.join(gates.team(50))
    gates.pooled.cache_clear()
    assert text.startswith('# Gate summary, stage team')
    for arm, group in gates.TEAM_ARMS:
        assert f'| {arm} | {group} | missing' in text and f'| {arm} | missing' in text
    for a, b in gates.TEAM_PAIRS:
        assert f'| {a} | {b} | missing | missing | missing |' in text
    for word in ('accuracy', 'precision', 'recall', 'TP', 'FN', 'FP', 'TN', 'Pre-registered rule', 'non-inferiority'):
        assert word in text


def fake_group(runs, cfg_of, signal, seeds=(0, 1, 2)):
    """Synthetic five-fold runs for every seed: cfg_of(fold, seed) -> RunConfig; the probabilities follow the label by
    `signal` (write_run's checks apply: the ids must be the manifest's)."""
    for seed in seeds:
        rng = np.random.default_rng(100 * seed + int(10 * signal))
        for fold in FOLDS:
            cfg, manifest = cfg_of(fold, seed), tp.fold_manifest(fold, seed)
            f1, auc = fake_rows(manifest, rng, signal), fake_rows(manifest, rng, signal)
            tp.write_run(runs / cfg.name, cfg, f1, auc, val_threshold_of(manifest.val, *f1['val'][:2]),
                         val_threshold_of(manifest.val, *auc['val'][:2]),
                         {'best_epoch': 2, 'epochs_run': 7, 'auc_best_epoch': 3, 'val_auc': 0.9, 'val_macro_f1': 0.8})


@needs_real_data
@needs_baselines
def test_team_gates_from_synthetic_runs(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    (runs / 'baselines').mkdir(parents=True)
    for fold in FOLDS:
        shutil.copytree(RUNS / 'baselines' / f'kfold{fold}_x86_s0', runs / 'baselines' / f'kfold{fold}_x86_s0')
    base = dict(protocol='kfold', arch='x86', model='gnn')
    fake_group(runs, lambda f, s: RunConfig(**base, fold=f, seed=s, mask_blocks=0.15), 0.5)          # R7, 3 seeds
    fake_group(runs, lambda f, s: RunConfig(**base, fold=f, seed=s, mask_blocks=0.15, subgraph='cap', budget=500,
                                            tag='b500'), 0.5)                                          # ours b500 typed
    fake_group(runs, lambda f, s: tp.team_config(f, s, 'gnn500'), 0.4)                                  # weak
    fake_group(runs, lambda f, s: tp.team_config(f, s, 'cb500'), 1.5)                                   # strong
    fake_group(runs, lambda f, s: tp.team_config(f, s, 'sig500'), 0.7, seeds=(0, 1))                    # seed 2 missing
    monkeypatch.setattr(stats, 'RUNS', runs)
    gates.pooled.cache_clear()
    text = '\n'.join(gates.team(60))
    gates.pooled.cache_clear()
    ops = {l.split('|')[1].strip(): [c.strip() for c in l.split('|')[2:-1]]
           for l in text.splitlines() if l.startswith('| ') and l.count('|') == 12 and l.split('|')[1].strip() in dict(gates.TEAM_ARMS)}
    assert len(ops) == len(gates.TEAM_ARMS)
    for arm in ('TF-IDF', 'R7 typed, whole graph', 'ours b500 typed', 'team gnn500', 'team cb500', 'team sig500'):
        auc, pr, f1, acc, prec, rec, tp_, fn, fp, tn = ops[arm]
        tp_, fn, fp, tn = int(tp_), int(fn), int(fp), int(tn)
        assert tp_ + fn == 1134 and fp + tn == 676                          # ransomware / goodware in the five test folds
        assert float(acc) == pytest.approx((tp_ + tn) / 1810, abs=6e-4)
        assert float(prec) == pytest.approx(tp_ / max(tp_ + fp, 1), abs=6e-4)
        assert float(rec) == pytest.approx(tp_ / 1134, abs=6e-4)
    assert ops['TF-IDF'][0] == '0.982'
    assert ops['ours b500 no-edges'][0].startswith('missing') and ops['team gnn500'][0] != ops['team cb500'][0]
    pair = {(l.split('|')[1].strip(), l.split('|')[2].strip()): l.split('|')[-2].strip()
            for l in text.splitlines() if l.startswith('| ') and l.count('|') == 6
            and (l.split('|')[1].strip(), l.split('|')[2].strip()) in gates.TEAM_PAIRS}
    assert len(pair) == len(gates.TEAM_PAIRS)
    assert pair[('R7 typed, whole graph', 'team cb500')] == 'beats'                                       # 3 seeds in both
    assert pair[('R7 typed, whole graph', 'team gnn500')] == 'does not beat'
    tf_cb = pair[('TF-IDF', 'team cb500')]
    assert tf_cb.startswith(('beats', 'does not beat')) and 'seeds [0] only: a direction, not the test' in tf_cb
    assert pair[gates.TEAM_COST] in ('non-inferior', 'not shown non-inferior')
    assert 'seeds [0, 1] only' in pair[('R7 typed, whole graph', 'team sig500')]                     # the seeds complete in both
    assert pair[('path 2: typed, CodeBERT vectors', 'team cb500')] == 'missing'
