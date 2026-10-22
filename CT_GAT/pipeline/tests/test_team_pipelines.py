"""python -m pytest tests/test_team_pipelines.py: the team-pipeline adapter (experiments/team_pipelines/common.py): fold
partition, bundle and collection lookup, run directories the harness reads, the sha256 pin on the teammates' code, and the Pipeline A / B
trainers.

What runs everywhere (CPU, no teammate code, no torch_geometric): the import guard of every module of the package, the
environment variables that locate the teammates' worktrees, the bundle and the IDA listings, the pin check and `import_team` on
a toy source, the fold partition and the run directories on the tiny cache (tests/tiny_cache.py), and the team stage of the gates
on synthetic runs. What skips with a reason when its input is absent: the pins of the real worktrees (GT_TEAM_GNN_ROOT,
GT_TEAM_MAIN_ROOT), the trainers (torch_geometric), the 1,810-binary checks, the bundle and the builds (marked `real_data`).
"""
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from common import manifests, paths
from common.config import DEFAULT_CACHE, RUNS, RunConfig
from experiments.team_pipelines import build_a, build_b, common as tp, fit, train_a, train_b
from experiments.train import val_threshold_of
from s04_evaluation import gate_tables, gates, report, stats
from tiny_cache import tiny  # noqa: F401  (a fixture)

FOLDS = range(5)
ROOT = Path(__file__).resolve().parents[1]
MODULES = ('build_a', 'build_b', 'common', 'compare_builds', 'fit', 'match_acceptance', 'match_diagnosis', 'match_rates',
           'measure_rss', 'train_a', 'train_b')


def _importable(name):
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def _unconfigured(*calls):
    """The message of the first call that raises TeamSourceNotConfigured, else None."""
    for call in calls:
        try:
            call()
        except tp.TeamSourceNotConfigured as e:
            return str(e)
    return None


NO_PYG = None if _importable('torch_geometric') else 'needs torch_geometric'
NO_WORKTREES = _unconfigured(lambda: tp.root_of('gnn-331ce0b'), lambda: tp.root_of('main-c7899f0'))
NO_BUNDLE = _unconfigured(tp.bundle_dir, tp.asm_dir)
needs_pyg = pytest.mark.skipif(NO_PYG is not None, reason=NO_PYG or '')
needs_worktrees = pytest.mark.skipif(NO_WORKTREES is not None, reason=f'needs the pinned teammate worktrees ({NO_WORKTREES})')
needs_bundle = pytest.mark.skipif(NO_BUNDLE is not None, reason=f'needs the Mendeley bundle and the IDA listings ({NO_BUNDLE})')
needs_baselines = pytest.mark.skipif(not (RUNS / 'baselines' / 'kfold0_x86_s0' / 'predictions.csv').exists(),
                                     reason='needs the tfidf kfold baselines')


def needs_real_data(fn):
    """The test reads the real cache of this machine (conftest skips `real_data` without it), through the caches of the
    adapter, and the bundle with the listings."""
    return pytest.mark.real_data(pytest.mark.usefixtures('real_cache')(needs_bundle(fn)))


def clear_team_caches():
    for cached in (tp._index, tp.sample_rows, tp.fold_manifest, tp.all_ids, tp._rows_of):
        cached.cache_clear()


@pytest.fixture(autouse=True)
def fresh_team_caches():
    """The adapter memoizes the index and the manifests of one cache; every test starts and ends without them."""
    clear_team_caches()
    yield
    clear_team_caches()


@pytest.fixture
def real_cache(monkeypatch):
    """The cache of this machine for a test that follows tests on the tiny cache (its module-scoped patch stays on)."""
    from common import config
    monkeypatch.setattr(manifests, 'CACHE', config.CACHE)


# ---- the import guard, the environment variables, the bundle lookup --------------------------------------------------------

def test_every_module_imports_without_torch_geometric_psutil_or_the_worktrees(tmp_path):
    fake = tmp_path / 'fake'
    (fake / 'torch_geometric').mkdir(parents=True)
    (fake / 'torch_geometric' / '__init__.py').write_text('raise ImportError("torch_geometric is not installed")', encoding='utf-8')
    (fake / 'psutil.py').write_text('raise ImportError("psutil is not installed")', encoding='utf-8')
    code = ('import importlib, sys\n'
            f'names = {list(MODULES)!r}\n'
            'for n in names:\n'
            '    importlib.import_module("experiments.team_pipelines." + n)\n'
            'assert "torch_geometric" not in sys.modules and "psutil" not in sys.modules\n'
            'from experiments.team_pipelines import common as tp\n'
            'try:\n'
            '    tp.root_of()\n'
            'except tp.TeamSourceNotConfigured as e:\n'
            '    assert "GT_TEAM_GNN_ROOT" in str(e), e\n'
            'else:\n'
            '    raise SystemExit("root_of() resolved without GT_TEAM_GNN_ROOT")\n'
            'print("ok", len(names))\n')
    env = {k: v for k, v in os.environ.items() if k not in ('GT_TEAM_GNN_ROOT', 'GT_TEAM_MAIN_ROOT', 'GT_TEAM_ASM_DIR', 'GT_BUNDLE')}
    env.update(GT_DATASETS=str(tmp_path / 'no-datasets.json'), PYTHONPATH=os.pathsep.join([str(fake), str(ROOT)]),
               CUDA_VISIBLE_DEVICES='-1')
    r = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == f'ok {len(MODULES)}'
    assert {p.stem for p in (ROOT / 'experiments' / 'team_pipelines').glob('*.py')} == set(MODULES) | {'__init__'}


def test_running_a_script_as_a_file_is_refused(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / 'experiments' / 'team_pipelines' / 'build_b.py'), '--help'], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode != 0 and 'python -m experiments.team_pipelines' in r.stderr


def test_a_source_root_comes_from_its_environment_variable_when_asked_for(tmp_path, monkeypatch):
    for source, var in (('gnn-331ce0b', 'GT_TEAM_GNN_ROOT'), ('main-c7899f0', 'GT_TEAM_MAIN_ROOT')):
        assert tp.SOURCES[source]['env'] == var
        monkeypatch.delenv(var, raising=False)
        with pytest.raises(tp.TeamSourceNotConfigured, match=var):
            tp.root_of(source)
        with pytest.raises(tp.TeamSourceNotConfigured, match=var):
            tp.check_pins('tokenize_asm_gz' if source == 'main-c7899f0' else 'construct_blocks', source=source)
        monkeypatch.setenv(var, str(tmp_path / 'absent'))
        with pytest.raises(tp.TeamSourceNotConfigured, match=var):
            tp.root_of(source)
        monkeypatch.setenv(var, str(tmp_path))
        assert tp.root_of(source) == tmp_path
    assert issubclass(tp.TeamSourceNotConfigured, FileNotFoundError)
    with pytest.raises(KeyError):
        tp.root_of('not-a-source')


def make_bundle(root, ids):
    """A bundle with a samples.csv row per id (cfg_path cfg/<id>.jsonl.gz) and the listings next to it."""
    (root / 'bundle' / 'cfg').mkdir(parents=True)
    (root / 'asm').mkdir()
    lines = ['sample_id,label,cfg_path,family']
    for i, sid in enumerate(ids):
        lines.append(f'{sid},{i % 2},cfg/{sid}.jsonl.gz,fam{i}')
        (root / 'bundle' / 'cfg' / f'{sid}.jsonl.gz').write_bytes(b'')
        (root / 'asm' / f'{sid}.asm.gz').write_bytes(b'')
    (root / 'bundle' / 'samples.csv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return root / 'bundle', root / 'asm'


def test_paths_for_reads_the_bundle_and_the_listing_directory_named_by_the_environment(tmp_path, monkeypatch):
    bundle, asm = make_bundle(tmp_path, ['a' * 64, 'b' * 64])
    monkeypatch.setenv('GT_DATASETS', str(tmp_path / 'no-datasets.json'))
    monkeypatch.setenv('GT_COLLECTIONS', str(tmp_path / 'no-collections'))
    monkeypatch.delenv('GT_BUNDLE', raising=False)
    monkeypatch.delenv('GT_TEAM_ASM_DIR', raising=False)
    with pytest.raises(tp.TeamSourceNotConfigured, match='GT_BUNDLE'):       # nothing configured, no collections either
        tp.paths_for('a' * 64)
    monkeypatch.setenv('GT_TEAM_ASM_DIR', str(asm))
    with pytest.raises(tp.TeamSourceNotConfigured, match='GT_BUNDLE'):       # the listings without the bundle: half configured
        tp.paths_for('a' * 64)
    monkeypatch.delenv('GT_TEAM_ASM_DIR')
    monkeypatch.setenv('GT_BUNDLE', str(bundle))
    with pytest.raises(tp.TeamSourceNotConfigured, match='GT_TEAM_ASM_DIR'):
        tp.paths_for('a' * 64)
    monkeypatch.setenv('GT_TEAM_ASM_DIR', str(asm))
    cfg_path, asm_path = tp.paths_for('a' * 64)
    assert cfg_path == bundle / 'cfg' / f'{"a" * 64}.jsonl.gz' and asm_path == asm / f'{"a" * 64}.asm.gz'
    row = tp.team_row('b' * 64)
    assert row['label'] == 1 and row['cfg_file'] == bundle / 'cfg' / f'{"b" * 64}.jsonl.gz' and row['family'] == 'fam1'
    with pytest.raises(KeyError):
        tp.paths_for('0' * 64)                                              # an id the bundle does not list
    (asm / f'{"a" * 64}.asm.gz').unlink()
    with pytest.raises(FileNotFoundError, match='missing'):
        tp.paths_for('a' * 64)


def make_collections(root, ids):
    """The two Mendeley collections under `root` (the `collections` source), a samples.csv row per id with its cfg_path and asm_path
    relative to its own collection, the files beside them. Ids alternate between the collections (ransomware, goodware)."""
    for k, name in enumerate(('Mendeley_Ransomware', 'Mendeley_Goodware')):
        lines = ['sample_id,label,family,cfg_path,asm_path']
        for sid in ids[k::2]:
            for kind in ('jsonl', 'asm'):
                path = root / name / 'samples' / 'train' / f'{sid}.{kind}.gz'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'')
            lines.append(f'{sid},{1 - k},fam{k},samples/train/{sid}.jsonl.gz,samples/train/{sid}.asm.gz')
        (root / name / 'samples.csv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return root


def zero_config(monkeypatch, tmp_path, collections):
    """No datasets.json, no bundle, no listing directory; the collections source is `collections`."""
    monkeypatch.setenv('GT_DATASETS', str(tmp_path / 'no-datasets.json'))
    monkeypatch.setenv('GT_COLLECTIONS', str(collections))
    monkeypatch.delenv('GT_BUNDLE', raising=False)
    monkeypatch.delenv('GT_TEAM_ASM_DIR', raising=False)


def test_paths_for_without_a_bundle_and_a_listing_directory_reads_the_collections(tmp_path, monkeypatch):
    ids = [c * 64 for c in 'abcd']
    collections = make_collections(tmp_path / 'data', ids)
    zero_config(monkeypatch, tmp_path, collections)
    for sid, name in zip(ids, ('Mendeley_Ransomware', 'Mendeley_Goodware') * 2):
        base = collections / name / 'samples' / 'train'
        assert tp.paths_for(sid) == (base / f'{sid}.jsonl.gz', base / f'{sid}.asm.gz')
    row = tp.team_row(ids[1])
    assert row['label'] == 0 and row['family'] == 'fam1' and 'root' not in row
    assert row['cfg_file'] == collections / 'Mendeley_Goodware' / 'samples' / 'train' / f'{ids[1]}.jsonl.gz'
    with pytest.raises(KeyError, match='no row'):
        tp.paths_for('0' * 64)
    (collections / 'Mendeley_Goodware' / 'samples' / 'train' / f'{ids[1]}.asm.gz').unlink()
    with pytest.raises(FileNotFoundError, match='missing'):
        tp.paths_for(ids[1])


def test_paths_for_names_the_collections_when_they_are_missing(tmp_path, monkeypatch):
    zero_config(monkeypatch, tmp_path, tmp_path / 'data')
    with pytest.raises(tp.TeamSourceNotConfigured, match='Mendeley_Ransomware'):
        tp.paths_for('a' * 64)


def test_a_configured_bundle_or_listing_directory_keeps_the_machines_behaviour_beside_collections(tmp_path, monkeypatch):
    ids = ['a' * 64, 'b' * 64]
    collections = make_collections(tmp_path / 'data', ids)
    bundle, asm = make_bundle(tmp_path, ids)
    zero_config(monkeypatch, tmp_path, collections)
    monkeypatch.setenv('GT_BUNDLE', str(bundle))
    with pytest.raises(tp.TeamSourceNotConfigured, match='GT_TEAM_ASM_DIR'):            # not the collections' listing
        tp.paths_for(ids[0])
    monkeypatch.setenv('GT_TEAM_ASM_DIR', str(asm))
    assert tp.paths_for(ids[0]) == (bundle / 'cfg' / f'{ids[0]}.jsonl.gz', asm / f'{ids[0]}.asm.gz')
    monkeypatch.delenv('GT_BUNDLE')
    with pytest.raises(tp.TeamSourceNotConfigured, match='GT_BUNDLE'):                  # the listings alone: still half configured
        tp.paths_for(ids[0])


class Stopped(Exception):
    """Raised by the stand-in for `build_b.load_state`, carrying what `main` had chosen by then."""


@pytest.fixture
def build_b_up_to_the_state(monkeypatch):
    """A function running `build_b.main(argv)` up to its first read of the build folder, returning (out, settings)."""
    monkeypatch.setattr(build_b, 'import_team', lambda *args, **kwargs: object())

    def stop(out, pp, settings):
        raise Stopped(out, settings)
    monkeypatch.setattr(build_b, 'load_state', stop)

    def run(argv):
        with pytest.raises(Stopped) as e:
            build_b.main(argv)
        return e.value.args
    return run


def test_build_b_writes_to_the_team_b_build_source_when_no_out_is_given(tiny, tmp_path, monkeypatch, build_b_up_to_the_state):  # noqa: F811
    monkeypatch.delenv('GT_TEAM_B_BUILD', raising=False)
    monkeypatch.setenv('GT_DATASETS', str(tmp_path / 'none.json'))
    monkeypatch.setattr(paths, 'PROCESSED', tmp_path / 'processed')                    # the catalog default: CT_GAT/processed/cfg_subgraphs
    out, settings = build_b_up_to_the_state(['--limit', '1'])
    assert out == tmp_path / 'processed' / 'cfg_subgraphs' and out.is_dir() and 'pins' not in settings
    monkeypatch.setenv('GT_TEAM_B_BUILD', str(tmp_path / 'elsewhere'))
    assert build_b_up_to_the_state(['--limit', '1'])[0] == tmp_path / 'elsewhere'
    out, settings = build_b_up_to_the_state(['--limit', '1', '--out', str(tmp_path / 'given')])
    assert out == tmp_path / 'given'
    assert build_b_up_to_the_state(['--source', 'repo', '--limit', '1'])[1]['pins'] == dict(tp.source_pins('repo'))


def test_build_b_stops_with_the_build_cache_command_when_the_cache_has_no_index(tmp_path, monkeypatch):
    monkeypatch.setattr(manifests, 'CACHE', tmp_path / 'no-cache')
    with pytest.raises(SystemExit, match=r'index.csv is missing.*build_cache --cache mendeley_x86'):
        build_b.main(['--out', str(tmp_path / 'b')])
    assert not (tmp_path / 'b').exists()


def test_build_b_does_not_touch_a_folder_of_sample_files_that_is_no_build(tiny, tmp_path, build_b_up_to_the_state):  # noqa: F811
    unzipped = tmp_path / 'cfg_subgraphs'
    unzipped.mkdir()
    sample = unzipped / ('a' * 64 + '.pt')
    torch.save({}, sample)
    torch.save({}, unzipped / 'channel_embedder.pt')
    with pytest.raises(SystemExit, match='not a build of this command'):
        build_b.main(['--out', str(unzipped), '--limit', '1'])
    assert sample.is_file()
    (unzipped / 'settings.json').write_text('{}', encoding='utf-8')                    # a build of this command: it goes on
    assert build_b_up_to_the_state(['--out', str(unzipped), '--limit', '1'])[0] == unzipped
    only_embedder = tmp_path / 'fresh'
    only_embedder.mkdir()
    torch.save({}, only_embedder / 'channel_embedder.pt')
    assert build_b_up_to_the_state(['--out', str(only_embedder), '--limit', '1'])[0] == only_embedder


def test_the_bundle_may_come_from_datasets_json(tmp_path, monkeypatch):
    bundle, _ = make_bundle(tmp_path, ['a' * 64])
    registry = tmp_path / 'datasets.json'
    registry.write_text(json.dumps({'x': {'cache': 'c', 'sources': {'bundle': str(bundle)}}}), encoding='utf-8')
    monkeypatch.setenv('GT_DATASETS', str(registry))
    monkeypatch.delenv('GT_BUNDLE', raising=False)
    assert tp.bundle_dir() == bundle


# ---- the folds -------------------------------------------------------------------------------------------------------------

def test_folds_partition_all_ids(tiny):  # noqa: F811
    ids = tp.all_ids()
    assert len(ids) == 40 == len(set(ids)) and ids == sorted(ids)
    tests = [{r['sample_id'] for r in tp.fold_manifest(f, 0).test} for f in FOLDS]
    assert sum(map(len, tests)) == 40
    assert set().union(*tests) == set(ids)
    for a in FOLDS:
        for b in range(a + 1, 5):
            assert not tests[a] & tests[b]


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_no_group_leaks_between_train_val_test(tiny, seed):  # noqa: F811
    for fold in FOLDS:
        m = tp.fold_manifest(fold, seed)
        groups = {name: {(r['family'] if r['label'] else r['identity_group']) for r in rows}
                  for name, rows in (('train', m.train), ('val', m.val), ('test', m.test))}
        assert groups['train'] and groups['val'] and groups['test']
        assert not groups['train'] & groups['val'], f'fold {fold}: a group is in train and val'
        assert not groups['train'] & groups['test'], f'fold {fold}: a group is in train and test'
        assert not groups['val'] & groups['test'], f'fold {fold}: a group is in val and test'
        ids = [r['sample_id'] for rows in (m.train, m.val, m.test) for r in rows]
        assert len(ids) == len(set(ids)) == 40
        assert all(r['arch'] == 'x86' for rows in (m.train, m.val, m.test) for r in rows)


def test_the_fold_manifest_is_the_one_of_the_kfold_gnn_runs(tiny):  # noqa: F811
    from common.manifests import load_index, make_manifest
    cfg = RunConfig(protocol='kfold', fold=3, arch='x86', seed=2, model='gnn', mask_blocks=0.15)
    assert tp.fold_config(3, 2) == cfg
    want = make_manifest(cfg, load_index())
    got = tp.fold_manifest(3, 2)
    assert [r['sample_id'] for r in got.val] == [r['sample_id'] for r in want.val]
    assert [r['sample_id'] for r in got.train] == [r['sample_id'] for r in want.train]
    assert set(tp.sample_rows()) == set(tiny.ids)


@pytest.mark.real_data
@pytest.mark.usefixtures('real_cache')
def test_the_real_folds_partition_the_1810_ids():
    ids = tp.all_ids()
    assert len(ids) == 1810 == len(set(ids)) and ids == sorted(ids)
    tests = [{r['sample_id'] for r in tp.fold_manifest(f, 0).test} for f in FOLDS]
    assert sum(map(len, tests)) == 1810 and set().union(*tests) == set(ids)


@needs_real_data
def test_paths_for_resolves_every_id():
    for sample_id in tp.all_ids():
        cfg_path, asm_path = tp.paths_for(sample_id)
        assert cfg_path.name == f'{sample_id}.jsonl.gz' and cfg_path.is_file()
        assert asm_path.name == f'{sample_id}.asm.gz' and asm_path.is_file()


# ---- run directories the harness reads -------------------------------------------------------------------------------------

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


def write_tfidf(runs, seed=0, signal=1.0):
    """The tfidf kfold baseline of the tiny cache's folds, in the layout stats.py reads (baselines/kfold<f>_x86_s<seed>)."""
    import csv
    rng = np.random.default_rng(11)
    for fold in FOLDS:
        d = runs / 'baselines' / f'kfold{fold}_x86_s{seed}'
        d.mkdir(parents=True)
        (d / 'metrics.json').write_text(json.dumps({'protocol': 'kfold', 'fold': fold, 'arch': 'x86', 'seed': seed,
                                                    'methods': {'tfidf_sig_api_lr': {'threshold_val': 0.4}},
                                                    'train_fraction': 1.0}), encoding='utf-8')
        with (d / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['sample_id', 'split', 'method', 'label', 'prob', 'pred', 'arch', 'family', 'fold'])
            for r in tp.fold_manifest(fold, seed).test:
                p = float(np.clip(0.5 + signal * (r['label'] - 0.5) + rng.normal(0, 0.2), 0.01, 0.99))
                w.writerow([r['sample_id'], 'test', 'tfidf_sig_api_lr', r['label'], f'{p:.6f}', int(p >= 0.5), 'x86',
                            r['family'], fold])


def test_synthetic_run_is_read_by_stats_and_report(tiny, tmp_path, monkeypatch):  # noqa: F811
    runs = tmp_path / 'runs'
    runs.mkdir()
    write_tfidf(runs)
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
    monkeypatch.setattr(report, 'all_labels', lambda: {r['sample_id']: r['label'] for r in tiny.rows})
    for selection in ('f1', 'auc'):
        result, info = stats.compare('tfidf', 'kfold_x86_gnn_team-fake', selection, resamples=40)
        assert info['n'] == 40 and info['seeds'] == [0] and not info['skipped']
        assert set(result) == {'auc', 'pr_auc', 'macro_f1'}
        assert np.isfinite(result['auc']['diff'])
    report.main(['--out', str(tmp_path / 'RESULTS.md')])
    text = (tmp_path / 'RESULTS.md').read_text(encoding='utf-8')
    assert 'team-fake' in text and 'edge-typed GNN' in text


def test_write_run_checks_the_ids_and_the_name(tiny, tmp_path):  # noqa: F811
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


def test_team_config_names_the_run_directory():
    cfg = tp.team_config(2, 1, 'sig500v2')
    assert cfg.name == 'kfold2_x86_gnn_s1_team-sig500v2' and cfg.group == 'kfold_x86_gnn_team-sig500v2'
    assert (cfg.protocol, cfg.arch, cfg.model, cfg.fold, cfg.seed) == ('kfold', 'x86', 'gnn', 2, 1)


# ---- the pins --------------------------------------------------------------------------------------------------------------

@needs_worktrees
@pytest.mark.parametrize('source', sorted(tp.SOURCES))
def test_pin_check_passes_on_the_review_worktree_and_raises_on_a_tampered_copy(tmp_path, source):
    src = tp.SOURCES[source]
    root = tp.root_of(source)
    for module in src['closure']:
        tp.check_pins(module, source=source)
    copy = tmp_path / root.name
    for rel in src['pins']:
        (copy / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, copy / rel)
    for module in src['closure']:
        tp.check_pins(module, copy, source)
    victim = next(copy / rel for rel in src['pins'] if rel.endswith('tokenize_asm_gz.py'))
    victim.write_bytes(victim.read_bytes() + b'\n# tampered\n')
    with pytest.raises(RuntimeError, match='tokenize_asm_gz.py'):
        tp.import_team('preprocess_full', root=copy, source=source)
    with pytest.raises(RuntimeError, match='tokenize_asm_gz.py'):
        tp.check_pins('tokenize_asm_gz', copy, source)
    with pytest.raises(KeyError):
        tp.check_pins('not_a_module', copy, source)
    with pytest.raises(FileNotFoundError):
        tp.check_pins('preprocess_full', tmp_path / 'empty', source)
    if source == 'gnn-331ce0b':
        tp.check_pins('train_cfg', copy)  # an untouched module of the same copy still passes
        with pytest.raises(FileNotFoundError):
            tp.check_pins('construct_blocks', tmp_path / 'empty')
    else:
        with pytest.raises(KeyError):      # main has no graph loader: it comes from the gnn source
            tp.check_pins('graph_full_loader', copy, source)
    with pytest.raises(KeyError):
        tp.import_team('preprocess_full', source='not-a-source')


def test_pins_cover_every_file_of_every_closure():
    for src in tp.SOURCES.values():
        assert all(rel in src['pins'] for files in src['closure'].values() for rel in files)
        assert all(len(sha) == 64 and set(sha) <= set('0123456789abcdef') for sha in src['pins'].values())
        assert set(src['pins']) == {rel for files in src['closure'].values() for rel in files}
    assert tp.PINNED_SHA256 is tp.SOURCES[tp.DEFAULT_SOURCE]['pins'] and tp.DEFAULT_SOURCE == 'gnn-331ce0b'
    assert set(tp.source_pins('main-c7899f0')) == {tp.STAGE + 'preprocess_full.py', tp.STAGE + 'tokenize_asm_gz.py'}
    assert (tp.TEAM_COMMIT, tp.MAIN_COMMIT) == ('331ce0b', 'c7899f0')
    assert tp.SOURCES['gnn-331ce0b']['commit'] == '331ce0b' and tp.SOURCES['main-c7899f0']['commit'] == 'c7899f0'


def repo_copy(tmp_path, monkeypatch):
    """The two files of the `repo` source copied under a temp root that the source is then pointed at."""
    root = tmp_path / 'repo_copy'
    for rel in tp.MAIN_CLOSURE['preprocess_full']:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tp.REPO_ROOT / rel, root / rel)
    monkeypatch.setitem(tp.SOURCES, 'repo', {**tp.SOURCES['repo'], 'root': root, 'pins': tp.FilePins(root, tp.MAIN_PINNED_SHA256)})
    return root


def test_the_repo_source_pins_the_files_of_the_repository_as_they_are(tmp_path, monkeypatch):
    rels = {tp.STAGE + 'preprocess_full.py', tp.STAGE + 'tokenize_asm_gz.py'}
    assert tp.SOURCES['repo']['env'] is None and tp.root_of('repo') == tp.REPO_ROOT == ROOT.parents[1]
    here = tp.source_pins('repo')
    assert set(here) == rels and all(here[rel] == hashlib.sha256((tp.REPO_ROOT / rel).read_bytes()).hexdigest() for rel in rels)
    assert tp.source_pins('repo', 'tokenize_asm_gz') == {tp.STAGE + 'tokenize_asm_gz.py': here[tp.STAGE + 'tokenize_asm_gz.py']}
    tp.check_pins('preprocess_full', source='repo')
    copy = repo_copy(tmp_path, monkeypatch)
    assert tp.root_of('repo') == copy and tp.source_pins('repo') == here                 # a copy of the files has the same pins
    victim = copy / tp.STAGE / 'tokenize_asm_gz.py'
    victim.write_bytes(victim.read_bytes() + b'\n# edited\n')
    changed = tp.source_pins('repo')                                                      # read when asked, not at import
    assert changed[tp.STAGE + 'tokenize_asm_gz.py'] != here[tp.STAGE + 'tokenize_asm_gz.py']
    assert changed[tp.STAGE + 'preprocess_full.py'] == here[tp.STAGE + 'preprocess_full.py']
    tp.check_pins('preprocess_full', source='repo')                                       # the pins follow the files of the root
    other = tmp_path / 'other'
    shutil.copytree(copy, other)
    (other / tp.STAGE / 'preprocess_full.py').write_bytes(b'x = 1\n')
    with pytest.raises(RuntimeError, match='preprocess_full.py.*the files of this repository'):
        tp.check_pins('preprocess_full', other, 'repo')                                   # files that are not the repo's
    with pytest.raises(KeyError):
        tp.check_pins('graph_full_loader', source='repo')


def test_a_resume_refuses_a_repo_file_that_changed_since_the_checkpoint(tmp_path, monkeypatch):
    copy = repo_copy(tmp_path, monkeypatch)
    out = tmp_path / 'build'
    out.mkdir()
    settings = {'sample': 'signal', 'num_blocks': 500, 'seed': 0, 'source': 'repo', 'pins': dict(tp.source_pins('repo'))}
    (out / 'checkpoint.json').write_text(json.dumps({'settings': settings, 'done': ['x'], 'api_names': 2}), encoding='utf-8')
    monkeypatch.setattr(build_b, 'load_vocab_embedder', lambda directory, pp: (type('V', (), {'names': [0, 1]})(), 'embedder'))
    _, embedder, done = build_b.load_state(out, None, dict(settings))
    assert (embedder, done) == ('embedder', {'x'})                                        # the same files: resumes
    victim = copy / tp.STAGE / 'preprocess_full.py'
    victim.write_bytes(victim.read_bytes() + b'\n# edited\n')
    with pytest.raises(SystemExit, match=r'preprocess_full.py.*changed since the checkpoint'):
        build_b.load_state(out, None, {**settings, 'pins': dict(tp.source_pins('repo'))})
    with pytest.raises(SystemExit, match='was built with'):                               # another source on a repo build
        build_b.load_state(out, None, {k: v for k, v in settings.items() if k != 'pins'} | {'source': 'gnn-331ce0b'})


def test_import_team_of_the_repo_source_imports_its_files_and_keeps_our_own_modules(monkeypatch):
    stage = (tp.REPO_ROOT / tp.STAGE).resolve()
    ours = type(sys)('token_mapping')
    ours.__file__ = str(stage / 'token_mapping.py')                                       # a module of ours inside the same tree
    monkeypatch.setitem(sys.modules, 'token_mapping', ours)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    mod = tp.import_team('preprocess_full', source='repo')
    assert Path(mod.__file__).resolve() == stage / 'preprocess_full.py'
    assert sys.modules['token_mapping'] is ours
    assert Path(sys.modules['tokenize_asm_gz'].__file__).resolve() == stage / 'tokenize_asm_gz.py'
    assert tp.import_team('preprocess_full', source='repo') is mod                         # the same pinned file twice: kept


@pytest.fixture
def toy_source(tmp_path, monkeypatch):
    """A pinned source of two tiny modules under a temp root (toy_b imports toy_a), registered in tp.SOURCES; sys.path and
    sys.modules are put back afterwards."""
    root = tmp_path / 'toy_root'
    root.mkdir()
    files = {'toy_a.py': 'VALUE = 1\n', 'toy_b.py': 'import toy_a\nTOTAL = toy_a.VALUE + 1\n'}
    for name, text in files.items():
        (root / name).write_bytes(text.encode())
    pins = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()}
    monkeypatch.setitem(tp.SOURCES, 'toy', {'env': 'GT_TEAM_TOY_ROOT', 'commit': '0000000', 'pins': pins, 'dirs': ('.',),
                                            'closure': {'toy_a': ('toy_a.py',), 'toy_b': ('toy_b.py', 'toy_a.py')}})
    monkeypatch.setenv('GT_TEAM_TOY_ROOT', str(root))
    saved_path = list(sys.path)
    yield root
    sys.path[:] = saved_path
    for name in ('toy_a', 'toy_b'):
        sys.modules.pop(name, None)


def test_pins_and_import_team_on_a_toy_source(toy_source):
    assert tp.root_of('toy') == toy_source
    assert tp.source_pins('toy', 'toy_b') == {'toy_b.py': tp.SOURCES['toy']['pins']['toy_b.py'],
                                              'toy_a.py': tp.SOURCES['toy']['pins']['toy_a.py']}
    tp.check_pins('toy_b', source='toy')
    wrote = set(p.name for p in toy_source.iterdir())
    mod = tp.import_team('toy_b', source='toy')
    assert mod.TOTAL == 2 and Path(mod.__file__).resolve() == (toy_source / 'toy_b.py').resolve()
    assert set(p.name for p in toy_source.iterdir()) == wrote                       # nothing written into the worktree
    assert tp.import_team('toy_b', source='toy') is mod                              # same pinned file twice: kept
    (toy_source / 'toy_a.py').write_text('VALUE = 5\n', encoding='utf-8')
    with pytest.raises(RuntimeError, match='toy_a.py'):
        tp.check_pins('toy_b', source='toy')
    with pytest.raises(RuntimeError, match='toy_a.py'):
        tp.import_team('toy_a', source='toy')
    (toy_source / 'toy_a.py').unlink()
    with pytest.raises(FileNotFoundError, match='toy_a.py'):
        tp.check_pins('toy_a', source='toy')
    with pytest.raises(KeyError):
        tp.check_pins('toy_c', source='toy')


def under(root):
    """Loaded modules whose file lies under `root`."""
    root = Path(root).resolve()
    return sorted(n for n, m in list(sys.modules.items())
                  if getattr(m, '__file__', None) and root in Path(m.__file__).resolve().parents)


@needs_worktrees
def test_importing_one_source_never_leaves_the_other_in_sys_modules():
    team_root, main_root = tp.root_of('gnn-331ce0b'), tp.root_of('main-c7899f0')
    old = tp.import_team('preprocess_full', source='gnn-331ce0b')
    assert under(main_root) == []
    assert {'preprocess_full', 'tokenize_asm_gz', 'transformer_model', 'token_mapping'} <= set(under(team_root))
    new = tp.import_team('preprocess_full', source='main-c7899f0')
    assert new is not old and under(team_root) == []                     # nothing of the gnn source is left loaded
    assert under(main_root) == ['preprocess_full', 'tokenize_asm_gz']
    assert Path(new.tok.__file__).resolve() == (main_root / tp.STAGE / 'tokenize_asm_gz.py').resolve()
    assert Path(sys.path[0]) == main_root / 'CT_GAT/pipeline/s02_preprocess' and not any(
        Path(p).resolve() == (team_root / 'CT_GAT/preprocessing').resolve() for p in sys.path)
    # both define the same sizes, so a payload pooled with one source's embedder is valid for the other's pooling
    for name in ('FIRST_API_ID', 'NO_API', 'UNK_API', 'BEHAVIOR_VOCAB_SIZE'):
        assert getattr(new.tok, name) == getattr(old.tok, name)
    assert (new.OPCODE_VOCAB_SIZE, new.OPERAND_VOCAB_SIZE) == (old.OPCODE_VOCAB_SIZE, old.OPERAND_VOCAB_SIZE)
    assert new.HANDCRAFTED_FEATURE_NAMES == old.HANDCRAFTED_FEATURE_NAMES
    back = tp.import_team('preprocess_full')
    assert back is not new and under(main_root) == []
    assert Path(back.__file__).resolve() == (team_root / 'CT_GAT/preprocessing/preprocess_full.py').resolve()
    again = tp.import_team('preprocess_full')                                # same source twice: the loaded module is kept
    assert again is back
    gl = tp.import_team('graph_full_loader')                                  # a gnn-only module does not evict the main pair
    tp.import_team('preprocess_full', source='main-c7899f0')
    assert tp.import_team('graph_full_loader') is gl and under(team_root) == ['graph_full_loader']


# ---- step 18: Pipeline A / B trainers ---------------------------------------------------------------------------------------
A_DIR, B_BUILD = train_a.GRAPH_DIR, train_b.B_DIR
needs_a_graphs = pytest.mark.skipif(not A_DIR.exists() or len(list(A_DIR.glob('*.pt'))) < 20,
                                    reason='needs a few build_a graphs')
needs_b_build = pytest.mark.skipif(not (B_BUILD / 'checkpoint.json').exists(), reason='needs the build_b output')


def b_done():
    return sorted(json.loads((B_BUILD / 'checkpoint.json').read_text(encoding='utf-8'))['done'])


@pytest.fixture(scope='module', autouse=True)
def pinned_team_modules():
    """Other test modules import our own modules under the teammates' flat names (token_mapping, tokenize_asm_gz, ...). For
    this module those are set aside, the teammates' pinned ones are imported afresh, and everything (sys.modules, sys.path)
    is put back afterwards."""
    stems = {Path(f).stem for src in tp.SOURCES.values() for files in src['closure'].values() for f in files}
    saved_path, aside = list(sys.path), {s: sys.modules.pop(s) for s in stems if s in sys.modules}
    yield
    for s in stems:
        sys.modules.pop(s, None)
    sys.modules.update(aside)
    sys.path[:] = saved_path


@pytest.fixture(scope='module')
def pp(pinned_team_modules):
    return train_b.import_team('preprocess_full')


@needs_worktrees
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


@needs_pyg
@needs_worktrees
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


@needs_pyg
@needs_worktrees
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


@needs_pyg
@needs_worktrees
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


def tiny_model():
    from torch import nn
    from torch_geometric.nn import global_mean_pool

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 2)

        def forward(self, data):
            return self.lin(global_mean_pool(data.x, data.batch))
    return Tiny


@needs_pyg
def test_run_arm_writes_a_run_the_harness_reads(tiny, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(fit, 'RUNS', tmp_path)
    graphs, sizes = synthetic_graphs(tp.all_ids(), np.random.default_rng(0))
    hp = {**train_a.HP, 'epochs': 3}
    threads = torch.get_num_threads()
    monkeypatch.setenv('TEAM_THREADS', '2')
    try:
        out = fit.run_arm(arm='fake18', fold=2, seed=1, graphs=graphs, sizes=sizes, build_model=tiny_model(), hp=hp,
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
    assert 0.0 <= metrics['test']['auc'] <= 1.0 and 'threshold_val' in metrics['selection']['auc']
    assert len((d / 'history.csv').read_text(encoding='utf-8').splitlines()) == metrics['epochs_run'] + 1
    del graphs[tp.all_ids()[5]]
    with pytest.raises(SystemExit, match='no graph'):
        fit.run_arm(arm='fake18', fold=2, seed=1, graphs=graphs, sizes=sizes, build_model=tiny_model(), hp=hp,
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


@needs_pyg
@needs_worktrees
@needs_real_data
@needs_a_graphs
def test_train_a_gnn500_smoke_creates_no_run_dir(no_write_run):
    out = train_a.run('gnn500', 0, 0, smoke=12, device='cpu', log=lambda *_: None)
    assert_smoke_clean(out, no_write_run)
    shutil.rmtree(out['dir'])


@needs_pyg
@needs_worktrees
@needs_real_data
@needs_b_build
def test_train_b_smoke_creates_no_run_dir(no_write_run):
    out = train_b.run('sig500', 0, 0, smoke=12, device='cpu', log=lambda *_: None)
    assert_smoke_clean(out, no_write_run)
    shutil.rmtree(out['dir'])


@needs_pyg
@needs_worktrees
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


# ---- the team stage of the gates ---------------------------------------------------------------------------------------------

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


def test_team_gates_from_synthetic_runs(tiny, tmp_path, monkeypatch):  # noqa: F811
    runs = tmp_path / 'runs'
    runs.mkdir()
    write_tfidf(runs)
    base = dict(protocol='kfold', arch='x86', model='gnn')
    fake_group(runs, lambda f, s: RunConfig(**base, fold=f, seed=s, mask_blocks=0.15), 0.5)          # R7, 3 seeds
    fake_group(runs, lambda f, s: RunConfig(**base, fold=f, seed=s, mask_blocks=0.15, subgraph='cap', budget=500,
                                            tag='b500'), 0.5)                                          # ours b500 typed
    fake_group(runs, lambda f, s: tp.team_config(f, s, 'gnn500'), 0.4)                                  # weak
    fake_group(runs, lambda f, s: tp.team_config(f, s, 'cb500'), 1.5)                                   # strong
    fake_group(runs, lambda f, s: tp.team_config(f, s, 'sig500'), 0.7, seeds=(0, 1))                    # seed 2 missing
    fake_group(runs, lambda f, s: tp.team_config(f, s, 'sig500v2'), 0.9)                                # the v2 arm, 3 seeds
    monkeypatch.setattr(stats, 'RUNS', runs)
    gate_tables.pooled.cache_clear()
    text = '\n'.join(gates.run_stage('team', 40))
    gate_tables.pooled.cache_clear()
    stage = gates.load_stage('team')
    arms = dict(stage.arms)
    ops = {l.split('|')[1].strip(): [c.strip() for c in l.split('|')[2:-1]]
           for l in text.splitlines() if l.startswith('| ') and l.count('|') == 12 and l.split('|')[1].strip() in arms}
    assert len(ops) == len(arms)
    for arm in ('TF-IDF', 'R7 typed, whole graph', 'ours b500 typed', 'team gnn500', 'team cb500', 'team sig500',
                'team sig500v2'):
        row = ops[arm]
        assert not row[0].startswith('missing'), (arm, row)
        auc, pr, f1, acc, prec, rec, tp_, fn, fp, tn = row[-10:]
        tp_, fn, fp, tn = int(tp_), int(fn), int(fp), int(tn)
        assert tp_ + fn == 20 and fp + tn == 20                              # ransomware / goodware in the five test folds
        assert float(acc) == pytest.approx((tp_ + tn) / 40, abs=6e-4)
        assert float(prec) == pytest.approx(tp_ / max(tp_ + fp, 1), abs=6e-4)
        assert float(rec) == pytest.approx(tp_ / 20, abs=6e-4)
    assert ops['ours b500 no-edges'][0].startswith('missing')
    pair = {(l.split('|')[1].strip(), l.split('|')[2].strip()): l.split('|')[-2].strip()
            for l in text.splitlines() if l.startswith('| ') and l.count('|') == 6
            and (l.split('|')[1].strip(), l.split('|')[2].strip()) in stage.pairs}
    assert len(pair) == len(stage.pairs)
    tf_cb = pair[('TF-IDF', 'team cb500')]
    assert tf_cb.startswith(('beats', 'does not beat')) and 'seeds [0] only: a direction, not the test' in tf_cb
    assert pair[('R7 typed, whole graph', 'ours b500 typed')] in ('non-inferior', 'not shown non-inferior')
    assert 'seeds [0, 1] only' in pair[('R7 typed, whole graph', 'team sig500')]                     # the seeds complete in both
    assert pair[('path 2: typed, CodeBERT vectors', 'team cb500')] == 'missing'
    assert 'seeds [0, 1] only' in pair[('team sig500', 'team sig500v2')]                          # sig500 has no seed 2
    assert pair[('R7 typed, whole graph', 'team sig500v2')] in ('beats', 'does not beat')           # 3 seeds in both
    assert 'seeds [0] only' in pair[('TF-IDF', 'team sig500v2')]


@needs_real_data
@needs_baselines
def test_a_synthetic_team_run_is_read_by_stats_against_the_real_baselines(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    (runs / 'baselines').mkdir(parents=True)
    for fold in FOLDS:
        shutil.copytree(RUNS / 'baselines' / f'kfold{fold}_x86_s0', runs / 'baselines' / f'kfold{fold}_x86_s0')
    write_fake_arm(runs)
    monkeypatch.setattr(stats, 'RUNS', runs)
    result, info = stats.compare('tfidf', 'kfold_x86_gnn_team-fake', 'auc', resamples=40)
    assert info['n'] == 1810 and info['seeds'] == [0] and not info['skipped'] and np.isfinite(result['auc']['diff'])
