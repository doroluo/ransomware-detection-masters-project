"""The team-pipeline adapter (experiments/team_pipelines/common.py): fold partition, bundle lookup, run directories the
harness reads, and the sha256 pin on the teammates' code. CPU only. Reads the real v2 cache index and the Mendeley bundle."""
import json
import shutil

import numpy as np
import pytest

from common.config import DEFAULT_CACHE, RUNS, RunConfig
from evaluation import report, stats
from experiments.team_pipelines import common as tp
from training.train import val_threshold_of

FOLDS = range(5)
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
