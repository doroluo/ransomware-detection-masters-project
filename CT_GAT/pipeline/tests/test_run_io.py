"""python -m pytest tests/test_run_io.py: `common.run_io.write_run` and its coverage check, on the tiny cache.

write_run takes the predictions of a model in its two states and writes the run directory that stats, report and gates read.
The ids it is given must be exactly the manifest's validation and test ids less the ones declared excluded.
"""
import csv
import json

import numpy as np
import pytest

from common import run_io
from common.config import RunConfig
from common.manifests import load_index, make_manifest
from s04_evaluation import report, stats, val_threshold
from tiny_cache import tiny  # noqa: F401  (a fixture)

EXTRA = {'best_epoch': 2, 'epochs_run': 5, 'auc_best_epoch': 3, 'val_auc': 0.9, 'val_macro_f1': 0.8}
WHY = 'process_sample raises MemoryError reading the listing'


def fake_rows(manifest, rng, signal=0.8, skip=()):
    out = {}
    for split in ('val', 'test'):
        rows = [r for r in getattr(manifest, split) if r['sample_id'] not in skip]
        probs = np.clip(0.5 + signal * (np.array([r['label'] for r in rows]) - 0.5) + rng.normal(0, 0.2, len(rows)), 0.01, 0.99)
        out[split] = ([r['sample_id'] for r in rows], probs, [(3, 5, 7)] * len(rows))
    return out


def fold_config(fold=0, seed=0, **kw):
    return RunConfig(protocol='kfold', fold=fold, seed=seed, arch='x86', model='gnn', **kw)


def test_write_run_with_the_default_manifest_equals_the_one_passed_in(tiny, tmp_path):  # noqa: F811
    cfg = fold_config(1, 2, mask_blocks=0.15, tag='t')
    manifest = make_manifest(cfg, load_index())
    rows = fake_rows(manifest, np.random.default_rng(0))
    a = run_io.write_run(tmp_path / 'a' / cfg.name, cfg, rows, rows, 0.4, 0.45, EXTRA)
    b = run_io.write_run(tmp_path / 'b' / cfg.name, cfg, rows, rows, 0.4, 0.45, EXTRA, manifest=manifest)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)                  # (NaN != NaN, so not ==)
    for name in ('config.json', 'manifest.json', 'metrics.json', 'predictions.csv', 'predictions_auc.csv'):
        assert (tmp_path / 'a' / cfg.name / name).read_bytes() == (tmp_path / 'b' / cfg.name / name).read_bytes()
    with pytest.raises(ValueError, match='RunConfig name'):
        run_io.write_run(tmp_path / 'wrong', cfg, rows, rows, 0.4, 0.45, EXTRA)
    with pytest.raises(AssertionError):
        run_io.write_run(tmp_path / 'c' / cfg.name, cfg, {**rows, 'test': tuple(x[:-1] for x in rows['test'])}, rows, 0.4,
                         0.45, EXTRA)
    with pytest.raises(KeyError):
        run_io.write_run(tmp_path / 'd' / cfg.name, cfg, rows, rows, 0.4, 0.45, {'best_epoch': 1})


def test_the_run_directory_is_what_stats_and_report_read(tiny, tmp_path, monkeypatch):  # noqa: F811
    cfg = fold_config(0, 0, tag='fake')
    manifest = make_manifest(cfg, load_index())
    rng = np.random.default_rng(0)
    f1, auc = fake_rows(manifest, rng, 0.8), fake_rows(manifest, rng, 0.9)
    d = tmp_path / cfg.name
    metrics = run_io.write_run(d, cfg, f1, auc, 0.3, 0.35, EXTRA)
    assert RunConfig.load(d / 'config.json') == cfg
    for f in ('config.json', 'manifest.json', 'metrics.json', 'predictions.csv', 'predictions_auc.csv'):
        assert (d / f).is_file()
    assert (d / 'predictions.csv').read_text(encoding='utf-8').splitlines()[0] == (
        'sample_id,split,label,prob,pred,arch,family,fold,n_segments_total,n_blocks_kept,n_blocks_total')
    saved = json.loads((d / 'metrics.json').read_text(encoding='utf-8'))
    for key in ('best_epoch', 'epochs_run', 'test', 'floors', 'family_recall', 'truncated_test'):
        assert key in saved
    assert saved['truncated_test'] == len(manifest.test) == metrics['truncated_test']          # (3, 5, 7): every binary is cut
    assert {'threshold_val', 'test', 'family_recall', 'rule', 'best_epoch', 'val_auc'} <= set(saved['selection']['auc'])
    assert saved['selection']['f1']['threshold_val'] == 0.3 and saved['selection']['auc']['threshold_val'] == 0.35
    assert saved['selection']['auc']['rule'] == run_io.AUC_RULES['overall']
    with (d / 'predictions.csv').open(newline='', encoding='utf-8') as f:                       # thresholded at cfg.threshold
        assert all(int(r['pred']) == int(float(r['prob']) >= cfg.threshold) for r in csv.DictReader(f))
    with (d / 'predictions_auc.csv').open(newline='', encoding='utf-8') as f:                   # at the AUC checkpoint's threshold
        assert all(int(r['pred']) == int(float(r['prob']) >= 0.35) for r in csv.DictReader(f))
    assert len(report.read_predictions(d / 'predictions.csv')) == len(manifest.test)
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    assert stats.folds_by_seed(cfg.group)[0] == {0: d}


def test_write_run_takes_an_explicit_excluded_list(tiny, tmp_path):  # noqa: F811
    cfg = fold_config(0, 0, tag='fake')
    manifest = make_manifest(cfg, load_index())
    victim = manifest.test[0]['sample_id']
    other = make_manifest(fold_config(1, 0), load_index()).test[0]['sample_id']              # in the train or val set of fold 0
    assert other in {r['sample_id'] for r in manifest.train + manifest.val}
    rows = fake_rows(manifest, np.random.default_rng(0), skip={victim})
    assert victim not in rows['test'][0]
    with pytest.raises(AssertionError, match='missing'):                      # not declared: the coverage check still fails
        run_io.write_run(tmp_path / 'a' / cfg.name, cfg, rows, rows, 0.5, 0.5, EXTRA)
    with pytest.raises(ValueError, match='excluded_reason'):
        run_io.write_run(tmp_path / 'b' / cfg.name, cfg, rows, rows, 0.5, 0.5, EXTRA, excluded=[victim])
    with pytest.raises(ValueError, match='no split'):
        run_io.write_run(tmp_path / 'c' / cfg.name, cfg, rows, rows, 0.5, 0.5, EXTRA, excluded=['0' * 64], excluded_reason='x')
    full = fake_rows(manifest, np.random.default_rng(0))
    with pytest.raises(AssertionError, match='extra'):                         # declared but its rows are there
        run_io.write_run(tmp_path / 'd' / cfg.name, cfg, full, full, 0.5, 0.5, EXTRA, excluded=[victim], excluded_reason='x')
    metrics = run_io.write_run(tmp_path / 'e' / cfg.name, cfg, rows, rows, 0.5, 0.5, EXTRA, excluded=[victim, other],
                               excluded_reason=WHY)
    assert metrics['excluded'] == sorted([victim, other]) and metrics['excluded_reason'] == WHY
    saved = json.loads((tmp_path / 'e' / cfg.name / 'metrics.json').read_text(encoding='utf-8'))
    assert saved['excluded'] == metrics['excluded'] and saved['excluded_reason'] == WHY
    predictions = (tmp_path / 'e' / cfg.name / 'predictions.csv').read_text(encoding='utf-8')
    assert victim not in predictions and len(predictions.splitlines()) - 1 == len(manifest.val) + len(manifest.test) - 1
    plain = run_io.write_run(tmp_path / 'f' / cfg.name, cfg, full, full, 0.5, 0.5, EXTRA)
    assert 'excluded' not in plain and 'excluded_reason' not in plain         # an ordinary run records nothing extra


def test_write_predictions_adds_trailing_block_columns_and_readers_still_work(tmp_path):
    rows = [{'sample_id': s, 'label': y, 'arch': a, 'family': 'f', 'fold': 1}
            for s, y, a in (('a', 1, 'x86'), ('b', 0, 'x64'), ('c', 1, 'x64'))]
    by_split = {'val': (['a'], np.array([0.9]), [(5, 10, 10)]),
                'test': (['b', 'c'], np.array([0.2, 0.7]), [(7, 400, 900), (3, None, None)])}
    path = tmp_path / 'predictions.csv'
    run_io.write_predictions(path, rows, by_split, 0.5)
    with path.open(newline='') as f:
        read = list(csv.DictReader(f))
    assert list(read[0]) == ['sample_id', 'split', 'label', 'prob', 'pred', 'arch', 'family', 'fold',
                             'n_segments_total', 'n_blocks_kept', 'n_blocks_total']
    assert [(r['n_segments_total'], r['n_blocks_kept'], r['n_blocks_total']) for r in read] == [
        ('5', '10', '10'), ('7', '400', '900'), ('3', '', '')]
    assert run_io.truncated_count(by_split['test'][2]) == 1 and run_io.truncated_count([(1, 5, 5)]) == 0
    assert [stats.truncation(r) for r in read] == [0, 1, -1]
    assert len(report.read_predictions(path)) == 2
    assert val_threshold.load(tmp_path)[2].tolist() == [0, 1]


def test_evaluate_scores_the_rows_it_is_given():
    rows = [{'sample_id': s, 'label': y, 'arch': a}
            for s, y, a in (('a', 1, 'x86'), ('b', 0, 'x64'), ('c', 1, 'x64'), ('d', 0, 'x86'))]
    s = run_io.evaluate(rows, ['d', 'c', 'b', 'a'], np.array([0.1, 0.8, 0.3, 0.6]), 0.5)
    assert s['n'] == 4 and s['accuracy'] == 1.0 and s['auc'] == 1.0 and s['recall_G_x64'] == 1.0
