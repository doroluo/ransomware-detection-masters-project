"""python -m pytest tests/test_eval_baselines.py: the TF-IDF / structural baselines of `s04_evaluation.baselines` on the tiny
cache: validation rows and thresholds, `--train-fraction`, `--arch-balance`, `--vocab noapi`, and the out-of-fold logits of
`--stack-logits`. `--imports` and `--vocab yanping|hybrid` read side arrays that come with the GNN and the experiments (later
steps) and are not run here."""
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from common.config import RunConfig
from common.manifests import Manifest, load_index, load_sample, make_manifest
from eval_synth import evaltiny, gnn  # noqa: F401  (a fixture)
from s04_evaluation import baselines, gate_tables, report, stats
from s04_evaluation.val_threshold import GRID
from tiny_cache import tiny  # noqa: F401  (a fixture)

FOLDS = range(5)


@pytest.fixture(scope='module')
def stack_files(evaltiny, tmp_path_factory):  # noqa: F811
    """stack_logits.csv for fold 0 of the tiny cache in a temporary baselines directory."""
    root = tmp_path_factory.mktemp('baselines')
    cfg = RunConfig(protocol='kfold', fold=0, arch='x86', seed=0, min_df=1)
    sizes = {k: len(v) for k, v in json.loads((baselines.CACHE / 'vocab.json').read_text(encoding='utf-8')).items()}
    out = baselines.write_stack_logits(cfg, sizes, root)
    return SimpleNamespace(root=root, dir=out, cfg=cfg, manifest=make_manifest(cfg, load_index()))


def test_baselines_write_val_rows_and_thresholds(evaltiny, tmp_path):  # noqa: F811
    sizes = {k: len(v) for k, v in json.loads((baselines.CACHE / 'vocab.json').read_text(encoding='utf-8')).items()}
    baselines.run(RunConfig(protocol='kfold', fold=0, seed=0), sizes, tmp_path)
    d = tmp_path / 'kfold0_x86_s0'
    m = json.loads((d / 'metrics.json').read_text())
    for v in m['methods'].values():
        assert v['threshold_val'] in [float(g) for g in GRID]
        assert v['test_at_val_threshold']['n'] == v['test']['n'] and 'pr_auc' in v['test']
    with (d / 'predictions.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert {r['split'] for r in rows} == {'val', 'test'}
    assert len(report.read_predictions(d / 'predictions.csv', method='structural_lr')) == m['n_test']
    assert list(rows[0])[:3] == ['sample_id', 'split', 'method']


def test_baselines_train_fraction(evaltiny, tmp_path):  # noqa: F811
    sizes = {k: len(v) for k, v in json.loads((baselines.CACHE / 'vocab.json').read_text(encoding='utf-8')).items()}
    baselines.run(RunConfig(protocol='kfold', fold=1, seed=0, train_fraction=0.25), sizes, tmp_path)
    d = tmp_path / 'kfold1_x86_s0_tf25'
    m = json.loads((d / 'metrics.json').read_text())
    full = make_manifest(RunConfig(protocol='kfold', fold=1, seed=0), load_index())
    assert m['train_fraction'] == 0.25 and m['n_val'] == len(full.val) and m['n_test'] == len(full.test)
    assert abs(m['n_train'] - 0.25 * len(full.train)) <= 2 and m['n_train'] < len(full.train)
    for v in m['methods'].values():
        assert v['threshold_val'] in [float(g) for g in GRID] and 'pr_auc' in v['test']
    with (d / 'predictions.csv').open() as f:
        assert {r['split'] for r in csv.DictReader(f)} == {'val', 'test'}
    assert [p.name for p in tmp_path.iterdir()] == ['kfold1_x86_s0_tf25']


def test_baselines_arch_balance_matches_class_weight_on_one_architecture(evaltiny, tmp_path):  # noqa: F811
    """On an x86-only manifest the only cells are the two classes, so N / (C n_cell) is the 'balanced' class weight:
    the arch-balanced fit must reproduce the default fit, and the output goes to a `_ab` directory."""
    sizes = {k: len(v) for k, v in json.loads((baselines.CACHE / 'vocab.json').read_text(encoding='utf-8')).items()}
    baselines.run(RunConfig(protocol='kfold', fold=0, seed=0), sizes, tmp_path)
    baselines.run(RunConfig(protocol='kfold', fold=0, seed=0, arch_balance=True), sizes, tmp_path)
    plain, ab = tmp_path / 'kfold0_x86_s0', tmp_path / 'kfold0_x86_s0_ab'
    m, mab = (json.loads((d / 'metrics.json').read_text()) for d in (plain, ab))
    assert m['arch_balance'] is False and mab['arch_balance'] is True
    for name, v in mab['methods'].items():
        for key in ('test', 'test_at_val_threshold'):
            cb = v[key]['cell_balanced']
            assert cb['n_cells'] == 2 and cb['n_R_x64'] == 0 and cb['n_G_x64'] == 0 and cb['n_R_x86'] > 0
        assert v['test']['auc'] == pytest.approx(m['methods'][name]['test']['auc'], abs=1e-9)
    assert 'cell_balanced' in m['floors']['x86_rule'] and 'cell_balanced' in mab['floors']['majority']
    assert report.read_predictions(ab / 'predictions.csv', method='tfidf_sig_api_lr')


def run_noapi_baselines(box, min_df=2):
    sizes = {k: len(v) for k, v in json.loads((box.root / 'vocab.json').read_text(encoding='utf-8')).items()}
    for f in FOLDS:
        baselines.run(RunConfig(protocol='kfold', fold=f, arch='x86', seed=0, min_df=min_df), sizes,
                      box.runs / 'baselines', 'noapi')


def test_noapi_baseline_counts_mn_sig_beh_and_writes_its_directory(evaltiny, monkeypatch):  # noqa: F811
    monkeypatch.setattr(baselines, 'load_ytok', lambda *a: pytest.fail('noapi must not read the reader side arrays'))
    run_noapi_baselines(evaltiny)
    gate_tables.pooled.cache_clear()
    for f in FOLDS:
        d = evaltiny.runs / 'baselines' / f'kfold{f}_x86_s0_voc-noapi'
        m = json.loads((d / 'metrics.json').read_text(encoding='utf-8'))
        assert list(m['methods']) == ['tfidf_noapi_lr', 'tfidf_sig_lr'] and m['vocab'] == 'noapi'
        assert 'threshold_val' in m['methods']['tfidf_noapi_lr']
        manifest = make_manifest(RunConfig(protocol='kfold', fold=f, arch='x86', seed=0), load_index())
        counts = {k: np.zeros(n, dtype=int) for k, n in (('mn', 30), ('sig', 50), ('beh', 9), ('api', 24))}
        for r in manifest.train:
            z = load_sample(r['sample_id'])
            for k in counts:
                counts[k][np.unique(z[k])] += 1
        assert m['tfidf_features'] == sum(int((counts[k] >= 2).sum()) for k in ('mn', 'sig', 'beh'))
        assert m['tfidf_features'] > int((counts['sig'] >= 2).sum())                    # more than sig alone: mn and beh count too
        rows = (d / 'predictions.csv').read_text(encoding='utf-8').splitlines()[1:]
        assert {r.split(',')[2] for r in rows} == {'tfidf_noapi_lr', 'tfidf_sig_lr'}
    assert not (evaltiny.runs / 'baselines' / 'kfold0_x86_s0').exists()                      # the plain directory is untouched
    folds = stats.folds_by_seed('tfidf-noapi')
    assert sorted(folds[0]) == [0, 1, 2, 3, 4] and stats.baseline_method('tfidf-noapi') == ('tfidf_noapi_lr', '_voc-noapi')
    r = gate_tables.pooled('tfidf-noapi', 'auc')
    assert not isinstance(r, str) and 0 <= r[0] <= 1
    gate_tables.pooled.cache_clear()


def test_the_default_baseline_is_unchanged(evaltiny):  # noqa: F811
    sizes = {k: len(v) for k, v in json.loads((evaltiny.root / 'vocab.json').read_text(encoding='utf-8')).items()}
    out = evaltiny.runs / 'plain_noapi_test'
    baselines.run(RunConfig(protocol='kfold', fold=1, arch='x86', seed=0), sizes, out)
    m = json.loads((out / 'kfold1_x86_s0' / 'metrics.json').read_text(encoding='utf-8'))
    assert list(m['methods']) == ['tfidf_sig_api_lr', 'structural_lr'] and 'vocab' not in m


def synthetic_problem(n=400, d=30, seed=0):
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    groups = np.array([f'g{i // 5}' for i in range(n)])
    y = (np.arange(n) // 5 % 2).astype(int)
    x = rng.poisson(0.3, (n, d)).astype(float)
    x[:, 0] += 3 * y                                                                  # a real signal
    return sp.csr_matrix(x), y, groups


def test_inner_folds_never_split_a_group_and_keep_both_classes():
    x, y, groups = synthetic_problem()
    folds = baselines.inner_folds(y, groups, seed=0)
    assert len(folds) == 5
    held_all = np.concatenate([held for _, held in folds])
    assert sorted(held_all) == list(range(len(y)))
    for fit, held in folds:
        assert not set(groups[fit]) & set(groups[held])
        assert set(y[held]) == {0, 1} and set(y[fit]) == {0, 1}
    assert [list(h) for _, h in folds] == [list(h) for _, h in baselines.inner_folds(y, groups, seed=0)]


def test_oof_logit_of_a_training_row_differs_from_the_logit_of_a_model_that_saw_it():
    x, y, groups = synthetic_problem()
    oof, which = baselines.oof_logits(x, y, groups, min_df=2, seed=0)
    folds = baselines.inner_folds(y, groups, seed=0)
    assert set(which) == {0, 1, 2, 3, 4}
    for k in (0, 3):
        fit, held = folds[k]
        row = held[0]
        # a model fitted without the row's group reproduces the stored logit exactly ...
        (without,) = baselines.fit_logits(x[fit], y[fit], [x[[row]]], 2)
        assert without[0] == pytest.approx(oof[row], abs=1e-12) and which[row] == k
        # ... and a model fitted with the row (the full fit) gives another one
        (with_row,) = baselines.fit_logits(x[np.append(fit, row)], y[np.append(fit, row)], [x[[row]]], 2)
        assert abs(with_row[0] - oof[row]) > 1e-6
    (full,) = baselines.fit_logits(x, y, [x], 2)
    assert np.abs(full - oof).mean() > 1e-3 and (full != oof).all()
    assert np.corrcoef(full, oof)[0, 1] > 0.5                                         # the signal survives


def test_a_memorised_row_shows_in_the_full_fit_but_not_in_the_out_of_fold_logit():
    import scipy.sparse as sp
    x, y, groups = synthetic_problem(seed=1)
    x = sp.hstack([x, sp.csr_matrix((np.arange(len(y)) == 7).astype(float)[:, None] * 5)]).tocsr()   # a feature only row 7 has
    y = y.copy()
    y[7] = 1 - y[7]                                                                   # and it disagrees with its group
    groups = groups.copy()
    oof, _ = baselines.oof_logits(x, y, groups, min_df=1, seed=0)
    (full,) = baselines.fit_logits(x, y, [x], 1)
    sign = 1 if y[7] == 1 else -1
    assert sign * full[7] > sign * oof[7]                                             # the full fit has learnt row 7
    assert sign * oof[7] < 0.5


def test_inner_groups_keep_classes_apart():
    rows = [dict(label=1, family='a', identity_group='x'), dict(label=0, family='a', identity_group='a'),
            dict(label=0, family='goodware', identity_group='x')]
    g = baselines.inner_groups(rows)
    assert len(set(g)) == 3 and g[0] == 'fam:a' and g[1] == 'id:a'


def test_baseline_name_matches_the_directories_baselines_writes():
    assert baselines.baseline_name(RunConfig(protocol='kfold', fold=0)) == 'kfold0_x86_s0'
    assert baselines.baseline_name(gnn(protocol='kfold', fold=3, seed=2, mask_blocks=0.15, tag='smoke')) == 'kfold3_x86_s2'
    assert baselines.baseline_name(RunConfig(protocol='kfold_v3', fold=1, arch='all')) == 'v3kfold1_all_s0'
    assert baselines.baseline_name(RunConfig()) == 'bundle_x86_s0'
    assert baselines.baseline_name(gnn(protocol='kfold', fold=0, seed=1, val_seed=0)) == 'kfold0_x86-vs0_s1'


def test_stack_logits_csv_roles_and_leakage(stack_files):
    import csv
    rows = list(csv.DictReader((stack_files.dir / 'stack_logits.csv').open(encoding='utf-8')))
    m = stack_files.manifest
    assert [r['split'] for r in rows].count('train') == len(m.train) and len(rows) == len(m.train) + len(m.val) + len(m.test)
    assert list(rows[0]) == ['sample_id', 'split', 'inner_fold', 'label', 'family', 'logit']
    by_split = {s: {r['sample_id'] for r in rows if r['split'] == s} for s in ('train', 'val', 'test')}
    assert by_split['train'] == {r['sample_id'] for r in m.train} and by_split['test'] == {r['sample_id'] for r in m.test}
    train_rows = [r for r in rows if r['split'] == 'train']
    assert {r['inner_fold'] for r in train_rows} == {'0', '1', '2', '3', '4'}
    assert all(r['inner_fold'] == '' for r in rows if r['split'] != 'train')
    # inner folds are grouped: a family or goodware identity group sits in one inner fold only
    index = {r['sample_id']: r for r in m.train}
    where = {}
    for r in train_rows:
        src = index[r['sample_id']]
        group = ('fam', src['family']) if src['label'] == 1 else ('id', src['identity_group'])
        where.setdefault(group, set()).add(r['inner_fold'])
    assert all(len(v) == 1 for v in where.values())
    meta = json.loads((stack_files.dir / 'stack_logits.json').read_text(encoding='utf-8'))
    assert meta['inner_splits'] == 5 and meta['n_train'] == len(m.train)
    # a training row's out-of-fold logit differs from the full-fit logit of a model that includes it
    sizes = {k: len(v) for k, v in json.loads((baselines.CACHE / 'vocab.json').read_text(encoding='utf-8')).items()}
    x_train, _ = baselines.load_features(m.train[:], sizes)
    y = np.array([r['label'] for r in m.train])
    (full,) = baselines.fit_logits(x_train, y, [x_train], stack_files.cfg.min_df)
    oof = np.array([float(next(r['logit'] for r in train_rows if r['sample_id'] == s['sample_id'])) for s in m.train])
    assert (full != oof).all() and np.abs(full - oof).mean() > 1e-3
    pos = y == 1
    assert np.abs(full[pos]).mean() > np.abs(oof[pos]).mean()                         # in-sample logits are more confident


def test_read_stack_logits_checks_roles(stack_files, tmp_path):
    m = stack_files.manifest
    path = stack_files.dir / 'stack_logits.csv'
    got = baselines.read_stack_logits(path, m)
    assert len(got) == len(m.train) + len(m.val) + len(m.test)
    swapped = Manifest(m.train[1:], m.val + m.train[:1], m.test, m.val_families)      # a train row asked as a val row
    with pytest.raises(ValueError, match='needs it as a val row'):
        baselines.read_stack_logits(path, swapped)
    extra = Manifest(m.train, m.val, m.test + [{**m.train[0], 'sample_id': 'f' * 64}], m.val_families)
    with pytest.raises(ValueError, match='missing'):
        baselines.read_stack_logits(path, extra)
    with pytest.raises(FileNotFoundError, match='--stack-logits'):
        baselines.read_stack_logits(tmp_path / 'none.csv', m)
    subset = Manifest(m.train[::2], m.val, m.test, m.val_families)                    # --train-fraction: a subset is fine
    assert set(baselines.read_stack_logits(path, subset)) == {r['sample_id'] for r in subset.train + subset.val + subset.test}
