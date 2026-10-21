"""python -m pytest tests/test_threshold_oof.py: the out-of-fold threshold rule (`--threshold-from oof`): the grouped folds of
`common.manifests`, `s04_evaluation.val_threshold.oof_rule`, `experiments.train.choose_thresholds`, the run names and the metrics text.
Pure numpy on synthetic rows and predictions; the models that use the rule are tested with their trainers
(tests/test_model1_trainer.py, tests/test_five_channel.py)."""
import hashlib

import numpy as np
import pytest

from common.config import THRESHOLD_FROM, RunConfig
from common.manifests import Manifest, grouped_folds
from common.run_io import AUC_RULES, auc_rule
from experiments.train import choose_thresholds, val_threshold_of
from s04_evaluation.val_threshold import GRID, OOF_K, grid_threshold, macro_f1, oof_rule


def rows(n_families=12, per_family=3, n_good=36, prefix='r'):
    """Ransomware binaries in families of `per_family`, goodware binaries that are their own identity group."""
    out = [{'sample_id': f'{prefix}{f}_{i}', 'label': 1, 'family': f'fam{f}', 'identity_group': f'fam{f}'}
           for f in range(n_families) for i in range(per_family)]
    return out + [{'sample_id': f'{prefix}g{i}', 'label': 0, 'family': '', 'identity_group': f'{prefix}g{i}'} for i in range(n_good)]


def noise(sample_id, scale):
    """A deterministic standard-ish draw per binary, so a fit_predict gives the same number for the same binary."""
    return (int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16) / 2**32 - 0.5) * 2 * scale


def overlapping(row, epoch=1):
    """The held-out probability of a model that is not saturated: ransomware around 0.65, goodware around 0.35."""
    return float(np.clip(0.5 + (0.15 if row['label'] else -0.15) + noise(row['sample_id'] + str(epoch), 0.25), 0.01, 0.99))


# ---- the grouped folds -----------------------------------------------------------------------------------------------------------

def test_grouped_folds_partition_the_rows_and_keep_every_group_whole():
    data = rows()
    folds = grouped_folds(data, 3, seed=0)
    ids = [r['sample_id'] for fold in folds for r in fold]
    assert len(folds) == 3 and sorted(ids) == sorted(r['sample_id'] for r in data) and len(set(ids)) == len(ids)
    group = lambda r: r['family'] if r['label'] else r['identity_group']
    where = {}
    for k, fold in enumerate(folds):
        for r in fold:
            assert where.setdefault((r['label'], group(r)), k) == k
    for fold in folds:
        labels = [r['label'] for r in fold]
        assert 0 < sum(labels) < len(labels)                                    # both classes in every fold
        assert [data.index(r) for r in fold] == sorted(data.index(r) for r in fold)      # the order of the rows is kept
    sizes = sorted(len(f) for f in folds)
    assert sizes[-1] - sizes[0] <= 3                                             # about the same size


def test_grouped_folds_depend_on_the_seed_only():
    data = rows()
    ids = lambda folds: [[r['sample_id'] for r in f] for f in folds]
    assert ids(grouped_folds(data, 3, 5)) == ids(grouped_folds(data, 3, 5))
    assert ids(grouped_folds(data, 3, 5)) != ids(grouped_folds(data, 3, 6))


def test_grouped_folds_refuse_a_class_with_fewer_groups_than_folds():
    with pytest.raises(ValueError, match='too few for 3 folds'):
        grouped_folds(rows(n_families=2), 3, 0)
    with pytest.raises(ValueError, match='too few for 3 folds'):
        grouped_folds(rows(n_good=2), 3, 0)


# ---- the rule --------------------------------------------------------------------------------------------------------------------

def test_the_oof_rule_refits_on_the_other_folds_and_pools_the_held_out_predictions():
    data = rows()
    calls = []

    def fit_predict(train_rows, held_rows, epochs, seed):
        calls.append((train_rows, held_rows, epochs, seed))
        return {e: [overlapping(r, e) for r in held_rows] for e in epochs}

    thresholds, record = oof_rule(data, fit_predict, {'f1': 3, 'auc': 5}, seed=2)
    assert len(calls) == OOF_K == 3
    held_ids = [r['sample_id'] for _, held, _, _ in calls for r in held]
    assert sorted(held_ids) == sorted(r['sample_id'] for r in data)                  # every binary is predicted once, out of fold
    for train_rows, held, epochs, seed in calls:
        assert not {r['sample_id'] for r in train_rows} & {r['sample_id'] for r in held}
        assert len(train_rows) + len(held) == len(data) and epochs == (3, 5)
        train_groups = {(r['label'], r['family'] or r['identity_group']) for r in train_rows}
        assert not train_groups & {(r['label'], r['family'] or r['identity_group']) for r in held}      # no group on both sides
    assert len({seed for *_, seed in calls}) == 3                                    # one seed per refit
    assert set(thresholds) == {'f1', 'auc'} and all(t in GRID for t in thresholds.values())
    assert record['rule'] == 'oof' and record['k'] == 3 and record['refit_epochs'] == {'f1': 3, 'auc': 5}
    assert record['n_oof'] == len(data) and sum(record['fold_sizes']) == len(data) and record['threshold'] == thresholds
    y = np.array([r['label'] for fold in grouped_folds(data, 3, 2) for r in fold])
    p = np.array([overlapping(r, 5) for fold in grouped_folds(data, 3, 2) for r in fold])
    assert thresholds['auc'] == grid_threshold(y, p) and record['oof_macro_f1']['auc'] == pytest.approx(macro_f1(y, p, thresholds['auc']))


def test_a_refit_that_gives_the_wrong_number_of_predictions_is_an_error():
    with pytest.raises(ValueError, match='held-out binaries'):
        oof_rule(rows(), lambda train, held, epochs, seed: {e: [0.5] for e in epochs}, {'auc': 1})


def test_when_validation_saturates_the_oof_rule_picks_a_different_better_calibrated_threshold():
    train = rows(prefix='t')
    val = rows(n_families=3, per_family=3, n_good=8, prefix='v')
    test = rows(n_families=10, per_family=3, n_good=30, prefix='x')
    # the validation families are separated perfectly: every threshold of the grid scores the same macro-F1 there
    val_probs = np.array([0.995 if r['label'] else 0.005 for r in val])
    val_y = np.array([r['label'] for r in val])
    assert all(macro_f1(val_y, val_probs, g) == 1.0 for g in GRID)
    manifest = Manifest(train, val, test, [])
    cfg = RunConfig(protocol='kfold', fold=0, model='tokenseq', threshold_from='oof')
    val_by_rule = {rule: ([r['sample_id'] for r in val], val_probs, None) for rule in ('f1', 'auc')}
    by_val, record_val = choose_thresholds(RunConfig(protocol='kfold', fold=0, model='tokenseq'), manifest, val_by_rule,
                                           {'f1': 4, 'auc': 4})
    assert record_val is None and by_val == {'f1': float(GRID[0]), 'auc': float(GRID[0])}             # the first of the plateau
    by_oof, record = choose_thresholds(cfg, manifest, val_by_rule, {'f1': 4, 'auc': 4},
                                       lambda tr, held, epochs, seed: {e: [overlapping(r) for r in held] for e in epochs})
    assert record['threshold_val'] == by_val and by_oof['auc'] != by_val['auc']
    assert 0.4 <= by_oof['auc'] <= 0.6                                                               # near where the classes cross
    # on binaries the model has not seen (the test families draw from the same overlapping scores) the OOF threshold is better
    y = np.array([r['label'] for r in test])
    p = np.array([overlapping(r) for r in test])
    at_val, at_oof = macro_f1(y, p, by_val['auc']), macro_f1(y, p, by_oof['auc'])
    assert at_val < 0.5 < 0.7 < at_oof


def test_the_default_rule_is_the_validation_rule_unchanged():
    val = rows(n_families=3, n_good=8, prefix='v')
    manifest = Manifest(rows(prefix='t'), val, [], [])
    rng = np.random.default_rng(0)
    probs = np.where([r['label'] for r in val], rng.uniform(0.3, 1.0, len(val)), rng.uniform(0.0, 0.7, len(val)))
    ids = [r['sample_id'] for r in val]
    cfg = RunConfig(protocol='kfold', fold=0, model='tokenseq')
    assert cfg.threshold_from == 'val'
    got, record = choose_thresholds(cfg, manifest, {'f1': (ids, probs, None), 'auc': (ids, probs[::-1].copy(), None)},
                                    {'f1': 1, 'auc': 1})
    assert record is None
    assert got == {'f1': val_threshold_of(manifest.val, ids, probs), 'auc': val_threshold_of(manifest.val, ids, probs[::-1].copy())}
    with pytest.raises(ValueError, match='no refit'):
        choose_thresholds(RunConfig(protocol='kfold', fold=0, model='tokenseq', threshold_from='oof'), manifest,
                          {'f1': (ids, probs, None)}, {'f1': 1})


# ---- the option, the names, the metrics text ---------------------------------------------------------------------------------------

def test_the_suffix_is_thoof_and_only_when_it_is_set():
    cfg = lambda **kw: RunConfig(protocol='kfold', fold=1, **kw)
    assert THRESHOLD_FROM == ('val', 'oof')
    assert cfg(model='tokenseq').name == 'kfold1_x86_tokenseq_s0'
    assert cfg(model='tokenseq', threshold_from='val').name == 'kfold1_x86_tokenseq_s0'
    assert cfg(model='tokenseq', threshold_from='oof').name == 'kfold1_x86_tokenseq-thoof_s0'
    assert cfg(model='tokenseq', threshold_from='oof', seq_len=256, max_windows=8).name == 'kfold1_x86_tokenseq-L256-w8-thoof_s0'
    assert cfg(model='fivechannel', threshold_from='oof').name == 'kfold1_x86_fivechannel-thoof_s0'
    assert cfg(model='fivechannel', threshold_from='oof', max_instructions=512).group == 'kfold_x86_fivechannel-L512-thoof'
    assert cfg(model='gnn', mask_blocks=0.15).name == 'kfold1_x86_gnn-mblk15_s0'


def test_only_the_sequence_models_take_the_option():
    with pytest.raises(ValueError, match="needs model='tokenseq' or 'fivechannel'"):
        RunConfig(protocol='kfold', fold=0, model='gnn', threshold_from='oof')
    with pytest.raises(ValueError, match='threshold_from'):
        RunConfig(protocol='kfold', fold=0, model='tokenseq', threshold_from='test')


def test_the_metrics_text_names_the_rule():
    cfg = RunConfig(protocol='kfold', fold=0, model='tokenseq')
    assert auc_rule(cfg) == AUC_RULES['overall']
    text = auc_rule(RunConfig(protocol='kfold', fold=0, model='tokenseq', threshold_from='oof'))
    assert text.startswith('checkpoint: validation AUC; threshold: out-of-fold macro-F1') and 'validation macro-F1' not in text


def test_the_report_labels_the_rule_and_does_not_list_it_as_an_override():
    from s04_evaluation.report import LABELLED, model_label
    assert {'threshold_from', 'max_instructions'} <= set(LABELLED)
    seq = lambda **kw: RunConfig(protocol='kfold', fold=0, **kw)
    assert model_label(seq(model='tokenseq')) == 'Model 1 token-sequence'
    assert model_label(seq(model='tokenseq', threshold_from='oof')) == 'Model 1 token-sequence thr-oof'
    assert model_label(seq(model='fivechannel', max_instructions=512, threshold_from='oof')) == 'Model 1 five-channel L512 thr-oof'
