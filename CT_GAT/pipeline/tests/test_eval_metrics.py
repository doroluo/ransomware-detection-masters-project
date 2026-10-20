"""python -m pytest tests/test_eval_metrics.py: PR-AUC in scores, the cell weights N / (C n_cell), the cell-balanced metrics and
the floors of `s04_evaluation.metrics`, and the validation-threshold rule of `val_threshold.py`."""
import json

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from common.config import RunConfig
from eval_synth import CELLS, cell_sample, needs_combined, v3_rows, write_v3_run
from s04_evaluation import val_threshold
from s04_evaluation.metrics import ARCHES, cell_balanced, cell_weights, floors, scores


def test_pr_auc_in_scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    prob = np.clip(y * 0.3 + rng.random(200) * 0.7, 0, 1)
    s = scores(y, prob, (prob >= 0.5).astype(int), ['x86'] * 200)
    assert s['pr_auc'] == pytest.approx(average_precision_score(y, prob))
    one = scores([1, 1, 1], [0.2, 0.9, 0.5], [0, 1, 1], ['x86'] * 3)
    assert np.isnan(one['pr_auc']) and np.isnan(one['auc'])


def test_cell_weights_on_a_synthetic_manifest():
    y, arch = cell_sample([60, 10, 20, 30])
    w = cell_weights(y, arch)
    assert len(w) == 120 and w.mean() == pytest.approx(1.0)
    assert w[0] == pytest.approx(120 / (4 * 60)) and w[60] == pytest.approx(120 / (4 * 10))
    assert w[70] == pytest.approx(120 / (4 * 20)) and w[90] == pytest.approx(120 / (4 * 30))
    for cell in CELLS:
        mask = (y == cell[0]) & (arch == cell[1])
        assert w[mask].sum() == pytest.approx(120 / 4)
    y, arch = cell_sample([60, 0, 20, 30])
    w = cell_weights(y, arch)
    assert w.mean() == pytest.approx(1.0) and w[0] == pytest.approx(110 / (3 * 60))
    y, arch = cell_sample([5, 0, 3, 0])
    assert cell_weights(y, arch) == pytest.approx([8 / (2 * 5)] * 5 + [8 / (2 * 3)] * 3)
    assert len(cell_weights([], [])) == 0


def test_cell_balanced_x86_rule_is_exactly_half():
    y, arch = cell_sample([60, 10, 20, 30])
    pred = (arch == 'x86').astype(int)
    cb = cell_balanced(y, pred.astype(float), pred, arch)
    assert cb['macro_f1'] == pytest.approx(0.5) and cb['accuracy'] == pytest.approx(0.5)
    assert cb['auc'] == pytest.approx(0.5) and cb['n_cells'] == 4
    assert [cb[f'n_{t}_{a}'] for t in 'RG' for a in ARCHES] == [60, 10, 20, 30]
    ordinary = scores(y, pred.astype(float), pred, arch)
    assert ordinary['accuracy'] != pytest.approx(0.5, abs=0.05) and ordinary['cell_balanced'] == cb
    fl = floors([0, 1], y, arch)
    assert fl['x86_rule']['cell_balanced']['macro_f1'] == pytest.approx(0.5)
    assert fl['x86_rule']['cell_balanced']['accuracy'] == pytest.approx(0.5)
    assert fl['majority']['cell_balanced']['accuracy'] == pytest.approx(0.5)
    assert fl['majority']['cell_balanced']['auc'] == pytest.approx(0.5)


def test_cell_balanced_hand_computed_one_inverted_cell():
    """Perfect on R x86, R x64 and G x86, inverted on G x64; every cell weighs 1/4 whatever its size. Positives: weight
    0.5 at prob 0.9; negatives: 0.25 at 0.1 and 0.25 at 0.9. Class 1: tp 0.5, fp 0.25, fn 0 -> F1 0.8. Class 0: tp 0.25,
    fp 0, fn 0.25 -> F1 2/3. AUC (0.5 * 0.25 + 0.5 * 0.25 / 2) / (0.5 * 0.5) = 0.75. AP: one step at recall 1 with
    precision 0.5 / 0.75."""
    y, arch = cell_sample([30, 50, 70, 20])
    pred = y.copy()
    pred[(y == 0) & (arch == 'x64')] = 1
    prob = np.where(pred == 1, 0.9, 0.1)
    cb = cell_balanced(y, prob, pred, arch)
    assert cb['accuracy'] == pytest.approx(0.75)
    assert cb['macro_f1'] == pytest.approx((0.8 + 2 / 3) / 2)
    assert cb['auc'] == pytest.approx(0.75)
    assert cb['pr_auc'] == pytest.approx(2 / 3)
    plain = scores(y, prob, pred, arch)
    assert plain['accuracy'] == pytest.approx(1 - 20 / 170)


def test_cell_balanced_with_an_empty_cell_and_one_class():
    y, arch = cell_sample([40, 0, 10, 0])
    pred = np.concatenate([np.ones(40, int), np.zeros(5, int), np.ones(5, int)])
    cb = cell_balanced(y, np.where(pred == 1, 0.8, 0.2), pred, arch)
    assert cb['n_cells'] == 2 and cb['n_R_x64'] == 0 and cb['n_G_x64'] == 0 and cb['n_R_x86'] == 40
    assert cb['n_G_x86'] == 10 and cb['accuracy'] == pytest.approx(0.75)
    y1, arch1 = cell_sample([5, 5, 0, 0])
    one = cell_balanced(y1, np.full(10, 0.5), np.ones(10, int), arch1)
    assert np.isnan(one['auc']) and np.isnan(one['pr_auc']) and one['n_cells'] == 2 and one['accuracy'] == 1.0
    s = scores([1, 1, 1], [0.2, 0.9, 0.5], [0, 1, 1], ['x86'] * 3)
    assert s['cell_balanced']['n_cells'] == 1 and np.isnan(s['cell_balanced']['auc'])
    json.dumps(s)


def test_the_threshold_grid_and_the_scores_at_a_threshold():
    assert len(val_threshold.GRID) == 49 and val_threshold.GRID[0] == pytest.approx(0.02) and val_threshold.GRID[-1] == pytest.approx(0.98)
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.9])
    best = max(val_threshold.GRID, key=lambda g: val_threshold.macro_f1(y, p, g))
    assert 0.3 < best <= 0.4 and val_threshold.macro_f1(y, p, best) == pytest.approx(1.0)
    s = val_threshold.scores(y, p, 0.5)
    assert s['recall_R'] == pytest.approx(2 / 3) and s['recall_G'] == pytest.approx(1.0) and s['auc'] == pytest.approx(1.0)
    assert val_threshold.fmt([0.5]) == '0.500' and val_threshold.fmt([0.4, 0.6]).startswith('0.500 ± ')


def test_val_threshold_pools_kfold_v3_like_kfold(tmp_path, monkeypatch, capsys):
    import csv
    monkeypatch.setattr(val_threshold, 'RUNS', tmp_path)
    for fold in range(5):
        d = write_v3_run(tmp_path, 'all', fold, 1.5)
        with (d / 'predictions.csv').open('a', newline='') as f:
            w = csv.writer(f)
            for sid, y, p, fam, a in v3_rows(fold, 3, 1.5):
                w.writerow([sid + 'v', 'val', y, f'{p:.6f}', int(p >= 0.5), a, fam, fold, 5])
    val_threshold.main()
    out = capsys.readouterr().out
    assert out.count('| kfold_v3_all_gnn_s0 | folds | 5 |') == 1 and '| kfold_v3_all_gnn_s0 | pooled | 600 |' in out
    assert '| kfold_v3_all_gnn | 1 |' in out


@needs_combined
def test_cell_weights_on_a_real_v3_manifest():
    from common.manifests import load_index, make_manifest
    from common.paths import COMBINED_CACHE
    m = make_manifest(RunConfig(protocol='kfold_v3', fold=0, arch='all'), load_index(COMBINED_CACHE))
    y, arch = np.array([r['label'] for r in m.train]), np.array([r['arch'] for r in m.train])
    w = cell_weights(y, arch)
    assert w.mean() == pytest.approx(1.0)
    for label, a in CELLS:
        assert w[(y == label) & (arch == a)].sum() == pytest.approx(len(y) / 4)
