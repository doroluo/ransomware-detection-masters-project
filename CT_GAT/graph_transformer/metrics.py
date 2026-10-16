"""Binary-classification metrics shared by train.py, baselines.py and report.py (ransomware = 1 is positive)."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score, recall_score,
                             roc_auc_score)

ARCHES = ('x86', 'x64')


def cell_weights(y, arch):
    """N / (C * n_cell) per sample, C the number of non-empty (label, arch) cells: every non-empty cell carries the same
    total weight and the mean weight is 1."""
    cells = list(zip(np.asarray(y).tolist(), np.asarray(arch).tolist()))
    if not cells:
        return np.zeros(0)
    counts = {c: cells.count(c) for c in set(cells)}
    return np.array([len(cells) / (len(counts) * counts[c]) for c in cells])


def cell_balanced(y, prob, pred, arch):
    """Accuracy, macro-F1, AUC and PR-AUC with sample weights proportional to 1 / (C * n_cell) (cell_weights), so each
    non-empty (label, arch) cell weighs the same (under it the x86 rule scores 0.5 when all four cells are non-empty),
    and the cell counts. A cell with no test sample is left out of C."""
    y, prob, pred, arch = np.asarray(y), np.asarray(prob, dtype=float), np.asarray(pred), np.asarray(arch)
    w = cell_weights(y, arch)
    two = len(set(y.tolist())) == 2
    out = {
        'accuracy': accuracy_score(y, pred, sample_weight=w),
        'macro_f1': f1_score(y, pred, average='macro', zero_division=0, sample_weight=w),
        'auc': roc_auc_score(y, prob, sample_weight=w) if two else float('nan'),
        'pr_auc': average_precision_score(y, prob, sample_weight=w) if two else float('nan'),
    }
    out = {k: float(v) for k, v in out.items()}
    out['n_cells'] = len({(a, b) for a, b in zip(y.tolist(), arch.tolist())})
    for a in ARCHES:
        for cls, tag in ((1, 'R'), (0, 'G')):
            out[f'n_{tag}_{a}'] = int(((arch == a) & (y == cls)).sum())
    return out


def scores(y, prob, pred, arch):
    y, prob, pred, arch = np.asarray(y), np.asarray(prob, dtype=float), np.asarray(pred), np.asarray(arch)
    out = {
        'n': int(len(y)), 'n_ransomware': int(y.sum()),
        'accuracy': accuracy_score(y, pred),
        'precision': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0),
        'macro_f1': f1_score(y, pred, average='macro', zero_division=0),
        'auc': roc_auc_score(y, prob) if len(set(y.tolist())) == 2 else float('nan'),
        'pr_auc': average_precision_score(y, prob) if len(set(y.tolist())) == 2 else float('nan'),
    }
    for a in ARCHES:
        for cls, tag in ((1, 'R'), (0, 'G')):
            mask = (arch == a) & (y == cls)
            out[f'recall_{tag}_{a}'] = float((pred[mask] == cls).mean()) if mask.any() else float('nan')
            out[f'n_{tag}_{a}'] = int(mask.sum())
    out = {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in out.items()}
    out['cell_balanced'] = cell_balanced(y, prob, pred, arch)
    return out


def floors(y_train, y_test, arch_test):
    """Majority class of the training manifest, and the x86 rule (x86 -> ransomware, x64 -> goodware)."""
    y_test, arch_test = np.asarray(y_test), np.asarray(arch_test)
    majority = int(np.mean(y_train) >= 0.5)
    pred_major = np.full(len(y_test), majority)
    pred_x86 = (arch_test == 'x86').astype(int)
    return {'majority': scores(y_test, pred_major.astype(float), pred_major, arch_test),
            'x86_rule': scores(y_test, pred_x86.astype(float), pred_x86, arch_test)}


def family_recall(y, pred, family):
    y, pred, family = np.asarray(y), np.asarray(pred), np.asarray(family)
    out = {}
    for f in sorted(set(family[y == 1].tolist())):
        mask = (family == f) & (y == 1)
        out[f] = {'n': int(mask.sum()), 'recall': float((pred[mask] == 1).mean())}
    return out
