"""Binary-classification metrics shared by train.py, baselines.py and report.py (ransomware = 1 is positive)."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ARCHES = ('x86', 'x64')


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
    }
    for a in ARCHES:
        for cls, tag in ((1, 'R'), (0, 'G')):
            mask = (arch == a) & (y == cls)
            out[f'recall_{tag}_{a}'] = float((pred[mask] == cls).mean()) if mask.any() else float('nan')
            out[f'n_{tag}_{a}'] = int(mask.sum())
    return {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in out.items()}


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
