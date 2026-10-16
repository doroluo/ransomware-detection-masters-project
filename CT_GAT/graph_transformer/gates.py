"""Gate summary from the runs on disk.

python gates.py --stage day1 [--resamples 2000]

Day 1 reads seed 0 of the kfold x86 arms (R0 TF-IDF baseline, R2 tfidf / mean, R4 shuffle, R5 untyped, R7 typed GNN):
a pooled table, the preliminary gate-A verdict of PLAN_GNN.md section 5, and the stats.py paired differences. An arm
with missing folds is reported as missing with the folds that are present; nothing is dropped silently.
"""
import argparse
from functools import lru_cache

import numpy as np

import stats

SEED = 0
ARMS = (('R0 TF-IDF', 'tfidf'), ('R2 tfidf', 'kfold_x86_gnn-noedges'), ('R2 mean', 'kfold_x86_gnn-noedges-enc-mean'),
        ('R4 shuffle', 'kfold_x86_gnn-shuffle'), ('R5 untyped', 'kfold_x86_gnn-untyped'), ('R7 typed', 'kfold_x86_gnn'))
GROUP = dict(ARMS)
PAIRS = (('R2 tfidf', 'R4 shuffle'), ('R2 tfidf', 'R7 typed'), ('R4 shuffle', 'R7 typed'), ('R5 untyped', 'R7 typed'),
         ('R0 TF-IDF', 'R2 tfidf'))
SELECTIONS = (('f1', 'F1 checkpoint (macro-F1 at 0.5)'), ('auc', 'AUC checkpoint (macro-F1 at validation threshold)'))
LABEL = 'seed 0 only, not the pre-registered test'
AUC_MARGIN, F1_MARGIN = 0.01, 0.02


@lru_cache(maxsize=None)
def pooled(group, selection):
    """(auc, pr_auc, macro_f1) of the seed-0 pooled predictions, or a string saying what is missing."""
    complete, partial = stats.load_group(group, selection)
    if SEED in complete:
        p = complete[SEED]
        order, grp = stats.ranking(p)
        return stats.metrics_weighted(p, order, grp, np.ones(len(p.y)))
    present = partial.get(SEED, [])
    if group in stats.BASELINES and selection == 'auc' and not present and SEED in partial:
        return 'missing (baseline has no validation threshold; re-run baselines.py)'
    other = f'; complete seeds {sorted(complete)}' if complete else ''
    return f'missing (seed {SEED} folds present {present} of 0-4{other})'


def table():
    head = ['| arm | group | ' + ' | '.join(f'{rule} {m}' for rule, _ in SELECTIONS for m in ('AUC', 'PR-AUC', 'macro-F1'))
            + ' |', '|' + '---|' * (2 + 3 * len(SELECTIONS))]
    rows = []
    for arm, group in ARMS:
        cells = []
        for rule, _ in SELECTIONS:
            r = pooled(group, rule)
            cells += [f'{v:.3f}' for v in r] if not isinstance(r, str) else [r] + [''] * 2
        rows.append(f'| {arm} | {group} | ' + ' | '.join(cells) + ' |')
    return head + rows


def gate_a():
    """R2 tfidf passes when pooled AUC >= R0 - 0.01 and macro-F1 >= R0 - 0.02, per selection rule."""
    lines = []
    for rule, title in SELECTIONS:
        r0, r2 = pooled(GROUP['R0 TF-IDF'], rule), pooled(GROUP['R2 tfidf'], rule)
        if isinstance(r0, str) or isinstance(r2, str):
            lines.append(f'- {title}: cannot assess. R0: {r0 if isinstance(r0, str) else "ok"}; '
                         f'R2 tfidf: {r2 if isinstance(r2, str) else "ok"}.')
            continue
        ok_auc, ok_f1 = r2[0] >= r0[0] - AUC_MARGIN, r2[2] >= r0[2] - F1_MARGIN
        lines.append(f"- {title}: {'PASS' if ok_auc and ok_f1 else 'FAIL'}. AUC {r2[0]:.3f} vs R0 {r0[0]:.3f} - "
                     f"{AUC_MARGIN} = {r0[0] - AUC_MARGIN:.3f} ({'ok' if ok_auc else 'below'}); macro-F1 {r2[2]:.3f} vs "
                     f"R0 {r0[2]:.3f} - {F1_MARGIN} = {r0[2] - F1_MARGIN:.3f} ({'ok' if ok_f1 else 'below'}).")
    return lines


def day1(resamples):
    out = [f'# Gate summary, stage day1 ({LABEL})', '', '## Pooled kfold x86, seed 0', ''] + table() + ['']
    out += [f'## Gate A preliminary verdict ({LABEL})', '',
            'R2 `--encoder tfidf` against R0 TF-IDF (PLAN_GNN.md section 5): pooled AUC >= R0 - 0.01 and macro-F1 at '
            'the validation threshold >= R0 - 0.02. The AUC-checkpoint line is the one that applies the validation '
            'threshold; the F1-checkpoint line scores at 0.5. This is one seed and cannot pass the pre-registered '
            'rule.', ''] + gate_a() + ['']
    out += [f'## Paired differences, second minus first ({LABEL})', '']
    for rule, _ in SELECTIONS:
        for a, b in PAIRS:
            try:
                result, info = stats.compare(GROUP[a], GROUP[b], rule, resamples, seeds=[SEED])
                out += [f'**{a} vs {b}**', '', stats.fmt_block(GROUP[a], GROUP[b], rule, resamples, result, info), '']
            except stats.MissingGroup as e:
                out += [f'**{a} vs {b}** (selection {rule}): missing: {e}', '']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['day1'], required=True)
    ap.add_argument('--resamples', type=int, default=2000)
    args = ap.parse_args()
    print('\n'.join(day1(args.resamples)))


if __name__ == '__main__':
    main()
