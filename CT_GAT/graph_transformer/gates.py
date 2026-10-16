"""Gate summary from the runs on disk.

python gates.py --stage day1|day1b [--resamples 2000]

Day 1 reads seed 0 of the kfold x86 arms (R0 TF-IDF baseline, R2 tfidf / mean, R4 shuffle, R5 untyped, R7 typed GNN):
a pooled table, the preliminary gate-A verdict of PLAN_GNN.md section 5, and the stats.py paired differences. An arm
with missing folds is reported as missing with the folds that are present; nothing is dropped silently.

Day 1b reads the extra arms of step 5b (direction, augmentation, masking, depth, train fraction): the pooled table of
every kfold x86 GNN group on disk, the AUC-selection paired differences of each new arm against R7 and R2 tfidf, and
the learning curve (train fraction 0.25 / 0.5 / 1.0 for R7 and TF-IDF).
"""
import argparse
from functools import lru_cache

import numpy as np

import stats
from config import RunConfig

SEED = 0
ARMS = (('R0 TF-IDF', 'tfidf'), ('R2 tfidf', 'kfold_x86_gnn-noedges'), ('R2 mean', 'kfold_x86_gnn-noedges-enc-mean'),
        ('R4 shuffle', 'kfold_x86_gnn-shuffle'), ('R5 untyped', 'kfold_x86_gnn-untyped'), ('R7 typed', 'kfold_x86_gnn'))
GROUP = dict(ARMS)
PAIRS = (('R2 tfidf', 'R4 shuffle'), ('R2 tfidf', 'R7 typed'), ('R4 shuffle', 'R7 typed'), ('R5 untyped', 'R7 typed'),
         ('R0 TF-IDF', 'R2 tfidf'))
SELECTIONS = (('f1', 'F1 checkpoint (macro-F1 at 0.5)'), ('auc', 'AUC checkpoint (macro-F1 at validation threshold)'))
LABEL = 'seed 0 only, not the pre-registered test'
AUC_MARGIN, F1_MARGIN = 0.01, 0.02
NEW_ARMS = tuple(f'kfold_x86_gnn-{v}' for v in ('mtok15', 'dfn5-dlib40', 'undir', 'fwd', 'dfn15', 'tf50', 'tf25',
                                                'mblk15', 'dedge20', 'flip20', 'mtok50', 'bwd', 'L8'))
REFERENCES = (('R7 typed', 'kfold_x86_gnn'), ('R2 tfidf', 'kfold_x86_gnn-noedges'))
CURVE = ((0.25, 'tf25'), (0.5, 'tf50'), (1.0, ''))


@lru_cache(maxsize=None)
def pooled(group, selection):
    """(auc, pr_auc, macro_f1) of the seed-0 pooled predictions, or a string saying what is missing."""
    complete, partial = stats.load_group(group, selection)
    if SEED in complete:
        p = complete[SEED]
        order, grp = stats.ranking(p)
        return stats.metrics_weighted(p, order, grp, np.ones(len(p.y)))
    present = partial.get(SEED, [])
    if stats.baseline_method(group) and selection == 'auc' and not present and SEED in partial:
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


def gnn_groups():
    """Every kfold x86 GNN group with a run directory, sorted."""
    found = set()
    for d in stats.RUNS.iterdir():
        if (d / 'config.json').exists():
            cfg = RunConfig.load(d / 'config.json')
            if cfg.model == 'gnn' and cfg.protocol == 'kfold' and cfg.arch == 'x86':
                found.add(cfg.group)
    return sorted(found)


def metric_cells(r):
    return [f'{v:.3f}' for v in r] if not isinstance(r, str) else [r] + [''] * 2


def learning_curve():
    """Train fraction x {R7, TF-IDF}: AUC checkpoint at its validation threshold (TF-IDF at its own)."""
    head = ['| train fraction | model | group | AUC | PR-AUC | macro-F1 |', '|---|---|---|---|---|---|']
    rows = []
    for p, suffix in CURVE:
        for name, base in (('R7 typed', 'kfold_x86_gnn'), ('TF-IDF', 'tfidf')):
            group = f'{base}-{suffix}' if suffix else base
            rows.append(f'| {p} | {name} | {group} | ' + ' | '.join(metric_cells(pooled(group, 'auc'))) + ' |')
    return head + rows


def day1b(resamples):
    out = [f'# Gate summary, stage day1b ({LABEL})', '', '## Pooled kfold x86 GNN groups, seed 0', '']
    groups = gnn_groups()
    head = ['| group | ' + ' | '.join(f'{rule} {m}' for rule, _ in SELECTIONS for m in ('AUC', 'PR-AUC', 'macro-F1'))
            + ' |', '|' + '---|' * (1 + 3 * len(SELECTIONS))]
    rows = [f'| {g} | ' + ' | '.join(c for rule, _ in SELECTIONS for c in metric_cells(pooled(g, rule))) + ' |'
            for g in groups]
    out += head + rows + ['', f'Selections: {"; ".join(f"{r} = {t}" for r, t in SELECTIONS)}.', '']
    out += [f'## Paired differences, new arm minus reference, AUC selection ({LABEL})', '']
    for arm in NEW_ARMS:
        for ref, ref_group in REFERENCES:
            try:
                result, info = stats.compare(ref_group, arm, 'auc', resamples, seeds=[SEED])
                out += [f'**{arm} vs {ref}**', '', stats.fmt_block(ref_group, arm, 'auc', resamples, result, info), '']
            except stats.MissingGroup as e:
                out += [f'**{arm} vs {ref}** (selection auc): missing: {e}', '']
    out += [f'## Learning curve: training fraction ({LABEL})', '',
            'AUC checkpoint at its validation threshold for R7; the TF-IDF logistic regression at its validation '
            'threshold. Test sets are identical across fractions.', ''] + learning_curve() + ['']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['day1', 'day1b'], required=True)
    ap.add_argument('--resamples', type=int, default=2000)
    args = ap.parse_args()
    print('\n'.join({'day1': day1, 'day1b': day1b}[args.stage](args.resamples)))


if __name__ == '__main__':
    main()
