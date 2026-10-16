"""Gate summary from the runs on disk.

python gates.py --stage day1|day1b|v3 [--resamples 2000]

Day 1 reads seed 0 of the kfold x86 arms (R0 TF-IDF baseline, R2 tfidf / mean, R4 shuffle, R5 untyped, R7 typed GNN):
a pooled table, the preliminary gate-A verdict of PLAN_GNN.md section 5, and the stats.py paired differences. An arm
with missing folds is reported as missing with the folds that are present; nothing is dropped silently.

Day 1b reads the extra arms of step 5b (direction, augmentation, masking, depth, train fraction): the pooled table of
every kfold x86 GNN group on disk, the AUC-selection paired differences of each new arm against R7 and R2 tfidf, and
the learning curve (train fraction 0.25 / 0.5 / 1.0 for R7 and TF-IDF).

Stage v3 reads seed 0 of the kfold_v3 arms (R0, R2, R4, R5, R7) on the combined cache for arch all and arch matched: the
pooled table under both checkpoint rules with per-architecture recall at the validation threshold and both floors
(majority, x86 rule), the paired AUC-selection differences R2-R7, R4-R7, R5-R7 and R0-R2, and a one-line verdict per
arch on whether R7 beats R2 and R4 under the pre-registered rule.
"""
import argparse
from functools import lru_cache

import numpy as np

import report
import stats
from config import RunConfig
from data import MAX_BINARY_BLOCKS

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
V3_ARCHES = ('all', 'matched')
V3_ARMS = (('R0 TF-IDF', 'tfidf'), ('R2 no-edges', 'gnn-noedges'), ('R4 shuffle', 'gnn-shuffle'),
           ('R5 untyped', 'gnn-untyped'), ('R7 typed', 'gnn'))
V3_PAIRS = (('R2 no-edges', 'R7 typed'), ('R4 shuffle', 'R7 typed'), ('R5 untyped', 'R7 typed'),
            ('R0 TF-IDF', 'R2 no-edges'))
V3_RECALLS = (('R x86', 1, 'x86'), ('R x64', 1, 'x64'), ('G x86', 0, 'x86'), ('G x64', 0, 'x64'))
MIN_AUC, MIN_F1 = 0.01, 0.02


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


def v3_group(arch, variant):
    return variant if stats.baseline_method(variant) else f'kfold_v3_{arch}_{variant}'


@lru_cache(maxsize=None)
def v3_cell(group, selection, arch):
    """[AUC, PR-AUC, macro-F1, R x86, R x64, G x86, G x64] of the seed-0 pooled kfold_v3 predictions, or a string saying
    what is missing. A baseline group resolves to the kfold_v3 baselines of `arch`."""
    complete, partial = stats.load_group(group, selection, ('kfold_v3', arch))
    if SEED not in complete:
        present = partial.get(SEED, [])
        other = f'; complete seeds {sorted(complete)}' if complete else ''
        return f'missing (seed {SEED} folds present {present} of 0-4{other})'
    p = complete[SEED]
    order, grp = stats.ranking(p)
    recalls = []
    for _, label, a in V3_RECALLS:
        mask = (p.y == label) & (p.arch == a)
        recalls.append(float(np.mean(p.pred[mask] == label)) if mask.any() else float('nan'))
    return [*stats.metrics_weighted(p, order, grp, np.ones(len(p.y))), *recalls]


def v3_floor_cells(arch):
    keys = ('auc', 'pr_auc', 'macro_f1', 'recall_R_x86', 'recall_R_x64', 'recall_G_x86', 'recall_G_x64')
    return {name: [s[k] for k in keys] for name, s in report.v3_floors(arch)[0].items()}


def v3_table(arch):
    metrics = ('AUC', 'PR-AUC', 'macro-F1')
    head = ['| arm | group | ' + ' | '.join(f'{rule} {m}' for rule in ('F1 ckpt, 0.5', 'AUC ckpt, val thr') for m in metrics)
            + ' | ' + ' | '.join(f'{t} recall' for t, _, _ in V3_RECALLS) + ' |',
            '|' + '---|' * (2 + 6 + len(V3_RECALLS))]
    fmt = lambda cells: [c if isinstance(c, str) else f'{c:.3f}' for c in cells]
    rows = []
    for arm, variant in V3_ARMS:
        group = v3_group(arch, variant)
        f1, auc = v3_cell(group, 'f1', arch), v3_cell(group, 'auc', arch)
        cells = (f1[:3] if not isinstance(f1, str) else [f1, '', '']) + (auc if not isinstance(auc, str)
                                                                         else [auc] + [''] * 6)
        rows.append(f'| {arm} | {group} | ' + ' | '.join(fmt(cells)) + ' |')
    for name, cells in v3_floor_cells(arch).items():
        rows.append(f'| floor: {name} | - | ' + ' | '.join(fmt(cells[:3] + cells)) + ' |')
    return head + rows


def beats(r):
    """The pre-registered minimum effect on one paired comparison (b - a): dAUC >= 0.01 and dmacro-F1 >= 0.02 with
    both 95% intervals above zero."""
    return all(r[k]['diff'] >= m and r[k]['lo'] > 0 for k, m in (('auc', MIN_AUC), ('macro_f1', MIN_F1)))


def v3_truncation(arch):
    """One line: R7's test binaries cut by the per-binary block cap (AUC checkpoint, validation threshold) and the
    ransomware / goodware recall on the cut and the uncut ones."""
    group = v3_group(arch, 'gnn')
    complete, partial = stats.load_group(group, 'auc', ('kfold_v3', arch))
    if SEED not in complete:
        return f"- Arch {arch}: truncation not assessed, R7 ({group}) has no complete seed {SEED}."
    p = complete[SEED]
    if (p.truncated < 0).any():
        return (f"- Arch {arch}: truncation not assessed, R7 predictions lack n_blocks_kept / n_blocks_total "
                "(run predates the cap).")

    def cell(mask):
        r, g = mask & (p.y == 1), mask & (p.y == 0)
        recall = lambda m, label: f'{np.mean(p.pred[m] == label):.3f} (n={int(m.sum())})' if m.any() else 'n/a (n=0)'
        return f"R recall {recall(r, 1)}, G recall {recall(g, 0)}"
    cut = p.truncated == 1
    return (f"- Arch {arch}: R7 truncated {int(cut.sum())} of {len(p.y)} test binaries (n_blocks_kept < n_blocks_total, "
            f"cap {MAX_BINARY_BLOCKS}); truncated: {cell(cut)}; untruncated: {cell(~cut)}.")


def v3_verdict(arch, results):
    parts, missing = [], []
    for ref in ('R2 no-edges', 'R4 shuffle'):
        r = results[ref, 'R7 typed']
        if isinstance(r, str):
            missing.append(f'{ref}: {r}')
            continue
        parts.append(f"{'beats' if beats(r) else 'does not beat'} {ref} (dAUC {r['auc']['diff']:+.3f} "
                     f"[{r['auc']['lo']:+.3f}, {r['auc']['hi']:+.3f}], dmacro-F1 {r['macro_f1']['diff']:+.3f} "
                     f"[{r['macro_f1']['lo']:+.3f}, {r['macro_f1']['hi']:+.3f}])")
    if missing:
        return f"- Arch {arch}: cannot assess, missing: {'; '.join(missing)} ({LABEL})."
    both = all(p.startswith('beats') for p in parts)
    return (f"- Arch {arch}: R7 {'beats both R2 and R4' if both else 'does not beat both R2 and R4'}: "
            f"{'; '.join(parts)} ({LABEL}).")


def v3(resamples):
    out = [f'# Gate summary, stage v3 ({LABEL})', '',
           'Combined cache, kfold_v3 (family-holdout v3 folds 0-4), seed 0 pooled over the five folds. The F1 checkpoint '
           "is scored at 0.5; the AUC checkpoint at the threshold that maximises validation macro-F1 (TF-IDF at its own "
           "per fold); recalls are at that threshold. A missing arm or fold is stated in its row. Floors: the majority "
           "class of each fold's training manifest, and the x86 rule (ransomware iff x86).", '']
    verdicts = []
    for arch in V3_ARCHES:
        pooled_floors, per_fold = report.v3_floors(arch)
        out += [f'## Arch {arch}: pooled table', ''] + v3_table(arch) + ['']
        out += ['x86-rule macro-F1 per fold: ' + ', '.join(f"{f['x86_rule']['macro_f1']:.3f}" for f in per_fold)
                + f" (pooled {pooled_floors['x86_rule']['macro_f1']:.3f}).", '']
        out += [v3_truncation(arch), '']
        out += [f'### Arch {arch}: paired differences, second minus first, AUC selection ({LABEL})', '']
        results = {}
        for a, b in V3_PAIRS:
            ga, gb = (v3_group(arch, dict(V3_ARMS)[x]) for x in (a, b))
            try:
                result, info = stats.compare(ga, gb, 'auc', resamples, seeds=[SEED], protocol='kfold_v3', arch=arch)
                results[a, b] = result
                out += [f'**{a} vs {b}**', '', stats.fmt_block(ga, gb, 'auc', resamples, result, info), '']
            except stats.MissingGroup as e:
                results[a, b] = str(e)
                out += [f'**{a} vs {b}** (selection auc): missing: {e}', '']
        verdicts.append(v3_verdict(arch, results))
    out += ['## Verdict: does R7 beat R2 and R4?', '',
            f'Pre-registered rule (PLAN_GNN.md section 5), applied to the AUC selection: dAUC >= {MIN_AUC} and '
            f'dmacro-F1 >= {MIN_F1} pooled, both 95% intervals excluding zero. With one seed it cannot pass the rule; '
            'these lines only say which way seed 0 points.', ''] + verdicts + ['']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['day1', 'day1b', 'v3'], required=True)
    ap.add_argument('--resamples', type=int, default=2000)
    args = ap.parse_args()
    print('\n'.join({'day1': day1, 'day1b': day1b, 'v3': v3}[args.stage](args.resamples)))


if __name__ == '__main__':
    main()
