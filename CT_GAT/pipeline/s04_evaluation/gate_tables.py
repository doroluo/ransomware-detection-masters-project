"""The tables every gate stage is made of: the pooled seed-0 row of a group, the paired-difference tables, the operating points.

Moved unchanged from the single-file `gates.py` of the 3-block-level-counting branch; the stage definitions that call them
live in `experiments/stages/`, the engine in `gates.py`.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from s04_evaluation import metrics, report, stats
from s04_evaluation.stage import DEFAULT_SCOPE, LABEL, SEED, signed_ci


@lru_cache(maxsize=None)
def pooled(group, selection, scope=DEFAULT_SCOPE):
    """(auc, pr_auc, macro_f1) of the seed-0 pooled predictions, or a string saying what is missing. `scope` = (protocol,
    arch) is the dataset a baseline group (`tfidf`) is read for; the run groups carry theirs in the name."""
    complete, partial = stats.load_group(group, selection, scope)
    if SEED in complete:
        p = complete[SEED]
        order, grp = stats.ranking(p)
        return stats.metrics_weighted(p, order, grp, np.ones(len(p.y)))
    present = partial.get(SEED, [])
    if stats.baseline_method(group) and selection == 'auc' and not present and SEED in partial:
        return 'missing (baseline has no validation threshold)'
    other = f'; complete seeds {sorted(complete)}' if complete else ''
    return f'missing (seed {SEED} folds present {present} of 0-4{other})'


def metric_cells(r):
    return [f'{v:.3f}' for v in r] if not isinstance(r, str) else [r] + [''] * 2


def paired_table(group, pairs, resamples, seeds, rule, scope=None, balanced=False):
    """(table lines, detail blocks) of the paired differences of `pairs` (labels, second minus first). `scope` = (protocol,
    arch) is the dataset a baseline group is read for (None: what stats.compare derives from the other group of the pair,
    else kfold x86); `balanced`: every sample weighted N / (C n_cell) by its class x arch cell."""
    protocol, arch = scope or (None, None)
    out = ['| first | second | dAUC [95% CI] | dmacro-F1 [95% CI] |' + (' rule |' if rule else ''),
           '|---|---|---|---|' + ('---|' if rule else '')]
    blocks = []
    for a, b in pairs:
        try:
            result, info = stats.compare(group[a], group[b], 'auc', resamples, seeds=list(seeds), protocol=protocol, arch=arch,
                                         balanced=balanced)
        except stats.MissingGroup as e:
            out.append(f'| {a} | {b} | missing | missing |' + (' missing |' if rule else ''))
            blocks += [f'**{b} vs {a}** (selection auc): missing: {e}', '']
            continue
        out.append(f"| {a} | {b} | {signed_ci(result['auc'])} | {signed_ci(result['macro_f1'])} |"
                   + (f' {rule(a, b, result, info)} |' if rule else ''))
        blocks += [stats.fmt_block(group[a], group[b], 'auc', resamples, result, info), '']
    return out, blocks


def paired_stage(title, intro, arms, pairs, resamples, seeds=(SEED,), after_pooled=(), rule=None, scope=DEFAULT_SCOPE,
                 balanced=False):
    """Pooled table of `arms` ((label, group), AUC checkpoint, seed 0) and the paired differences of `pairs` (labels,
    second minus first) over the `seeds` complete in both arms. A missing arm or fold prints as missing. `after_pooled`: lines
    printed right after the pooled table. `rule(first, second, result, info)`: one more column of the paired table. `scope` =
    (protocol, arch): the dataset the baseline arms are read for (the default, kfold x86, leaves the output of a stage that
    gives none as it was). `balanced`: a second paired table, with every sample weighted N / (C n_cell) by its class x
    architecture cell."""
    group = dict(arms)
    out = [f'# Gate summary, stage {title} ({LABEL})', '', intro, '', '## Pooled, AUC checkpoint', '',
           '| arm | group | AUC | PR-AUC | macro-F1 |', '|---|---|---|---|---|']
    for arm, g in arms:
        out.append(f'| {arm} | {g} | ' + ' | '.join(metric_cells(pooled(g, 'auc', scope))) + ' |')
    out += list(after_pooled)
    table, blocks = paired_table(group, pairs, resamples, seeds, rule, None if scope == DEFAULT_SCOPE else scope)
    out += ['', '## Paired differences, second minus first', ''] + table
    if balanced:
        table_b, blocks_b = paired_table(group, pairs, resamples, seeds, rule, scope, balanced=True)
        out += ['', '## Paired differences, cell-balanced', '',
                'Every test binary is weighted N / (C n_cell) by its class x architecture cell, so ransomware x86, '
                'ransomware x64, goodware x86 and goodware x64 weigh alike; the x86 rule scores 0.5 under these weights.', ''
                ] + table_b
        blocks += ['### Cell-balanced differences, detail', ''] + blocks_b
    return out + [''] + blocks


def operating_points(arms, scope=DEFAULT_SCOPE):
    """Markdown table lines: seed-0 pooled AUC checkpoint at its validation threshold, per arm, with accuracy, precision,
    recall (ransomware = positive class) and the confusion matrix. A missing arm prints the reason pooled() gives."""
    out = ['', 'Operating point of each arm, seed 0 pooled over the five folds, AUC checkpoint at its validation threshold '
           '(TF-IDF and the header-imports LR: their own validation threshold). Ransomware is the positive class; TP / FN '
           'are ransomware binaries called ransomware / goodware, FP / TN goodware binaries called ransomware / goodware.', '',
           '| arm | AUC | PR-AUC | macro-F1 | accuracy | precision | recall | TP | FN | FP | TN |',
           '|---|---|---|---|---|---|---|---|---|---|---|']
    for arm, g in arms:
        complete, _ = stats.load_group(g, 'auc', scope)
        if SEED not in complete:
            out.append(f'| {arm} | {pooled(g, "auc", scope)} |' + ' |' * 9)
            continue
        p = complete[SEED]
        tp, fn = int(((p.y == 1) & (p.pred == 1)).sum()), int(((p.y == 1) & (p.pred == 0)).sum())
        fp, tn = int(((p.y == 0) & (p.pred == 1)).sum()), int(((p.y == 0) & (p.pred == 0)).sum())
        ratio = lambda a, b: a / b if b else 0.0
        cells = ([f'{v:.3f}' for v in pooled(g, 'auc', scope)]
                 + [f'{ratio(tp + tn, len(p.y)):.3f}', f'{ratio(tp, tp + fp):.3f}', f'{ratio(tp, tp + fn):.3f}']
                 + [str(v) for v in (tp, fn, fp, tn)])
        out.append(f'| {arm} | ' + ' | '.join(cells) + ' |')
    return out


BUILD_TOTAL = {('kfold', 'x86'): 1810, ('kfold', 'all'): 2411, ('kfold_v3', 'all'): 5652, ('kfold_v3', 'matched'): 5652}
# binaries of the build a stage's scope reads (a matched run reads the all build)


def match_rate_line(build, name, scope=DEFAULT_SCOPE, total=None):
    """One line with the per-label match rates of the pipeline-B build in `build`, called `name` in the line
    (experiments/team_pipelines/match_rates.py), or why it cannot be given: no build, or one whose build_summary.json does not
    account for all the binaries of the scope (`BUILD_TOTAL`, or `total`). The kfold x86 builds hold 1810, with none excluded;
    for another scope `processed` plus the ids excluded on purpose (`--exclude`, `--max-instructions`) must equal the summary's
    total, which must be the scope's: the copy of an x86 build that `build_b --extend` is still extending holds the summary of
    the x86 build (1810 of 1810)."""
    from experiments.team_pipelines import match_rates
    summary = Path(build) / 'build_summary.json'
    if not summary.is_file():
        return f'Match rates of `{name}`: missing (no {summary.name} under {build}).'
    done = json.loads(summary.read_text(encoding='utf-8'))
    total = BUILD_TOTAL[tuple(scope)] if total is None else total
    complete = (done.get('processed') == done.get('total') == 1810 if scope == DEFAULT_SCOPE
                else done.get('total') == total and (done.get('processed') or 0) + len(done.get('excluded') or {}) == total)
    if not complete:
        return (f"Match rates of `{name}`: missing (build incomplete: {done.get('processed')} of {done.get('total')} "
                'binaries processed).')
    rates = match_rates.summarize(match_rates.sample_rates(Path(build)))
    parts = [f"{name} {s['n']} binaries, mean {s['mean_match']:.3f}, below 50% {s['n_below_50']} ({100 * s['share_below_50']:.1f}%), "
             f"below 90% {s['n_below_90']} ({100 * s['share_below_90']:.1f}%), exactly 0 {100 * s['share_zero']:.1f}%"
             for name, s in rates.items()]
    return f'Match rates of `{name}` (share of each binary\'s selected instructions paired with a listing line): ' + '; '.join(parts) + '.'


def match_rate_lines(builds, scope=DEFAULT_SCOPE):
    """The lines a stage's `options['match_rate_builds']` asks for: `match_rate_line` of each (name, folder under the run
    directory), each after a blank line. The folder is resolved when the lines are made, with the run directory `stats` reads."""
    out = []
    for name, folder in builds:
        out += ['', match_rate_line(stats.RUNS / folder, name, tuple(scope))]
    return out


CELL_RECALLS = (('R x86', 1, 'x86'), ('R x64', 1, 'x64'), ('G x86', 0, 'x86'), ('G x64', 0, 'x64'))


def cell_row(group, scope):
    """[AUC, PR-AUC, macro-F1, R x86, R x64, G x86, G x64 recall, cell-balanced AUC, cell-balanced macro-F1] of the seed-0
    pooled predictions of the AUC checkpoint at its validation threshold (a baseline group is read for `scope`), or a string
    saying what is missing. A cell without a test binary has recall nan."""
    complete, _ = stats.load_group(group, 'auc', scope)
    if SEED not in complete:
        return pooled(group, 'auc', scope)                               # the reason the pooled table gives
    p = complete[SEED]
    order, grp = stats.ranking(p)
    recalls = [float(np.mean(p.pred[m] == label)) if (m := (p.y == label) & (p.arch == arch)).any() else float('nan')
               for _, label, arch in CELL_RECALLS]
    cb = metrics.cell_balanced(p.y, p.prob, p.pred, p.arch)
    return [*stats.metrics_weighted(p, order, grp, np.ones(len(p.y))), *recalls, cb['auc'], cb['macro_f1']]


def scope_floors(scope):
    """(pooled floors, per-fold floors) of the test manifests of `scope`, or None when the cache of its protocol is not
    built (a table prints that instead of failing)."""
    try:
        return report.scope_floors(*scope)
    except (FileNotFoundError, ValueError):
        return None


def cells(arms, scope=DEFAULT_SCOPE):
    """Markdown table lines: per arm the seed-0 pooled AUC checkpoint at its validation threshold with the recall of each class
    x architecture cell (ransomware = positive class) and the cell-balanced AUC and macro-F1, then the floors (majority class
    and the x86 rule) of the scope's test manifests and the x86 rule per fold. Read these rows before the pooled AUC when the
    architecture predicts the label (stage team_mendeley_all)."""
    protocol, arch = scope
    out = ['', f'## Per-architecture recalls and cell-balanced metrics ({protocol}, arch {arch})', '',
           'AUC checkpoint at its validation threshold (TF-IDF: its own validation threshold), seed 0 pooled over the five folds. '
           'The recalls are those of ransomware (R) and goodware (G) binaries of each architecture. The cell-balanced columns '
           'weight each non-empty class x architecture cell of the pooled test set equally; under them the x86 rule scores 0.5.', '',
           '| arm | group | AUC | PR-AUC | macro-F1 | ' + ' | '.join(f'{name} recall' for name, _, _ in CELL_RECALLS)
           + ' | cell-bal AUC | cell-bal macro-F1 |', '|' + '---|' * (2 + 3 + len(CELL_RECALLS) + 2)]
    fmt = lambda values: [c if isinstance(c, str) else f'{c:.3f}' for c in values]
    for label, group in arms:
        row = cell_row(group, scope)
        out.append(f'| {label} | {group} | ' + ' | '.join(fmt(row if not isinstance(row, str) else [row] + [''] * 8)) + ' |')
    floors = scope_floors(scope)
    if floors is None:
        return out + ['', f'Floors: missing (no cache for protocol {protocol}).']
    pooled_floors, per_fold = floors
    keys = ('auc', 'pr_auc', 'macro_f1', 'recall_R_x86', 'recall_R_x64', 'recall_G_x86', 'recall_G_x64')
    for name, s_ in pooled_floors.items():
        out.append(f'| floor: {name} | - | ' + ' | '.join(fmt([*(s_[k] for k in keys), s_['cell_balanced']['auc'],
                                                              s_['cell_balanced']['macro_f1']])) + ' |')
    return out + ['', 'x86-rule macro-F1 per fold: ' + ', '.join(f"{f['x86_rule']['macro_f1']:.3f}" for f in per_fold)
                  + f" (pooled {pooled_floors['x86_rule']['macro_f1']:.3f}).", '']


EXTRAS = {'operating_points': operating_points, 'cells': cells}      # Stage.extra -> a function of the stage's arms and scope, printed after the pooled table
