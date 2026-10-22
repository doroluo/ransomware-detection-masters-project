"""The tables every gate stage is made of: the pooled seed-0 row of a group, the paired-difference tables, the operating points.

Moved unchanged from the single-file `gates.py` of the 3-block-level-counting branch; the stage definitions that call them
live in `experiments/stages/`, the layouts that arrange them in `gate_layouts.py`, the engine in `gates.py`.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from s04_evaluation import stats
from s04_evaluation.stage import LABEL, SEED, signed_ci

SELECTIONS = (('f1', 'F1 checkpoint (macro-F1 at 0.5)'), ('auc', 'AUC checkpoint (macro-F1 at validation threshold)'))


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


def metric_cells(r):
    return [f'{v:.3f}' for v in r] if not isinstance(r, str) else [r] + [''] * 2


def paired_stage(title, intro, arms, pairs, resamples, seeds=(SEED,), after_pooled=(), rule=None):
    """Pooled table of `arms` ((label, group), AUC checkpoint, seed 0) and the paired differences of `pairs` (labels,
    second minus first) over the `seeds` complete in both arms. A missing arm or fold prints as missing. `after_pooled`: lines
    printed right after the pooled table. `rule(first, second, result, info)`: one more column of the paired table."""
    group = dict(arms)
    out = [f'# Gate summary, stage {title} ({LABEL})', '', intro, '', '## Pooled, AUC checkpoint', '',
           '| arm | group | AUC | PR-AUC | macro-F1 |', '|---|---|---|---|---|']
    for arm, g in arms:
        out.append(f'| {arm} | {g} | ' + ' | '.join(metric_cells(pooled(g, 'auc'))) + ' |')
    out += list(after_pooled)
    out += ['', '## Paired differences, second minus first', '',
            '| first | second | dAUC [95% CI] | dmacro-F1 [95% CI] |' + (' rule |' if rule else ''),
            '|---|---|---|---|' + ('---|' if rule else '')]
    blocks = []
    for a, b in pairs:
        try:
            result, info = stats.compare(group[a], group[b], 'auc', resamples, seeds=list(seeds))
        except stats.MissingGroup as e:
            out.append(f'| {a} | {b} | missing | missing |' + (' missing |' if rule else ''))
            blocks += [f'**{b} vs {a}** (selection auc): missing: {e}', '']
            continue
        out.append(f"| {a} | {b} | {signed_ci(result['auc'])} | {signed_ci(result['macro_f1'])} |"
                   + (f' {rule(a, b, result, info)} |' if rule else ''))
        blocks += [stats.fmt_block(group[a], group[b], 'auc', resamples, result, info), '']
    return out + [''] + blocks


def operating_points(arms):
    """Markdown table lines: seed-0 pooled AUC checkpoint at its validation threshold, per arm, with accuracy, precision,
    recall (ransomware = positive class) and the confusion matrix. A missing arm prints the reason pooled() gives."""
    out = ['', 'Operating point of each arm, seed 0 pooled over the five folds, AUC checkpoint at its validation threshold '
           '(TF-IDF and the header-imports LR: their own validation threshold). Ransomware is the positive class; TP / FN '
           'are ransomware binaries called ransomware / goodware, FP / TN goodware binaries called ransomware / goodware.', '',
           '| arm | AUC | PR-AUC | macro-F1 | accuracy | precision | recall | TP | FN | FP | TN |',
           '|---|---|---|---|---|---|---|---|---|---|---|']
    for arm, g in arms:
        complete, _ = stats.load_group(g, 'auc')
        if SEED not in complete:
            out.append(f'| {arm} | {pooled(g, "auc")} |' + ' |' * 9)
            continue
        p = complete[SEED]
        tp, fn = int(((p.y == 1) & (p.pred == 1)).sum()), int(((p.y == 1) & (p.pred == 0)).sum())
        fp, tn = int(((p.y == 0) & (p.pred == 1)).sum()), int(((p.y == 0) & (p.pred == 0)).sum())
        ratio = lambda a, b: a / b if b else 0.0
        cells = ([f'{v:.3f}' for v in pooled(g, 'auc')]
                 + [f'{ratio(tp + tn, len(p.y)):.3f}', f'{ratio(tp, tp + fp):.3f}', f'{ratio(tp, tp + fn):.3f}']
                 + [str(v) for v in (tp, fn, fp, tn)])
        out.append(f'| {arm} | ' + ' | '.join(cells) + ' |')
    return out


def match_rate_line(build, name):
    """One line with the per-label match rates of the pipeline-B build in `build`, called `name` in the line
    (experiments/team_pipelines/match_rates.py), or why it cannot be given: no build, or one whose build_summary.json does not
    account for all 1810 binaries."""
    from experiments.team_pipelines import match_rates
    summary = Path(build) / 'build_summary.json'
    if not summary.is_file():
        return f'Match rates of `{name}`: missing (no {summary.name} under {build}).'
    done = json.loads(summary.read_text(encoding='utf-8'))
    if done.get('processed') != done.get('total') or done.get('total') != 1810:
        return (f"Match rates of `{name}`: missing (build incomplete: {done.get('processed')} of {done.get('total')} "
                'binaries processed).')
    rates = match_rates.summarize(match_rates.sample_rates(Path(build)))
    parts = [f"{name} {s['n']} binaries, mean {s['mean_match']:.3f}, below 50% {s['n_below_50']} ({100 * s['share_below_50']:.1f}%), "
             f"below 90% {s['n_below_90']} ({100 * s['share_below_90']:.1f}%), exactly 0 {100 * s['share_zero']:.1f}%"
             for name, s in rates.items()]
    return f'Match rates of `{name}` (share of each binary\'s selected instructions paired with a listing line): ' + '; '.join(parts) + '.'


def match_rate_lines(builds):
    """The lines a stage's `options['match_rate_builds']` asks for: `match_rate_line` of each (name, folder under the run
    directory), each after a blank line. The folder is resolved when the lines are made, with the run directory `stats` reads."""
    out = []
    for name, folder in builds:
        out += ['', match_rate_line(stats.RUNS / folder, name)]
    return out


EXTRAS = {'operating_points': operating_points}      # Stage.extra -> a function of the stage's arms, printed after the pooled table
