"""The four gate stages whose tables are not the generic 'paired' shape: day1, day1b, v3 and subgraph.

Each layout is `layout(stage, resamples) -> list of markdown lines`; the arms, pairs and constants it arranges come from the
stage definition in `experiments/stages/`. The code is that of the single-file `gates.py` of the 3-block-level-counting
branch, with the module constants replaced by the stage's data.
"""
import json
from functools import lru_cache

import numpy as np

from common.config import RunConfig
from s02_preprocess.subgraph import MAX_BINARY_BLOCKS
from s04_evaluation import metrics, report, stats
from s04_evaluation.gate_tables import SELECTIONS, metric_cells, pooled
from s04_evaluation.stage import LABEL, MIN_AUC, MIN_F1, NI_AUC, NI_F1, SEED, beats, non_inferior, signed_ci, x86_group

V3_RECALLS = (('R x86', 1, 'x86'), ('R x64', 1, 'x64'), ('G x86', 0, 'x86'), ('G x64', 0, 'x64'))


# ---- day1 ------------------------------------------------------------------------------------------------------------

def table(arms):
    head = ['| arm | group | ' + ' | '.join(f'{rule} {m}' for rule, _ in SELECTIONS for m in ('AUC', 'PR-AUC', 'macro-F1'))
            + ' |', '|' + '---|' * (2 + 3 * len(SELECTIONS))]
    rows = []
    for arm, group in arms:
        cells = []
        for rule, _ in SELECTIONS:
            r = pooled(group, rule)
            cells += [f'{v:.3f}' for v in r] if not isinstance(r, str) else [r] + [''] * 2
        rows.append(f'| {arm} | {group} | ' + ' | '.join(cells) + ' |')
    return head + rows


def gate_a(group, baseline, candidate, margins):
    """The candidate passes when pooled AUC >= baseline - margin and macro-F1 >= baseline - margin, per selection rule."""
    auc_margin, f1_margin = margins
    lines = []
    for rule, title in SELECTIONS:
        r0, r2 = pooled(group[baseline], rule), pooled(group[candidate], rule)
        if isinstance(r0, str) or isinstance(r2, str):
            lines.append(f'- {title}: cannot assess. R0: {r0 if isinstance(r0, str) else "ok"}; '
                         f'R2 tfidf: {r2 if isinstance(r2, str) else "ok"}.')
            continue
        ok_auc, ok_f1 = r2[0] >= r0[0] - auc_margin, r2[2] >= r0[2] - f1_margin
        lines.append(f"- {title}: {'PASS' if ok_auc and ok_f1 else 'FAIL'}. AUC {r2[0]:.3f} vs R0 {r0[0]:.3f} - "
                     f"{auc_margin} = {r0[0] - auc_margin:.3f} ({'ok' if ok_auc else 'below'}); macro-F1 {r2[2]:.3f} vs "
                     f"R0 {r0[2]:.3f} - {f1_margin} = {r0[2] - f1_margin:.3f} ({'ok' if ok_f1 else 'below'}).")
    return lines


def day1(stage, resamples):
    group = dict(stage.arms)
    baseline, candidate = stage.options['gate_a']
    out = [f'# Gate summary, stage {stage.name} ({LABEL})', '', '## Pooled kfold x86, seed 0', ''] + table(stage.arms) + ['']
    out += [f'## Gate A preliminary verdict ({LABEL})', '', stage.intro_text(resamples), ''] + gate_a(
        group, baseline, candidate, stage.options['margins']) + ['']
    out += [f'## Paired differences, second minus first ({LABEL})', '']
    for rule, _ in SELECTIONS:
        for a, b in stage.pairs:
            try:
                result, info = stats.compare(group[a], group[b], rule, resamples, seeds=[SEED])
                out += [f'**{a} vs {b}**', '', stats.fmt_block(group[a], group[b], rule, resamples, result, info), '']
            except stats.MissingGroup as e:
                out += [f'**{a} vs {b}** (selection {rule}): missing: {e}', '']
    return out


# ---- day1b -----------------------------------------------------------------------------------------------------------

def gnn_groups():
    """Every kfold x86 GNN group with a run directory, sorted."""
    found = set()
    for d in stats.RUNS.iterdir():
        if (d / 'config.json').exists():
            cfg = RunConfig.try_load(d / 'config.json')
            if cfg and cfg.model == 'gnn' and cfg.protocol == 'kfold' and cfg.arch == 'x86':
                found.add(cfg.group)
    return sorted(found)


def learning_curve(curve, models):
    """Train fraction x {R7, TF-IDF}: AUC checkpoint at its validation threshold (TF-IDF at its own)."""
    head = ['| train fraction | model | group | AUC | PR-AUC | macro-F1 |', '|---|---|---|---|---|---|']
    rows = []
    for p, suffix in curve:
        for name, base in models:
            group = f'{base}-{suffix}' if suffix else base
            rows.append(f'| {p} | {name} | {group} | ' + ' | '.join(metric_cells(pooled(group, 'auc'))) + ' |')
    return head + rows


def day1b(stage, resamples):
    out = [f'# Gate summary, stage {stage.name} ({LABEL})', '', '## Pooled kfold x86 GNN groups, seed 0', '']
    groups = gnn_groups()
    head = ['| group | ' + ' | '.join(f'{rule} {m}' for rule, _ in SELECTIONS for m in ('AUC', 'PR-AUC', 'macro-F1'))
            + ' |', '|' + '---|' * (1 + 3 * len(SELECTIONS))]
    rows = [f'| {g} | ' + ' | '.join(c for rule, _ in SELECTIONS for c in metric_cells(pooled(g, rule))) + ' |'
            for g in groups]
    out += head + rows + ['', f'Selections: {"; ".join(f"{r} = {t}" for r, t in SELECTIONS)}.', '']
    out += [f'## Paired differences, new arm minus reference, AUC selection ({LABEL})', '']
    for arm in stage.options['new_arms']:
        for ref, ref_group in stage.arms:
            try:
                result, info = stats.compare(ref_group, arm, 'auc', resamples, seeds=[SEED])
                out += [f'**{arm} vs {ref}**', '', stats.fmt_block(ref_group, arm, 'auc', resamples, result, info), '']
            except stats.MissingGroup as e:
                out += [f'**{arm} vs {ref}** (selection auc): missing: {e}', '']
    out += [f'## Learning curve: training fraction ({LABEL})', '',
            'AUC checkpoint at its validation threshold for R7; the TF-IDF logistic regression at its validation '
            'threshold. Test sets are identical across fractions.', ''] + learning_curve(
        stage.options['curve'], stage.options['curve_models']) + ['']
    return out


# ---- v3 --------------------------------------------------------------------------------------------------------------

def v3_group(arch, variant):
    return variant if stats.baseline_method(variant) else f'kfold_v3_{arch}_{variant}'


@lru_cache(maxsize=None)
def v3_cell(group, selection, arch):
    """[AUC, PR-AUC, macro-F1, R x86, R x64, G x86, G x64, cell-balanced AUC, cell-balanced macro-F1] of the seed-0
    pooled kfold_v3 predictions, or a string saying what is missing. A baseline group resolves to the kfold_v3 baselines of `arch`."""
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
    cb = metrics.cell_balanced(p.y, p.prob, p.pred, p.arch)
    return [*stats.metrics_weighted(p, order, grp, np.ones(len(p.y))), *recalls, cb['auc'], cb['macro_f1']]


def v3_floor_cells(arch):
    keys = ('auc', 'pr_auc', 'macro_f1', 'recall_R_x86', 'recall_R_x64', 'recall_G_x86', 'recall_G_x64')
    return {name: [*(s[k] for k in keys), s['cell_balanced']['auc'], s['cell_balanced']['macro_f1']]
            for name, s in report.v3_floors(arch)[0].items()}


def v3_table(arch, arms):
    metric_names = ('AUC', 'PR-AUC', 'macro-F1')
    rules = ('F1 ckpt, 0.5', 'AUC ckpt, val thr')
    head = ['| arm | group | ' + ' | '.join(f'{rule} {m}' for rule in rules for m in metric_names)
            + ' | ' + ' | '.join(f'{t} recall' for t, _, _ in V3_RECALLS)
            + ' | ' + ' | '.join(f'{rule} cell-bal {m}' for rule in rules for m in ('AUC', 'macro-F1')) + ' |',
            '|' + '---|' * (2 + 6 + len(V3_RECALLS) + 4)]
    fmt = lambda cells: [c if isinstance(c, str) else f'{c:.3f}' for c in cells]
    rows = []
    for arm, variant in arms:
        group = v3_group(arch, variant)
        f1, auc = v3_cell(group, 'f1', arch), v3_cell(group, 'auc', arch)
        cells = ((f1[:3] if not isinstance(f1, str) else [f1, '', ''])
                 + (auc[:7] if not isinstance(auc, str) else [auc] + [''] * 6)
                 + (f1[7:] if not isinstance(f1, str) else ['', ''])
                 + (auc[7:] if not isinstance(auc, str) else ['', '']))
        rows.append(f'| {arm} | {group} | ' + ' | '.join(fmt(cells)) + ' |')
    for name, cells in v3_floor_cells(arch).items():
        rows.append(f'| floor: {name} | - | ' + ' | '.join(fmt(cells[:3] + cells[:7] + cells[7:] + cells[7:])) + ' |')
    return head + rows


def v3_truncation(arch, arms):
    """One line: R7's test binaries cut by the per-binary block cap (AUC checkpoint, validation threshold) and the
    ransomware / goodware recall on the cut and the uncut ones."""
    group = v3_group(arch, dict(arms)['R7 typed'])
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


def v3_verdict(arch, results, suffix='', note=''):
    """One line on whether R7 beats R2 and R4 in `results`; `suffix` ' ab' reads the arch-balanced arms and `note`
    says which weighting the differences use."""
    parts, missing = [], []
    for name in ('R2 no-edges', 'R4 shuffle'):
        ref = name + suffix
        r = results[ref, 'R7 typed' + suffix]
        if isinstance(r, str):
            missing.append(f'{ref}: {r}')
            continue
        parts.append(f"{'beats' if beats(r) else 'does not beat'} {ref} (dAUC {r['auc']['diff']:+.3f} "
                     f"[{r['auc']['lo']:+.3f}, {r['auc']['hi']:+.3f}], dmacro-F1 {r['macro_f1']['diff']:+.3f} "
                     f"[{r['macro_f1']['lo']:+.3f}, {r['macro_f1']['hi']:+.3f}])")
    if missing:
        return f"- Arch {arch}: {note}cannot assess, missing: {'; '.join(missing)} ({LABEL})."
    both = all(p.startswith('beats') for p in parts)
    return (f"- Arch {arch}: {note}R7{suffix} {'beats both R2 and R4' if both else 'does not beat both R2 and R4'}: "
            f"{'; '.join(parts)} ({LABEL}).")


def v3_pairs(arch, pairs, resamples, balanced, arms):
    """(markdown lines, {(a, b): result or the missing reason}) of the paired AUC-selection differences."""
    out, results = [], {}
    for a, b in pairs:
        ga, gb = (v3_group(arch, dict(arms)[x]) for x in (a, b))
        try:
            result, info = stats.compare(ga, gb, 'auc', resamples, seeds=[SEED], protocol='kfold_v3', arch=arch,
                                         balanced=balanced)
            results[a, b] = result
            out += [f'**{a} vs {b}**', '', stats.fmt_block(ga, gb, 'auc', resamples, result, info), '']
        except stats.MissingGroup as e:
            results[a, b] = str(e)
            out += [f'**{a} vs {b}** (selection auc): missing: {e}', '']
    return out, results


def v3(stage, resamples):
    plain, ab = stage.arms, stage.options['ab_arms']
    every = plain + ab
    out = [f'# Gate summary, stage {stage.name} ({LABEL})', '', stage.intro_text(resamples), '']
    verdicts = []
    for arch in stage.options['arches']:
        pooled_floors, per_fold = report.v3_floors(arch)
        out += [f'## Arch {arch}: pooled table', ''] + v3_table(arch, every) + ['']
        out += ['x86-rule macro-F1 per fold: ' + ', '.join(f"{f['x86_rule']['macro_f1']:.3f}" for f in per_fold)
                + f" (pooled {pooled_floors['x86_rule']['macro_f1']:.3f}).", '']
        out += [v3_truncation(arch, plain), '']
        out += [f'### Arch {arch}: paired differences, second minus first, AUC selection ({LABEL})', '']
        lines, results = v3_pairs(arch, stage.pairs, resamples, False, every)
        out += lines
        verdicts.append(v3_verdict(arch, results))
        out += [f'### Arch {arch}: paired cell-balanced differences, second minus first, AUC selection ({LABEL})', '']
        lines, balanced = v3_pairs(arch, stage.pairs, resamples, True, every)
        out += lines
        verdicts.append(v3_verdict(arch, balanced, note='cell-balanced: '))
        out += [f'### Arch {arch}: arch-balanced arms, paired differences, second minus first, AUC selection ({LABEL})', '']
        lines, plain_ab = v3_pairs(arch, stage.options['ab_pairs'], resamples, False, every)
        out += lines
        verdicts.append(v3_verdict(arch, plain_ab, ' ab'))
        out += [f'### Arch {arch}: arch-balanced arms, paired cell-balanced differences ({LABEL})', '']
        lines, balanced_ab = v3_pairs(arch, stage.options['ab_pairs'], resamples, True, every)
        out += lines
        verdicts.append(v3_verdict(arch, balanced_ab, ' ab', 'cell-balanced: '))
        out += [f'### Arch {arch}: each arch-balanced arm minus its unweighted twin, cell-balanced ({LABEL})', '']
        lines, _ = v3_pairs(arch, stage.options['twins'], resamples, True, every)
        out += lines
    out += ['## Verdict: does R7 beat R2 and R4?', '',
            f'Pre-registered rule (PLAN_GNN.md section 5), applied to the AUC selection: dAUC >= {MIN_AUC} and '
            f'dmacro-F1 >= {MIN_F1} pooled, both 95% intervals excluding zero. With one seed it cannot pass the rule; '
            'these lines only say which way seed 0 points. Per arch, the first line uses the ordinary metrics on the '
            'unweighted arms, the second the cell-balanced metrics on them, the third the ordinary metrics on the '
            'arch-balanced arms, the fourth the cell-balanced metrics on the arch-balanced arms.', ''] + verdicts + ['']
    return out


# ---- subgraph --------------------------------------------------------------------------------------------------------

def truncated_share(group):
    """'k of n' test binaries that kept fewer blocks than they have (AUC checkpoint, seed 0), or '-'."""
    complete, _ = stats.load_group(group, 'auc')
    if SEED not in complete or (complete[SEED].truncated < 0).any():
        return '-'
    cut = complete[SEED].truncated
    return f'{int(cut.sum())} of {len(cut)}'


def chain_line(chain_bar):
    path = stats.RUNS / 'subgraph_chain.json'
    if not path.exists():
        return f'- Chain merge: not measured ({path.name} is missing; run subgraph_chain.py).'
    c = json.loads(path.read_text(encoding='utf-8'))
    verdict = ('a chain-merge policy is worth building' if c['reduction'] >= chain_bar
               else f'below the {100 * chain_bar:.0f}% bar, so no chain-merge policy is built')
    return (f"- Chain merge ({c['pool']}): {c['n_chain_blocks']} of {c['n_blocks']} blocks in {c['n_binaries']} binaries "
            f"could merge, a reduction of {100 * c['reduction']:.2f}% (per binary: mean "
            f"{100 * c['mean_binary_reduction']:.2f}%, maximum {100 * c['max_binary_reduction']:.2f}%): {verdict}.")


def paired_rows(reference, arms, resamples):
    """[(label, group, result or None)] of `arms` against the `reference` variant (kfold x86, AUC selection, seed 0) and
    the markdown blocks that go with them."""
    rows, blocks = [], []
    for label, variant in arms:
        group = x86_group(variant)
        try:
            result, info = stats.compare(x86_group(reference), group, 'auc', resamples, seeds=[SEED])
        except stats.MissingGroup as e:
            rows.append((label, group, None))
            blocks += [f'**{label} vs {x86_group(reference)}** (selection auc): missing: {e}', '']
            continue
        rows.append((label, group, result))
        blocks += [stats.fmt_block(x86_group(reference), group, 'auc', resamples, result, info), '']
    return rows, blocks


def subgraph(stage, resamples):
    ref_label, ref = stage.options['reference']
    noedges, weightings = stage.options['noedges'], stage.options['weightings']
    out = [f'# Gate summary, stage {stage.name} ({LABEL})', '', stage.intro_text(resamples), '', '## Pooled, AUC checkpoint', '',
           '| arm | group | AUC | PR-AUC | macro-F1 |', '|---|---|---|---|---|']
    for label, variant in (stage.options['reference'],) + stage.arms + (('R2 no-edges, tfidf', noedges),) + weightings:
        out.append(f'| {label} | {x86_group(variant)} | ' + ' | '.join(metric_cells(pooled(x86_group(variant), 'auc')))
                   + ' |')
    out += ['', '## Subgraph policies: non-inferiority against the whole graph', '',
            f'Reference: {ref_label} ({x86_group(ref)}). An arm is non-inferior when the lower bound of its dAUC '
            f'interval is above -{NI_AUC} and the lower bound of its dmacro-F1 interval is above -{NI_F1}.', '',
            '| arm | group | dAUC [95% CI] | dmacro-F1 [95% CI] | truncated test binaries | non-inferior |',
            '|---|---|---|---|---|---|']
    rows, blocks = paired_rows(ref, stage.arms, resamples)
    for label, group, r in rows:
        out.append(f'| {label} | {group} | missing | missing | - | - |' if r is None else
                   f"| {label} | {group} | {signed_ci(r['auc'])} | {signed_ci(r['macro_f1'])} | {truncated_share(group)} "
                   f"| {'yes' if non_inferior(r) else 'no'} |")
    out += [''] + blocks
    out += ['## Chain merge', '', chain_line(stage.options['chain_bar']), '']
    out += ['## Block-level counting weightings', '',
            'No-edge arm with the raw (counts), presence (each distinct token once) and rawlog (log1p of the count) '
            f'weightings against the TF-IDF no-edge arm ({x86_group(noedges)}).', '',
            '| arm | group | dAUC [95% CI] | dmacro-F1 [95% CI] |', '|---|---|---|---|']
    rows, blocks = paired_rows(noedges, weightings, resamples)
    for label, group, r in rows:
        out.append(f'| {label} | {group} | missing | missing |' if r is None else
                   f"| {label} | {group} | {signed_ci(r['auc'])} | {signed_ci(r['macro_f1'])} |")
    out += [''] + blocks
    return out


LAYOUTS = {'day1': day1, 'day1b': day1b, 'v3': v3, 'subgraph': subgraph}
