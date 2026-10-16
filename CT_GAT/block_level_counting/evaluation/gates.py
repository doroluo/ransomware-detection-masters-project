"""Gate summary from the runs on disk.

python gates.py --stage day1|day1b|v3|subgraph|paths [--resamples 2000]

Day 1 reads seed 0 of the kfold x86 arms (R0 TF-IDF baseline, R2 tfidf / mean, R4 shuffle, R5 untyped, R7 typed GNN):
a pooled table, the preliminary gate-A verdict of PLAN_GNN.md section 5, and the stats.py paired differences. An arm
with missing folds is reported as missing with the folds that are present; nothing is dropped silently.

Day 1b reads the extra arms of step 5b (direction, augmentation, masking, depth, train fraction): the pooled table of
every kfold x86 GNN group on disk, the AUC-selection paired differences of each new arm against R7 and R2 tfidf, and
the learning curve (train fraction 0.25 / 0.5 / 1.0 for R7 and TF-IDF).

Stage v3 reads seed 0 of the kfold_v3 block-masked arms (R2 no-edges, R4 shuffle, R7 typed with 15% block masking and
cell-balanced checkpoint selection, `-mblk15-selcell`) and the R0 TF-IDF baseline on the combined cache, for arch all
and arch matched: the pooled table under both checkpoint rules with per-architecture recall at the validation threshold,
cell-balanced AUC and macro-F1 (each class x arch cell weighted equally) and both floors (majority, x86 rule), the
arch-balanced (`-ab`) arms beside their unweighted twins, the paired AUC-selection differences R2-R7, R4-R7 and
R0-R2 (ordinary and cell-balanced metrics, for the plain and for the `-ab` arms), the cell-balanced difference of each
`-ab` arm against its unweighted twin, and one-line verdicts per arch on whether R7 beats R2 and R4 under the
pre-registered rule.

Stage subgraph reads the kfold x86 arms of step 11: R7 typed with 15% block masking under the subgraph policies cap and
lcc at 20k and 5k blocks against the whole-graph `gnn-mblk15` reference (non-inferiority: the lower bound of the paired
95% interval of dAUC above -0.01 and of dmacro-F1 above -0.02, AUC selection), the chain-merge measurement of
RUNS/subgraph_chain.json, and the no-edge arm with the raw, presence and rawlog block weightings against
`gnn-noedges-mblk15`.

Stage paths reads the kfold x86 arms of step 12: R7 with the out-of-fold TF-IDF logit (`gnn-stack-mblk15`) against the
`gnn-mblk15` reference and against the TF-IDF baseline, and the block-vector arms (path 1: no-edge GNN on precomputed
block vectors, path 2: typed GNN on them) against each other and against the token-counting no-edge arm
`gnn-noedges-mblk15` (R2 tokens). The vectors are the reference encoder's until the teammates' checkpoints arrive. An
arm that is not on disk, or has fewer than five folds (the reference arms run on fold 0 only), is reported as missing.
"""
import argparse
import json
from functools import lru_cache

import numpy as np

from common.config import RunConfig
from evaluation import metrics, report, stats
from pipeline.s04_select_subgraph.subgraph import MAX_BINARY_BLOCKS

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


def variant_of(**kw):
    """RunConfig.variant of a GNN arm, so the group names here cannot drift from the run names."""
    return RunConfig(**{'model': 'gnn', 'protocol': 'kfold_v3', 'fold': 0, 'arch': 'all', **kw}).variant


V3_MASKED = dict(mask_blocks=0.15, select_on='cell')
V3_GNN = (('R2 no-edges', dict(control='no_edges')), ('R4 shuffle', dict(control='shuffle')), ('R7 typed', {}))
V3_ARMS = (('R0 TF-IDF', 'tfidf'),) + tuple((arm, variant_of(**kw, **V3_MASKED)) for arm, kw in V3_GNN)
V3_AB_ARMS = (('R0 TF-IDF ab', 'tfidf-ab'),) + tuple(
    (f'{arm} ab', variant_of(**kw, **V3_MASKED, arch_balance=True)) for arm, kw in V3_GNN)
V3_PAIRS = (('R2 no-edges', 'R7 typed'), ('R4 shuffle', 'R7 typed'), ('R0 TF-IDF', 'R2 no-edges'))
V3_AB_PAIRS = tuple((f'{a} ab', f'{b} ab') for a, b in V3_PAIRS)
V3_TWINS = tuple((arm, f'{arm} ab') for arm, _ in V3_ARMS)
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


def v3_table(arch):
    metric_names = ('AUC', 'PR-AUC', 'macro-F1')
    rules = ('F1 ckpt, 0.5', 'AUC ckpt, val thr')
    head = ['| arm | group | ' + ' | '.join(f'{rule} {m}' for rule in rules for m in metric_names)
            + ' | ' + ' | '.join(f'{t} recall' for t, _, _ in V3_RECALLS)
            + ' | ' + ' | '.join(f'{rule} cell-bal {m}' for rule in rules for m in ('AUC', 'macro-F1')) + ' |',
            '|' + '---|' * (2 + 6 + len(V3_RECALLS) + 4)]
    fmt = lambda cells: [c if isinstance(c, str) else f'{c:.3f}' for c in cells]
    rows = []
    for arm, variant in V3_ARMS + V3_AB_ARMS:
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


def beats(r):
    """The pre-registered minimum effect on one paired comparison (b - a): dAUC >= 0.01 and dmacro-F1 >= 0.02 with
    both 95% intervals above zero."""
    return all(r[k]['diff'] >= m and r[k]['lo'] > 0 for k, m in (('auc', MIN_AUC), ('macro_f1', MIN_F1)))


def v3_truncation(arch):
    """One line: R7's test binaries cut by the per-binary block cap (AUC checkpoint, validation threshold) and the
    ransomware / goodware recall on the cut and the uncut ones."""
    group = v3_group(arch, dict(V3_ARMS)['R7 typed'])
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


def v3_pairs(arch, pairs, resamples, balanced):
    """(markdown lines, {(a, b): result or the missing reason}) of the paired AUC-selection differences."""
    out, results = [], {}
    for a, b in pairs:
        ga, gb = (v3_group(arch, dict(V3_ARMS + V3_AB_ARMS)[x]) for x in (a, b))
        try:
            result, info = stats.compare(ga, gb, 'auc', resamples, seeds=[SEED], protocol='kfold_v3', arch=arch,
                                         balanced=balanced)
            results[a, b] = result
            out += [f'**{a} vs {b}**', '', stats.fmt_block(ga, gb, 'auc', resamples, result, info), '']
        except stats.MissingGroup as e:
            results[a, b] = str(e)
            out += [f'**{a} vs {b}** (selection auc): missing: {e}', '']
    return out, results


def v3(resamples):
    out = [f'# Gate summary, stage v3 ({LABEL})', '',
           'Combined cache, kfold_v3 (family-holdout v3 folds 0-4), seed 0 pooled over the five folds. Every GNN arm is '
           'block-masked (15% of the blocks in training, `mblk15`). Its AUC checkpoint is chosen by the validation '
           'cell-balanced AUC and its threshold by the validation cell-balanced macro-F1 (`selcell`); the F1 checkpoint '
           'rule is unchanged. The F1 checkpoint is scored at 0.5; the AUC checkpoint at its validation threshold '
           "(TF-IDF: the validation macro-F1 maximiser, per fold); recalls are at that threshold. A missing arm or fold "
           "is stated in its row. Floors: the majority "
           "class of each fold's training manifest, and the x86 rule (ransomware iff x86). `ab` arms were trained with "
           "--arch-balance (each class x arch cell weighted equally). Cell-bal columns weight each class x arch cell "
           "of the pooled test set equally; under them the x86 rule scores 0.5.", '']
    verdicts = []
    for arch in V3_ARCHES:
        pooled_floors, per_fold = report.v3_floors(arch)
        out += [f'## Arch {arch}: pooled table', ''] + v3_table(arch) + ['']
        out += ['x86-rule macro-F1 per fold: ' + ', '.join(f"{f['x86_rule']['macro_f1']:.3f}" for f in per_fold)
                + f" (pooled {pooled_floors['x86_rule']['macro_f1']:.3f}).", '']
        out += [v3_truncation(arch), '']
        out += [f'### Arch {arch}: paired differences, second minus first, AUC selection ({LABEL})', '']
        lines, results = v3_pairs(arch, V3_PAIRS, resamples, balanced=False)
        out += lines
        verdicts.append(v3_verdict(arch, results))
        out += [f'### Arch {arch}: paired cell-balanced differences, second minus first, AUC selection ({LABEL})', '']
        lines, balanced = v3_pairs(arch, V3_PAIRS, resamples, balanced=True)
        out += lines
        verdicts.append(v3_verdict(arch, balanced, note='cell-balanced: '))
        out += [f'### Arch {arch}: arch-balanced arms, paired differences, second minus first, AUC selection ({LABEL})', '']
        lines, plain_ab = v3_pairs(arch, V3_AB_PAIRS, resamples, balanced=False)
        out += lines
        verdicts.append(v3_verdict(arch, plain_ab, ' ab'))
        out += [f'### Arch {arch}: arch-balanced arms, paired cell-balanced differences ({LABEL})', '']
        lines, balanced_ab =v3_pairs(arch, V3_AB_PAIRS, resamples, balanced=True)
        out += lines
        verdicts.append(v3_verdict(arch, balanced_ab, ' ab', 'cell-balanced: '))
        out += [f'### Arch {arch}: each arch-balanced arm minus its unweighted twin, cell-balanced ({LABEL})', '']
        lines, _ = v3_pairs(arch, V3_TWINS, resamples, balanced=True)
        out += lines
    out += ['## Verdict: does R7 beat R2 and R4?', '',
            f'Pre-registered rule (PLAN_GNN.md section 5), applied to the AUC selection: dAUC >= {MIN_AUC} and '
            f'dmacro-F1 >= {MIN_F1} pooled, both 95% intervals excluding zero. With one seed it cannot pass the rule; '
            'these lines only say which way seed 0 points. Per arch, the first line uses the ordinary metrics on the '
            'unweighted arms, the second the cell-balanced metrics on them, the third the ordinary metrics on the '
            'arch-balanced arms, the fourth the cell-balanced metrics on the arch-balanced arms.', ''] + verdicts + ['']
    return out


SG_REFERENCE = ('R7 typed, whole graph', variant_of(mask_blocks=0.15))
SG_NOEDGES = variant_of(control='no_edges', mask_blocks=0.15)
SG_ARMS = tuple((f'R7 typed, {policy} {budget // 1000}k', variant_of(mask_blocks=0.15, subgraph=policy, budget=budget))
                for policy in ('cap', 'lcc') for budget in (20000, 5000))
WEIGHTINGS = tuple((f'R2 no-edges, {enc}', variant_of(control='no_edges', mask_blocks=0.15, encoder=enc))
                   for enc in ('raw', 'presence', 'rawlog'))
NI_AUC, NI_F1 = 0.01, 0.02
CHAIN_BAR = 0.05


def x86_group(variant):
    return f'kfold_x86_{variant}'


def signed_ci(r):
    return f"{r['diff'] + 0.0:+.3f} [{r['lo'] + 0.0:+.3f}, {r['hi'] + 0.0:+.3f}]"


def non_inferior(result):
    """Lower bounds of the paired 95% intervals above -0.01 (dAUC) and -0.02 (dmacro-F1)."""
    return result['auc']['lo'] > -NI_AUC and result['macro_f1']['lo'] > -NI_F1


def truncated_share(group):
    """'k of n' test binaries that kept fewer blocks than they have (AUC checkpoint, seed 0), or '-'."""
    complete, _ = stats.load_group(group, 'auc')
    if SEED not in complete or (complete[SEED].truncated < 0).any():
        return '-'
    cut = complete[SEED].truncated
    return f'{int(cut.sum())} of {len(cut)}'


def chain_line():
    path = stats.RUNS / 'subgraph_chain.json'
    if not path.exists():
        return f'- Chain merge: not measured ({path.name} is missing; run subgraph_chain.py).'
    c = json.loads(path.read_text(encoding='utf-8'))
    verdict = ('a chain-merge policy is worth building' if c['reduction'] >= CHAIN_BAR
               else f'below the {100 * CHAIN_BAR:.0f}% bar, so no chain-merge policy is built')
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


def subgraph(resamples):
    ref_label, ref = SG_REFERENCE
    out = [f'# Gate summary, stage subgraph ({LABEL})', '',
           'kfold x86 on the v2 cache, seed 0 pooled over the five folds, 15% block masking on every arm. Differences '
           f'are arm minus reference for the AUC checkpoint at its validation threshold, with the 95% paired cluster '
           f'bootstrap interval ({resamples} resamples).', '', '## Pooled, AUC checkpoint', '',
           '| arm | group | AUC | PR-AUC | macro-F1 |', '|---|---|---|---|---|']
    for label, variant in (SG_REFERENCE,) + SG_ARMS + (('R2 no-edges, tfidf', SG_NOEDGES),) + WEIGHTINGS:
        out.append(f'| {label} | {x86_group(variant)} | ' + ' | '.join(metric_cells(pooled(x86_group(variant), 'auc')))
                   + ' |')
    out += ['', '## Subgraph policies: non-inferiority against the whole graph', '',
            f'Reference: {ref_label} ({x86_group(ref)}). An arm is non-inferior when the lower bound of its dAUC '
            f'interval is above -{NI_AUC} and the lower bound of its dmacro-F1 interval is above -{NI_F1}.', '',
            '| arm | group | dAUC [95% CI] | dmacro-F1 [95% CI] | truncated test binaries | non-inferior |',
            '|---|---|---|---|---|---|']
    rows, blocks = paired_rows(ref, SG_ARMS, resamples)
    for label, group, r in rows:
        out.append(f'| {label} | {group} | missing | missing | - | - |' if r is None else
                   f"| {label} | {group} | {signed_ci(r['auc'])} | {signed_ci(r['macro_f1'])} | {truncated_share(group)} "
                   f"| {'yes' if non_inferior(r) else 'no'} |")
    out += [''] + blocks
    out += ['## Chain merge', '', chain_line(), '']
    out += ['## Block-level counting weightings', '',
            'No-edge arm with the raw (counts), presence (each distinct token once) and rawlog (log1p of the count) '
            f'weightings against the TF-IDF no-edge arm ({x86_group(SG_NOEDGES)}).', '',
            '| arm | group | dAUC [95% CI] | dmacro-F1 [95% CI] |', '|---|---|---|---|']
    rows, blocks = paired_rows(SG_NOEDGES, WEIGHTINGS, resamples)
    for label, group, r in rows:
        out.append(f'| {label} | {group} | missing | missing |' if r is None else
                   f"| {label} | {group} | {signed_ci(r['auc'])} | {signed_ci(r['macro_f1'])} |")
    out += [''] + blocks
    return out


REF_VEC = 'ours_cap0_ref'
PATHS_X86 = dict(protocol='kfold', arch='x86', mask_blocks=0.15)
PATHS_VEC = dict(**PATHS_X86, block_input='vec', block_vec=REF_VEC, val_seed=0)
PATHS_ARMS = (('TF-IDF', 'tfidf'),
              ('R7 typed, tokens', x86_group(variant_of(**PATHS_X86))),
              ('R7 + TF-IDF stack', x86_group(variant_of(**PATHS_X86, stack_tfidf=True))),
              ('R2 no-edges, tokens', x86_group(variant_of(**PATHS_X86, control='no_edges'))),
              ('path 1: no-edges, block vectors', x86_group(variant_of(**PATHS_VEC, control='no_edges'))),
              ('path 2: typed, block vectors', x86_group(variant_of(**PATHS_VEC))))
PATHS_PAIRS = (('R7 typed, tokens', 'R7 + TF-IDF stack'), ('TF-IDF', 'R7 + TF-IDF stack'),
               ('path 1: no-edges, block vectors', 'path 2: typed, block vectors'),
               ('R2 no-edges, tokens', 'path 1: no-edges, block vectors'))


def paths(resamples):
    group = dict(PATHS_ARMS)
    out = [f'# Gate summary, stage paths ({LABEL})', '',
           'kfold x86 on the v2 cache, seed 0 pooled over the five folds, 15% block masking on every GNN arm. Stack: R7 '
           'with the standardised out-of-fold TF-IDF logit as an extra head input (`--stack-tfidf`). Paths 1 and 2 train '
           f'on the frozen block vectors `{REF_VEC}` of the reference encoder (random weights, a plumbing check, run on '
           'fold 0 only until the teammates deliver checkpoints); path 1 has no edges, path 2 has the typed edges. Differences '
           f'are second minus first for the AUC checkpoint at its validation threshold, with the 95% paired cluster '
           f'bootstrap interval ({resamples} resamples). A missing arm or fold is stated.', '', '## Pooled, AUC checkpoint', '',
           '| arm | group | AUC | PR-AUC | macro-F1 |', '|---|---|---|---|---|']
    for arm, g in PATHS_ARMS:
        out.append(f'| {arm} | {g} | ' + ' | '.join(metric_cells(pooled(g, 'auc'))) + ' |')
    out += ['', '## Paired differences, second minus first', '',
            '| first | second | dAUC [95% CI] | dmacro-F1 [95% CI] |', '|---|---|---|---|']
    blocks = []
    for a, b in PATHS_PAIRS:
        try:
            result, info = stats.compare(group[a], group[b], 'auc', resamples, seeds=[SEED])
        except stats.MissingGroup as e:
            out.append(f'| {a} | {b} | missing | missing |')
            blocks += [f'**{b} vs {a}** (selection auc): missing: {e}', '']
            continue
        out.append(f"| {a} | {b} | {signed_ci(result['auc'])} | {signed_ci(result['macro_f1'])} |")
        blocks += [stats.fmt_block(group[a], group[b], 'auc', resamples, result, info), '']
    return out + [''] + blocks


STAGES = {'day1': day1, 'day1b': day1b, 'v3': v3, 'subgraph': subgraph, 'paths': paths}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=list(STAGES), required=True)
    ap.add_argument('--resamples', type=int, default=2000)
    args = ap.parse_args()
    print('\n'.join(STAGES[args.stage](args.resamples)))


if __name__ == '__main__':
    main()
