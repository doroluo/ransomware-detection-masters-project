"""Paired cluster bootstrap between two kfold arms.

python stats.py --a kfold_x86_gnn-noedges --b kfold_x86_gnn [--selection f1|auc] [--resamples 2000]
[--protocol kfold|kfold_v3] [--arch x86|all|matched] [--balanced]

A group is a RunConfig.group of kfold or kfold_v3 runs, or `tfidf` / `structural` for the baselines (`tfidf-tf50`
for those trained on a train-fraction subset, `tfidf-ab` for the arch-balanced ones). A baseline group takes the
protocol and arch of the other group
(`kfold_v3_all_gnn` -> the kfold_v3 all baselines); --protocol / --arch override that, and two baseline groups
without an override mean kfold x86. Per seed present
(with all five folds) in both groups, the pooled out-of-fold test predictions are matched by sample_id and the
binaries both have are compared; those that one arm declares as excluded (`excluded` in its metrics.json) are named in the
output as "N excluded from both arms". Clusters are
ransomware families and single goodware binaries; every resample draws clusters with replacement and is applied to
every seed and both groups. Prints the mean difference (b - a) over seeds and its 95% percentile interval for AUC,
PR-AUC and macro-F1: at 0.5 for `f1` (predictions.csv), at each run's validation threshold for `auc`
(predictions_auc.csv; baselines are thresholded at their metrics.json threshold_val).

--balanced weights every test sample by N / (C n_cell) (metrics.cell_weights, fixed per sample from the pooled test set
of the first seed, since the cells are the same for every seed) times its bootstrap count, so the metrics are the
cell-balanced ones: each non-empty class x arch cell weighs equally.
"""
import argparse
import csv
import json
import re
from dataclasses import dataclass

import numpy as np

from common.config import RUNS, RunConfig
from evaluation.metrics import cell_weights

METRICS = (('auc', 'AUC'), ('pr_auc', 'PR-AUC'), ('macro_f1', 'macro-F1'))
BASELINES = {'tfidf': 'tfidf_sig_api_lr', 'structural': 'structural_lr',
             'wl0': 'wl_h0_lr', 'wl2': 'wl_h2_lr', 'wl2-shuffle': 'wl_h2_shuffle_lr',
             'tfidf-yanping': 'tfidf_yanping_lr', 'tfidf-hybrid': 'tfidf_hybrid_lr', 'tfidf-noapi': 'tfidf_noapi_lr',
             'imports-header': 'imports_header_lr'}
# the baselines of `--vocab` and `--imports` live in directories of their own: kfold0_x86_s0_voc-yanping, kfold0_x86_s0_imp
BASELINE_KIND = {'tfidf-yanping': '_voc-yanping', 'tfidf-hybrid': '_voc-hybrid', 'tfidf-noapi': '_voc-noapi',
                 'imports-header': '_imp'}
FOLDS = range(5)
KFOLD_PROTOCOLS = ('kfold', 'kfold_v3')
BASELINE_PREFIX = {'kfold': 'kfold', 'kfold_v3': 'v3kfold'}
GROUP_SCOPE = re.compile(r'(kfold_v3|kfold)_(x86|all|matched)_')


BASELINE_GROUP = re.compile(rf"({'|'.join(BASELINES)})(?:-tf(\d+))?(-ab)?")


def baseline_method(group):
    """(method, directory suffix) for a baseline group such as `tfidf`, `structural-tf25` or `tfidf-ab` (`_tf25`, `_ab`
    and `_tf50_ab` are the suffixes), `tfidf-yanping` (`_voc-yanping`), `tfidf-noapi` (`_voc-noapi`) or `imports-header` (`_imp`), else None."""
    m = BASELINE_GROUP.fullmatch(group)
    return (BASELINES[m[1]], BASELINE_KIND.get(m[1], '') + (f'_tf{m[2]}' if m[2] else '') + ('_ab' if m[3] else '')) if m else None


class MissingGroup(Exception):
    pass


@dataclass
class Preds:
    """One group's pooled test predictions for one seed, sorted by sample_id."""
    ids: np.ndarray
    y: np.ndarray
    prob: np.ndarray
    pred: np.ndarray
    cluster: np.ndarray
    arch: np.ndarray = None
    truncated: np.ndarray = None


def read_rows(path, method=None):
    with path.open(encoding='utf-8', newline='') as f:
        return [r for r in csv.DictReader(f) if r.get('split', 'test') == 'test' and r.get('method') == method]


def truncation(row):
    """1 if the binary kept fewer blocks than it has, 0 if all, -1 when the predictions file has no such columns."""
    kept, total = row.get('n_blocks_kept'), row.get('n_blocks_total')
    return -1 if kept in (None, '') or total in (None, '') else int(int(kept) < int(total))


def pool(rows_by_fold, thresholds=None):
    rows = [(f, r) for f, rs in rows_by_fold.items() for r in rs]
    rows.sort(key=lambda fr: fr[1]['sample_id'])
    prob = np.array([float(r['prob']) for _, r in rows])
    pred = (np.array([int(r['pred']) for _, r in rows]) if thresholds is None
            else (prob >= np.array([thresholds[f] for f, _ in rows])).astype(int))
    y = np.array([int(r['label']) for _, r in rows])
    cluster = np.array([f"fam:{r['family']}" if r['label'] == '1' else f"id:{r['sample_id']}" for _, r in rows])
    return Preds(np.array([r['sample_id'] for _, r in rows]), y, prob, pred, cluster,
                 np.array([r.get('arch', '') for _, r in rows]), np.array([truncation(r) for _, r in rows]))


def group_scope(group):
    """(protocol, arch) of a RunConfig.group name such as `kfold_v3_all_gnn`, else None."""
    m = GROUP_SCOPE.match(group)
    return (m[1], m[2]) if m else None


def baseline_scopes(group_a, group_b, protocol=None, arch=None):
    """{group: (protocol, arch)} for the baseline groups among the two: the override, else the other group's, else
    kfold x86."""
    scopes = {}
    for g, other in ((group_a, group_b), (group_b, group_a)):
        if baseline_method(g):
            from_other = group_scope(other) if not baseline_method(other) else None
            p, a = from_other or ('kfold', 'x86')
            scopes[g] = (protocol or p, arch or a)
    return scopes


def folds_by_seed(group, scope=('kfold', 'x86')):
    """{seed: {fold: (path of the run or baseline directory)}} for a group name. `scope` = (protocol, arch) picks
    the baseline directories of a baseline group."""
    out = {}
    if base := baseline_method(group):
        tf, (protocol, arch) = base[1], scope
        root = RUNS / ('baselines_wl' if base[0].startswith('wl_') else 'baselines')
        for d in root.glob(f'{BASELINE_PREFIX[protocol]}*_{arch}_s*'):
            m = re.fullmatch(rf'{BASELINE_PREFIX[protocol]}(\d)_{arch}_s(\d+){tf}', d.name)
            if m and (d / 'metrics.json').exists() and base[0] in json.loads((d / 'metrics.json').read_text(encoding='utf-8')).get('methods', {}):
                out.setdefault(int(m[2]), {})[int(m[1])] = d
        return out
    for d in sorted(p for p in RUNS.iterdir() if (p / 'config.json').exists()):
        cfg = RunConfig.try_load(d / 'config.json')
        if cfg and cfg.protocol in KFOLD_PROTOCOLS and cfg.group == group and d.name == cfg.name:
            out.setdefault(cfg.seed, {})[cfg.fold] = d
    return out


def load_group(group, selection, scope=('kfold', 'x86')):
    """({seed: Preds} for the seeds with all five folds, {seed: folds present} for the others)."""
    files = {'f1': 'predictions.csv', 'auc': 'predictions_auc.csv'}[selection]
    complete, partial = {}, {}
    for seed, by_fold in sorted(folds_by_seed(group, scope).items()):
        rows, thresholds = {}, {}
        for fold, d in by_fold.items():
            if base := baseline_method(group):
                method = base[0]
                t = json.loads((d / 'metrics.json').read_text(encoding='utf-8'))['methods'][method].get('threshold_val')
                if selection == 'auc' and t is None:
                    continue
                rows[fold], thresholds[fold] = read_rows(d / 'predictions.csv', method), t
            elif (d / files).exists():
                rows[fold] = read_rows(d / files)
        if len(rows) == 5:
            complete[seed] = pool(rows, thresholds if baseline_method(group) and selection == 'auc' else None)
        else:
            partial[seed] = sorted(rows)
    return complete, partial


def group_exclusions(group, scope=('kfold', 'x86')):
    """{sample_id: reason} that the runs of `group` declare as left out on purpose (`excluded` / `excluded_reason` in
    metrics.json, written by experiments/team_pipelines/common.write_run). Empty for baselines and ordinary runs."""
    out = {}
    if baseline_method(group):
        return out
    for by_fold in folds_by_seed(group, scope).values():
        for d in by_fold.values():
            path = d / 'metrics.json'
            m = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
            out.update({i: m.get('excluded_reason', '') for i in m.get('excluded', [])})
    return out


def metrics_weighted(p: Preds, order, grp, w):
    """(AUC, PR-AUC, macro-F1) of p with integer-like sample weights w; equals the metrics of the samples repeated
    w times. Ties in prob share a threshold (as sklearn does)."""
    ws = w[order]
    pos, neg = np.bincount(grp, ws * p.y[order]), np.bincount(grp, ws * (1 - p.y[order]))
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        auc = pr_auc = float('nan')
    else:
        auc = float((pos * (np.cumsum(neg) - neg + 0.5 * neg)).sum() / (n_pos * n_neg))
        ctp, cfp = np.cumsum(pos[::-1]), np.cumsum(neg[::-1])
        seen = (pos + neg)[::-1] > 0
        pr_auc = float((pos[::-1][seen] / n_pos * ctp[seen] / (ctp + cfp)[seen]).sum())
    tp, fp = (w * p.y * p.pred).sum(), (w * (1 - p.y) * p.pred).sum()
    fn, tn = (w * p.y * (1 - p.pred)).sum(), (w * (1 - p.y) * (1 - p.pred)).sum()
    f1 = lambda a, b, c: 2 * a / (2 * a + b + c) if 2 * a + b + c > 0 else 0.0
    return auc, pr_auc, (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2


def ranking(p: Preds):
    order = np.argsort(p.prob, kind='stable')
    sorted_prob = p.prob[order]
    return order, np.concatenate([[0], np.cumsum(sorted_prob[1:] != sorted_prob[:-1])])


def paired_bootstrap(a, b, resamples=2000, seed=0, balanced=False):
    """a, b: {seed: Preds} over the same seeds and sample_ids. Returns {metric: dict(a, b, diff, lo, hi)} where
    a and b are the observed means over seeds and diff/lo/hi describe b - a. With `balanced`, every sample carries its
    cell weight (arch is required) in the observed means and in each resample."""
    seeds = sorted(a)
    assert sorted(b) == seeds and all(np.array_equal(a[s].ids, b[s].ids) and np.array_equal(a[s].ids, a[seeds[0]].ids)
                                      for s in seeds), 'groups must cover the same seeds and sample_ids'
    clusters, cidx = np.unique(a[seeds[0]].cluster, return_inverse=True)
    arms = {name: {s: (p, *ranking(p)) for s, p in group.items()} for name, group in (('a', a), ('b', b))}
    mean_over_seeds = lambda name, w: np.mean([metrics_weighted(p, o, g, w) for p, o, g in arms[name].values()], 0)
    base = cell_weights(a[seeds[0]].y, a[seeds[0]].arch) if balanced else np.ones(len(cidx))
    observed = {name: mean_over_seeds(name, base) for name in arms}
    draws = np.random.default_rng(seed).integers(0, len(clusters), size=(resamples, len(clusters)))
    diffs = np.empty((resamples, len(METRICS)))
    for i, draw in enumerate(draws):
        w = np.bincount(draw, minlength=len(clusters))[cidx] * base
        diffs[i] = mean_over_seeds('b', w) - mean_over_seeds('a', w)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5], axis=0)
    return {name: {'a': observed['a'][j], 'b': observed['b'][j], 'diff': observed['b'][j] - observed['a'][j],
                   'lo': lo[j], 'hi': hi[j]} for j, (name, _) in enumerate(METRICS)}, len(clusters)


def compare(group_a, group_b, selection='f1', resamples=2000, seed=0, seeds=None, protocol=None, arch=None,
            balanced=False):
    """Loads both groups, keeps the seeds complete in both (and in `seeds`, if given), and returns (result, info).
    Baseline groups resolve to the other group's protocol and arch unless `protocol` / `arch` say otherwise.
    `balanced`: cell-balanced weights (see paired_bootstrap). Raises MissingGroup."""
    scopes = baseline_scopes(group_a, group_b, protocol, arch)
    loaded = {g: load_group(g, selection, scopes.get(g, ('kfold', 'x86'))) for g in (group_a, group_b)}
    wanted = None if seeds is None else set(seeds)
    for g, (complete, partial) in loaded.items():
        if not set(complete) & (wanted if wanted is not None else set(complete)):
            where = '' if wanted is None else f' among seeds {sorted(wanted)}'
            raise MissingGroup(f'{g}: no complete seed{where} with all five folds'
                               + (f' (folds present: {partial})' if partial else ''))
    seeds = sorted(set(loaded[group_a][0]) & set(loaded[group_b][0]) & (wanted or set(loaded[group_a][0])))
    if not seeds:
        raise MissingGroup(f'{group_a} and {group_b} share no complete seed')
    ids = [set(loaded[g][0][s].ids.tolist()) for g in (group_a, group_b) for s in seeds]
    common = np.array(sorted(set.intersection(*ids)))
    declared = {}
    for g in (group_a, group_b):
        declared.update(group_exclusions(g, scopes.get(g, ('kfold', 'x86'))))
    left_out = sorted((set.union(*ids) - set(common.tolist())) & set(declared))
    fields = ('ids', 'y', 'prob', 'pred', 'cluster', 'arch', 'truncated')
    pick = lambda p: Preds(*(getattr(p, f)[np.isin(p.ids, common)] for f in fields))
    result, n_clusters = paired_bootstrap({s: pick(loaded[group_a][0][s]) for s in seeds},
                                          {s: pick(loaded[group_b][0][s]) for s in seeds}, resamples, seed, balanced)
    info = {'seeds': seeds, 'n': len(common), 'n_clusters': n_clusters, 'balanced': balanced,
            'excluded': left_out, 'excluded_reason': '; '.join(sorted({declared[i] for i in left_out})),
            'skipped': {g: partial for g, (_, partial) in loaded.items() if partial}}
    return result, info


def fmt_block(group_a, group_b, selection, resamples, result, info):
    signed = lambda x: f'{x + 0.0:+.3f}'
    lines = [f'### {group_b} minus {group_a} (selection {selection}, {resamples} resamples)', '',
             f"a = {group_a}, b = {group_b}; seeds {info['seeds']}; {info['n']} test binaries in "
             f"{info['n_clusters']} clusters. Differences are b - a, 95% percentile interval."
             + (' Cell-balanced: every test sample weighted N / (C n_cell) by its class x arch cell.'
                if info.get('balanced') else ''), '']
    if info.get('excluded'):
        lines += [f"{len(info['excluded'])} excluded from both arms: {', '.join(info['excluded'])} "
                  f"({info['excluded_reason']}). The differences are on the {info['n']} binaries both arms have.", '']
    if info['skipped']:
        lines += [f"Seeds without all five folds are left out: {info['skipped']}.", '']
    lines += ['| metric | a | b | b - a | 95% CI |', '|---|---|---|---|---|']
    for key, title in METRICS:
        r = result[key]
        lines.append(f"| {title} | {r['a']:.3f} | {r['b']:.3f} | {signed(r['diff'])} | "
                     f"[{signed(r['lo'])}, {signed(r['hi'])}] |")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--selection', choices=['f1', 'auc'], default='f1')
    ap.add_argument('--resamples', type=int, default=2000)
    ap.add_argument('--protocol', choices=KFOLD_PROTOCOLS, help='baseline groups: default from the other group')
    ap.add_argument('--arch', choices=['x86', 'all', 'matched'], help='baseline groups: default from the other group')
    ap.add_argument('--balanced', action='store_true', help='weight each class x arch cell equally')
    args = ap.parse_args()
    try:
        result, info = compare(args.a, args.b, args.selection, args.resamples, protocol=args.protocol, arch=args.arch,
                               balanced=args.balanced)
    except MissingGroup as e:
        raise SystemExit(f'missing: {e}')
    print(fmt_block(args.a, args.b, args.selection, args.resamples, result, info))


if __name__ == '__main__':
    main()
