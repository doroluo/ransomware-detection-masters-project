"""PASS or FAIL of a pipeline-B build against the acceptance criteria for the listing-to-CFG match (plan v2, item 2).

python -m experiments.team_pipelines.match_acceptance --build RUNS/team/b_sig500 [--source gnn-331ce0b|main-c7899f0]
       [--rebuild-n 5] [--rebuild-seed 0] [--skip-rebuild] [--json out.json]
       (run from the block_level_counting root; a leading RUNS/ is graph_transformer_runs; exit status 0 only if all PASS)

The criteria, each printed with its measured value:
  1 complete       all 1,810 kfold-x86 ids are built (the checkpoint holds exactly `all_ids()`), none excluded
                   (`build_summary.json` and `settings.json` list no exclusions, processed == total == 1810);
  2 label gap      the share of binaries with under 90% of their selected instructions matched differs between ransomware
                   and goodware by at most 2 percentage points;
  3 no bad tail    the share with under 50% matched is below 1% for both labels;
  4 no label leak  the AUC of `1 - match_rate` as a ransomware score is below 0.55 (the point estimate decides; the 95%
                   cluster-bootstrap interval of match_diagnosis is printed);
  5 deterministic  rebuilding --rebuild-n sampled ids (random.Random(--rebuild-seed) over the build's ids) into a temporary
                   directory gives identical tensors and metadata. The rebuild starts from the build's own `api_vocab.json`
                   and `channel_embedder.pt` (`build_b --frozen-from <build>`, read only): a fresh vocabulary would number the
                   API names differently, so those tensors could not be compared with the build's. Arrays are compared with
                   `torch.equal`, other fields with `==`; --skip-rebuild prints SKIPPED and the build cannot pass.
On a build that is not complete, criteria 2-5 are evaluated on the ids it holds and marked partial. A criterion that
cannot be evaluated (no samples of a label) is FAIL.
"""
import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

import torch

from experiments.team_pipelines.build_b import resolve_dir
from experiments.team_pipelines.match_diagnosis import auc_of_loss, build_ids, build_source, label_table, load_samples

GAP_MAX = 0.02
TAIL_MAX = 0.01
AUC_MAX = 0.55
EXPECTED_IDS = 1810


def expected_ids():
    from experiments.team_pipelines.common import all_ids
    return all_ids()


def check_complete(build, expected):
    """(ok, detail): the build holds exactly the expected ids and excludes none."""
    build = Path(build)
    held = set(build_ids(build))
    expected = set(expected)
    excluded = {}
    summary_ok, processed, total = False, None, None
    summary = build / 'build_summary.json'
    if summary.is_file():
        data = json.loads(summary.read_text(encoding='utf-8'))
        excluded.update(data.get('excluded') or {})
        processed, total = data.get('processed'), data.get('total')
        summary_ok = processed == total == len(expected)
    settings = build / 'settings.json'
    if settings.is_file():
        excluded.update(json.loads(settings.read_text(encoding='utf-8')).get('excluded') or {})
    missing, extra = sorted(expected - held), sorted(held - expected)
    ok = not missing and not extra and not excluded and summary_ok
    detail = (f'{len(held & expected)} of {len(expected)} ids built, {len(missing)} missing, {len(extra)} outside the fold set, '
              f'{len(excluded)} excluded, build_summary processed {processed} of {total}'
              + ('' if summary else ' (no build_summary.json)'))
    return ok, detail


def check_label_gap(table):
    r, g = table['ransomware'], table['goodware']
    if not r['n'] or not g['n']:
        return False, 'a label has no binaries'
    gap = abs(r['share_below_90'] - g['share_below_90'])
    return gap <= GAP_MAX + 1e-9, (f"under 90%: ransomware {100 * r['share_below_90']:.1f}% ({r['n_below_90']} of {r['n']}), goodware "
                            f"{100 * g['share_below_90']:.1f}% ({g['n_below_90']} of {g['n']}), gap {100 * gap:.1f} points "
                            f"(limit {100 * GAP_MAX:.0f})")


def check_tail(table):
    r, g = table['ransomware'], table['goodware']
    if not r['n'] or not g['n']:
        return False, 'a label has no binaries'
    ok = r['share_below_50'] < TAIL_MAX and g['share_below_50'] < TAIL_MAX
    return ok, (f"under 50%: ransomware {100 * r['share_below_50']:.1f}% ({r['n_below_50']} of {r['n']}), goodware "
                f"{100 * g['share_below_50']:.1f}% ({g['n_below_50']} of {g['n']}) (limit {100 * TAIL_MAX:.0f}% each)")


def check_auc(auc):
    if auc['auc'] != auc['auc']:
        return False, 'AUC undefined (one label only)'
    return auc['auc'] < AUC_MAX, (f"AUC of 1 - match rate {auc['auc']:.3f}, 95% interval [{auc['lo']:.3f}, {auc['hi']:.3f}] over "
                                  f"{auc['n_clusters']} clusters (limit {AUC_MAX})")


def payload_diff(a, b):
    """Names of the fields in which two saved samples differ (tensors with torch.equal, the rest with ==)."""
    diff = []
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            diff.append(key)
        elif torch.is_tensor(a[key]) or torch.is_tensor(b[key]):
            if not (torch.is_tensor(a[key]) and torch.is_tensor(b[key]) and a[key].dtype == b[key].dtype
                    and torch.equal(a[key], b[key])):
                diff.append(key)
        elif a[key] != b[key]:
            diff.append(key)
    return diff


def pick_rebuild_ids(ids, n, seed):
    return sorted(random.Random(seed).sample(sorted(ids), min(n, len(ids))))


def frozen_rebuild(build, source):
    """A rebuild function `(ids, out_dir)` that runs `build_b` for those ids with the build's settings and its own
    vocabulary and embedder frozen."""
    from experiments.team_pipelines import build_b
    settings = json.loads((Path(build) / 'settings.json').read_text(encoding='utf-8'))

    def run(ids, out):
        build_b.main(['--sample', settings['sample'], '--num-blocks', str(settings['num_blocks']), '--seed', str(settings['seed']),
                      '--source', source, '--frozen-from', str(build), '--out', str(out), '--ids', *ids])
    return run


def check_deterministic(build, ids, rebuild):
    """(ok, detail): `rebuild(ids, out_dir)` writes `<id>.pt` for the ids; each must equal the build's own."""
    with tempfile.TemporaryDirectory(prefix='match_acceptance_') as tmp:
        out = Path(tmp) / 'rebuild'
        rebuild(list(ids), out)
        different = {}
        for sample_id in ids:
            a = torch.load(Path(build) / f'{sample_id}.pt', map_location='cpu', weights_only=False)
            b = torch.load(out / f'{sample_id}.pt', map_location='cpu', weights_only=False)
            d = payload_diff(a, b)
            if d:
                different[sample_id] = d
    if different:
        return False, f'{len(different)} of {len(ids)} rebuilt ids differ: ' + '; '.join(
            f'{i[:12]} in {", ".join(d)}' for i, d in different.items())
    return True, f'{len(ids)} of {len(ids)} rebuilt ids identical in every field (' + ', '.join(i[:12] for i in ids) + ')'


def evaluate(build, expected, source=None, rebuild=None, rebuild_n=5, rebuild_seed=0, resamples=1000):
    """The list of (criterion, verdict, detail) for a build; verdict is PASS, FAIL or SKIPPED."""
    build = Path(build)
    samples = load_samples(build)
    complete, detail = check_complete(build, expected)
    partial = '' if complete else f' [partial: {len(samples)} binaries]'
    rows = [('1 complete build, nothing excluded', 'PASS' if complete else 'FAIL', detail)]
    table = label_table(samples)
    ok, detail = check_label_gap(table)
    rows.append(('2 label gap of the under-90% share <= 2 points', 'PASS' if ok else 'FAIL', detail + partial))
    ok, detail = check_tail(table)
    rows.append(('3 under-50% share below 1% for both labels', 'PASS' if ok else 'FAIL', detail + partial))
    ok, detail = check_auc(auc_of_loss(samples, resamples))
    rows.append(('4 AUC of 1 - match rate below 0.55', 'PASS' if ok else 'FAIL', detail + partial))
    if rebuild is None:
        rows.append(('5 deterministic rebuild of sampled ids', 'SKIPPED', 'not run'))
    else:
        ids = pick_rebuild_ids([s['sample_id'] for s in samples], rebuild_n, rebuild_seed)
        ok, detail = check_deterministic(build, ids, rebuild)
        rows.append(('5 deterministic rebuild of sampled ids', 'PASS' if ok else 'FAIL', detail))
    return rows


def format_rows(build, rows):
    lines = [f'acceptance of {build}']
    lines += [f'{verdict:<8} {name}: {detail}' for name, verdict, detail in rows]
    overall = 'PASS' if all(v == 'PASS' for _, v, _ in rows) else 'FAIL'
    lines.append(f'OVERALL  {overall}' + ('' if overall == 'PASS' else
                                          f"  ({sum(1 for _, v, _ in rows if v == 'PASS')} of {len(rows)} criteria pass)"))
    return '\n'.join(lines), overall


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--build', type=Path, required=True)
    ap.add_argument('--source', help='teammate source of the build (default: from its settings.json)')
    ap.add_argument('--rebuild-n', type=int, default=5)
    ap.add_argument('--rebuild-seed', type=int, default=0)
    ap.add_argument('--skip-rebuild', action='store_true')
    ap.add_argument('--json', type=Path)
    args = ap.parse_args(argv)
    build = resolve_dir(args.build)
    source = args.source or build_source(build)
    rebuild = None if args.skip_rebuild else frozen_rebuild(build, source)
    rows = evaluate(build, expected_ids(), source, rebuild, args.rebuild_n, args.rebuild_seed)
    text, overall = format_rows(build, rows)
    print(text)
    if args.json:
        args.json.write_text(json.dumps({'build': str(build), 'overall': overall,
                                         'criteria': [{'criterion': n, 'verdict': v, 'detail': d} for n, v, d in rows]},
                                        indent=1), encoding='utf-8')
    return 0 if overall == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
