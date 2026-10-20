"""Per-label match rate of a pipeline-B build: how many of each binary's selected instructions were paired with a listing line.

python -m experiments.team_pipelines.match_rates --build RUNS/team/b_sig500 [--among RUNS/team/b_sig500_v2] [--json out.json]
       (run from the block_level_counting root; a leading RUNS/ is graph_transformer_runs)

For every `<id>.pt` of the build (the ids of its checkpoint.json; with no checkpoint, every .pt) the match rate is the mean of
`instruction_matched` over the selected instructions of the saved subgraph. The table gives, per label, the number of binaries,
the mean rate, the share with a rate below 50% and below 90%, and the share with rate 0. --among DIR keeps only the ids that DIR's checkpoint
also holds (to read a finished build against a partial one on the same binaries). Nothing is parsed apart from the
tensors the build saved, and nothing is written next to them.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

from experiments.team_pipelines.build_b import resolve_dir

LABELS = {1: 'ransomware', 0: 'goodware'}


def build_ids(build):
    ckpt = Path(build) / 'checkpoint.json'
    if ckpt.is_file():
        return sorted(json.loads(ckpt.read_text(encoding='utf-8'))['done'])
    return sorted(p.stem for p in Path(build).glob('*.pt') if p.name != 'channel_embedder.pt')


def sample_rates(build, ids=None):
    """{sample_id: (label, match rate, n instructions)} for the build's samples."""
    out = {}
    for sample_id in (build_ids(build) if ids is None else ids):
        p = torch.load(Path(build) / f'{sample_id}.pt', map_location='cpu', weights_only=False)
        m = p['instruction_matched']
        out[sample_id] = (int(p['label']), float(m.float().mean()) if m.numel() else 1.0, int(m.numel()))
    return out


def summarize(rates):
    """{label name: {n, mean_match, share_below_50, share_below_90, share_zero}} over {id: (label, rate, n)}."""
    out = {}
    for label, name in LABELS.items():
        vals = [r for (lab, r, _) in rates.values() if lab == label]
        n = len(vals)
        share = lambda pred: sum(1 for v in vals if pred(v)) / n if n else float('nan')   # noqa: E731
        out[name] = {'n': n, 'mean_match': sum(vals) / n if n else float('nan'), 'share_below_50': share(lambda v: v < 0.5),
                     'share_below_90': share(lambda v: v < 0.9), 'share_zero': share(lambda v: v == 0.0),
                     'n_below_50': sum(1 for v in vals if v < 0.5), 'n_below_90': sum(1 for v in vals if v < 0.9)}
    return out


def format_table(summary, build):
    lines = [f'match rates of {build}', '',
             '| label | binaries | mean match | below 50% | below 90% | exactly 0 |', '|---|---|---|---|---|---|']
    for name, s in summary.items():
        lines.append(f"| {name} | {s['n']} | {s['mean_match']:.3f} | {s['n_below_50']} ({100 * s['share_below_50']:.1f}%) | "
                     f"{s['n_below_90']} ({100 * s['share_below_90']:.1f}%) | {100 * s['share_zero']:.1f}% |")
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--build', type=Path, required=True)
    ap.add_argument('--among', type=Path, help='only the ids that this build also holds')
    ap.add_argument('--json', type=Path)
    args = ap.parse_args(argv)
    build = resolve_dir(args.build)
    ids = build_ids(build)
    if args.among:
        ids = sorted(set(ids) & set(build_ids(resolve_dir(args.among))))
    rates = sample_rates(build, ids)
    summary = summarize(rates)
    print(format_table(summary, str(build) + (f' ({len(ids)} ids also in {resolve_dir(args.among)})' if args.among else '')))
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
