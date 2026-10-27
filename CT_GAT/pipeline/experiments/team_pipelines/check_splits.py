"""Check that the finished runs of one fold and seed share their train / val / test ids (Model 1, 2 and 3 alike).

python -m experiments.team_pipelines.check_splits [--runs DIR]      (run from CT_GAT/pipeline; default RUNS)

Every run directory with a config.json and a manifest.json is grouped by (protocol, arch, fold, seed); within a group the
manifest.json of every run must hold the same ids in each split. Prints one line per group and exits 1 when a group differs.
"""
import argparse
import json
import sys
from pathlib import Path

from common.config import RUNS


def split_key(manifest):
    return tuple(tuple(sorted(manifest[s])) for s in ('train', 'val', 'test'))


def groups(runs):
    """{(protocol, arch, fold, seed): {run name: split key}} of the finished runs under `runs`."""
    out = {}
    for config in sorted(Path(runs).glob('*/config.json')):
        manifest = config.parent / 'manifest.json'
        if not manifest.is_file():
            continue
        cfg = json.loads(config.read_text(encoding='utf-8'))
        key = (cfg.get('protocol'), cfg.get('arch'), cfg.get('fold'), cfg.get('seed'))
        out.setdefault(key, {})[config.parent.name] = split_key(json.loads(manifest.read_text(encoding='utf-8')))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', type=Path, default=RUNS)
    args = ap.parse_args(argv)
    found = groups(args.runs)
    if not found:
        print(f'no finished run (config.json + manifest.json) under {args.runs}')
        return 0
    bad = 0
    for (protocol, arch, fold, seed), runs in sorted(found.items(), key=lambda kv: tuple(str(k) for k in kv[0])):
        distinct = set(runs.values())
        train, val, test = (len(s) for s in next(iter(distinct)))
        status = 'same' if len(distinct) == 1 else f'DIFFERENT ({len(distinct)} versions)'
        print(f'{protocol} {arch} fold {fold} seed {seed}: {len(runs)} runs, train {train} val {val} test {test}: {status}')
        if len(distinct) > 1:
            bad += 1
            for name, key in sorted(runs.items()):
                print(f'    {name}: train {len(key[0])} val {len(key[1])} test {len(key[2])}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
