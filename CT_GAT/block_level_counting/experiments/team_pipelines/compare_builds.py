"""Compare two pipeline-B builds sample by sample (the equivalence check of Yanping's streaming preprocess).

python -m experiments.team_pipelines.compare_builds pick    --build RUNS/team/b_sig500 --out ids.json [--n 60] [--low 10] [--seed 0]
python -m experiments.team_pipelines.compare_builds compare --a RUNS/team/b_sig500 --b <dir> [--ids-file ids.json] [--json out.json]

`pick`: N ids of a build: the `--low` ids with the lowest mean `instruction_matched`, then ransomware and goodware ids drawn
with `random.Random(seed)` from the rest until each label holds N / 2, sorted.
`compare`: for every id both builds hold, compares the saved tensors: `block_keys`; `instruction_matched`; `instruction_ids`
columns 0-2 and 4 (opcode, operand classes, behaviour) and column 3 (api); `block_features`; `edge_index`; `edge_attr`; plus the
match rate (mean `instruction_matched`) of each side. A sample is identical when every one of these is equal. Reads `.pt` files
only; nothing is written next to them.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

from common.config import RUNS


def resolve(path):
    path = Path(path)
    return RUNS.joinpath(*path.parts[1:]) if path.parts and path.parts[0] == 'RUNS' else path


def load(build, sample_id):
    return torch.load(Path(build) / f'{sample_id}.pt', map_location='cpu', weights_only=False)


def match_rate(payload):
    m = payload['instruction_matched']
    return float(m.float().mean()) if m.numel() else 1.0


def done_ids(build):
    return sorted(json.loads((Path(build) / 'checkpoint.json').read_text(encoding='utf-8'))['done'])


def pick(build, n=60, low=10, seed=0):
    """(ids, info): the `low` lowest-match ids, then label-balanced random ones up to n."""
    rows = sorted((match_rate(p), i, int(p['label'])) for i in done_ids(build) for p in [load(build, i)])
    lowest = rows[:low]
    chosen = {i for _, i, _ in lowest}
    rng = random.Random(seed)
    for label in (1, 0):
        have = sum(1 for r in lowest if r[2] == label)
        pool = [r for r in rows[low:] if r[2] == label]
        chosen.update(i for _, i, _ in rng.sample(pool, n // 2 - have))
    ids = sorted(chosen)
    labels = {i: label for _, i, label in rows}
    info = {'n': len(ids), 'ransomware': sum(labels[i] for i in ids), 'goodware': sum(1 - labels[i] for i in ids),
            'lowest': [{'sample_id': i, 'match': round(m, 4), 'label': label} for m, i, label in lowest]}
    return ids, info


def compare_payloads(a, b):
    """{array name: {'equal': bool, ...}} for the arrays the equivalence check names."""
    ia, ib = a['instruction_ids'], b['instruction_ids']
    cols = [0, 1, 2, 4]
    parts = {
        'block_keys': (list(map(tuple, a['block_keys'])), list(map(tuple, b['block_keys']))),
        'instruction_matched': (a['instruction_matched'], b['instruction_matched']),
        'ids_opcode_operands_behaviour': (ia[:, cols], ib[:, cols]),
        'ids_api': (ia[:, 3], ib[:, 3]),
        'block_features': (a['block_features'], b['block_features']),
        'edge_index': (a['edge_index'], b['edge_index']),
        'edge_attr': (a['edge_attr'], b['edge_attr']),
    }
    out = {}
    for name, (x, y) in parts.items():
        if name == 'block_keys':
            out[name] = {'equal': x == y, 'n_a': len(x), 'n_b': len(y)}
        elif tuple(x.shape) != tuple(y.shape):
            out[name] = {'equal': False, 'shape_a': list(x.shape), 'shape_b': list(y.shape)}
        elif torch.equal(x, y):
            out[name] = {'equal': True}
        else:
            diff = (x.float() - y.float()).abs()
            rows = diff if diff.dim() == 1 else diff.flatten(1).any(1)
            out[name] = {'equal': False, 'rows_differing': int((rows != 0).sum()), 'rows': int(rows.shape[0]),
                         'max_abs': float(diff.max())}
    return out


def compare(a_dir, b_dir, ids):
    per = {}
    for sample_id in ids:
        pa, pb = load(a_dir, sample_id), load(b_dir, sample_id)
        res = compare_payloads(pa, pb)
        per[sample_id] = {'label': int(pa['label']), 'identical': all(v['equal'] for v in res.values()),
                          'match_a': match_rate(pa), 'match_b': match_rate(pb), 'arrays': res,
                          'n_instructions': int(pa['instruction_matched'].numel())}
    names = {}
    for r in per.values():
        for name, v in r['arrays'].items():
            if not v['equal']:
                names[name] = names.get(name, 0) + 1
    identical = sum(r['identical'] for r in per.values())
    return {'compared': len(per), 'identical': identical, 'differing': len(per) - identical,
            'differing_arrays': names, 'samples': per}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p1 = sub.add_parser('pick')
    p1.add_argument('--build', type=Path, required=True)
    p1.add_argument('--out', type=Path, required=True)
    p1.add_argument('--n', type=int, default=60)
    p1.add_argument('--low', type=int, default=10)
    p1.add_argument('--seed', type=int, default=0)
    p2 = sub.add_parser('compare')
    p2.add_argument('--a', type=Path, required=True)
    p2.add_argument('--b', type=Path, required=True)
    p2.add_argument('--ids-file', type=Path)
    p2.add_argument('--json', type=Path)
    args = ap.parse_args(argv)
    if args.cmd == 'pick':
        ids, info = pick(resolve(args.build), args.n, args.low, args.seed)
        args.out.write_text(json.dumps({'ids': ids, **info}, indent=1), encoding='utf-8')
        print(f"{len(ids)} ids ({info['ransomware']} ransomware, {info['goodware']} goodware) -> {args.out}")
        return 0
    a, b = resolve(args.a), resolve(args.b)
    if args.ids_file:
        ids = json.loads(args.ids_file.read_text(encoding='utf-8'))['ids']
    else:
        ids = sorted(set(done_ids(a)) & set(done_ids(b)))
    result = compare(a, b, ids)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(f"compared {result['compared']}: {result['identical']} identical, {result['differing']} differing; "
          f"arrays that differ (samples): {result['differing_arrays']}")
    for i, r in sorted(result['samples'].items()):
        if not r['identical']:
            bad = {k: v for k, v in r['arrays'].items() if not v['equal']}
            print(f"  {i[:12]} label {r['label']} match {r['match_a']:.4f} -> {r['match_b']:.4f}: {bad}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
