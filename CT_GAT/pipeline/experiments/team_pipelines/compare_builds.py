"""Compare two pipeline-B builds sample by sample (the equivalence check of Yanping's streaming preprocess).

python -m experiments.team_pipelines.compare_builds pick    --build RUNS/team/b_sig500 --out ids.json [--n 60] [--low 10] [--seed 0]
python -m experiments.team_pipelines.compare_builds compare --a RUNS/team/b_sig500 --b <dir> [--ids-file ids.json] [--json out.json]
       [--state]

`pick`: N ids of a build: the `--low` ids with the lowest mean `instruction_matched`, then ransomware and goodware ids drawn
with `random.Random(seed)` from the rest until each label holds N / 2, sorted.
`compare`: for every id both builds hold, compares the saved tensors: `block_keys`; `instruction_matched`; `instruction_ids`
columns 0-2 and 4 (opcode, operand classes, behaviour) and column 3 (api); `block_features`; `edge_index`; `edge_attr`; plus the
match rate (mean `instruction_matched`) of each side. A sample is identical when every one of these is equal. Reads `.pt` files
only; nothing is written next to them.
`--state` (two builds that are meant to be the same, such as a build from main and the published one) adds what the arrays above
leave out: `block_embeddings` compared exactly (`torch.equal`) with the largest absolute difference printed, the API vocabulary
`api_vocab.json` (the names, whose position is the id: id = FIRST_API_ID + position) and every tensor of `channel_embedder.pt` (the
largest absolute difference of each). `identical` then also needs the vocabulary and the embedder to be equal.
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


def compare_payloads(a, b, state=False):
    """{array name: {'equal': bool, ...}} for the arrays the equivalence check names; with `state` also `block_embeddings`,
    exactly, with the largest absolute difference (0.0 when equal)."""
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
    if state:
        parts['block_embeddings'] = (a['block_embeddings'], b['block_embeddings'])
    out = {}
    for name, (x, y) in parts.items():
        if name == 'block_keys':
            out[name] = {'equal': x == y, 'n_a': len(x), 'n_b': len(y)}
        elif tuple(x.shape) != tuple(y.shape):
            out[name] = {'equal': False, 'shape_a': list(x.shape), 'shape_b': list(y.shape)}
        elif torch.equal(x, y):
            out[name] = {'equal': True, **({'max_abs': 0.0} if name == 'block_embeddings' else {})}
        else:
            diff = (x.float() - y.float()).abs()
            rows = diff if diff.dim() == 1 else diff.flatten(1).any(1)
            out[name] = {'equal': False, 'rows_differing': int((rows != 0).sum()), 'rows': int(rows.shape[0]),
                         'max_abs': float(diff.max())}
    return out


def api_names(build):
    """The API names of a build's `api_vocab.json` in id order (the file is a JSON list; id = FIRST_API_ID + position), or None
    when it has none."""
    path = Path(build) / 'api_vocab.json'
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding='utf-8'))
    return list(payload['names'] if isinstance(payload, dict) else payload)


def compare_state(a_dir, b_dir):
    """The vocabulary and the embedder of two builds: {'api_vocab': {...}, 'channel_embedder': {tensor: {...}}, 'identical': bool}.
    A vocabulary is equal when the names are, in the same order (so the ids are); a tensor is equal under `torch.equal`, and
    `max_abs` is the largest absolute difference of two tensors of one shape."""
    na, nb = api_names(a_dir), api_names(b_dir)
    vocab = {'equal': na is not None and na == nb, 'n_a': None if na is None else len(na), 'n_b': None if nb is None else len(nb)}
    if na is not None and nb is not None and na != nb:
        vocab['first_difference'] = next((i for i, (x, y) in enumerate(zip(na, nb)) if x != y), min(len(na), len(nb)))
        vocab['same_names_other_order'] = sorted(na) == sorted(nb)
    weights = []
    for build in (a_dir, b_dir):
        path = Path(build) / 'channel_embedder.pt'
        weights.append(torch.load(path, map_location='cpu', weights_only=True) if path.is_file() else None)
    embedder = {}
    if weights[0] is None or weights[1] is None:
        embedder['channel_embedder.pt'] = {'equal': False, 'missing': 'a' if weights[0] is None else 'b'}
    else:
        for key in sorted(weights[0].keys() | weights[1].keys()):
            x, y = weights[0].get(key), weights[1].get(key)
            if x is None or y is None:
                embedder[key] = {'equal': False, 'missing': 'a' if x is None else 'b'}
            elif tuple(x.shape) != tuple(y.shape):
                embedder[key] = {'equal': False, 'shape_a': list(x.shape), 'shape_b': list(y.shape)}
            else:
                embedder[key] = {'equal': bool(torch.equal(x, y)), 'max_abs': float((x.float() - y.float()).abs().max()) if x.numel() else 0.0}
    return {'api_vocab': vocab, 'channel_embedder': embedder,
            'identical': vocab['equal'] and bool(embedder) and all(v['equal'] for v in embedder.values())}


def compare(a_dir, b_dir, ids, state=False):
    """Per-sample comparison of the ids in both builds; with `state` also `block_embeddings` exactly, the largest difference of
    them (`max_block_embedding_diff`) and `compare_state` of the two builds (key `state`)."""
    per = {}
    for sample_id in ids:
        pa, pb = load(a_dir, sample_id), load(b_dir, sample_id)
        res = compare_payloads(pa, pb, state)
        per[sample_id] = {'label': int(pa['label']), 'identical': all(v['equal'] for v in res.values()),
                          'match_a': match_rate(pa), 'match_b': match_rate(pb), 'arrays': res,
                          'n_instructions': int(pa['instruction_matched'].numel())}
    names = {}
    for r in per.values():
        for name, v in r['arrays'].items():
            if not v['equal']:
                names[name] = names.get(name, 0) + 1
    identical = sum(r['identical'] for r in per.values())
    result = {'compared': len(per), 'identical': identical, 'differing': len(per) - identical,
              'differing_arrays': names, 'samples': per}
    if state:
        diffs = [r['arrays']['block_embeddings'].get('max_abs') for r in per.values()]
        result['max_block_embedding_diff'] = max([d for d in diffs if d is not None], default=None)
        result['state'] = compare_state(a_dir, b_dir)
    return result


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
    p2.add_argument('--state', action='store_true', help='also compare block_embeddings exactly, api_vocab.json and channel_embedder.pt')
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
    result = compare(a, b, ids, args.state)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(f"compared {result['compared']}: {result['identical']} identical, {result['differing']} differing; "
          f"arrays that differ (samples): {result['differing_arrays']}")
    for i, r in sorted(result['samples'].items()):
        if not r['identical']:
            bad = {k: v for k, v in r['arrays'].items() if not v['equal']}
            print(f"  {i[:12]} label {r['label']} match {r['match_a']:.4f} -> {r['match_b']:.4f}: {bad}")
    if args.state:
        state = result['state']
        vocab, tensors = state['api_vocab'], state['channel_embedder']
        print(f"state: block_embeddings largest |difference| {result['max_block_embedding_diff']}; api_vocab "
              f"{'equal' if vocab['equal'] else 'DIFFERS'} ({vocab['n_a']} / {vocab['n_b']} names); channel_embedder "
              f"{sum(v['equal'] for v in tensors.values())} of {len(tensors)} tensors equal"
              + ''.join(f"; {k}: {v}" for k, v in tensors.items() if not v['equal']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
