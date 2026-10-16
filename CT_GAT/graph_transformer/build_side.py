"""Build <cache>/side/<sample_id>.npz from cache-aligned IDA exports; main cache is read-only.

python build_side.py --cache C:/.../graph_transformer_cache_v2 [--limit 2]
python build_side.py --cache C:/.../graph_transformer_cache_combined
Optional --corpus-root name=path overrides build_cache.CORPUS_ROOTS.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

import numpy as np

from config import CACHE
from side_common import sample_rng, shuffle_edges
from side_defuse import EDGE_KIND_NAMES, defuse_edges
from side_tags import TAG_NAMES, function_tags

SCHEMA = 'ida-side/1'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def code_hash():
    h = hashlib.sha256()
    for name in ('build_side.py', 'side_common.py', 'side_defuse.py', 'side_tags.py'):
        h.update(Path(__file__).with_name(name).read_bytes())
    return h.hexdigest()


def atomic_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def encode_side(z, source, sample_id, vocab, seed=0):
    if int(z['cache_version']) != 2 or str(z['ids_kind']) != 'global':
        raise ValueError('requires cache v2 with finalized global token IDs')
    all_edges, coverage = [], Counter()
    fn, base, footer = 0, 0, None
    with gzip.open(source, 'rt', encoding='utf-8') as stream:
        header = json.loads(next(stream))
        if (header.get('record'), header.get('sha256'), header.get('schema')) != ('header', sample_id, 'ida-cfg-research/2.1'):
            raise ValueError('export schema or identity mismatch')
        if header.get('analysis_options', {}).get('processor') != 'metapc':
            raise ValueError('requires x86/x64 metapc export')
        for line in stream:
            # Skip large strings/orphan records without decoding them.
            if line.startswith('{"record":"function"') or line.startswith('{"record": "function"'):
                record = json.loads(line)
            elif '"footer"' in line[:40]:
                footer = json.loads(line)
                continue
            else:
                continue
            blocks = sorted(record['blocks'], key=lambda b: b['start_rva'])
            if fn >= len(z['fn_flags']):
                raise ValueError('extra function in export')
            flags = int(bool(record['library'])) | (int(bool(record['thunk'])) << 1)
            if flags != z['fn_flags'][fn] or tuple(z['fn_blk_ptr'][fn:fn + 2]) != (base, base + len(blocks)):
                raise ValueError(f'function alignment mismatch at {fn}')
            rv = np.array([(b['start_rva'], b['end_rva']) for b in blocks]).reshape(-1, 2)
            if not np.array_equal(rv, z['blk_rva'][base:base + len(blocks)]):
                raise ValueError(f'block RVA/order mismatch at function {fn}')
            lengths = [len(b['instructions']) for b in blocks]
            if not np.array_equal(lengths, np.diff(z['blk_insn_ptr'][base:base + len(blocks) + 1])):
                raise ValueError('instruction alignment mismatch')
            # Verify mnemonic stream, not just block counts/addresses.
            lo, hi = z['blk_insn_ptr'][base], z['blk_insn_ptr'][base + len(blocks)]
            cached_mn = [vocab['mn'][i] for i in z['mn'][lo:hi]]
            if cached_mn != [ins['mnemonic'] for b in blocks for ins in b['instructions']]:
                raise ValueError('instruction content mismatch')
            local = {b['id']: i for i, b in enumerate(blocks)}
            if len(local) != len(blocks):
                raise ValueError('duplicate block ID')
            edges = [(local[a], local[b]) for a, b in record['edges']]
            # Cache CFG edges are stored in function/export order.
            start = int(coverage['cfg_edges'])
            expected = np.array(edges, dtype=np.int64).reshape(-1, 2) + base
            if not np.array_equal(z['cfg_edges'][start:start + len(edges)], expected):
                raise ValueError('CFG alignment mismatch')
            coverage['cfg_edges'] += len(edges)
            df, counts = defuse_edges(blocks, edges)
            all_edges.append(df + base)
            coverage.update(counts)
            fn += 1
            base += len(blocks)
    if not footer or not footer.get('complete'):
        raise ValueError('export missing complete footer')
    if fn != len(z['fn_flags']) or base != len(z['blk_rva']) or coverage['cfg_edges'] != len(z['cfg_edges']):
        raise ValueError('export/cache final counts mismatch')
    edges = np.concatenate(all_edges) if all_edges else np.empty((0, 2), dtype=np.int32)
    kinds = np.zeros(len(edges), dtype=np.int8)
    shuffled = shuffle_edges(edges, z['fn_blk_ptr'], sample_rng(sample_id, seed), kinds)
    tags, callee = function_tags(z, vocab['api'])
    return dict(schema=np.array(SCHEMA), sample_id=np.array(sample_id), control_seed=np.array(seed),
                df_edges=edges, df_kind=kinds, df_edges_shuffled=shuffled,
                df_kind_names=np.array(EDGE_KIND_NAMES), fn_tags=tags, fn_tags_callee=callee,
                tag_names=np.array(TAG_NAMES), fn_blk_ptr=z['fn_blk_ptr'], fn_flags=z['fn_flags'],
                coverage_json=np.array(json.dumps(dict(coverage), sort_keys=True)),
                df_shuffle_changed=np.array(int(np.count_nonzero(edges[:, 1] != shuffled[:, 1]))))


def build_one(cache, sample_id, source, vocab, seed=0, replace_stale=False):
    if not re.fullmatch(r'[0-9a-f]{64}', sample_id):
        raise ValueError('sample_id must be a lowercase SHA-256')
    cache = Path(cache)
    source = Path(source)
    path = cache / 'samples' / f'{sample_id}.npz'
    out = cache / 'side' / f'{sample_id}.npz'
    provenance = dict(schema=SCHEMA, sample_id=sample_id, cache_sha256=sha256(path),
                      source_sha256=sha256(source), vocab_sha256=sha256(cache / 'vocab.json'),
                      code_sha256=code_hash(), control_seed=seed)
    if out.exists():
        with np.load(out, allow_pickle=False) as old:
            if json.loads(str(old['provenance_json'])) == provenance:
                return {'sample_id': sample_id, 'status': 'skipped', 'path': str(out)}
        if not replace_stale:
            raise ValueError(f'stale side artifact {out}; use --replace-stale to replace only side output')
    with np.load(path, allow_pickle=False) as z:
        # NpzFile does not memoize reads: materialize once, never reload full token
        # arrays for every function/block of a million-block binary.
        keys = ('cache_version', 'ids_kind', 'fn_flags', 'fn_blk_ptr', 'blk_rva', 'blk_insn_ptr',
                'blk_api_ptr', 'mn', 'api', 'cfg_edges', 'call_edges')
        arrays = encode_side({k: z[k] for k in keys}, source, sample_id, vocab, seed)
    # Refuse publication if a cache rebuild/export replacement raced extraction.
    if sha256(path) != provenance['cache_sha256'] or sha256(source) != provenance['source_sha256']:
        raise ValueError('input changed during side extraction')
    arrays['provenance_json'] = np.array(json.dumps(provenance, sort_keys=True))
    atomic_npz(out, arrays)
    return {'sample_id': sample_id, 'status': 'ok', 'path': str(out),
            'functions': len(arrays['fn_tags']), 'tagged_functions': int(np.count_nonzero(arrays['fn_tags'].any(1))),
            'df_edges': len(arrays['df_edges']), 'df_shuffle_changed': int(arrays['df_shuffle_changed']),
            'coverage': json.loads(str(arrays['coverage_json']))}


def source_paths(roots):
    paths = {}
    for corpus, root in roots.items():
        manifest = root / 'samples.csv'
        if not manifest.exists():
            continue
        with manifest.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                paths[corpus, row['sample_id']] = root / row['cfg_path']
    return paths


def main():
    from build_cache import CORPUS_ROOTS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', type=Path, default=CACHE)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--sample-id', nargs='+')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--replace-stale', action='store_true', help='atomically replace stale side artifacts only')
    ap.add_argument('--corpus-root', action='append', default=[], metavar='NAME=PATH')
    args = ap.parse_args()
    if args.limit is not None and args.limit < 1:
        ap.error('--limit must be positive')
    roots = dict(CORPUS_ROOTS)
    for value in args.corpus_root:
        name, sep, path = value.partition('=')
        if not sep:
            ap.error('--corpus-root requires NAME=PATH')
        roots[name] = Path(path)
    paths = source_paths(roots)
    vocab = json.loads((args.cache / 'vocab.json').read_text(encoding='utf-8'))
    with (args.cache / 'index.csv').open(encoding='utf-8', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['status'] == 'ok']
    if args.sample_id:
        wanted = set(args.sample_id)
        rows = [r for r in rows if r['sample_id'] in wanted]
        if {r['sample_id'] for r in rows} != wanted:
            ap.error('requested sample missing from cache index')
    if args.limit:
        rows = rows[:args.limit]
    results = []
    for r in rows:
        sid = r['sample_id']
        try:
            result = build_one(args.cache, sid, paths[r.get('corpus') or 'mendeley', sid], vocab, args.seed, args.replace_stale)
        except Exception as exc:
            result = {'sample_id': sid, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
        results.append(result)
        print(json.dumps(result), flush=True)
    side = args.cache / 'side'
    side.mkdir(exist_ok=True)
    # One audit file per invocation, so a limited smoke never replaces a full-build summary.
    (side / f'build-{uuid.uuid4().hex}.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    if any(r['status'] == 'failed' for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
