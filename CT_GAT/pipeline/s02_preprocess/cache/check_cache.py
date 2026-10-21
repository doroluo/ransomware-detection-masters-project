"""Check cache invariants for every built sample, and compare a few samples against the bundle's own reader.

python -m s02_preprocess.cache.check_cache [--cache NAME] [--reference N] [--compare-v1 N] [--compare-base DIR]

--cache NAME checks the cache of that dataset (common/paths.py reads it from the command line, as for build_cache); without it
the cache is GT_CACHE or the first dataset of the registry.

Every built sample is checked against its corpus's samples.csv and the vocab generation its ids belong to.
--compare-v1 also requires N random Mendeley samples' v1 arrays to be byte-identical to the v1 cache, whose
vocab.json must be a prefix of this one. --compare-base requires every npz file that DIR built to be
byte-identical here, and its index.csv rows to reappear with the same values.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from common.paths import registry
from s02_preprocess.cache.build_cache import (CACHE, CACHE_VERSION, COUNT_FIELDS, EDGE_KINDS, FCB_EXTERN, TOKEN_KINDS,
                         corpus_samples, npz_path, read_csv, read_vocab)

V1_CACHE = CACHE.with_name('graph_transformer_cache')
V1_ARRAYS = ('mn', 'sig', 'beh', 'api', 'blk_insn_ptr', 'blk_api_ptr', 'blk_feat', 'fn_blk_ptr', 'fn_flags',
             'cfg_edges', 'call_edges', 'mn_u', 'sig_u', 'api_u')


def is_acyclic(n, edges):
    """Kahn's algorithm, one frontier at a time."""
    indeg = np.bincount(edges[:, 1], minlength=n)
    order = np.argsort(edges[:, 0], kind='stable')
    dst = edges[order, 1]
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(edges[:, 0], minlength=n), out=ptr[1:])
    frontier, done = np.flatnonzero(indeg == 0), 0
    while len(frontier):
        done += len(frontier)
        counts = ptr[frontier + 1] - ptr[frontier]
        take = np.repeat(ptr[frontier] - np.cumsum(counts) + counts, counts) + np.arange(counts.sum())
        hit, times = np.unique(dst[take], return_counts=True)
        indeg[hit] -= times
        frontier = hit[indeg[hit] == 0]
    return done == n


def check_sample(z, row, vocab, generations):
    """generations: vocab_sha -> sizes of each vocab this cache's vocab.json grew through."""
    errors = []

    def need(ok, what):
        if not ok:
            errors.append(what)

    B, F, I, A = len(z['blk_feat']), len(z['fn_flags']), len(z['mn']), len(z['api'])
    got = (F, B, len(z['cfg_edges']), I)
    need(got == tuple(int(row[k]) for k in COUNT_FIELDS), f'counts {got} vs samples.csv')
    vocab_sizes = generations.get(str(z['vocab_sha']))
    need(str(z['ids_kind']) == 'global' and vocab_sizes is not None, 'ids not global for a vocab of this cache')
    if errors:
        return errors
    for name, ptr, end in (('blk_insn_ptr', z['blk_insn_ptr'], I), ('blk_api_ptr', z['blk_api_ptr'], A),
                           ('fn_blk_ptr', z['fn_blk_ptr'], B)):
        n = F if name == 'fn_blk_ptr' else B
        need(len(ptr) == n + 1 and ptr[0] == 0 and ptr[-1] == end and np.all(np.diff(ptr) >= 0), f'{name} not a valid ptr')
    need(len(z['sig']) == I and len(z['beh']) == I, 'sig/beh length')
    for k in TOKEN_KINDS:
        ids = z[k]
        need(len(ids) == 0 or (ids.min() >= 0 and ids.max() < vocab_sizes[k]), f'{k} ids out of vocab')
        need(np.array_equal(np.unique(ids), z[f'{k}_u']), f'{k}_u != unique({k})')
        need(z[f'{k}_strings'].tolist() == [vocab[k][i] for i in z[f'{k}_u'].tolist()], f'{k}_strings != vocab[{k}_u]')
    need(len(z['beh']) == 0 or (z['beh'].min() >= 0 and z['beh'].max() < vocab_sizes['beh']), 'beh out of range')
    e, c = z['cfg_edges'], z['call_edges']
    need(e.shape[1:] == (2,) and (len(e) == 0 or (e.min() >= 0 and e.max() < B)), 'cfg edge index out of range')
    need(c.shape[1:] == (2,) and (len(c) == 0 or (c.min() >= 0 and c.max() < F)), 'call edge index out of range')
    fn_of_block = np.repeat(np.arange(F), np.diff(z['fn_blk_ptr']))
    need(len(e) == 0 or np.array_equal(fn_of_block[e[:, 0]], fn_of_block[e[:, 1]]), 'cfg edge crosses functions')
    feat = z['blk_feat'].astype(np.float32)
    need(feat.shape == (B, 6) and np.isfinite(feat).all(), 'blk_feat shape or non-finite')
    want = np.stack([np.log1p(np.diff(z['blk_insn_ptr'])), np.log1p(np.diff(z['blk_api_ptr'])),
                     np.log1p(np.bincount(e[:, 1], minlength=B)), np.log1p(np.bincount(e[:, 0], minlength=B))], 1)
    need(np.allclose(feat[:, [0, 2, 4, 5]], want, atol=1e-2), 'blk_feat disagrees with ptrs/edges')
    need(np.all(np.diff(fn_of_block) >= 0), 'blocks not grouped by function')

    need(int(z['cache_version']) == CACHE_VERSION, 'cache_version')
    kind, back, back_addr, btype, rva = z['edge_kind'], z['edge_back'], z['edge_back_addr'], z['blk_type'], z['blk_rva']
    E = len(e)
    need(rva.shape == (B, 2) and rva.dtype == np.uint32 and len(btype) == B, 'blk_rva/blk_type shape')
    need(all(len(a) == E for a in (kind, back, back_addr)), 'edge array length')
    need(len(kind) == 0 or (kind.min() >= 0 and kind.max() < len(EDGE_KINDS)), 'edge_kind out of range')
    need(np.isin(back, (0, 1)).all() and np.isin(back_addr, (0, 1)).all(), 'edge_back values not 0/1')
    if errors:
        return errors
    src, dst = e[:, 0], e[:, 1]
    need(np.array_equal(kind == EDGE_KINDS.index('extern'), btype[dst] == FCB_EXTERN), 'extern edges != edges into type 6')
    adjacent = rva[dst, 0] == rva[src, 1]
    need(np.array_equal(kind == EDGE_KINDS.index('fallthrough'), adjacent & (btype[dst] != FCB_EXTERN)),
         'fallthrough edges != RVA-adjacent non-extern edges')
    need(np.array_equal(back_addr.astype(bool), rva[dst, 0] <= rva[src, 0]), 'edge_back_addr != RVA order')
    need(np.all(rva[:, 0] <= rva[:, 1]), 'blk_rva start > end')
    need(np.all((np.diff(rva[:, 0].astype(np.int64)) > 0) | (np.diff(fn_of_block) > 0)),
         'blocks not address-sorted within function')
    need(is_acyclic(B, e[back == 0].astype(np.int64)), 'cycle remains after removing back edges')
    return errors


def vocab_prefix(old, new):
    """Every kind of `old` is a prefix of the same kind of `new`."""
    return all(new[k][:len(v)] == v for k, v in old.items())


def compare_base(base, index):
    """Rows of base's index.csv that are missing here or differ in any of base's columns, and built npz whose
    bytes differ."""
    here = {r['sample_id']: r for r in index}
    rows = read_csv(base / 'index.csv')
    row_diff = [r['sample_id'] for r in rows if r['sample_id'] not in here
                or any(here[r['sample_id']][k] != v for k, v in r.items())]
    file_diff = [r['sample_id'] for r in rows if r['status'] == 'ok'
                 and (base / 'samples' / f"{r['sample_id']}.npz").read_bytes() != npz_path(r['sample_id']).read_bytes()]
    return len(rows), row_diff, file_diff


def compare_v1(sample_id):
    """Names of v1 arrays that are not byte-identical between the v1 cache and this one."""
    with np.load(V1_CACHE / 'samples' / f'{sample_id}.npz') as old, np.load(npz_path(sample_id)) as new:
        return [k for k in V1_ARRAYS if old[k].dtype != new[k].dtype or old[k].shape != new[k].shape
                or old[k].tobytes() != new[k].tobytes()]


def reference_compare(row, vocab):
    """Rebuild one sample with the reader shipped beside its samples.csv (the bundle's dataset.py or a collection's
    gnn_dataset.py, the same file) and compare tokens, APIs and features."""
    root = Path(row['root'])
    spec = importlib.util.spec_from_file_location('bundle_dataset', next(f for f in (root / 'dataset.py', root / 'gnn_dataset.py') if f.exists()))
    ds = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ds)
    z = np.load(npz_path(row['sample_id']))
    mn = [vocab['mn'][i] for i in z['mn']]
    api = [vocab['api'][i] for i in z['api']]
    blk = 0
    for f, graph in enumerate(ds.iter_file_functions(root / row['cfg_path'], row['sample_id'])):
        order = sorted(range(graph['num_nodes']), key=lambda i: graph['node_start_rva'][i])
        assert z['fn_blk_ptr'][f] == blk, 'function block offset'
        for i in order:
            a, b = z['blk_insn_ptr'][blk], z['blk_insn_ptr'][blk + 1]
            assert mn[a:b] == graph['node_tokens'][i], f'mnemonics differ at block {blk}'
            ref_api = [n.split('!', 1)[0].lower() + '!' + n.split('!', 1)[1] for ins in graph['node_api_references'][i] for n in ins]
            assert api[z['blk_api_ptr'][blk]:z['blk_api_ptr'][blk + 1]] == ref_api, f'apis differ at block {blk}'
            assert np.allclose(z['blk_feat'][blk].astype(np.float32), graph['x'][i], atol=1e-2), f'features differ at block {blk}'
            blk += 1
    assert blk == len(z['blk_feat']), 'block total'
    return blk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', choices=sorted(registry()), metavar='NAME',
                    help='dataset whose cache to check (default: GT_CACHE, else the registry default)')
    ap.add_argument('--reference', type=int, default=3, help='samples to compare against the bundle reader')
    ap.add_argument('--compare-v1', type=int, default=0,
                    help='random Mendeley samples to compare byte-for-byte with the v1 cache')
    ap.add_argument('--compare-base', type=Path, help='cache whose npz and index rows must reappear unchanged')
    args = ap.parse_args()
    vocab, generations = read_vocab()
    generations = {g['vocab_sha']: g['sizes'] for g in generations}
    full_index = read_csv(CACHE / 'index.csv')
    index = [r for r in full_index if r['status'] == 'ok']
    corpora = dict.fromkeys(r.get('corpus', 'mendeley') for r in index)
    samples = {r['sample_id']: r for c in corpora for r in corpus_samples(c)}
    bad = {}
    for r in index:
        with np.load(npz_path(r['sample_id'])) as z:
            errors = check_sample(z, samples[r['sample_id']], vocab, generations)
        if errors:
            bad[r['sample_id']] = errors
    print(json.dumps({'checked': len(index), 'corpora': {c: sum(r.get('corpus', 'mendeley') == c for r in index)
                                                         for c in corpora},
                      'vocab_generations': len(generations), 'failed': len(bad),
                      'failures': dict(list(bad.items())[:20])}, indent=2))
    mendeley = [r for r in index if r.get('corpus', 'mendeley') == 'mendeley']
    if args.compare_v1:
        v1_prefix = vocab_prefix(json.loads((V1_CACHE / 'vocab.json').read_text(encoding='utf-8')), vocab)
        picks_v1 = np.random.default_rng(0).choice(len(mendeley), min(args.compare_v1, len(mendeley)), replace=False)
        differ = {mendeley[i]['sample_id']: d for i in picks_v1 if (d := compare_v1(mendeley[i]['sample_id']))}
        print(json.dumps({'compare_v1': len(picks_v1), 'v1_vocab_is_prefix': v1_prefix, 'differing': differ}, indent=2))
        if differ or not v1_prefix:
            bad['compare_v1'] = ['v1 mismatch']
    if args.compare_base:
        n, row_diff, file_diff = compare_base(args.compare_base, full_index)
        base_prefix = vocab_prefix(json.loads((args.compare_base / 'vocab.json').read_text(encoding='utf-8')), vocab)
        print(json.dumps({'compare_base': str(args.compare_base), 'rows': n, 'rows_differing': row_diff[:20],
                          'npz_bytes_differing': file_diff[:20], 'base_vocab_is_prefix': base_prefix}, indent=2))
        if row_diff or file_diff or not base_prefix:
            bad['compare_base'] = ['base mismatch']
    smallest = sorted(mendeley, key=lambda r: int(r['num_blocks']))
    picks = [smallest[0], smallest[len(smallest) // 2]] + [r for r in smallest if r['label'] != smallest[0]['label']][:1]
    for r in picks[:args.reference]:
        print('reference match', r['sample_id'], r['label'], reference_compare(samples[r['sample_id']], vocab), 'blocks')
    raise SystemExit(1 if bad else 0)


if __name__ == '__main__':
    main()
