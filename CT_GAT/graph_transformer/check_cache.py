"""Check cache invariants for every built sample, and compare a few samples against the bundle's own reader.

python check_cache.py [--reference N]
"""
import argparse
import csv
import hashlib
import importlib.util
import json

import numpy as np

from build_cache import BUNDLE, CACHE, COUNT_FIELDS, TOKEN_KINDS, npz_path


def check_sample(z, row, vocab_sizes, vocab_sha):
    errors = []

    def need(ok, what):
        if not ok:
            errors.append(what)

    B, F, I, A = len(z['blk_feat']), len(z['fn_flags']), len(z['mn']), len(z['api'])
    got = (F, B, len(z['cfg_edges']), I)
    need(got == tuple(int(row[k]) for k in COUNT_FIELDS), f'counts {got} vs samples.csv')
    need(str(z['ids_kind']) == 'global' and str(z['vocab_sha']) == vocab_sha, 'ids not global for current vocab')
    for name, ptr, end in (('blk_insn_ptr', z['blk_insn_ptr'], I), ('blk_api_ptr', z['blk_api_ptr'], A),
                           ('fn_blk_ptr', z['fn_blk_ptr'], B)):
        n = F if name == 'fn_blk_ptr' else B
        need(len(ptr) == n + 1 and ptr[0] == 0 and ptr[-1] == end and np.all(np.diff(ptr) >= 0), f'{name} not a valid ptr')
    need(len(z['sig']) == I and len(z['beh']) == I, 'sig/beh length')
    for k in TOKEN_KINDS:
        ids = z[k]
        need(len(ids) == 0 or (ids.min() >= 0 and ids.max() < vocab_sizes[k]), f'{k} ids out of vocab')
        need(np.array_equal(np.unique(ids), z[f'{k}_u']), f'{k}_u != unique({k})')
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
    return errors


def reference_compare(row, vocab):
    """Rebuild one sample with the bundle's dataset.py reader and compare tokens, APIs and features."""
    spec = importlib.util.spec_from_file_location('bundle_dataset', BUNDLE / 'dataset.py')
    ds = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ds)
    z = np.load(npz_path(row['sample_id']))
    mn = [vocab['mn'][i] for i in z['mn']]
    api = [vocab['api'][i] for i in z['api']]
    blk = 0
    for f, graph in enumerate(ds.iter_file_functions(BUNDLE / row['cfg_path'], row['sample_id'])):
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
    ap.add_argument('--reference', type=int, default=3, help='samples to compare against the bundle reader')
    args = ap.parse_args()
    text = (CACHE / 'vocab.json').read_text(encoding='utf-8')
    vocab, vocab_sha = json.loads(text), hashlib.sha256(text.encode('utf-8')).hexdigest()
    sizes = {k: len(v) for k, v in vocab.items()}
    with (BUNDLE / 'samples.csv').open(encoding='utf-8', newline='') as f:
        samples = {r['sample_id']: r for r in csv.DictReader(f)}
    with (CACHE / 'index.csv').open(encoding='utf-8', newline='') as f:
        index = [r for r in csv.DictReader(f) if r['status'] == 'ok']
    bad = {}
    for r in index:
        with np.load(npz_path(r['sample_id'])) as z:
            errors = check_sample(z, samples[r['sample_id']], sizes, vocab_sha)
        if errors:
            bad[r['sample_id']] = errors
    print(json.dumps({'checked': len(index), 'failed': len(bad), 'failures': dict(list(bad.items())[:20])}, indent=2))
    smallest = sorted(index, key=lambda r: int(r['num_blocks']))
    picks = [smallest[0], smallest[len(smallest) // 2]] + [r for r in smallest if r['label'] != smallest[0]['label']][:1]
    for r in picks[:args.reference]:
        print('reference match', r['sample_id'], r['label'], reference_compare(samples[r['sample_id']], vocab), 'blocks')
    raise SystemExit(1 if bad else 0)


if __name__ == '__main__':
    main()
