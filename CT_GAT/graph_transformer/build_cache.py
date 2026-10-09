"""Stream the Mendeley_GNN_v1 IDA CFG exports into one npz per sample, plus index.csv and vocab.json.

python build_cache.py [--limit N] [--workers 4]

Idempotent: a sample whose npz loads and whose counts match samples.csv is skipped. Workers write
ids into a sorted per-sample string table; the finalize pass merges the tables into vocab.json
(sorted, so deterministic) and rewrites every npz to global ids.
"""
import argparse
import array
import csv
import gzip
import hashlib
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from config import CACHE

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'preprocessing'))
from token_mapping import BEHAVIOR_MAP, _match_api_in_operands, categorize_behavior, strip_opcode_prefixes  # noqa: E402

DOWNLOADS = Path('C:/Users/chaoa/Downloads')
BUNDLE = DOWNLOADS / 'ransomware-detection-masters-project/reports/google_drive_upload/2026-10-07/Mendeley_GNN_v1'
FOLDS = DOWNLOADS / 'rdmp-llm/results/family_holdout/folds_mendeley.csv'
SCHEMA = 'ida-cfg-research/2.1'

OPERAND_ABBREV = {'register': 'r', 'immediate': 'i', 'phrase': 'p', 'displacement': 'd',
                  'memory': 'm', 'code_target': 'c', 'processor_specific': 's'}
TOKEN_KINDS = ('mn', 'sig', 'api')
COUNT_FIELDS = ('num_functions', 'num_blocks', 'num_edges', 'num_instructions')
INDEX_FIELDS = ('sample_id', 'label', 'split', 'arch', 'family', 'identity_group', 'fold', 'fold_sha256',
                *COUNT_FIELDS, 'num_api', 'num_call_edges', 'status', 'error')


def eligible_rows():
    with (BUNDLE / 'samples.csv').open(encoding='utf-8', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['recommended_split'] and r['representation'] != 'upx_packed']
    with FOLDS.open(encoding='utf-8', newline='') as f:
        folds = {r['sha256']: r for r in csv.DictReader(f)}
    with (BUNDLE / 'provenance/identity_aliases.csv').open(encoding='utf-8', newline='') as f:
        aliases = {r['sha256']: r['cohort_sha256'] for r in csv.DictReader(f)}
    for r in rows:
        key = r['sample_id'] if r['sample_id'] in folds else aliases.get(r['sample_id'])
        if key not in folds:
            raise SystemExit(f"{r['sample_id']} joins neither folds_mendeley.csv nor identity_aliases.csv")
        r['fold'], r['fold_sha256'] = folds[key]['fold'], key
    return rows


def npz_path(sample_id):
    return CACHE / 'samples' / f'{sample_id}.npz'


def stored_counts(z):
    return tuple(int(z[k]) for k in COUNT_FIELDS)


def expected_counts(row):
    return tuple(int(row[k]) for k in COUNT_FIELDS)


def verified(row, want_kind=None):
    path = npz_path(row['sample_id'])
    if not path.exists():
        return False
    try:
        with np.load(path) as z:
            return stored_counts(z) == expected_counts(row) and (want_kind is None or str(z['ids_kind']) == want_kind)
    except Exception:
        return False


class Interner:
    def __init__(self):
        self.ids = {}

    def __call__(self, key):
        i = self.ids.get(key)
        if i is None:
            i = self.ids[key] = len(self.ids)
        return i

    def sorted_table(self, as_string=lambda k: k):
        strings = [as_string(k) for k in self.ids]
        order = sorted(range(len(strings)), key=strings.__getitem__)
        remap = np.empty(len(strings), dtype=np.int32)
        remap[np.asarray(order, dtype=np.int64)] = np.arange(len(strings), dtype=np.int32)
        return np.array([strings[i] for i in order], dtype=np.str_), remap


def sig_string(key):
    mnemonic, cats = key
    return mnemonic + ' ' + ','.join(cats) if cats else mnemonic


def api_string(symbol):
    name = symbol.get('name')
    return symbol['dll'].lower() + '!' + (name if name else '#' + str(symbol.get('ordinal')))


def encode_sample(row):
    """Stream one export, one function record at a time, into flat arrays."""
    sample_id = row['sample_id']
    mn, sig, beh, api = array.array('i'), array.array('i'), array.array('b'), array.array('i')
    blk_insn_ptr, blk_api_ptr, blk_feat = array.array('i', [0]), array.array('i', [0]), array.array('f')
    fn_blk_ptr, fn_flags, cfg_edges = array.array('i', [0]), array.array('b'), array.array('i')
    pending_calls = array.array('q')
    fn_index_by_rva = {}
    mn_ids, sig_ids, api_ids = Interner(), Interner(), Interner()
    crypto_beh = {}
    import_beh = {}
    footer = None

    with gzip.open(BUNDLE / row['cfg_path'], 'rt', encoding='utf-8') as stream:
        header = json.loads(stream.readline())
        if header.get('record') != 'header' or header.get('sha256') != sample_id:
            raise ValueError('header identity mismatch')
        if header.get('schema') != SCHEMA:
            raise ValueError(f"schema {header.get('schema')}")
        imports = header.get('pe', {}).get('imports', [])
        import_strings = [api_string(s) for s in imports]
        import_names = [s.get('name') or '' for s in imports]
        del header

        for line in stream:
            if line.startswith('{"record":"footer"'):
                footer = json.loads(line)
                continue
            if not line.startswith('{"record":"function"'):
                continue
            record = json.loads(line)
            fn_index = len(fn_flags)
            fn_index_by_rva.setdefault(record['rva'], fn_index)
            fn_flags.append(int(bool(record['library'])) | (int(bool(record['thunk'])) << 1))
            blocks = sorted(record['blocks'], key=lambda b: b['start_rva'])
            base = len(blk_insn_ptr) - 1
            local = {b['id']: base + i for i, b in enumerate(blocks)}
            if len(local) != len(blocks):
                raise ValueError(f'duplicate block id in function {fn_index}')
            indeg, outdeg = [0] * len(blocks), [0] * len(blocks)
            for src, dst in record['edges']:
                a, b = local[src], local[dst]
                cfg_edges.append(a)
                cfg_edges.append(b)
                outdeg[a - base] += 1
                indeg[b - base] += 1
            for i, block in enumerate(blocks):
                calls = refs = 0
                for insn in block['instructions']:
                    mnemonic = insn['mnemonic']
                    mn.append(mn_ids(mnemonic))
                    sig.append(sig_ids((mnemonic, tuple(OPERAND_ABBREV[o['category']] for o in insn['operands']))))
                    ids = insn['import_ids']
                    if ids:
                        key = tuple(ids)
                        b_id = import_beh.get(key)
                        if b_id is None:
                            matched = _match_api_in_operands([import_names[j] for j in ids])
                            b_id = import_beh[key] = categorize_behavior(strip_opcode_prefixes(mnemonic), 0, matched)
                        for j in ids:
                            api.append(api_ids(import_strings[j]))
                        refs += len(ids)
                    else:
                        b_id = crypto_beh.get(mnemonic)
                        if b_id is None:
                            b_id = crypto_beh[mnemonic] = categorize_behavior(strip_opcode_prefixes(mnemonic), 0, None)
                    beh.append(b_id)
                    if insn['is_call']:
                        calls += 1
                        for target in insn['code_refs']:
                            pending_calls.append(fn_index)
                            pending_calls.append(target)
                n = len(block['instructions'])
                blk_insn_ptr.append(len(mn))
                blk_api_ptr.append(len(api))
                blk_feat.extend((math.log1p(n), math.log1p(calls), math.log1p(refs), float(block['external']),
                                 math.log1p(indeg[i]), math.log1p(outdeg[i])))
            fn_blk_ptr.append(len(blk_insn_ptr) - 1)
            del record, blocks, local

    if not footer or not footer.get('complete'):
        raise ValueError('incomplete export (footer missing or complete=false)')

    calls = np.frombuffer(pending_calls, dtype=np.int64).reshape(-1, 2)
    callee = np.array([fn_index_by_rva.get(int(t), -1) for t in calls[:, 1]], dtype=np.int64)
    keep = callee >= 0
    call_edges = np.unique(np.stack([calls[keep, 0], callee[keep]], axis=1), axis=0).astype(np.int32).reshape(-1, 2)

    out = dict(
        mn=np.frombuffer(mn, dtype=np.int32), sig=np.frombuffer(sig, dtype=np.int32),
        beh=np.frombuffer(beh, dtype=np.int8), api=np.frombuffer(api, dtype=np.int32),
        blk_insn_ptr=np.frombuffer(blk_insn_ptr, dtype=np.int32), blk_api_ptr=np.frombuffer(blk_api_ptr, dtype=np.int32),
        blk_feat=np.frombuffer(blk_feat, dtype=np.float32).reshape(-1, 6).astype(np.float16),
        fn_blk_ptr=np.frombuffer(fn_blk_ptr, dtype=np.int32), fn_flags=np.frombuffer(fn_flags, dtype=np.int8),
        cfg_edges=np.frombuffer(cfg_edges, dtype=np.int32).reshape(-1, 2), call_edges=call_edges,
    )
    for kind, interner, as_string in (('mn', mn_ids, str), ('sig', sig_ids, sig_string), ('api', api_ids, str)):
        table, remap = interner.sorted_table(as_string)
        out[kind] = remap[out[kind]] if len(out[kind]) else out[kind].copy()
        out[f'{kind}_strings'] = table
    out.update(num_functions=len(fn_flags), num_blocks=len(blk_insn_ptr) - 1, num_edges=len(cfg_edges) // 2,
               num_instructions=len(mn), ids_kind='local', vocab_sha='')
    return out


def write_npz(path, arrays):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def build_one(row):
    start = time.time()
    try:
        arrays = encode_sample(row)
        got = (arrays['num_functions'], arrays['num_blocks'], arrays['num_edges'], arrays['num_instructions'])
        if got != expected_counts(row):
            raise ValueError(f'counts {got} != samples.csv {expected_counts(row)}')
        write_npz(npz_path(row['sample_id']), arrays)
        return row['sample_id'], 'ok', '', time.time() - start
    except Exception as exc:
        return row['sample_id'], 'failed', f'{type(exc).__name__}: {exc}', time.time() - start


def finalize(rows):
    """Merge per-sample string tables into vocab.json and rewrite ids to global, skipping npz already current."""
    built = [r for r in rows if npz_path(r['sample_id']).exists()]
    tables = {k: set() for k in TOKEN_KINDS}
    for r in built:
        with np.load(npz_path(r['sample_id'])) as z:
            for k in TOKEN_KINDS:
                tables[k].update(z[f'{k}_strings'].tolist())
    vocab = {k: sorted(tables[k]) for k in TOKEN_KINDS}
    vocab['beh'] = sorted(BEHAVIOR_MAP, key=BEHAVIOR_MAP.get)
    text = json.dumps(vocab, indent=0, ensure_ascii=False)
    vocab_sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    tmp = CACHE / 'vocab.json.tmp'
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, CACHE / 'vocab.json')
    lookup = {k: {s: i for i, s in enumerate(vocab[k])} for k in TOKEN_KINDS}
    rewritten = 0
    for r in built:
        path = npz_path(r['sample_id'])
        with np.load(path) as z:
            if str(z['ids_kind']) == 'global' and str(z['vocab_sha']) == vocab_sha:
                continue
            arrays = {k: z[k] for k in z.files}
        for k in TOKEN_KINDS:
            ids = arrays[k]
            if str(arrays['ids_kind']) == 'global':
                # Sorted vocab and sorted per-sample tables share one order, so rank among present ids recovers the local id.
                ids = np.searchsorted(np.unique(ids), ids).astype(np.int32)
            to_global = np.array([lookup[k][s] for s in arrays[f'{k}_strings'].tolist()], dtype=np.int32)
            arrays[k] = to_global[ids] if len(ids) else ids
            arrays[f'{k}_u'] = to_global
        arrays['ids_kind'], arrays['vocab_sha'] = 'global', vocab_sha
        write_npz(path, arrays)
        rewritten += 1
    return {k: len(vocab[k]) for k in vocab}, vocab_sha, rewritten


def write_index(rows, results):
    out = []
    for r in rows:
        status, error = results.get(r['sample_id'], ('missing', 'not built'))
        entry = {'sample_id': r['sample_id'], 'label': r['label'], 'split': r['recommended_split'],
                 'arch': r['architecture'], 'family': r['family'], 'identity_group': r['identity_group'],
                 'fold': r['fold'], 'fold_sha256': r['fold_sha256'], 'status': status, 'error': error}
        for k in COUNT_FIELDS:
            entry[k] = r[k]
        entry['num_api'] = entry['num_call_edges'] = ''
        if status == 'ok':
            with np.load(npz_path(r['sample_id'])) as z:
                entry['num_api'], entry['num_call_edges'] = len(z['api']), len(z['call_edges'])
        out.append(entry)
    tmp = CACHE / 'index.csv.tmp'
    with tmp.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(out)
    os.replace(tmp, CACHE / 'index.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, help='build only N samples, half of each class')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    if args.workers > 4:
        raise SystemExit('at most 4 workers')
    (CACHE / 'samples').mkdir(parents=True, exist_ok=True)
    rows = eligible_rows()
    if args.limit:
        by_class = {c: sorted((r for r in rows if r['label'] == c), key=lambda r: r['sample_id']) for c in ('0', '1')}
        rows = by_class['0'][:args.limit // 2] + by_class['1'][:args.limit - args.limit // 2]
    todo = sorted((r for r in rows if not verified(r)), key=lambda r: -int(r['num_instructions']))
    todo_ids = {r['sample_id'] for r in todo}
    print(f'{len(rows)} eligible, {len(rows) - len(todo)} already built, {len(todo)} to build', flush=True)
    results = {r['sample_id']: ('ok', '') for r in rows if r['sample_id'] not in todo_ids}
    start = time.time()
    with Pool(args.workers) as pool:
        for n, (sample_id, status, error, secs) in enumerate(pool.imap_unordered(build_one, todo, chunksize=1), 1):
            results[sample_id] = (status, error)
            print(f'[{n}/{len(todo)}] {time.time() - start:7.0f}s {sample_id} {status} {secs:.1f}s {error}', flush=True)
    build_secs = time.time() - start
    sizes, vocab_sha, rewritten = finalize([r for r in rows if results[r['sample_id']][0] == 'ok'])
    write_index(rows, results)
    failed = {s: e for s, (st, e) in results.items() if st != 'ok'}
    summary = {'eligible': len(rows), 'ok': len(rows) - len(failed), 'failed': failed, 'vocab_sizes': sizes,
               'vocab_sha': vocab_sha, 'npz_rewritten': rewritten, 'build_seconds': round(build_secs, 1),
               'total_seconds': round(time.time() - start, 1),
               'disk_bytes': sum(p.stat().st_size for p in (CACHE / 'samples').glob('*.npz'))}
    (CACHE / 'build_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
