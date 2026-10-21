"""Stream IDA CFG exports (the Mendeley_GNN_v1 bundle or the Mendeley_Ransomware and Mendeley_Goodware collections, and
the other collection-cfg-asm/1 collections) into one npz per sample, plus index.csv and vocab.json.

python -m s02_preprocess.cache.build_cache [--corpora mendeley ...] [--base-cache DIR] [--limit N] [--workers 4]
(from CT_GAT/pipeline; the cache is GT_CACHE, or the dataset picked by --cache <name> / the protocol, see common/paths.py)

Idempotent: a sample whose npz loads and whose counts match samples.csv is skipped. Workers write
ids into a sorted per-sample string table; the finalize pass appends the strings vocab.json lacks
(sorted among themselves) and rewrites the new npz to global ids. vocab.json only grows, so global ids
of npz built against an earlier vocab stay valid; vocab_generations.json lists every vocab's sha and sizes.

Corpora other than mendeley go into a new cache (GT_CACHE), never the default one. --base-cache copies
that cache's vocab.json and the npz of rows it already holds, so those files stay byte-identical.

Inputs (common/paths.py `source_for`; GT_BUNDLE, GT_MENDELEY_COLLECTIONS, GT_COLLECTIONS, GT_FOLDS, GT_FOLDS_V3 or "sources"
in datasets.json): Mendeley comes from the Mendeley_GNN_v1 bundle when one is configured, else from the two collections
under `mendeley_collections` (rows sorted by sample_id as the bundle's are, cfg_path relative to the collection of the row;
the cache comes out the same). The other collections, the Mendeley fold CSV and the v3 fold files are the remaining inputs.

Behaviour tags: each instruction's behaviour id comes from `s02_preprocess/token_mapping.py` (`categorize_behavior`,
`_match_api_in_operands`, `strip_opcode_prefixes`, `BEHAVIOR_MAP`), the module the caches were built with. It is Yanping's
file, copied unchanged from origin/yanping (tests/test_token_mapping.py pins its sha256). It is not
`s02_preprocess/tokenize_asm_gz.py`: that module's `_matched_api` assigns a different id to 57 of the 86,127 API names of
the combined cache's vocabulary (44 of 12,327 in the v2 cache), so a cache built with it would not be this cache.
"""
import argparse
import array
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from common.config import CACHE, DEFAULT_CACHE
from common.paths import UNSET, MissingSource, datasets_file, source_for
from s02_preprocess import token_mapping

CORPUS_NAMES = ('mendeley', 'ransomware_balanced', 'balanced_goodware')
MENDELEY_COLLECTIONS = ('Mendeley_Ransomware', 'Mendeley_Goodware')
SCHEMA = 'ida-cfg-research/2.1'

OPERAND_ABBREV = {'register': 'r', 'immediate': 'i', 'phrase': 'p', 'displacement': 'd',
                  'memory': 'm', 'code_target': 'c', 'processor_specific': 's'}
TOKEN_KINDS = ('mn', 'sig', 'api')
COUNT_FIELDS = ('num_functions', 'num_blocks', 'num_edges', 'num_instructions')
CACHE_VERSION = 2
EDGE_KINDS = ('fallthrough', 'cond_taken', 'jmp', 'indirect', 'extern', 'other')
FCB_INDJUMP, FCB_EXTERN = 1, 6
INDEX_FIELDS = ('sample_id', 'label', 'split', 'arch', 'family', 'identity_group', 'fold', 'fold_sha256',
                *COUNT_FIELDS, 'num_api', 'num_call_edges', 'status', 'error',
                'corpus', 'fold_v3', 'fold_v3_sha256', 'arch_matched')


def behavior_rules():
    """The `token_mapping` module the caches were built with (the file beside this package, unchanged)."""
    return token_mapping


def configured_bundle():
    """The Mendeley_GNN_v1 bundle when GT_BUNDLE or datasets.json names one, else None."""
    try:
        return source_for('bundle')
    except MissingSource:
        return None


def corpus_roots(corpus):
    """The folders of a corpus's samples.csv, cfg files and provenance/: the bundle, or for Mendeley without a configured
    bundle its two collections (each row's cfg_path is relative to its own folder)."""
    if corpus != 'mendeley':
        return [source_for('collections') / {'ransomware_balanced': 'Ransomware_Balanced', 'balanced_goodware': 'Balanced_Goodware'}[corpus]]
    bundle = configured_bundle()
    if bundle:
        return [bundle]
    try:
        base = source_for('mendeley_collections')
    except MissingSource:
        base = None
    roots = [base / name for name in MENDELEY_COLLECTIONS] if base else []
    if not roots or not all((root / 'samples.csv').is_file() for root in roots):
        raise MissingSource('no Mendeley data: set GT_BUNDLE (or "bundle" under "sources" in '
                            f'{datasets_file()}) to the Mendeley_GNN_v1 bundle, or GT_MENDELEY_COLLECTIONS (or '
                            '"mendeley_collections") to the folder holding Mendeley_Ransomware/ and Mendeley_Goodware/'
                            + (f'; {base} lacks one of them' if base else ''))
    return roots


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def read_aliases(root):
    """sha256 -> alias record (cohort_sha256, current_representation, ...) from the bundle's csv or a collection's json."""
    path = root / 'provenance/identity_aliases.csv'
    rows = read_csv(path) if path.exists() else json.loads((root / 'provenance/identity_aliases.json').read_text(encoding='utf-8'))
    return {r['sha256']: r for r in rows}


def corpus_aliases(corpus):
    return {sha: rec for root in corpus_roots(corpus) for sha, rec in read_aliases(root).items()}


def corpus_samples(corpus):
    """The samples.csv rows of a corpus, each with its `corpus` and the `root` its cfg_path is relative to. Mendeley's two
    collections merge sorted by sample_id, the order of the bundle's samples.csv."""
    roots = corpus_roots(corpus)
    rows = []
    for root in roots:
        for r in read_csv(root / 'samples.csv'):
            r['corpus'], r['root'] = corpus, str(root)
            rows.append(r)
    return sorted(rows, key=lambda r: r['sample_id']) if len(roots) > 1 else rows


def exclusion(r, alias):
    """Why a samples.csv row stays out of the cache, or '' if it goes in. UPX stubs are flagged by representation
    in the bundle, by alias current_representation (stubs of bundle samples) or by the _upx_packed folder elsewhere."""
    if r['corpus'] == 'mendeley':
        if not r['recommended_split']:
            return 'no recommended split'
    elif r['eligible_for_gnn'] != '1' or r['architecture'] not in ('x86', 'x64'):
        return 'ineligible'
    if r['corpus'] == 'mendeley' and r['representation'] == 'upx_packed':
        return 'upx stub'
    if r['corpus'] != 'mendeley' and (r['family'] == '_upx_packed' or alias.get('current_representation') == 'upx_packed'):
        return 'upx stub'
    return ''


def join_v3(rows, aliases):
    """fold_v3, fold_v3_sha256 and arch_matched from folds_all.csv, by sha256 or the identity alias's cohort_sha256.
    Joined non-mendeley rows take family and identity group from the fold file, which spans all corpora."""
    folds_v3 = source_for('folds_v3')
    folds = {r['sha256']: r for r in read_csv(folds_v3 / 'folds_all.csv')}
    matched = {r['sha256']: r['keep'] for r in read_csv(folds_v3 / 'arch_matched.csv')}
    for r in rows:
        sid = r['sample_id']
        key = sid if sid in folds else aliases[r['corpus']].get(sid, {}).get('cohort_sha256')
        r['fold_v3'] = r['fold_v3_sha256'] = r['arch_matched'] = ''
        if key not in folds:
            continue
        f = folds[key]
        if (f['label'], f['arch']) != (r['label'], r['architecture']):
            raise SystemExit(f"{sid}: label/arch {r['label']}/{r['architecture']} vs folds_all {f['label']}/{f['arch']}")
        r['fold_v3'], r['fold_v3_sha256'], r['arch_matched'] = f['fold'], key, matched[key]
        if r['corpus'] != 'mendeley':
            r['family'], r['identity_group'] = f['family'], f['group']


def eligible_rows(corpora=('mendeley',)):
    """Cache rows of the corpora in order, and {(corpus, reason): sample_ids} for the rows left out. A sample_id
    already taken by an earlier corpus is left out as a duplicate."""
    rows, dropped, seen, aliases = [], {}, set(), {}
    for corpus in corpora:
        aliases[corpus] = corpus_aliases(corpus)
        for r in corpus_samples(corpus):
            alias = aliases[corpus].get(r['sample_id'], {})
            why = exclusion(r, alias)
            if not why and r['sample_id'] in seen:
                why = 'duplicate sample_id'
            if not why and corpus != 'mendeley' and alias.get('cohort_sha256') in seen:
                why = 'duplicate identity'
            if why:
                dropped.setdefault((corpus, why), []).append(r['sample_id'])
            else:
                seen.add(r['sample_id'])
                rows.append(r)
    folds = {r['sha256']: r for r in read_csv(source_for('folds'))}
    for r in rows:
        r['fold'] = r['fold_sha256'] = ''
        if r['corpus'] != 'mendeley':
            continue
        key = r['sample_id'] if r['sample_id'] in folds else aliases['mendeley'].get(r['sample_id'], {}).get('cohort_sha256')
        if key not in folds:
            raise SystemExit(f"{r['sample_id']} joins neither folds_mendeley.csv nor identity_aliases.csv")
        r['fold'], r['fold_sha256'] = folds[key]['fold'], key
    join_v3(rows, aliases)
    return rows, dropped


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
            return (int(z['cache_version']) == CACHE_VERSION and stored_counts(z) == expected_counts(row)
                    and (want_kind is None or str(z['ids_kind']) == want_kind))
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


def is_cond(mnemonic):
    """Conditional branch: j* other than jmp, or loop*."""
    m = mnemonic.lower()
    return (m.startswith('j') and m != 'jmp') or m.startswith('loop')


def classify_edge_kind(src_block, dst_block):
    """Index into EDGE_KINDS for one CFG edge, by precedence: extern, fallthrough, cond_taken, jmp/indirect, other."""
    if dst_block['type'] == FCB_EXTERN:
        return EDGE_KINDS.index('extern')
    if dst_block['start_rva'] == src_block['end_rva']:
        return EDGE_KINDS.index('fallthrough')
    insns = src_block['instructions']
    last = insns[-1]['mnemonic'].lower() if insns else ''
    if is_cond(last):
        return EDGE_KINDS.index('cond_taken')
    if last == 'jmp':
        return EDGE_KINDS.index('indirect' if src_block['type'] == FCB_INDJUMP else 'jmp')
    return EDGE_KINDS.index('other')


def dfs_back_edges(n_blocks, edges_local, entry):
    """Edges whose target is on the DFS stack. Iterative; starts at entry, then restarts at the lowest unvisited block."""
    edges = np.asarray(edges_local, dtype=np.int64).reshape(-1, 2)
    back = np.zeros(len(edges), dtype=bool)
    if n_blocks == 0:
        return back
    order = np.argsort(edges[:, 0], kind='stable')
    ptr = np.zeros(n_blocks + 1, dtype=np.int64)
    np.cumsum(np.bincount(edges[:, 0], minlength=n_blocks), out=ptr[1:])
    dst, order, end, nxt = edges[order, 1].tolist(), order.tolist(), ptr[1:].tolist(), ptr[:-1].tolist()
    state = [0] * n_blocks
    for root in (entry, *range(n_blocks)):
        if state[root]:
            continue
        state[root] = 1
        stack = [root]
        while stack:
            u = stack[-1]
            i = nxt[u]
            if i == end[u]:
                state[u] = 2
                stack.pop()
                continue
            nxt[u] = i + 1
            v = dst[i]
            if state[v] == 1:
                back[order[i]] = True
            elif state[v] == 0:
                state[v] = 1
                stack.append(v)
    return back


def function_edge_arrays(blocks, edges_local, entry_rva):
    """edge_kind, edge_back, edge_back_addr (int8, aligned with edges_local) for one function's address-sorted blocks."""
    edges = np.asarray(edges_local, dtype=np.int64).reshape(-1, 2)
    kind = np.array([classify_edge_kind(blocks[a], blocks[b]) for a, b in edges.tolist()], dtype=np.int8)
    start = np.array([b['start_rva'] for b in blocks], dtype=np.int64)
    entry = next((i for i, b in enumerate(blocks) if b['start_rva'] == entry_rva), 0)
    back = dfs_back_edges(len(blocks), edges, entry)
    back_addr = start[edges[:, 1]] <= start[edges[:, 0]]
    return kind, back.astype(np.int8), back_addr.astype(np.int8)


def encode_sample(row):
    """Stream one export, one function record at a time, into flat arrays."""
    rules = behavior_rules()
    categorize_behavior, strip_opcode_prefixes, match_api = rules.categorize_behavior, rules.strip_opcode_prefixes, rules._match_api_in_operands
    sample_id = row['sample_id']
    mn, sig, beh, api = array.array('i'), array.array('i'), array.array('b'), array.array('i')
    blk_insn_ptr, blk_api_ptr, blk_feat = array.array('i', [0]), array.array('i', [0]), array.array('f')
    fn_blk_ptr, fn_flags, cfg_edges = array.array('i', [0]), array.array('b'), array.array('i')
    edge_kind, edge_back, edge_back_addr = array.array('b'), array.array('b'), array.array('b')
    blk_type, blk_rva = array.array('b'), array.array('I')
    pending_calls = array.array('q')
    fn_index_by_rva = {}
    mn_ids, sig_ids, api_ids = Interner(), Interner(), Interner()
    crypto_beh = {}
    import_beh = {}
    footer = None

    with gzip.open(Path(row['root']) / row['cfg_path'], 'rt', encoding='utf-8') as stream:
        header = json.loads(stream.readline())
        if header.get('record') != 'header' or header.get('sha256') != sample_id:
            raise ValueError('header identity mismatch')
        if header.get('schema') != SCHEMA:
            raise ValueError(f"schema {header.get('schema')}")
        processor = header.get('analysis_options', {}).get('processor')
        if processor != 'metapc':
            raise ValueError(f'IDA processor {processor}: not x86 code (a managed binary exports CIL methods)')
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
            fn_edges = []
            for src, dst in record['edges']:
                a, b = local[src], local[dst]
                cfg_edges.append(a)
                cfg_edges.append(b)
                outdeg[a - base] += 1
                indeg[b - base] += 1
                fn_edges.append((a - base, b - base))
            for arr, new in zip((edge_kind, edge_back, edge_back_addr), function_edge_arrays(blocks, fn_edges, record['rva'])):
                arr.extend(new.tolist())
            for block in blocks:
                blk_type.append(block['type'])
                blk_rva.extend((block['start_rva'], block['end_rva']))
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
                            matched = match_api([import_names[j] for j in ids])
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
        edge_kind=np.frombuffer(edge_kind, dtype=np.int8), edge_back=np.frombuffer(edge_back, dtype=np.int8),
        edge_back_addr=np.frombuffer(edge_back_addr, dtype=np.int8), blk_type=np.frombuffer(blk_type, dtype=np.int8),
        blk_rva=np.frombuffer(blk_rva, dtype=np.uint32).reshape(-1, 2), cache_version=np.array(CACHE_VERSION),
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


def vocab_sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_vocab(cache=None):
    """vocab.json and the generations it grew through, oldest first; an empty vocab and [] for a new cache."""
    cache = cache or CACHE
    if not (cache / 'vocab.json').exists():
        return {k: [] for k in TOKEN_KINDS}, []
    text = (cache / 'vocab.json').read_text(encoding='utf-8')
    vocab = json.loads(text)
    path = cache / 'vocab_generations.json'
    if path.exists():
        return vocab, json.loads(path.read_text(encoding='utf-8'))
    return vocab, [{'vocab_sha': vocab_sha256(text), 'sizes': {k: len(v) for k, v in vocab.items()}}]


def seed_from_base(base, rows):
    """Copy base's vocab into a cache that has none, and base's npz for rows it built that this cache lacks."""
    if not (CACHE / 'vocab.json').exists():
        for name in ('vocab.json', 'vocab_generations.json'):
            if (base / name).exists():
                shutil.copyfile(base / name, CACHE / name)
    built = {r['sample_id'] for r in read_csv(base / 'index.csv') if r['status'] == 'ok'}
    copied = 0
    for r in rows:
        if r['sample_id'] in built and not npz_path(r['sample_id']).exists():
            shutil.copyfile(base / 'samples' / f"{r['sample_id']}.npz", npz_path(r['sample_id']))
            copied += 1
    return copied


def finalize(rows):
    """Append the strings of locally numbered npz to vocab.json and rewrite those npz to global ids. npz numbered
    against an earlier generation of this vocab are left alone, since appending keeps their ids."""
    prior, generations = read_vocab()
    known = {g['vocab_sha'] for g in generations}
    local, new = [], {k: set() for k in TOKEN_KINDS}
    for r in rows:
        path = npz_path(r['sample_id'])
        if not path.exists():
            continue
        with np.load(path) as z:
            if str(z['ids_kind']) == 'global':
                if str(z['vocab_sha']) not in known:
                    raise SystemExit(f"{r['sample_id']}: ids belong to vocab {z['vocab_sha']}, not one of this cache's")
                continue
            for k in TOKEN_KINDS:
                new[k].update(z[f'{k}_strings'].tolist())
        local.append(path)
    vocab = {k: prior[k] + sorted(new[k].difference(prior[k])) for k in TOKEN_KINDS}
    behavior_map = behavior_rules().BEHAVIOR_MAP
    vocab['beh'] = sorted(behavior_map, key=behavior_map.get)
    if prior.get('beh', vocab['beh']) != vocab['beh']:
        raise SystemExit('behavior categories differ from vocab.json')
    text = json.dumps(vocab, indent=0, ensure_ascii=False)
    vocab_sha = vocab_sha256(text)
    if vocab_sha not in known:
        generations.append({'vocab_sha': vocab_sha, 'sizes': {k: len(v) for k, v in vocab.items()}})
    for name, body in (('vocab.json', text), ('vocab_generations.json', json.dumps(generations, indent=1))):
        tmp = CACHE / f'{name}.tmp'
        tmp.write_text(body, encoding='utf-8')
        os.replace(tmp, CACHE / name)
    lookup = {k: {s: i for i, s in enumerate(vocab[k])} for k in TOKEN_KINDS}
    for path in local:
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
        for k in TOKEN_KINDS:
            ids, strings = arrays[k], arrays[f'{k}_strings']
            to_global = np.array([lookup[k][s] for s in strings.tolist()], dtype=np.int32)
            order = np.argsort(to_global, kind='stable')
            arrays[k] = to_global[ids] if len(ids) else ids
            arrays[f'{k}_u'], arrays[f'{k}_strings'] = to_global[order], strings[order]
        arrays['ids_kind'], arrays['vocab_sha'] = 'global', vocab_sha
        write_npz(path, arrays)
    return {k: len(vocab[k]) for k in vocab}, vocab_sha, len(local)


def edge_stats(rows):
    kinds, back, agree, total = np.zeros(len(EDGE_KINDS), dtype=np.int64), 0, 0, 0
    for r in rows:
        with np.load(npz_path(r['sample_id'])) as z:
            kind, eb, ea = z['edge_kind'], z['edge_back'], z['edge_back_addr']
        kinds += np.bincount(kind, minlength=len(EDGE_KINDS))
        back += int(eb.sum())
        agree += int((eb == ea).sum())
        total += len(kind)
    return {'edge_kind_counts': dict(zip(EDGE_KINDS, kinds.tolist())), 'back_edges': back,
            'back_addr_agreement': round(agree / total, 6) if total else None}


def write_index(rows, results):
    out = []
    for r in rows:
        status, error = results.get(r['sample_id'], ('missing', 'not built'))
        entry = {'sample_id': r['sample_id'], 'label': r['label'], 'split': r['recommended_split'],
                 'arch': r['architecture'], 'family': r['family'], 'identity_group': r['identity_group'],
                 'fold': r['fold'], 'fold_sha256': r['fold_sha256'], 'status': status, 'error': error,
                 'corpus': r['corpus'], 'fold_v3': r['fold_v3'], 'fold_v3_sha256': r['fold_v3_sha256'],
                 'arch_matched': r['arch_matched']}
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
    return out


def x86_rule_macro_f1(n):
    """Macro-F1 of 'ransomware iff x86' from counts keyed (label, arch)."""
    tp, fn, fp, tn = n[('1', 'x86')], n[('1', 'x64')], n[('0', 'x86')], n[('0', 'x64')]
    f1 = lambda a, b, c: 2 * a / (2 * a + b + c) if a else 0.0
    return (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2


def fold_table(index):
    """Markdown tables of the built rows: per v3 fold, class x architecture with the x86-rule floor (all rows and
    the arch-matched subset), per fold x corpus, and the rows without a v3 fold."""
    ok = [r for r in index if r['status'] == 'ok']
    cells = (('1', 'x86'), ('1', 'x64'), ('0', 'x86'), ('0', 'x64'))
    out = ['| fold | R x86 | R x64 | G x86 | G x64 | total | x86-rule mF1 | arch-matched n | arch-matched x86-rule mF1 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for fold in ('0', '1', '2', '3', '4', None):
        sel = [r for r in ok if r['fold_v3'] != '' and (fold is None or r['fold_v3'] == fold)]
        n = {c: sum((r['label'], r['arch']) == c for r in sel) for c in cells}
        am = {c: sum((r['label'], r['arch']) == c and r['arch_matched'] == '1' for r in sel) for c in cells}
        out.append(f"| {fold or 'all'} | " + ' | '.join(str(n[c]) for c in cells)
                   + f' | {len(sel)} | {x86_rule_macro_f1(n):.3f} | {sum(am.values())} | {x86_rule_macro_f1(am):.3f} |')
    parts = list(dict.fromkeys((r['corpus'], r['label']) for r in ok))
    out += ['', '| fold | ' + ' | '.join(f"{c} {'R' if y == '1' else 'G'}" for c, y in parts) + ' |',
            '|---|' + '---:|' * len(parts)]
    for fold in ('0', '1', '2', '3', '4'):
        sel = [r for r in ok if r['fold_v3'] == fold]
        out.append(f'| {fold} | ' + ' | '.join(str(sum((r['corpus'], r['label']) == p for r in sel)) for p in parts) + ' |')
    unjoined = [r for r in ok if r['fold_v3'] == '']
    groups = {}
    for r in unjoined:
        groups.setdefault((r['corpus'], r['label'], r['arch']), []).append(r['family'])
    out += ['', f'Rows without a v3 fold: {len(unjoined)}.', '']
    for (corpus, label, arch), fams in sorted(groups.items()):
        top = ', '.join(f'{f} {c}' for f, c in sorted(((f, fams.count(f)) for f in set(fams)), key=lambda t: -t[1])[:8])
        out.append(f"- {corpus} {'R' if label == '1' else 'G'} {arch}: {len(fams)} ({top})")
    return '\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, help='build only N samples, half of each class')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--corpora', nargs='+', choices=list(CORPUS_NAMES), default=['mendeley'])
    ap.add_argument('--base-cache', type=Path, help='cache whose vocab.json and npz to start from')
    args = ap.parse_args()
    if CACHE == UNSET:
        raise SystemExit('no cache configured: set GT_CACHE, or list a dataset in datasets.json and pass --cache <name>')
    if args.workers > 4:
        raise SystemExit('at most 4 workers')
    if args.corpora != ['mendeley'] and CACHE.resolve() == DEFAULT_CACHE.resolve():
        raise SystemExit('set GT_CACHE to a new directory before adding corpora; the default cache stays Mendeley-only')
    if args.base_cache and args.base_cache.resolve() == CACHE.resolve():
        raise SystemExit('--base-cache must differ from the cache being built')
    rows, dropped = eligible_rows(args.corpora)
    (CACHE / 'samples').mkdir(parents=True, exist_ok=True)
    if args.limit:
        by_class = {c: sorted((r for r in rows if r['label'] == c), key=lambda r: r['sample_id']) for c in ('0', '1')}
        rows = by_class['0'][:args.limit // 2] + by_class['1'][:args.limit - args.limit // 2]
    copied = seed_from_base(args.base_cache, rows) if args.base_cache else 0
    todo = sorted((r for r in rows if not verified(r)), key=lambda r: -int(r['num_instructions']))
    todo_ids = {r['sample_id'] for r in todo}
    print(f'{len(rows)} eligible, {copied} copied from base, {len(rows) - len(todo)} already built, {len(todo)} to build',
          flush=True)
    results = {r['sample_id']: ('ok', '') for r in rows if r['sample_id'] not in todo_ids}
    start = time.time()
    with Pool(args.workers) as pool:
        for n, (sample_id, status, error, secs) in enumerate(pool.imap_unordered(build_one, todo, chunksize=1), 1):
            results[sample_id] = (status, error)
            print(f'[{n}/{len(todo)}] {time.time() - start:7.0f}s {sample_id} {status} {secs:.1f}s {error}', flush=True)
    build_secs = time.time() - start
    sizes, vocab_sha, rewritten = finalize([r for r in rows if results[r['sample_id']][0] == 'ok'])
    index = write_index(rows, results)
    (CACHE / 'folds_v3.md').write_text(fold_table(index), encoding='utf-8')
    failed = {s: e for s, (st, e) in results.items() if st != 'ok'}
    stats = edge_stats([r for r in rows if r['sample_id'] not in failed])
    summary = {'corpora': args.corpora, 'base_cache': str(args.base_cache or ''), 'copied_from_base': copied,
               'eligible': {c: sum(r['corpus'] == c for r in rows) for c in args.corpora},
               'dropped': {f'{c}: {why}': len(ids) for (c, why), ids in dropped.items()},
               'no_fold_v3': {c: sum(r['corpus'] == c and not r['fold_v3'] for r in rows) for c in args.corpora},
               'ok': len(rows) - len(failed), 'failed': failed, 'vocab_sizes': sizes,
               'vocab_sha': vocab_sha, **stats, 'npz_rewritten': rewritten, 'build_seconds': round(build_secs, 1),
               'total_seconds': round(time.time() - start, 1),
               'disk_bytes': sum(p.stat().st_size for p in (CACHE / 'samples').glob('*.npz'))}
    (CACHE / 'build_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
