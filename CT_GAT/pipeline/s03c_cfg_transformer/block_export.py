"""Block dataset export: python -m s03c_cfg_transformer.block_export --protocol kfold --arch x86 --out RUNS/export/ours_cap0

Writes <out>/store/ (fold-independent arrays in global vocabulary ids) and <out>/fold{k}/ (the fold's manifest,
global -> fold vocabulary table and the key of every block). Every array, dtype and shape is in EXPORT_SCHEMA.md. Reads
only the CFG cache samples; the blocks are the ones the GNN sees in evaluation: select_subgraph(policy 'cap', budget 0,
rng None) over the non-empty functions. A block longer than MAX_TOKENS tokens is cut at instruction boundaries.
"""
import argparse
import hashlib
import json
import struct
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from common.config import CACHE, EXPORT, RUNS, RunConfig
from common.manifests import Manifest, load_index, load_sample, make_manifest
from s03b_cfg_feature.gnn.batching import keep_functions
from s02_preprocess.tokenizer.block_tokens import token_table
from s02_preprocess.tokenizer.vocab import TOKEN_KINDS, TokenVocab
from s02_preprocess.subgraph import MAX_BINARY_BLOCKS, select_subgraph
from s03b_cfg_feature.gnn.s06_gnn_block_cfg.edges import block_edges
from s03b_cfg_feature.gnn.s08_gnn_call_graph.calls import call_pairs
from s03b_cfg_feature.gnn.s11_final_classifier.head import bin_feat

FORMAT = 1
MAX_TOKENS, HEAD_TOKENS, TAIL_TOKENS = 192, 160, 32
SPECIALS = {'PAD': 0, 'CLS': 1, 'SEP': 2, 'MASK': 3}
DEGREE_CLIP = RunConfig().degree_clip
TRUNCATION_RULE = (
    f'A block of 3 * n_instructions + n_api_refs > {MAX_TOKENS} tokens keeps its first instructions with at most '
    f'{HEAD_TOKENS} tokens (3 each: mnemonic, signature, behaviour) and its last units with at most {TAIL_TOKENS} '
    'tokens, where the units of a block are its instructions in order followed by its API references (1 token each); '
    'a head that holds every instruction continues into the API references. Cuts fall between units, never inside '
    'one. The kept tokens keep the layout [mn | sig | beh | api] of the kept instructions and API references.')
STORE_ARRAYS = ('bin_blk_ptr', 'bin_fn_ptr', 'bin_cfg_ptr', 'bin_call_ptr', 'blk_tok_ptr', 'tokens', 'tok_kind',
                'blk_fn', 'blk_cache_id', 'blk_feat', 'blk_truncated', 'fn_flags', 'fn_cache_id', 'cfg_edges',
                'edge_type', 'call_edges', 'bin_feat')


def sha_of_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha_of_ids(ids):
    """sha256 of a sample-id list as a set: the sorted ids joined by newlines."""
    return hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()


def global_vocab(cache):
    """The identity vocabulary over the whole cache vocabulary, in `TokenVocab` kind order: `token_table` with it gives
    global ids, kind j offset by the sizes of the kinds before it."""
    sizes = {k: len(v) for k, v in json.loads((Path(cache) / 'vocab.json').read_text(encoding='utf-8')).items()}
    offset = dict(zip(TOKEN_KINDS, np.cumsum([0] + [sizes[k] for k in TOKEN_KINDS[:-1]]).tolist()))
    return SimpleNamespace(lut={k: np.arange(sizes[k], dtype=np.int64) for k in TOKEN_KINDS}, offset=offset,
                           sizes=sizes, total=sum(sizes.values()))


def truncation_plan(n_ins, n_api):
    """(head_ins, tail_ins, head_api, tail_api, cut) per block: the first head_ins and last tail_ins instructions and
    the first head_api and last tail_api API references stay (TRUNCATION_RULE). Blocks that fit keep everything."""
    n_ins, n_api = np.asarray(n_ins, dtype=np.int64), np.asarray(n_api, dtype=np.int64)
    cut = 3 * n_ins + n_api > MAX_TOKENS
    head_ins = np.minimum(n_ins, HEAD_TOKENS // 3)
    head_api = np.where(head_ins == n_ins, np.minimum(n_api, HEAD_TOKENS - 3 * n_ins), 0)
    tail_api = np.minimum(n_api - head_api, TAIL_TOKENS)
    tail_ins = np.minimum(n_ins - head_ins, (TAIL_TOKENS - tail_api) // 3)
    return (np.where(cut, head_ins, n_ins), np.where(cut, tail_ins, 0), np.where(cut, head_api, n_api),
            np.where(cut, tail_api, 0), cut)


def binary_arrays(z, sample_id, gvocab, base_cfg):
    """One binary's rows of the store. Block ids in `cfg_edges` and function ids in `blk_fn` and `call_edges` are
    relative to the binary (0 = its first block / function); `write_store` shifts them by the binary's offsets."""
    fn_ptr = z['fn_blk_ptr'].astype(np.int64)
    sizes = np.diff(fn_ptr)
    keep = keep_functions(z['fn_flags'], sizes, base_cfg, None)
    fn, blk = select_subgraph(z, 'cap', 0, None, sample_id, keep=keep)
    length, n_seg, n_blk = sizes[fn], len(fn), len(blk)
    blk_seg = np.repeat(np.arange(n_seg), length)
    blk_pos = np.arange(n_blk) - np.repeat(np.cumsum(length) - length, length)

    tokens, kind, tok_offsets, per_block, _, _ = token_table(z, gvocab, blk)
    insn_ptr, api_ptr = z['blk_insn_ptr'].astype(np.int64), z['blk_api_ptr'].astype(np.int64)
    n_ins, n_api = insn_ptr[blk + 1] - insn_ptr[blk], api_ptr[blk + 1] - api_ptr[blk]
    head_ins, tail_ins, head_api, tail_api, cut = truncation_plan(n_ins, n_api)
    owner = np.repeat(np.arange(n_blk), per_block)
    local = np.arange(len(tokens)) - tok_offsets[owner]
    is_api = kind == 3
    index = np.where(is_api, local - 3 * n_ins[owner], local - kind * n_ins[owner])
    first = np.where(is_api, head_api[owner], head_ins[owner])
    last = np.where(is_api, tail_api[owner], tail_ins[owner])
    count = np.where(is_api, n_api[owner], n_ins[owner])
    kept = (index < first) | (index >= count - last)
    blk_ntok = np.bincount(owner[kept], minlength=n_blk)

    _, _, (edge_seg, edge_src, edge_dst, edge_type) = block_edges(z, blk, blk_seg, blk_pos, DEGREE_CLIP)
    seg_start = np.cumsum(length) - length
    call_src, call_dst = call_pairs(z, fn)
    return {
        'tokens': tokens[kept].astype(np.int32), 'tok_kind': kind[kept].astype(np.int8), 'blk_ntok': blk_ntok,
        'blk_fn': blk_seg.astype(np.int32), 'blk_cache_id': blk.astype(np.int32),
        'blk_feat': z['blk_feat'][blk].astype(np.float16), 'blk_truncated': cut.astype(np.int8),
        'fn_flags': z['fn_flags'][fn].astype(np.int8), 'fn_cache_id': fn.astype(np.int32),
        'cfg_edges': np.stack([seg_start[edge_seg] + edge_src, seg_start[edge_seg] + edge_dst], 1).astype(np.int32),
        'edge_type': edge_type.astype(np.int8), 'call_edges': np.stack([call_src, call_dst], 1).astype(np.int32),
        'bin_feat': bin_feat(z), 'n_blocks_total': int(sizes.sum()), 'n_blocks_kept': n_blk,
    }


def raw_to_npy(raw, dest, dtype, chunk=1 << 24):
    """Copy a raw dtype stream into a .npy file and delete the stream."""
    src = np.memmap(raw, dtype=dtype, mode='r')
    out = np.lib.format.open_memmap(dest, mode='w+', dtype=dtype, shape=src.shape)
    for lo in range(0, len(src), chunk):
        out[lo:lo + chunk] = src[lo:lo + chunk]
    out.flush()
    del out, src
    Path(raw).unlink()


def write_store(rows, store, cache, base_cfg, log=print):
    """Write store/ for the binaries `rows` (in the given order); returns the counts that go into store.json."""
    store.mkdir(parents=True, exist_ok=True)
    gvocab = global_vocab(cache)
    parts = {k: [] for k in ('blk_ntok', 'blk_fn', 'blk_cache_id', 'blk_feat', 'blk_truncated', 'fn_flags',
                             'fn_cache_id', 'cfg_edges', 'edge_type', 'call_edges', 'bin_feat')}
    ptr = {k: [0] for k in ('blk', 'fn', 'cfg', 'call')}
    binaries, vocab_shas = [], set()
    n_tokens, start = 0, time.time()
    with (store / 'tokens.raw').open('wb') as f_tok, (store / 'tok_kind.raw').open('wb') as f_kind:
        for i, row in enumerate(rows):
            z = load_sample(row['sample_id'])
            with np.load(Path(cache) / 'samples' / f"{row['sample_id']}.npz") as zz:
                vocab_shas.add(str(zz['vocab_sha']))
            a = binary_arrays(z, row['sample_id'], gvocab, base_cfg)
            a['cfg_edges'] = a['cfg_edges'] + np.int32(ptr['blk'][-1])
            a['call_edges'] = a['call_edges'] + np.int32(ptr['fn'][-1])
            a['blk_fn'] = a['blk_fn'] + np.int32(ptr['fn'][-1])
            f_tok.write(a['tokens'].tobytes())
            f_kind.write(a['tok_kind'].tobytes())
            n_tokens += len(a['tokens'])
            for k in parts:
                parts[k].append(a[k])
            ptr['blk'].append(ptr['blk'][-1] + len(a['blk_fn']))
            ptr['fn'].append(ptr['fn'][-1] + len(a['fn_flags']))
            ptr['cfg'].append(ptr['cfg'][-1] + len(a['edge_type']))
            ptr['call'].append(ptr['call'][-1] + len(a['call_edges']))
            binaries.append({'index': i, 'sample_id': row['sample_id'], 'label': row['label'], 'arch': row['arch'],
                             'family': row['family'], 'identity_group': row['identity_group'],
                             'corpus': row.get('corpus', 'mendeley'), 'fold': row['fold'],
                             'n_blocks_total': a['n_blocks_total'], 'n_blocks_kept': a['n_blocks_kept']})
            if (i + 1) % 200 == 0:
                log(f'  {i + 1}/{len(rows)} binaries, {ptr["blk"][-1]} blocks, {n_tokens} tokens, '
                    f'{time.time() - start:.0f} s')
    if len(vocab_shas) != 1:
        raise ValueError(f'samples were built with different vocabularies: {sorted(vocab_shas)}')
    raw_to_npy(store / 'tokens.raw', store / 'tokens.npy', np.int32)
    raw_to_npy(store / 'tok_kind.raw', store / 'tok_kind.npy', np.int8)
    blk_ntok = np.concatenate(parts.pop('blk_ntok'))
    np.save(store / 'blk_tok_ptr.npy', np.concatenate([[0], np.cumsum(blk_ntok)]).astype(np.int64))
    for name, key in (('bin_blk_ptr', 'blk'), ('bin_fn_ptr', 'fn'), ('bin_cfg_ptr', 'cfg'), ('bin_call_ptr', 'call')):
        np.save(store / f'{name}.npy', np.asarray(ptr[key], dtype=np.int64))
    for name, chunks in parts.items():
        np.save(store / f'{name}.npy', np.stack(chunks) if name == 'bin_feat' else np.concatenate(chunks))
    (store / 'binaries.json').write_text(json.dumps(binaries, indent=0), encoding='utf-8')
    return {'n_binaries': len(rows), 'n_blocks': int(ptr['blk'][-1]), 'n_tokens': n_tokens,
            'n_functions': int(ptr['fn'][-1]), 'n_cfg_edges': int(ptr['cfg'][-1]), 'n_call_edges': int(ptr['call'][-1]),
            'n_blocks_truncated': int(np.concatenate(parts['blk_truncated']).sum()),
            'n_binaries_block_capped': int(sum(b['n_blocks_kept'] < b['n_blocks_total'] for b in binaries)),
            'vocab_sha': vocab_shas.pop(), 'global_vocab': {'sizes': gvocab.sizes, 'offsets': gvocab.offset}}


def fold_lut(vocab, gvocab):
    """int32 table: global id -> fold id. Specials 0-3; then the ids of `TokenVocab` shifted by 4 (per kind UNK = 0
    of the kind and the tokens kept by document frequency; behaviour tags are all kept and have no UNK)."""
    lut = np.zeros(gvocab.total, dtype=np.int32)
    for k in TOKEN_KINDS:
        lut[gvocab.offset[k]:gvocab.offset[k] + gvocab.sizes[k]] = len(SPECIALS) + vocab.offset[k] + vocab.lut[k]
    return lut


def block_keys(tokens, ptr, lut, chunk_blocks=1 << 20):
    """blake2b-64 (little-endian int64) of each block's fold-id sequence, as int32 bytes."""
    keys = np.empty(len(ptr) - 1, dtype=np.int64)
    unpack = struct.Struct('<q').unpack
    for lo in range(0, len(keys), chunk_blocks):
        hi = min(lo + chunk_blocks, len(keys))
        base = int(ptr[lo])
        buf = memoryview(lut[np.asarray(tokens[base:int(ptr[hi])])].astype('<i4').tobytes())
        rel = ((ptr[lo:hi + 1] - base) * 4).tolist()
        keys[lo:hi] = [unpack(hashlib.blake2b(buf[rel[j]:rel[j + 1]], digest_size=8).digest())[0]
                       for j in range(hi - lo)]
    return keys


def manifest_sha_of(doc):
    keep = {k: doc[k] for k in ('fold', 'train', 'val', 'test', 'val_families', 'seed', 'val_seed')}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()


def write_fold(store, fold_dir, fold, manifest: Manifest, cfg: RunConfig, binaries, log=print):
    """fold{k}/: manifest.json, lut.npy, vocab.json and block_key.npy. The vocabulary is
    TokenVocab(manifest.train, cfg), the one a GNN run of `cfg` builds."""
    fold_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    index_of = {b['sample_id']: b['index'] for b in binaries}
    doc = {'fold': fold, 'seed': cfg.seed, 'val_seed': cfg.val_seed,
           **{name: [index_of[r['sample_id']] for r in getattr(manifest, name)] for name in ('train', 'val', 'test')},
           'val_families': manifest.val_families}
    vocab = TokenVocab(manifest.train, cfg)
    gvocab = global_vocab(CACHE)
    lut = fold_lut(vocab, gvocab)
    np.save(fold_dir / 'lut.npy', lut)
    tokens = np.load(store / 'tokens.npy', mmap_mode='r')
    ptr = np.load(store / 'blk_tok_ptr.npy')
    keys = block_keys(tokens, ptr, lut)
    np.save(fold_dir / 'block_key.npy', keys)
    n_distinct = len(np.unique(keys))
    if len(np.unique(np.stack([keys, np.diff(ptr)], 1), axis=0)) != n_distinct:
        raise ValueError(f'fold {fold}: two blocks of different length share a block key')
    doc.update(manifest_sha=manifest_sha_of(doc), n_blocks=len(keys), n_distinct_blocks=n_distinct)
    (fold_dir / 'manifest.json').write_text(json.dumps(doc, indent=1), encoding='utf-8')
    unk = {k: None if k == 'beh' else len(SPECIALS) + vocab.offset[k] for k in TOKEN_KINDS}
    vdoc = {'specials': SPECIALS, 'n_ids': len(SPECIALS) + sum(vocab.size.values()), 'min_df': cfg.min_df,
            'n_train_binaries': len(manifest.train),
            'kinds': {k: {'offset': len(SPECIALS) + vocab.offset[k], 'size': vocab.size[k], 'unk': unk[k]}
                      for k in TOKEN_KINDS},
            'global_offsets': gvocab.offset, 'lut_sha': hashlib.sha256(lut.tobytes()).hexdigest()}
    (fold_dir / 'vocab.json').write_text(json.dumps(vdoc, indent=1), encoding='utf-8')
    log(f'fold {fold}: train {len(manifest.train)} val {len(manifest.val)} test {len(manifest.test)}, '
        f'{vdoc["n_ids"]} ids, {n_distinct} distinct of {len(keys)} blocks, {time.time() - start:.0f} s')
    return doc


def code_sha():
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__).resolve(), *(root / p for p in (
        's02_preprocess/tokenizer/block_tokens.py', 's02_preprocess/tokenizer/vocab.py',
        's02_preprocess/subgraph.py', 's03b_cfg_feature/gnn/s06_gnn_block_cfg/edges.py',
        's03b_cfg_feature/gnn/s08_gnn_call_graph/calls.py'))]
    return hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()


def git_head():
    here = Path(__file__).resolve().parent
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=here, capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--', '.'], cwd=here, capture_output=True, text=True,
                               check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ''
    return head + ('+dirty' if dirty else '')


def export_sha_of(root, folds):
    """sha256 over the store arrays, binaries.json and every fold's manifest, lut and block keys."""
    h = hashlib.sha256()
    names = [root / 'store' / f'{n}.npy' for n in STORE_ARRAYS] + [root / 'store' / 'binaries.json']
    for k in folds:
        names += [root / f'fold{k}' / n for n in ('manifest.json', 'lut.npy', 'block_key.npy')]
    for p in names:
        h.update(p.name.encode() + sha_of_file(p).encode())
    return h.hexdigest()


def resolve_out(out):
    """`RUNS/export/<name>` (or an absolute path) -> a path under the export directory (RUNS/export, or GT_EXPORT);
    anything else is refused."""
    path = Path(out)
    if path.parts and path.parts[0] == 'RUNS':
        rest = path.parts[1:]
        path = EXPORT.joinpath(*rest[1:]) if rest[:1] == ('export',) else RUNS.joinpath(*rest)
    path = path.resolve()
    if EXPORT.resolve() not in path.parents:
        raise ValueError(f"exports go under {EXPORT}: {out}")
    return path


def build_export(cfg: RunConfig, out, rows, manifests, log=print):
    """Write store/ for `rows` and a fold directory for each {fold: Manifest} of `manifests`, then store.json. `cfg` is
    the base GNN config (protocol, arch, seed 0, val_seed 0) that builds each fold's vocabulary."""
    start = time.time()
    out.mkdir(parents=True, exist_ok=True)
    counts = write_store(rows, out / 'store', CACHE, cfg, log)
    binaries = json.loads((out / 'store' / 'binaries.json').read_text(encoding='utf-8'))
    for k, manifest in manifests.items():
        write_fold(out / 'store', out / f'fold{k}', k, manifest, replace(cfg, fold=k), binaries, log)
    meta = {'format': FORMAT, 'protocol': cfg.protocol, 'arch': cfg.arch, 'cache': str(CACHE),
            'folds': list(manifests), 'policy': 'cap', 'budget': 0, 'max_blocks_per_binary': MAX_BINARY_BLOCKS,
            'selection': "select_subgraph(policy 'cap', budget 0, rng None) over the non-empty functions",
            'max_tokens_per_block': MAX_TOKENS, 'head_tokens': HEAD_TOKENS, 'tail_tokens': TAIL_TOKENS,
            'truncation_rule': TRUNCATION_RULE, 'token_layout': '[mn | sig | beh | api] per block (block_tokens)',
            'specials': SPECIALS, 'code_sha': code_sha(), 'git_head': git_head(), 'seed': cfg.seed,
            'val_seed': cfg.val_seed, 'export_secs': round(time.time() - start, 1), **counts}
    meta['export_sha'] = export_sha_of(out, manifests)
    (out / 'store' / 'store.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    log(f"export {out}: {counts['n_binaries']} binaries, {counts['n_blocks']} blocks, {counts['n_tokens']} tokens, "
        f"{counts['n_blocks_truncated']} blocks cut at {MAX_TOKENS} tokens, {size / 2**30:.2f} GiB, "
        f"{meta['export_secs']:.0f} s, export_sha {meta['export_sha'][:16]}")
    return meta


def export(cfg: RunConfig, out, folds=range(5), log=print):
    """Store of every binary with a fold under cfg's protocol and arch, and a fold directory per fold, from the
    manifests of seed 0 and val_seed 0."""
    index = load_index()
    base_cfg = replace(cfg, model='gnn', fold=0, seed=0, val_seed=0)
    first = make_manifest(base_cfg, index)
    rows = sorted(first.train + first.val + first.test, key=lambda r: r['sample_id'])
    log(f'{len(rows)} binaries of {cfg.protocol} {cfg.arch} from {CACHE}')
    manifests = {k: make_manifest(replace(base_cfg, fold=k), index) for k in folds}
    return build_export(base_cfg, out, rows, manifests, log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', choices=['kfold', 'kfold_v3'], default='kfold')
    ap.add_argument('--arch', choices=['x86', 'all', 'matched'], default='x86')
    ap.add_argument('--out', required=True, help='RUNS/export/<name> (RUNS is GT_RUNS), or a path under the export directory')
    ap.add_argument('--folds', type=int, nargs='*', default=list(range(5)))
    args = ap.parse_args()
    export(RunConfig(protocol=args.protocol, fold=0, arch=args.arch, model='gnn'), resolve_out(args.out), args.folds)


if __name__ == '__main__':
    main()
