"""python -m pytest tests/test_transformer_paths.py: step 12, the block export and its reader, the encoder contract, the
precomputed block vectors and the GNN switch that reads them, the out-of-fold TF-IDF stacking, the `paths` queue and gates.

The export tests build a small export (ten real binaries, one fold) in a temporary directory and need the CFG cache of
`common.config.CACHE`. GPU-only tests (the 2-epoch smoke runs of path 1, path 2 and the stack) are marked needs_cuda.
"""
import ast
import copy
import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from common.config import CACHE, RUNS, RunConfig
from common.manifests import Manifest, load_index, load_sample, make_manifest
from evaluation import baselines, gates, stats
from pipeline.batching import BinaryDataset, collate, encode_binary, loader
from pipeline.model import HierGNN
from pipeline.s03_tokenizer.vocab import TOKEN_KINDS, TokenVocab
from pipeline.s04_select_subgraph.subgraph import select_subgraph
from pipeline.s05_block_counting.block_vecs import BlockVecs, BlockVecSource, MissingKeys
from training import train
from training.train import parse_config
from transformer_paths import block_export as be
from transformer_paths import encoder_contract as ec
from transformer_paths import precompute_blockvec as pb
from transformer_paths import reference_encoder as re_
from transformer_paths.reader import BlockStore, FoldView, collate_blocks

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which('bash')
needs_cache = pytest.mark.skipif(not (CACHE / 'index.csv').exists(), reason='needs the CFG cache')
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
needs_bash = pytest.mark.skipif(BASH is None, reason='needs bash')


def gnn(**kw):
    return RunConfig(**{'model': 'gnn', 'num_workers': 0, **kw})


# configuration: val_seed, block_input, names ----------------------------------------------------------------------


def test_default_names_are_unchanged():
    assert RunConfig().name == 'bundle_x86_graph_s0'
    assert RunConfig(protocol='kfold', fold=2, model='gnn').name == 'kfold2_x86_gnn_s0'
    cfg = RunConfig(protocol='kfold', fold=0, model='gnn', control='no_edges', mask_blocks=0.15, seed=3)
    assert cfg.name == 'kfold0_x86_gnn-noedges-mblk15_s3'
    assert RunConfig(protocol='kfold', fold=0, model='gnn', stack_tfidf=True, mask_blocks=0.15).variant == 'gnn-stack-mblk15'
    assert (RunConfig().val_seed, RunConfig().block_input, RunConfig().block_vec) == (None, 'tokens', '')


def test_val_seed_suffix_only_when_it_differs_from_the_seed():
    base = dict(protocol='kfold', fold=0, model='gnn')
    assert gnn(**base, seed=1).name == gnn(**base, seed=1, val_seed=1).name == 'kfold0_x86_gnn_s1'
    assert gnn(**base, seed=1, val_seed=0).name == 'kfold0_x86_gnn-vs0_s1'
    assert gnn(**base, seed=0, val_seed=0).name == 'kfold0_x86_gnn_s0'
    assert RunConfig(protocol='kfold', fold=0, seed=2, val_seed=0).name == 'kfold0_x86_graph-vs0_s2'
    assert gnn(**base, seed=1, val_seed=0).group == 'kfold_x86_gnn-vs0'
    with pytest.raises(ValueError):
        RunConfig(val_seed=-1)


def test_block_input_config_rules():
    vec = dict(protocol='kfold', fold=0, model='gnn', block_input='vec', block_vec='ours_cap0_ref')
    cfg = gnn(**vec)
    assert cfg.name == 'kfold0_x86_gnn-bv-ours_cap0_ref_s0'
    assert gnn(**vec, control='no_edges', mask_blocks=0.15).variant == 'gnn-noedges-mblk15-bv-ours_cap0_ref'
    assert gnn(**{**vec, 'block_vec': 'ref'}).variant == 'gnn-bv-ref'
    for bad in (dict(block_input='vec'), dict(block_vec='x'), dict(block_input='bag'),
                dict(**vec, mask_tokens=0.1), dict(**vec, encoder='mean'),
                {**vec, 'block_vec': 'a b'}, {**vec, 'block_vec': '../x'}, {**vec, 'protocol': 'kfold_v3'}):
        with pytest.raises(ValueError):
            RunConfig(**{'model': 'gnn', 'protocol': 'kfold', 'fold': 0, **bad})
    with pytest.raises(ValueError):
        RunConfig(protocol='kfold', fold=0, block_input='vec', block_vec='x')       # the transformer path has no vectors


def test_cli_parses_the_new_flags():
    cfg, out = parse_config(['--protocol', 'kfold', '--fold', '0', '--model', 'gnn', '--val-seed', '0', '--seed', '2',
                             '--block-input', 'vec', '--block-vec', 'e_n', '--stack-tfidf', '--out', 'x'])
    assert (cfg.val_seed, cfg.seed, cfg.block_input, cfg.block_vec, cfg.stack_tfidf, out) == (0, 2, 'vec', 'e_n', True, 'x')
    assert parse_config(['--protocol', 'kfold', '--fold', '0'])[0].val_seed is None
    train.check_supported(cfg)                                                   # stacking is supported now
    with pytest.raises(NotImplementedError):
        train.check_supported(replace(cfg, aux_recon=0.1))


@needs_cache
def test_val_seed_fixes_the_split_across_seeds():
    index = load_index()
    ids = lambda m: ([r['sample_id'] for r in m.train], [r['sample_id'] for r in m.val], [r['sample_id'] for r in m.test])
    fixed = [ids(make_manifest(gnn(protocol='kfold', fold=1, seed=s, val_seed=0), index)) for s in (0, 1, 2)]
    assert fixed[0] == fixed[1] == fixed[2]
    free = [ids(make_manifest(gnn(protocol='kfold', fold=1, seed=s), index)) for s in (0, 1)]
    assert free[0][2] == free[1][2] and free[0][1] != free[1][1]                   # same test fold, another carve
    assert ids(make_manifest(gnn(protocol='kfold', fold=1, seed=1, val_seed=1), index)) == free[1]
    assert fixed[1] == free[0]                                                     # val_seed 0 is the seed-0 split
    sub = make_manifest(gnn(protocol='kfold', fold=1, seed=2, val_seed=0, train_fraction=0.5), index)
    assert [r['sample_id'] for r in sub.val] == fixed[0][1]
    assert sub.train == make_manifest(gnn(protocol='kfold', fold=1, seed=0, train_fraction=0.5), index).train


# the export ---------------------------------------------------------------------------------------------------------


def reference_cut(n_ins, n_api):
    """The truncation rule written as a walk over units: (kept instruction indices, kept API indices)."""
    units = [('i', k) for k in range(n_ins)] + [('a', k) for k in range(n_api)]
    size = lambda u: 3 if u[0] == 'i' else 1
    if sum(size(u) for u in units) <= be.MAX_TOKENS:
        return list(range(n_ins)), list(range(n_api))
    head, used = [], 0
    for u in units:
        if used + size(u) > be.HEAD_TOKENS:
            break
        head.append(u)
        used += size(u)
    tail, used = [], 0
    for u in reversed(units[len(head):]):
        if used + size(u) > be.TAIL_TOKENS:
            break
        tail.insert(0, u)
        used += size(u)
    kept = head + tail
    return [k for t, k in kept if t == 'i'], [k for t, k in kept if t == 'a']


def test_truncation_plan_matches_the_unit_walk():
    cases = [(a, p) for a in list(range(0, 130, 3)) + [53, 54, 55, 63, 64, 65, 200] for p in (0, 1, 5, 31, 32, 33, 100, 500)]
    a, p = np.array(cases).T
    head_ins, tail_ins, head_api, tail_api, cut = be.truncation_plan(a, p)
    for i, (n_ins, n_api) in enumerate(cases):
        ins, api = reference_cut(n_ins, n_api)
        got_ins = list(range(head_ins[i])) + list(range(n_ins - tail_ins[i], n_ins))
        got_api = list(range(head_api[i])) + list(range(n_api - tail_api[i], n_api))
        assert (got_ins, got_api) == (ins, api), (n_ins, n_api)
        assert bool(cut[i]) == (3 * n_ins + n_api > be.MAX_TOKENS)
        assert 3 * len(got_ins) + len(got_api) <= be.MAX_TOKENS
    assert tuple(be.truncation_plan([70], [0])[i][0] for i in range(4)) == (53, 10, 0, 0)       # 159 + 30 tokens
    assert tuple(be.truncation_plan([10], [500])[i][0] for i in range(4)) == (10, 0, 130, 32)    # 30 + 130 + 32 tokens
    assert (be.MAX_TOKENS, be.HEAD_TOKENS, be.TAIL_TOKENS) == (192, 160, 32)


@pytest.fixture(scope='module')
def mini(tmp_path_factory):
    """An export of ten real binaries (the five smallest of each class), fold 0, and its tuple of rows."""
    if not (CACHE / 'index.csv').exists():
        pytest.skip('needs the CFG cache')
    x86 = sorted((r for r in load_index() if r['arch'] == 'x86' and r['fold'] is not None),
                 key=lambda r: (int(r['num_blocks']), r['sample_id']))
    ransomware, goodware = [r for r in x86 if r['label'] == 1][:5], [r for r in x86 if r['label'] == 0][:5]
    train_rows, val_rows, test_rows = ransomware[:3] + goodware[:3], ransomware[3:4] + goodware[3:4], ransomware[4:] + goodware[4:]
    rows = sorted(train_rows + val_rows + test_rows, key=lambda r: r['sample_id'])
    cfg = RunConfig(model='gnn', protocol='kfold', fold=0, arch='x86', seed=0, val_seed=0)
    out = tmp_path_factory.mktemp('export') / 'mini'
    manifest = Manifest(train_rows, val_rows, test_rows, [])
    meta = be.build_export(cfg, out, rows, {0: manifest}, log=lambda *_: None)
    return SimpleNamespace(root=out, rows=rows, manifest=manifest, cfg=cfg, store=BlockStore(out), meta=meta,
                           tmp=out.parent)


@needs_cache
def test_round_trip_tokens_edges_and_calls_equal_batching(mini):
    """For every binary of the export: the store's tokens (global ids through the fold table, minus the 4 specials)
    equal the run ids `encode_binary` builds from `TokenVocab(manifest.train)`, block by block for the blocks that were
    not cut and as a prefix/suffix cut for the ones that were; cfg edges (shifted by the function starts), edge types,
    call edges and bin_feat equal the batch arrays exactly."""
    store, fold = mini.store, FoldView(mini.store, 0, 'all')
    vocab = TokenVocab(mini.manifest.train, mini.cfg)
    cut_seen = 0
    for b in store.binaries:
        sid, i = b['sample_id'], b['index']
        item = encode_binary(load_sample(sid), vocab, mini.cfg, None, sid)
        first, last = store.block_range(i)
        offsets = np.append(item['tok_offsets'], len(item['tokens']))
        z = load_sample(sid)
        assert last - first == len(item['blk_seg'])
        _, kept_blocks = select_subgraph(z, 'cap', 0, None, sid)
        assert np.array_equal(store.blk_cache_id[first:last], kept_blocks)
        n_ins = np.diff(z['blk_insn_ptr'].astype(np.int64))[np.asarray(store.blk_cache_id[first:last])]
        n_api = np.diff(z['blk_api_ptr'].astype(np.int64))[np.asarray(store.blk_cache_id[first:last])]
        for j, block in enumerate(range(first, last)):
            want = item['tokens'][offsets[j]:offsets[j + 1]]
            got = fold.sequence(block).astype(np.int64) - 4
            if not store.blk_truncated[block]:
                assert np.array_equal(got, want), (sid, j)
                continue
            cut_seen += 1
            ins, api = reference_cut(int(n_ins[j]), int(n_api[j]))
            keep = np.concatenate([np.array(ins, dtype=np.int64) + k * n_ins[j] for k in range(3)]
                                  + [3 * n_ins[j] + np.array(api, dtype=np.int64)])
            assert len(want) > be.MAX_TOKENS and np.array_equal(got, want[keep]), (sid, j)
            assert len(got) <= be.MAX_TOKENS
        seg_start = np.cumsum(item['seg_len']) - item['seg_len']
        edges, edge_type = store.cfg_of(i)
        assert np.array_equal(edges[:, 0] - first, seg_start[item['edge_seg']] + item['edge_src'])
        assert np.array_equal(edges[:, 1] - first, seg_start[item['edge_seg']] + item['edge_dst'])
        assert np.array_equal(edge_type, item['edge_type'])
        calls = store.calls_of(i)
        fn_first, fn_last = int(store.bin_fn_ptr[i]), int(store.bin_fn_ptr[i + 1])
        assert np.array_equal(calls[:, 0] - fn_first, item['call_src'])
        assert np.array_equal(calls[:, 1] - fn_first, item['call_dst'])
        assert np.array_equal(np.asarray(store.bin_feat[i]), item['bin_feat'])
        assert fn_last - fn_first == len(item['seg_len'])
        assert np.array_equal(np.asarray(store.blk_fn[first:last]) - fn_first, item['blk_seg'])
        assert np.array_equal(np.asarray(store.fn_cache_id[fn_first:fn_last]), np.asarray(select_subgraph(
            z, 'cap', 0, None, sid)[0]))
        assert np.array_equal(np.asarray(store.blk_feat[first:last]), item['blk_feat'].astype(np.float16))
        assert b['n_blocks_kept'] == item['n_blocks_kept'] and b['n_blocks_total'] == item['n_blocks_total']
    assert cut_seen > 0 and int(np.asarray(store.blk_truncated).sum()) == cut_seen == mini.meta['n_blocks_truncated']


@needs_cache
def test_store_metadata_and_kinds(mini):
    store, meta = mini.store, mini.meta
    assert store.tokens.dtype == np.int32 and store.tok_kind.dtype == np.int8 and store.blk_feat.dtype == np.float16
    assert store.bin_feat.dtype == np.float32 and store.bin_feat.shape == (len(mini.rows), 5)
    assert store.cfg_edges.dtype == store.call_edges.dtype == np.int32 and store.edge_type.dtype == np.int8
    assert store.bin_blk_ptr.dtype == store.blk_tok_ptr.dtype == np.int64
    assert [b['sample_id'] for b in store.binaries] == sorted(r['sample_id'] for r in mini.rows)
    assert set(store.binaries[0]) >= {'index', 'sample_id', 'label', 'arch', 'family', 'corpus', 'fold', 'n_blocks_total',
                                      'n_blocks_kept'}
    assert (meta['max_tokens_per_block'], meta['policy'], meta['budget']) == (192, 'cap', 0)
    assert meta['n_binaries_block_capped'] == 0 and 'between units' in meta['truncation_rule']
    assert {'cache', 'vocab_sha', 'code_sha', 'export_sha', 'truncation_rule'} <= set(meta)
    assert np.diff(store.blk_tok_ptr).max() <= 192 and np.diff(store.blk_tok_ptr).min() >= 0
    sizes = meta['global_vocab']['sizes']
    off = meta['global_vocab']['offsets']
    for k, kind in enumerate(TOKEN_KINDS):                                           # global ids sit in the kind's range
        mask = np.asarray(store.tok_kind) == k
        ids = np.asarray(store.tokens)[mask]
        assert ids.min() >= off[kind] and ids.max() < off[kind] + sizes[kind]
    for block in range(len(store.blk_cache_id)):                                     # [mn | sig | beh | api] layout
        kinds = np.asarray(store.kinds_of(block))
        counts = np.bincount(kinds, minlength=4)
        assert (np.diff(kinds) >= 0).all() and counts[0] == counts[1] == counts[2]


@needs_cache
def test_fold_vocabulary_is_the_train_token_vocab_shifted_by_four(mini):
    store = mini.store
    data, vocab = store.fold(0), TokenVocab(mini.manifest.train, mini.cfg)
    lut, sizes = data['lut'], mini.meta['global_vocab']['sizes']
    off = mini.meta['global_vocab']['offsets']
    assert lut.dtype == np.int32 and len(lut) == sum(sizes.values()) and lut.min() >= 4
    for k in TOKEN_KINDS:
        local = np.arange(sizes[k])
        assert np.array_equal(lut[off[k]:off[k] + sizes[k]], 4 + vocab.offset[k] + vocab.lut[k][local]), k
    assert data['vocab']['n_ids'] == 4 + sum(vocab.size.values()) and lut.max() < data['vocab']['n_ids']
    assert data['vocab']['specials'] == {'PAD': 0, 'CLS': 1, 'SEP': 2, 'MASK': 3}
    for k in TOKEN_KINDS:
        info = data['vocab']['kinds'][k]
        assert info['offset'] == 4 + vocab.offset[k] and info['size'] == vocab.size[k]
        assert info['unk'] == (None if k == 'beh' else 4 + vocab.offset[k])
    assert data['vocab']['lut_sha'] == hashlib.sha256(lut.tobytes()).hexdigest()
    # the tokens kept are those in >= min_df training binaries: UNK-mapped ids are exactly the rare ones
    unk_api = np.flatnonzero(lut[off['api']:off['api'] + sizes['api']] == info_unk(data, 'api'))
    assert len(unk_api) > 0 and (vocab.lut['api'][unk_api] == 0).all()


def info_unk(data, kind):
    return data['vocab']['kinds'][kind]['unk']


@needs_cache
def test_manifest_partitions_the_store_and_matches_make_manifest(mini):
    data = mini.store.fold(0)['manifest']
    index_of = mini.store.index_of
    for split, rows in (('train', mini.manifest.train), ('val', mini.manifest.val), ('test', mini.manifest.test)):
        assert data[split] == [index_of[r['sample_id']] for r in rows]
    assert sorted(data['train'] + data['val'] + data['test']) == list(range(len(mini.rows)))
    assert (data['fold'], data['seed'], data['val_seed']) == (0, 0, 0)
    assert len(data['manifest_sha']) == 64 and data['n_blocks'] == len(mini.store.blk_cache_id)


@needs_cache
def test_block_key_is_blake2b_of_the_fold_id_sequence(mini):
    store, fold = mini.store, FoldView(mini.store, 0, 'all')
    keys = np.asarray(store.fold(0)['block_key'])
    assert keys.dtype == np.int64 and len(keys) == len(store.blk_cache_id)
    for block in (0, 1, 7, len(keys) // 2, len(keys) - 1):
        digest = hashlib.blake2b(fold.sequence(block).astype('<i4').tobytes(), digest_size=8).digest()
        assert keys[block] == int.from_bytes(digest, 'little', signed=True)
    # equal sequences have equal keys and unequal sequences different ones
    seqs = {}
    for block in range(len(keys)):
        seqs.setdefault(fold.sequence(block).tobytes(), set()).add(int(keys[block]))
    assert all(len(v) == 1 for v in seqs.values())
    assert len({next(iter(v)) for v in seqs.values()}) == len(seqs) == mini.store.fold(0)['manifest']['n_distinct_blocks']
    assert len(seqs) < len(keys)                                                     # the near-duplicate binaries share blocks


@needs_cache
def test_reader_views_and_collate(mini):
    store = mini.store
    train_view, val_view, test_view = (FoldView(store, 0, s) for s in ('train', 'val', 'test'))
    assert len(train_view.binaries()) == 6 and len(val_view.binaries()) == 2 and len(test_view.binaries()) == 2
    assert train_view.sample_ids() == [r['sample_id'] for r in mini.manifest.train]
    blocks = train_view.blocks()
    assert len(blocks) == sum(int(store.bin_blk_ptr[b + 1] - store.bin_blk_ptr[b]) for b in train_view.binaries())
    ids, keys = train_view.distinct_blocks()
    assert (np.diff(keys) > 0).all() and len(ids) == len(np.unique(train_view.keys()))
    assert np.array_equal(train_view.keys(ids), keys) and set(ids) <= set(blocks)
    assert len(FoldView(store, 0, 'all').blocks()) == len(store.blk_cache_id)
    with pytest.raises(ValueError):
        FoldView(store, 0, 'dev')
    # padded() and collate_blocks() agree
    pick = ids[:37]
    tokens, mask = train_view.padded(pick, max_len=192)
    t2, m2 = collate_blocks([train_view.sequence(b) for b in pick], pad=0, max_len=192)
    assert torch.equal(tokens, t2) and torch.equal(mask, m2)
    assert tokens.dtype == torch.long and mask.dtype == torch.bool and tokens.shape == mask.shape
    assert (tokens[~mask] == 0).all() and (tokens[mask] >= 4).all()
    assert (mask.sum(1).numpy() == np.diff(store.blk_tok_ptr)[pick]).all()


def test_collate_blocks_pads_and_clips():
    tokens, mask = collate_blocks([np.array([5, 6, 7]), np.array([8]), np.array([], dtype=np.int32)], pad=0)
    assert tokens.tolist() == [[5, 6, 7], [8, 0, 0], [0, 0, 0]] and mask.tolist() == [[True] * 3, [True, False, False],
                                                                                 [False] * 3]
    tokens, mask = collate_blocks([np.arange(1, 8), np.array([9, 9])], pad=3, max_len=4)
    assert tokens.tolist() == [[1, 2, 3, 4], [9, 9, 3, 3]] and mask.sum().item() == 6
    tokens, mask = collate_blocks([])
    assert tokens.shape == (0, 0) and mask.dtype == torch.bool


@needs_cache
def test_export_refuses_a_directory_outside_runs_export(tmp_path):
    with pytest.raises(ValueError):
        be.resolve_out(str(tmp_path / 'x'))
    with pytest.raises(ValueError):
        be.resolve_out('RUNS/other/x')
    with pytest.raises(ValueError):
        be.resolve_out('RUNS/export')
    assert be.resolve_out('RUNS/export/ours_cap0') == (RUNS / 'export' / 'ours_cap0').resolve()


def test_schema_documents_every_file_and_the_example_runs(mini):
    text = (ROOT / 'transformer_paths' / 'EXPORT_SCHEMA.md').read_text(encoding='utf-8')
    for name in list(be.STORE_ARRAYS) + ['store.json', 'binaries.json', 'manifest.json', 'lut.npy', 'vocab.json',
                                          'block_key.npy']:
        assert f'`{name.removesuffix(".npy")}' in text or name in text, name
    for dtype in ('int32', 'int8', 'int64', 'float16', 'float32'):
        assert dtype in text
    code = re.search(r'```python\n(.*?)```', text, re.S).group(1)
    assert 1 <= len([l for l in code.splitlines() if l.strip()]) <= 15
    ast.parse(code)
    code = re.sub(r"BlockStore\('[^']*'\)", f'BlockStore({mini.root.as_posix()!r})', code)
    scope = {}
    exec(compile(code, 'EXPORT_SCHEMA.md', 'exec'), scope)                           # iterates fold 0's training blocks
    assert scope['loss'].item() >= 0 and len(scope['block_ids']) == len(scope['keys'])


# the encoder contract -----------------------------------------------------------------------------------------------


def make_checkpoint(mini, tmp_path, name='ok', model=None, init=None, **override):
    store = mini.store
    ids = lambda split: [store.binaries[i]['sample_id'] for i in store.fold(0)['manifest'][split]]
    init = init or {'n_ids': store.fold(0)['vocab']['n_ids']}
    model = model or re_.ReferenceEncoder(**init)
    ckpt = ec.build_checkpoint(model, init, mini.root, 0, ids('train'), ids('val'))
    ckpt.update(override)
    path = tmp_path / f'{name}.pt'
    torch.save(ckpt, path)
    return path, ckpt


@needs_cache
def test_reference_checkpoint_passes_every_check(mini, tmp_path):
    path, ckpt = make_checkpoint(mini, tmp_path)
    assert set(ec.CHECKPOINT_KEYS) <= set(ckpt)
    assert (ckpt['module'], ckpt['class']) == ('transformer_paths.reference_encoder', 'ReferenceEncoder')
    assert (ckpt['d_out'], ckpt['max_len'], ckpt['fold']) == (64, 192, 0)
    passed = ec.check_checkpoint(path, mini.root, 0)
    assert {'export_sha', 'manifest_sha', 'vocab_sha', 'train_ids_in_train', 'val_ids_in_val', 'no_test_ids',
            'output_shape', 'eval_deterministic', 'batch_independent', 'padding_independent'} <= set(passed)
    encoder = ec.load_encoder(path, 'cpu')
    assert isinstance(encoder, ec.BlockEncoder) and not encoder.training
    tokens, mask = collate_blocks([np.array([4, 5, 6]), np.array([7])])
    assert encoder(tokens, mask).shape == (2, 64)
    # the helper writes one valid checkpoint per fold
    paths = re_.write_reference_checkpoints(mini.root, 'ref', [0], out_root=tmp_path / 'encoders')
    assert paths == [tmp_path / 'encoders' / 'ref' / 'fold0' / 'best.pt']
    ec.check_checkpoint(paths[0], mini.root, 0)
    assert re_.ReferenceEncoder(10).embed.weight.equal(re_.ReferenceEncoder(10).embed.weight)
    assert not re_.ReferenceEncoder(10, seed=1).embed.weight.equal(re_.ReferenceEncoder(10).embed.weight)


class NoisyEncoder(torch.nn.Module):
    d_out, max_len = 8, 192

    def __init__(self, n_ids):
        super().__init__()
        self.embed = torch.nn.Embedding(n_ids, 8, padding_idx=0)

    def forward(self, tokens, mask):
        return (self.embed(tokens) * mask[..., None]).sum(1) + torch.rand(len(tokens), 8)


class BatchContextEncoder(NoisyEncoder):
    def forward(self, tokens, mask):
        out = (self.embed(tokens) * mask[..., None]).sum(1)
        return out - out.mean(0, keepdim=True)


class PaddingSensitiveEncoder(NoisyEncoder):
    def forward(self, tokens, mask):
        return self.embed(tokens).mean(1)                                              # padding counts in the mean


class NanOnEmptyEncoder(NoisyEncoder):
    def forward(self, tokens, mask):
        return (self.embed(tokens) * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)    # 0 / 0 on an empty block


class WrongShapeEncoder(NoisyEncoder):
    def forward(self, tokens, mask):
        return (self.embed(tokens) * mask[..., None]).sum(1)[:, :4]


@needs_cache
@pytest.mark.parametrize('name, override, message', [
    ('export_sha', dict(export_sha='0' * 64), 'export_sha'),
    ('fold', dict(fold=1), 'fold'),
    ('manifest_sha', dict(manifest_sha='1' * 64), 'manifest_sha'),
    ('vocab_sha', dict(vocab_sha='2' * 64), 'vocab_sha'),
    ('d_out', dict(d_out=65), 'd_out_max_len'),
    ('max_len', dict(max_len=100), 'd_out_max_len'),
])
def test_check_checkpoint_rejects_mismatched_metadata(mini, tmp_path, name, override, message):
    path, _ = make_checkpoint(mini, tmp_path, name, **override)
    with pytest.raises(ec.ContractError, match=message):
        ec.check_checkpoint(path, mini.root, 0)


@needs_cache
def test_check_checkpoint_rejects_leaked_or_foreign_ids(mini, tmp_path):
    store = mini.store
    ids = lambda split: [store.binaries[i]['sample_id'] for i in store.fold(0)['manifest'][split]]
    test_id, val_id, train_id = ids('test')[0], ids('val')[0], ids('train')[0]
    cases = {
        'test_in_train': dict(train_ids=ids('train') + [test_id], train_ids_sha=ec.sha_of_ids(ids('train') + [test_id])),
        'test_in_val': dict(val_ids=ids('val') + [test_id], val_ids_sha=ec.sha_of_ids(ids('val') + [test_id])),
        'val_in_train': dict(train_ids=ids('train') + [val_id], train_ids_sha=ec.sha_of_ids(ids('train') + [val_id])),
        'train_in_val': dict(val_ids=ids('val') + [train_id], val_ids_sha=ec.sha_of_ids(ids('val') + [train_id])),
        'stale_sha': dict(train_ids=ids('train')[:-1]),
        'no_train': dict(train_ids=[], train_ids_sha=ec.sha_of_ids([])),
    }
    for name, override in cases.items():
        path, _ = make_checkpoint(mini, tmp_path, name, **override)
        with pytest.raises(ec.ContractError) as e:
            ec.check_checkpoint(path, mini.root, 0)
        text = str(e.value)
        assert {'test_in_train': 'no_test_ids', 'test_in_val': 'no_test_ids', 'val_in_train': 'train_ids_in_train',
                'train_in_val': 'val_ids_in_val', 'stale_sha': 'train_ids_sha', 'no_train': 'train_ids_in_train'}[name] in text


@needs_cache
def test_check_checkpoint_rejects_encoders_that_break_the_function_rule(mini, tmp_path):
    n_ids = mini.store.fold(0)['vocab']['n_ids']
    assert (np.diff(mini.store.blk_tok_ptr) == 0).any()                              # the sample has an empty block
    for cls, check in ((NoisyEncoder, 'eval_deterministic'), (BatchContextEncoder, 'batch_independent'),
                       (PaddingSensitiveEncoder, 'padding_independent'), (WrongShapeEncoder, 'output_shape'),
                       (NanOnEmptyEncoder, 'finite')):
        path, _ = make_checkpoint(mini, tmp_path, cls.__name__, model=cls(n_ids), init={'n_ids': n_ids})
        with pytest.raises(ec.ContractError, match=check):
            ec.check_checkpoint(path, mini.root, 0)


@needs_cache
def test_check_checkpoint_rejects_an_incomplete_or_unimportable_checkpoint(mini, tmp_path):
    path, ckpt = make_checkpoint(mini, tmp_path, 'missing')
    torch.save({k: v for k, v in ckpt.items() if k != 'state_dict'}, path)
    with pytest.raises(ec.ContractError, match='missing keys'):
        ec.check_checkpoint(path, mini.root, 0)
    path, _ = make_checkpoint(mini, tmp_path, 'nomodule', module='no_such_module_anywhere')
    with pytest.raises(ec.ContractError, match='cannot load'):
        ec.check_checkpoint(path, mini.root, 0)


# precomputed block vectors ------------------------------------------------------------------------------------------


@pytest.fixture(scope='module')
def vectors(mini):
    """Reference checkpoints and block vectors of the mini export, in a temporary cache directory."""
    enc_root, cache = mini.tmp / 'encoders', mini.tmp / 'cache'
    re_.write_reference_checkpoints(mini.root, 'ref', [0], out_root=enc_root)
    metas = pb.precompute(mini.root, 'ref', [0], enc_root=enc_root, cache=cache, device='cpu', log=lambda *_: None)
    return SimpleNamespace(cache=cache, name='mini_ref', meta=metas[0], enc_root=enc_root)


@needs_cache
def test_precompute_embeds_each_distinct_block_once(mini, vectors):
    meta = vectors.meta
    assert meta['name'] == 'mini_ref' and meta['n_distinct'] == mini.store.fold(0)['manifest']['n_distinct_blocks']
    assert meta['n_distinct'] < meta['n_blocks'] and meta['dtype'] == 'float16' and meta['secs'] > 0
    bv = BlockVecs(vectors.cache, vectors.name, 0)
    assert bv.keys.dtype == np.int64 and (np.diff(bv.keys) > 0).all() and bv.vectors.dtype == np.float16
    assert bv.vectors.shape == (meta['n_distinct'], 64) and bv.d == 64
    all_keys = np.asarray(mini.store.fold(0)['block_key'])
    assert np.array_equal(bv.keys, np.unique(all_keys))                              # n_distinct and zero missing keys
    got = bv.lookup(all_keys)                                                        # every key is found
    assert got.shape == (len(all_keys), 64) and got.dtype == np.float16
    order = np.argsort(all_keys, kind='stable')
    same = np.flatnonzero(np.diff(all_keys[order]) == 0)
    assert len(same) > 0
    assert np.array_equal(got[order[same]], got[order[same + 1]])                    # identical sequences, identical vectors
    # the stored vector is the encoder's output on that block (bf16 autocast and float16 storage)
    encoder = ec.load_encoder(vectors.enc_root / 'ref' / 'fold0' / 'best.pt', 'cpu')
    view = FoldView(mini.store, 0, 'all')
    blocks = np.array([0, 3, 11, len(all_keys) - 1])
    tokens, mask = view.padded(blocks, encoder.max_len)
    with torch.no_grad():
        want = encoder(tokens, mask).numpy()
    assert np.allclose(got[blocks].astype(np.float32), want, atol=2e-2, rtol=2e-2)
    for key in ('export_sha', 'checkpoint_sha', 'manifest_sha', 'blocks_per_sec'):
        assert key in meta


@needs_cache
def test_lookup_asserts_that_every_key_is_found(vectors):
    bv = BlockVecs(vectors.cache, vectors.name, 0)
    missing = np.array([bv.keys[0], bv.keys.max() + 1, bv.keys[5] + 0])
    missing[1] = bv.keys.max() - 1 if bv.keys.max() - 1 not in bv.keys else bv.keys.min() - 5
    with pytest.raises(AssertionError, match='not in'):
        bv.lookup(missing)
    with pytest.raises(KeyError):
        bv.lookup(np.array([bv.keys.min() - 1]))
    assert issubclass(MissingKeys, AssertionError) and issubclass(MissingKeys, KeyError)
    assert bv.lookup(np.zeros(0, dtype=np.int64)).shape == (0, 64)


@needs_cache
def test_precompute_refuses_a_checkpoint_that_breaks_the_contract(mini, tmp_path):
    bad = tmp_path / 'enc'
    store = mini.store
    ids = lambda split: [store.binaries[i]['sample_id'] for i in store.fold(0)['manifest'][split]]
    n_ids = store.fold(0)['vocab']['n_ids']
    ckpt = ec.build_checkpoint(re_.ReferenceEncoder(n_ids), {'n_ids': n_ids}, mini.root, 0,
                               ids('train') + ids('test'), ids('val'))
    (bad / 'ref' / 'fold0').mkdir(parents=True)
    torch.save(ckpt, bad / 'ref' / 'fold0' / 'best.pt')
    with pytest.raises(ec.ContractError, match='no_test_ids'):
        pb.precompute(mini.root, 'ref', [0], enc_root=bad, cache=tmp_path / 'cache', device='cpu', log=lambda *_: None)
    assert not (tmp_path / 'cache').exists()


def test_length_buckets_cover_every_block_once():
    lengths = np.random.default_rng(0).integers(1, 193, 5000)
    batches = pb.length_buckets(lengths, max_tokens=4096, max_blocks=300)
    flat = np.concatenate(batches)
    assert sorted(flat) == list(range(5000))
    for b in batches:
        assert len(b) <= 300 and len(b) * lengths[b].max() <= 4096 or len(b) == 1
        assert lengths[b].max() - lengths[b].min() <= 192
    assert pb.length_buckets(np.zeros(0, dtype=int)) == []


@needs_cache
def test_block_vec_source_resolves_blocks_by_cache_id_and_pickles(mini, vectors):
    sid = mini.store.binaries[3]['sample_id']
    first, last = mini.store.block_range(3)
    cache_ids = np.asarray(mini.store.blk_cache_id[first:last])
    source = BlockVecSource(vectors.cache, vectors.name, 0)
    assert source.d == 64 and Path(source.export_root) == mini.root
    full = source.block_vectors(sid, cache_ids)
    assert full.shape == (len(cache_ids), 64) and full.dtype == np.float16 and full.flags.writeable
    subset = cache_ids[::3]                                                          # functions dropped at training time
    assert np.array_equal(source.block_vectors(sid, subset), full[::3])
    assert np.array_equal(source.block_vectors(sid, cache_ids[:0]), full[:0])
    clone = pickle.loads(pickle.dumps(source))                                       # DataLoader workers get a copy
    assert np.array_equal(clone.block_vectors(sid, subset), full[::3])
    with pytest.raises(MissingKeys):
        source.block_vectors(sid, np.array([cache_ids.max() + 1000]))
    with pytest.raises(KeyError):
        source.block_vectors('0' * 64, subset)
    vec_meta = json.loads((vectors.cache / 'side' / 'blockvec' / vectors.name / 'fold0' / 'meta.json').read_text())
    assert source.store_meta()['export_sha'] == vec_meta['export_sha']


# the vec input of the GNN -------------------------------------------------------------------------------------------


def vec_config(**kw):
    return gnn(protocol='kfold', fold=0, arch='x86', seed=0, val_seed=0, block_input='vec', block_vec='mini_ref', **kw)


@needs_cache
def test_encode_binary_with_vectors(mini, vectors):
    source = BlockVecSource(vectors.cache, vectors.name, 0)
    sid = mini.store.binaries[3]['sample_id']
    z = load_sample(sid)
    cfg = vec_config()
    out = encode_binary(z, None, cfg, None, sid, source)
    token_out = encode_binary(z, TokenVocab(mini.manifest.train, cfg), replace(cfg, block_input='tokens', block_vec=''),
                              None, sid)
    assert not {'tokens', 'tok_offsets', 'tok_weights'} & set(out)
    assert set(out) == (set(token_out) - {'tokens', 'tok_offsets', 'tok_weights'}) | {'blk_vec'}
    for k in set(out) - {'blk_vec'}:
        assert np.array_equal(out[k], token_out[k]), k
    n_blk = len(out['blk_seg'])
    assert out['blk_vec'].shape == (n_blk, 64) and out['blk_vec'].dtype == np.float16
    first, last = mini.store.block_range(3)
    keys = np.asarray(mini.store.fold(0)['block_key'][first:last])
    assert np.array_equal(out['blk_vec'], BlockVecs(vectors.cache, vectors.name, 0).lookup(keys))
    # identical sequences get identical vectors within one binary
    same = np.flatnonzero(np.unique(keys, return_counts=True)[1] > 1)
    assert len(same) > 0
    uniq, inverse = np.unique(keys, return_inverse=True)
    dup = np.flatnonzero(inverse == same[0])
    assert len(dup) > 1 and (out['blk_vec'][dup] == out['blk_vec'][dup[0]]).all()
    # block masking zeroes the vector row and the block features of the hit blocks; evaluation never masks
    masked_cfg = vec_config(mask_blocks=0.5)
    masked = encode_binary(z, None, masked_cfg, np.random.default_rng(5), sid, source)
    hit = np.random.default_rng(5).random(n_blk) < 0.5                                 # the draw the token path makes
    assert 0.2 < hit.mean() < 0.8
    assert not masked['blk_vec'][hit].any() and (masked['blk_feat'][hit] == 0).all()
    assert np.array_equal(masked['blk_vec'][~hit], out['blk_vec'][~hit])
    assert np.array_equal(masked['blk_feat'][~hit], out['blk_feat'][~hit]) and out['blk_vec'][~hit].any()
    assert np.array_equal(encode_binary(z, None, masked_cfg, None, sid, source)['blk_vec'], out['blk_vec'])
    # function dropping keeps the vectors of the blocks that stay
    dropped = encode_binary(z, None, vec_config(drop_functions=0.5), np.random.default_rng(1), sid, source)
    assert 0 < len(dropped['blk_vec']) < n_blk


@needs_cache
def test_collate_with_vectors_keeps_edges_and_bin_feat(mini, vectors):
    source = BlockVecSource(vectors.cache, vectors.name, 0)
    cfg = vec_config()
    rows = mini.manifest.train[:3]
    ds = BinaryDataset(rows, None, cfg, source)
    batch = collate([ds[(i, None)] for i in range(3)])
    assert 'tokens' not in batch and 'tok_weights' not in batch and 'tok_offsets' not in batch
    assert batch['blk_vec'].dtype == torch.float16 and batch['blk_vec'].shape == (len(batch['blk_seg']), 64)
    assert batch['edge_type'].shape == batch['edge_src'].shape and batch['bin_feat'].shape == (3, 5)
    assert 'stack_logit' not in batch and batch['n_fn'] == len(batch['seg_len'])
    token_ds = BinaryDataset(rows, TokenVocab(mini.manifest.train, cfg), replace(cfg, block_input='tokens', block_vec=''))
    token_batch = collate([token_ds[(i, None)] for i in range(3)])
    assert set(batch) == (set(token_batch) - {'tokens', 'tok_offsets', 'tok_weights'}) | {'blk_vec'}
    for k in set(batch) - {'blk_vec'}:
        same = torch.equal(batch[k], token_batch[k]) if torch.is_tensor(batch[k]) else batch[k] == token_batch[k]
        assert same, k
    loaded = list(loader(rows, None, cfg, train=False, vecs=source))
    assert sum(len(b['label']) for b in loaded) == 3 and all('blk_vec' in b for b in loaded)


sys.path.insert(0, str(Path(__file__).parent))
from test_gnn_model import D, DEV, collate as model_collate, make_bins  # noqa: E402


def vec_batch(bins, d_vec=6, **kw):
    """A synthetic contract batch whose block vectors are a function of the block's token ids (so equal token lists give
    equal vectors) in place of the token keys."""
    batch = model_collate(bins, **kw)
    table = torch.randn(40, d_vec, generator=torch.Generator().manual_seed(0))
    tokens, offsets = batch['tokens'].cpu(), batch['tok_offsets'].cpu()
    ends = torch.cat([offsets[1:], torch.tensor([len(tokens)])])
    vec = torch.stack([table[tokens[a:b]].sum(0) if b > a else torch.zeros(d_vec) for a, b in zip(offsets, ends)])
    for k in ('tokens', 'tok_offsets', 'tok_weights'):
        del batch[k]
    batch['blk_vec'] = vec.to(batch['blk_feat'].device).half()
    return batch


def vec_model(d_vec=6, **kw):
    torch.manual_seed(0)
    cfg = RunConfig(model='gnn', d_model=D, dropout=0.0, protocol='kfold', fold=0, block_input='vec', block_vec='x', **kw)
    return HierGNN(cfg, 0, d_vec).to(DEV).eval()


def scrambled(batch, seed=0):
    """The same batch with its CFG edges, degrees and call edges replaced by random ones."""
    g = torch.Generator().manual_seed(seed)
    out = {k: (v.clone() if torch.is_tensor(v) else copy.copy(v)) for k, v in batch.items()}
    n = len(out['edge_src'])
    out['edge_src'] = torch.randint(0, 3, (n,), generator=g).to(DEV)
    out['edge_dst'] = torch.randint(0, 3, (n,), generator=g).to(DEV)
    out['edge_type'] = torch.randint(0, 12, (n,), generator=g).to(DEV)
    out['deg_in'] = torch.randint(0, 9, out['deg_in'].shape, generator=g).to(DEV)
    out['deg_out'] = torch.randint(0, 9, out['deg_out'].shape, generator=g).to(DEV)
    out['blk_feat'][:, 4:6] = torch.rand(len(out['blk_feat']), 2, generator=g).to(DEV) * 4
    out['call_src'], out['call_dst'] = out['call_dst'].clone(), out['call_src'].clone()
    return out


def test_vec_no_edges_forward_ignores_edges_degrees_and_calls_but_typed_does_not():
    bins = make_bins()
    batch = vec_batch(bins)
    plain, ctl = vec_model(), vec_model(control='no_edges')
    with torch.no_grad():
        a, b = ctl(batch), ctl(scrambled(batch))
        assert torch.allclose(a, b, atol=1e-6), (a, b)
        assert torch.isfinite(a).all() and a.shape == (3,)
        typed_a, typed_b = plain(batch), plain(scrambled(batch))
        assert (typed_a - typed_b).abs().max().item() > 1e-4
        # the no-edge model still reads the vectors and the other block features
        changed = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
        changed['blk_vec'] = changed['blk_vec'] + 1
        assert (ctl(changed) - a).abs().max().item() > 1e-4


def test_vec_model_gives_identical_vectors_to_identical_blocks_and_has_its_own_parameters():
    batch = vec_batch(make_bins())
    model = vec_model()
    keys = set(model.state_dict())
    assert 'vec.weight' in keys and 'vec.bias' in keys and 'blk_feat.weight' in keys
    assert not any(k.startswith('tokens.') for k in keys) and model.vec.in_features == 6 and model.vec.out_features == D
    batch['blk_vec'][1], batch['blk_feat'][1] = batch['blk_vec'][0], batch['blk_feat'][0]
    with torch.no_grad():
        h = model.encode_blocks(batch, batch['blk_feat'])
    assert torch.equal(h[0], h[1])
    token_keys = set(HierGNN(RunConfig(model='gnn', d_model=D), 40).state_dict())
    assert 'tokens.weight' in token_keys and not any(k.startswith('vec.') for k in token_keys)
    assert {k for k in keys if not k.startswith('vec.')} == {k for k in token_keys if not k.startswith('tokens.')}
    with pytest.raises(ValueError, match='d_vec'):
        HierGNN(RunConfig(model='gnn', protocol='kfold', fold=0, block_input='vec', block_vec='x'), 0)


def test_vec_model_trains_on_float16_vectors():
    batch = vec_batch(make_bins())
    assert batch['blk_vec'].dtype == torch.float16
    model = vec_model().train()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(batch), batch['label'])
    loss.backward()
    assert model.vec.weight.grad is not None and torch.isfinite(model.vec.weight.grad).all()
    assert model.vec.weight.grad.abs().sum() > 0


# the token path stays as it was -------------------------------------------------------------------------------------


@needs_cache
def test_token_path_batches_have_no_vector_or_stack_keys():
    index = load_index()
    rows = make_manifest(gnn(protocol='kfold', fold=0), index).train[:2]
    cfg = gnn(protocol='kfold', fold=0)
    batch = next(iter(loader(rows, TokenVocab(rows, cfg), cfg, train=False)))
    assert 'blk_vec' not in batch and 'stack_logit' not in batch and 'tokens' in batch and 'tok_weights' in batch


# stacking -----------------------------------------------------------------------------------------------------------


def synthetic_problem(n=400, d=30, seed=0):
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    groups = np.array([f'g{i // 5}' for i in range(n)])
    y = (np.arange(n) // 5 % 2).astype(int)
    x = rng.poisson(0.3, (n, d)).astype(float)
    x[:, 0] += 3 * y                                                                  # a real signal
    return sp.csr_matrix(x), y, groups


def test_inner_folds_never_split_a_group_and_keep_both_classes():
    x, y, groups = synthetic_problem()
    folds = baselines.inner_folds(y, groups, seed=0)
    assert len(folds) == 5
    held_all = np.concatenate([held for _, held in folds])
    assert sorted(held_all) == list(range(len(y)))
    for fit, held in folds:
        assert not set(groups[fit]) & set(groups[held])
        assert set(y[held]) == {0, 1} and set(y[fit]) == {0, 1}
    assert [list(h) for _, h in folds] == [list(h) for _, h in baselines.inner_folds(y, groups, seed=0)]


def test_oof_logit_of_a_training_row_differs_from_the_logit_of_a_model_that_saw_it():
    x, y, groups = synthetic_problem()
    oof, which = baselines.oof_logits(x, y, groups, min_df=2, seed=0)
    folds = baselines.inner_folds(y, groups, seed=0)
    assert set(which) == {0, 1, 2, 3, 4}
    for k in (0, 3):
        fit, held = folds[k]
        row = held[0]
        # a model fitted without the row's group reproduces the stored logit exactly ...
        (without,) = baselines.fit_logits(x[fit], y[fit], [x[[row]]], 2)
        assert without[0] == pytest.approx(oof[row], abs=1e-12) and which[row] == k
        # ... and a model fitted with the row (the full fit) gives another one
        (with_row,) = baselines.fit_logits(x[np.append(fit, row)], y[np.append(fit, row)], [x[[row]]], 2)
        assert abs(with_row[0] - oof[row]) > 1e-6
    (full,) = baselines.fit_logits(x, y, [x], 2)
    assert np.abs(full - oof).mean() > 1e-3 and (full != oof).all()
    assert np.corrcoef(full, oof)[0, 1] > 0.5                                         # the signal survives


def test_a_memorised_row_shows_in_the_full_fit_but_not_in_the_out_of_fold_logit():
    import scipy.sparse as sp
    x, y, groups = synthetic_problem(seed=1)
    x = sp.hstack([x, sp.csr_matrix((np.arange(len(y)) == 7).astype(float)[:, None] * 5)]).tocsr()   # a feature only row 7 has
    y = y.copy()
    y[7] = 1 - y[7]                                                                   # and it disagrees with its group
    groups = groups.copy()
    oof, _ = baselines.oof_logits(x, y, groups, min_df=1, seed=0)
    (full,) = baselines.fit_logits(x, y, [x], 1)
    sign = 1 if y[7] == 1 else -1
    assert sign * full[7] > sign * oof[7]                                             # the full fit has learnt row 7
    assert sign * oof[7] < 0.5


def test_inner_groups_keep_classes_apart():
    rows = [dict(label=1, family='a', identity_group='x'), dict(label=0, family='a', identity_group='a'),
            dict(label=0, family='goodware', identity_group='x')]
    g = baselines.inner_groups(rows)
    assert len(set(g)) == 3 and g[0] == 'fam:a' and g[1] == 'id:a'


@needs_cache
def test_baseline_name_matches_the_directories_baselines_writes():
    assert baselines.baseline_name(RunConfig(protocol='kfold', fold=0)) == 'kfold0_x86_s0'
    assert baselines.baseline_name(gnn(protocol='kfold', fold=3, seed=2, mask_blocks=0.15, tag='smoke')) == 'kfold3_x86_s2'
    assert baselines.baseline_name(RunConfig(protocol='kfold_v3', fold=1, arch='all')) == 'v3kfold1_all_s0'
    assert baselines.baseline_name(RunConfig()) == 'bundle_x86_s0'
    assert baselines.baseline_name(gnn(protocol='kfold', fold=0, seed=1, val_seed=0)) == 'kfold0_x86-vs0_s1'


@pytest.fixture(scope='module')
def stack_files(tmp_path_factory):
    """stack_logits.csv for fold 0 of the real cache in a temporary baselines directory."""
    if not (CACHE / 'index.csv').exists():
        pytest.skip('needs the CFG cache')
    root = tmp_path_factory.mktemp('baselines')
    cfg = RunConfig(protocol='kfold', fold=0, arch='x86', seed=0)
    sizes = {k: len(v) for k, v in json.loads((CACHE / 'vocab.json').read_text(encoding='utf-8')).items()}
    out = baselines.write_stack_logits(cfg, sizes, root)
    return SimpleNamespace(root=root, dir=out, cfg=cfg, manifest=make_manifest(cfg, load_index()))


@needs_cache
def test_stack_logits_csv_roles_and_leakage(stack_files):
    import csv
    rows = list(csv.DictReader((stack_files.dir / 'stack_logits.csv').open(encoding='utf-8')))
    m = stack_files.manifest
    assert [r['split'] for r in rows].count('train') == len(m.train) and len(rows) == len(m.train) + len(m.val) + len(m.test)
    assert list(rows[0]) == ['sample_id', 'split', 'inner_fold', 'label', 'family', 'logit']
    by_split = {s: {r['sample_id'] for r in rows if r['split'] == s} for s in ('train', 'val', 'test')}
    assert by_split['train'] == {r['sample_id'] for r in m.train} and by_split['test'] == {r['sample_id'] for r in m.test}
    train_rows = [r for r in rows if r['split'] == 'train']
    assert {r['inner_fold'] for r in train_rows} == {'0', '1', '2', '3', '4'}
    assert all(r['inner_fold'] == '' for r in rows if r['split'] != 'train')
    # inner folds are grouped: a family or goodware identity group sits in one inner fold only
    index = {r['sample_id']: r for r in m.train}
    where = {}
    for r in train_rows:
        src = index[r['sample_id']]
        group = ('fam', src['family']) if src['label'] == 1 else ('id', src['identity_group'])
        where.setdefault(group, set()).add(r['inner_fold'])
    assert all(len(v) == 1 for v in where.values())
    meta = json.loads((stack_files.dir / 'stack_logits.json').read_text(encoding='utf-8'))
    assert meta['inner_splits'] == 5 and meta['n_train'] == len(m.train)
    # a training row's out-of-fold logit differs from the full-fit logit of a model that includes it
    sizes = {k: len(v) for k, v in json.loads((CACHE / 'vocab.json').read_text(encoding='utf-8')).items()}
    x_train, _ = baselines.load_features(m.train[:], sizes)
    y = np.array([r['label'] for r in m.train])
    (full,) = baselines.fit_logits(x_train, y, [x_train], stack_files.cfg.min_df)
    oof = np.array([float(next(r['logit'] for r in train_rows if r['sample_id'] == s['sample_id'])) for s in m.train])
    assert (full != oof).all() and np.abs(full - oof).mean() > 1e-3
    pos = y == 1
    assert np.abs(full[pos]).mean() > np.abs(oof[pos]).mean()                         # in-sample logits are more confident


@needs_cache
def test_read_stack_logits_checks_roles(stack_files, tmp_path):
    m = stack_files.manifest
    path = stack_files.dir / 'stack_logits.csv'
    got = baselines.read_stack_logits(path, m)
    assert len(got) == len(m.train) + len(m.val) + len(m.test)
    swapped = Manifest(m.train[1:], m.val + m.train[:1], m.test, m.val_families)      # a train row asked as a val row
    with pytest.raises(ValueError, match='needs it as a val row'):
        baselines.read_stack_logits(path, swapped)
    extra = Manifest(m.train, m.val, m.test + [{**m.train[0], 'sample_id': 'f' * 64}], m.val_families)
    with pytest.raises(ValueError, match='missing'):
        baselines.read_stack_logits(path, extra)
    with pytest.raises(FileNotFoundError, match='--stack-logits'):
        baselines.read_stack_logits(tmp_path / 'none.csv', m)
    subset = Manifest(m.train[::2], m.val, m.test, m.val_families)                    # --train-fraction: a subset is fine
    assert set(baselines.read_stack_logits(path, subset)) == {r['sample_id'] for r in subset.train + subset.val + subset.test}


@needs_cache
def test_train_standardises_stack_logits_on_the_training_rows(stack_files, monkeypatch):
    monkeypatch.setattr(train, 'RUNS', stack_files.root.parent)
    (stack_files.root.parent / 'baselines').mkdir(exist_ok=True)
    link = stack_files.root.parent / 'baselines' / stack_files.dir.name
    if not link.exists():
        shutil.copytree(stack_files.dir, link)
    cfg = gnn(protocol='kfold', fold=0, arch='x86', seed=0, stack_tfidf=True, mask_blocks=0.15)
    z = train.stack_logits(cfg, stack_files.manifest)
    t = np.array([z[r['sample_id']] for r in stack_files.manifest.train])
    assert abs(t.mean()) < 1e-5 and abs(t.std() - 1) < 1e-4
    raw = baselines.read_stack_logits(stack_files.dir / 'stack_logits.csv', stack_files.manifest)
    val = np.array([raw[r['sample_id']] for r in stack_files.manifest.val])
    raw_train = np.array([raw[r['sample_id']] for r in stack_files.manifest.train])
    assert np.allclose([z[r['sample_id']] for r in stack_files.manifest.val],
                       (val - raw_train.mean()) / raw_train.std(), atol=1e-5)
    assert all(isinstance(v, np.float32) for v in z.values())


@needs_cache
def test_dataset_and_collate_carry_the_stack_logit(mini):
    cfg = gnn(protocol='kfold', fold=0)
    rows = mini.manifest.train[:3]
    stack = {r['sample_id']: np.float32(i - 1.0) for i, r in enumerate(rows)}
    ds = BinaryDataset(rows, TokenVocab(mini.manifest.train, cfg), cfg, stack=stack)
    item = ds[(0, None)]
    assert item['stack_logit'] == np.float32(-1.0)
    batch = collate([ds[(i, None)] for i in range(3)])
    assert batch['stack_logit'].dtype == torch.float32 and batch['stack_logit'].tolist() == [-1.0, 0.0, 1.0]


# GPU smoke runs -----------------------------------------------------------------------------------------------------


@pytest.fixture
def smoke_env(mini, vectors, tmp_path, monkeypatch):
    """train.run on the mini manifest, with runs written to a temporary directory and the vectors read from the
    temporary cache (the export's recorded cache is pointed at it so that the run's consistency check holds)."""
    store_json = mini.root / 'store' / 'store.json'
    original = store_json.read_text(encoding='utf-8')
    meta = json.loads(original)
    meta['cache'] = str(vectors.cache)
    store_json.write_text(json.dumps(meta), encoding='utf-8')
    monkeypatch.setattr(train, 'RUNS', tmp_path / 'runs')
    monkeypatch.setattr(train, 'CACHE', vectors.cache)
    monkeypatch.setattr(train, 'make_manifest', lambda cfg, index: mini.manifest)
    monkeypatch.setattr(train, 'TokenVocab', lambda rows, cfg: TokenVocab(rows, cfg))
    yield SimpleNamespace(runs=tmp_path / 'runs', mini=mini)
    store_json.write_text(original, encoding='utf-8')


def smoke_config(**kw):
    return gnn(protocol='kfold', fold=0, arch='x86', seed=0, val_seed=0, model='gnn', fixed_epochs=2, mask_blocks=0.15,
               batch_blocks=100000, tag='smoke', **kw)


def check_run_outputs(runs, name):
    d = runs / name
    for f in ('config.json', 'manifest.json', 'history.csv', 'val_probs.csv', 'best.pt', 'best_auc.pt', 'predictions.csv',
              'predictions_auc.csv', 'metrics.json'):
        assert (d / f).exists(), f
    metrics = json.loads((d / 'metrics.json').read_text())
    assert metrics['epochs_run'] == 2 and 0 <= metrics['test']['auc'] <= 1 and np.isfinite(metrics['val_macro_f1'])
    lines = (d / 'predictions.csv').read_text().splitlines()
    assert len(lines) == 1 + 2 + 2 and lines[0].startswith('sample_id,split,label,prob')
    return metrics


@needs_cuda
@needs_cache
@pytest.mark.parametrize('control', ['no_edges', ''], ids=['path1', 'path2'])
def test_smoke_two_epochs_of_path_1_and_path_2(smoke_env, control):
    cfg = smoke_config(block_input='vec', block_vec='mini_ref', control=control)
    metrics = train.run(cfg)
    assert metrics['run'] == cfg.name
    check_run_outputs(smoke_env.runs, cfg.name)
    saved = RunConfig.load(smoke_env.runs / cfg.name / 'config.json')
    assert (saved.block_input, saved.block_vec, saved.val_seed) == ('vec', 'mini_ref', 0)
    state = torch.load(smoke_env.runs / cfg.name / 'best.pt', map_location='cpu')
    assert 'vec.weight' in state and 'tokens.weight' not in state
    assert json.loads((smoke_env.runs / cfg.name / 'vocab.json').read_text())['block_vec'] == 'mini_ref'


@needs_cuda
@needs_cache
def test_smoke_two_epochs_of_the_stack_arm(smoke_env, monkeypatch, tmp_path):
    m = smoke_env.mini.manifest
    cfg = smoke_config(stack_tfidf=True)
    rng = np.random.default_rng(0)
    base = tmp_path / 'runs' / 'baselines' / baselines.baseline_name(cfg)
    base.mkdir(parents=True)
    with (base / 'stack_logits.csv').open('w') as f:
        f.write('sample_id,split,inner_fold,label,family,logit\n')
        for split, rows in (('train', m.train), ('val', m.val), ('test', m.test)):
            for r in rows:
                f.write(f"{r['sample_id']},{split},,{r['label']},{r['family']},{rng.normal() + 2 * r['label']}\n")
    metrics = train.run(cfg)
    check_run_outputs(smoke_env.runs, cfg.name)
    assert cfg.name.endswith('gnn-stack-mblk15_s0_smoke') and metrics['run'] == cfg.name
    state = torch.load(smoke_env.runs / cfg.name / 'best.pt', map_location='cpu')
    assert state['head.0.weight'].shape[1] == cfg.d_model + 5 + 1                      # the head takes the logit
    missing = replace(cfg, seed=1)                                                   # no logits for seed 1
    with pytest.raises(FileNotFoundError):
        train.run(missing)


# the paths queue and gates ------------------------------------------------------------------------------------------


def queue(*args, **env):
    r = subprocess.run([BASH, 'queue_gnn.sh', *args], cwd=ROOT, capture_output=True, text=True, env={**os.environ, **env})
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


@needs_bash
def test_queue_paths_dry_names_match_run_configs():
    lines = queue('--stage', 'paths', '--dry')
    runs = [l.split()[1:] for l in lines if l.startswith('run ')]
    assert len(runs) == 7
    for name, *argv in runs:
        cfg, _ = parse_config(argv)
        assert cfg.name == name, (name, cfg.name)
        assert cfg.protocol == 'kfold' and cfg.arch == 'x86' and cfg.seed == 0 and cfg.model == 'gnn'
        assert cfg.mask_blocks == 0.15 and cfg.select_on == 'overall' and not cfg.arch_balance
    names = [n for n, *_ in runs]
    stack = [parse_config(a)[0] for _, *a in runs[:5]]
    assert [n.split('_')[0] for n in names[:5]] == [f'kfold{f}' for f in range(5)]
    assert all(c.stack_tfidf and c.block_input == 'tokens' and c.name.split('_')[2] == 'gnn-stack-mblk15' for c in stack)
    path1, path2 = (parse_config(a)[0] for _, *a in runs[5:])
    assert (path1.control, path1.block_input, path1.block_vec, path1.fold, path1.val_seed) == ('no_edges', 'vec', 'ours_cap0_ref', 0, 0)
    assert (path2.control, path2.block_input, path2.block_vec, path2.fold, path2.val_seed) == ('', 'vec', 'ours_cap0_ref', 0, 0)
    assert path1.name == 'kfold0_x86_gnn-noedges-mblk15-bv-ours_cap0_ref_s0'
    assert path2.name == 'kfold0_x86_gnn-mblk15-bv-ours_cap0_ref_s0'
    assert lines[0] == 'cpu baselines --protocol kfold --fold 0 1 2 3 4 --arch x86 --seed 0 --stack-logits'
    assert lines[-3:] == ['cpu report.py', 'cpu gates.py --stage paths > RUNS/gates_paths.md',
                          'cpu val_threshold.py > RUNS/val_threshold.md']
    cpu = [l for l in lines if l.startswith('cpu transformer_paths')]
    assert [l.split()[1] for l in cpu] == ['transformer_paths.block_export', 'transformer_paths.reference_encoder',
                                           'transformer_paths.precompute_blockvec']
    assert not any(l.startswith(('skipped', 'stopped')) for l in lines)
    # the groups the gates read are the groups the queue writes
    groups = {parse_config(a)[0].group for _, *a in runs}
    arms = dict(gates.PATHS_ARMS)
    assert groups == {arms[k] for k in ('R7 + TF-IDF stack', 'path 1: no-edges, block vectors',
                                       'path 2: typed, block vectors')}


@needs_bash
def test_queue_paths_budget_guard_and_usage():
    lines = queue('--stage', 'paths', '--dry', GPU_BUDGET_MIN='290', DRY_RUN_MIN='100')
    assert sum(l.startswith('run ') for l in lines) == 3 and sum(l.startswith('skipped (budget) ') for l in lines) == 4
    r = subprocess.run([BASH, 'queue_gnn.sh', '--stage', 'v4', '--dry'], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and 'day1|day1b|v3|subgraph|paths' in r.stdout


def test_gates_paths_prints_missing_arms(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    gates.pooled.cache_clear()
    text = '\n'.join(gates.paths(50))
    assert text.startswith('# Gate summary, stage paths')
    for arm, group in gates.PATHS_ARMS:
        assert f'| {arm} | {group} | missing' in text
    for a, b in gates.PATHS_PAIRS:
        assert f'| {a} | {b} | missing | missing |' in text
    assert 'kfold_x86_gnn-stack-mblk15' in text and 'gnn-mblk15-bv-ours_cap0_ref' in text
    gates.pooled.cache_clear()


@pytest.mark.skipif(not (RUNS / 'baselines').exists(), reason='needs the runs')
def test_gates_paths_cli_runs_on_the_real_runs_dir():
    r = subprocess.run([sys.executable, 'gates.py', '--stage', 'paths', '--resamples', '20'], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert '## Pooled, AUC checkpoint' in r.stdout and '## Paired differences, second minus first' in r.stdout


def test_gates_group_names_follow_the_run_configs():
    arms = dict(gates.PATHS_ARMS)
    assert arms['R7 + TF-IDF stack'] == RunConfig(model='gnn', protocol='kfold', fold=0, stack_tfidf=True,
                                                  mask_blocks=0.15).group
    assert arms['R7 typed, tokens'] == RunConfig(model='gnn', protocol='kfold', fold=0, mask_blocks=0.15).group
    assert arms['R2 no-edges, tokens'] == RunConfig(model='gnn', protocol='kfold', fold=0, mask_blocks=0.15,
                                                    control='no_edges').group
    assert arms['TF-IDF'] == 'tfidf' and stats.baseline_method('tfidf')
