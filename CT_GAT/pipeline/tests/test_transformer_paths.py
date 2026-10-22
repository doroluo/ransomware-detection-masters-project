"""python -m pytest tests/test_transformer_paths.py: the block export and its reader, the encoder contract, the precomputed
block vectors and the GNN switch that reads them (`--block-input vec`), and two epochs of paths 1 and 2.

Ported from the 3-block-level-counting branch (tests/test_transformer_paths.py); the stacking, queue and gate parts of that
file belong to the stages that own them (tests/test_eval_baselines.py, the experiment queue). The export tests build a small
export (ten real binaries, one fold) in a temporary directory and need the CFG cache of this machine: they are marked
`real_data`. The two-epoch runs also need CUDA (`gpu`).
"""
import ast
import copy
import hashlib
import json
import pickle
import re
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from common.config import CACHE, RUNS, RunConfig
from common.manifests import Manifest, load_index, load_sample, make_manifest
from common.reader import BlockStore, FoldView, collate_blocks
from experiments import train
from s02_preprocess.subgraph import select_subgraph
from s02_preprocess.tokenizer.vocab import TOKEN_KINDS, TokenVocab
from reference.gnn import spec as gnn_spec
from reference.gnn.batching import BinaryDataset, collate, encode_binary, loader
from reference.gnn.model import HierGNN
from reference.gnn.s05_block_counting.block_vecs import BlockVecs, BlockVecSource, MissingKeys
from reference.cfg_transformer import block_export as be
from reference.cfg_transformer import encoder_contract as ec
from reference.cfg_transformer import precompute_blockvec as pb
from reference.cfg_transformer.encoders import reference as re_

ROOT = Path(__file__).resolve().parents[1]
needs_cache = pytest.mark.real_data
needs_cuda = pytest.mark.gpu


def gnn(**kw):
    return RunConfig(**{'model': 'gnn', 'num_workers': 0, **kw})


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


@needs_cache
def test_schema_documents_every_file_and_the_example_runs(mini):
    text = (ROOT / 'reference' / 'cfg_transformer' / 'EXPORT_SCHEMA.md').read_text(encoding='utf-8')
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
    assert (ckpt['module'], ckpt['class']) == ('reference.cfg_transformer.encoders.reference', 'ReferenceEncoder')
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
    monkeypatch.setattr(gnn_spec, 'CACHE', vectors.cache)
    monkeypatch.setattr(train, 'check_requires', lambda spec, cache: None)         # the cache here holds only the vectors
    monkeypatch.setattr(train, 'make_manifest', lambda cfg, index: mini.manifest)
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


