"""common/reader.py on a block export written by hand (the exporter, and the real-export tests, move with S5-4):
BlockStore's lazy arrays and ranges, FoldView's splits, the fold vocabulary and padding, collate_blocks."""
import json

import numpy as np
import pytest
import torch

from common.reader import STORE_ARRAYS, BlockStore, FoldView, collate_blocks

TOKENS = [[1, 2, 3], [4, 5], [1, 2, 3], [6], [7, 8]]               # global ids of five blocks
BLOCKS_OF = [2, 1, 2]                                             # blocks per binary
KINDS = [[0, 1, 2], [0, 3], [0, 1, 2], [0], [0, 3]]


@pytest.fixture
def export(tmp_path):
    store = tmp_path / 'store'
    store.mkdir()
    np.save(store / 'tokens.npy', np.array([t for b in TOKENS for t in b], dtype=np.int32))
    np.save(store / 'tok_kind.npy', np.array([k for b in KINDS for k in b], dtype=np.int8))
    np.save(store / 'blk_tok_ptr.npy', np.concatenate([[0], np.cumsum([len(b) for b in TOKENS])]).astype(np.int64))
    np.save(store / 'bin_blk_ptr.npy', np.concatenate([[0], np.cumsum(BLOCKS_OF)]).astype(np.int64))
    np.save(store / 'bin_cfg_ptr.npy', np.array([0, 1, 1, 3], dtype=np.int64))
    np.save(store / 'cfg_edges.npy', np.array([[0, 1], [3, 4], [4, 3]], dtype=np.int32))
    np.save(store / 'edge_type.npy', np.array([0, 1, 2], dtype=np.int8))
    np.save(store / 'bin_call_ptr.npy', np.array([0, 0, 0, 1], dtype=np.int64))
    np.save(store / 'call_edges.npy', np.array([[0, 1]], dtype=np.int32))
    (store / 'store.json').write_text(json.dumps({'export_sha': 'abc'}), encoding='utf-8')
    (store / 'binaries.json').write_text(json.dumps([{'sample_id': f's{i}', 'index': i} for i in range(3)]), encoding='utf-8')
    fold = tmp_path / 'fold0'
    fold.mkdir()
    (fold / 'manifest.json').write_text(json.dumps({'train': [0, 2], 'val': [], 'test': [1]}), encoding='utf-8')
    (fold / 'vocab.json').write_text('{}', encoding='utf-8')
    lut = np.zeros(9, dtype=np.int32)
    lut[[1, 2, 3, 4, 5, 6]] = [11, 12, 13, 14, 15, 16]                   # global ids 7 and 8 are unknown to the fold: 0
    np.save(fold / 'lut.npy', lut)
    np.save(fold / 'block_key.npy', np.array([5, 6, 5, 7, 8], dtype=np.int64))      # blocks 0 and 2 hold the same sequence
    return tmp_path


def test_a_directory_without_a_store_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match='not a block export'):
        BlockStore(tmp_path)


def test_store_metadata_ranges_and_lazy_arrays(export):
    store = BlockStore(export)
    assert store.export_sha == 'abc' and store.index_of == {'s0': 0, 's1': 1, 's2': 2}
    assert 'tokens' in STORE_ARRAYS and 'tokens' not in vars(store)                  # loaded on first access
    assert store.tokens.shape == (11,) and 'tokens' in vars(store)
    with pytest.raises(AttributeError):
        store.not_an_array
    assert [store.block_range(b) for b in range(3)] == [(0, 2), (2, 3), (3, 5)]
    assert store.block_ids([2, 0]).tolist() == [3, 4, 0, 1] and store.block_ids([]).tolist() == []
    assert [store.tokens_of(b).tolist() for b in range(5)] == TOKENS
    assert [store.kinds_of(b).tolist() for b in range(5)] == KINDS
    edges, types = store.cfg_of(2)
    assert edges.tolist() == [[3, 4], [4, 3]] and types.tolist() == [1, 2]
    assert len(store.cfg_of(1)[0]) == 0
    assert store.calls_of(2).tolist() == [[0, 1]] and len(store.calls_of(0)) == 0
    assert store.fold(0) is store.fold(0) and store.fold_dir(0) == export / 'fold0'


def test_fold_view_splits_sequences_and_keys(export):
    store = BlockStore(export)
    train, test, every = FoldView(store, 0, 'train'), FoldView(store, 0, 'test'), FoldView(store, 0, 'all')
    assert train.binaries().tolist() == [0, 2] and test.binaries().tolist() == [1]
    assert every.binaries().tolist() == [0, 1, 2]
    assert train.sample_ids() == ['s0', 's2'] and train.blocks().tolist() == [0, 1, 3, 4]
    assert train.sequence(0).tolist() == [11, 12, 13] and train.sequence(4).tolist() == [0, 0]
    seq, kinds = train.sequence(1, with_kinds=True)
    assert seq.tolist() == [14, 15] and kinds.tolist() == [0, 3]
    assert train.keys().tolist() == [5, 6, 7, 8] and train.keys([2]).tolist() == [5]
    blocks, keys = every.distinct_blocks()
    assert keys.tolist() == [5, 6, 7, 8] and blocks.tolist() == [0, 1, 3, 4]         # the first block of each key
    with pytest.raises(ValueError, match='split'):
        FoldView(store, 0, 'dev')


def test_padded_uses_the_fold_vocabulary_and_clips(export):
    view = FoldView(BlockStore(export), 0, 'all')
    tokens, mask = view.padded([0, 1, 3])
    assert tokens.tolist() == [[11, 12, 13], [14, 15, 0], [16, 0, 0]]
    assert mask.tolist() == [[True, True, True], [True, True, False], [True, False, False]]
    assert tokens.dtype == torch.int64 and mask.dtype == torch.bool
    clipped, cmask = view.padded([0, 1], max_len=2)
    assert clipped.tolist() == [[11, 12], [14, 15]] and cmask.all()
    empty, emask = view.padded([])
    assert empty.shape == (0, 0) and emask.shape == (0, 0)


def test_collate_blocks_pads_and_clips():
    tokens, mask = collate_blocks([[1, 2, 3], [4], []])
    assert tokens.tolist() == [[1, 2, 3], [4, 0, 0], [0, 0, 0]]
    assert mask.tolist() == [[True, True, True], [True, False, False], [False, False, False]]
    tokens, mask = collate_blocks([np.arange(5), np.arange(2)], max_len=3)
    assert tokens.tolist() == [[0, 1, 2], [0, 1, 0]] and mask.tolist() == [[True] * 3, [True, True, False]]
    assert collate_blocks([[7]], pad=9)[0].tolist() == [[7]] and collate_blocks([])[0].shape == (0, 0)
    assert collate_blocks([[1], [2, 3]], pad=-1)[0].tolist() == [[1, -1], [2, 3]]
