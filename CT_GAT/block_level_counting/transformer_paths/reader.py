"""Read a block export (see EXPORT_SCHEMA.md): BlockStore for the fold-independent arrays, FoldView for one fold's
train / val / test / all binaries and blocks in the fold vocabulary, collate_blocks to pad sequences into tensors.
Arrays are memory-mapped; nothing here needs the CFG cache."""
import json
from pathlib import Path

import numpy as np
import torch

STORE_ARRAYS = ('bin_blk_ptr', 'bin_fn_ptr', 'bin_cfg_ptr', 'bin_call_ptr', 'blk_tok_ptr', 'tokens', 'tok_kind',
                'blk_fn', 'blk_cache_id', 'blk_feat', 'blk_truncated', 'fn_flags', 'fn_cache_id', 'cfg_edges',
                'edge_type', 'call_edges', 'bin_feat')
SPLITS = ('train', 'val', 'test', 'all')


class BlockStore:
    """The store/ directory of an export plus the fold{k}/ directories next to it. Every store array is a lazy
    attribute of the same name (see STORE_ARRAYS); `meta` is store.json and `binaries` is binaries.json."""

    def __init__(self, root):
        self.root = Path(root)
        self.dir = self.root / 'store'
        if not (self.dir / 'store.json').exists():
            raise FileNotFoundError(f'{self.dir / "store.json"}: not a block export')
        self.meta = json.loads((self.dir / 'store.json').read_text(encoding='utf-8'))
        self.binaries = json.loads((self.dir / 'binaries.json').read_text(encoding='utf-8'))
        self.index_of = {b['sample_id']: b['index'] for b in self.binaries}
        self._folds = {}

    def __getattr__(self, name):
        if name in STORE_ARRAYS:
            array = np.load(self.dir / f'{name}.npy', mmap_mode='r')
            setattr(self, name, array)
            return array
        raise AttributeError(name)

    @property
    def export_sha(self):
        return self.meta['export_sha']

    def fold_dir(self, fold):
        return self.root / f'fold{fold}'

    def fold(self, fold):
        """{'manifest', 'vocab', 'lut', 'block_key'} of a fold, loaded once."""
        if fold not in self._folds:
            d = self.fold_dir(fold)
            self._folds[fold] = {
                'manifest': json.loads((d / 'manifest.json').read_text(encoding='utf-8')),
                'vocab': json.loads((d / 'vocab.json').read_text(encoding='utf-8')),
                'lut': np.load(d / 'lut.npy'), 'block_key': np.load(d / 'block_key.npy', mmap_mode='r')}
        return self._folds[fold]

    def block_range(self, binary):
        """(first, last + 1) store block ids of a binary."""
        return int(self.bin_blk_ptr[binary]), int(self.bin_blk_ptr[binary + 1])

    def block_ids(self, binaries):
        """Store block ids of the given binaries, in the order given."""
        ptr = np.asarray(self.bin_blk_ptr)
        parts = [np.arange(ptr[b], ptr[b + 1]) for b in binaries]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    def tokens_of(self, block):
        """Global vocabulary ids of one block (int32 view)."""
        return self.tokens[int(self.blk_tok_ptr[block]):int(self.blk_tok_ptr[block + 1])]

    def kinds_of(self, block):
        """Kind of each token of one block: 0 mn, 1 sig, 2 beh, 3 api (int8 view)."""
        return self.tok_kind[int(self.blk_tok_ptr[block]):int(self.blk_tok_ptr[block + 1])]

    def cfg_of(self, binary):
        """(edges [E, 2] of store block ids, edge_type [E]) of one binary."""
        a, b = int(self.bin_cfg_ptr[binary]), int(self.bin_cfg_ptr[binary + 1])
        return np.asarray(self.cfg_edges[a:b]), np.asarray(self.edge_type[a:b])

    def calls_of(self, binary):
        """Call edges [C, 2] of store function ids of one binary."""
        return np.asarray(self.call_edges[int(self.bin_call_ptr[binary]):int(self.bin_call_ptr[binary + 1])])


class FoldView:
    """One fold and split ('train', 'val', 'test' or 'all') of a BlockStore, in the fold vocabulary."""

    def __init__(self, store: BlockStore, fold, split):
        if split not in SPLITS:
            raise ValueError(f'split {split}: one of {SPLITS}')
        self.store, self.fold, self.split = store, fold, split
        self.data = store.fold(fold)
        m = self.data['manifest']
        self._binaries = (np.arange(len(store.binaries)) if split == 'all' else np.asarray(m[split], dtype=np.int64))

    @property
    def lut(self):
        return self.data['lut']

    def binaries(self):
        """Store indices of the split's binaries (int64), in manifest order."""
        return self._binaries

    def sample_ids(self):
        return [self.store.binaries[i]['sample_id'] for i in self._binaries]

    def blocks(self):
        """Store block ids of every block of the split's binaries (int64), binary by binary."""
        return self.store.block_ids(self._binaries)

    def keys(self, blocks=None):
        """block_key of the given store block ids (default: all the split's blocks)."""
        return np.asarray(self.data['block_key'][self.blocks() if blocks is None else blocks])

    def sequence(self, block, with_kinds=False):
        """Fold-vocabulary ids of one block (int32); with_kinds also returns the token kinds."""
        seq = self.lut[self.store.tokens_of(block)]
        return (seq, np.asarray(self.store.kinds_of(block))) if with_kinds else seq

    def distinct_blocks(self):
        """(block_ids, keys): one store block id (the first in store order) per distinct block_key among the split's
        blocks, ordered by key. Two blocks with the same key have the same fold-id sequence."""
        blocks = self.blocks()
        keys, first = np.unique(self.keys(blocks), return_index=True)
        return blocks[first], keys

    def padded(self, blocks, max_len=None):
        """(tokens [B, L] int64 fold ids with 0 for padding, mask [B, L] bool) of store block ids, without a Python
        loop over the tokens. A block longer than max_len keeps its first max_len tokens."""
        blocks = np.asarray(blocks, dtype=np.int64)
        ptr = self.store.blk_tok_ptr
        start = np.asarray(ptr[blocks])
        length = np.asarray(ptr[blocks + 1]) - start
        if max_len is not None:
            length = np.minimum(length, max_len)
        width = int(length.max()) if len(length) else 0
        mask = np.arange(width)[None, :] < length[:, None]
        at = np.where(mask, start[:, None] + np.arange(width)[None, :], 0)
        tokens = np.where(mask, self.lut[np.asarray(self.store.tokens[at.reshape(-1)]).reshape(at.shape)], 0)
        return torch.from_numpy(tokens.astype(np.int64)), torch.from_numpy(mask)


def collate_blocks(seqs, pad=0, max_len=None):
    """Pad 1-D integer sequences into (LongTensor [B, L], BoolTensor [B, L]); the mask is True on tokens and False on
    padding. L is the longest sequence, or max_len when that is shorter; longer sequences keep their first max_len
    tokens."""
    seqs = [np.asarray(s) for s in seqs]
    width = max((len(s) for s in seqs), default=0)
    if max_len is not None:
        width = min(width, max_len)
    tokens = np.full((len(seqs), width), pad, dtype=np.int64)
    mask = np.zeros((len(seqs), width), dtype=bool)
    for i, s in enumerate(seqs):
        n = min(len(s), width)
        tokens[i, :n], mask[i, :n] = s[:n], True
    return torch.from_numpy(tokens), torch.from_numpy(mask)
