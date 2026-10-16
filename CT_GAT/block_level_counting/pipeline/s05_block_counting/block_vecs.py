"""Precomputed block vectors: one vector per distinct block key of a fold, written by
transformer_paths/precompute_blockvec.py to <cache>/side/blockvec/<export>_<name>/fold{k}/ and read back here, either by
key (`BlockVecs`) or as the vectors of the blocks of one binary (`BlockVecSource`, what `encode_binary_gnn` calls)."""
import json
from pathlib import Path

import numpy as np

from common.config import RUNS
from transformer_paths.reader import BlockStore


class MissingKeys(AssertionError, KeyError):
    """A lookup asked for block keys the vectors do not have."""


def vec_dir(cache, name, fold):
    return Path(cache) / 'side' / 'blockvec' / name / f'fold{fold}'


class BlockVecs:
    """keys.npy (sorted int64), vectors.npy (float16 [n, d], memory-mapped) and meta.json of one fold."""

    def __init__(self, cache, name, fold):
        self.dir = vec_dir(cache, name, fold)
        self.meta = json.loads((self.dir / 'meta.json').read_text(encoding='utf-8'))
        self.keys = np.load(self.dir / 'keys.npy')
        self.vectors = np.load(self.dir / 'vectors.npy', mmap_mode='r')
        if len(self.keys) != len(self.vectors):
            raise ValueError(f'{self.dir}: {len(self.keys)} keys but {len(self.vectors)} vectors')

    @property
    def d(self):
        return int(self.vectors.shape[1])

    def lookup(self, keys):
        """float16 [len(keys), d] vectors of the given block keys; every key must be present."""
        keys = np.asarray(keys, dtype=np.int64)
        pos = np.minimum(np.searchsorted(self.keys, keys), max(len(self.keys) - 1, 0))
        found = self.keys[pos] == keys if len(self.keys) else np.zeros(len(keys), dtype=bool)
        if not found.all():
            raise MissingKeys(f'{int((~found).sum())} of {len(keys)} block keys are not in {self.dir}')
        return np.asarray(self.vectors[pos])


def export_root_of(meta):
    root = Path(meta['export_root'])
    return root if root.exists() else RUNS / 'export' / meta['export']


class BlockVecSource:
    """The vectors of the blocks a binary keeps. The store's `blk_cache_id` and the fold's `block_key` turn a binary's
    kept cache block ids into keys, and the keys into vectors, so a function dropped at training time still
    resolves. Only paths are pickled; the arrays are opened on first use in each process."""

    def __init__(self, cache, name, fold):
        self.cache, self.name, self.fold = str(cache), name, fold
        meta = json.loads((vec_dir(cache, name, fold) / 'meta.json').read_text(encoding='utf-8'))
        self.d, self.export_root = int(meta['d']), str(export_root_of(meta))
        self._vecs = self._store = None

    def _open(self):
        if self._vecs is None:
            self._store = BlockStore(self.export_root)
            self._vecs = BlockVecs(self.cache, self.name, self.fold)
            if self._store.export_sha != self._vecs.meta['export_sha']:
                raise ValueError(f'{self.name} fold {self.fold} was computed for another export than {self.export_root}')
            if self._store.fold(self.fold)['manifest']['manifest_sha'] != self._vecs.meta['manifest_sha']:
                raise ValueError(f'{self.name} fold {self.fold}: the fold manifest changed since the vectors')
        return self._store, self._vecs

    def store_meta(self):
        return self._open()[0].meta

    def block_keys(self, sample_id, blk):
        """block_key of the cache blocks `blk` (sorted cache block ids) of a binary."""
        store, _ = self._open()
        i = store.index_of[sample_id]
        first, last = store.block_range(i)
        cache_ids = np.asarray(store.blk_cache_id[first:last])
        pos = np.searchsorted(cache_ids, blk)
        if len(blk) and (pos.max() >= len(cache_ids) or (cache_ids[pos] != blk).any()):
            raise MissingKeys(f'{sample_id}: blocks outside the export')
        return np.asarray(store.fold(self.fold)['block_key'][first:last])[pos]

    def block_vectors(self, sample_id, blk):
        """float16 [len(blk), d] (a writable copy)."""
        return self._open()[1].lookup(self.block_keys(sample_id, blk)).copy()

    def __getstate__(self):
        return {**self.__dict__, '_vecs': None, '_store': None}
