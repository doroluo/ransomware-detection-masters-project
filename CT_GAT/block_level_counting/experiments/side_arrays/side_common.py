"""Deterministic controls shared by static side arrays and the CPU WL baseline."""
import hashlib

import numpy as np


def sample_rng(sample_id, seed=0):
    digest = hashlib.sha256(f'{seed}:{sample_id}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'little'))


def shuffle_edges(edges, fn_blk_ptr, rng, kinds=None):
    """Directed configuration-model control, within function and (optionally) kind.

    Keeps each node's in/out degree, edge count and kind-specific degrees exactly.
    Parallel edges and self loops are allowed and must NOT be deduplicated afterward.
    This matches the existing GNN's destination-permutation null. Tiny graphs can
    remain unchanged; callers record the fraction of changed endpoint pairs.
    """
    out = np.asarray(edges, dtype=np.int32).reshape(-1, 2).copy()
    if not len(out):
        return out
    ptr = np.asarray(fn_blk_ptr)
    if out.min() < 0 or out.max() >= ptr[-1]:
        raise ValueError('edge outside block range')
    src_fn = np.searchsorted(ptr[1:], out[:, 0], side='right')
    dst_fn = np.searchsorted(ptr[1:], out[:, 1], side='right')
    if not np.array_equal(src_fn, dst_fn):
        raise ValueError('inter-function edge in intra-function control')
    kinds = np.zeros(len(out), dtype=np.int8) if kinds is None else np.asarray(kinds)
    if kinds.shape != (len(out),):
        raise ValueError('edge kind alignment mismatch')
    order = np.lexsort((kinds, src_fn))
    boundaries = np.flatnonzero((np.diff(src_fn[order]) != 0) | (np.diff(kinds[order]) != 0)) + 1
    for group in np.split(order, boundaries):
        out[group, 1] = rng.permutation(out[group, 1])
    return out


def tag_control(tags, mode, rng=None):
    """Apply to FULL cache-order function rows, before function dropping/capping."""
    tags = np.asarray(tags)
    if mode == 'off':
        return np.zeros_like(tags)
    if mode == 'real':
        return tags.copy()
    if mode == 'permute' and rng is not None:
        return tags[rng.permutation(len(tags))].copy()
    raise ValueError('tag control must be real, off, or permute (with RNG)')
