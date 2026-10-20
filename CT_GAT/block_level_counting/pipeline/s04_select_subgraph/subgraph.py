"""Which whole functions of a binary enter the batch."""
from pathlib import Path

import numpy as np

from pipeline.s03_tokenizer.block_tokens import ranges

MAX_BINARY_BLOCKS = 500_000


def cap_functions(fn_blk_ptr, fn_flags, blk_api_ptr, keep, rng, cap=None):
    """Whole functions of `keep` whose blocks sum to at most `cap` (default MAX_BINARY_BLOCKS), at least one. Unchanged
    when they already fit. Order of preference: evaluation (rng None) takes non-library
    first, then API references, then size, ties by index; training draws the same Efraimidis-Spirakis keys from `rng`.
    The longest prefix of that order that fits is kept; if even the first function is over the cap, the first one in
    order that fits is kept (else the smallest)."""
    cap = MAX_BINARY_BLOCKS if cap is None else cap
    fn_ptr = np.asarray(fn_blk_ptr, dtype=np.int64)
    cand = np.flatnonzero(keep)
    length = (fn_ptr[cand + 1] - fn_ptr[cand])
    if length.sum() <= cap:
        return keep
    api_ptr = np.asarray(blk_api_ptr, dtype=np.int64)
    n_api = api_ptr[fn_ptr[cand + 1]] - api_ptr[fn_ptr[cand]]
    library = (np.asarray(fn_flags)[cand] & 1).astype(np.int64)
    if rng is None:
        order = np.lexsort((np.arange(len(cand)), -length, -n_api, library))
    else:
        weight = (1.0 + 9.0 * (1 - library)) * (1.0 + np.log1p(n_api)) * (1.0 + np.log1p(length))
        order = np.argsort(-(np.log(rng.random(len(cand))) / weight), kind='stable')
    fits = np.cumsum(length[order]) <= cap
    n_keep = int(fits.sum()) if fits.all() else int(np.argmin(fits))
    if n_keep == 0:
        small = np.flatnonzero(length[order] <= cap)
        order = order[[small[0] if len(small) else np.argmin(length[order])]]
        n_keep = 1
    capped = np.zeros(len(keep), dtype=bool)
    capped[cand[order[:n_keep]]] = True
    return capped


def select_subgraph(z, policy, budget, rng, sample_id='', keep=None):
    """(kept_fn_ids, kept_blk_ids), both sorted: whole functions of a binary that fit in `budget` blocks (0 means
    MAX_BINARY_BLOCKS), from the functions in `keep` (default: the non-empty ones). Labels are never read. The only policy
    is 'cap', the `cap_functions` order. rng None is deterministic; an rng is used as `cap_functions` uses it. `sample_id`
    is not used by the current policy; it is there so a policy can seed from it as `control_edges` does."""
    if policy != 'cap':
        raise ValueError(f'subgraph policy {policy}')
    fn_ptr = np.asarray(z['fn_blk_ptr'], dtype=np.int64)
    if keep is None:
        keep = np.diff(fn_ptr) > 0
    kept = cap_functions(fn_ptr, z['fn_flags'], z['blk_api_ptr'], keep, rng, budget or MAX_BINARY_BLOCKS)
    fn = np.flatnonzero(kept)
    return fn, ranges(fn_ptr[fn], fn_ptr[fn + 1] - fn_ptr[fn])


def chain_edges(z):
    """Boolean mask over `cfg_edges` of the edges u -> v whose endpoints could merge into one block: a fallthrough
    (edge_kind 0), out-degree(u) == 1 and in-degree(v) == 1 over all CFG edges, u != v, same function."""
    edges = np.asarray(z['cfg_edges'], dtype=np.int64).reshape(-1, 2)
    n_blk = len(z['blk_feat'])
    out_deg = np.bincount(edges[:, 0], minlength=n_blk)
    in_deg = np.bincount(edges[:, 1], minlength=n_blk)
    fn_ptr = np.asarray(z['fn_blk_ptr'], dtype=np.int64)
    same_fn = np.searchsorted(fn_ptr, edges[:, 0], 'right') == np.searchsorted(fn_ptr, edges[:, 1], 'right')
    return ((np.asarray(z['edge_kind']) == 0) & (out_deg[edges[:, 0]] == 1) & (in_deg[edges[:, 1]] == 1)
            & (edges[:, 0] != edges[:, 1]) & same_fn)


def chain_reduction(rows, cache):
    """Measurement only: over the binaries `rows` (sample_id keys) of the cache directory `cache`, the blocks that a
    chain merge could remove (one per mergeable edge, see `chain_edges`) against all blocks. Returns the counts and the
    reduction fraction pooled over the binaries, and the mean and maximum of the per-binary fraction."""
    n_blocks = n_chain = 0
    per_binary = []
    for r in rows:
        with np.load(Path(cache) / 'samples' / f"{r['sample_id']}.npz") as z:
            if 'edge_kind' not in z.files:
                raise KeyError('sample lacks edge_kind; point the cache at the v2 cache')
            blocks, chain = len(z['blk_feat']), int(chain_edges(z).sum())
        n_blocks, n_chain = n_blocks + blocks, n_chain + chain
        per_binary.append(chain / max(blocks, 1))
    return {'n_binaries': len(per_binary), 'n_blocks': n_blocks, 'n_chain_blocks': n_chain,
            'reduction': n_chain / max(n_blocks, 1), 'mean_binary_reduction': float(np.mean(per_binary)),
            'max_binary_reduction': float(np.max(per_binary))}
