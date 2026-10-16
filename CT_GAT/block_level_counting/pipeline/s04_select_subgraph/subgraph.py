"""Which functions (or capped segments) of a binary enter the batch."""
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pipeline.s03_tokenizer.block_tokens import ranges

MAX_BINARY_BLOCKS = 500_000


def segment_table(z, max_blocks):
    """Address-ordered segments of at most max_blocks blocks per function; empty functions have none."""
    fn_ptr = z['fn_blk_ptr'].astype(np.int64)
    sizes = np.diff(fn_ptr)
    per_fn = -(-sizes // max_blocks)
    fn = np.repeat(np.arange(len(sizes)), per_fn)
    k = np.arange(len(fn)) - np.repeat(np.cumsum(per_fn) - per_fn, per_fn)
    start = fn_ptr[fn] + k * max_blocks
    length = np.minimum(max_blocks, fn_ptr[fn + 1] - start)
    api_ptr = z['blk_api_ptr'].astype(np.int64)
    return fn, start, length, api_ptr[start + length] - api_ptr[start], (z['fn_flags'][fn] & 1).astype(np.int64)


def select_segments(length, n_api, library, limit, rng):
    """All segments if they fit. Otherwise non-library first, then API references, then size; training
    draws a weighted random subset (Efraimidis-Spirakis keys) with weights that follow the same order."""
    n = len(length)
    if n <= limit:
        return np.arange(n)
    if rng is None:
        order = np.lexsort((np.arange(n), -length, -n_api, library))
        return np.sort(order[:limit])
    weight = (1.0 + 9.0 * (1 - library)) * (1.0 + np.log1p(n_api)) * (1.0 + np.log1p(length))
    keys = np.log(rng.random(n)) / weight
    return np.sort(np.argpartition(-keys, limit - 1)[:limit])


def cap_functions(fn_blk_ptr, fn_flags, blk_api_ptr, keep, rng, cap=None):
    """Whole functions of `keep` whose blocks sum to at most `cap` (default MAX_BINARY_BLOCKS), at least one. Unchanged
    when they already fit. Order of preference as in `select_segments`: evaluation (rng None) takes non-library
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


def lcc_functions(z, fn_blk_ptr, keep, rng, budget):
    """Mask of the `keep` functions chosen by the 'lcc' policy. The weakly connected components of the call graph among
    the `keep` functions are ordered by descending block count (ties: lowest function index) and taken whole while they fit
    in `budget`; the first one that does not fit contributes the longest prefix of its `cap_functions` order that fits the
    remaining budget, and nothing follows it. Only when nothing at all was kept is one function forced, as in
    `cap_functions`."""
    fn_ptr = np.asarray(fn_blk_ptr, dtype=np.int64)
    cand = np.flatnonzero(keep)
    length = fn_ptr[cand + 1] - fn_ptr[cand]
    if length.sum() <= budget:
        return keep
    local = np.full(len(keep), -1, dtype=np.int64)
    local[cand] = np.arange(len(cand))
    call = np.asarray(z['call_edges'], dtype=np.int64).reshape(-1, 2)
    src, dst = local[call[:, 0]], local[call[:, 1]]
    ok = (src >= 0) & (dst >= 0)
    graph = coo_matrix((np.ones(ok.sum()), (src[ok], dst[ok])), shape=(len(cand), len(cand)))
    n_comp, comp = connected_components(graph, directed=False)
    comp_blocks = np.bincount(comp, weights=length, minlength=n_comp).astype(np.int64)
    first_fn = np.full(n_comp, len(cand), dtype=np.int64)
    np.minimum.at(first_fn, comp, np.arange(len(cand)))
    kept, used = np.zeros(len(keep), dtype=bool), 0
    for c in np.lexsort((first_fn, -comp_blocks)):
        members = cand[comp == c]
        if used + comp_blocks[c] <= budget:
            kept[members] = True
            used += comp_blocks[c]
            continue
        room = np.zeros(len(keep), dtype=bool)
        room[members] = True
        part = cap_functions(fn_ptr, z['fn_flags'], z['blk_api_ptr'], room, rng, budget - used)
        size = int((fn_ptr[np.flatnonzero(part) + 1] - fn_ptr[np.flatnonzero(part)]).sum())
        if used + size <= budget or not kept.any():
            kept |= part
        break
    return kept


def select_subgraph(z, policy, budget, rng, sample_id='', keep=None):
    """(kept_fn_ids, kept_blk_ids), both sorted: whole functions of a binary that fit in `budget` blocks (0 means
    MAX_BINARY_BLOCKS), from the functions in `keep` (default: the non-empty ones). Labels are never read. Policies:
    'cap' takes `cap_functions` order; 'lcc' takes call-graph components, largest first (`lcc_functions`). rng None is
    deterministic; an rng is used as `cap_functions` uses it. `sample_id` is not used by the current policies; it is
    there so a policy can seed from it as `control_edges` does."""
    if policy not in ('cap', 'lcc'):
        raise ValueError(f'subgraph policy {policy}')
    fn_ptr = np.asarray(z['fn_blk_ptr'], dtype=np.int64)
    if keep is None:
        keep = np.diff(fn_ptr) > 0
    budget = budget or MAX_BINARY_BLOCKS
    if policy == 'cap':
        kept = cap_functions(fn_ptr, z['fn_flags'], z['blk_api_ptr'], keep, rng, budget)
    else:
        kept = lcc_functions(z, fn_ptr, keep, rng, budget)
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
