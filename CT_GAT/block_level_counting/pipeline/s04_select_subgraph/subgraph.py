"""Which functions (or capped segments) of a binary enter the batch."""
import numpy as np

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
