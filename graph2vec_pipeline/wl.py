#!/usr/bin/env python3
"""Vectorised Weisfeiler-Lehman over a whole cohort at once.

`cfg.wl_labels` walks one graph in Python and hashes a sorted successor tuple
per node. That is fine for 5,000-block graphs and one hyper-parameter; it is
not fine for a search over block caps, label granularities and h=1..5 on 20M
nodes. This module does the same relabelling with numpy on the concatenated
node/edge arrays of `graph_cache.GraphSet`:

  * a node's context is aggregated with an order-independent **sum of mixed
    neighbour hashes**, kept separately for successors and predecessors, plus
    the two degrees. That is the hashed-WL variant: same feature map up to
    64-bit collisions, but it is four `np.bincount` calls per iteration instead
    of a Python loop with a sort per node.
  * edge kinds (fall-through / branch / call / extern) can be salted into the
    neighbour hash, so "keep the fall-through/branch distinction as edge
    labels" is a flag rather than a different code path.
  * per-layer document matrices are built with the hashing trick into a 2**28
    column space. The retained vocabulary after a min_df filter is ~1e5 wide,
    so at 2.7e8 buckets an accidental collision with the singleton mass costs
    well under a tenth of a spurious count per retained column.

Everything here is deterministic: no RNG, no dict iteration order, no
`hash()`.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from graph2vec_pipeline.cfg import (E_BRANCH, E_BRANCH_XSEC, E_CALL,
                                    E_CALL_XSEC, E_EXTERN, E_FALL)

HASH_BITS = 28
HASH_SIZE = 1 << HASH_BITS

# distinct salts per edge kind, used only when edge_labels=True
_ETYPE_SALT = np.array([
    0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9,
    0x27D4EB2F165667C5, 0x85EBCA77C2B2AE63, 0xD6E8FEB86659FD93,
], dtype=np.uint64)

ALL_EDGES = (E_FALL, E_BRANCH, E_BRANCH_XSEC, E_CALL, E_CALL_XSEC, E_EXTERN)


def _mix64(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser; uint64 arithmetic wraps, which is what we want."""
    x = x.astype(np.uint64, copy=True)
    with np.errstate(over="ignore"):
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return x


def _ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Concatenated arange(start, start+count) for each (start, count)."""
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return np.repeat(starts - off, counts) + np.arange(total, dtype=np.int64)


class Batch:
    """A filtered, concatenated view of many graphs.

    gid      (int32)  graph number per node
    dom/term/szb/flg/blen/seq   per node, as cached
    src/dst  (int64)  batch-global node indices
    etype    (uint8)
    """

    __slots__ = ("n_graphs", "n_nodes", "gid", "dom", "term", "szb", "flg",
                 "blen", "seq", "is_extern", "src", "dst", "etype",
                 "outdeg", "indeg", "node_off")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def assemble(gs, rows, max_blocks: int | None = None,
             keep_edges=ALL_EDGES, drop_isolated_extern: bool = True) -> Batch:
    """Build a Batch from cached graphs.

    max_blocks   keep only the first K basic blocks of each graph (the cache is
                 built at the widest cap; a tighter cap is a prefix of it, as
                 in build_cfg, because leaders are in address order).
    keep_edges   which cfg.E_* kinds survive. Dropping E_CALL/E_CALL_XSEC is
                 `include_calls=False`; dropping the *_XSEC kinds is the old
                 same-section-only target rule; dropping E_EXTERN removes the
                 import-thunk nodes' edges.
    """
    rows = np.asarray(rows, dtype=np.int64)
    nstart, nend = gs.node_ptr[rows], gs.node_ptr[rows + 1]
    estart, eend = gs.edge_ptr[rows], gs.edge_ptr[rows + 1]
    ncnt, ecnt = nend - nstart, eend - estart

    nidx = _ranges(nstart, ncnt)
    gid = np.repeat(np.arange(len(rows), dtype=np.int32), ncnt)
    local = nidx - np.repeat(nstart, ncnt)
    nb = gs.n_blocks[rows]
    is_ext = local >= np.repeat(nb, ncnt)

    cap = np.minimum(nb, max_blocks) if max_blocks else nb
    keep = (local < np.repeat(cap, ncnt)) | is_ext

    # edges, in batch-global (pre-filter) node ids
    eidx = _ranges(estart, ecnt)
    noff = np.concatenate([[0], np.cumsum(ncnt)[:-1]]).astype(np.int64)
    base = np.repeat(noff, ecnt)
    src = base + gs.esrc[eidx].astype(np.int64)
    dst = base + gs.edst[eidx].astype(np.int64)
    et = gs.etype[eidx]

    emask = np.isin(et, np.asarray(keep_edges, dtype=np.uint8))
    emask &= keep[src] & keep[dst]
    src, dst, et = src[emask], dst[emask], et[emask]

    if drop_isolated_extern and len(src):
        touched = np.zeros(len(keep), dtype=bool)
        touched[src] = True
        touched[dst] = True
        keep &= ~is_ext | touched
    elif drop_isolated_extern:
        keep &= ~is_ext

    newid = np.cumsum(keep) - 1
    src, dst = newid[src], newid[dst]
    sel = np.flatnonzero(keep)
    take = nidx[sel]

    n_nodes = len(sel)
    outdeg = np.bincount(src, minlength=n_nodes).astype(np.int32)
    indeg = np.bincount(dst, minlength=n_nodes).astype(np.int32)
    return Batch(n_graphs=len(rows), n_nodes=n_nodes,
                 gid=gid[sel], dom=gs.dom[take], term=gs.term[take],
                 szb=gs.szb[take], flg=gs.flg[take], blen=gs.blen[take],
                 seq=gs.seq[take], is_extern=is_ext[sel],
                 src=src, dst=dst, etype=et, outdeg=outdeg, indeg=indeg,
                 node_off=None)


# ---------------------------------------------------------------------------
# node labelling granularity
# ---------------------------------------------------------------------------
LABELLINGS = ("class", "class_noflag", "dom_term", "seq", "seq_term", "deg",
              "deg_dom", "lenbucket")


def _deg_bucket(d: np.ndarray) -> np.ndarray:
    b = np.zeros(len(d), dtype=np.uint64)
    b[d >= 1] = 1
    b[d >= 2] = 2
    b[d >= 3] = 3
    b[d >= 5] = 4
    b[d >= 9] = 5
    b[d >= 17] = 6
    return b


def base_labels(b: Batch, labelling: str = "class") -> np.ndarray:
    """uint64 depth-0 label per node."""
    dom = b.dom.astype(np.uint64)
    term = b.term.astype(np.uint64)
    szb = b.szb.astype(np.uint64)
    flg = b.flg.astype(np.uint64)
    if labelling == "class":                 # what the current pipeline uses
        k = dom | (term << np.uint64(8)) | (szb << np.uint64(16)) \
            | (flg << np.uint64(24))
    elif labelling == "class_noflag":
        k = dom | (term << np.uint64(8)) | (szb << np.uint64(16))
    elif labelling == "dom_term":
        k = dom | (term << np.uint64(8))
    elif labelling == "lenbucket":
        k = szb | (term << np.uint64(8))
    elif labelling == "seq":                 # hashed mnemonic-class sequence
        k = b.seq.astype(np.uint64)
    elif labelling == "seq_term":
        k = b.seq.astype(np.uint64) | (term << np.uint64(32)) \
            | (flg << np.uint64(40))
    elif labelling == "deg":                 # (in-deg, out-deg, dom)
        k = (_deg_bucket(b.indeg) | (_deg_bucket(b.outdeg) << np.uint64(8))
             | (dom << np.uint64(16)))
    elif labelling == "deg_dom":             # (in-deg, out-deg, dom, term)
        k = (_deg_bucket(b.indeg) | (_deg_bucket(b.outdeg) << np.uint64(8))
             | (dom << np.uint64(16)) | (term << np.uint64(24)))
    else:
        raise ValueError(f"unknown labelling {labelling!r}")
    # depth tag, so layer 0 cannot collide with a deeper layer's labels
    return _mix64(k ^ np.uint64(0xD0))


def _aggregate(target: np.ndarray, vals: np.ndarray, n: int) -> np.ndarray:
    """Exact int64 sum of 64-bit values grouped by `target`.

    The sum is done in two 32-bit halves so float64 accumulation stays exact:
    each half is < 2**32 and no node has 2**20 neighbours, so a partial sum
    never reaches 2**53.
    """
    lo = np.bincount(target, weights=(vals & np.uint64(0xFFFFFFFF)).astype(
        np.float64), minlength=n)
    hi = np.bincount(target, weights=(vals >> np.uint64(32)).astype(
        np.float64), minlength=n)
    with np.errstate(over="ignore"):
        return (lo.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
                + hi.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F))


def wl_layers(b: Batch, h: int, labelling: str = "class",
              edge_labels: bool = False) -> list[np.ndarray]:
    """Depth-0..h node labels. Returns h+1 uint64 arrays of length n_nodes."""
    cur = base_labels(b, labelling)
    out = [cur]
    if h <= 0 or not len(b.src):
        return out + [cur.copy() for _ in range(max(0, h))]
    salt_s = (_ETYPE_SALT[b.etype] if edge_labels
              else np.zeros(len(b.etype), dtype=np.uint64))
    n = b.n_nodes
    deg_out = b.outdeg.astype(np.uint64)
    deg_in = b.indeg.astype(np.uint64)
    for d in range(1, h + 1):
        with np.errstate(over="ignore"):
            fwd = _mix64((cur[b.dst] ^ salt_s) + np.uint64(0x51))
            bwd = _mix64((cur[b.src] ^ salt_s) + np.uint64(0xA7))
        s_out = _aggregate(b.src, fwd, n)
        s_in = _aggregate(b.dst, bwd, n)
        with np.errstate(over="ignore"):
            nxt = _mix64(cur * np.uint64(0x100000001B3)
                         ^ (s_out + np.uint64(0x27D4EB2F165667C5))
                         ^ _mix64(s_in)
                         ^ _mix64(deg_out * np.uint64(31) + deg_in
                                  + np.uint64(d) * np.uint64(0x9E3779B1)))
        cur = nxt
        out.append(cur)
    return out


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------
def layer_matrix(gid: np.ndarray, lab: np.ndarray, n_graphs: int,
                 train_mask: np.ndarray | None = None, min_df: int = 3):
    """(graph x hashed-label) count matrix, columns pruned by train df."""
    cols = (lab & np.uint64(HASH_SIZE - 1)).astype(np.int32)
    M = sparse.coo_matrix(
        (np.ones(len(cols), dtype=np.float32), (gid.astype(np.int32), cols)),
        shape=(n_graphs, HASH_SIZE)).tocsr()
    M.sum_duplicates()
    if min_df <= 0:
        return M
    ref = M if train_mask is None else M[train_mask]
    uniq, cnt = np.unique(ref.indices, return_counts=True)
    keep = uniq[cnt >= min_df]
    if not len(keep):
        return sparse.csr_matrix((n_graphs, 0), dtype=np.float32)
    # column restriction done by hand: CSR fancy column indexing on a 2**28
    # wide matrix is the slowest thing in the search otherwise.
    pos = np.clip(np.searchsorted(keep, M.indices), 0, len(keep) - 1)
    ok = keep[pos] == M.indices
    rowid = np.repeat(np.arange(n_graphs, dtype=np.int32),
                      np.diff(M.indptr))
    return sparse.coo_matrix(
        (M.data[ok], (rowid[ok], pos[ok])),
        shape=(n_graphs, len(keep))).tocsr()


def wl_documents(b: Batch, h: int, labelling: str = "class",
                 edge_labels: bool = False, train_mask=None, min_df: int = 3):
    """Horizontally stacked per-depth count matrices for depths 0..h."""
    mats = [layer_matrix(b.gid, lab, b.n_graphs, train_mask, min_df)
            for lab in wl_layers(b, h, labelling, edge_labels)]
    return sparse.hstack([m for m in mats if m.shape[1]], format="csr")


# ---------------------------------------------------------------------------
# graph-level scalar features (the size_only control, recomputed per batch so
# it always matches the construction under test)
# ---------------------------------------------------------------------------
SIZE_FEATURES = ("n_nodes", "n_edges", "n_blocks", "n_extern", "insns",
                 "mean_block_len", "max_block_len", "edges_per_node",
                 "back_edges", "self_loops", "call_edges", "extern_edges",
                 "fall_edges", "branch_edges", "mean_outdeg", "max_outdeg",
                 "frac_outdeg0", "frac_outdeg2plus", "mean_indeg",
                 "max_indeg", "frac_indeg0", "n_leaves")


def size_features(b: Batch) -> np.ndarray:
    """One row of SIZE_FEATURES per graph."""
    g = b.gid.astype(np.int64)
    n = b.n_graphs
    nodes = np.bincount(g, minlength=n).astype(np.float64)
    nz = np.maximum(nodes, 1.0)
    egid = b.gid[b.src].astype(np.int64)
    edges = np.bincount(egid, minlength=n).astype(np.float64)
    ext = np.bincount(g, weights=b.is_extern.astype(np.float64), minlength=n)
    blen = b.blen.astype(np.float64)
    insns = np.bincount(g, weights=blen, minlength=n)
    maxlen = np.zeros(n)
    np.maximum.at(maxlen, g, blen)
    back = np.bincount(egid, weights=(b.dst < b.src).astype(np.float64),
                       minlength=n)
    loops = np.bincount(egid, weights=(b.dst == b.src).astype(np.float64),
                        minlength=n)

    def _et(kinds):
        m = np.isin(b.etype, np.asarray(kinds, dtype=np.uint8))
        return np.bincount(egid, weights=m.astype(np.float64), minlength=n)

    od = b.outdeg.astype(np.float64)
    idg = b.indeg.astype(np.float64)
    maxod = np.zeros(n)
    np.maximum.at(maxod, g, od)
    maxid = np.zeros(n)
    np.maximum.at(maxid, g, idg)
    cols = [
        nodes, edges, nodes - ext, ext, insns,
        insns / nz, maxlen, edges / nz,
        back, loops, _et((E_CALL, E_CALL_XSEC)), _et((E_EXTERN,)),
        _et((E_FALL,)), _et((E_BRANCH, E_BRANCH_XSEC)),
        np.bincount(g, weights=od, minlength=n) / nz, maxod,
        np.bincount(g, weights=(od == 0).astype(np.float64), minlength=n) / nz,
        np.bincount(g, weights=(od >= 2).astype(np.float64), minlength=n) / nz,
        np.bincount(g, weights=idg, minlength=n) / nz, maxid,
        np.bincount(g, weights=(idg == 0).astype(np.float64), minlength=n) / nz,
        np.bincount(g, weights=((od == 0) & (idg > 0)).astype(np.float64),
                    minlength=n),
    ]
    return np.vstack(cols).T.astype(np.float64)
