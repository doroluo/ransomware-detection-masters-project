"""Typed block edges of a function CFG, and the structural controls and edge augmentations applied to them."""
import zlib

import numpy as np

N_EDGE_TYPES = 12


def permute_within(seg, values, rng):
    """`values` shuffled among the entries that share a `seg`."""
    out = np.empty_like(values)
    out[np.argsort(seg, kind='stable')] = values[np.lexsort((rng.random(len(seg)), seg))]
    return out


def block_edges(z, blk, blk_seg, blk_pos, degree_clip):
    """Edges between kept blocks of the same function as (segment, source position, target position, type) with
    type = 2 * edge_kind + edge_back, and the in/out degree of each kept block over all CFG edges, clipped."""
    n_total_blocks = len(z['blk_feat'])
    e = z['cfg_edges'].astype(np.int64)
    seg_of = np.full(n_total_blocks, -1, dtype=np.int64)
    pos_of = np.zeros(n_total_blocks, dtype=np.int64)
    seg_of[blk], pos_of[blk] = blk_seg, blk_pos
    s, t = seg_of[e[:, 0]], seg_of[e[:, 1]]
    keep = (s >= 0) & (s == t)
    deg_in = np.minimum(np.bincount(e[:, 1], minlength=n_total_blocks)[blk], degree_clip)
    deg_out = np.minimum(np.bincount(e[:, 0], minlength=n_total_blocks)[blk], degree_clip)
    edge_seg, edge_src, edge_dst = s[keep], pos_of[e[keep, 0]], pos_of[e[keep, 1]]
    edge_type = z['edge_kind'].astype(np.int64)[keep] * 2 + z['edge_back'].astype(np.int64)[keep]
    return deg_in, deg_out, (edge_seg, edge_src, edge_dst, edge_type)


def control_edges(control, rng, sample_id, edges, blk_seg, blk_pos, length):
    """The structural control: `permute_types` shuffles edge types within each function, `chain` replaces the CFG by
    a chain over the blocks in address order, `shuffle` permutes edge targets within each function. Evaluation
    (rng None) seeds the permutation from the sample id."""
    edge_seg, edge_src, edge_dst, edge_type = edges
    crng = rng if rng is not None else np.random.default_rng(zlib.crc32(sample_id.encode()))
    if control == 'permute_types':
        edge_type = permute_within(edge_seg, edge_type, crng)
    elif control == 'chain':
        first = blk_pos < length[blk_seg] - 1
        edge_seg, edge_src = blk_seg[first], blk_pos[first]
        edge_dst = edge_src + 1
        edge_type = np.zeros(len(edge_seg), dtype=np.int64)
    elif control == 'shuffle':
        edge_dst = permute_within(edge_seg, edge_dst, crng)
        edge_type = np.zeros(len(edge_seg), dtype=np.int64)
    return edge_seg, edge_src, edge_dst, edge_type


def drop_and_flip_edges(edges, drop, flip, rng):
    """Training only: drop each edge with probability `drop`, then reverse each survivor with probability `flip`."""
    edge_seg, edge_src, edge_dst, edge_type = edges
    if rng is not None and drop > 0:
        alive = rng.random(len(edge_seg)) >= drop
        edge_seg, edge_src, edge_dst, edge_type = edge_seg[alive], edge_src[alive], edge_dst[alive], edge_type[alive]
    if rng is not None and flip > 0:
        flipped = rng.random(len(edge_seg)) < flip
        edge_src, edge_dst = np.where(flipped, edge_dst, edge_src), np.where(flipped, edge_src, edge_dst)
    return edge_seg, edge_src, edge_dst, edge_type
