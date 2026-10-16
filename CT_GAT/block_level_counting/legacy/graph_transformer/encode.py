"""Per-binary arrays for the legacy graph transformer: address-ordered function segments with no edge types."""
import numpy as np

from pipeline.s03_tokenizer.block_tokens import ranges
from pipeline.s04_select_subgraph.subgraph import segment_table, select_segments


def encode_binary_transformer(z, vocab, cfg, rng):
    fn, start, length, n_api, library = segment_table(z, cfg.max_segment_blocks)
    n_segments_total = len(length)
    sel = select_segments(length, n_api, library, cfg.max_segments, rng)
    fn, start, length, library = fn[sel], start[sel], length[sel], library[sel]

    blk = ranges(start, length)
    n_blk = len(blk)
    blk_seg = np.repeat(np.arange(len(sel)), length)
    blk_pos = np.arange(n_blk) - np.repeat(np.cumsum(length) - length, length)

    insn_ptr, api_ptr = z['blk_insn_ptr'].astype(np.int64), z['blk_api_ptr'].astype(np.int64)
    n_ins, n_apis = insn_ptr[blk + 1] - insn_ptr[blk], api_ptr[blk + 1] - api_ptr[blk]
    ins, apis = ranges(insn_ptr[blk], n_ins), ranges(api_ptr[blk], n_apis)
    per_block = 3 * n_ins + n_apis
    tok_offsets = np.cumsum(per_block) - per_block
    ins_rank = np.arange(len(ins)) - np.repeat(np.cumsum(n_ins) - n_ins, n_ins)
    ins_base = np.repeat(tok_offsets, n_ins) + ins_rank
    ins_len = np.repeat(n_ins, n_ins)
    api_rank = np.arange(len(apis)) - np.repeat(np.cumsum(n_apis) - n_apis, n_apis)
    tokens = np.empty(int(per_block.sum()), dtype=np.int64)
    for j, k in enumerate(('mn', 'sig', 'beh')):
        tokens[ins_base + j * ins_len] = vocab.lut[k][z[k][ins]] + vocab.offset[k]
    tokens[np.repeat(tok_offsets + 3 * n_ins, n_apis) + api_rank] = vocab.lut['api'][z['api'][apis]] + vocab.offset['api']

    n_total_blocks = len(z['blk_feat'])
    e = z['cfg_edges'].astype(np.int64)
    seg_of = np.full(n_total_blocks, -1, dtype=np.int64)
    pos_of = np.zeros(n_total_blocks, dtype=np.int64)
    seg_of[blk], pos_of[blk] = blk_seg, blk_pos
    s, t = seg_of[e[:, 0]], seg_of[e[:, 1]]
    keep = (s >= 0) & (s == t)
    clip = cfg.degree_clip
    deg_in = np.minimum(np.bincount(e[:, 1], minlength=n_total_blocks)[blk], clip)
    deg_out = np.minimum(np.bincount(e[:, 0], minlength=n_total_blocks)[blk], clip)

    fn_used, seg_fn = np.unique(fn, return_inverse=True)
    fn_local = np.full(len(z['fn_flags']), -1, dtype=np.int64)
    fn_local[fn_used] = np.arange(len(fn_used))
    c = z['call_edges'].astype(np.int64).reshape(-1, 2)
    cs, ct = fn_local[c[:, 0]], fn_local[c[:, 1]]
    ckeep = (cs >= 0) & (ct >= 0) & (cs != ct)

    return {
        'tokens': tokens, 'tok_offsets': tok_offsets, 'blk_feat': z['blk_feat'][blk].astype(np.float32),
        'deg_in': deg_in, 'deg_out': deg_out, 'blk_seg': blk_seg, 'blk_pos': blk_pos,
        'seg_len': length, 'seg_feat': np.stack([np.log1p(length), library], 1).astype(np.float32),
        'seg_fn': seg_fn, 'n_fn': len(fn_used),
        'edge_seg': s[keep], 'edge_src': pos_of[e[keep, 0]], 'edge_dst': pos_of[e[keep, 1]],
        'call_src': cs[ckeep], 'call_dst': ct[ckeep],
        'n_segments_total': n_segments_total,
    }
