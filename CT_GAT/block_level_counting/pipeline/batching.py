"""Per-binary arrays and batches: function dropping, the stage calls that build one binary's inputs, collate,
the block-budget sampler and the loader. The legacy transformer path is `legacy.graph_transformer.encode`."""
import numpy as np
import torch

from common.config import RunConfig
from common.manifests import load_sample
from legacy.graph_transformer.encode import encode_binary_transformer
from pipeline.s03_tokenizer.block_tokens import ranges, token_table
from pipeline.s03_tokenizer.vocab import TOKEN_KINDS, TokenVocab
from pipeline.s04_select_subgraph import subgraph
from pipeline.s04_select_subgraph.subgraph import cap_functions
from pipeline.s05_block_counting.weights import token_weights
from pipeline.s06_gnn_block_cfg.edges import block_edges, control_edges, drop_and_flip_edges
from pipeline.s08_gnn_call_graph.calls import call_pairs
from pipeline.s11_final_classifier.head import bin_feat


def keep_functions(fn_flags, sizes, cfg, rng):
    """Mask over functions: non-empty ones, minus the training-time drops (at least one function stays).
    `drop_library` replaces `drop_functions` as the rate of FLIRT/thunk functions (`fn_flags & 3`) when set."""
    keep = sizes > 0
    if rng is None or not (cfg.drop_functions or cfg.drop_library):
        return keep
    library_rate = cfg.drop_library or cfg.drop_functions
    u = rng.random(len(sizes))
    drop = (u < np.where((fn_flags & 3) > 0, library_rate, cfg.drop_functions)) & keep
    if keep.any() and not (keep & ~drop).any():
        drop[np.argmax(np.where(keep, u, -1.0))] = False
    return keep & ~drop


def encode_binary_gnn(z, vocab, cfg, rng, sample_id):
    """Batch contract: one segment per kept function, edge types, tf-idf weights, controls, masking and, in training,
CFG edge drop and flip (after the control). A binary over MAX_BINARY_BLOCKS keeps a subset of whole functions
(`cap_functions`, after function dropping); `bin_feat` counts the full binary."""
    if 'edge_kind' not in z:
        raise KeyError('sample lacks edge_kind/edge_back; point config.CACHE at the v2 cache')
    fn_ptr = z['fn_blk_ptr'].astype(np.int64)
    sizes = np.diff(fn_ptr)
    keep = keep_functions(z['fn_flags'], sizes, cfg, rng)
    fn = np.flatnonzero(cap_functions(fn_ptr, z['fn_flags'], z['blk_api_ptr'], keep, rng))
    length, n_seg = sizes[fn], len(fn)
    library = (z['fn_flags'][fn] & 1).astype(np.int64)

    blk = ranges(fn_ptr[fn], length)
    n_blk = len(blk)
    blk_seg = np.repeat(np.arange(n_seg), length)
    blk_pos = np.arange(n_blk) - np.repeat(np.cumsum(length) - length, length)

    tokens, kind, tok_offsets, per_block, ins_pos, api_pos = token_table(z, vocab, blk)
    blk_feat = z['blk_feat'][blk].astype(np.float32)
    mask_of_kind = np.array([vocab.mask_id[k] for k in TOKEN_KINDS])
    if rng is not None and cfg.mask_tokens > 0:
        hit = rng.random(len(ins_pos[0])) < cfg.mask_tokens
        for j in range(3):
            tokens[ins_pos[j][hit]] = mask_of_kind[j]
        tokens[api_pos[rng.random(len(api_pos)) < cfg.mask_tokens]] = mask_of_kind[3]
    if rng is not None and cfg.mask_blocks > 0:
        hit = rng.random(n_blk) < cfg.mask_blocks
        in_hit = np.repeat(hit, per_block)
        tokens[in_hit] = mask_of_kind[kind[in_hit]]
        blk_feat[hit] = 0
    tok_weights = token_weights(tokens, per_block, vocab, cfg.block_encoder)

    deg_in, deg_out, edges = block_edges(z, blk, blk_seg, blk_pos, cfg.degree_clip)
    if cfg.control:
        edges = control_edges(cfg.control, rng, sample_id, edges, blk_seg, blk_pos, length)
    edge_seg, edge_src, edge_dst, edge_type = drop_and_flip_edges(edges, cfg.drop_edges, cfg.flip_edges, rng)
    call_src, call_dst = call_pairs(z, fn)

    return {
        'tokens': tokens, 'tok_offsets': tok_offsets, 'tok_weights': tok_weights, 'blk_feat': blk_feat,
        'deg_in': deg_in, 'deg_out': deg_out, 'blk_seg': blk_seg, 'blk_pos': blk_pos,
        'seg_len': length, 'seg_feat': np.stack([np.log1p(length), library], 1).astype(np.float32),
        'seg_fn': np.arange(n_seg), 'n_fn': n_seg,
        'edge_seg': edge_seg, 'edge_src': edge_src, 'edge_dst': edge_dst, 'edge_type': edge_type,
        'call_src': call_src, 'call_dst': call_dst,
        'bin_feat': bin_feat(z),
        'n_segments_total': int((sizes > 0).sum()),
        'n_blocks_kept': n_blk, 'n_blocks_total': int(sizes.sum()),
    }


def encode_binary(z, vocab: TokenVocab, cfg: RunConfig, rng, sample_id=None):
    if cfg.model == 'gnn':
        return encode_binary_gnn(z, vocab, cfg, rng, sample_id)
    return encode_binary_transformer(z, vocab, cfg, rng)


class EpochSampler:
    """Yields (index, epoch) so dataset workers draw fresh segment subsets each epoch; epoch None means eval."""

    def __init__(self, n, seed, train):
        self.n, self.seed, self.train, self.epoch = n, seed, train, 0

    def __iter__(self):
        if not self.train:
            yield from ((i, None) for i in range(self.n))
            return
        self.epoch += 1
        for i in np.random.default_rng([self.seed, self.epoch]).permutation(self.n):
            yield int(i), self.epoch

    def __len__(self):
        return self.n


class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, rows, vocab, cfg):
        self.rows, self.vocab, self.cfg = rows, vocab, cfg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        i, epoch = item
        row = self.rows[i]
        rng = None if epoch is None else np.random.default_rng([self.cfg.seed, epoch, i])
        out = encode_binary(load_sample(row['sample_id']), self.vocab, self.cfg, rng, row['sample_id'])
        out['label'], out['sample_id'] = row['label'], row['sample_id']
        return out


def collate(items):
    """Concatenate binaries; per-binary indices are shifted into batch-global block, segment and function ids."""
    cat = lambda key, dtype=torch.long: torch.from_numpy(np.concatenate([it[key] for it in items])).to(dtype)
    tok_shift = np.cumsum([0] + [len(it['tokens']) for it in items[:-1]])
    seg_shift = np.cumsum([0] + [len(it['seg_len']) for it in items[:-1]])
    fn_shift = np.cumsum([0] + [it['n_fn'] for it in items[:-1]])
    shifted = lambda key, shifts: torch.from_numpy(np.concatenate([it[key] + s for it, s in zip(items, shifts)])).long()
    batch = {
        'tokens': cat('tokens'), 'tok_offsets': shifted('tok_offsets', tok_shift),
        'blk_feat': cat('blk_feat', torch.float32), 'deg_in': cat('deg_in'), 'deg_out': cat('deg_out'),
        'blk_seg': shifted('blk_seg', seg_shift), 'blk_pos': cat('blk_pos'),
        'seg_len': cat('seg_len'), 'seg_feat': cat('seg_feat', torch.float32), 'seg_fn': shifted('seg_fn', fn_shift),
        'seg_bin': torch.from_numpy(np.repeat(np.arange(len(items)), [len(it['seg_len']) for it in items])).long(),
        'n_fn': int(sum(it['n_fn'] for it in items)),
        'edge_seg': shifted('edge_seg', seg_shift), 'edge_src': cat('edge_src'), 'edge_dst': cat('edge_dst'),
        'call_src': shifted('call_src', fn_shift), 'call_dst': shifted('call_dst', fn_shift),
        'label': torch.tensor([it['label'] for it in items], dtype=torch.float32),
        'sample_id': [it['sample_id'] for it in items],
        'n_segments_total': [it['n_segments_total'] for it in items],
    }
    if 'n_blocks_kept' in items[0]:
        batch['n_blocks_kept'] = [it['n_blocks_kept'] for it in items]
        batch['n_blocks_total'] = [it['n_blocks_total'] for it in items]
    if 'tok_weights' in items[0]:
        batch['tok_weights'] = cat('tok_weights', torch.float32)
        batch['edge_type'] = cat('edge_type')
        batch['bin_feat'] = torch.from_numpy(np.stack([it['bin_feat'] for it in items])).float()
    return batch


class BlockBudgetSampler:
    """Batches of up to batch_size binaries in sampler order, closed early when the next binary would push the
    batch past `budget` blocks; a binary over budget on its own forms a batch."""

    def __init__(self, sampler, blocks, batch_size, budget):
        self.sampler, self.blocks, self.batch_size, self.budget = sampler, blocks, batch_size, budget

    def __iter__(self):
        batch, total = [], 0
        for item in self.sampler:
            n = self.blocks[item[0]]
            if batch and (len(batch) == self.batch_size or total + n > self.budget):
                yield batch
                batch, total = [], 0
            batch.append(item)
            total += n
        if batch:
            yield batch


def loader(rows, vocab, cfg, train):
    sampler = EpochSampler(len(rows), cfg.seed, train)
    if cfg.model == 'gnn':
        blocks = [min(int(r['num_blocks']), subgraph.MAX_BINARY_BLOCKS) for r in rows]
        batches = BlockBudgetSampler(sampler, blocks, cfg.batch_size, cfg.batch_blocks)
        return torch.utils.data.DataLoader(
            BinaryDataset(rows, vocab, cfg), batch_sampler=batches, collate_fn=collate,
            num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0, pin_memory=True)
    return torch.utils.data.DataLoader(
        BinaryDataset(rows, vocab, cfg), batch_size=cfg.batch_size, sampler=sampler, collate_fn=collate,
        num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0, pin_memory=True)
