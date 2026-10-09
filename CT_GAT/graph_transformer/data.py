"""Manifests, validation carve, per-run vocabulary cut, and per-binary tensors built from the cache."""
import csv
import json
from dataclasses import dataclass

import numpy as np
import torch

from config import CACHE, RunConfig

TOKEN_KINDS = ('mn', 'sig', 'beh', 'api')


def load_index():
    with (CACHE / 'index.csv').open(encoding='utf-8', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['status'] == 'ok']
    for r in rows:
        r['label'], r['fold'] = int(r['label']), int(r['fold'])
    return rows


@dataclass
class Manifest:
    train: list
    val: list
    test: list
    val_families: list

    def save(self, path):
        doc = {name: [r['sample_id'] for r in getattr(self, name)] for name in ('train', 'val', 'test')}
        doc['val_families'] = self.val_families
        path.write_text(json.dumps(doc, indent=1), encoding='utf-8')


def carve_validation(rows, fraction, seed):
    """Hold out whole ransomware families and whole goodware identity groups, about `fraction` of each class."""
    rng = np.random.default_rng(seed)
    held = set()
    val_families = []
    for label, key in ((1, 'family'), (0, 'identity_group')):
        members = {}
        for r in rows:
            if r['label'] == label:
                members.setdefault(r[key], []).append(r['sample_id'])
        groups = sorted(members)
        target, taken = fraction * sum(len(v) for v in members.values()), 0
        for i in rng.permutation(len(groups)):
            if taken >= target:
                break
            held.update(members[groups[i]])
            taken += len(members[groups[i]])
            if label == 1:
                val_families.append(groups[i])
    return [r for r in rows if r['sample_id'] not in held], [r for r in rows if r['sample_id'] in held], sorted(val_families)


def make_manifest(cfg: RunConfig, index):
    rows = [r for r in index if cfg.arch == 'all' or r['arch'] == cfg.arch]
    if cfg.protocol == 'bundle':
        pool, test = [r for r in rows if r['split'] == 'train'], [r for r in rows if r['split'] == 'test']
    else:
        pool, test = [r for r in rows if r['fold'] != cfg.fold], [r for r in rows if r['fold'] == cfg.fold]
    train, val, val_families = carve_validation(pool, cfg.val_fraction, cfg.seed)
    return Manifest(train, val, test, val_families)


def load_sample(sample_id):
    with np.load(CACHE / 'samples' / f'{sample_id}.npz') as z:
        return {k: z[k] for k in ('mn', 'sig', 'beh', 'api', 'blk_insn_ptr', 'blk_api_ptr', 'blk_feat',
                                  'fn_blk_ptr', 'fn_flags', 'cfg_edges', 'call_edges')}


class TokenVocab:
    """Global vocab id -> run id; 0 is UNK. A token survives if it occurs in >= min_df training binaries."""

    def __init__(self, train_rows, min_df):
        sizes = {k: len(v) for k, v in json.loads((CACHE / 'vocab.json').read_text(encoding='utf-8')).items()}
        df = {k: np.zeros(sizes[k], dtype=np.int64) for k in ('mn', 'sig', 'api')}
        for r in train_rows:
            with np.load(CACHE / 'samples' / f"{r['sample_id']}.npz") as z:
                for k in df:
                    df[k][z[f'{k}_u']] += 1
        self.lut, self.size = {}, {}
        for k, counts in df.items():
            keep = counts >= min_df
            self.lut[k] = np.where(keep, np.cumsum(keep), 0).astype(np.int64)
            self.size[k] = int(keep.sum()) + 1
        self.lut['beh'] = np.arange(sizes['beh'], dtype=np.int64)
        self.size['beh'] = sizes['beh']
        self.offset = dict(zip(TOKEN_KINDS, np.cumsum([0] + [self.size[k] for k in TOKEN_KINDS[:-1]]).tolist()))
        self.total = sum(self.size.values())

    def summary(self):
        return {'sizes_with_unk': self.size, 'total': self.total}


def ranges(starts, lengths):
    """Concatenation of arange(s, s + n) for each (s, n)."""
    lengths = np.asarray(lengths, dtype=np.int64)
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    first = np.cumsum(lengths) - lengths
    return np.repeat(np.asarray(starts, dtype=np.int64) - first, lengths) + np.arange(total, dtype=np.int64)


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


def encode_binary(z, vocab: TokenVocab, cfg: RunConfig, rng):
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
        out = encode_binary(load_sample(row['sample_id']), self.vocab, self.cfg, rng)
        out['label'], out['sample_id'] = row['label'], row['sample_id']
        return out


def collate(items):
    """Concatenate binaries; per-binary indices are shifted into batch-global block, segment and function ids."""
    cat = lambda key, dtype=torch.long: torch.from_numpy(np.concatenate([it[key] for it in items])).to(dtype)
    tok_shift = np.cumsum([0] + [len(it['tokens']) for it in items[:-1]])
    seg_shift = np.cumsum([0] + [len(it['seg_len']) for it in items[:-1]])
    fn_shift = np.cumsum([0] + [it['n_fn'] for it in items[:-1]])
    shifted = lambda key, shifts: torch.from_numpy(np.concatenate([it[key] + s for it, s in zip(items, shifts)])).long()
    return {
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


def loader(rows, vocab, cfg, train):
    sampler = EpochSampler(len(rows), cfg.seed, train)
    return torch.utils.data.DataLoader(
        BinaryDataset(rows, vocab, cfg), batch_size=cfg.batch_size, sampler=sampler, collate_fn=collate,
        num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0, pin_memory=True)
