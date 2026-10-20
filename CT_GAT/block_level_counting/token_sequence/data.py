"""Token-sequence data for Model 1: a binary becomes one stream of positions (instruction triplets and API ids in block
order), the stream is cut into windows of L positions, and windows of several binaries are collated into a batch.

Reads a block export through `transformer_paths.reader` only (fold vocabulary ids; the CFG cache is not needed).
"""
from typing import NamedTuple

import numpy as np
import torch

from transformer_paths.reader import BlockStore, FoldView

INSTRUCTION, API = 0, 1
SPLITS = ('train', 'val', 'test')


class Stream(NamedTuple):
    """ids [N, 3] int64 (mn, sig, beh; an API position has its id in slot 0 and zeros after) and kind [N] int64
    (0 instruction, 1 api)."""
    ids: np.ndarray
    kind: np.ndarray


def regroup(tokens, tok_kind, ptr) -> Stream:
    """Regroup kind-major blocks into positions. `tokens` are fold ids, `tok_kind` the export kinds (0 mn, 1 sig, 2 beh,
    3 api) and `ptr` the block boundaries ([n_blocks + 1], offsets into `tokens`). Block j contributes its n_j
    instruction triplets (mn_i, sig_i, beh_i) in order and then its m_j API ids. Raises ValueError when a block does not
    have the layout [mn_1..mn_n | sig_1..sig_n | beh_1..beh_n | api_1..api_m]."""
    tokens = np.asarray(tokens, dtype=np.int64)
    tok_kind = np.asarray(tok_kind, dtype=np.int64)
    ptr = np.asarray(ptr, dtype=np.int64)
    n_blocks = len(ptr) - 1
    owner = np.repeat(np.arange(n_blocks), np.diff(ptr))
    count = np.stack([np.bincount(owner[tok_kind == k], minlength=n_blocks) for k in range(4)])
    unequal = (count[0] != count[1]) | (count[1] != count[2])
    if unequal.any():
        bad = int(np.flatnonzero(unequal)[0])
        raise ValueError(f'block {bad}: {count[0][bad]} mnemonics, {count[1][bad]} signatures, {count[2][bad]} '
                         'behaviour tags: not one each per instruction')
    n_ins, n_api = count[0], count[3]
    local = np.arange(len(tokens)) - ptr[:-1][owner]
    n_of = n_ins[owner]
    expected = np.where(local < n_of, 0, np.where(local < 2 * n_of, 1, np.where(local < 3 * n_of, 2, 3)))
    if (expected != tok_kind).any():
        bad = int(owner[np.flatnonzero(expected != tok_kind)[0]])
        raise ValueError(f'block {bad}: tokens are not laid out [mn | sig | beh | api]')
    size = n_ins + n_api
    start = np.cumsum(size) - size
    is_api = tok_kind == 3
    slot = np.where(is_api, 0, tok_kind)
    position = np.where(is_api, start[owner] + local - 2 * n_of, start[owner] + local - tok_kind * n_of)
    ids = np.zeros((int(size.sum()), 3), dtype=np.int64)
    kind = np.zeros(int(size.sum()), dtype=np.int64)
    ids[position, slot] = tokens
    kind[position[is_api]] = API
    return Stream(ids, kind)


def instruction_stream(fold_view: FoldView, binary: int) -> Stream:
    """The positions of one binary (a store binary index) in store block order, in the fold vocabulary of `fold_view`:
    for each block its instruction triplets and then its API ids as API positions."""
    store = fold_view.store
    first, last = store.block_range(binary)
    ptr = np.asarray(store.blk_tok_ptr[first:last + 1], dtype=np.int64)
    lo, hi = int(ptr[0]), int(ptr[-1])
    tokens = fold_view.lut[np.asarray(store.tokens[lo:hi])]
    return regroup(tokens, np.asarray(store.tok_kind[lo:hi]), ptr - lo)


def windows(stream: Stream, L: int = 512, train: bool = False, max_windows: int = 64, rng=None):
    """Contiguous windows of L positions, block boundaries ignored, as (ids [W, L, 3], kind [W, L], pad_mask [W, L]);
    the last window is padded (pad_mask True on padding, ids and kind 0). Eval (train=False) returns every window in
    order, ceil(N / L) of them. Train returns at most `max_windows` of the windows that start at a random offset in
    [0, L) (rng: a numpy Generator, default a fresh one), in stream order."""
    ids, kind = stream
    n = len(ids)
    offset, picks = 0, None
    if train:
        rng = rng if rng is not None else np.random.default_rng()
        offset = int(rng.integers(0, L))
        if offset >= n:
            offset = 0
    n_win = -(-(n - offset) // L)
    if train and n_win > max_windows:
        picks = np.sort(rng.choice(n_win, size=max_windows, replace=False))
    picks = np.arange(n_win) if picks is None else picks
    w_ids = np.zeros((len(picks), L, 3), dtype=np.int64)
    w_kind = np.zeros((len(picks), L), dtype=np.int64)
    pad = np.ones((len(picks), L), dtype=bool)
    for row, w in enumerate(picks):
        a = offset + int(w) * L
        b = min(a + L, n)
        w_ids[row, :b - a], w_kind[row, :b - a], pad[row, :b - a] = ids[a:b], kind[a:b], False
    return torch.from_numpy(w_ids), torch.from_numpy(w_kind), torch.from_numpy(pad)


def collate(binaries) -> dict:
    """Items of FoldSequences ({'ids' [W, L, 3], 'kind' [W, L], 'pad_mask' [W, L], 'label', 'sample_id'}) -> one batch:
    ids [W, L, 3], kind [W, L], pad_mask [W, L] (windows of all binaries stacked), win2bin [W] (the binary of each
    window, 0..n-1), labels [n], sample_ids [n]. A binary without a position has no window."""
    binaries = list(binaries)
    if not binaries:
        raise ValueError('collate of no binaries')
    if len({b['ids'].shape[1] for b in binaries}) != 1:
        raise ValueError('binaries were cut into windows of different lengths')
    return {
        'ids': torch.cat([b['ids'] for b in binaries]),
        'kind': torch.cat([b['kind'] for b in binaries]),
        'pad_mask': torch.cat([b['pad_mask'] for b in binaries]),
        'win2bin': torch.cat([torch.full((len(b['ids']),), i, dtype=torch.long) for i, b in enumerate(binaries)]),
        'labels': torch.tensor([int(b['label']) for b in binaries], dtype=torch.long),
        'sample_ids': [b['sample_id'] for b in binaries]}


class FoldSequences:
    """The binaries of one split ('train', 'val' or 'test') of one fold of a block export, as windowed streams. It holds
    a FoldView of that split only, so nothing here can read the ids of another split: training code that is given
    `FoldSequences(root, fold, 'train')` cannot reach the test binaries. Item i is a dict for `collate`.

    Windows are random (at most `max_windows`, offset and choice seeded by (seed, epoch, i)) for the 'train' split and
    all of them in order for 'val' and 'test'; `set_epoch` changes the draw."""

    def __init__(self, export_root, fold: int, split: str, L: int = 512, max_windows: int = 64, seed: int = 0) -> None:
        if split not in SPLITS:
            raise ValueError(f'split {split}: one of {SPLITS}')
        self.fold, self.split, self.L, self.max_windows, self.seed, self.epoch = fold, split, L, max_windows, seed, 0
        store = export_root if isinstance(export_root, BlockStore) else BlockStore(export_root)
        self.view = FoldView(store, fold, split)
        self.n_ids = int(self.view.data['vocab']['n_ids'])
        self.export_sha = store.export_sha

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.view.binaries())

    def binary(self, i: int) -> int:
        return int(self.view.binaries()[i])

    def sample_ids(self):
        return self.view.sample_ids()

    def labels(self):
        return [int(self.view.store.binaries[b]['label']) for b in self.view.binaries()]

    def stream(self, i: int) -> Stream:
        return instruction_stream(self.view, self.binary(i))

    def __getitem__(self, i: int) -> dict:
        if not -len(self) <= i < len(self):
            raise IndexError(i)
        i %= len(self)
        train = self.split == 'train'
        rng = np.random.default_rng((self.seed, self.epoch, i)) if train else None
        ids, kind, pad = windows(self.stream(i), self.L, train, self.max_windows, rng)
        info = self.view.store.binaries[self.binary(i)]
        return {'ids': ids, 'kind': kind, 'pad_mask': pad, 'label': int(info['label']), 'sample_id': info['sample_id']}

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
