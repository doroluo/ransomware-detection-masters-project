"""Token vocabulary: global vocab id -> run id by training document frequency, one [MASK] id per kind, idf."""
import json

import numpy as np

from common.config import CACHE, RunConfig
from pipeline.s03_tokenizer.import_reader_side import YTOK_SIZES, load_ytok

TOKEN_KINDS = ('mn', 'sig', 'beh', 'api')
Y_KINDS = ('y0', 'y1', 'y2')
Y_SIZES = dict(zip(Y_KINDS, YTOK_SIZES))
VOCAB_KINDS = {'ours': TOKEN_KINDS,
               'yanping': Y_KINDS + ('api',),
               'hybrid': ('mn', 'beh', 'y1', 'y2', 'api')}


def insn_values(z, kind):
    """The per-instruction global ids of a kind: a cache array (`mn`, `sig`, `beh`) or a column of the reader's `ytok`."""
    return z['ytok'][:, int(kind[1])] if kind in Y_KINDS else z[kind]


class TokenVocab:
    """Global vocab id -> run id; 0 is UNK. A token survives if it occurs in >= min_df training binaries.

    `cfg.vocab` picks the kinds (`kinds`, the last one is always `api`; `insn_kinds` are the per-instruction ones):
    'ours' mn, sig, beh, api; 'yanping' the reader's three ids y0 (opcode or API, < 256), y1, y2 (operand categories, < 76)
    plus api; 'hybrid' our mn and beh plus the reader's y1, y2 plus api. Every kind but `beh` is cut on the training
    binaries the same way (document frequency >= min_df, UNK = 0); `beh` keeps all its ids, as before. The reader kinds
    need `<cache>/side/yanping/` (import_reader_side.py).

    For gnn configs, a [MASK] id per kind is appended after the kinds' ranges (`mask_id`, counted in `total`) and
    `idf[run id] = ln((1 + N) / (1 + df)) + 1` over the training binaries (UNK and MASK get 1)."""

    def __init__(self, train_rows, cfg: RunConfig):
        gnn = cfg.model == 'gnn'
        if cfg.vocab != 'ours' and not gnn:
            raise ValueError("vocab other than 'ours' needs model='gnn'")
        self.name, self.kinds = cfg.vocab, VOCAB_KINDS[cfg.vocab]
        self.insn_kinds = self.kinds[:-1]
        sizes = {k: len(v) for k, v in json.loads((CACHE / 'vocab.json').read_text(encoding='utf-8')).items()}
        sizes.update(Y_SIZES)
        cut = [k for k in self.kinds if k != 'beh']
        df = {k: np.zeros(sizes[k], dtype=np.int64) for k in cut}
        df_beh = np.zeros(sizes['beh'], dtype=np.int64)
        for r in train_rows:
            with np.load(CACHE / 'samples' / f"{r['sample_id']}.npz") as z:
                ours = {k: z[f'{k}_u'] for k in cut if k in ('mn', 'sig', 'api')}
                if gnn and 'beh' in self.kinds:
                    df_beh[np.unique(z['beh'])] += 1
            if cfg.vocab != 'ours':
                ytok = load_ytok(CACHE, r['sample_id'])
                ours.update({k: np.unique(ytok[:, int(k[1])]) for k in cut if k in Y_KINDS})
            for k, ids in ours.items():
                df[k][ids] += 1
        self.lut, self.size = {}, {}
        kept_df = {}
        for k, counts in df.items():
            keep = counts >= cfg.min_df
            self.lut[k] = np.where(keep, np.cumsum(keep), 0).astype(np.int64)
            self.size[k] = int(keep.sum()) + 1
            kept_df[k] = counts[keep]
        if 'beh' in self.kinds:
            self.lut['beh'] = np.arange(sizes['beh'], dtype=np.int64)
            self.size['beh'] = sizes['beh']
        self.offset = dict(zip(self.kinds, np.cumsum([0] + [self.size[k] for k in self.kinds[:-1]]).tolist()))
        self.total = sum(self.size.values())
        self.idf = self.mask_id = None
        if gnn:
            if 'beh' in self.kinds:
                kept_df['beh'] = df_beh
            self.idf = np.ones(self.total + len(self.kinds), dtype=np.float32)
            for k, counts in kept_df.items():
                lo = self.offset[k] + (k != 'beh')
                self.idf[lo:lo + len(counts)] = np.log((1 + len(train_rows)) / (1 + counts)) + 1
            self.mask_id = {k: self.total + j for j, k in enumerate(self.kinds)}
            self.total += len(self.kinds)

    def summary(self):
        return {'sizes_with_unk': self.size, 'total': self.total}
