"""Token vocabulary: global vocab id -> run id by training document frequency, one [MASK] id per kind, idf."""
import json

import numpy as np

from common.config import CACHE, RunConfig

TOKEN_KINDS = ('mn', 'sig', 'beh', 'api')


class TokenVocab:
    """Global vocab id -> run id; 0 is UNK. A token survives if it occurs in >= min_df training binaries.

    For gnn configs, a [MASK] id per kind is appended after the kinds' ranges (`mask_id`, counted in `total`) and
    `idf[run id] = ln((1 + N) / (1 + df)) + 1` over the training binaries (UNK and MASK get 1)."""

    def __init__(self, train_rows, cfg: RunConfig):
        gnn = cfg.model == 'gnn'
        sizes = {k: len(v) for k, v in json.loads((CACHE / 'vocab.json').read_text(encoding='utf-8')).items()}
        df = {k: np.zeros(sizes[k], dtype=np.int64) for k in ('mn', 'sig', 'api')}
        df_beh = np.zeros(sizes['beh'], dtype=np.int64)
        for r in train_rows:
            with np.load(CACHE / 'samples' / f"{r['sample_id']}.npz") as z:
                for k in df:
                    df[k][z[f'{k}_u']] += 1
                if gnn:
                    df_beh[np.unique(z['beh'])] += 1
        self.lut, self.size = {}, {}
        kept_df = {}
        for k, counts in df.items():
            keep = counts >= cfg.min_df
            self.lut[k] = np.where(keep, np.cumsum(keep), 0).astype(np.int64)
            self.size[k] = int(keep.sum()) + 1
            kept_df[k] = counts[keep]
        self.lut['beh'] = np.arange(sizes['beh'], dtype=np.int64)
        self.size['beh'] = sizes['beh']
        self.offset = dict(zip(TOKEN_KINDS, np.cumsum([0] + [self.size[k] for k in TOKEN_KINDS[:-1]]).tolist()))
        self.total = sum(self.size.values())
        self.idf = self.mask_id = None
        if gnn:
            kept_df['beh'] = df_beh
            self.idf = np.ones(self.total + len(TOKEN_KINDS), dtype=np.float32)
            for k, counts in kept_df.items():
                lo = self.offset[k] + (k != 'beh')
                self.idf[lo:lo + len(counts)] = np.log((1 + len(train_rows)) / (1 + counts)) + 1
            self.mask_id = {k: self.total + j for j, k in enumerate(TOKEN_KINDS)}
            self.total += len(TOKEN_KINDS)

    def summary(self):
        return {'sizes_with_unk': self.size, 'total': self.total}
