"""Step 10, the unreferenced-import feature: a vocabulary of `dll!name` identities fitted on the training binaries and a
bag of the ones a binary imports but its code never references.

`ImportVocab(train_rows, min_df=2)` keeps an identity that is unreferenced in at least `min_df` training binaries
(validation and test binaries never enter the fit) and gives it `idf = ln((1 + N) / (1 + df)) + 1` over the training
binaries. `ImportVocab.item(sample_id)` is one binary's input: the kept identities it leaves unreferenced as run ids,
their idf weights divided by the L2 norm of the weights (an identity is present once), and two log counts,
log1p(unreferenced imports) and log1p(header imports), counted before the vocabulary cut. `ImportBag` sums the
embeddings of those ids with the weights (`EmbeddingBag`, d = IMPORT_DIM) and `head.make_head` takes it, with the two
counts, as extra inputs. "Unreferenced" is a static fact of the export (CT_GAT/IDA_FEATURE_READERS.md), not run-time
evidence.

`which='header'` is `RunConfig.imports == 'header'` (run-name suffix `-imph`): the same vocabulary, idf, bag and two log counts, over
every header import identity instead of the unreferenced ones (min_df and idf again from the training binaries only). The baselines
(`baselines.py --imports`) use both. `which` leaves the unreferenced item bit for bit as it was.
"""
import numpy as np
from torch import nn

from common.config import CACHE
from pipeline.s10_unused_import_features.import_side import load_imports

IMPORT_DIM = 32
WHICH = {'unreferenced': 'unreferenced_imports', 'header': 'header_imports', 'referenced': 'referenced_imports'}


class ImportVocab:
    def __init__(self, train_rows, min_df=2, which='unreferenced', cache=None):
        self.which, self.min_df, self.cache = which, min_df, str(cache or CACHE)
        count = {}
        for r in train_rows:
            for name in set(load_imports(self.cache, r['sample_id'])[WHICH[which]]):
                count[name] = count.get(name, 0) + 1
        self.names = sorted(n for n, c in count.items() if c >= min_df)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.idf = np.array([np.log((1 + len(train_rows)) / (1 + count[n])) + 1 for n in self.names], dtype=np.float32)

    def __len__(self):
        return len(self.names)

    def ids(self, names):
        """Sorted vocabulary indices of the kept identities among `names`."""
        return np.array(sorted(self.index[n] for n in set(names) if n in self.index), dtype=np.int64)

    def item(self, sample_id):
        """{'imp_idx' int64 [k], 'imp_w' float32 [k], 'imp_feat' float32 [2]} of one binary (cache = the vocabulary's)."""
        got = load_imports(self.cache, sample_id)
        idx = self.ids(got[WHICH[self.which]])
        w = self.idf[idx]
        if len(w):
            w = w / np.float32(np.sqrt((w.astype(np.float64) ** 2).sum()))
        feat = np.log1p([len(got['unreferenced_imports']), len(got['header_imports'])]).astype(np.float32)
        return {'imp_idx': idx, 'imp_w': w.astype(np.float32), 'imp_feat': feat}

    def summary(self):
        return {'which': self.which, 'min_df': self.min_df, 'size': len(self.names)}


class ImportBag(nn.Module):
    """Weighted sum of the embeddings of a binary's kept imports (unreferenced or header, as the vocabulary was fitted): [N, IMPORT_DIM].
    An empty bag is zero."""

    def __init__(self, n_names):
        super().__init__()
        self.bag = nn.EmbeddingBag(max(n_names, 1), IMPORT_DIM, mode='sum')

    def forward(self, batch):
        return self.bag(batch['imp_idx'], batch['imp_offsets'], per_sample_weights=batch['imp_w'])
