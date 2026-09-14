#!/usr/bin/env python3
"""Graph embeddings from cached Weisfeiler-Lehman documents.

Three representations, all fitted on TRAIN graphs only:

wl_tfidf      WL subtree label counts -> TF-IDF. This is the WL subtree kernel
              feature map with IDF weighting; a linear model on it is a WL
              kernel machine. Used for the "plain WL histogram" baseline.
wl_svd        wl_tfidf -> TruncatedSVD(dim). A dense graph embedding.
graph2vec     PV-DBOW (doc2vec skip-gram with negative sampling) over the WL
              documents. This *is* the graph2vec algorithm (Narayanan et al.
              2017); what is not used is karateclub's implementation.

Why not karateclub
------------------
karateclub is sdist-only and needs gensim>=4 for its Doc2Vec. gensim has no
cp314 wheel (pip offers only the pure-python 0.10.x line, which is a different
library) and building it needs a C toolchain that is not present. So PV-DBOW is
implemented here in ~80 lines of vectorised numpy with a fixed seed. It is the
same objective, not a stand-in for it; the difference from karateclub is the
optimiser schedule, not the model.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


# ---------------------------------------------------------------------------
# cache -> sparse count matrix
# ---------------------------------------------------------------------------
def load_cache(path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    z = np.load(path, allow_pickle=False)
    shas, indptr, labels, counts = (z["shas"], z["indptr"], z["labels"],
                                    z["counts"])
    out = {}
    for i, s in enumerate(shas):
        a, b = indptr[i], indptr[i + 1]
        out[str(s)] = (labels[a:b], counts[a:b])
    return out


def build_counts(docs: list[tuple[np.ndarray, np.ndarray]],
                 train_mask: np.ndarray, min_df: int = 5):
    """Sparse count matrix over a vocabulary taken from TRAIN docs only.

    Returns (X csr float32, vocab int64 sorted, df int32 over train).
    Labels not seen in >= min_df training graphs are dropped everywhere, which
    is what keeps the h=2 layer (nearly one label per node) from dominating.
    """
    tr = [docs[i][0] for i in range(len(docs)) if train_mask[i]]
    if not tr:
        raise ValueError("no training documents")
    allv = np.concatenate(tr)
    uniq, df = np.unique(allv, return_counts=True)   # per-doc labels are unique
    vocab = uniq[df >= min_df]
    dfk = df[df >= min_df]

    indptr = [0]
    idx_parts, dat_parts = [], []
    for lab, cnt in docs:
        pos = np.searchsorted(vocab, lab)
        pos_c = np.clip(pos, 0, len(vocab) - 1)
        ok = vocab[pos_c] == lab
        cols = pos_c[ok]
        vals = cnt[ok].astype(np.float32)
        idx_parts.append(cols.astype(np.int32))
        dat_parts.append(vals)
        indptr.append(indptr[-1] + len(cols))
    X = sparse.csr_matrix(
        (np.concatenate(dat_parts) if dat_parts else np.empty(0, np.float32),
         np.concatenate(idx_parts) if idx_parts else np.empty(0, np.int32),
         np.array(indptr, dtype=np.int64)),
        shape=(len(docs), len(vocab)))
    return X, vocab, dfk


# ---------------------------------------------------------------------------
# TF-IDF / SVD
# ---------------------------------------------------------------------------
def tfidf_fit_transform(X, train_mask):
    from sklearn.feature_extraction.text import TfidfTransformer
    tf = TfidfTransformer(sublinear_tf=True)
    tf.fit(X[train_mask])
    return tf.transform(X), tf


def svd_fit_transform(Xt, train_mask, dim=128, seed=42):
    from sklearn.decomposition import TruncatedSVD
    dim = int(min(dim, min(Xt.shape) - 1))
    sv = TruncatedSVD(n_components=dim, random_state=seed, algorithm="randomized")
    sv.fit(Xt[train_mask])
    return sv.transform(Xt).astype(np.float32), sv


# ---------------------------------------------------------------------------
# PV-DBOW == graph2vec
# ---------------------------------------------------------------------------
class PVDBOW:
    """doc2vec PV-DBOW with negative sampling, vectorised minibatch SGD."""

    def __init__(self, dim=128, negatives=3, epochs=12, batch=8192,
                 lr=0.05, min_lr=1e-4, seed=42, max_seconds=420.0,
                 max_steps=9000, max_steps_infer=4500):
        self.dim, self.neg, self.epochs = dim, negatives, epochs
        self.batch, self.lr0, self.min_lr = batch, lr, min_lr
        self.seed, self.max_seconds = seed, max_seconds
        self.max_steps, self.max_steps_infer = max_steps, max_steps_infer
        self.Wo = None
        self.noise_cum = None
        self.vocab_size = None

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _row_cum(X):
        cum = np.cumsum(X.data, dtype=np.float64)
        return cum, float(cum[-1]) if len(cum) else 0.0

    def _sample_pairs(self, X, cum, total, rng, n):
        u = rng.random(n) * total
        nz = np.searchsorted(cum, u, side="left")
        np.clip(nz, 0, len(X.data) - 1, out=nz)
        rows = np.searchsorted(X.indptr, nz, side="right") - 1
        cols = X.indices[nz].astype(np.int64)
        return rows.astype(np.int64), cols

    def _sample_neg(self, rng, shape):
        u = rng.random(shape) * self.noise_cum[-1]
        return np.clip(np.searchsorted(self.noise_cum, u, side="left"),
                       0, self.vocab_size - 1)

    def _run(self, X, Wd, train_out: bool, rng, log=None):
        import time
        cum, total = self._row_cum(X)
        if total <= 0:
            return Wd
        per_epoch = max(1, int(total // self.batch))
        cap = self.max_steps if train_out else self.max_steps_infer
        steps = min(per_epoch * self.epochs, cap)
        if log:
            log(f"    PV-DBOW {'fit' if train_out else 'infer'}: {steps} steps "
                f"x {self.batch} samples ({per_epoch} steps = 1 epoch over "
                f"{int(total):,} tokens)")
        t0 = time.time()
        for s in range(steps):
            lr = max(self.min_lr, self.lr0 * (1.0 - s / steps))
            d, w = self._sample_pairs(X, cum, total, rng, self.batch)
            negs = self._sample_neg(rng, (self.batch, self.neg))
            tgt = np.concatenate([w[:, None], negs], axis=1)        # B,K+1
            y = np.zeros(tgt.shape, dtype=np.float32)
            y[:, 0] = 1.0
            h = Wd[d]                                               # B,dim
            o = self.Wo[tgt]                                        # B,K+1,dim
            score = np.einsum("bd,bkd->bk", h, o, optimize=True)
            np.clip(score, -12, 12, out=score)
            g = (y - 1.0 / (1.0 + np.exp(-score))) * lr             # B,K+1
            gh = np.einsum("bk,bkd->bd", g, o, optimize=True)
            if train_out:
                go = g[:, :, None] * h[:, None, :]
                np.add.at(self.Wo, tgt, go.astype(np.float32))
            np.add.at(Wd, d, gh.astype(np.float32))
            if (s + 1) % 200 == 0 and time.time() - t0 > self.max_seconds:
                if log:
                    log(f"    PV-DBOW stopped early at step {s+1}/{steps} "
                        f"({time.time()-t0:.0f}s budget)")
                break
        return Wd

    # -- API --------------------------------------------------------------
    def fit(self, X_train, log=None):
        rng = np.random.default_rng(self.seed)
        self.vocab_size = X_train.shape[1]
        freq = np.asarray(X_train.sum(axis=0)).ravel() ** 0.75
        freq[freq <= 0] = 1e-9
        self.noise_cum = np.cumsum(freq)
        scale = 0.5 / self.dim
        self.Wo = np.zeros((self.vocab_size, self.dim), dtype=np.float32)
        Wd = ((rng.random((X_train.shape[0], self.dim), dtype=np.float32) - .5)
              * scale).astype(np.float32)
        self.Wd_train = self._run(X_train, Wd, True, rng, log)
        return self

    def infer(self, X_new, log=None):
        """Freeze the label vectors, fit doc vectors for unseen graphs."""
        rng = np.random.default_rng(self.seed + 1)
        scale = 0.5 / self.dim
        Wd = ((rng.random((X_new.shape[0], self.dim), dtype=np.float32) - .5)
              * scale).astype(np.float32)
        return self._run(X_new, Wd, False, rng, log)


def l2norm(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return A / n
