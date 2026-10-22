"""Per-token weights of the block bag: equal weight (mean), counts (raw, presence, rawlog) or L2-normalised sublinear
tf x idf."""
import numpy as np


def token_weights(tokens, per_block, vocab, encoder):
    """'mean': 1 / tokens in the block. 'raw': 1 per occurrence, so the bag sums to the token counts. 'presence': 1 on
    the first occurrence of each distinct token of a block, 0 on repeats. 'rawlog': per distinct token log1p(c) / c on
    each of its c occurrences. 'tfidf': per block, run id t with count c gets idf[t] (1 + ln c) / c / norm, so the
    occurrences of t sum to the L2-normalised sublinear-tf x idf entry."""
    if encoder == 'mean':
        return np.repeat(1.0 / np.maximum(per_block, 1), per_block).astype(np.float32)
    if encoder == 'raw':
        return np.ones(len(tokens), dtype=np.float32)
    n_vocab = len(vocab.idf)
    blk_of = np.repeat(np.arange(len(per_block)), per_block)
    if encoder in ('presence', 'rawlog'):
        _, first, inv, count = np.unique(blk_of * n_vocab + tokens, return_index=True, return_inverse=True,
                                         return_counts=True)
        if encoder == 'presence':
            out = np.zeros(len(tokens), dtype=np.float32)
            out[first] = 1
            return out
        c = count[inv.reshape(-1)]
        return (np.log1p(c) / c).astype(np.float32)
    idf = vocab.idf.astype(np.float64)
    uniq, inv, count = np.unique(blk_of * n_vocab + tokens, return_inverse=True, return_counts=True)
    value = idf[uniq % n_vocab] * (1 + np.log(count))
    norm = np.sqrt(np.bincount(uniq // n_vocab, weights=value ** 2, minlength=len(per_block)))
    c = count[inv.reshape(-1)]
    return (idf[tokens] * (1 + np.log(c)) / c / norm[blk_of]).astype(np.float32)
