"""Per-token weights of the block bag: equal weight (mean) or L2-normalised sublinear tf x idf."""
import numpy as np


def token_weights(tokens, per_block, vocab, encoder):
    """'mean': 1 / tokens in the block. 'tfidf': per block, run id t with count c gets idf[t] (1 + ln c) / c / norm,
    so the occurrences of t sum to the L2-normalised sublinear-tf x idf entry."""
    if encoder == 'mean':
        return np.repeat(1.0 / np.maximum(per_block, 1), per_block).astype(np.float32)
    n_vocab = len(vocab.idf)
    idf = vocab.idf.astype(np.float64)
    blk_of = np.repeat(np.arange(len(per_block)), per_block)
    uniq, inv, count = np.unique(blk_of * n_vocab + tokens, return_inverse=True, return_counts=True)
    value = idf[uniq % n_vocab] * (1 + np.log(count))
    norm = np.sqrt(np.bincount(uniq // n_vocab, weights=value ** 2, minlength=len(per_block)))
    c = count[inv.reshape(-1)]
    return (idf[tokens] * (1 + np.log(c)) / c / norm[blk_of]).astype(np.float32)
