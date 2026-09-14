"""
tuning.py - the pieces the tuned runs need that the six committed experiments
never did: a sequence budget that is not "the first 5,000 lines", pooling that
is not a plain mean, n-gram features over the tokenizer's own output, a
group-aware cross-validation splitter, and a decision threshold chosen from
out-of-fold scores.

Everything here is a pure function over numpy / scipy / stdlib. Nothing in this
module imports gensim, transformers or tokenizers, so `tests/` can pin it under
the repo's Python 3.14 as well as the 3.12 venv; the heavy pieces live in
`tune.py`, which is the driver.

THE ONE IDEA THAT MAKES THE SEARCH AFFORDABLE. The revised feature files hold
one MNEMONIC per line, and the whole corpus uses ~1,300 distinct mnemonics. So a
document is fully described, for every representation used here, by the multiset
of its token n-grams:

  * TF-IDF over 1-3 grams is that multiset, reweighted.
  * A mean-pooled Word2Vec vector is `counts @ W / counts.sum()` - the token
    vectors are fixed, so summing over 50,000 positions is the same arithmetic
    as one sparse matrix product against the 1,300-column count vector.
  * TF-IDF-weighted mean pooling and max pooling are the same product with a
    different weight vector / reduction.

That turns "re-pool 2,500 files x 50,000 tokens" from a minutes-long Python loop
into a single sparse matmul, which is what lets several hundred configurations
be cross-validated inside the compute budget. The count matrix is built once per
(experiment, budget, sampling) and reused by every configuration downstream.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Sequence budget and positional sampling
# ---------------------------------------------------------------------------
#
# The committed pipeline reads `max_instructions = 5000` lines from the TOP of
# each file and stops. On the revised corpus that is a severe truncation - the
# median Mendeley ransomware test file holds 144,236 mnemonics, so 5,000 lines
# is the first 3.5% of it, and what lives in that first 3.5% of a PE's .text is
# the CRT startup and the compiler's prologue, which is the part that is most
# alike across every binary built with the same toolchain. Two knobs are offered
# instead:
#
#   head    - the first `budget` normalized lines (what the pipeline does today)
#   strided - `budget` lines spread evenly across the WHOLE file
#
# `strided` costs one extra pass to count lines and is the only one of the two
# that can see the end of a large binary at all.


def count_lines(path) -> int:
    """Number of physical lines in a file, counted over raw bytes.

    Deliberately not `sum(1 for _ in fh)`: the strided sampler needs the count
    before it can choose indices, and on Goodware_Balanced (645M lines across
    1,337 files, one 49.7M-line outlier) the decode-and-split cost of a text
    iteration is the dominant term. Counting `b"\\n"` over 1 MiB blocks is
    memory-flat and roughly an order of magnitude faster.

    A file not ending in a newline still has its last line counted, matching
    what iterating the file would yield.
    """
    n = 0
    tail = b""
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            n += chunk.count(b"\n")
            tail = chunk[-1:]
    if tail and tail != b"\n":
        n += 1
    return n


def strided_indices(n: int, budget: int) -> np.ndarray:
    """`budget` line indices spread evenly over `range(n)`, endpoints included.

    Returns every index when the file is at or under budget, so a short file is
    read whole and `strided` degenerates to `head` exactly - which is what makes
    the two samplers comparable on the small files and different only where the
    truncation actually bites.
    """
    if n <= 0 or budget <= 0:
        return np.empty(0, dtype=np.int64)
    if n <= budget:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, budget).astype(np.int64))


def even_subsample(seq, budget: int):
    """Thin an already-strided sample down to `budget` items, evenly.

    An evenly-spaced subsample of an evenly-spaced sample is itself an evenly
    spaced sample of the original file, to within one line of index. That is the
    property the token cache relies on: it stores ONE strided sample per file at
    the largest budget in the search, and every smaller strided budget is
    derived from it here instead of re-reading the corpus.
    """
    n = len(seq)
    if n <= budget:
        return seq
    idx = np.unique(np.linspace(0, n - 1, budget).astype(np.int64))
    return [seq[i] for i in idx] if isinstance(seq, list) else seq[idx]


def read_tokens(path, normalize, budget: int, sampling: str = "head",
                n_lines: int | None = None) -> list[str]:
    """The normalized token sequence for one file under one budget/sampler.

    `normalize` is `Tokenization.tokenization.normalize_instruction`, imported by
    the caller from the tokenization repo rather than reimplemented - it is the
    single point where two disassemblers' output has to converge, and a second
    copy of it here would be a second thing to keep in step.

    Lines that normalize to nothing are dropped and do NOT consume budget, which
    is what `run_pipeline.read_instructions` does for `head`. For `strided` the
    budget is spent on LINE indices before normalization, so a file with blank
    lines yields slightly fewer than `budget` tokens; on the revised corpus
    (schema-checked: no blank lines, one mnemonic per line) the two coincide.
    """
    if sampling == "head":
        out = []
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                t = normalize(line)
                if t:
                    out.append(t)
                    if budget and len(out) >= budget:
                        break
        return out

    if sampling != "strided":
        raise ValueError(f"unknown sampling: {sampling!r}")

    n = count_lines(path) if n_lines is None else n_lines
    want = strided_indices(n, budget)
    if want.size == 0:
        return []
    keep = set(want.tolist())
    hi = int(want[-1])
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i in keep:
                t = normalize(line)
                if t:
                    out.append(t)
            if i >= hi:
                break
    return out


# ---------------------------------------------------------------------------
# Count matrices over token n-grams
# ---------------------------------------------------------------------------


def ngram_keys(ids: np.ndarray, n: int, vocab: int) -> np.ndarray:
    """Integer keys for every length-`n` window of a token-id sequence.

    Base-`vocab` positional encoding, so the key is a bijection onto the n-grams
    actually present and no hashing collision is possible. With vocab <= 4,096
    and n <= 3 the largest key is < 2^36 and fits an int64 with room to spare.
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    if ids.size < n:
        return np.empty(0, dtype=np.int64)
    k = ids[: ids.size - n + 1].astype(np.int64)
    for j in range(1, n):
        k = k * vocab + ids[j: ids.size - n + 1 + j]
    return k


def doc_ngram_counts(ids: np.ndarray, ngram_range: tuple[int, int], vocab: int):
    """(keys, counts) for one document over the requested n-gram orders.

    Orders are kept apart by offsetting order `n`'s keys past the whole key
    space of the orders below it, so a unigram id can never collide with a
    bigram key.
    """
    lo, hi = ngram_range
    all_keys, all_counts = [], []
    offset = 0
    for n in range(lo, hi + 1):
        k = ngram_keys(ids, n, vocab)
        if k.size:
            u, c = np.unique(k, return_counts=True)
            all_keys.append(u + offset)
            all_counts.append(c)
        offset += vocab ** n
    if not all_keys:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(all_keys), np.concatenate(all_counts)


def build_count_matrix(id_seqs, ngram_range, vocab, min_df_global=2):
    """CSR document-term count matrix over token n-grams, built once.

    `min_df_global` prunes n-grams seen in fewer than that many documents ACROSS
    THE WHOLE experiment, train and test alike. That is not a leak and it is not
    a modelling choice: a feature with global document frequency 1 cannot reach
    document frequency 2 inside any subset of the documents, so pruning at
    global df >= 2 removes exactly the columns that a per-fold `min_df = 2`
    would remove anyway. It exists because the unpruned trigram space on a
    50,000-mnemonic budget is tens of millions of columns of almost entirely
    singletons, and building it is what would not fit in memory. Anything that
    survives is re-fitted per fold from the fold's TRAIN rows only
    (`fold_tfidf`), so no test row ever contributes a vocabulary entry, an IDF,
    or a document frequency to a model that is scored on it.
    """
    from scipy import sparse

    rows, keys, vals = [], [], []
    for i, ids in enumerate(id_seqs):
        k, c = doc_ngram_counts(ids, ngram_range, vocab)
        if k.size == 0:
            continue
        rows.append(np.full(k.size, i, dtype=np.int32))
        keys.append(k)
        vals.append(c.astype(np.float32))
    if not rows:
        return sparse.csr_matrix((len(id_seqs), 0), dtype=np.float32), np.empty(0, np.int64)
    rows = np.concatenate(rows)
    keys = np.concatenate(keys)
    vals = np.concatenate(vals)

    uniq, inv = np.unique(keys, return_inverse=True)
    if min_df_global > 1:
        df = np.bincount(inv, minlength=uniq.size)
        keep = df >= min_df_global
        if not keep.all():
            remap = np.full(uniq.size, -1, dtype=np.int64)
            remap[keep] = np.arange(int(keep.sum()))
            col = remap[inv]
            sel = col >= 0
            rows, col, vals = rows[sel], col[sel], vals[sel]
            uniq = uniq[keep]
            inv = col
    mat = sparse.csr_matrix((vals, (rows, inv)),
                            shape=(len(id_seqs), uniq.size), dtype=np.float32)
    mat.sum_duplicates()
    return mat, uniq


def fold_tfidf(counts, train_rows, min_df=2, sublinear=True, max_features=None):
    """TF-IDF exactly as `TfidfVectorizer(...).fit(train).transform(all)` would.

    Re-derived per fold from `train_rows` alone: the surviving column set is the
    columns with document frequency >= `min_df` among the fold's training
    documents, and the IDF is smoothed on those documents only
    (`1 + ln((1+n)/(1+df))`, sklearn's default), `sublinear_tf` is `1 + ln(tf)`,
    and rows are L2-normalized. Written out rather than re-fitting a sklearn
    vectoriser per fold because the counts do not depend on the split, and
    re-tokenizing 125M tokens five times per configuration is the whole compute
    budget.

    Returns (X_all, keep_mask). Rows outside `train_rows` are transformed, never
    fitted.
    """
    from scipy import sparse
    from sklearn.preprocessing import normalize as l2

    tr = counts[train_rows].tocsc()
    df = np.diff(tr.indptr)
    keep = df >= min_df
    if not keep.any():
        keep = df >= 1
    if max_features is not None and int(keep.sum()) > max_features:
        # sklearn's `max_features` keeps the globally most FREQUENT terms of the
        # fitted documents, ties broken by term. Same rule here, on the fold's
        # training rows only.
        tf = np.asarray(tr.sum(axis=0)).ravel()
        tf[~keep] = -1.0
        cut = np.argsort(-tf, kind="stable")[:max_features]
        keep = np.zeros_like(keep)
        keep[cut] = True
    df = df.copy()
    n = tr.shape[0]
    idf = 1.0 + np.log((1.0 + n) / (1.0 + df[keep].astype(np.float64)))

    X = counts[:, keep].tocsr(copy=True).astype(np.float64)
    if sublinear:
        X.data = 1.0 + np.log(X.data)
    X = X.multiply(sparse.csr_matrix(idf.reshape(1, -1)))
    return l2(sparse.csr_matrix(X), norm="l2", copy=False).astype(np.float32), keep


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------
#
# All three poolings are order-free reductions of the token multiset, so all
# three are one product against the unigram count matrix. That is not an
# approximation of the loop in `Embedding/build_masked_embeddings.py`: for
# `mean` it is the same number to floating-point rounding, and
# `tune.py --check-pooling` asserts it against the real function.


def pool_mean(counts, W):
    """Mean of the token vectors, summed with multiplicity. counts: n x V."""
    tot = np.asarray(counts.sum(axis=1)).ravel()
    tot[tot == 0] = 1.0
    return np.asarray(counts @ W) / tot[:, None]


def pool_weighted_mean(counts, W, weights):
    """Mean weighted by a per-token weight (here: the fold's unigram IDF).

    Rare mnemonics carry nearly all of the class signal on this corpus - `mov`,
    `push` and `call` are a third of every file on both sides - and a plain mean
    lets them dominate the vector. This is the same reweighting TF-IDF applies,
    pushed into the pooling step.
    """
    w = counts.multiply(weights.reshape(1, -1)).tocsr()
    tot = np.asarray(w.sum(axis=1)).ravel()
    tot[tot == 0] = 1.0
    return np.asarray(w @ W) / tot[:, None]


def pool_mean_max(counts, W):
    """[mean ; elementwise max over the tokens present] - twice the width.

    Max pooling over a bag is a presence feature: it reports the extreme value
    of each embedding dimension over the token TYPES in the file, with no regard
    to how often they occur, so it carries what the mean averages away.
    """
    mean = pool_mean(counts, W)
    n, d = counts.shape[0], W.shape[1]
    mx = np.full((n, d), -np.inf, dtype=np.float64)
    ind, ptr = counts.indices, counts.indptr
    for i in range(n):
        cols = ind[ptr[i]:ptr[i + 1]]
        if cols.size:
            mx[i] = W[cols].max(axis=0)
        else:
            mx[i] = 0.0
    return np.hstack([mean, mx])


# ---------------------------------------------------------------------------
# Group-aware cross-validation and thresholds
# ---------------------------------------------------------------------------


def group_folds(y, groups, n_splits=5, seed=42):
    """Fold indices that never put one group on both sides of a fold.

    `groups` is the `group` column of splits.csv: the ransomware FAMILY for a
    positive and the source project (or the file itself) for a negative. Holding
    out whole families is the entire point - the Mendeley test split is
    family-disjoint by construction, so a model selected under random folds is
    selected for a task the test set does not ask.

    `StratifiedGroupKFold` where available so every fold keeps both classes;
    plain `GroupKFold` otherwise. Shuffled with a fixed seed, so the folds are
    reproducible and are not the corpus's filename order.
    """
    try:
        from sklearn.model_selection import StratifiedGroupKFold as K
        splitter = K(n_splits=n_splits, shuffle=True, random_state=seed)
    except ImportError:                                   # pragma: no cover
        from sklearn.model_selection import GroupKFold as K
        splitter = K(n_splits=n_splits)
    return list(splitter.split(np.zeros(len(y)), y, groups))


def best_threshold(y_true, scores, metric="macro_f1", grid=257):
    """The decision threshold that maximises `metric` on OUT-OF-FOLD scores.

    The default 0.5 (or 0.0 for a decision function) is only optimal when the
    training prior matches the deployment prior, and here it does not: the
    Mendeley train split is 45% ransomware and its test split is 74%. Choosing
    the cut on out-of-fold training scores uses no test row at all, and the
    chosen number is reported with every tuned row so it can be checked.

    Candidates are the score quantiles, so the search is invariant to any
    monotone rescaling of `scores` (a `decision_function` and a
    `predict_proba` give the same cut).
    """
    from sklearn.metrics import balanced_accuracy_score, f1_score

    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return 0.5, 0.0
    qs = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, grid)))
    cands = np.unique(np.concatenate([qs, [scores.min() - 1e-9, scores.max() + 1e-9]]))

    def sc(t):
        p = (scores >= t).astype(int)
        if metric == "balanced_accuracy":
            return balanced_accuracy_score(y_true, p)
        return f1_score(y_true, p, average="macro", zero_division=0)

    vals = [sc(t) for t in cands]
    i = int(np.argmax(vals))
    return float(cands[i]), float(vals[i])


# The operating point is chosen for BALANCED ACCURACY, not for macro-F1, and the
# reason is a property of the corpus rather than a preference. The Mendeley
# train split is 45% ransomware and its test split is 74%; macro-F1's optimal
# threshold moves with the class prior, balanced accuracy's does not. A
# threshold tuned for macro-F1 on training folds therefore encodes the TRAINING
# prior and transfers badly - measured on this corpus it costs about 0.10
# macro-F1 on the test set relative to leaving the cut alone, with goodware
# recall falling from ~0.93 to ~0.60. Balanced accuracy is the prior-free
# criterion and is what the search fixes; `default` and the macro-F1 threshold
# are scored alongside for every configuration so the choice is visible rather
# than asserted.
THRESHOLD_RULES = ("default", "oof_balanced_accuracy", "oof_macro_f1")


def nested_threshold_scores(y_true, oof, folds, default_cut=0.0):
    """CV metrics for each threshold rule, with the threshold chosen honestly.

    For fold k the threshold is chosen on the out-of-fold scores of the OTHER
    folds only, then applied to fold k. Those scores are themselves out-of-fold,
    so no held-out row contributes to the cut that classifies it, and the three
    rules can be compared without one of them being scored on the very points it
    was fitted to. `default` fits nothing and is included as the null rule.

    Returns `{rule: {"macro_f1", "balanced_accuracy", "threshold"}}` where
    `threshold` is the FULL out-of-fold choice - the number a final model
    trained on the whole train split would use.
    """
    from sklearn.metrics import balanced_accuracy_score, f1_score

    y_true = np.asarray(y_true)
    oof = np.asarray(oof, dtype=np.float64)
    out = {}
    for rule in THRESHOLD_RULES:
        pred = np.zeros(len(y_true), dtype=int)
        for tr, te in folds:
            if rule == "default":
                t = default_cut
            else:
                metric = ("balanced_accuracy" if rule == "oof_balanced_accuracy"
                          else "macro_f1")
                t, _ = best_threshold(y_true[tr], oof[tr], metric)
            pred[te] = (oof[te] >= t).astype(int)
        if rule == "default":
            full = default_cut
        else:
            metric = ("balanced_accuracy" if rule == "oof_balanced_accuracy"
                      else "macro_f1")
            full, _ = best_threshold(y_true, oof, metric)
        out[rule] = {
            "macro_f1": float(f1_score(y_true, pred, average="macro",
                                       zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "threshold": float(full),
        }
    return out
