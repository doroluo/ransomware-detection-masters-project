"""
Unit tests for `llm_features_pipeline/tuning.py` and the parts of
`llm_features_pipeline/tune.py` that are not a model fit - the pieces the tuned
expC/expD runs added on top of the committed pipeline.

What is pinned here, and why each one matters to a number in
`results/exp*_tuned/metrics.json`:

1. **The sequence budget and the positional sampler.** `count_lines` over raw
   bytes has to agree with iterating the file, including the no-trailing-newline
   case, because the strided sampler chooses its line indices from that count.
   `strided_indices` has to hit both endpoints and degenerate to "read
   everything" below budget, and `even_subsample` has to be a strided sample of
   a strided sample - that identity is what lets the token cache store ONE
   50,000-line sample per file and derive every smaller strided budget from it
   instead of re-reading 645M lines.

2. **The n-gram count matrix and the per-fold TF-IDF.** The whole search runs on
   an integer-key count matrix built once, with the vectoriser re-derived per
   fold. That is only legitimate if it equals
   `TfidfVectorizer(...).fit(train).transform(all)` exactly, so it is asserted
   against sklearn's own vectoriser here, min_df / max_features / sublinear_tf
   included.

3. **The poolings.** Mean, IDF-weighted mean and mean+max are computed as sparse
   products against the count matrix rather than by iterating 50,000 token
   vectors. Asserted against the literal loop.

4. **The group folds.** No group may appear on both sides of a fold. The folds
   hold out whole ransomware families, which is the shape of the real test
   split; a fold that leaks a family makes every CV number in `cv_search.csv`
   an overestimate.

5. **The threshold rules.** `best_threshold` maximises what it says it does, and
   `nested_threshold_scores` chooses fold k's cut from the OTHER folds' scores
   only - the property that makes the three operating-point rules comparable
   rather than one of them being scored on the points it was fitted to.

6. **The architecture reweighting.** `class` reproduces sklearn's `balanced`,
   and `class_arch` equalises the architectures INSIDE each class, which is the
   x64-ransomware reweighting expD needed.

Runs under the repo's own Python 3.14 as well as the 3.12 venv: nothing here
imports gensim, transformers or tokenizers, and the two sklearn/scipy-backed
groups are guarded with `importorskip`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

np = pytest.importorskip("numpy")

from llm_features_pipeline import tuning as T  # noqa: E402


# ---------------------------------------------------------------------------
# 1. sequence budget and positional sampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("body,expect", [
    ("a\nb\nc\n", 3),
    ("a\nb\nc", 3),       # no trailing newline: the last line still counts
    ("", 0),
    ("\n", 1),
    ("a\n\nb\n", 3),      # blank lines are lines
])
def test_count_lines_matches_iteration(tmp_path, body, expect):
    p = tmp_path / "f.txt"
    p.write_text(body, encoding="utf-8", newline="")
    assert T.count_lines(p) == expect
    with p.open(encoding="utf-8") as fh:
        assert sum(1 for _ in fh) == expect


def test_strided_indices_endpoints_and_degeneracy():
    # Below budget: every line, so `strided` and `head` are the same file.
    assert T.strided_indices(7, 10).tolist() == list(range(7))
    assert T.strided_indices(10, 10).tolist() == list(range(10))
    idx = T.strided_indices(1000, 5)
    assert idx[0] == 0 and idx[-1] == 999          # both ends are seen
    assert len(idx) == 5
    gaps = np.diff(idx)
    assert gaps.max() - gaps.min() <= 1            # evenly spread
    assert T.strided_indices(0, 10).size == 0
    assert T.strided_indices(10, 0).size == 0


def test_even_subsample_of_a_strided_sample_is_strided():
    """The identity the token cache depends on.

    The cache stores one 50,000-line strided sample per file; every smaller
    strided budget is derived from it by `even_subsample` rather than by
    re-reading the file. That is only sound if thinning a strided sample gives
    (nearly) the same line indices as striding the original file directly.
    """
    n = 1_000_000
    big = T.strided_indices(n, 50_000)
    thinned = T.even_subsample(big, 500)
    direct = T.strided_indices(n, 500)
    assert len(thinned) == len(direct) == 500
    # Same positions to within the rounding of two integer linspaces.
    assert np.abs(np.asarray(thinned) - direct).max() <= n / 500
    assert thinned[0] == direct[0] == 0
    assert thinned[-1] == direct[-1] == n - 1
    # Below budget it is a no-op.
    assert T.even_subsample([1, 2, 3], 10) == [1, 2, 3]


def test_read_tokens_head_and_strided(tmp_path):
    p = tmp_path / "mn.txt"
    p.write_text("".join(f"m{i}\n" for i in range(100)), encoding="utf-8")

    def normalize(line):
        return line.strip().lower() or None

    head = T.read_tokens(p, normalize, 10, "head")
    assert head == [f"m{i}" for i in range(10)]

    strided = T.read_tokens(p, normalize, 10, "strided")
    assert len(strided) == 10
    assert strided[0] == "m0" and strided[-1] == "m99"
    # The point of the sampler: it reaches the end of the file, head does not.
    assert "m99" not in head

    # Under budget the two coincide exactly.
    assert (T.read_tokens(p, normalize, 500, "head")
            == T.read_tokens(p, normalize, 500, "strided"))

    with pytest.raises(ValueError):
        T.read_tokens(p, normalize, 10, "reservoir")


def test_read_tokens_head_skips_blank_lines_without_spending_budget(tmp_path):
    p = tmp_path / "gaps.txt"
    p.write_text("mov\n\n\npush\n\ncall\n", encoding="utf-8")

    def normalize(line):
        return line.strip().lower() or None

    assert T.read_tokens(p, normalize, 3, "head") == ["mov", "push", "call"]


# ---------------------------------------------------------------------------
# 2. n-gram keys, count matrix and per-fold TF-IDF
# ---------------------------------------------------------------------------


def test_ngram_keys_are_a_bijection():
    ids = np.array([1, 2, 3, 1, 2], dtype=np.int32)
    assert T.ngram_keys(ids, 1, 10).tolist() == [1, 2, 3, 1, 2]
    assert T.ngram_keys(ids, 2, 10).tolist() == [12, 23, 31, 12]
    assert T.ngram_keys(ids, 3, 10).tolist() == [123, 231, 312]
    # Shorter than the window: no n-grams, not an error.
    assert T.ngram_keys(np.array([1], dtype=np.int32), 3, 10).size == 0
    with pytest.raises(ValueError):
        T.ngram_keys(ids, 0, 10)


def test_doc_ngram_counts_keeps_orders_apart():
    ids = np.array([1, 1, 1], dtype=np.int32)
    keys, counts = T.doc_ngram_counts(ids, (1, 2), vocab=10)
    # unigram `1` three times, bigram `1,1` twice - and the bigram key is
    # offset past the whole unigram key space, so they cannot collide.
    assert sorted(zip(keys.tolist(), counts.tolist())) == [(1, 3), (10 + 11, 2)]


def _tfidf_reference(docs, train_rows, ngram_range, min_df, max_features):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+",
                          ngram_range=ngram_range, min_df=min_df,
                          sublinear_tf=True, max_features=max_features)
    text = [" ".join(d) for d in docs]
    vec.fit([text[i] for i in train_rows])
    return vec.transform(text)


def test_fold_tfidf_equals_sklearn_fitted_on_train_rows():
    """The load-bearing equivalence of the whole search.

    `build_count_matrix` + `fold_tfidf` exist so the n-grams are counted once
    and the vectoriser is re-derived per fold from that. If it is not exactly
    `TfidfVectorizer.fit(train).transform(all)`, every CV number is measuring
    something other than what the summary says it is.
    """
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    vocab = ["mov", "push", "call", "add", "xor", "int3", "ret"]
    docs_ids = [rng.integers(0, len(vocab), size=int(rng.integers(20, 60)))
                .astype(np.int32) for _ in range(40)]
    docs = [[vocab[i] for i in d] for d in docs_ids]
    train_rows = np.arange(28)

    for ngram_range in [(1, 1), (1, 2), (1, 3)]:
        for min_df in (1, 2, 5):
            counts, _ = T.build_count_matrix(docs_ids, ngram_range, len(vocab),
                                             min_df_global=min_df)
            X, keep = T.fold_tfidf(counts, train_rows, min_df=min_df,
                                   sublinear=True)
            ref = _tfidf_reference(docs, train_rows, ngram_range, min_df, None)
            assert X.shape[0] == ref.shape[0]
            assert X.shape[1] == ref.shape[1], (ngram_range, min_df)
            # Column ORDER differs (integer key order vs sklearn's lexical
            # order), so compare the multiset of values in each row.
            a = np.sort(X.toarray(), axis=1)
            b = np.sort(ref.toarray(), axis=1)
            assert np.allclose(a, b, atol=1e-5), (ngram_range, min_df)


def test_fold_tfidf_max_features_matches_sklearn():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(7)
    vocab = [f"m{i}" for i in range(12)]
    docs_ids = [rng.integers(0, len(vocab), size=40).astype(np.int32)
                for _ in range(30)]
    docs = [[vocab[i] for i in d] for d in docs_ids]
    train_rows = np.arange(20)
    counts, _ = T.build_count_matrix(docs_ids, (1, 2), len(vocab),
                                     min_df_global=2)
    X, keep = T.fold_tfidf(counts, train_rows, min_df=2, sublinear=True,
                           max_features=25)
    assert X.shape[1] == 25 == int(keep.sum())
    ref = _tfidf_reference(docs, train_rows, (1, 2), 2, 25)
    assert X.shape == ref.shape

    # Both keep "the 25 most frequent of the terms that survive min_df on the
    # training rows", so the multiset of kept training term frequencies must
    # match exactly. WHICH term sits on the frequency boundary need not:
    # sklearn breaks ties with an unstable argsort and this implementation with
    # a stable one, and on a tie that is an arbitrary choice in both.
    tr = counts[train_rows].tocsc()
    tf = np.asarray(tr.sum(axis=0)).ravel()
    df = np.diff(tr.indptr)
    survivors = sorted(tf[df >= 2], reverse=True)
    assert sorted(tf[keep], reverse=True) == survivors[:25]
    assert df[keep].min() >= 2                     # min_df respected


def test_fold_tfidf_never_uses_a_held_out_row():
    """A term that occurs ONLY outside `train_rows` must not become a column."""
    pytest.importorskip("scipy")
    docs = [np.array([0, 0, 1], dtype=np.int32) for _ in range(6)]
    docs.append(np.array([2, 2, 2], dtype=np.int32))   # token 2 is test-only
    counts, keys = T.build_count_matrix(docs, (1, 1), 3, min_df_global=1)
    X, keep = T.fold_tfidf(counts, np.arange(6), min_df=1)
    assert set(keys[keep].tolist()) == {0, 1}
    assert X[6].nnz == 0          # the held-out row transforms to nothing


def test_global_min_df_prune_cannot_change_what_the_train_fit_keeps():
    """The one operation in `tune.py --stage final` that sees a test row.

    `_features_for` counts n-grams over TRAIN AND TEST documents together and
    prunes at global document frequency `min_df_global`, because the unpruned
    trigram space at a 50,000-token budget does not fit in memory. It is only
    legitimate because a column's training document frequency can never exceed
    its global one, so every column the train-only fit would keep survives the
    global prune and the fitted vocabulary, IDF and `max_features` cut are
    identical either way. Asserted here rather than argued: the same documents
    are put through both routes and the resulting matrices must match.
    """
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(23)
    V = 9
    docs = [rng.integers(0, V, size=int(rng.integers(15, 50))).astype(np.int32)
            for _ in range(50)]
    train_rows = np.arange(34)          # 34 train, 16 held out

    for ngram in [(1, 1), (1, 2), (1, 3)]:
        for min_df in (2, 5):
            # Route A: the search's route - count TRAIN rows only.
            train_counts, tkeys = T.build_count_matrix(
                [docs[i] for i in train_rows], ngram, V, min_df_global=min_df)
            Xa, keep_a = T.fold_tfidf(train_counts, np.arange(len(train_rows)),
                                      min_df=min_df, sublinear=True)
            # Route B: the final stage's route - count EVERY row, prune
            # globally, then fit the vectoriser on the train rows.
            all_counts, akeys = T.build_count_matrix(docs, ngram, V,
                                                     min_df_global=min_df)
            Xb, keep_b = T.fold_tfidf(all_counts, train_rows, min_df=min_df,
                                      sublinear=True)
            assert (sorted(tkeys[keep_a].tolist())
                    == sorted(akeys[keep_b].tolist())), (ngram, min_df)
            # ...and the train rows get the identical numbers, not merely the
            # identical column set.
            assert np.allclose(Xa.toarray(), Xb[train_rows].toarray(),
                               atol=1e-6), (ngram, min_df)


# ---------------------------------------------------------------------------
# 3. pooling
# ---------------------------------------------------------------------------


def _counts_matrix(docs, V):
    from scipy import sparse
    m = np.zeros((len(docs), V))
    for i, d in enumerate(docs):
        for t in d:
            m[i, t] += 1
    return sparse.csr_matrix(m)


def test_poolings_match_the_literal_loop():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(3)
    V, dim = 6, 4
    W = rng.normal(size=(V, dim))
    docs = [rng.integers(0, V, size=int(rng.integers(3, 30))) for _ in range(9)]
    counts = _counts_matrix(docs, V)

    mean = T.pool_mean(counts, W)
    for i, d in enumerate(docs):
        assert np.allclose(mean[i], np.mean([W[t] for t in d], axis=0))

    idf = rng.uniform(0.5, 3.0, size=V)
    wm = T.pool_weighted_mean(counts, W, idf)
    for i, d in enumerate(docs):
        num = np.sum([idf[t] * W[t] for t in d], axis=0)
        assert np.allclose(wm[i], num / np.sum([idf[t] for t in d]))

    mm = T.pool_mean_max(counts, W)
    assert mm.shape == (len(docs), 2 * dim)
    for i, d in enumerate(docs):
        assert np.allclose(mm[i, :dim], mean[i])
        # max is over the token TYPES present, with no multiplicity
        assert np.allclose(mm[i, dim:], W[sorted(set(d.tolist()))].max(axis=0))


def test_pool_mean_of_an_empty_document_is_zero_not_nan():
    pytest.importorskip("scipy")
    from scipy import sparse
    W = np.ones((3, 2))
    counts = sparse.csr_matrix(np.zeros((1, 3)))
    assert np.allclose(T.pool_mean(counts, W), 0.0)


# ---------------------------------------------------------------------------
# 4. group folds
# ---------------------------------------------------------------------------


def test_group_folds_never_split_a_group():
    pytest.importorskip("sklearn")
    # 6 ransomware families of 20 files each, 120 one-file goodware groups.
    y = np.array([1] * 120 + [0] * 120)
    groups = np.array([f"fam{i // 20}" for i in range(120)]
                      + [f"file:{i}" for i in range(120)])
    folds = T.group_folds(y, groups, n_splits=5, seed=42)
    assert len(folds) == 5
    seen = set()
    for tr, te in folds:
        assert not (set(groups[tr]) & set(groups[te])), "group on both sides"
        assert len(te) > 0
        seen |= set(te.tolist())
    assert seen == set(range(len(y)))            # every row held out exactly once
    # Whole families are held out: a fold that holds any member of a family
    # holds all of it.
    for tr, te in folds:
        for fam in {g for g in groups[te] if g.startswith("fam")}:
            assert (groups[te] == fam).sum() == 20


def test_group_folds_are_reproducible():
    pytest.importorskip("sklearn")
    y = np.array([1] * 60 + [0] * 60)
    groups = np.array([f"fam{i // 10}" for i in range(60)]
                      + [f"file:{i}" for i in range(60)])
    a = T.group_folds(y, groups, 4, seed=42)
    b = T.group_folds(y, groups, 4, seed=42)
    for (a_tr, a_te), (b_tr, b_te) in zip(a, b):
        assert a_te.tolist() == b_te.tolist()


# ---------------------------------------------------------------------------
# 5. thresholds
# ---------------------------------------------------------------------------


def test_best_threshold_finds_the_separating_cut():
    pytest.importorskip("sklearn")
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr, f1 = T.best_threshold(y, scores, "macro_f1")
    assert f1 == pytest.approx(1.0)
    assert 0.3 < thr <= 0.7


def test_best_threshold_is_invariant_to_monotone_rescaling():
    """A decision_function and a predict_proba must give the same partition."""
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, size=200)
    s = rng.normal(size=200) + y
    t1, f1 = T.best_threshold(y, s, "macro_f1")
    t2, f2 = T.best_threshold(y, 1 / (1 + np.exp(-s)), "macro_f1")
    assert f1 == pytest.approx(f2)
    assert ((s >= t1).astype(int) == (1 / (1 + np.exp(-s)) >= t2).astype(int)).all()


def test_best_threshold_optimises_balanced_accuracy_when_asked():
    pytest.importorskip("sklearn")
    from sklearn.metrics import balanced_accuracy_score
    rng = np.random.default_rng(5)
    y = np.array([1] * 180 + [0] * 20)          # heavily skewed, as the test split is
    s = rng.normal(size=200) + y * 1.5
    thr, val = T.best_threshold(y, s, "balanced_accuracy")
    assert balanced_accuracy_score(y, (s >= thr).astype(int)) == pytest.approx(val)
    grid = np.quantile(s, np.linspace(0, 1, 50))
    assert val + 1e-12 >= max(
        balanced_accuracy_score(y, (s >= t).astype(int)) for t in grid)


def test_nested_threshold_uses_only_the_other_folds():
    """The cut applied to a fold must not have seen that fold's scores.

    Planted: fold 0's held-out rows are the only ones whose optimal cut is
    unusual. If the implementation chose the threshold on all of the scores, the
    nested `oof_macro_f1` number would equal the non-nested one; it must not.
    """
    pytest.importorskip("sklearn")
    y = np.array([0] * 20 + [1] * 20)
    scores = np.concatenate([np.linspace(-2, -1, 20), np.linspace(1, 2, 20)])
    # Fold 0 holds rows where the classes are swapped in score order.
    scores[:5] = np.linspace(3, 4, 5)
    idx = np.arange(40)
    folds = [(np.setdiff1d(idx, blk), blk)
             for blk in np.array_split(idx, 4)]
    out = T.nested_threshold_scores(y, scores, folds, default_cut=0.0)
    assert set(out) == set(T.THRESHOLD_RULES)
    non_nested_thr, non_nested_f1 = T.best_threshold(y, scores, "macro_f1")
    assert out["oof_macro_f1"]["macro_f1"] < non_nested_f1
    # The reported full-data threshold is still the non-nested one - that is
    # the cut a model trained on the whole train split would use.
    assert out["oof_macro_f1"]["threshold"] == pytest.approx(non_nested_thr)
    assert out["default"]["threshold"] == 0.0


def test_nested_threshold_default_rule_fits_nothing():
    pytest.importorskip("sklearn")
    from sklearn.metrics import f1_score
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    scores = np.array([-1.0, 0.5, 2.0, -0.2, -3.0, 1.0, 0.1, 0.9])
    idx = np.arange(8)
    folds = [(np.setdiff1d(idx, b), b) for b in np.array_split(idx, 4)]
    out = T.nested_threshold_scores(y, scores, folds, default_cut=0.0)
    assert out["default"]["macro_f1"] == pytest.approx(
        f1_score(y, (scores >= 0.0).astype(int), average="macro"))


# ---------------------------------------------------------------------------
# 6. architecture-aware reweighting (tune.py, but no heavy imports on the path)
# ---------------------------------------------------------------------------


def test_arch_sample_weight_reproduces_sklearn_balanced():
    pytest.importorskip("sklearn")
    pytest.importorskip("pandas")
    pytest.importorskip("yaml")
    from sklearn.utils.class_weight import compute_sample_weight
    from llm_features_pipeline import tune as X

    y = np.array([1] * 90 + [0] * 10)
    arch = np.array(["x86"] * 100)
    assert np.allclose(X.arch_sample_weight(y, arch, "none"), 1.0)
    assert np.allclose(X.arch_sample_weight(y, arch, "class"),
                       compute_sample_weight("balanced", y))


def test_arch_sample_weight_lifts_the_rare_architecture_inside_its_class():
    """expD's actual shape: 862 x86 and 42 x64 ransomware in train.

    `class_arch` has to make the 42 x64 positives weigh as much in aggregate as
    the 862 x86 ones, without changing the balance between the two classes.
    """
    pytest.importorskip("pandas")
    pytest.importorskip("yaml")
    from llm_features_pipeline import tune as X

    y = np.array([1] * 904 + [0] * 1112)
    arch = np.array(["x86"] * 862 + ["x64"] * 42
                    + ["x64"] * 913 + ["x86"] * 199)
    w = X.arch_sample_weight(y, arch, "class_arch")
    ran, good = y == 1, y == 0
    # Class totals are untouched by the architecture adjustment.
    assert w[ran].sum() == pytest.approx(w[good].sum())
    # Inside the ransomware class the two architectures now carry equal mass...
    x86 = ran & (arch == "x86")
    x64 = ran & (arch == "x64")
    assert w[x86].sum() == pytest.approx(w[x64].sum())
    # ...which means each of the 42 x64 files weighs ~20x an x86 one.
    assert w[x64][0] / w[x86][0] == pytest.approx(862 / 42, rel=1e-6)

    with pytest.raises(ValueError):
        X.arch_sample_weight(y, arch, "by_family")
