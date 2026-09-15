"""What the mnemonic sequence model promises, asserted.

The six changes `seq_model/` makes to the CNN-ViT setup are each a claim about
the DATA path, not about accuracy, and each of them is checkable without a GPU:

  * the model vocabulary is fitted on the training files and nothing else;
  * windows are strided across the WHOLE stream and never exceed the cap;
  * padding is structurally unreadable - junk written into the padded region
    of a window does not move the pooled representation at all;
  * the sampler equalises the four (label, arch) cells;
  * the imports side-input hashes deterministically and reaches the head;
  * the index sets the runner hands the model satisfy the same fold invariants
    `family_holdout/common.py` asserts for every other runner.

Everything that needs torch is behind `importorskip`, and everything else runs
on a synthetic six-file corpus built in tmp_path, so the file is fast and does
not need the 1.9 GB real token cache.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from seq_model import config as CFG                       # noqa: E402
from seq_model import data as D                           # noqa: E402

OUT = REPO / "results" / "family_holdout"
DATASETS = ["mendeley", "balanced"]


# ---------------------------------------------------------------------------
# a synthetic corpus
# ---------------------------------------------------------------------------
TRAIN_ONLY = "vzeroupper"
TEST_ONLY = "cpuid"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Six files: four 'training', two 'test'. Two mnemonics are exclusive to
    one side, which is what makes the vocabulary test meaningful."""
    root = tmp_path_factory.mktemp("mn")
    cache = tmp_path_factory.mktemp("cache")
    rng = np.random.default_rng(0)
    base = ["push", "mov", "call", "ret", "jmp", "test", "je", "xor"]
    shas, lengths = [], {}
    for i in range(6):
        sha = f"{i:064x}"
        n = int(rng.integers(50, 40000))
        toks = list(rng.choice(base, size=n))
        toks.append(TRAIN_ONLY if i < 4 else TEST_ONLY)
        # two sections, to prove they are concatenated into one stream
        half = len(toks) // 2
        (root / f"{sha}.txt").write_text(
            " ".join(toks[:half]) + "\n" + " ".join(toks[half:]) + "\n",
            encoding="utf-8")
        shas.append(sha)
        lengths[sha] = len(toks)
    old = D.MN_DIRS
    D.MN_DIRS = (root,)
    try:
        D.build_cache(shas, cache_root=cache, workers=1, log=lambda *a: None)
    finally:
        D.MN_DIRS = old
    return {"root": root, "cache": cache, "shas": shas, "lengths": lengths,
            "train": shas[:4], "test": shas[4:]}


def test_cache_is_the_concatenated_stream(corpus):
    """One file's cached array is exactly its whitespace tokens, both sections,
    with no separator invented between them."""
    sha = corpus["shas"][0]
    ids = np.asarray(D.load_ids(sha, corpus["cache"]))
    text = (corpus["root"] / f"{sha}.txt").read_text(encoding="utf-8")
    assert len(ids) == len(text.split()) == corpus["lengths"][sha]
    vocab = D.global_vocab(corpus["cache"])
    assert [vocab[i] for i in ids[:5]] == text.split()[:5]
    assert D.cache_lengths(corpus["cache"])[sha] == len(ids)


# ---------------------------------------------------------------------------
# 1. the vocabulary is fitted on the training files only
# ---------------------------------------------------------------------------
def test_vocab_is_fitted_on_training_files_only(corpus):
    v = D.Vocab.fit(corpus["train"], corpus["cache"])
    assert v.tokens[D.PAD_ID] == "<pad>" and v.tokens[D.UNK_ID] == "<unk>"
    assert TRAIN_ONLY in v.tokens, "a training mnemonic is missing from the vocab"
    assert TEST_ONLY not in v.tokens, "a test-only mnemonic leaked into the vocab"
    assert v.size == len(set(v.tokens)), "duplicate entries in the vocabulary"


def test_unseen_mnemonics_become_unk_at_encode_time(corpus):
    v = D.Vocab.fit(corpus["train"], corpus["cache"])
    g = D.global_vocab(corpus["cache"])
    gid_test_only = g.index(TEST_ONLY)
    gid_train_only = g.index(TRAIN_ONLY)
    assert int(v.encode(np.array([gid_test_only]))[0]) == D.UNK_ID
    assert int(v.encode(np.array([gid_train_only]))[0]) != D.UNK_ID
    # and a whole test file encodes with no id outside the fitted range
    enc = v.encode(np.asarray(D.load_ids(corpus["test"][0], corpus["cache"])))
    assert enc.min() >= 0 and enc.max() < v.size
    assert (enc == D.UNK_ID).sum() == 1, "exactly the one unseen mnemonic is <unk>"


def test_vocab_fit_is_a_function_of_the_training_set(corpus):
    a = D.Vocab.fit(corpus["train"], corpus["cache"])
    b = D.Vocab.fit(corpus["shas"], corpus["cache"])
    assert b.size == a.size + 1 and a.size < b.size


# ---------------------------------------------------------------------------
# 2. windowing covers the whole stream and never exceeds N
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("W,N", [(16, 4), (4096, 16), (1024, 8), (21840, 1)])
def test_window_offsets_never_exceed_N(W, N):
    rng = np.random.default_rng(1)
    for n in list(rng.integers(1, 400000, size=200)) + [1, W - 1, W, W + 1, W * N]:
        offs = D.window_offsets(int(n), W, N)
        assert 1 <= len(offs) <= N
        assert offs[0] == 0, "the first window always starts at the head"
        assert (np.diff(offs) > 0).all(), "offsets must be strictly increasing"
        assert offs[-1] + W >= min(n, offs[-1] + W)
        assert offs[-1] <= max(0, n - W), "a window starts past the usable range"


@pytest.mark.parametrize("W", [16, 128, 4096])
def test_windows_cover_the_whole_stream_when_not_capped(W):
    """With enough windows allowed, their union is the entire stream - the
    property the image encoder's first-21,845-instructions canvas lacked."""
    rng = np.random.default_rng(2)
    for n in list(rng.integers(1, W * 40, size=120)):
        n = int(n)
        need = -(-n // W)
        offs = D.window_offsets(n, W, N=need)          # uncapped
        covered = np.zeros(n, dtype=bool)
        for o in offs:
            covered[o:o + W] = True
        assert covered.all(), f"gap in coverage for n={n}, W={W}"
        assert offs[-1] + W >= n, "the tail of the stream is never reached"


def test_capped_windowing_still_reaches_the_tail():
    """Capped at N, the windows are spread over the file rather than truncating
    it: the LAST window still ends at the end of the stream."""
    n, W, N = 10_000_000, 4096, 16
    offs = D.window_offsets(n, W, N)
    assert len(offs) == N
    assert offs[0] == 0 and offs[-1] + W == n
    assert len(set(offs)) == N


def test_first_window_only_ablation_is_the_head_of_the_file():
    cfg = CFG.ablation(CFG.load(), "first_window_only")
    W, N = cfg["windows"]["window"], cfg["windows"]["n_windows"]
    assert N == 1 and W % 16 == 0 and abs(W - 21845) <= 16
    assert D.window_offsets(10 ** 7, W, N).tolist() == [0]


def test_file_windows_shape_and_lengths(corpus):
    v = D.Vocab.fit(corpus["train"], corpus["cache"])
    for sha in corpus["shas"]:
        n = corpus["lengths"][sha]
        w, lens = D.file_windows(sha, v, 1024, 8, corpus["cache"])
        assert w.shape[1] == 1024 and w.shape[0] == len(lens) <= 8
        assert (lens > 0).all() and (lens <= 1024).all()
        assert lens.max() == min(n, 1024)
        # windows may overlap (that is how the even spread reaches the tail),
        # so the only bound on the total is the bag size
        assert lens.sum() <= len(lens) * 1024
        # every position past `lens` is exactly PAD
        for i, L in enumerate(lens):
            assert (w[i, L:] == D.PAD_ID).all()
        assert (w[:, 0] != D.PAD_ID).all()


# ---------------------------------------------------------------------------
# 3. padding is structurally unreadable
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def torch_mod():
    return pytest.importorskip("torch", reason="torch not installed")


def _tiny_model(torch, window=256, vocab=40, **kw):
    from seq_model.model import SeqTransformer
    torch.manual_seed(0)
    m = SeqTransformer(vocab, window=window, dim=64, depth=2, heads=4,
                       mlp_dim=64, emb_dim=16, dropout=0.0, **kw)
    return m.eval()


def _padded_batch(torch, lens, window, vocab, seed=0):
    g = torch.Generator().manual_seed(seed)
    lens = torch.as_tensor(lens, dtype=torch.long)
    B, M = lens.shape
    pos = torch.arange(window).view(1, 1, window)
    ids = torch.randint(D.N_SPECIAL, vocab, (B, M, window), generator=g)
    ids = torch.where(pos < lens.unsqueeze(-1), ids, torch.zeros_like(ids))
    return ids, lens, pos


def test_padded_positions_cannot_change_the_pooled_output(torch_mod):
    """The claim that killed the image encoder's constant pad: whatever sits in
    the padded region, the file's representation is identical."""
    torch = torch_mod
    from seq_model.model import RECEPTIVE_FIELD
    W, V = 256, 40
    m = _tiny_model(torch, W, V)
    lens = [[200, 256, 33], [256, RECEPTIVE_FIELD, 64]]
    ids, lens_t, pos = _padded_batch(torch, lens, W, V)
    win = torch.tensor([[True, True, True], [True, True, False]])
    with torch.no_grad():
        a, wa = m(ids, lens_t, win)
        for seed in (1, 2, 3):
            junk = torch.randint(D.N_SPECIAL, V, ids.shape,
                                 generator=torch.Generator().manual_seed(seed))
            noisy = torch.where(pos < lens_t.unsqueeze(-1), ids, junk)
            b, wb = m(noisy, lens_t, win)
            assert torch.equal(a, b), "padded content reached the file logits"
            valid = win.unsqueeze(-1).expand_as(wa)
            assert torch.equal(wa[valid], wb[valid])


def test_short_windows_only_see_the_stems_own_receptive_field(torch_mod):
    """A window with fewer real tokens than one block position can see keeps
    that position anyway, so the guarantee weakens exactly there and nowhere
    else: content past the receptive field is still unreachable. Two of the
    3,846 cohort files are this short."""
    torch = torch_mod
    from seq_model.model import RECEPTIVE_FIELD
    W, V = 256, 40
    m = _tiny_model(torch, W, V)
    ids, lens, pos = _padded_batch(torch, [[3]], W, V, seed=11)
    win = torch.tensor([[True]])
    with torch.no_grad():
        a = m(ids, lens, win)[0]
        junk = torch.randint(D.N_SPECIAL, V, ids.shape,
                             generator=torch.Generator().manual_seed(12))
        far = torch.where(pos < RECEPTIVE_FIELD, ids, junk)
        assert torch.equal(a, m(far, lens, win)[0])


def test_a_masked_out_window_cannot_change_the_file_logits(torch_mod):
    """MIL aggregation ignores window slots that the file does not have."""
    torch = torch_mod
    W, V = 256, 40
    m = _tiny_model(torch, W, V)
    ids, lens, _ = _padded_batch(torch, [[128, 256, 90]], W, V, seed=5)
    with torch.no_grad():
        two = m(ids[:, :2], lens[:, :2], torch.tensor([[True, True]]))[0]
        three = m(ids, lens, torch.tensor([[True, True, False]]))[0]
    assert torch.allclose(two, three, atol=1e-6)


def test_a_file_scores_the_same_alone_as_in_a_batch(torch_mod):
    """No batch statistic anywhere: batching a short file with a long one does
    not move the short file's logits."""
    torch = torch_mod
    W, V = 256, 40
    m = _tiny_model(torch, W, V)
    ids, lens, _ = _padded_batch(torch, [[40, 1, 1], [256, 256, 256]], W, V, seed=7)
    win = torch.tensor([[True, False, False], [True, True, True]])
    with torch.no_grad():
        both = m(ids, lens, win)[0]
        alone = m(ids[:1], lens[:1], win[:1])[0]
    assert torch.allclose(both[0], alone[0], atol=1e-6)


def test_minpool_mask_marks_only_fully_valid_positions(torch_mod):
    torch = torch_mod
    from seq_model.model import minpool_mask
    m = torch.zeros(1, 64, dtype=torch.bool)
    m[0, :32] = True
    out = minpool_mask(m, 2, 2, 0)
    assert out.shape == (1, 32)
    assert out[0, :16].all() and not out[0, 16:].any()
    full = torch.ones(1, 64, dtype=torch.bool)
    assert minpool_mask(full, 3, 2, 1).all(), "structural conv padding is not content"


def test_pad_embedding_is_exactly_zero(torch_mod):
    torch = torch_mod
    m = _tiny_model(torch_mod, 256, 40)
    assert torch.equal(m.encoder.embed.weight[D.PAD_ID],
                       torch.zeros_like(m.encoder.embed.weight[0]))


# ---------------------------------------------------------------------------
# 4. the sampler balances the (label, arch) cells
# ---------------------------------------------------------------------------
def test_arch_balanced_weights_equalise_every_cell():
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 2, 4000)
    archs = np.where(rng.random(4000) < 0.95, "x86", "x64")
    w = D.arch_balanced_weights(labels, archs)
    mass = {(l, a): w[(labels == l) & (archs == a)].sum()
            for l in (0, 1) for a in ("x86", "x64")}
    assert abs(sum(mass.values()) - 1.0) < 1e-9
    assert max(mass.values()) - min(mass.values()) < 1e-9, mass


def test_class_balanced_weights_ignore_arch():
    labels = np.array([0] * 90 + [1] * 10)
    archs = np.array(["x86"] * 95 + ["x64"] * 5)
    w = D.class_balanced_weights(labels)
    assert abs(w[labels == 0].sum() - w[labels == 1].sum()) < 1e-12
    # the ablation's point: x64 gets strictly less mass than arch balancing
    # would give it. Three (label, arch) cells exist here, so arch balancing
    # puts a third of the mass on the x64 files and class balancing a tenth.
    bal = D.arch_balanced_weights(labels, archs)
    assert w[archs == "x64"].sum() < bal[archs == "x64"].sum()
    assert abs(bal[archs == "x64"].sum() - 1 / 3) < 1e-9
    assert abs(w[archs == "x64"].sum() - 0.25) < 1e-9


def test_sampler_draws_are_architecture_balanced():
    torch = pytest.importorskip("torch")
    from seq_model.train import make_sampler
    rng = np.random.default_rng(4)
    n = 2000
    labels = rng.integers(0, 2, n)
    archs = np.where(rng.random(n) < 0.95, "x86", "x64")
    for balanced, tol in ((True, 0.06), (False, None)):
        s = make_sampler(labels, archs, balanced, seed=1, n=20000)
        idx = np.array(list(iter(s)))
        share = {(l, a): float(((labels[idx] == l) & (archs[idx] == a)).mean())
                 for l in (0, 1) for a in ("x86", "x64")}
        if balanced:
            assert max(share.values()) - min(share.values()) < tol, share
        else:
            assert share[(1, "x64")] < 0.15, share
    assert len(idx) == 20000


# ---------------------------------------------------------------------------
# 5. the imports side-input
# ---------------------------------------------------------------------------
def test_hash_imports_is_deterministic_and_normalised():
    a = D.hash_imports(["CreateFileW", "WriteFile", "CryptEncrypt"], 2048)
    b = D.hash_imports(["createfilew", " WriteFile ", "CryptEncrypt"], 2048)
    assert a.shape == (2048,) and a.dtype == np.float32
    assert np.allclose(a, b), "hashing must not depend on case or whitespace"
    assert abs(np.linalg.norm(a) - 1.0) < 1e-5
    assert not np.allclose(a, D.hash_imports(["CreateFileW"], 2048))
    assert np.allclose(D.hash_imports([], 2048), 0.0)
    assert D.hash_imports(["CreateFileW"], 256).shape == (256,)


def test_load_imports_reads_the_synthetic_json(tmp_path):
    doc = {"a" * 64: ["CreateFileW", "WriteFile"], "b" * 64: ["RegOpenKeyExW"]}
    p = tmp_path / "imports.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    got = D.load_imports(p, 64)
    assert set(got) == set(doc)
    assert all(v.shape == (64,) for v in got.values())
    assert np.allclose(got["a" * 64], D.hash_imports(doc["a" * 64], 64))


def test_imports_reach_the_classifier_head(torch_mod):
    """The untested-on-real-data code path, exercised end to end on synthetic
    imports: widening the head, feeding a per-file vector, and changing the
    prediction when the imports change."""
    torch = torch_mod
    W, V, DIM = 256, 40, 64
    m = _tiny_model(torch, W, V, imports_dim=DIM)
    assert m.mlp_head[1].in_features == 64 + DIM
    ids, lens, _ = _padded_batch(torch, [[100, 256]], W, V, seed=9)
    win = torch.tensor([[True, True]])
    imp = torch.from_numpy(D.hash_imports(["CreateFileW", "CryptEncrypt"], DIM))[None]
    other = torch.from_numpy(D.hash_imports(["RegOpenKeyExW"], DIM))[None]
    with torch.no_grad():
        a = m(ids, lens, win, imp)[0]
        b = m(ids, lens, win, other)[0]
        z = m(ids, lens, win, None)[0]
    assert not torch.allclose(a, b), "the imports vector does not reach the head"
    assert torch.allclose(z, m(ids, lens, win,
                               torch.zeros_like(imp))[0], atol=1e-6)


def test_batches_have_a_constant_shape(corpus):
    """Every batch is padded to the configured window count, not to the batch's
    own maximum: a varying shape makes PyTorch's pinned-memory allocator cache
    a new block per shape and never release it."""
    pytest.importorskip("torch")
    from seq_model.train import Collate, FileWindowDataset
    v = D.Vocab.fit(corpus["train"], corpus["cache"])
    ds = FileWindowDataset(corpus["shas"], [0] * 6, v, 512, 6, corpus["cache"])
    c = Collate(ds.N, ds.W)
    shapes = set()
    for lo in (0, 2, 4):
        b = c([ds[i] for i in range(lo, lo + 2)])
        shapes.add(tuple(b["ids"].shape[1:]))
        assert b["ids"].dtype.itemsize == 4, "ids should be int32, not int64"
        assert b["win_mask"].shape == b["lens"].shape == (2, ds.N)
        # padded window slots are marked absent, whatever the file's length
        for i in range(2):
            k = int(b["win_mask"][i].sum())
            assert 1 <= k <= ds.N
            assert not b["win_mask"][i, k:].any()
    assert shapes == {(ds.N, ds.W)}, shapes


def test_imports_dataset_supplies_a_zero_vector_for_missing_files(corpus):
    pytest.importorskip("torch")
    from seq_model.train import FileWindowDataset, collate
    v = D.Vocab.fit(corpus["train"], corpus["cache"])
    ds = FileWindowDataset(corpus["shas"][:2], [0, 1], v, 256, 2,
                           corpus["cache"],
                           imports={corpus["shas"][0]: D.hash_imports(["X"], 32)},
                           imports_dim=32)
    b = collate([ds[0], ds[1]])
    assert b["imports"].shape == (2, 32)
    assert float(b["imports"][1].abs().sum()) == 0.0
    assert float(b["imports"][0].abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# 6. fold invariants, through family_holdout/common.py
# ---------------------------------------------------------------------------
pytestmark_folds = pytest.mark.skipif(
    not (OUT / "folds_mendeley.csv").is_file(),
    reason="run python family_holdout/folds.py --out results/family_holdout first")


@pytest.fixture(scope="module")
def folds():
    if not (OUT / "folds_mendeley.csv").is_file():
        pytest.skip("fold file missing")
    from family_holdout.common import Folds
    return {ds: Folds(ds) for ds in DATASETS}


@pytest.mark.parametrize("ds", DATASETS)
def test_runner_uses_the_shared_fold_invariants(folds, ds):
    from family_holdout.common import check_kfold, check_lofo
    rep = check_kfold(folds[ds])
    assert all(r["sha_overlap"] == 0 and not r["family_overlap"] for r in rep.values())
    assert len(check_lofo(folds[ds])) == len(folds[ds].families) == 38


@pytest.mark.parametrize("ds", DATASETS)
def test_val_carve_leaks_no_group_into_train_or_test(folds, ds):
    """The runner's own split, not just the fold file's: the early-stopping
    signal must not share a family or a goodware group with what it selects on."""
    pytest.importorskip("pandas")
    from seq_model.run_family_holdout import val_split
    f = folds[ds]
    for k, tr_all, te in list(f.kfold()) + list(f.lofo())[:3]:
        tr, va, te2 = val_split(f, tr_all, te)
        assert sorted(te2) == sorted(te)
        assert len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0
        assert len(tr) + len(va) == len(tr_all)
        g = lambda i: set(np.where(f.y[i] == 1, f.family[i], f.group[i]))  # noqa: E731
        assert not g(tr) & g(va)
        assert not (g(tr) | g(va)) & g(te)
        assert len(va) > 0


@pytest.mark.parametrize("ds", DATASETS)
def test_every_cohort_file_has_a_cached_token_stream(folds, ds):
    """The runner would otherwise discover this 40 minutes into a sweep."""
    cache = Path(CFG.load()["paths"]["token_cache"])
    if not (cache / "lengths.json").exists():
        pytest.skip("token cache not built (python seq_model/data.py)")
    have = D.cache_lengths(cache)
    missing = [s for s in folds[ds].sha if s not in have]
    assert not missing, f"{len(missing)} files have no cached stream, e.g. {missing[:3]}"


# ---------------------------------------------------------------------------
# the frozen configuration
# ---------------------------------------------------------------------------
def test_fallback_yaml_parser_agrees_with_pyyaml():
    yaml = pytest.importorskip("yaml")
    ref = yaml.safe_load(CFG.CONFIG_PATH.read_text(encoding="utf-8"))
    import builtins
    real = builtins.__import__

    def blocked(name, *a, **k):
        if name == "yaml":
            raise ImportError("blocked for the test")
        return real(name, *a, **k)

    builtins.__import__ = blocked
    try:
        mine = CFG.load_yaml()
    finally:
        builtins.__import__ = real
    assert mine == ref


def test_config_is_the_one_the_study_describes():
    c = CFG.load()
    assert c["model"]["depth"] == 6 and c["model"]["dim"] == 256
    assert c["model"]["heads"] == 8 and c["model"]["mlp_dim"] == 512
    assert c["windows"]["window"] % 16 == 0
    assert c["windows"]["window"] // 16 == 256, "the ViT must see 256 positions"
    assert c["sampler"]["arch_balanced"] is True
    assert c["imports"]["enabled"] is False, "the committed run has no imports"
    assert c["vocabulary"]["fit_on"] == "training_folds_only"
    assert c["evaluation"]["decision"] == "argmax_0.5"
    assert len(c["seeds"]["kfold"]) >= 3 and c["seeds"]["lofo"] == [1]


@pytest.mark.parametrize("name", CFG.ABLATIONS)
def test_each_ablation_changes_exactly_one_thing(name):
    base = CFG.load()
    ab = CFG.ablation(base, name)
    flat = lambda d, p="": {  # noqa: E731
        f"{p}{k}": v for k, v in d.items() if not isinstance(v, dict)} | {
        kk: vv for k, v in d.items() if isinstance(v, dict)
        for kk, vv in flat(v, f"{p}{k}.").items()}
    assert ab["seeds"]["kfold"] == [base["seeds"]["kfold"][0]], \
        "every ablation is a one-seed study, at the main run's first seed"
    assert ab["seeds"]["lofo"] == [], "ablations do not run LOFO"
    a, b = flat(base), flat(ab)
    # the seed list is budget, not configuration: it changes how many runs are
    # averaged, never what a run computes (it is not in the fingerprint)
    diff = ({k for k in set(a) | set(b) if a.get(k) != b.get(k)}
            - {"ablation", "seeds.kfold", "seeds.lofo"})
    if name == "first_window_only":
        assert diff == {"windows.window", "windows.n_windows",
                        "training.batch_size", "training.eval_batch_size"}
    elif name == "no_pretrain":
        assert diff == {"pretrain.enabled"}
    else:
        assert diff == {"sampler.arch_balanced"}


def test_fingerprint_tracks_what_changes_the_result():
    from seq_model.run_family_holdout import fingerprint
    base = CFG.load()
    assert fingerprint(base) == fingerprint(CFG.load())
    for name in CFG.ABLATIONS:
        assert fingerprint(CFG.ablation(base, name)) != fingerprint(base)
    spun = CFG.load()
    spun["training"]["num_workers"] = 99
    spun["training"]["eval_workers"] = 99
    assert fingerprint(spun) == fingerprint(base), \
        "dataloader worker count must not invalidate finished runs"
