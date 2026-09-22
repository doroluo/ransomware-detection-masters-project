"""
Unit tests for the tokenization step and the pipeline's own data loader.

Two things are pinned here:

1. `normalize_instruction` from the tokenization repo (imported through the
   path in `llm_features_pipeline/config.yaml`, never from a copy) against a
   hand-written instruction snippet with the tokens written out by hand. This
   is the check the plan's Task 3 asks for: the normalizer is the single point
   where two different disassemblers' output has to converge, so a silent
   change to it invalidates every experiment in `results/`.

2. `llm_features_pipeline/data.py`'s loader: deterministic sorted order,
   filename carried, empty files counted as dropped, family used as a split
   group and never as a label, label always 0 or 1.

Runs under the repo's own Python (3.14) as well as the 3.12 venv: nothing here
imports gensim or transformers. The one external dependency,
`Tokenization.tokenization`, needs pandas + tokenizers only, and is guarded
with `importorskip` so a bare interpreter skips rather than errors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from llm_features_pipeline import data as D  # noqa: E402

CONFIG = REPO / "llm_features_pipeline" / "config.yaml"


# --------------------------------------------------------------------------
# normalize_instruction
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def normalize():
    """The real `normalize_instruction`, reached through config.yaml's
    `paths.tokenization_repo` so the test cannot drift onto a second copy."""
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    repo = Path(cfg["paths"]["tokenization_repo"])
    if not repo.is_dir():
        pytest.skip(f"tokenization repo not present: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    pytest.importorskip("tokenizers")
    pytest.importorskip("pandas")
    from Tokenization.tokenization import normalize_instruction
    return normalize_instruction


# (raw line, exact expected token) - written by hand, not generated.
CASES = [
    # 0x hex immediates and displacements become <HEX>
    ("mov eax, 0x10", "mov eax <HEX>"),
    ("call 0x401000", "call <HEX>"),
    # bare decimals become <OFFSET>; digits inside a mnemonic or register are
    # not word-bounded and must survive
    ("add esp, 8", "add esp <OFFSET>"),
    ("cvtsi2sd xmm0, eax", "cvtsi2sd xmm0 eax"),
    ("vbroadcasti32x4 zmm1, zmmword ptr [rax]", "vbroadcasti32x4 zmm1 zmmword ptr [rax]"),
    # bracket spacing is collapsed, then operator spacing
    ("mov eax, dword ptr [ ebp - 8 ]", "mov eax dword ptr [ebp-<OFFSET>]"),
    ("mov eax, dword ptr [ebp+0x1c]", "mov eax dword ptr [ebp+<HEX>]"),
    ("lea rax, [rip + 0x2f1a]", "lea rax [rip+<HEX>]"),
    # operator spacing outside brackets too
    ("add eax , 0x1 + 0x2", "add eax <HEX>+<HEX>"),
    # prefixed mnemonics stay whole: the normalizer never extracts a mnemonic
    # with a regex, so `rep`/`lock` are not stripped or split off
    ("rep movsb", "rep movsb"),
    ("lock cmpxchg dword ptr [eax], ecx", "lock cmpxchg dword ptr [eax] ecx"),
    ("repne scasb", "repne scasb"),
    # uppercase input is lowercased; runs of whitespace and commas collapse
    ("MOV     EAX ,  0X10", "mov eax <HEX>"),
    ("PUSH\tEBP", "push ebp"),
    ("  RET  ", "ret"),
    # no-operand instructions
    ("nop", "nop"),
]


@pytest.mark.parametrize("raw,expected", CASES)
def test_normalize_exact_tokens(normalize, raw, expected):
    assert normalize(raw) == expected


@pytest.mark.parametrize("blank", ["", "\n", "   ", "\t", "  \r\n"])
def test_blank_lines_return_none(normalize, blank):
    assert normalize(blank) is None


def test_snippet_end_to_end(normalize, tmp_path):
    """A hand-written snippet as it would arrive from extract.py, including the
    blank lines a file ends with, through the pipeline's own reader."""
    snippet = """\
push ebp
mov ebp, esp
sub esp, 0x40

MOV EAX, DWORD PTR [ EBP - 8 ]
rep movsb

lock cmpxchg dword ptr [eax], ecx
add esp, 12
ret
"""
    expected = [
        "push ebp",
        "mov ebp esp",
        "sub esp <HEX>",
        "mov eax dword ptr [ebp-<OFFSET>]",
        "rep movsb",
        "lock cmpxchg dword ptr [eax] ecx",
        "add esp <OFFSET>",
        "ret",
    ]
    p = tmp_path / "root_snippet.exe.txt"
    p.write_text(snippet, encoding="utf-8")

    from llm_features_pipeline.run_pipeline import read_instructions
    got = read_instructions(p, normalize, cap=5000)
    assert got == expected
    assert len(got) == 8, "the two blank lines must be dropped, not kept"


def test_truncation_at_5000_instructions(normalize, tmp_path):
    """`tokenization.py` keeps `instructions[:5000]`; the pipeline stops reading
    at the same point. Blank lines must not count toward the cap."""
    from llm_features_pipeline.run_pipeline import read_instructions

    lines = []
    for i in range(6000):
        lines.append("nop")
        if i % 100 == 0:
            lines.append("")          # blank lines are not instructions
    p = tmp_path / "root_long.exe.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert len(read_instructions(p, normalize, cap=5000)) == 5000
    assert len(read_instructions(p, normalize, cap=10)) == 10
    assert len(read_instructions(p, normalize, cap=0)) == 6000  # 0 = uncapped


def test_truncation_matches_upstream(normalize, tmp_path):
    """Early-exit truncation must give the same list as upstream's
    normalize-everything-then-slice, not just the same length."""
    from llm_features_pipeline.run_pipeline import read_instructions

    raw = [f"mov eax, {i}" for i in range(120)]
    p = tmp_path / "root_cmp.exe.txt"
    p.write_text("\n".join(raw) + "\n", encoding="utf-8")

    upstream = [normalize(l) for l in raw if normalize(l)][:50]
    assert read_instructions(p, normalize, cap=50) == upstream


# --------------------------------------------------------------------------
# llm_features_pipeline/data.py loader
# --------------------------------------------------------------------------

def _make_corpus(root: Path) -> Path:
    """A miniature LLM_Features tree: two ransomware families, goodware, and
    one empty file in each class."""
    sha = "%064x"
    layout = {
        "mal_train": [f"avaddon_{sha % 1}.txt", f"avaddon_{sha % 2}.txt",
                      f"phobos_{sha % 3}.txt"],
        "mal_test": [f"hive_{sha % 4}.txt", f"clop_{sha % 5}.txt"],
        "good_train": ["root_alpha.exe.txt", "root_beta.exe.txt", "root_zeta.exe.txt"],
        "good_test": ["root_gamma.exe.txt"],
    }
    empty = {"mal_train": f"phobos_{sha % 3}.txt", "good_train": "root_zeta.exe.txt"}
    for folder, names in layout.items():
        d = root / folder
        d.mkdir(parents=True)
        for n in names:
            body = "" if empty.get(folder) == n else "push ebp\nmov ebp, esp\nret\n"
            (d / n).write_text(body, encoding="utf-8")
    return root


def test_loader_is_sorted_and_deterministic(tmp_path):
    root = _make_corpus(tmp_path / "corpus")
    first = [s.name for s in D.load_mendeley(root, "goodware")]
    second = [s.name for s in D.load_mendeley(root, "goodware")]
    assert first == second, "two loads of the same tree must agree"
    train = [s.name for s in D.load_mendeley(root, "goodware") if s.split == "train"]
    assert train == sorted(train), "row order is the label mapping; it must be sorted"


def test_loader_carries_filename_and_path(tmp_path):
    root = _make_corpus(tmp_path / "corpus")
    for s in D.load_mendeley(root, "ransomware"):
        assert s.name == s.path.name
        assert s.path.is_file()


def test_labels_are_binary_and_correct(tmp_path):
    root = _make_corpus(tmp_path / "corpus")
    ran = D.load_mendeley(root, "ransomware")
    good = D.load_mendeley(root, "goodware")
    assert {s.label for s in ran} == {1}
    assert {s.label for s in good} == {0}
    assert {s.label for s in ran + good} <= {0, 1}


def test_family_is_a_group_never_a_label(tmp_path):
    root = _make_corpus(tmp_path / "corpus")
    ran = D.load_mendeley(root, "ransomware")
    by_name = {s.name: s for s in ran}
    a1 = next(s for n, s in by_name.items() if n.startswith("avaddon_"))
    assert a1.group == "avaddon"
    assert a1.meta["family"] == "avaddon"
    assert a1.label == 1, "the family must not leak into the label"
    # every ransomware sample of one family shares one group
    groups = {s.group for s in ran if s.name.startswith("avaddon_")}
    assert groups == {"avaddon"}
    # goodware gets a per-file group, so it is never grouped by family
    for s in D.load_mendeley(root, "goodware"):
        assert s.group == f"file:{s.name}"
        assert s.meta["family"] == ""


def test_mendeley_family_split_is_disjoint(tmp_path):
    root = _make_corpus(tmp_path / "corpus")
    ran = D.load_mendeley(root, "ransomware")
    tr = {s.group for s in ran if s.split == "train"}
    te = {s.group for s in ran if s.split == "test"}
    assert tr and te and not (tr & te)
    assert D.summarize(ran)["group_overlap"] == []


def test_empty_files_stop_the_run(tmp_path):
    """The loader keeps every file. `build_frames` used to drop the empty ones
    silently, which left sample_counts.json, splits.csv and the floors on a
    larger denominator than the model rows; now it stops and names them, so the
    cohort file is where such files get excluded."""
    pytest.importorskip("pandas")
    root = _make_corpus(tmp_path / "corpus")
    samples = D.load_mendeley(root, "ransomware") + D.load_mendeley(root, "goodware")
    assert len(samples) == 9, "the loader itself drops nothing"

    from llm_features_pipeline.run_pipeline import build_frames
    with pytest.raises(SystemExit, match="2 files normalise to nothing"):
        build_frames(samples, lambda s: (s.strip().lower() or None), cap=5000)
    full = [s for s in samples if not (s.name.startswith("phobos_") or s.name == "root_zeta.exe.txt")]
    df = build_frames(full, lambda s: (s.strip().lower() or None), cap=5000)
    assert len(df) == 7


def test_content_leak_finds_a_planted_duplicate(tmp_path):
    root = tmp_path / "corpus"
    _make_corpus(root)
    # make one test goodware file a byte-identical copy of a train one
    (root / "good_test" / "root_gamma.exe.txt").write_bytes(
        (root / "good_train" / "root_alpha.exe.txt").read_bytes())
    samples = D.load_mendeley(root, "goodware") + D.load_mendeley(root, "ransomware")
    leak = D.content_leak(samples)
    assert leak["test_goodware_duplicated_in_train"] == 1
    assert leak["test_goodware_n"] == 1
    assert leak["test_goodware_leak_rate"] == 1.0
    # 'push ebp/mov ebp, esp/ret' is shared by both classes here, so the
    # both-labels counter must fire too
    assert leak["streams_labelled_both_classes"] >= 1


def _balanced_pool(tmp_path, layout):
    """layout: {bucket: [group_size, ...]} - mirrors Goodware_Balanced, where
    `everyday`/`hard_negative` are multi-file projects and `system_local` is
    286 independent one-file groups."""
    pool = []
    for bucket, sizes in layout.items():
        for e, size in enumerate(sizes):
            for f in range(size):
                p = tmp_path / f"{bucket}_{e}_{f}.txt"
                p.write_text("nop\n", encoding="utf-8")
                pool.append(D.Sample(p, 0, "balanced_goodware", f"{bucket}_{e}",
                                     "", {"bucket": bucket,
                                          "entry_id": f"{bucket}_{e}"}))
    return pool


def test_group_split_keeps_groups_whole_and_is_seeded(tmp_path):
    """Balanced-goodware split: whole entry_id groups on one side only, bucket
    proportions preserved, identical across runs with the same seed."""
    layout = {"everyday": [5, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1],   # 25
              "hard_negative": [4, 3, 2, 1, 1, 1, 1, 1, 1],       # 15
              "system": [1] * 10}                                 # 10
    pool = _balanced_pool(tmp_path, layout)
    assert len(pool) == 50

    a = D.group_split(list(pool), 40, 10, seed=42)
    for s in pool:
        s.split = ""
    b = D.group_split(list(pool), 40, 10, seed=42)
    assert [(s.path.name, s.split) for s in a] == [(s.path.name, s.split) for s in b]

    tr = {s.group for s in a if s.split == "train"}
    te = {s.group for s in a if s.split == "test"}
    assert not (tr & te), "a source project must not straddle the split"
    assert sum(1 for s in a if s.split == "train") == 40
    assert sum(1 for s in a if s.split == "test") == 10
    # stratification: each bucket's share of the test side tracks its share of
    # the pool (25/15/10 of 50 -> 5/3/2 of 10)
    import collections as _c
    got = _c.Counter(s.meta["bucket"] for s in a if s.split == "test")
    assert got == {"everyday": 5, "hard_negative": 3, "system": 2}


def test_group_split_undershoots_rather_than_tearing_a_group(tmp_path):
    """With coarse groups the quotas cannot be hit exactly. The contract is that
    a group is never torn: leftover groups sit the experiment out instead.

    On the real corpus this costs nothing - `system_local`'s 286 one-file
    groups make the quotas reachable exactly, which is why Exp B lands on
    1,116/131 on the nose - but the guarantee must hold either way."""
    pool = _balanced_pool(tmp_path, {"everyday": [5] * 6,
                                     "hard_negative": [5] * 4,
                                     "system": [5] * 2})
    a = D.group_split(list(pool), 40, 10, seed=42)
    n_tr = sum(1 for s in a if s.split == "train")
    n_te = sum(1 for s in a if s.split == "test")
    assert n_tr <= 40 and n_te <= 10
    assert n_tr + n_te < 50, "coarse groups cannot fill the quotas exactly"
    by_group = collections_defaultdict_splits(a)
    for g, splits in by_group.items():
        assert len(splits) == 1, f"group {g} was torn across {splits}"


def collections_defaultdict_splits(samples):
    import collections as _c
    out = _c.defaultdict(set)
    for s in samples:
        out[s.group].add(s.split)
    return out


def test_group_split_refuses_to_oversubscribe(tmp_path):
    p = tmp_path / "everyday_0_0.txt"
    p.write_text("nop\n", encoding="utf-8")
    pool = [D.Sample(p, 0, "balanced_goodware", "e0", "", {"bucket": "everyday"})]
    with pytest.raises(ValueError):
        D.group_split(pool, 5, 5, seed=42)


def test_cross_source_dedup_removes_a_planted_overlap(tmp_path):
    """A Goodware_Balanced file that is really a Mendeley goodware binary must
    be removed at load time, by either identity."""
    import csv as _csv
    import json as _json

    mend_dir = tmp_path / "mendeley"
    mend_dir.mkdir()
    m1 = mend_dir / "root_shared.exe.txt"
    m1.write_text("push ebp\nret\n", encoding="utf-8")
    m2 = mend_dir / "root_other.exe.txt"
    m2.write_text("nop\nnop\n", encoding="utf-8")
    mendeley = [D.Sample(m1, 0, "mendeley_goodware", "file:a"),
                D.Sample(m2, 0, "mendeley_goodware", "file:b")]

    bal_dir = tmp_path / "balanced"
    bal_dir.mkdir()
    # (a) same opcode stream as m1 -> caught by content hash
    dup_content = bal_dir / "everyday_dup.exe.txt"
    dup_content.write_text("push ebp\nret\n", encoding="utf-8")
    # (b) different stream, but its source binary sha256 is a Mendeley one
    dup_sha = bal_dir / "everyday_sha.exe.txt"
    dup_sha.write_text("xor eax, eax\n", encoding="utf-8")
    # (c) genuinely new
    keep = bal_dir / "everyday_keep.exe.txt"
    keep.write_text("mov eax, ebx\n", encoding="utf-8")
    pool = [D.Sample(p, 0, "balanced_goodware", "g", "", {})
            for p in (dup_content, dup_sha, keep)]

    manifest = tmp_path / "opcode_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=["txt_file", "sha256"])
        w.writeheader()
        w.writerow({"txt_file": dup_content.name, "sha256": "a" * 64})
        w.writerow({"txt_file": dup_sha.name, "sha256": "b" * 64})
        w.writerow({"txt_file": keep.name, "sha256": "c" * 64})

    sha_json = tmp_path / "mendeley_goodware_sha256.json"
    sha_json.write_text(_json.dumps({"b" * 64: ["other.exe"]}), encoding="utf-8")

    kept, rep = D.dedup_goodware_sources(pool, mendeley, manifest, sha_json)
    assert [s.name for s in kept] == [keep.name]
    assert rep["removed_total"] == 2
    assert rep["removed_by_source_sha256"] == [dup_sha.name]
    assert rep["removed_by_content_hash"] == [dup_content.name]


def test_tokenizer_cache_reuses_and_invalidates(tmp_path):
    """The WordPiece trainer has no seed and picks a different vocabulary in
    every process (audit §1.10), so the trained tokenizer is cached against a
    fingerprint of its training corpus. Two things must hold: an unchanged
    corpus reuses the cached tokenizer without retraining, and a changed corpus
    retrains."""
    pytest.importorskip("tokenizers")
    pd = pytest.importorskip("pandas")
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    repo = Path(cfg["paths"]["tokenization_repo"])
    if not repo.is_dir():
        pytest.skip(f"tokenization repo not present: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from Tokenization.tokenization import train_tokenizer
    from llm_features_pipeline.run_pipeline import get_tokenizer

    calls = []

    def counting_train(*a, **k):
        calls.append(1)
        return train_tokenizer(*a, **k)

    rows = [" <SEP> ".join(["mov eax <HEX>", "push ebp", "ret"] * 20)
            for _ in range(30)]
    fit = pd.DataFrame({"Instructions": rows})
    cache = tmp_path / "tokenizers"

    t1 = get_tokenizer("WPC", counting_train, fit, 200, cache, "expTest")
    assert len(calls) == 1
    assert (cache / "expTest_WPC.json").is_file()
    assert (cache / "expTest_WPC.fingerprint").is_file()

    t2 = get_tokenizer("WPC", counting_train, fit, 200, cache, "expTest")
    assert len(calls) == 1, "an unchanged corpus must not retrain"
    assert t1.get_vocab() == t2.get_vocab()

    changed = pd.DataFrame({"Instructions": rows + ["xor eax eax <SEP> nop"]})
    get_tokenizer("WPC", counting_train, changed, 200, cache, "expTest")
    assert len(calls) == 2, "a changed corpus must retrain"

    # a different vocab size is a different tokenizer, same corpus
    get_tokenizer("WPC", counting_train, changed, 300, cache, "expTest")
    assert len(calls) == 3


def test_cross_source_dedup_is_a_noop_when_there_is_no_overlap(tmp_path):
    import json as _json
    mend_dir = tmp_path / "mendeley"
    mend_dir.mkdir()
    m = mend_dir / "root_a.exe.txt"
    m.write_text("push ebp\n", encoding="utf-8")
    bal = tmp_path / "everyday_b.exe.txt"
    bal.write_text("nop\n", encoding="utf-8")
    sha_json = tmp_path / "s.json"
    sha_json.write_text(_json.dumps({}), encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("txt_file,sha256\neveryday_b.exe.txt,\n", encoding="utf-8")
    kept, rep = D.dedup_goodware_sources(
        [D.Sample(bal, 0, "balanced_goodware", "g")],
        [D.Sample(m, 0, "mendeley_goodware", "file:a")],
        manifest, sha_json)
    assert len(kept) == 1 and rep["removed_total"] == 0
    # a missing manifest used to degrade the check to a no-op; it is fatal now
    with pytest.raises(FileNotFoundError):
        D.dedup_goodware_sources([D.Sample(bal, 0, "balanced_goodware", "g")],
                                 [D.Sample(m, 0, "mendeley_goodware", "file:a")],
                                 tmp_path / "missing.csv", sha_json)
