"""
Unit tests for the cohort loaders, the cohort filter, split reuse, and the
per-architecture / per-family reporting added for expA_cohort, expB_cohort,
expC and expD.

What is pinned here:

1. **The join key.** The cohort CSVs are joined to feature files on
   `<family>_<filename>.txt`, not on sha256. A sha256-first join is wrong for
   the 25 Mendeley ransomware rows whose recorded sha256 differs from the
   sha256 in their filename, and for two of those it maps two different samples
   onto one row. A planted copy of that exact shape is asserted below.
2. **The filter.** `in_cohort == 1` and nothing else; an unmatched file is
   dropped, not kept, because its packing status is unknown.
3. **Split reuse.** expB_cohort and expD take expB's committed goodware
   membership verbatim instead of re-splitting, and take its groups too - the
   revised Goodware_Balanced manifest has no `entry_id`, so without that the
   train/test group-disjointness assertion would go vacuous.
4. **Mnemonic-only input.** `normalize_instruction`, the `<SEP>` join and the
   WordPiece tokenizer on one-token lines. WPC provably degenerates to
   whole-word tokenization in this regime, which is the caveat the revised
   columns of results/summary.md have to carry.
5. **The new metrics.** Per-architecture metrics for both classes and
   per-family ransomware recall, against hand-computed numbers.
6. **The floors.** The majority-class and architecture-only baselines, and the
   fact that they are a pure function of `samples.arch` - which is what lets
   `--summary-only` put a floor beside a score in a `metrics.json` written
   before the floors existed.

Runs under the repo's own Python (3.14) as well as the 3.12 venv: every heavy
import is guarded with `importorskip`, as in test_tokenization_pipeline.py.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from llm_features_pipeline import data as D  # noqa: E402

CONFIG = REPO / "llm_features_pipeline" / "config.yaml"

COHORT_HEADER = ["corpus", "sha256", "set", "label", "family", "filename",
                 "arch", "tag", "in_cohort", "exclude_reason"]


def write_cohort(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COHORT_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COHORT_HEADER})
    return path


def row(**kw) -> dict:
    base = {"corpus": "mendeley", "sha256": "0" * 64, "set": "mal_train",
            "label": "1", "family": "fam", "filename": "f.exe", "arch": "x86",
            "tag": "plain", "in_cohort": "1", "exclude_reason": ""}
    base.update(kw)
    return base


def make_sample(tmp_path: Path, name: str, label: int, source: str,
                set_: str = "", text: str = "mov\npush\n") -> D.Sample:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    meta = {"family": "", "cohort_set": set_}
    return D.Sample(p, label, source, f"g:{name}", "", meta)


# --------------------------------------------------------------------------
# Cohort index and lookup
# --------------------------------------------------------------------------

def test_cohort_is_keyed_on_the_feature_filename(tmp_path):
    csvp = write_cohort(tmp_path / "c.csv", [
        row(family="avaddon", filename="aa" * 32, sha256="bb" * 32, arch="x64"),
        row(set="good_train", label="0", family="root", filename="ARP.EXE",
            arch="x86"),
    ])
    coh = D.load_cohort(csvp)
    assert len(coh) == 2
    assert coh.lookup(f"avaddon_{'aa' * 32}.txt")["arch"] == "x64"
    assert coh.lookup("root_ARP.EXE.txt", "good_train")["arch"] == "x86"
    # case-insensitive: the traditional and revised writers disagree on case
    # for some Windows filenames.
    assert coh.lookup("root_arp.exe.txt") is not None
    assert coh.lookup("root_NOT_HERE.exe.txt") is None


def test_lookup_is_scoped_by_set_when_a_set_is_given(tmp_path):
    csvp = write_cohort(tmp_path / "c.csv", [
        row(set="mal_train", family="fam", filename="x", arch="x86"),
        row(set="mal_test", family="fam", filename="x", arch="x64"),
    ])
    coh = D.load_cohort(csvp)
    assert coh.lookup("fam_x.txt", "mal_train")["arch"] == "x86"
    assert coh.lookup("fam_x.txt", "mal_test")["arch"] == "x64"


def test_in_cohort_row_wins_over_a_dup_row(tmp_path):
    """`tag: dup` rows share a sha256 with a kept row and are always out of the
    cohort. A lookup that lands on one of them would report the wrong
    architecture and, worse, drop a file that belongs in the cohort."""
    csvp = write_cohort(tmp_path / "c.csv", [
        row(set="mal_train", family="fam", filename="x", arch="",
            tag="dup", in_cohort="0", exclude_reason="tag:dup"),
        row(set="mal_train", family="fam", filename="x", arch="x86",
            tag="plain", in_cohort="1"),
    ])
    coh = D.load_cohort(csvp)
    hit = coh.lookup("fam_x.txt", "mal_train")
    assert hit["in_cohort"] == "1" and hit["arch"] == "x86"


def test_load_cohort_rejects_a_csv_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("sha256,set\nabc,mal_train\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        D.load_cohort(p)


def test_load_cohort_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        D.load_cohort(tmp_path / "nope.csv")


def test_filename_join_separates_what_a_sha256_join_collides(tmp_path):
    """The real collision, planted.

    `darkside_4d9432e8....txt` and `darkside_ec368752....txt` are two different
    samples. The cohort row for the first records sha256 `ec368752...` - the
    Mendeley filename is not always the sha256 of the file. A sha256-first join
    maps BOTH files onto that one row; the filename join keeps them apart.
    """
    a, b = "4d" * 32, "ec" * 32
    csvp = write_cohort(tmp_path / "c.csv", [
        row(family="darkside", filename=a, sha256=b, arch="x86"),
        row(family="darkside", filename=b, sha256="99" * 32, arch="x64"),
    ])
    coh = D.load_cohort(csvp)
    ra = coh.lookup(f"darkside_{a}.txt")
    rb = coh.lookup(f"darkside_{b}.txt")
    assert ra is not rb
    assert (ra["arch"], rb["arch"]) == ("x86", "x64")
    # And the thing the naive join would have done:
    by_sha = {r["sha256"]: r for r in coh.rows}
    assert by_sha[b] is ra, "the sha256 index really does point both at one row"


# --------------------------------------------------------------------------
# annotate / filter
# --------------------------------------------------------------------------

@pytest.fixture
def annotated(tmp_path):
    csvp = write_cohort(tmp_path / "c.csv", [
        row(set="mal_train", family="avaddon", filename="a", arch="x86"),
        row(set="mal_train", family="thanos", filename="b", arch="x86",
            in_cohort="0", exclude_reason="family:thanos"),
        row(set="mal_train", family="conti", filename="c", arch="x64",
            tag="packed_other", in_cohort="0", exclude_reason="tag:packed_other"),
        row(set="good_train", label="0", family="root", filename="d",
            arch="x64"),
    ])
    samples = [
        make_sample(tmp_path, "avaddon_a.txt", 1, "mendeley_ransomware", "mal_train"),
        make_sample(tmp_path, "thanos_b.txt", 1, "mendeley_ransomware", "mal_train"),
        make_sample(tmp_path, "conti_c.txt", 1, "mendeley_ransomware", "mal_train"),
        make_sample(tmp_path, "root_d.txt", 0, "mendeley_goodware", "good_train"),
        make_sample(tmp_path, "root_unknown.txt", 0, "mendeley_goodware", "good_train"),
    ]
    rep = D.annotate_cohort(samples, D.load_cohort(csvp))
    return samples, rep


def test_annotate_attaches_arch_tag_and_family(annotated):
    samples, rep = annotated
    by = {s.name: s for s in samples}
    assert by["avaddon_a.txt"].meta["arch"] == "x86"
    assert by["avaddon_a.txt"].meta["family"] == "avaddon"
    assert by["conti_c.txt"].meta["cohort_tag"] == "packed_other"
    assert by["root_d.txt"].meta["arch"] == "x64"
    # goodware keeps an empty family: the family field is a ransomware split
    # group, and "root" is a folder name, not a family.
    assert by["root_d.txt"].meta["family"] == ""
    assert rep["annotated"] == 4 and rep["unmatched"] == 1
    assert rep["in_cohort"] == 2


def test_annotate_never_drops_anything(annotated):
    samples, _ = annotated
    assert len(samples) == 5


def test_unmatched_sample_is_flagged_not_guessed(annotated):
    samples, rep = annotated
    unk = next(s for s in samples if s.name == "root_unknown.txt")
    assert unk.meta["in_cohort"] is False
    assert unk.meta["arch"] == ""
    assert "root_unknown.txt" in rep["unmatched_examples"]


def test_filter_keeps_only_in_cohort_and_reports_reasons(annotated):
    samples, _ = annotated
    kept, rep = D.filter_cohort(samples)
    assert sorted(s.name for s in kept) == ["avaddon_a.txt", "root_d.txt"]
    assert rep == {"in": 5, "out": 2, "dropped": 3,
                   "dropped_by_reason": {"family:thanos": 1,
                                         "no_cohort_row": 1,
                                         "tag:packed_other": 1}}


def test_filter_drops_an_unmatched_file_rather_than_keeping_it(annotated):
    samples, _ = annotated
    kept, _ = D.filter_cohort(samples)
    assert "root_unknown.txt" not in {s.name for s in kept}


# --------------------------------------------------------------------------
# split reuse
# --------------------------------------------------------------------------

def _committed_split(tmp_path: Path, rows: list[tuple]) -> Path:
    p = tmp_path / "splits.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "source", "label", "group", "split"])
        for r in rows:
            w.writerow(r)
    return p


def test_reuse_split_assigns_the_committed_membership(tmp_path):
    committed = _committed_split(tmp_path, [
        ("a.txt", "balanced_goodware", 0, "proj1", "train"),
        ("b.txt", "balanced_goodware", 0, "proj1", "train"),
        ("c.txt", "balanced_goodware", 0, "proj2", "test"),
        ("gone.txt", "balanced_goodware", 0, "proj9", "test"),
        ("r.txt", "mendeley_ransomware", 1, "avaddon", "train"),
    ])
    pool = [make_sample(tmp_path, n, 0, "balanced_goodware")
            for n in ("a.txt", "b.txt", "c.txt", "extra.txt")]
    chosen, rep = D.reuse_split(pool, committed, "balanced_goodware")

    assert {s.name for s in chosen} == {"a.txt", "b.txt", "c.txt"}
    assert {s.name: s.split for s in chosen} == {"a.txt": "train", "b.txt": "train",
                                                 "c.txt": "test"}
    assert (rep["selected_train"], rep["selected_test"]) == (2, 1)
    # "extra.txt" is in the pool but not in the committed split: excluded.
    assert rep["pool"] == 4 and rep["selected"] == 3
    # "gone.txt" is in the committed split but not in this pool: reported.
    assert rep["named_but_absent_from_pool"] == 1
    assert rep["named_but_absent_examples"] == ["gone.txt"]
    # the ransomware rows of the committed file are not touched
    assert rep["named_by_split"] == 4


def test_reuse_split_takes_the_groups_too(tmp_path):
    """The REVISED Goodware_Balanced manifest carries no `entry_id`, so every
    sample would fall back to a per-file group and the train/test group-overlap
    assertion in run_experiment would become vacuous exactly where project
    disjointness has to be proven."""
    committed = _committed_split(tmp_path, [
        ("a.txt", "balanced_goodware", 0, "entry:7zip", "train"),
        ("b.txt", "balanced_goodware", 0, "entry:7zip", "train"),
    ])
    pool = [make_sample(tmp_path, n, 0, "balanced_goodware") for n in ("a.txt", "b.txt")]
    for s in pool:
        s.group = f"file:{s.name}"          # the degraded fallback
    chosen, rep = D.reuse_split(pool, committed, "balanced_goodware")
    assert {s.group for s in chosen} == {"entry:7zip"}
    assert rep["groups_taken_from_splits_csv"] == 2


def test_reuse_split_filters_by_source(tmp_path):
    committed = _committed_split(tmp_path, [
        ("a.txt", "mendeley_goodware", 0, "g", "train"),
    ])
    pool = [make_sample(tmp_path, "a.txt", 0, "balanced_goodware")]
    chosen, rep = D.reuse_split(pool, committed, "balanced_goodware")
    assert chosen == [] and rep["named_by_split"] == 0


def test_reuse_split_refuses_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        D.reuse_split([], tmp_path / "nope.csv", "balanced_goodware")


# --------------------------------------------------------------------------
# breakdowns
# --------------------------------------------------------------------------

def test_arch_breakdown_covers_both_classes_and_both_splits(tmp_path):
    samples = []
    for name, label, split, arch in [
            ("r1.txt", 1, "train", "x86"), ("r2.txt", 1, "train", "x86"),
            ("r3.txt", 1, "test", "x64"), ("g1.txt", 0, "train", "x64"),
            ("g2.txt", 0, "test", "x86"), ("g3.txt", 0, "test", "")]:
        s = make_sample(tmp_path, name, label, "src")
        s.split, s.meta["arch"] = split, arch
        samples.append(s)
    out = D.arch_breakdown(samples)
    assert out["train_ransomware"] == {"x86": 2}
    assert out["test_ransomware"] == {"x64": 1}
    assert out["train_goodware"] == {"x64": 1}
    assert out["test_goodware"] == {"unknown": 1, "x86": 1}


def test_family_breakdown_counts_ransomware_only(tmp_path):
    samples = []
    for name, label, split, fam in [("a.txt", 1, "train", "avaddon"),
                                    ("b.txt", 1, "train", "avaddon"),
                                    ("c.txt", 1, "test", "hive"),
                                    ("g.txt", 0, "train", "")]:
        s = make_sample(tmp_path, name, label, "src")
        s.split, s.meta["family"] = split, fam
        samples.append(s)
    out = D.family_breakdown(samples)
    assert out["train"] == {"avaddon": 2}
    assert out["test"] == {"hive": 1}


# --------------------------------------------------------------------------
# trivial-rule floors
# --------------------------------------------------------------------------

def test_baselines_majority_class_against_hand_computed_numbers():
    # 3 goodware (2 x86), 7 ransomware (6 x86) on the test side.
    arch = {"test_goodware": {"x86": 2, "x64": 1},
            "test_ransomware": {"x86": 6, "x64": 1}}
    m = D.baselines(arch)["majority_class"]
    assert m["predicts"] == "ransomware"
    assert m["confusion_matrix"] == {"tn": 0, "fp": 3, "fn": 0, "tp": 7}
    assert m["accuracy"] == pytest.approx(0.7)
    # calling everything ransomware: recall(ran) 1, recall(good) 0
    assert m["balanced_accuracy"] == pytest.approx(0.5)
    # F1(good) = 0, F1(ran) = 2*0.7*1/1.7
    assert m["macro_f1"] == pytest.approx((2 * 0.7 * 1 / 1.7) / 2, abs=1e-6)


def test_baselines_majority_class_flips_when_goodware_is_larger():
    arch = {"test_goodware": {"x86": 9}, "test_ransomware": {"x86": 1}}
    m = D.baselines(arch)["majority_class"]
    assert m["predicts"] == "goodware"
    assert m["confusion_matrix"] == {"tn": 9, "fp": 0, "fn": 1, "tp": 0}
    assert m["accuracy"] == pytest.approx(0.9)
    assert m["balanced_accuracy"] == pytest.approx(0.5)


def test_baselines_x86_rule_reads_only_architecture():
    arch = {"test_goodware": {"x86": 2, "x64": 1},
            "test_ransomware": {"x86": 6, "x64": 1}}
    a = D.baselines(arch)["x86_is_ransomware"]
    # every x86 -> ransomware: tp 6, fp 2, fn 1, tn 1
    assert a["confusion_matrix"] == {"tn": 1, "fp": 2, "fn": 1, "tp": 6}
    assert a["test_x86"] == 8
    assert a["test_unknown_arch"] == 0
    assert a["recall_ransomware"] == pytest.approx(6 / 7)
    assert a["recall_goodware"] == pytest.approx(1 / 3)
    assert a["accuracy"] == pytest.approx(0.7)


def test_baselines_count_unknown_architecture_as_not_x86():
    arch = {"test_goodware": {"unknown": 4},
            "test_ransomware": {"unknown": 2, "x86": 2}}
    a = D.baselines(arch)["x86_is_ransomware"]
    assert a["test_unknown_arch"] == 6
    assert a["test_x86"] == 2
    # the 2 unknown-arch ransomware are predicted goodware, so they are misses
    assert a["confusion_matrix"] == {"tn": 4, "fp": 0, "fn": 2, "tp": 2}
    assert a["recall_goodware"] == pytest.approx(1.0)


def test_baselines_are_recoverable_from_a_committed_metrics_json():
    """The summary writer recomputes floors for runs that predate them, so the
    block must be a pure function of `samples.arch` and nothing else."""
    import json
    stored = {}
    for name in ("expA", "expB", "expC", "expD"):
        f = REPO / "results" / name / "metrics.json"
        if not f.is_file():
            continue
        s = json.loads(f.read_text(encoding="utf-8"))["samples"]
        stored[name] = s
        assert D.baselines(s["arch"]) == s["baselines"], name
    if not stored:
        pytest.skip("no committed metrics.json to check against")


# --------------------------------------------------------------------------
# mnemonic-only input
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tok_repo():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    repo = Path(cfg["paths"]["tokenization_repo"])
    if not repo.is_dir():
        pytest.skip(f"tokenization repo not present: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    pytest.importorskip("tokenizers")
    pytest.importorskip("pandas")
    return repo


@pytest.mark.parametrize("raw,expected", [
    ("mov\n", "mov"),
    ("PUSH\n", "push"),
    ("  int3  \n", "int3"),
    ("vpxor\n", "vpxor"),
    ("cmpxchg16b\n", "cmpxchg16b"),
    # A digit-only suffix is still an <OFFSET> substitution, because
    # normalize_instruction does not know it is looking at a bare mnemonic.
    ("rep movsb\n", "rep movsb"),
])
def test_normalize_passes_single_mnemonics_through(tok_repo, raw, expected):
    from Tokenization.tokenization import normalize_instruction
    assert normalize_instruction(raw) == expected


def test_normalize_still_rejects_blank_mnemonic_lines(tok_repo):
    from Tokenization.tokenization import normalize_instruction
    for blank in ("", "\n", "   \n", "\t\n"):
        assert normalize_instruction(blank) is None


def test_sep_join_of_mnemonics_round_trips(tok_repo):
    """The WPC/BPE path joins the instruction list with ' <SEP> '. On
    mnemonic-only input that produces alternating mnemonic/<SEP> words, with no
    whitespace inside a 'word' - the case the pre-tokenizer's
    `<[^>]+>|\\[[^\\]]+\\]|\\w+` regex has to keep whole."""
    insns = ["mov", "push", "int3"]
    text = " <SEP> ".join(insns)
    assert text == "mov <SEP> push <SEP> int3"
    assert text.split(" <SEP> ") == insns


def test_wpc_degenerates_to_whole_words_on_mnemonic_only_input(tok_repo):
    """The caveat the revised columns of results/summary.md must carry.

    The distinct-mnemonic vocabulary is a few hundred, well under
    `vocab_size: 1000`, so the WordPiece trainer exhausts the corpus before it
    reaches the size cutoff. Every mnemonic then encodes as exactly one
    whole-word token: no `##` continuation pieces, no `<UNK>`. WPC and SW are
    reading the same stream, and a WPC row in Exp C or Exp D is not evidence
    about subword tokenization.
    """
    pd = pytest.importorskip("pandas")
    from Tokenization.tokenization import train_tokenizer

    mnemonics = ["mov", "push", "pop", "add", "sub", "call", "ret", "jmp",
                 "cmp", "je", "jne", "lea", "int3", "xor", "test", "nop",
                 "vpxor", "cmpxchg", "rorx", "vmovaps"]
    docs = [" <SEP> ".join(mnemonics[i % len(mnemonics):] + mnemonics[:i % len(mnemonics)])
            for i in range(40)]
    fit = pd.DataFrame({"Instructions": docs})
    tok = train_tokenizer("WPC", fit, 1000, "<UNK>",
                          ["<UNK>", "<SEP>", "<MASK>", "<CLS>", "<HEX>", "<OFFSET>"],
                          1000)

    vocab = tok.get_vocab()
    assert len(vocab) < 1000, "the trainer should run out of corpus, not hit the cap"

    enc = tok.encode(" <SEP> ".join(mnemonics))
    assert "<UNK>" not in enc.tokens
    assert not [t for t in enc.tokens if t.startswith("##")], \
        "a continuation piece means WPC did NOT degenerate to whole words"
    # one token per mnemonic plus one <SEP> between each pair
    assert len(enc.tokens) == 2 * len(mnemonics) - 1
    assert [t for t in enc.tokens if t != "<SEP>"] == mnemonics


def test_wpc_does_fragment_full_instruction_input(tok_repo):
    """The control for the test above: on the traditional feature form the same
    tokenizer does produce continuation pieces, so the degeneracy really is a
    property of mnemonic-only input and not of the settings."""
    pd = pytest.importorskip("pandas")
    from Tokenization.tokenization import train_tokenizer

    lines = ["push ebp", "mov ebp esp", "sub esp <HEX>", "mov eax [ebp+<HEX>]",
             "lea ecx [ebp-<HEX>]", "call <HEX>", "add esp <HEX>", "pop ebp",
             "ret", "xor eax eax", "cmp dword ptr [ebp+<HEX>] <HEX>"]
    docs = [" <SEP> ".join(lines[i % len(lines):] + lines[:i % len(lines)])
            for i in range(40)]
    tok = train_tokenizer("WPC", pd.DataFrame({"Instructions": docs}), 60, "<UNK>",
                          ["<UNK>", "<SEP>", "<MASK>", "<CLS>", "<HEX>", "<OFFSET>"],
                          1000)
    enc = tok.encode(" <SEP> ".join(lines))
    assert [t for t in enc.tokens if t.startswith("##")], \
        "a small vocab over full instructions must fragment something"


# --------------------------------------------------------------------------
# per-architecture and per-family metrics
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rp():
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    pytest.importorskip("yaml")
    import importlib
    return importlib.import_module("llm_features_pipeline.run_pipeline")


def _frame(pd, rows):
    return pd.DataFrame(rows, columns=["file", "Label", "arch", "family"])


def test_per_arch_metrics_against_hand_computed_numbers(rp):
    pd = pytest.importorskip("pandas")
    te = _frame(pd, [
        # x86: 2 ransomware (1 caught), 2 goodware (2 caught)
        ("r1", 1, "x86", "hive"), ("r2", 1, "x86", "hive"),
        ("g1", 0, "x86", ""), ("g2", 0, "x86", ""),
        # x64: 1 ransomware (0 caught), 1 goodware (1 caught)
        ("r3", 1, "x64", "clop"), ("g3", 0, "x64", ""),
    ])
    y_pred = [1, 0, 0, 0, 0, 0]
    out = rp.per_arch_metrics(te, y_pred)

    assert out["x86"]["n"] == 4
    assert out["x86"]["support_ransomware"] == 2
    assert out["x86"]["recall_ransomware"] == pytest.approx(0.5)
    assert out["x86"]["recall_goodware"] == pytest.approx(1.0)
    assert out["x86"]["accuracy"] == pytest.approx(0.75)
    assert out["x64"]["recall_ransomware"] == pytest.approx(0.0)
    assert out["x64"]["recall_goodware"] == pytest.approx(1.0)
    assert out["x64"]["accuracy"] == pytest.approx(0.5)
    # both classes present in both slices -> both macro-F1 numbers exist
    assert out["x86"]["macro_f1"] is not None
    assert out["x64"]["macro_f1"] is not None


def test_per_arch_macro_f1_is_null_when_a_slice_has_one_class(rp):
    """Averaging an F1 over a class with zero support reports "the model
    failed" when the truth is "the question was not asked"."""
    pd = pytest.importorskip("pandas")
    te = _frame(pd, [("r1", 1, "x64", "hive"), ("r2", 1, "x64", "hive"),
                     ("g1", 0, "x86", "")])
    out = rp.per_arch_metrics(te, [1, 1, 0])
    assert out["x64"]["macro_f1"] is None
    assert out["x64"]["recall_goodware"] is None
    assert out["x64"]["recall_ransomware"] == pytest.approx(1.0)
    assert out["x86"]["macro_f1"] is None


def test_per_arch_buckets_blank_architecture_as_unknown(rp):
    pd = pytest.importorskip("pandas")
    te = _frame(pd, [("r1", 1, "", "hive"), ("g1", 0, "", "")])
    out = rp.per_arch_metrics(te, [1, 0])
    assert set(out) == {"unknown"}
    assert out["unknown"]["n"] == 2


def test_ransomware_recall_by_family_ignores_goodware(rp):
    pd = pytest.importorskip("pandas")
    te = _frame(pd, [
        ("r1", 1, "x86", "hive"), ("r2", 1, "x86", "hive"),
        ("r3", 1, "x86", "clop"),
        ("g1", 0, "x86", ""), ("g2", 0, "x86", ""),
    ])
    out = rp.ransomware_recall_by_family(te, [1, 0, 1, 1, 1])
    assert set(out) == {"hive", "clop"}
    assert out["hive"] == {"n": 2, "recall": pytest.approx(0.5), "detected": 1}
    assert out["clop"] == {"n": 1, "recall": pytest.approx(1.0), "detected": 1}


def test_family_recall_denominators_sum_to_the_ransomware_support(rp):
    pd = pytest.importorskip("pandas")
    te = _frame(pd, [("r1", 1, "x86", "hive"), ("r2", 1, "x64", "clop"),
                     ("r3", 1, "x64", "maui"), ("g1", 0, "x86", "")])
    out = rp.ransomware_recall_by_family(te, [1, 1, 0, 0])
    assert sum(v["n"] for v in out.values()) == 3


# --------------------------------------------------------------------------
# Against the real corpora, when they are present
# --------------------------------------------------------------------------

def _cfg():
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _need(path_key, sub=""):
    cfg = _cfg()
    p = Path(cfg["paths"][path_key])
    if sub:
        p = p / sub
    if not p.exists():
        pytest.skip(f"{path_key} not present: {p}")
    return p


def test_real_cohort_joins_every_revised_feature_file():
    """The revised folders ARE the cohort: `mn_to_features.py` applies the
    filter as it writes. Every file must join, and every join must land on an
    `in_cohort == 1` row - that is the assertion that the two agree."""
    root = _need("llm_features_revised")
    coh = D.load_cohort(_need("cohort_mendeley"))
    ran = D.load_mendeley(root, "ransomware")
    good = D.load_mendeley(root, "goodware")
    rep = D.annotate_cohort(ran + good, coh)
    assert rep["unmatched"] == 0, rep["unmatched_examples"]
    assert rep["in_cohort"] == rep["annotated"] == len(ran) + len(good)


def test_real_filename_join_claims_no_cohort_row_twice():
    """A sha256-first join double-books 5 rows across mal_train and mal_test.
    The filename join must be a bijection."""
    root = _need("llm_features_revised")
    coh = D.load_cohort(_need("cohort_mendeley"))
    seen = {}
    for which in ("ransomware", "goodware"):
        for s in D.load_mendeley(root, which):
            r = coh.lookup(s.name, s.meta["cohort_set"])
            assert r is not None, s.name
            key = (r["set"], r["family"], r["filename"])
            assert key not in seen, f"{s.name} and {seen[key]} claim the same row"
            seen[key] = s.name


def test_real_cohort_filter_reproduces_the_revised_counts():
    """expA_cohort's counts have to be the traditional corpus filtered to the
    same cohort the revised folders were built from. The two feature trees do
    not hold the same files - the traditional extractor dropped .NET/UPX itself
    and the revised sweep recovered some samples the linear sweep could not
    finish - so these are the numbers, asserted rather than assumed."""
    trad = _need("llm_features")
    coh = D.load_cohort(_need("cohort_mendeley"))
    samples = (D.load_mendeley(trad, "ransomware") + D.load_mendeley(trad, "goodware"))
    D.annotate_cohort(samples, coh)
    kept, _ = D.filter_cohort(samples)
    counts = {(s.split, s.label) for s in kept}
    assert counts == {("train", 0), ("train", 1), ("test", 0), ("test", 1)}
    n = {(sp, lb): sum(1 for s in kept if s.split == sp and s.label == lb)
         for sp, lb in counts}
    assert n[("train", 1)] == 900 and n[("test", 1)] == 362
    assert n[("train", 0)] == 1109 and n[("test", 0)] == 129


def test_real_revised_corpus_matches_the_shared_cohort_module():
    """`cnn_vit_pipeline/cohort.py` is the shared definition the EMBER and
    CNN-ViT harnesses score against. Exp C's membership must equal it, or the
    three pipelines are not being compared on the same samples."""
    pytest.importorskip("pandas")
    root = _need("llm_features_revised")
    try:
        from cnn_vit_pipeline import cohort as shared
    except Exception as exc:                         # pragma: no cover
        pytest.skip(f"shared cohort module unavailable: {exc}")
    if not shared.COHORT_MENDELEY.exists():
        pytest.skip("cohort CSVs not present")
    df = shared.load_split("mendeley")
    want = df.groupby(["split", "label"]).size().to_dict()
    ours = D.load_mendeley(root, "ransomware") + D.load_mendeley(root, "goodware")
    got = {}
    for s in ours:
        got[(s.split, s.label)] = got.get((s.split, s.label), 0) + 1
    assert got == want


def test_real_expB_membership_intersected_with_the_cohort():
    """expB_cohort's goodware: expB's committed membership, cohort-filtered.
    1,116/131 in, 1,113/127 out."""
    committed = REPO / "results" / "expB" / "splits.csv"
    if not committed.is_file():
        pytest.skip("results/expB/splits.csv not present")
    pool_dir = _need("balanced_goodware")
    cfg = _cfg()
    pool = D.load_balanced(pool_dir, Path(cfg["paths"]["balanced_manifest"]),
                           cfg["split"]["goodware_group_field"],
                           cfg["split"].get("goodware_ungrouped_entries", []))
    chosen, rep = D.reuse_split(pool, committed, "balanced_goodware")
    assert (rep["selected_train"], rep["selected_test"]) == (1116, 131)
    D.annotate_cohort(chosen, D.load_cohort(_need("cohort_balanced")))
    kept, _ = D.filter_cohort(chosen)
    assert sum(1 for s in kept if s.split == "train") == 1113
    assert sum(1 for s in kept if s.split == "test") == 127
