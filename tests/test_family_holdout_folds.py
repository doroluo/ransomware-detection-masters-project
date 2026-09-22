"""Invariants of the family-holdout fold definition (family_holdout/folds.py)."""
from __future__ import annotations

import collections
import csv
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from family_holdout import folds  # noqa: E402

OUT = REPO / "results" / "family_holdout"
pytestmark = pytest.mark.skipif(not (OUT / "folds_mendeley.csv").is_file(),
                                reason="run python family_holdout/folds.py first")


def _load(ds):
    with (OUT / f"folds_{ds}.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_assign_greedy_is_deterministic_and_balanced():
    counts = {f"f{i}": (i * 7) % 23 + 1 for i in range(40)}
    a = folds.assign_greedy(counts)
    b = folds.assign_greedy(dict(reversed(list(counts.items()))))
    assert a == b
    sizes = collections.Counter()
    for name, f in a.items():
        sizes[f] += counts[name]
    assert set(a.values()) == set(range(folds.K))
    assert max(sizes.values()) - min(sizes.values()) <= max(counts.values())


@pytest.mark.parametrize("ds", ["mendeley", "balanced"])
def test_every_family_in_exactly_one_fold(ds):
    rows = _load(ds)
    fam_folds = collections.defaultdict(set)
    for r in rows:
        if r["label"] == "1":
            fam_folds[r["family"]].add(r["fold"])
    assert len(fam_folds) == 38
    assert all(len(v) == 1 for v in fam_folds.values())
    assert "thanos" not in fam_folds and "nightsky" not in fam_folds


@pytest.mark.parametrize("ds", ["mendeley", "balanced"])
def test_goodware_groups_never_straddle_folds(ds):
    rows = _load(ds)
    grp_folds = collections.defaultdict(set)
    for r in rows:
        if r["label"] == "0":
            grp_folds[r["group"]].add(r["fold"])
    assert all(len(v) == 1 for v in grp_folds.values())


def test_ransomware_folds_identical_across_datasets():
    a = {r["sha256"]: r["fold"] for r in _load("mendeley") if r["label"] == "1"}
    b = {r["sha256"]: r["fold"] for r in _load("balanced") if r["label"] == "1"}
    assert a == b and len(a) == 1266


@pytest.mark.parametrize("ds,n_good", [("mendeley", 1243), ("balanced", 1337)])
def test_pool_sizes_and_no_duplicate_sha(ds, n_good):
    rows = _load(ds)
    shas = [r["sha256"] for r in rows]
    assert len(shas) == len(set(shas))
    assert sum(r["label"] == "0" for r in rows) == n_good
    assert set(r["fold"] for r in rows) == {str(i) for i in range(folds.K)}


def test_rebuild_is_byte_identical(tmp_path):
    folds.build(tmp_path)
    for name in ("folds_mendeley.csv", "folds_balanced.csv", "families.csv"):
        assert (tmp_path / name).read_bytes() == (OUT / name).read_bytes()


# ---------------------------------------------------------------------------
# cross-corpus rules (folds.py v2): aliases, dedup, label collision, balancing
# ---------------------------------------------------------------------------
def _cohort(path, rows):
    cols = ["corpus", "sha256", "set", "label", "family", "filename", "arch", "tag", "in_cohort", "exclude_reason"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({**{c: "" for c in cols}, "set": "mal_train", "label": "1", "tag": "plain",
                        "in_cohort": "1", **r})


def test_normalise_family_maps_aliases_onto_mendeley_spelling():
    assert folds.normalise_family("Sodinokibi") == "revil"
    assert folds.normalise_family("Play") == "playcrypt"
    assert folds.normalise_family("Black Basta") == "blackbasta"
    assert folds.normalise_family("conti") == "conti"
    assert folds.normalise_family("MedusaLocker") == "medusalocker"     # not Medusa
    assert folds.normalise_family("Nemty") == "nemty"                   # not Nefilim


def test_extra_corpus_pools_families_and_drops_repeated_sha(tmp_path, monkeypatch):
    mend = tmp_path / "m.csv"
    extra = tmp_path / "x.csv"
    _cohort(mend, [{"sha256": f"m{i:063d}", "family": "conti", "arch": "x86"} for i in range(6)]
                  + [{"sha256": "shared" + "0" * 58, "family": "hive", "arch": "x64"}])
    _cohort(extra, [{"sha256": f"x{i:063d}", "family": "Conti", "arch": "x64"} for i in range(3)]
                   + [{"sha256": "SHARED" + "0" * 58, "family": "hive", "arch": "x64"}]   # same sha, upper case
                   + [{"sha256": "y" * 64, "family": "hive", "arch": "x64", "in_cohort": "0"}])
    monkeypatch.setitem(folds.COHORT, "mendeley", mend)
    rows, fam_fold = folds.ransomware_rows({"vs": extra})
    assert len(rows) == 10                       # 7 mendeley + 3 vs; the shared sha and the out-of-cohort row dropped
    assert {r["family"] for r in rows} == {"conti", "hive"}
    assert {r["corpus"] for r in rows if r["family"] == "conti"} == {"mendeley", "vs"}
    assert len({r["fold"] for r in rows if r["family"] == "conti"}) == 1   # one family, one fold, across corpora


def test_check_dataset_rejects_sha_with_both_labels_and_bad_arch():
    good = {"sha256": "a" * 64, "label": 0, "arch": "x64"}
    with pytest.raises(ValueError, match="both goodware and ransomware"):
        folds.check_dataset("t", [good, {"sha256": "a" * 64, "label": 1, "arch": "x86"}])
    with pytest.raises(ValueError, match="listed twice"):
        folds.check_dataset("t", [good, dict(good)])
    with pytest.raises(ValueError, match="unexpected arch"):
        folds.check_dataset("t", [good, {"sha256": "b" * 64, "label": 1, "arch": "arm64"}])


def test_arch_aware_balancer_spreads_x64_across_folds():
    # ten families of 20 files; the first five are all-x64, the rest x86-only
    counts = {f"f{i}": (20, 20 if i < 5 else 0) for i in range(10)}
    a = folds.assign_greedy(counts)
    x64_per_fold = collections.Counter(a[f"f{i}"] for i in range(5))
    assert max(x64_per_fold.values()) == 1        # one x64 family per fold, not five in one
    plain = folds.assign_greedy({k: v[0] for k, v in counts.items()})
    assert set(plain.values()) == set(range(folds.K))


def test_pool_small_keeps_tiny_families_together(tmp_path, monkeypatch):
    mend = tmp_path / "m.csv"
    rows = [{"sha256": f"a{i:063d}", "family": "big", "arch": "x86"} for i in range(30)]
    rows += [{"sha256": f"b{i:063d}", "family": f"tiny{i}", "arch": "x86"} for i in range(4)]
    _cohort(mend, rows)
    monkeypatch.setitem(folds.COHORT, "mendeley", mend)
    r_default, _ = folds.ransomware_rows()
    r_pooled, _ = folds.ransomware_rows(pool_small=True)
    assert len({r["fold"] for r in r_pooled if r["family"].startswith("tiny")}) == 1
    assert len({r["fold"] for r in r_default if r["family"].startswith("tiny")}) == 4
    assert {r["family"] for r in r_pooled} == {r["family"] for r in r_default}   # names kept for LOFO
