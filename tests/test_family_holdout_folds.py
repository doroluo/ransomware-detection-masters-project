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
