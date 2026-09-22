"""tools/arch_balance.py: the x86/x64 report, the matched selection, the need estimate."""
from __future__ import annotations

import collections
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools import arch_balance as A  # noqa: E402


def _rows():
    rows = []
    i = 0
    def add(n, label, arch, family, corpus):
        nonlocal i
        for _ in range(n):
            rows.append({"sha256": f"{i:064d}", "label": label, "arch": arch,
                         "family": family if label else "goodware", "corpus": corpus})
            i += 1
    add(40, 1, "x86", "gandcrab", "m"); add(30, 1, "x86", "phobos", "m")
    add(20, 1, "x64", "hive", "m"); add(5, 1, "x86", "hive", "m")
    add(10, 1, "x64", "akira", "vs"); add(2, 1, "x86", "akira", "vs")
    add(50, 0, "x86", "", "m"); add(120, 0, "x64", "", "b")
    return rows


def test_report_and_x86_rule():
    rows = _rows()
    c = A.counts(rows)
    assert c[1] == {"x86": 77, "x64": 30} and c[0] == {"x86": 50, "x64": 120}
    text = A.report(rows)
    assert "x86-rule accuracy" in text and "akira" in text and "vs" in text


def test_match_gives_both_classes_identical_arch_counts():
    rows = _rows()
    keep = A.match(rows, seed=3)
    kept = [r for r in rows if keep[r["sha256"]]]
    c = A.counts(kept)
    assert c[0] == c[1]                                   # identical (x86, x64) counts
    assert c[1]["x64"] == 30 and c[1]["x86"] == 50        # scarce arch fully kept
    # thinning is spread over families: every family survives, the x64 files all do
    fam_kept = collections.Counter(r["family"] for r in kept if r["label"] == 1)
    assert set(fam_kept) == {"gandcrab", "phobos", "hive", "akira"}
    assert all(keep[r["sha256"]] for r in rows if r["label"] == 1 and r["arch"] == "x64")
    assert keep == A.match(rows, seed=3)                  # deterministic


def test_need_counts_missing_x64():
    rows = _rows()
    text = A.need(rows, 0.5)
    assert "needs 47 more x64 ransomware files" in text    # 77 x86 -> 77 x64 wanted, have 30
    assert "hive" in text and "akira" in text and "gandcrab" not in text.split("families that ship")[1]
