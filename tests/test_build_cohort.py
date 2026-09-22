"""tools/build_cohort.py: manifest rows -> cohort rows under the plan's rules."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools import build_cohort as B  # noqa: E402


def _row(sha, family, arch="x86", tag="plain", label="1", set_="mal_train"):
    return {"sha256": sha, "set": set_, "label": label, "family": family,
            "filename": sha, "arch": arch, "tag": tag}


def test_rules_and_family_normalisation():
    rows = [
        _row("A" * 64, "LockBit", "x64"),                       # in, normalised, upper-case sha lowered
        _row("b" * 64, "Sodinokibi", "x86", "upx_unpacked"),    # in, alias -> revil
        _row("c" * 64, "conti", "x86", "packed_other"),          # out: tag
        _row("d" * 64, "conti", "x86", "dotnet"),                # out: tag
        _row("e" * 64, "conti", "arm64", "plain"),               # out: arch
        _row("f" * 64, "Thanos", "x86", "plain"),                # out: family
        _row("g" * 64, "root", "x64", "plain", label="0", set_="good_train"),  # goodware keeps no family
    ]
    out = B.build(rows, "vs", None, None)
    by = {r["sha256"][0]: r for r in out}
    assert [r["in_cohort"] for r in out] == [1, 1, 0, 0, 0, 0, 1]
    assert by["a"]["sha256"] == "a" * 64 and by["a"]["family"] == "lockbit"
    assert by["b"]["family"] == "revil"
    assert by["c"]["exclude_reason"] == "packed_other" and by["d"]["exclude_reason"] == "dotnet"
    assert by["e"]["exclude_reason"] == "arch:arm64"
    assert by["f"]["exclude_reason"] == "family:thanos"
    assert by["g"]["family"] == "goodware" and by["g"]["label"] == 0 and by["g"]["set"] == "good_train"
    assert all(r["corpus"] == "vs" for r in out)
    assert list(out[0]) == B.COLUMNS


def test_set_and_label_overrides_and_summary():
    rows = [_row("a" * 64, "akira", "x64", set_="whatever"), _row("b" * 64, "akira", "x86")]
    out = B.build(rows, "vs", "mal_train", 1)
    assert {r["set"] for r in out} == {"mal_train"}
    text = B.summary(out)
    assert "2 of 2 rows in cohort" in text and "akira" in text
    line = next(l for l in text.splitlines() if l.startswith("akira"))
    assert line.split() == ["akira", "1", "1"]
