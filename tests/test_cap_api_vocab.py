"""asm_tool/cap_api_vocab.py: rare API tokens collapse to <mnemonic>:api_rare,
kept ones survive, plain mnemonics and line structure are untouched."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from asm_tool import cap_api_vocab as C  # noqa: E402


def test_rewrite_one_collapses_only_rare_api_tokens(tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("push call:k32!a mov call:k32!b\nret jmp:u32!c\n", encoding="utf-8")
    dst = tmp_path / "out" / "a.txt"
    sha, n_api, n_rare = C._rewrite_one((str(src), str(dst), frozenset({"call:k32!a"})))
    assert (sha, n_api, n_rare) == ("a", 3, 2)
    assert dst.read_text(encoding="utf-8").splitlines() == [
        "push call:k32!a mov call:api_rare", "ret jmp:api_rare"]


def test_df_one_counts_api_tokens_once_per_file(tmp_path):
    p = tmp_path / "b.txt"
    p.write_text("call:x!y call:x!y mov call:x!z\n", encoding="utf-8")
    assert C._df_one(str(p)) == {"call:x!y", "call:x!z"}
