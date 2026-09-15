"""
Acceptance tests for imports/extract_imports.py.

Run:
    python tests/make_fixtures.py --corpus ../Goodware_Balanced
    python -m pytest tests/test_imports.py -v

The PE cases run against the fixture corpus (benign binaries copied from
VirusTotal-clean Goodware_Balanced); nothing is executed, every check parses
headers.  The transcript cases run against hand-written `.asm` snippets, so
the RIP-relative arithmetic is pinned without needing a disassembler.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import pefile
import pytest

REPO = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(REPO))

from imports.extract_imports import (  # noqa: E402
    MANIFEST_FIELDS, dumps, extract_dir, extract_pe, format_va,
    indirect_targets, main, parse_asm, resolve_asm, write_json,
)

SYMBOL_RE = re.compile(r"^[^!]+![^!]+$")
VA_RE = re.compile(r"^0x[0-9a-f]+$")

needs_fixtures = pytest.mark.skipif(
    not (FIXTURES / "pe32.bin").is_file(),
    reason="fixtures not built; run tests/make_fixtures.py first",
)


# --------------------------------------------------------------------------
# record shape
# --------------------------------------------------------------------------

@needs_fixtures
@pytest.mark.parametrize("name", ["pe32.bin", "pe64.bin", "dll64.bin"])
def test_record_shape(name):
    rec = extract_pe((FIXTURES / name).read_bytes())
    assert set(rec) == {"imports", "iat", "dll_count", "func_count", "error"}
    assert rec["error"] == ""
    assert rec["func_count"] == len(rec["imports"]) > 0
    assert rec["dll_count"] > 0
    assert rec["dll_count"] <= rec["func_count"]
    assert len(set(rec["imports"])) == len(rec["imports"])


@needs_fixtures
@pytest.mark.parametrize("name", ["pe32.bin", "pe64.bin", "dll64.bin"])
def test_names_are_lowercase_and_dll_prefixed(name):
    rec = extract_pe((FIXTURES / name).read_bytes())
    for sym in rec["imports"]:
        assert sym == sym.lower(), sym
        assert SYMBOL_RE.match(sym), sym
        dll, func = sym.split("!", 1)
        assert dll.endswith(".dll") or "." in dll, dll
        assert func, sym
        if func.startswith("#"):
            assert func[1:].isdigit(), sym


@needs_fixtures
@pytest.mark.parametrize("name", ["pe32.bin", "pe64.bin", "dll64.bin"])
def test_iat_addresses(name):
    """Keys are `0x` + lowercase unpadded hex VAs inside the mapped image,
    and every value is one of the strings in `imports`."""
    path = FIXTURES / name
    rec = extract_pe(path.read_bytes())
    pe = pefile.PE(str(path), fast_load=True)
    base, span = pe.OPTIONAL_HEADER.ImageBase, pe.OPTIONAL_HEADER.SizeOfImage
    pe.close()

    assert rec["iat"], "fixture has imports but no IAT slots"
    symbols = set(rec["imports"])
    for key, value in rec["iat"].items():
        assert VA_RE.match(key), key
        assert key == format_va(int(key, 16)), key      # unpadded round-trip
        assert base <= int(key, 16) < base + span, key
        assert value in symbols, value


@needs_fixtures
def test_ordinal_imports_are_hash_numbered():
    """Whatever the fixtures happen to import by ordinal must render as
    `dll!#123`; if none do, at least assert the encoder on a stub."""
    seen = False
    for name in ("pe32.bin", "pe64.bin", "dll64.bin", "no_exec.bin"):
        for sym in extract_pe((FIXTURES / name).read_bytes())["imports"]:
            if "!#" in sym:
                seen = True
                assert re.match(r"^[^!]+!#\d+$", sym), sym
    if not seen:
        from imports.extract_imports import _symbol

        class _Ord:
            import_by_ordinal, name, ordinal = True, None, 123
        assert _symbol("ws2_32.dll", _Ord()) == "ws2_32.dll!#123"


@needs_fixtures
def test_unparseable_and_non_pe():
    with pytest.raises(Exception):
        extract_pe((FIXTURES / "garbage_mz.bin").read_bytes())
    assert (FIXTURES / "notpe.txt").read_bytes()[:2] != b"MZ"


@needs_fixtures
def test_dotnet_has_the_single_mscoree_stub():
    rec = extract_pe((FIXTURES / "dotnet.bin").read_bytes())
    assert rec["error"] == ""
    assert all(s == s.lower() for s in rec["imports"])
    if rec["imports"]:
        assert any("mscoree" in s for s in rec["imports"]), rec["imports"]


# --------------------------------------------------------------------------
# directory walk + manifest
# --------------------------------------------------------------------------

@needs_fixtures
def test_manifest_has_one_row_per_input(tmp_path):
    records, rows = extract_dir(FIXTURES, set(), progress_every=0)
    inputs = [p for p in FIXTURES.rglob("*") if p.is_file()]
    assert len(rows) == len(inputs)
    assert {r["rel_path"] for r in rows} == {
        p.relative_to(FIXTURES).as_posix() for p in inputs}
    assert all(set(r) == set(MANIFEST_FIELDS) for r in rows)
    # notpe.txt is the one non-MZ fixture and gets a row but no JSON record.
    statuses = {r["rel_path"]: r["status"] for r in rows}
    assert statuses["notpe.txt"] == "not_pe"
    assert "garbage_mz.bin" in statuses
    assert statuses["garbage_mz.bin"] in ("parse_error", "no_imports")
    assert all(len(sha) == 64 for sha in records)


@needs_fixtures
def test_excludes_and_hidden_dirs_are_pruned(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "keep").mkdir(parents=True)
    (corpus / "_upx_packed").mkdir()
    (corpus / ".tools").mkdir()
    data = (FIXTURES / "pe32.bin").read_bytes()
    for sub in ("keep", "_upx_packed", ".tools"):
        (corpus / sub / "a.bin").write_bytes(data)

    _, rows = extract_dir(corpus, {"_upx_packed"}, progress_every=0)
    assert [r["rel_path"] for r in rows] == ["keep/a.bin"]


@needs_fixtures
def test_cli_end_to_end(tmp_path):
    out = tmp_path / "out.json"
    man = tmp_path / "out.csv"
    rc = main(["--in", str(FIXTURES), "--out", str(out), "--manifest", str(man)])
    assert rc == 0

    records = json.loads(out.read_text(encoding="utf-8"))
    with man.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == MANIFEST_FIELDS
    assert len(rows) == len([p for p in FIXTURES.rglob("*") if p.is_file()])
    pe_shas = {r["sha256"] for r in rows if r["status"] != "not_pe"}
    assert set(records) == pe_shas
    for rec in records.values():
        assert set(rec) == {"imports", "iat", "dll_count", "func_count", "error"}


def test_iat_is_spilled_when_the_json_is_large(tmp_path):
    records = {f"{i:064x}": {"imports": ["kernel32.dll!createfilew"],
                             "iat": {"0x4291f0": "kernel32.dll!createfilew"},
                             "dll_count": 1, "func_count": 1, "error": ""}
               for i in range(200)}
    out, side = write_json(dict(records), tmp_path / "big.json",
                           split_mb=len(dumps(records).encode()) / 2e6)
    assert side == tmp_path / "big.iat.json"
    main_blob = json.loads(out.read_text(encoding="utf-8"))
    assert all("iat" not in rec for rec in main_blob.values())
    assert json.loads(side.read_text(encoding="utf-8"))[f"{0:064x}"] == \
        {"0x4291f0": "kernel32.dll!createfilew"}

    out2, side2 = write_json(dict(records), tmp_path / "small.json",
                             split_mb=50.0)
    assert side2 is None
    assert json.loads(out2.read_text(encoding="utf-8"))[f"{0:064x}"]["iat"]


# --------------------------------------------------------------------------
# .asm transcript resolution
# --------------------------------------------------------------------------

X86_ASM = """\
0x401000:  push\tebp
0x401001:  mov\tebp, esp
0x401003:  call\tdword ptr [0x4291f0]
0x401009:  call\tdword ptr [0x4291f4]
0x40100f:  call\teax
0x401011:  jmp\tdword ptr [0x4291f0]
0x401017:  ret\t
"""

# `call qword ptr [rip + 0xd639]` at 0x140001020 is 7 bytes long, so the next
# transcript line is at 0x140001027 and the slot is 0x140001027 + 0xd639.
X64_ASM = """\
0x140001010:  sub\trsp, 0x28
0x140001020:  call\tqword ptr [rip + 0xd639]
0x140001027:  nop\tdword ptr [rax + rax]
0x14000102c:  call\tqword ptr [rip - 0x20]
0x140001032:  bnd jmp\tqword ptr [rip + 0x10]
0x140001039:  call\tqword ptr [rip + 0x1]
"""


def test_parse_asm_reads_address_mnemonic_operands():
    ins = parse_asm(X86_ASM)
    assert ins[0] == (0x401000, "push", "ebp")
    assert ins[2] == (0x401003, "call", "dword ptr [0x4291f0]")
    assert len(ins) == 7


def test_absolute_indirect_targets():
    found = indirect_targets(parse_asm(X86_ASM))
    assert found == [(0x401003, "abs", 0x4291f0),
                     (0x401009, "abs", 0x4291f4),
                     (0x401011, "abs", 0x4291f0)]
    assert all(kind == "abs" for _, kind, _ in found)


def test_rip_relative_targets_use_the_next_line_address():
    found = indirect_targets(parse_asm(X64_ASM))
    assert found[0] == (0x140001020, "rip", 0x140001027 + 0xD639)
    # negative displacement, against the next line at 0x140001032
    assert found[1] == (0x14000102C, "rip", 0x140001032 - 0x20)
    # `bnd jmp` is still a memory-indirect branch; next line is 0x140001039
    assert found[2] == (0x140001032, "rip", 0x140001039 + 0x10)
    # last line of the transcript: no next address, so no target
    assert found[3] == (0x140001039, "rip", None)


def test_resolve_asm_against_an_iat_map():
    iat = {"0x4291f0": "kernel32.dll!createfilew",
           "0x4291f4": "kernel32.dll!#123"}
    resolved, total, misses = resolve_asm(X86_ASM, iat)
    assert (resolved, total) == (3, 3)
    assert misses == []

    resolved, total, misses = resolve_asm(X86_ASM, {"0x4291f0": "a.dll!b"})
    assert (resolved, total) == (2, 3)
    assert misses == [(0x401009, "abs", 0x4291f4)]

    # unresolved trailing rip target is reported, not silently dropped
    _, _, misses = resolve_asm(X64_ASM, {})
    assert (0x140001039, "rip", None) in misses


@needs_fixtures
def test_iat_map_resolves_a_real_transcript_slot(tmp_path):
    """A round trip with no disassembler: take a real IAT slot out of a
    fixture and synthesise the call that would reference it."""
    rec = extract_pe((FIXTURES / "pe64.bin").read_bytes())
    slot = int(sorted(rec["iat"])[0], 16)
    call_at, next_at = 0x1000, 0x1007
    asm = (f"0x{call_at:x}:  call\tqword ptr [rip + 0x{slot - next_at:x}]\n"
           f"0x{next_at:x}:  ret\t\n")
    resolved, total, misses = resolve_asm(asm, rec["iat"])
    assert (resolved, total, misses) == (1, 1, [])
