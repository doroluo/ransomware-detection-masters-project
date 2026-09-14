"""
Acceptance tests for asm_parse.py and asm_tool/asm_to_opcodes.py.

Run:
    python tests/make_fixtures.py --corpus ../Goodware_Balanced
    python -m pytest tests/ -v

Fixtures are benign binaries copied from the VirusTotal-clean Goodware_Balanced
corpus. Nothing is executed: every check parses headers or disassembles bytes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(REPO))

from asm_parse import FIELDS, analyze, disassemble  # noqa: E402
from asm_tool.asm_to_opcodes import convert_line  # noqa: E402

MAX_INSN = 20_000

# The per-file row the worker brief specifies. `instructions` is kept as the
# original name for `n_instructions`.
PLAN_COLUMNS = ["sha256", "machine", "bitness", "is_dotnet", "packed_flag",
                "packer_guess", "sections_disassembled", "n_instructions",
                "status", "error"]

pytestmark = pytest.mark.skipif(
    not (FIXTURES / "pe32.bin").is_file(),
    reason="fixtures not built; run tests/make_fixtures.py first",
)


def run(name):
    return disassemble(FIXTURES / name, MAX_INSN)


# --- architecture detection -------------------------------------------------

def test_pe32_detected_as_x86():
    status, lines, arch, n = run("pe32.bin")
    assert status in ("ok", "truncated")
    assert arch == "x86"
    assert n > 0 and len(lines) == n


def test_pe64_detected_as_x64():
    status, lines, arch, n = run("pe64.bin")
    assert status in ("ok", "truncated")
    assert arch == "x64"
    assert n > 0


def test_dll_disassembles():
    status, _, arch, n = run("dll64.bin")
    assert status in ("ok", "truncated")
    assert arch == "x64" and n > 0


def test_x86_and_x64_do_not_collapse_to_one_mode():
    """A 32/64 mix-up is silent - both decode - so compare against a known
    64-bit-only encoding rather than trusting the status field."""
    _, l32, _, _ = run("pe32.bin")
    _, l64, _, _ = run("pe64.bin")
    rex64 = sum(1 for x in l64 if " r8" in x or " r9" in x or " rsp" in x or " rbp" in x)
    rex32 = sum(1 for x in l32 if " r8" in x or " r9" in x or " rsp" in x or " rbp" in x)
    assert rex64 > 0, "x64 disassembly produced no 64-bit registers"
    assert rex32 == 0, "x86 disassembly produced 64-bit registers - wrong CS_MODE"


# --- the flags that keep garbage out of the corpus --------------------------

def test_dotnet_flagged_not_disassembled():
    status, lines, _, n = run("dotnet.bin")
    assert status == "dotnet_ilonly"
    assert lines == [] and n == 0


def test_upx_flagged_not_disassembled():
    status, lines, _, _ = run("upx.bin")
    assert status == "upx_packed"
    assert lines == []


def test_no_exec_section_flagged():
    status, lines, _, _ = run("no_exec.bin")
    assert status == "no_exec_section"
    assert lines == []


# --- malformed input produces a status row, never a crash -------------------

@pytest.mark.parametrize("name", ["truncated.bin", "garbage_mz.bin", "notpe.txt"])
def test_malformed_returns_status_not_exception(name):
    status, lines, _, n = run(name)
    assert isinstance(status, str) and status
    assert status not in ("ok", "truncated") or n > 0
    if status.startswith(("pe_error", "not_pe", "read_error")):
        assert lines == []


def test_notpe_rejected_before_pefile():
    status, _, _, _ = run("notpe.txt")
    assert status == "not_pe"


# --- output format contract with asm_parser.py ------------------------------

def test_line_format_matches_parse_asm_line():
    """asm_parser.parse_asm_line splits on ':' then tab; asm_parse must emit
    exactly `0x<hex>:  <mnemonic>\t<operands>`."""
    _, lines, _, _ = run("pe64.bin")
    for line in lines[:200]:
        head, sep, rest = line.partition(":")
        assert sep == ":", line
        assert head.startswith("0x"), line
        int(head, 16)
        assert rest.startswith("  "), line
        assert "\t" in rest, line


def test_instruction_cap_is_enforced():
    status, lines, _, _ = disassemble(FIXTURES / "pe64.bin", 10)
    assert len(lines) <= 10
    if status == "truncated":
        assert len(lines) == 10


# --- the .asm -> LLM_Features bridge ---------------------------------------

def test_convert_line_strips_address_and_tab():
    assert convert_line("0x00401000:  mov\teax, ebx") == "mov eax, ebx"
    assert convert_line("0x0000000180001000:  ret\t") == "ret"
    assert convert_line("") is None
    assert convert_line("   \n") is None


def test_convert_line_leaves_extract_py_format_untouched():
    """Idempotent, so running the bridge twice cannot corrupt a corpus."""
    for s in ("mov eax, ebx", "ret", "call qword ptr [rip + 0x1234]"):
        assert convert_line(s) == s
        assert convert_line(convert_line(s)) == s


def test_bridge_output_has_no_address_prefix(tmp_path):
    """The shortcut this whole script exists to prevent."""
    _, lines, _, _ = run("pe64.bin")
    converted = [convert_line(x) for x in lines[:500]]
    assert all(c and not c.startswith("0x") for c in converted)


# --- packing flags ----------------------------------------------------------
#
# packed_flag is advisory metadata. It must never change which files get
# disassembled: the UPX fixture is skipped because of its section name (the
# pre-existing upx_packed status), and the high-entropy fixture is
# disassembled normally while still being flagged.

def test_upx_fixture_flagged_packed():
    rec, _ = analyze(FIXTURES / "upx.bin", MAX_INSN)
    assert rec["packed_flag"] == 1
    assert "upx" in rec["packer_guess"]


def test_high_entropy_fixture_flagged_but_still_disassembled():
    """Synthetic positive for the entropy branch: pe64.bin with its code
    section overwritten by random bytes. Never packed, never executed."""
    fx = FIXTURES / "highentropy.bin"
    if not fx.is_file():
        pytest.skip("highentropy.bin not built; rerun tests/make_fixtures.py")
    rec, _ = analyze(fx, MAX_INSN)
    assert rec["packed_flag"] == 1
    assert "high_entropy" in rec["packer_guess"]
    assert rec["status"] in ("ok", "truncated", "empty_disassembly"), \
        "the packing flag must not divert a file out of disassembly"
    assert rec["arch"] == "x64"


@pytest.mark.parametrize("name", ["pe32.bin", "pe64.bin", "dll64.bin"])
def test_ordinary_binaries_not_flagged_packed(name):
    rec, _ = analyze(FIXTURES / name, MAX_INSN)
    assert rec["packed_flag"] == 0, rec["packer_guess"]
    assert rec["packer_guess"] == ""


# --- manifest schema --------------------------------------------------------

def test_analyze_record_carries_every_plan_column():
    rec, _ = analyze(FIXTURES / "pe64.bin", MAX_INSN)
    for col in PLAN_COLUMNS:
        assert col in rec, col
    assert len(rec["sha256"]) == 64
    assert rec["machine"] == "0x8664"
    assert rec["bitness"] == 64
    assert rec["is_dotnet"] == 0
    assert rec["sections_disassembled"] >= 1
    assert rec["n_instructions"] == rec["instructions"] > 0


def test_disassemble_wrapper_agrees_with_analyze():
    """consistency_check.py and half these tests use the four-tuple API."""
    rec, lines = analyze(FIXTURES / "pe32.bin", MAX_INSN)
    status, wlines, arch, n = disassemble(FIXTURES / "pe32.bin", MAX_INSN)
    assert (status, arch, n) == (rec["status"], rec["arch"], rec["n_instructions"])
    assert wlines == lines


# --- CLI smoke test ---------------------------------------------------------

def run_cli(out, *extra, cap="500"):
    return subprocess.run(
        [sys.executable, str(REPO / "asm_parse.py"),
         "--in-dir", str(FIXTURES), "--out-dir", str(out),
         "--max-instructions", cap, "--jobs", "1", *extra],
        capture_output=True, text=True, timeout=600)


def read_manifest(out):
    import csv
    with (out / "asm_manifest.csv").open(encoding="utf-8", newline="") as fh:
        return {r["rel_path"]: r for r in csv.DictReader(fh)}


def n_fixture_files():
    return sum(1 for p in FIXTURES.iterdir()
               if p.is_file() and p.suffix.lower() not in {".csv", ".md", ".json"})


def asm_hashes(out):
    import hashlib
    return {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob("*.asm"))}


def test_asm_parse_cli_runs(tmp_path):
    out = tmp_path / "asm"
    r = run_cli(out)
    assert r.returncode == 0, r.stderr
    manifest = out / "asm_manifest.csv"
    assert manifest.is_file()
    body = manifest.read_text(encoding="utf-8").splitlines()
    # header + one row per fixture, minus the .txt that asm_parse skips by suffix
    n_fixtures = n_fixture_files()
    assert len(body) - 1 == n_fixtures, f"{len(body)-1} rows for {n_fixtures} fixtures"


def test_cli_manifest_has_every_plan_column(tmp_path):
    out = tmp_path / "asm"
    assert run_cli(out).returncode == 0
    rows = read_manifest(out)
    header = list(next(iter(rows.values())))
    for col in ["rel_path", "asm_path", "status", "arch", "instructions"]:
        assert col in header, f"original column {col} dropped"
    for col in PLAN_COLUMNS:
        assert col in header, f"plan column {col} missing"
    assert header == FIELDS
    pe64 = rows["pe64.bin"]
    assert len(pe64["sha256"]) == 64
    assert pe64["bitness"] == "64" and pe64["machine"] == "0x8664"
    assert pe64["n_instructions"] == pe64["instructions"]
    assert rows["upx.bin"]["packed_flag"] == "1"


# --- resume must never degrade the manifest ---------------------------------

def test_resume_preserves_status_arch_and_counts(tmp_path):
    """The bug this guards: a second run finding the .asm already there and
    writing a placeholder row, losing ok/truncated, arch and the count."""
    out = tmp_path / "asm"
    assert run_cli(out, cap="10").returncode == 0
    first = read_manifest(out)
    assert any(r["status"] == "truncated" for r in first.values()), \
        "cap of 10 should truncate something, else the test proves nothing"

    before = asm_hashes(out)
    r2 = run_cli(out, cap="10")
    assert r2.returncode == 0, r2.stderr
    second = read_manifest(out)

    assert set(first) == set(second)
    assert len(second) == n_fixture_files()
    for rel, row in second.items():
        assert row["status"] != "already_done"
        for col in ("status", "arch", "instructions", "n_instructions",
                    "sha256", "machine", "bitness", "is_dotnet",
                    "packed_flag", "packer_guess", "asm_path"):
            assert row[col] == first[rel][col], f"{rel}.{col} changed on resume"
    assert any(r["source"] == "resumed" for r in second.values())
    assert asm_hashes(out) == before, "resume rewrote .asm files"


def test_resume_repairs_a_legacy_already_done_manifest(tmp_path):
    """The manifest this fix had to repair: five columns, `already_done` in
    place of ok/truncated, blank arch and counts. A resume run must rebuild it
    from the binaries and the .asm files, not carry the placeholder forward."""
    import csv
    out = tmp_path / "asm"
    assert run_cli(out, cap="10").returncode == 0
    truth = read_manifest(out)

    legacy = [{"rel_path": r["rel_path"], "asm_path": r["asm_path"],
               "status": "already_done" if r["asm_path"] else r["status"],
               "arch": "", "instructions": ""} for r in truth.values()]
    with (out / "asm_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["rel_path", "asm_path", "status",
                                           "arch", "instructions"])
        w.writeheader()
        w.writerows(legacy)

    assert run_cli(out, cap="10").returncode == 0
    repaired = read_manifest(out)
    assert not any(r["status"] == "already_done" for r in repaired.values())
    for rel, row in repaired.items():
        for col in ("status", "arch", "instructions", "n_instructions",
                    "sha256", "machine", "bitness", "packed_flag"):
            assert row[col] == truth[rel][col], f"{rel}.{col} not repaired"


def test_refresh_manifest_rebuilds_rows_without_touching_asm(tmp_path):
    out = tmp_path / "asm"
    assert run_cli(out, cap="500").returncode == 0
    first = read_manifest(out)
    before = asm_hashes(out)

    r = run_cli(out, "--refresh-manifest", cap="500")
    assert r.returncode == 0, r.stderr
    refreshed = read_manifest(out)

    assert asm_hashes(out) == before, "--refresh-manifest wrote to the .asm tree"
    assert set(refreshed) == set(first)
    for rel, row in refreshed.items():
        assert row["source"] == "refreshed"
        assert "asm_mismatch" not in row["error"], rel
        assert "asm_missing" not in row["error"], rel
        for col in ("status", "arch", "instructions", "n_instructions",
                    "sections_disassembled", "packed_flag", "sha256"):
            assert row[col] == first[rel][col], f"{rel}.{col} changed on refresh"


def test_refresh_manifest_detects_a_tampered_asm(tmp_path):
    """A refresh is only worth running if it can fail."""
    out = tmp_path / "asm"
    assert run_cli(out, cap="500").returncode == 0
    victim = out / "pe64.bin.asm"
    victim.write_text(victim.read_text(encoding="utf-8") + "\n0x0:  nop\t",
                      encoding="utf-8")
    assert run_cli(out, "--refresh-manifest", cap="500").returncode == 0
    assert "asm_mismatch" in read_manifest(out)["pe64.bin"]["error"]


# --- check_arch.py sees the same corpus as asm_parse.py ---------------------

def test_check_arch_excludes_the_same_paths_as_asm_parse(tmp_path):
    """Before this was fixed, check_arch.py pointed at Goodware_Balanced
    reported 1,525 PEs and 25 "UPX packed" against a 1,500-PE corpus with zero
    UPX: it was counting _upx_packed/ (the packed originals of files already
    unpacked in place) and .tools/upx.exe."""
    import shutil
    from check_arch import scan

    corpus = tmp_path / "corpus"
    (corpus / "everyday").mkdir(parents=True)
    (corpus / "_upx_packed").mkdir()
    (corpus / ".tools").mkdir()
    shutil.copy2(FIXTURES / "pe64.bin", corpus / "everyday" / "a.exe")
    shutil.copy2(FIXTURES / "upx.bin", corpus / "_upx_packed" / "a.exe")
    shutil.copy2(FIXTURES / "pe32.bin", corpus / ".tools" / "upx.exe")
    (corpus / "corpus_index.csv").write_text("sha256\n", encoding="utf-8")

    kept = scan(corpus)
    assert [r["rel_path"] for r in kept] == [str(Path("everyday") / "a.exe")]
    assert sum(r["upx"] for r in kept) == 0
    assert len(scan(corpus, skip=False)) == 4


def test_overwrite_and_refresh_are_mutually_exclusive(tmp_path):
    out = tmp_path / "asm"
    assert run_cli(out, cap="500").returncode == 0
    r = run_cli(out, "--overwrite", "--refresh-manifest", cap="500")
    assert r.returncode != 0
