"""asm_tool/rewrite_streams.py: operand splitting, operand classes, IAT
resolution (absolute and RIP-relative against the next line's address),
prefixes as their own token, one output line per section, and token-count
parity between the mn_api stream and a whitespace split of the mnemonics."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from asm_tool import rewrite_streams as R  # noqa: E402

TRANSCRIPT = """; section .text va=0x401000 size=64
0x00401000:  push\tebx
0x00401001:  mov\tdword ptr [edi + 4], 0
0x00401008:  push\t0x4203ac
0x0040100d:  call\tdword ptr [0x41c008]
0x00401013:  mov\tesi, dword ptr [0x41c010]
0x00401019:  call\tesi
0x0040101b:  je\t0x4010a5
0x0040101d:  rep movsb
0x0040101f:  call\tqword ptr [rip + 0x100]
0x00401025:  .skip\t3 bytes
0x00401028:  bnd jmp\tqword ptr [rip - 0x28]
0x0040102e:  ret
; section .text2 va=0x402000 size=8
0x00402000:  xor\teax, eax
0x00402002:  call\tqword ptr [rip + 0x10]
"""
IAT = {"0x41c008": "kernel32.dll!createfilew", "0x41c010": "kernel32.dll!getprocaddress",
       "0x401125": "advapi32.dll!cryptencrypt",      # 0x401025 + 0x100
       "0x401006": "user32.dll!messageboxw"}          # 0x40102e - 0x28


def _run(tmp_path):
    src = tmp_path / "x.asm"; src.write_text(TRANSCRIPT, encoding="utf-8")
    rw = R.Rewriter(IAT)
    st = rw.run(src, tmp_path / "api.txt", tmp_path / "cls.txt")
    return st, (tmp_path / "api.txt").read_text().splitlines(), (tmp_path / "cls.txt").read_text().splitlines()


def test_split_operands_respects_brackets():
    assert R.split_operands("dword ptr [edi + 4], 0") == ["dword ptr [edi + 4]", "0"]
    assert R.split_operands("xmm0, xmmword ptr [rax + rcx*8], 3") == ["xmm0", "xmmword ptr [rax + rcx*8]", "3"]
    assert R.split_operands("") == []


def test_classify():
    assert R.classify("mov", "dword ptr [edi + 4]", None) == "mem"
    assert R.classify("mov", "0", None) == "imm"
    assert R.classify("je", "0x4010a5", None) == "addr"
    assert R.classify("call", "0x401000", None) == "addr"
    assert R.classify("push", "ebx", None) == "reg"
    assert R.classify("call", "dword ptr [0x41c008]", "k!f") == "api"


def test_streams_and_resolution(tmp_path):
    st, api, cls = _run(tmp_path)
    assert len(api) == 2 and len(cls) == 2, "one line per section"
    assert api[0].split() == ["push", "mov", "push", "call:kernel32.dll!createfilew",
                              "mov:kernel32.dll!getprocaddress", "call", "je", "rep", "movsb",
                              "call:advapi32.dll!cryptencrypt", "bnd", "jmp:user32.dll!messageboxw", "ret"]
    assert api[1].split() == ["xor", "call"]          # rip at end of section: unresolved
    assert cls[0].split() == ["push_reg", "mov_mem_imm", "push_imm", "call_api", "kernel32.dll!createfilew",
                              "mov_reg_api", "kernel32.dll!getprocaddress", "call_reg", "je_addr", "rep", "movsb",
                              "call_api", "advapi32.dll!cryptencrypt", "bnd", "jmp_api", "user32.dll!messageboxw", "ret"]
    assert cls[1].split() == ["xor_reg_reg", "call_mem"]
    assert st["n_sections"] == 2 and st["n_insns"] == 13   # prefixes and .skip are not instructions
    assert st["n_mem_indirect_branch"] == 4 and st["n_api_resolved"] == 3 and st["n_api_any"] == 4


def test_mn_api_has_one_token_per_mnemonic_token(tmp_path):
    _, api, _ = _run(tmp_path)
    mn_tokens = ["push", "mov", "push", "call", "mov", "call", "je", "rep", "movsb", "call", "bnd", "jmp", "ret"]
    assert [t.split(":")[0] for t in api[0].split()] == mn_tokens


def test_no_iat_gives_the_plain_mnemonic_stream(tmp_path):
    src = tmp_path / "y.asm"; src.write_text(TRANSCRIPT, encoding="utf-8")
    R.Rewriter({}).run(src, tmp_path / "a.txt", tmp_path / "c.txt")
    assert ":" not in (tmp_path / "a.txt").read_text()
