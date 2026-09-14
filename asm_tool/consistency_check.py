#!/usr/bin/env python3
"""
consistency_check.py - prove that asm_parse.py and extract.py produce the same
instruction stream, so LLM_Features (built with extract.py) and
asm_output/goodware_balanced (built with asm_parse.py) can be mixed in one
experiment.

They are not byte-identical by design:

  * extract.py  disassembles from `section.VirtualAddress` and emits
    "mov eax, ebx".
  * asm_parse.py disassembles from `ImageBase + section.VirtualAddress` and
    emits "0x00401000:  mov\teax, ebx".

The differing base changes the *printed* target of relative branches
(`je 0x103d` vs `je 0x40103d`) but not the decoded bytes, and
Tokenization/tokenization.py rewrites every `0x...` to `<HEX>`. So the two
streams must agree exactly on:

    1. the mnemonic sequence, and
    2. the normalized instruction sequence (after <HEX>/<OFFSET> folding).

Any disagreement on (1) is a real decoding difference and must be explained
before the corpora are combined.

    python asm_tool/consistency_check.py --corpus ../Goodware_Balanced -n 10
"""

from __future__ import annotations

import argparse
import csv
import difflib
import random
import re
import sys
from pathlib import Path

import pefile
from capstone import Cs, CS_ARCH_X86, CS_MODE_32, CS_MODE_64

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from asm_parse import disassemble  # noqa: E402
from asm_tool.asm_to_opcodes import convert_line  # noqa: E402


def normalize(line: str) -> str:
    """A local copy of Tokenization/tokenization.py:normalize_instruction.

    Duplicated deliberately: this script has to run without the tokenization
    repo on sys.path, and it is the one place where drift between the two
    would be caught by the test rather than hidden by it.
    """
    line = line.strip().lower()
    if not line:
        return ""
    line = re.sub(r"0x[0-9a-f]+", "<HEX>", line)
    line = re.sub(r"\b\d+\b", "<OFFSET>", line)
    line = re.sub(r"\[\s*(.*?)\s*\]", r"[\1]", line)
    line = re.sub(r"\s*([-+])\s*", r"\1", line)
    return re.sub(r"[\s,]+", " ", line).strip()


def extract_py_stream(path: Path, cap: int) -> list[str]:
    """extract.py's extract_instructions(), verbatim in behaviour."""
    pe = pefile.PE(str(path), fast_load=True)
    try:
        if pe.FILE_HEADER.Machine == 0x014C:
            md = Cs(CS_ARCH_X86, CS_MODE_32)
        elif pe.FILE_HEADER.Machine == 0x8664:
            md = Cs(CS_ARCH_X86, CS_MODE_64)
        else:
            return []
        out = []
        for section in pe.sections:
            if section.Characteristics & 0x20000000:
                for i in md.disasm(section.get_data(), section.VirtualAddress):
                    out.append(f"{i.mnemonic} {i.op_str}".strip())
                    if cap and len(out) >= cap:
                        return out
        return out
    finally:
        pe.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="../Goodware_Balanced")
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--cap", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    corpus = Path(args.corpus).resolve()
    rows = [r for r in csv.DictReader((corpus / "corpus_index.csv").open(encoding="utf-8"))
            if r["pipeline"] == "ok"]
    # Sample both architectures - a 32/64 mode mismatch is the failure mode
    # that a single-arch sample would miss.
    rng = random.Random(args.seed)
    x86 = [r for r in rows if r["arch"] == "x86"]
    x64 = [r for r in rows if r["arch"] == "x64"]
    half = max(1, args.n // 2)
    picked = rng.sample(x86, min(half, len(x86))) + \
             rng.sample(x64, min(args.n - half, len(x64)))

    print(f"comparing {len(picked)} files (cap {args.cap} instructions)\n")
    print(f"{'file':<42}{'insn':>7}{'mnemonic':>10}{'normalized':>12}")
    print("-" * 71)

    fails = 0
    for row in picked:
        src = corpus / row["rel_path"]
        a_status, a_lines, _, _ = disassemble(src, args.cap)
        a_stream = [convert_line(x) for x in a_lines]
        a_stream = [x for x in a_stream if x]
        b_stream = extract_py_stream(src, args.cap)

        mn_a = [x.split(" ", 1)[0] for x in a_stream]
        mn_b = [x.split(" ", 1)[0] for x in b_stream]
        nz_a = [normalize(x) for x in a_stream]
        nz_b = [normalize(x) for x in b_stream]

        mn_ok = mn_a == mn_b
        nz_ok = nz_a == nz_b
        fails += not (mn_ok and nz_ok)
        name = row["filename"][:40]
        print(f"{name:<42}{len(a_stream):>7}"
              f"{'match' if mn_ok else 'DIFF':>10}"
              f"{'match' if nz_ok else 'DIFF':>12}")

        if not nz_ok:
            diff = list(difflib.unified_diff(nz_b, nz_a, "extract.py",
                                             "asm_parse.py", n=1, lineterm=""))
            for line in diff[:14]:
                print(f"    {line}")

    print("-" * 71)
    if fails:
        print(f"{fails}/{len(picked)} files disagree - do NOT mix the corpora "
              f"until this is explained.")
        return 1
    print(f"all {len(picked)} files agree on both the mnemonic sequence and the\n"
          f"normalized instruction sequence. The corpora are interchangeable\n"
          f"once asm_to_opcodes.py has stripped the address prefix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
