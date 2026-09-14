#!/usr/bin/env python3
"""
check_schema.py - prove a candidate opcode-feature directory is schema-identical
to the reference one, so the two can be mixed in a single experiment.

    python llm_features_pipeline/check_schema.py
    python llm_features_pipeline/check_schema.py --candidate DIR --reference DIR

Why this exists. Exp B feeds Goodware_Balanced features (produced by
`asm_tool/asm_to_opcodes.py` from `asm_parse.py` output) to a model trained
alongside Mendeley ransomware features (produced by `extract.py`). If the two
writers disagree about anything visible in the text - an address prefix, a tab
between mnemonic and operands, a comment line, case - that difference is present
in exactly one class and is a content-free feature that separates the classes
perfectly. `normalize_instruction` hides some of it (`0x...` becomes `<HEX>`),
which makes the failure quieter, not smaller.

Hard checks (exit 1 on any failure):
  * filename pattern `<prefix>_<name>.txt`
  * UTF-8 decodable
  * one instruction per line
  * no line starting with an `0x...:` address prefix
  * no tab characters
  * no `;` comment lines
  * no `.skip` lines (or any other assembler directive line)
  * no empty files

Soft check (reported, never fatal): the symmetric difference between the two
mnemonic vocabularies. Different corpora legitimately use different
instructions; a *huge* difference is a hint that one side is being decoded in
the wrong mode, but it is not a schema violation.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

DEFAULT_CANDIDATE = "C:/Users/chaoa/Downloads/LLM_Features_Balanced/good_all"
DEFAULT_REFERENCE = "C:/Users/chaoa/Downloads/LLM_Features/Features_Extraction/good_train"

NAME_RE = re.compile(r"^[^_]+_.+\.txt$")
ADDR_RE = re.compile(r"^\s*0x[0-9a-fA-F]+\s*:")
DIRECTIVE_RE = re.compile(r"^\s*\.(skip|byte|word|long|quad|align|section|text|data)\b")
# extract.py writes "mov eax, ebx"; a mnemonic is the first whitespace-run token.
MNEMONIC_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_.]*$")


def scan(directory: Path, max_report: int = 5) -> dict:
    files = sorted(directory.glob("*.txt"))
    rep: dict = {
        "dir": str(directory), "n_files": len(files),
        "bad_name": [], "not_utf8": [], "address_prefix": [], "tab": [],
        "comment": [], "directive": [], "empty": [], "bad_head": [],
        "mnemonics": collections.Counter(), "n_lines": 0,
    }
    if not files:
        rep["bad_name"].append("<no .txt files found>")
        return rep

    for p in files:
        if not NAME_RE.match(p.name):
            rep["bad_name"].append(p.name)
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            rep["not_utf8"].append(f"{p.name}: {exc}")
            continue

        n_nonblank = 0
        for lineno, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            n_nonblank += 1
            rep["n_lines"] += 1
            if ADDR_RE.match(line):
                _add(rep["address_prefix"], f"{p.name}:{lineno}: {line[:60]!r}", max_report)
            if "\t" in line:
                _add(rep["tab"], f"{p.name}:{lineno}: {line[:60]!r}", max_report)
            if line.lstrip().startswith(";"):
                _add(rep["comment"], f"{p.name}:{lineno}: {line[:60]!r}", max_report)
            if DIRECTIVE_RE.match(line):
                _add(rep["directive"], f"{p.name}:{lineno}: {line[:60]!r}", max_report)
            # "one instruction per line", made checkable: the first
            # whitespace-separated token of every line must be a bare mnemonic.
            # Anything else - an address, a byte dump, a label, a directive, a
            # second instruction glued on - fails this.
            head = line.strip().split()[0].lower()
            if MNEMONIC_RE.match(head):
                rep["mnemonics"][head] += 1
            else:
                _add(rep["bad_head"], f"{p.name}:{lineno}: {line[:60]!r}", max_report)
        if n_nonblank == 0:
            rep["empty"].append(p.name)
    return rep


def _add(bucket: list, item: str, cap: int) -> None:
    """Keep the first `cap` examples but keep counting past them."""
    if len(bucket) < cap:
        bucket.append(item)
    else:
        bucket.append(None)


def _shown(bucket: list) -> tuple[int, list]:
    real = [b for b in bucket if b is not None]
    return len(bucket), real


HARD = [("bad_name", "filename does not match <prefix>_<name>.txt"),
        ("not_utf8", "file is not UTF-8 decodable"),
        ("address_prefix", "line carries an 0x...: address prefix"),
        ("tab", "line contains a tab character"),
        ("comment", "line is a ';' comment"),
        ("directive", "line is an assembler directive (.skip/.byte/...)"),
        ("bad_head", "line does not start with a bare mnemonic "
                      "(not one instruction per line)"),
        ("empty", "file has no instruction lines")]


def report(rep: dict, label: str) -> int:
    print(f"\n{label}: {rep['dir']}")
    print(f"  {rep['n_files']} files, {rep['n_lines']} instruction lines, "
          f"{len(rep['mnemonics'])} distinct mnemonics")
    failures = 0
    for key, desc in HARD:
        n, examples = _shown(rep[key])
        if n:
            failures += n
            print(f"  FAIL  {n:>6}  {desc}")
            for e in examples:
                print(f"           {e}")
        else:
            print(f"  ok    {0:>6}  {desc}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    ap.add_argument("--reference", default=DEFAULT_REFERENCE)
    ap.add_argument("--top", type=int, default=15,
                    help="how many vocabulary differences to print per side")
    args = ap.parse_args()

    cand, ref = Path(args.candidate), Path(args.reference)
    for d in (cand, ref):
        if not d.is_dir():
            sys.exit(f"not a directory: {d}")

    rc = scan(cand)
    rr = scan(ref)
    fails = report(rr, "REFERENCE") + report(rc, "CANDIDATE")

    cv, rv = set(rc["mnemonics"]), set(rr["mnemonics"])
    print("\nMnemonic vocabulary (soft check, never fatal)")
    print(f"  reference {len(rv)}, candidate {len(cv)}, shared {len(cv & rv)}")
    jac = len(cv & rv) / len(cv | rv) if (cv | rv) else 1.0
    print(f"  Jaccard {jac:.3f}")
    only_c = sorted(cv - rv, key=lambda m: -rc["mnemonics"][m])[:args.top]
    only_r = sorted(rv - cv, key=lambda m: -rr["mnemonics"][m])[:args.top]
    print(f"  candidate-only ({len(cv - rv)}), most frequent first: "
          + ", ".join(f"{m}x{rc['mnemonics'][m]}" for m in only_c))
    print(f"  reference-only ({len(rv - cv)}), most frequent first: "
          + ", ".join(f"{m}x{rr['mnemonics'][m]}" for m in only_r))

    if fails:
        print(f"\nFAILED: {fails} hard schema violation(s)")
        return 1
    print("\nPASS: candidate is schema-identical to reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
