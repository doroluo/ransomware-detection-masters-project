#!/usr/bin/env python3
"""
rewrite_streams.py - two richer token streams from the .asm transcripts,
written beside the mnemonic-only `mn/` trees and read the same way.

    python asm_tool/rewrite_streams.py --workers 8

Why. Dropping operands gained the tokenization pipeline 0.10 macro-F1 because
raw operands carry bitness (rbp, r8-r15, rip-relative) and per-binary
addresses. The import side input then gained the sequence transformer 0.04
because the mnemonic stream never says WHAT a program calls. These two
variants put back exactly the operand information that is not an
architecture tell:

    mn_api      the mnemonic stream, except that an instruction whose memory
                operand is an import-table slot carries the resolved name:
                    call dword ptr [0x41c008]      ->  call:kernel32.dll!createfilew
                    mov  esi, dword ptr [0x41c010] ->  mov:kernel32.dll!getprocaddress
                    call qword ptr [rip + 0xd639]  ->  call:advapi32.dll!cryptencrypt
                Every other instruction is its bare mnemonic, so a file with
                no import table gives exactly its `mn/` stream.

    mn_opclass  mnemonic plus an OPERAND CLASS per operand, no register names,
                no widths, no addresses:
                    mov edi, ecx                 ->  mov_reg_reg
                    mov dword ptr [edi + 4], 0   ->  mov_mem_imm
                    push 0x4203ac                ->  push_imm
                    je 0x4010a5                  ->  je_addr
                    call dword ptr [0x41c008]    ->  call_api kernel32.dll!createfilew
                classes: reg (any register, any width), mem (any memory
                operand), imm (a literal), addr (a literal target of a
                branch/call), api (an import-table slot; the name follows as
                its own token).

Both are written from one pass over `asm/<sha256>.asm`, one output line per
executable section (like `mn/`), for exactly the cohort files listed in the
folds_*.csv of the fold directory (results/family_holdout, or $RANSOM_FH_DIR);
each file is read from and written under the extraction tree of its `corpus`
column (family_holdout.common.TREE_OF_CORPUS). Prefixes (rep, lock,
bnd, ...) are emitted as their own token, which is what a whitespace split of
the `mn/` line already gives. IAT slots come from the per-corpus JSONs of
imports/extract_imports.py (their `iat` maps); a RIP-relative operand is
resolved against the address printed on the NEXT transcript line, exactly as
`extract_imports.py --verify-asm` does.

Output
    <corpus tree>/{mn_api,mn_opclass}/<sha256>.txt
    <shared>/rewrite_stats.csv   sha256, tree (= corpus), n_sections, n_insns,
                                 n_mem_indirect_branch, n_api_resolved, n_api_any
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from family_holdout.common import FOLD_DIR, SHARED, TREE_OF_CORPUS, ALL_DATASETS  # noqa: E402

TREES = TREE_OF_CORPUS
FOLDS = FOLD_DIR
IMPORT_JSONS = [REPO / "manifests" / "imports" / n for n in
                ("mendeley_mal_train.json", "mendeley_mal_test.json", "mendeley_good_train.json",
                 "mendeley_good_test.json", "goodware_balanced.json", "vs.json",
                 "goodware_hostx86.json")]
VARIANTS = ("mn_api", "mn_opclass")

LINE_RE = re.compile(r"^\s*0x([0-9a-fA-F]+)\s*:\s*(\S+)\s*(.*?)\s*$")
ABS_MEM_RE = re.compile(r"^\s*\[\s*(0x[0-9a-fA-F]+)\s*\]\s*$")
RIP_MEM_RE = re.compile(r"\[\s*rip\s*([+-])\s*(0x[0-9a-fA-F]+)\s*\]")
NUM_RE = re.compile(r"^-?(0x[0-9a-fA-F]+|\d+)$")
PREFIXES = {"rep", "repe", "repz", "repne", "repnz", "lock", "bnd", "notrack", "data16", "xacquire", "xrelease"}
BRANCH = {"call", "jmp", "loop", "loope", "loopne", "loopz", "loopnz", "jecxz", "jrcxz", "jcxz"}


def split_operands(ops: str) -> list:
    """Comma split that ignores commas inside brackets."""
    out, depth, cur = [], 0, []
    for ch in ops:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip()); cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return out


def mem_body(op: str):
    """The bracket content of a memory operand, or None."""
    i = op.find("[")
    return None if i < 0 else op[i:]


def is_branch(mnem: str) -> bool:
    return mnem in BRANCH or (mnem.startswith("j") and mnem not in ("jmp",)) or mnem == "jmp"


def classify(mnem: str, op: str, api: str | None) -> str:
    if api is not None:
        return "api"
    if "[" in op:
        return "mem"
    if NUM_RE.match(op):
        return "addr" if is_branch(mnem) else "imm"
    return "reg"


class Rewriter:
    """Streams one transcript; keeps one instruction pending so a RIP-relative
    operand can be resolved against the next line's address."""

    def __init__(self, iat: dict):
        self.iat = iat
        self.stats = {"n_sections": 0, "n_insns": 0, "n_mem_indirect_branch": 0,
                      "n_api_resolved": 0, "n_api_any": 0}

    def resolve(self, mnem: str, ops: list, next_addr):
        """-> [api name or None per operand]"""
        out = []
        for op in ops:
            body = mem_body(op)
            name = None
            if body is not None:
                m = ABS_MEM_RE.match(body)
                if m:
                    name = self.iat.get(m.group(1).lower())
                else:
                    r = RIP_MEM_RE.search(body)
                    if r and next_addr is not None:
                        disp = int(r.group(2), 16)
                        tgt = next_addr + disp if r.group(1) == "+" else next_addr - disp
                        name = self.iat.get(f"0x{tgt:x}")
                if mnem in ("call", "jmp"):
                    self.stats["n_mem_indirect_branch"] += 1
                    if name:
                        self.stats["n_api_resolved"] += 1
            if name:
                self.stats["n_api_any"] += 1
            out.append(name)
        return out

    def emit(self, mnem: str, ops: list, next_addr, api_tokens: list, cls_tokens: list):
        names = self.resolve(mnem, ops, next_addr)
        self.stats["n_insns"] += 1
        hit = next((n for n in names if n), None)
        api_tokens.append(f"{mnem}:{hit}" if hit else mnem)
        if ops:
            cls = [classify(mnem, op, n) for op, n in zip(ops, names)]
            cls_tokens.append(mnem + "_" + "_".join(cls))
            for n in names:
                if n:
                    cls_tokens.append(n)
        else:
            cls_tokens.append(mnem)

    def run(self, src: Path, out_api: Path, out_cls: Path) -> dict:
        pending = None                    # (mnem, ops)
        api_line, cls_line = [], []
        api_lines, cls_lines = [], []

        def flush_section():
            nonlocal pending, api_line, cls_line
            if pending is not None:
                self.emit(pending[0], pending[1], None, api_line, cls_line)
                pending = None
            if api_line or cls_line:
                api_lines.append(" ".join(api_line)); cls_lines.append(" ".join(cls_line))
            api_line, cls_line = [], []

        with src.open("r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                if raw.startswith(";"):
                    if raw.startswith("; section"):
                        flush_section(); self.stats["n_sections"] += 1
                    continue
                m = LINE_RE.match(raw)
                if not m:
                    continue
                addr = int(m.group(1), 16)
                if pending is not None:
                    self.emit(pending[0], pending[1], addr, api_line, cls_line)
                    pending = None
                mnem = m.group(2).lower()
                rest = m.group(3)
                if mnem == ".skip":
                    continue
                while mnem in PREFIXES and rest:
                    api_line.append(mnem); cls_line.append(mnem)
                    parts = rest.split(None, 1)
                    mnem = parts[0].lower(); rest = parts[1] if len(parts) > 1 else ""
                pending = (mnem, split_operands(rest.lower()) if rest else [])
        flush_section()
        out_api.parent.mkdir(parents=True, exist_ok=True)
        out_cls.parent.mkdir(parents=True, exist_ok=True)
        out_api.write_text("\n".join(api_lines) + ("\n" if api_lines else ""), encoding="utf-8")
        out_cls.write_text("\n".join(cls_lines) + ("\n" if cls_lines else ""), encoding="utf-8")
        return self.stats


def _task(args):
    sha, tree, src, iat = args
    root = TREES[tree]
    try:
        st = Rewriter(iat).run(Path(src), root / "mn_api" / f"{sha}.txt", root / "mn_opclass" / f"{sha}.txt")
        return {"sha256": sha, "tree": tree, **st, "error": ""}
    except Exception as e:                      # noqa: BLE001
        return {"sha256": sha, "tree": tree, "error": f"{type(e).__name__}: {e}"}


def cohort_files() -> list:
    """[(sha, corpus)] for every cohort file of every fold file present, each once."""
    seen, out = set(), []
    for ds in ALL_DATASETS:
        f = FOLDS / f"folds_{ds}.csv"
        if not f.is_file():
            continue
        with f.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if r["sha256"] not in seen:
                    seen.add(r["sha256"]); out.append((r["sha256"], r["corpus"]))
    if not out:
        raise SystemExit(f"no folds_*.csv under {FOLDS}; run family_holdout/folds.py first")
    return out


def load_iat(paths) -> dict:
    iat = {}
    for p in paths:
        p = Path(p)
        if not p.is_file():
            print(f"warning: {p} missing", file=sys.stderr); continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        side = p.with_suffix(".iat.json")
        spill = json.loads(side.read_text(encoding="utf-8")) if side.is_file() else {}
        for sha, rec in doc.items():
            m = rec.get("iat") or spill.get(sha) or {}
            iat[sha] = {k.lower(): v for k, v in m.items()}
    return iat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=0, help="first N files only (smoke test)")
    ap.add_argument("--force", action="store_true", help="rewrite files that already exist")
    a = ap.parse_args()
    t0 = time.time()
    iat = load_iat(IMPORT_JSONS)
    files = cohort_files()
    if a.limit:
        files = files[:a.limit]
    tasks = []
    for sha, tree in files:
        root = TREES[tree]
        src = root / "asm" / f"{sha}.asm"
        if not src.is_file():
            print(f"missing transcript: {src}", file=sys.stderr); continue
        if not a.force and all((root / v / f"{sha}.txt").is_file() for v in VARIANTS):
            continue
        tasks.append((sha, tree, str(src), iat.get(sha, {})))
    print(f"{len(files)} cohort files, {len(tasks)} to rewrite, {len(iat)} IAT maps, {a.workers} workers", flush=True)
    stats_path = SHARED / "rewrite_stats.csv"
    fields = ["sha256", "tree", "n_sections", "n_insns", "n_mem_indirect_branch", "n_api_resolved", "n_api_any", "error"]
    existing = {}
    if stats_path.is_file():
        with stats_path.open(encoding="utf-8", newline="") as fh:
            existing = {r["sha256"]: r for r in csv.DictReader(fh)}
    done = 0
    with Pool(a.workers) as pool:
        for rec in pool.imap_unordered(_task, tasks, chunksize=4):
            existing[rec["sha256"]] = {k: rec.get(k, "") for k in fields}
            done += 1
            if done % 200 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    with stats_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for sha in sorted(existing):
            w.writerow(existing[sha])
    err = sum(1 for r in existing.values() if r.get("error"))
    ins = sum(int(r["n_insns"] or 0) for r in existing.values() if not r.get("error"))
    br = sum(int(r["n_mem_indirect_branch"] or 0) for r in existing.values() if not r.get("error"))
    res = sum(int(r["n_api_resolved"] or 0) for r in existing.values() if not r.get("error"))
    print(f"done: {len(existing)} files, {err} errors, {ins:,} instructions, "
          f"{res:,}/{br:,} memory-indirect call/jmp resolved ({100*res/max(br,1):.1f}%), "
          f"{time.time()-t0:.0f}s; stats -> {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
