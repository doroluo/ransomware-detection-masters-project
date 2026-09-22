#!/usr/bin/env python3
"""
stage_host_goodware.py - a native x86 goodware corpus from this Windows host.

Why: the goodware pool is 59% x64 and the ransomware pool is 18-25% x64, so
"x64 means goodware" is a free 0.70 accuracy. The host's own 32-bit Windows
binaries (SysWOW64, Program Files (x86)) are signed, benign, native x86 code
from the same vendors and toolchains as the x64 goodware already in the set.

What it does
  * walks the source roots, keeps PE files with machine 0x14c (x86),
  * takes every EXE and samples DLLs across programs (round-robin over groups,
    deterministic) up to --max-dll, so one program cannot dominate,
  * de-duplicates by sha256,
  * copies the selection to --out/<group>/<name> (extract_unified.py reads a
    tree in place) and writes --out/hostgood_index.csv with
    sha256, group, is_dll, arch, source_path, size.

group = "sys:<file name>" for SysWOW64 (its x64 twin lives in System32, and
Goodware_Balanced's system bucket was drawn from there, so twins share a
name) and "pf:<top-level application folder>" for Program Files (x86); the
fold builder keeps a group on one side of every split.

    python tools/stage_host_goodware.py --out C:/Users/chaoa/Downloads/Goodware_HostX86 --max-dll 1000
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import os
import random
import shutil
import struct
from pathlib import Path

ROOTS = {
    "sys": (Path(r"C:\Windows\SysWOW64"), 1),
    "pf": (Path(r"C:\Program Files (x86)"), 8),
}


def pe_info(p: Path):
    """(machine, is_dll) or None if not a parseable PE."""
    try:
        with p.open("rb") as f:
            if f.read(2) != b"MZ":
                return None
            f.seek(0x3C)
            off = struct.unpack("<I", f.read(4))[0]
            f.seek(off)
            if f.read(4) != b"PE\0\0":
                return None
            machine = struct.unpack("<H", f.read(2))[0]
            f.seek(off + 4 + 18)
            chars = struct.unpack("<H", f.read(2))[0]
            return machine, bool(chars & 0x2000)
    except (OSError, struct.error):
        return None


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def group_of(source: str, root: Path, p: Path) -> str:
    if source == "sys":
        return "sys:" + p.name.lower()
    rel = p.relative_to(root)
    return "pf:" + (rel.parts[0].lower() if len(rel.parts) > 1 else p.name.lower())


def collect(max_depth_override=None):
    found = []
    for source, (root, depth) in ROOTS.items():
        if not root.is_dir():
            continue
        base = len(root.parts)
        for dp, dns, fns in os.walk(root):
            dp = Path(dp)
            if len(dp.parts) - base >= (max_depth_override or depth):
                dns[:] = []
            for fn in fns:
                if not fn.lower().endswith((".exe", ".dll")):
                    continue
                p = dp / fn
                info = pe_info(p)
                if not info or info[0] != 0x14C:
                    continue
                found.append((source, root, p, info[1]))
    return found


def select(found, max_dll: int, seed: int):
    rng = random.Random(seed)
    exes = [f for f in found if not f[3]]
    dlls = collections.defaultdict(list)
    for f in found:
        if f[3]:
            dlls[group_of(f[0], f[1], f[2])].append(f)
    for g in dlls:
        rng.shuffle(dlls[g])
    groups = sorted(dlls)
    rng.shuffle(groups)
    picked, i = [], 0
    while len(picked) < max_dll and any(dlls.values()):
        g = groups[i % len(groups)]
        if dlls[g]:
            picked.append(dlls[g].pop())
        i += 1
    return exes + picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-dll", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    found = collect()
    print(f"{len(found)} native x86 PE files found "
          f"({sum(1 for f in found if not f[3])} exe, {sum(1 for f in found if f[3])} dll)")
    chosen = select(found, a.max_dll, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    for source, root, p, is_dll in sorted(chosen, key=lambda f: str(f[2]).lower()):
        sha = sha256_file(p)
        if sha in seen:
            continue
        seen.add(sha)
        g = group_of(source, root, p)
        dest_dir = a.out / g.replace(":", "_").replace(" ", "_")
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / p.name
        if not dest.exists():
            shutil.copy2(p, dest)
        rows.append({"sha256": sha, "group": g, "is_dll": int(is_dll), "arch": "x86",
                     "source_path": str(p), "size": p.stat().st_size,
                     "rel_path": str(dest.relative_to(a.out)).replace("\\", "/")})
    with (a.out / "hostgood_index.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_dll = sum(r["is_dll"] for r in rows)
    print(f"staged {len(rows)} unique files ({len(rows) - n_dll} exe, {n_dll} dll) in "
          f"{len({r['group'] for r in rows})} groups, {sum(r['size'] for r in rows) / 1e9:.2f} GB -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
