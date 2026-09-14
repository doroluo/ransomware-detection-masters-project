#!/usr/bin/env python3
"""Zip the files the VM needs (see vm_package/README.md). Run from the repo root."""
from __future__ import annotations
import argparse, zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INCLUDE = ["asm_parse.py", "check_arch.py", "asm_tool", "ember_pipeline", "vm_package/README.md"]
SKIP_DIRS = {"__pycache__", ".pytest_cache"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO.parent / "vm_package.zip"))
    a = ap.parse_args()
    out = Path(a.out)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            p = REPO / item
            if p.is_file():
                z.write(p, item); n += 1
                continue
            for f in sorted(p.rglob("*")):
                if f.is_file() and not (SKIP_DIRS & set(f.relative_to(REPO).parts)) and f.suffix != ".pyc":
                    z.write(f, f.relative_to(REPO).as_posix()); n += 1
    print(f"wrote {out} ({n} files, {out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
