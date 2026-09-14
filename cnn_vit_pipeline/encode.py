#!/usr/bin/env python3
"""Configurable token-image encoder for the CNN-ViT pipeline.

`CNN-ViT/asm_parser.py` is left exactly as the teammate's branch has it. This
module reuses its *parsing* (``parse_asm_line``, ``TOKEN_MAP``, ``API_MAP``,
``asm_id``) and re-implements only the last step - how a parsed instruction
stream becomes a 256x256 uint8 canvas plus a 16x16 ViT patch mask - so that
alternative encodings can be measured against the original.

Two stages
----------
1. ``cache``  parse every .asm under an asm tree exactly once with
   ``asm_parser.parse_asm_line`` and store the result as a flat uint8 array of
   instruction triplets ``(n_instructions, 3)``. Writes
   ``<cache>/<tree>/{tokens.u8,index.csv}``. This is the expensive step
   (~4 minutes over the 3,941 files on 16 cores) and it is variant-agnostic.
2. ``render`` / ``arrays``  turn the cache into canvases under one of the
   encodings below. ``render`` writes a PNG + ``_vit_mask.npy`` tree that
   ``build_dataset.py --images-root ... --source unified`` consumes unchanged;
   ``arrays`` returns them in memory for the trainer.

The encodings
-------------
Every canvas is 256x256 = 65,536 uint8 token slots. The ViT mask marks a 16x16
patch active when it holds at least one real token. Unused canvas is filled
with ``END_PAD`` (3), the constant the current pipeline uses.

``head3``    *the baseline; byte-for-byte what asm_parser.py produces today.*
             Instruction triplets (opcode-or-API id, operand-1 class,
             operand-2 class) laid down in file order, truncated at 65,536
             tokens - the first 21,845 whole instructions plus the opcode of
             the 21,846th.

``stride3``  Same triplets, but when a file holds more than that window the
             21,846 kept are sampled on an even stride over the *whole* file
             (``np.linspace(0, n-1, 21846)``) instead of taking the first
             ones. Files at or below the window are identical to ``head3``.
             Answers "does the model lose by only ever seeing the program's
             first fifth?" - 50% of Mendeley and 61% of balanced-goodware
             files exceed the window.

``mnem1``    Mnemonic-only: one token per instruction (the opcode/API id;
             operand classes dropped), file order, first 65,536
             instructions. Three times the code fits on the same canvas and
             the operand channel - which is only ever one of five coarse
             classes - stops diluting it. This is the representation the
             mnemonic TF-IDF baseline wins with.

``mnem1s``   Mnemonic-only with the same even-stride subsampling as
             ``stride3`` for files longer than 65,536 instructions.

``crop3``    Instruction triplets, K fixed windows of 21,846 consecutive
             instructions per file, their starts evenly spaced over the file
             (window 0 always starts at 0, so crop 0 == ``head3``). The
             trainer picks one crop at random per sample per epoch and
             averages the K scores at test time. Files shorter than the
             window yield K identical crops.

Row alignment
-------------
A canvas row is 256 tokens. With triplets that is 85 1/3 instructions, so an
instruction straddles the row boundary and the phase shifts every row; the
CNN stem therefore sees the same instruction at three different alignments.
``--row-align`` pads each row out to a whole number of instructions
(255 tokens = 85 instructions + one END_PAD) so every row starts on an opcode.
It applies to the triplet encodings only (mnemonic encodings are already
aligned) and is off by default.

    python cnn_vit_pipeline/encode.py cache --asm-tree unified_mendeley
    python cnn_vit_pipeline/encode.py render --asm-tree unified_mendeley \
        --variant mnem1 --out-root .../cnn_vit_images/tuned_mnem1
    python cnn_vit_pipeline/encode.py selftest
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT / "CNN-ViT", REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import asm_parser  # noqa: E402  (imported unmodified; import is side-effect free)

SQUARE = asm_parser.SQUARE_RESOLUTION          # 256
CAPACITY = asm_parser.TOTAL_TOKEN_CAPACITY     # 65536
PATCH = asm_parser.VIT_PATCH_SIZE              # 16
END_PAD = asm_parser.TOKEN_MAP["END_PAD"]      # 3
PATCHES = SQUARE // PATCH                      # 16

#: instructions touched by one canvas, per encoding width. 65536/3 is not a
#: whole number, so the 21846th instruction of a triplet encoding is drawn
#: partially - one token of it lands on the last canvas slot. asm_parser.py
#: does exactly this (it truncates the flattened token stream, not the
#: instruction list), and `head3` reproduces it byte for byte.
INSN_WINDOW = {1: CAPACITY, 3: -(-CAPACITY // 3)}  # 65536 and 21846

ASM_ROOT = Path(os.environ.get("RANSOM_ASM_OUTPUT",
                               REPO_ROOT.parent / "asm_output"))
CACHE_ROOT = Path(os.environ.get("RANSOM_CNN_VIT_CACHE",
                                 REPO_ROOT.parent / "cnn_vit_cache"))

VARIANTS = ("head3", "stride3", "mnem1", "mnem1s", "crop3")
CROP_K = 4

CLASS_DIRS = {0: "Class_0_Goodware", 1: "Class_1_Ransomware"}


# --------------------------------------------------------------- stage 1 ---
def parse_asm_file(path: str) -> np.ndarray:
    """Every instruction of one .asm as uint8 triplets, via asm_parser."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            triplet = asm_parser.parse_asm_line(line)
            if triplet:
                rows.append(triplet)
    if not rows:
        return np.zeros((0, 3), dtype=np.uint8)
    return np.asarray(rows, dtype=np.uint8)


def _parse_job(args):
    key, path = args
    arr = parse_asm_file(path)
    return key, arr


def build_cache(asm_tree: str, cache_root: Path = None, workers: int = 0) -> Path:
    """Parse one asm tree into <cache>/<tree>/{tokens.u8,index.csv}."""
    import pandas as pd
    cache_root = Path(cache_root or CACHE_ROOT)
    asm_dir = ASM_ROOT / asm_tree
    man = pd.read_csv(asm_dir / "asm_manifest.csv", dtype=str,
                      keep_default_na=False)
    man = man[man["asm_path"] != ""]

    jobs = []
    meta = []
    for _, r in man.iterrows():
        rel = r["asm_path"]
        # asm_manifest paths are relative to the tree root and may carry the
        # tree name as their first component (goodware_balanced/...).
        cand = asm_dir / rel
        if not cand.exists():
            cand = asm_dir / Path(rel).relative_to(Path(rel).parts[0])
        stem = asm_parser.asm_id(cand, asm_dir)
        jobs.append((stem, str(cand)))
        meta.append({"asm_id": stem, "sha256": r["sha256"].strip().lower(),
                     "label": int(r["label"]), "set": r["set"],
                     "family": r["family"], "arch": r["arch"],
                     "filename": r["filename"]})

    out_dir = cache_root / asm_tree
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or os.cpu_count() or 4

    print(f"[cache] {asm_tree}: parsing {len(jobs)} .asm on {workers} workers")
    results: dict[str, np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (key, arr) in enumerate(ex.map(_parse_job, jobs, chunksize=8), 1):
            results[key] = arr
            if i % 250 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)

    offsets, total = [], 0
    for m in meta:
        n = len(results[m["asm_id"]])
        offsets.append((total, n))
        total += n
    flat = np.empty((total, 3), dtype=np.uint8)
    for m, (off, n) in zip(meta, offsets):
        if n:
            flat[off:off + n] = results[m["asm_id"]]
    flat.tofile(out_dir / "tokens.u8")

    for m, (off, n) in zip(meta, offsets):
        m["offset"], m["n_insns"] = off, n
    pd.DataFrame(meta).to_csv(out_dir / "index.csv", index=False)
    print(f"[cache] wrote {out_dir/'tokens.u8'} "
          f"({total} instructions, {flat.nbytes/1e6:.0f} MB) and index.csv")
    return out_dir


def load_cache(asm_tree: str, cache_root: Path = None):
    """(index DataFrame, memmapped (total,3) uint8 token array)."""
    import pandas as pd
    cache_root = Path(cache_root or CACHE_ROOT)
    d = cache_root / asm_tree
    if not (d / "tokens.u8").exists():
        raise FileNotFoundError(
            f"no token cache at {d}; run "
            f"`python cnn_vit_pipeline/encode.py cache --asm-tree {asm_tree}`")
    idx = pd.read_csv(d / "index.csv", dtype={"sha256": str})
    tok = np.memmap(d / "tokens.u8", dtype=np.uint8, mode="r").reshape(-1, 3)
    return idx, tok


# --------------------------------------------------------------- stage 2 ---
def _select(n: int, window: int, stride: bool, start: int = 0) -> np.ndarray:
    """Which instruction indices land on the canvas."""
    if n <= window:
        return np.arange(n)
    if stride:
        return np.unique(np.linspace(0, n - 1, window).round().astype(np.int64))
    start = min(start, n - window)
    return np.arange(start, start + window)


def canvas_from_insns(insns: np.ndarray, variant: str, crop: int = 0,
                      row_align: bool = False):
    """One (256,256) uint8 canvas and its (16,16) uint8 ViT mask.

    `insns` is the file's full (n,3) uint8 triplet array.
    """
    n = len(insns)
    if variant in ("mnem1", "mnem1s"):
        width, stride = 1, (variant == "mnem1s")
    elif variant in ("head3", "stride3", "crop3"):
        width, stride = 3, (variant == "stride3")
    else:
        raise ValueError(f"unknown variant {variant!r}")

    window = INSN_WINDOW[width]
    if row_align and width == 3:
        # 85 instructions (255 tokens) per row + 1 END_PAD => 85*256 = 21760
        window = (SQUARE // width) * SQUARE

    start = 0
    if variant == "crop3" and n > window:
        # K windows with evenly spaced starts; crop 0 == head3
        start = int(round(crop * (n - window) / max(CROP_K - 1, 1)))

    pick = _select(n, window, stride, start)
    if width == 1:
        stream = insns[pick, 0]
    else:
        # flatten, then truncate the TOKEN stream at the canvas capacity, which
        # is what asm_parser.py does: the last instruction may be drawn partly.
        stream = insns[pick].reshape(-1)[:CAPACITY]

    flat = np.full(CAPACITY, END_PAD, dtype=np.uint8)
    mask_flat = np.zeros(CAPACITY, dtype=np.uint8)

    if row_align and width == 3:
        per_row = (SQUARE // width) * width          # 255 tokens = 85 insns
        canvas = flat.reshape(SQUARE, SQUARE)
        mcanvas = mask_flat.reshape(SQUARE, SQUARE)
        usable = min(len(stream), SQUARE * per_row)
        r = usable // per_row                        # whole rows available
        if r:
            canvas[:r, :per_row] = stream[:r * per_row].reshape(r, per_row)
            mcanvas[:r, :per_row] = 1
        rest = stream[r * per_row:usable]            # the partial final row
        if len(rest) and r < SQUARE:
            canvas[r, :len(rest)] = rest
            mcanvas[r, :len(rest)] = 1
    else:
        k = min(len(stream), CAPACITY)
        flat[:k] = stream[:k]
        mask_flat[:k] = 1

    img = flat.reshape(SQUARE, SQUARE)
    mask2d = mask_flat.reshape(SQUARE, SQUARE)
    patch_mask = (mask2d.reshape(PATCHES, PATCH, PATCHES, PATCH)
                  .max(axis=(1, 3))).astype(np.uint8)
    return img, patch_mask


def encode_tree(asm_tree: str, variant: str, row_align: bool = False,
                cache_root: Path = None, crops: int = 1):
    """(index DataFrame, images, masks) for a whole asm tree.

    images is (N,256,256) uint8, or (N,K,256,256) when crops > 1.
    masks   is (N,16,16)   uint8, or (N,K,16,16)   when crops > 1.
    """
    idx, tok = load_cache(asm_tree, cache_root)
    n = len(idx)
    if crops > 1:
        imgs = np.empty((n, crops, SQUARE, SQUARE), dtype=np.uint8)
        msks = np.empty((n, crops, PATCHES, PATCHES), dtype=np.uint8)
    else:
        imgs = np.empty((n, SQUARE, SQUARE), dtype=np.uint8)
        msks = np.empty((n, PATCHES, PATCHES), dtype=np.uint8)
    off = idx["offset"].to_numpy()
    cnt = idx["n_insns"].to_numpy()
    for i in range(n):
        insns = np.asarray(tok[off[i]:off[i] + cnt[i]])
        if crops > 1:
            for k in range(crops):
                imgs[i, k], msks[i, k] = canvas_from_insns(
                    insns, variant, crop=k, row_align=row_align)
        else:
            imgs[i], msks[i] = canvas_from_insns(insns, variant,
                                                 row_align=row_align)
    return idx, imgs, msks


def render_tree(asm_tree: str, variant: str, out_root: Path,
                row_align: bool = False, cache_root: Path = None) -> dict:
    """Write a PNG + mask tree build_dataset.py can consume unchanged.

    Layout mirrors the unified image trees: <out_root>/<asm_tree>/Class_<n>_*/
    with the PNG stem equal to asm_parser.asm_id() of the .asm, so
    build_dataset.py's `_unified_stem_map` joins it back to a sha256.
    """
    from PIL import Image
    idx, imgs, msks = encode_tree(asm_tree, variant, row_align, cache_root)
    out = Path(out_root) / asm_tree
    counts = {}
    for i, r in idx.iterrows():
        cls = CLASS_DIRS[int(r["label"])]
        d = out / cls
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(imgs[i], mode="L").save(d / f"{r['asm_id']}.png")
        np.save(d / f"{r['asm_id']}_vit_mask.npy", msks[i])
        counts[cls] = counts.get(cls, 0) + 1
    print(f"[render] {asm_tree} variant={variant} row_align={row_align} -> "
          f"{out}: {counts}")
    return counts


# ---------------------------------------------------------------- selftest --
_TINY_ASM = """0x00401000:  push	0x40
0x00401002:  mov	ecx, 0x4f89f8
0x00401007:  call	0x40a390
0x0040100c:  ret
0x0040100d:  nop
0x0040100e:  xor	eax, eax
0x00401010:  call	dword ptr [VirtualAlloc]
; a comment-only line
0x00401016:  mov	dword ptr [ebp-0x4], eax
"""


def selftest() -> int:
    """Hand-written .asm -> expected token-image properties."""
    import tempfile
    T = asm_parser.TOKEN_MAP
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{' - ' + detail if detail and not cond else ''}")
        ok = ok and bool(cond)

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "tiny.asm"
        p.write_text(_TINY_ASM, encoding="utf-8")
        insns = parse_asm_file(str(p))

    print("selftest: hand-written 8-instruction .asm")
    check("8 instructions parsed", len(insns) == 8, f"got {len(insns)}")
    # NOTE `call 0x40a390` is a direct call to an address, which matches no
    # name in API_MAP, so asm_parser maps it to UNKNOWN_API (75) rather than
    # CALL (21). CALL's own id is unreachable in practice.
    expect_ops = [T["PUSH"], T["MOV"], T["UNKNOWN_API"], T["RET"],
                  T["NOP_SLED"], T["XOR"],
                  asm_parser.API_MAP["virtualalloc"], T["MOV"]]
    check("opcode column matches the hand-computed ids",
          [int(v) for v in insns[:, 0]] == expect_ops,
          f"{[int(v) for v in insns[:,0]]} != {expect_ops}")
    check("`call VirtualAlloc` resolves to the API id, not CALL",
          insns[6, 0] == asm_parser.API_MAP["virtualalloc"])
    check("`ret` has two PADDING operand slots",
          insns[3, 1] == T["PADDING"] and insns[3, 2] == T["PADDING"])
    check("`mov [ebp-0x4], eax` -> MEM_STACK_REF, REG_DATA",
          insns[7, 1] == T["MEM_STACK_REF"] and insns[7, 2] == T["REG_DATA"])

    # --- canvas properties, per variant -------------------------------------
    img, mask = canvas_from_insns(insns, "head3")
    check("head3 canvas is 256x256 uint8", img.shape == (256, 256)
          and img.dtype == np.uint8)
    check("head3 writes 24 real tokens then END_PAD",
          (img.reshape(-1)[:24] == insns.reshape(-1)).all()
          and (img.reshape(-1)[24:] == END_PAD).all())
    # 24 tokens occupy row 0 columns 0..23, which straddles patches (0,0)
    # and (0,1); nothing else on the canvas is real.
    check("head3 mask activates exactly the two patches the 24 tokens touch",
          mask.sum() == 2 and mask[0, 0] == 1 and mask[0, 1] == 1)

    img1, mask1 = canvas_from_insns(insns, "mnem1")
    check("mnem1 writes 8 tokens (opcodes only)",
          (img1.reshape(-1)[:8] == insns[:, 0]).all()
          and (img1.reshape(-1)[8:] == END_PAD).all())

    # --- synthetic long file: window and stride ----------------------------
    n = 60000
    rng = np.random.default_rng(0)
    long = np.stack([rng.integers(10, 60, n), rng.integers(70, 75, n),
                     rng.integers(70, 75, n)], axis=1).astype(np.uint8)

    h, hm = canvas_from_insns(long, "head3")
    check("head3 on a 60k-instruction file keeps the FIRST 65536 tokens",
          (h.reshape(-1) == long.reshape(-1)[:65536]).all())
    check("head3 fills the canvas - no END_PAD on a capped file",
          not (h == END_PAD).any() or (long.reshape(-1)[:65536] == END_PAD).any())
    check("head3 mask is fully active on a capped file", hm.sum() == 256)

    s, _ = canvas_from_insns(long, "stride3")
    pick = np.unique(np.linspace(0, n - 1, 21846).round().astype(np.int64))
    check("stride3 samples evenly across the whole file",
          (s.reshape(-1) == long[pick].reshape(-1)[:65536]).all())
    check("stride3 reaches the last instruction, head3 does not",
          (long[pick][-1] == long[-1]).all() and pick[-1] == n - 1)

    m, _ = canvas_from_insns(long, "mnem1")
    check("mnem1 fits all 60000 instructions - head3 reached only 21845",
          (m.reshape(-1)[:n] == long[:, 0]).all()
          and (m.reshape(-1)[n:] == END_PAD).all())
    vlong = np.repeat(long, 2, axis=0)  # 120000 instructions
    mv, _ = canvas_from_insns(vlong, "mnem1")
    check("mnem1 window is 65536 instructions",
          (mv.reshape(-1) == vlong[:65536, 0]).all())

    c0, _ = canvas_from_insns(long, "crop3", crop=0)
    c3, _ = canvas_from_insns(long, "crop3", crop=CROP_K - 1)
    check("crop 0 == head3", (c0 == h).all())
    # the window ends on the file's final instruction, whose opcode lands on
    # the very last canvas slot (its two operand tokens fall off the edge,
    # exactly as asm_parser.py's token-stream truncation does)
    check("last crop reaches the end of the file",
          c3.reshape(-1)[-1] == long[-1][0]
          and (c3.reshape(-1)[-4:-1] == long[-2]).all())

    ra, ram = canvas_from_insns(long, "head3", row_align=True)
    rows = ra.reshape(256, 256)
    check("row_align: every row starts on an opcode",
          all(rows[r, 0] in set(range(10, 60)) for r in range(256)))
    check("row_align: column 255 is END_PAD on every row",
          (rows[:, 255] == END_PAD).all())
    check("row_align mask fully active", ram.sum() == 256)

    # --- short file: mask geometry -----------------------------------------
    short = long[:1000]
    _, sm = canvas_from_insns(short, "head3")
    # 3000 tokens -> rows 0..11 full (3072 > 3000), patch rows 0 active only
    check("short file activates ceil(3000/4096) patch rows",
          sm[0].sum() == 16 and sm[1:].sum() == 0)

    print("\nselftest:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


# --------------------------------------------------------------------- cli --
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cache", help="parse .asm -> token cache (once)")
    c.add_argument("--asm-tree", required=True)
    c.add_argument("--asm-root", default=None)
    c.add_argument("--cache-root", default=None)
    c.add_argument("--workers", type=int, default=0)

    r = sub.add_parser("render", help="token cache -> PNG/mask tree")
    r.add_argument("--asm-tree", required=True)
    r.add_argument("--variant", choices=VARIANTS, required=True)
    r.add_argument("--row-align", action="store_true")
    r.add_argument("--out-root", required=True)
    r.add_argument("--cache-root", default=None)

    sub.add_parser("selftest", help="hand-written .asm -> expected properties")
    a = ap.parse_args()

    global ASM_ROOT
    if getattr(a, "asm_root", None):
        ASM_ROOT = Path(a.asm_root)

    if a.cmd == "cache":
        build_cache(a.asm_tree, a.cache_root, a.workers)
        return 0
    if a.cmd == "render":
        render_tree(a.asm_tree, a.variant, Path(a.out_root), a.row_align,
                    a.cache_root)
        return 0
    return selftest()


if __name__ == "__main__":
    raise SystemExit(main())
