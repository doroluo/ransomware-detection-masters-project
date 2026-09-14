"""Tests for the configurable token-image encoder, cnn_vit_pipeline/encode.py.

The encoder is the one place where a change to the *input representation* can
silently invalidate every CNN-ViT number in results/. Two things are asserted:

  1. a tiny hand-written .asm produces exactly the token image the TOKEN_MAP
     says it should - opcode ids, operand classes, canvas fill, ViT mask
     geometry - for every encoding variant; and
  2. the default variant `head3` is byte-for-byte what the unmodified
     CNN-ViT/asm_parser.py writes, so "tuned" numbers are only ever compared
     against a baseline that really is the baseline.

Nothing here needs the corpus; everything is synthesised in-process.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "CNN-ViT"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cnn_vit_pipeline import encode as E  # noqa: E402
import asm_parser  # noqa: E402

T = asm_parser.TOKEN_MAP
END_PAD = T["END_PAD"]

TINY_ASM = """0x00401000:  push	0x40
0x00401002:  mov	ecx, 0x4f89f8
0x00401007:  call	0x40a390
0x0040100c:  ret
0x0040100d:  nop
0x0040100e:  xor	eax, eax
0x00401010:  call	dword ptr [VirtualAlloc]
; a comment-only line
0x00401016:  mov	dword ptr [ebp-0x4], eax
"""


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    p = tmp_path_factory.mktemp("asm") / "tiny.asm"
    p.write_text(TINY_ASM, encoding="utf-8")
    return E.parse_asm_file(str(p))


@pytest.fixture(scope="module")
def long_file():
    """60,000 synthetic instructions - longer than the triplet canvas."""
    rng = np.random.default_rng(0)
    n = 60_000
    return np.stack([rng.integers(10, 60, n), rng.integers(70, 75, n),
                     rng.integers(70, 75, n)], axis=1).astype(np.uint8)


# ------------------------------------------------------------- parsing -----
def test_tiny_asm_parses_to_the_hand_computed_tokens(tiny):
    assert tiny.shape == (8, 3)
    # `call 0x40a390` is a direct call to an address: it matches no API name,
    # so asm_parser maps it to UNKNOWN_API, not CALL.
    assert [int(v) for v in tiny[:, 0]] == [
        T["PUSH"], T["MOV"], T["UNKNOWN_API"], T["RET"], T["NOP_SLED"],
        T["XOR"], asm_parser.API_MAP["virtualalloc"], T["MOV"]]


def test_operand_classes(tiny):
    assert int(tiny[3, 1]) == T["PADDING"] and int(tiny[3, 2]) == T["PADDING"]
    assert int(tiny[5, 1]) == T["REG_DATA"] and int(tiny[5, 2]) == T["REG_DATA"]
    assert int(tiny[7, 1]) == T["MEM_STACK_REF"]
    assert int(tiny[7, 2]) == T["REG_DATA"]


def test_comment_only_line_is_dropped(tiny):
    assert len(tiny) == 8  # nine lines in, one of them a bare comment


# -------------------------------------------------------------- canvas -----
@pytest.mark.parametrize("variant", E.VARIANTS)
def test_canvas_shape_and_dtype(tiny, variant):
    img, mask = E.canvas_from_insns(tiny, variant)
    assert img.shape == (256, 256) and img.dtype == np.uint8
    assert mask.shape == (16, 16) and mask.dtype == np.uint8


def test_head3_writes_triplets_then_end_pad(tiny):
    img, mask = E.canvas_from_insns(tiny, "head3")
    flat = img.reshape(-1)
    assert (flat[:24] == tiny.reshape(-1)).all()
    assert (flat[24:] == END_PAD).all()
    # 24 tokens occupy row 0 columns 0..23, straddling patches (0,0) and (0,1)
    assert mask.sum() == 2 and mask[0, 0] == 1 and mask[0, 1] == 1


def test_mnem1_drops_the_operand_tokens(tiny):
    img, _ = E.canvas_from_insns(tiny, "mnem1")
    flat = img.reshape(-1)
    assert (flat[:8] == tiny[:, 0]).all()
    assert (flat[8:] == END_PAD).all()


def test_mask_marks_a_patch_active_if_it_holds_any_real_token(long_file):
    short = long_file[:1000]                     # 3,000 tokens -> rows 0..11
    _, mask = E.canvas_from_insns(short, "head3")
    assert mask[0].sum() == 16 and mask[1:].sum() == 0
    _, full = E.canvas_from_insns(long_file, "head3")
    assert full.sum() == 256                     # capped file: every patch live


# ------------------------------------------- window, stride, crops, align ---
def test_head3_keeps_the_first_65536_tokens_only(long_file):
    img, _ = E.canvas_from_insns(long_file, "head3")
    assert (img.reshape(-1) == long_file.reshape(-1)[:65536]).all()
    # the file's last instruction is NOT on the canvas
    assert not (img.reshape(-1)[-3:] == long_file[-1]).all()


def test_stride3_spans_the_whole_file(long_file):
    img, _ = E.canvas_from_insns(long_file, "stride3")
    pick = np.unique(np.linspace(0, len(long_file) - 1,
                                 E.INSN_WINDOW[3]).round().astype(np.int64))
    assert (img.reshape(-1) == long_file[pick].reshape(-1)[:65536]).all()
    assert pick[0] == 0 and pick[-1] == len(long_file) - 1


def test_stride_is_a_no_op_for_files_inside_the_window(tiny):
    assert (E.canvas_from_insns(tiny, "head3")[0]
            == E.canvas_from_insns(tiny, "stride3")[0]).all()
    assert (E.canvas_from_insns(tiny, "mnem1")[0]
            == E.canvas_from_insns(tiny, "mnem1s")[0]).all()


def test_mnem1_reaches_three_times_as_far_as_head3(long_file):
    img, _ = E.canvas_from_insns(long_file, "mnem1")
    assert (img.reshape(-1)[:len(long_file)] == long_file[:, 0]).all()
    assert (img.reshape(-1)[len(long_file):] == END_PAD).all()
    very_long = np.repeat(long_file, 2, axis=0)          # 120,000 instructions
    img2, _ = E.canvas_from_insns(very_long, "mnem1")
    assert (img2.reshape(-1) == very_long[:65536, 0]).all()


def test_crop0_is_head3_and_the_last_crop_reaches_the_end(long_file):
    head, _ = E.canvas_from_insns(long_file, "head3")
    c0, _ = E.canvas_from_insns(long_file, "crop3", crop=0)
    last, _ = E.canvas_from_insns(long_file, "crop3", crop=E.CROP_K - 1)
    assert (c0 == head).all()
    assert not (last == head).all()
    # the window ends on the final instruction; its opcode is the last slot
    assert last.reshape(-1)[-1] == long_file[-1][0]


def test_row_align_puts_an_opcode_in_every_row_0(long_file):
    img, mask = E.canvas_from_insns(long_file, "head3", row_align=True)
    rows = img.reshape(256, 256)
    opcodes = set(range(10, 60))
    assert all(int(rows[r, 0]) in opcodes for r in range(256))
    assert (rows[:, 255] == END_PAD).all()       # 85*3 = 255, one slot spare
    assert mask.sum() == 256


def test_unknown_variant_is_rejected(tiny):
    with pytest.raises(ValueError):
        E.canvas_from_insns(tiny, "no_such_variant")


# --------------------------------------- the default really is the baseline --
def test_head3_matches_unmodified_asm_parser_byte_for_byte(tmp_path, tiny):
    """Render the same .asm both ways and compare the PNG and the mask."""
    from PIL import Image
    src = tmp_path / "in"
    src.mkdir()
    (src / "tiny.asm").write_text(TINY_ASM, encoding="utf-8")
    out = tmp_path / "out"
    asm_parser.process_single_file(str(src / "tiny.asm"), str(out), "1", "tiny")
    ref_png = np.array(Image.open(out / "Class_1_Ransomware" / "tiny.png")
                       .convert("L"))
    ref_mask = np.load(out / "Class_1_Ransomware" / "tiny_vit_mask.npy")

    img, mask = E.canvas_from_insns(tiny, "head3")
    assert (img == ref_png).all()
    assert (mask == ref_mask).all()


def test_selftest_entry_point_passes():
    assert E.selftest() == 0
