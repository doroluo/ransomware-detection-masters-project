"""Yanping's s02 preprocess (`preprocess_full.py`, `tokenize_asm_gz.py`) on hand-written listings and hand-built CFGs.

The files are loaded as her scripts load them (`import tokenize_asm_gz`), from the stage folder.
"""
import sys
from pathlib import Path

import pytest

STAGE = Path(__file__).resolve().parents[1] / 's02_preprocess'
sys.path.insert(0, str(STAGE))
import preprocess_full as pf  # noqa: E402
import tokenize_asm_gz as tok  # noqa: E402

IMAGEBASE = 0x400000


def make_cfg(functions, edges=()):
    """FullCFG from `functions`: [(name, function_id, rva, [(block_id, [(rva, mnemonic), ...]), ...]), ...].

    A block spans from its first instruction to one byte past its last.
    """
    blocks = {}
    function_rvas = {}
    for name, function_id, rva, block_list in functions:
        function_rvas[name] = rva
        for block_id, rows in block_list:
            instructions = [
                pf.CfgInstruction(
                    file_id='f', function_id=function_id, function_name=name, block_id=block_id, index_in_block=index,
                    rva=row_rva, size=1, mnemonic=mnemonic, import_ids=[], call_resolution=None,
                    is_call=mnemonic == 'call',
                )
                for index, (row_rva, mnemonic) in enumerate(rows)
            ]
            blocks[(function_id, block_id)] = pf.CfgBlock(
                file_id='f', function_id=function_id, function_name=name, block_id=block_id,
                start_rva=rows[0][0], end_rva=rows[-1][0] + 1, block_type=None, external=False, instructions=instructions,
            )
    return pf.FullCFG(
        file_id='f', imagebase=IMAGEBASE, blocks=blocks,
        edges=[pf.CfgEdge(src=src, dst=dst) for src, dst in edges],
        imports=[], function_rvas=function_rvas, header={},
    )


# one function, an inner label, an epilogue, then `endp`
LISTING = '''\
sub_401000      proc near
                push    ebp
                mov     ebp, esp
                test    eax, eax
                jz      short loc_401010
                inc     eax

loc_401010:
                mov     esp, ebp
                pop     ebp
                retn
sub_401000      endp
'''.splitlines()

EPILOGUE = [(0x1010, 'mov'), (0x1011, 'pop'), (0x1012, 'retn')]
ENDP_CFG = [
    ('sub_401000', 0, 0x1000, [
        (0, [(0x1000, 'push'), (0x1001, 'mov'), (0x1003, 'test'), (0x1005, 'jz'), (0x1007, 'inc')]),
        (1, EPILOGUE),
    ]),
]


def aligned(lines, functions=ENDP_CFG):
    cfg = make_cfg(functions)
    keys = sorted(cfg.blocks)
    report = pf.align_sampled_blocks(cfg, keys, lines, [])
    return cfg, report


def epilogue_matched(cfg):
    return [instruction.matched for instruction in cfg.blocks[(0, 1)].instructions]


def old_note_location(line, state):
    """`_note_location` as it was before the fix: the first token of every line is a label address."""
    imagebase = tok._IMAGEBASE.search(line)
    if imagebase:
        state['imagebase'] = int(imagebase.group(1), 16)
    org = tok._ORG.search(line)
    if org:
        state['va'] = int(org.group(1), 16)
    code = line.split(';', 1)[0].strip()
    if not code or code.startswith('.'):
        return
    va = tok._va_from_token(code.split()[0])
    if va is not None:
        state['va'] = va


def test_endp_keeps_the_label_address():
    state = {'imagebase': IMAGEBASE, 'va': None}
    tok._note_location('loc_401010:', state)
    tok._note_location('sub_401000      endp', state)
    assert state['va'] == 0x401010
    tok._note_location('sub_401000      ends', state)
    assert state['va'] == 0x401010
    tok._note_location('sub_402000      proc near', state)
    assert state['va'] == 0x402000


def test_every_epilogue_instruction_is_matched_after_the_endp_fix():
    cfg, report = aligned(LISTING)
    assert epilogue_matched(cfg) == [True, True, True]
    assert all(instruction.matched for block in cfg.blocks.values() for instruction in block.instructions)
    assert report.missing_asm_count == 0


def test_without_the_fix_the_epilogue_is_unmatched(monkeypatch):
    monkeypatch.setattr(tok, '_note_location', old_note_location)
    cfg, report = aligned(LISTING)
    assert epilogue_matched(cfg) == [False, False, False]
    # the span after `endp` is re-anchored at the function start, so the report also counts the 5 matched instructions
    assert report.missing_asm_count == 8
