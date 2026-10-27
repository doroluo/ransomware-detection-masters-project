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


# ---- library functions are not sampled ---------------------------------------------------------------------------

def blocks_of(function_id, count, first_rva, call=False):
    rows = []
    for block_id in range(count):
        base = first_rva + 0x10 * block_id
        rows.append((block_id, [(base, 'call' if call and block_id == 0 else 'mov'), (base + 1, 'retn')]))
    return rows


def mixed_cfg():
    """Two library functions (0, 3) and two others (1, 2): 3 + 3 + 2 + 2 blocks. The library function 0 calls an import
    in its first block, so it would be ranked first if library functions were ranked with the others."""
    cfg = make_cfg(
        [
            ('lib_a', 0, 0x1000, blocks_of(0, 3, 0x1000, call=True)),
            ('app_b', 1, 0x2000, blocks_of(1, 3, 0x2000)),
            ('app_c', 2, 0x3000, blocks_of(2, 2, 0x3000, call=True)),
            ('lib_d', 3, 0x4000, blocks_of(3, 2, 0x4000)),
        ],
        edges=[((0, 0), (0, 1)), ((0, 1), (0, 2)), ((1, 0), (1, 1)), ((1, 1), (1, 2)), ((2, 0), (2, 1)), ((3, 0), (3, 1))],
    )
    cfg.library_functions = {0, 3}
    return cfg


def library_blocks(cfg, sample):
    return [key for key in sample.selected_keys if key[0] in cfg.library_functions]


@pytest.mark.parametrize('budget', [1, 2, 3, 4, 5])
def test_signal_sampler_never_picks_a_library_block_while_the_others_fill_the_budget(budget):
    cfg = mixed_cfg()
    for seed in range(8):
        sample = pf.sample_signal_subgraph(cfg, seed=seed, num_blocks=budget)
        assert len(sample.selected_keys) == budget
        assert library_blocks(cfg, sample) == []


def chain_cfg(non_library_blocks, library_functions=3, library_blocks_each=200):
    """One non-library function of `non_library_blocks` blocks in a chain, plus library functions with many blocks."""
    functions = [('app', 0, 0x1000, blocks_of(0, non_library_blocks, 0x1000, call=True))]
    edges = [((0, i), (0, i + 1)) for i in range(non_library_blocks - 1)]
    for index in range(1, library_functions + 1):
        functions.append((f'lib{index}', index, 0x100000 * index, blocks_of(index, library_blocks_each, 0x100000 * index, call=True)))
        edges += [((index, i), (index, i + 1)) for i in range(library_blocks_each - 1)]
    cfg = make_cfg(functions, edges)
    cfg.library_functions = set(range(1, library_functions + 1))
    return cfg


@pytest.mark.parametrize('sampler', [pf.sample_signal_subgraph, pf.sample_connected_subgraph])
def test_a_binary_with_fewer_non_library_blocks_than_the_budget_samples_just_those(sampler):
    cfg = chain_cfg(120)
    for seed in range(5):
        sample = sampler(cfg, seed=seed, num_blocks=500)
        assert sorted(sample.selected_keys) == [(0, i) for i in range(120)]


def test_include_library_restores_the_top_up():
    cfg = chain_cfg(120)
    sample = pf.sample_signal_subgraph(cfg, seed=0, num_blocks=500, include_library=True)
    assert len(sample.selected_keys) == 500 and any(key[0] != 0 for key in sample.selected_keys)


def test_the_signal_sampler_is_not_topped_up_when_a_max_instructions_budget_is_left():
    cfg = chain_cfg(4)
    sample = pf.sample_signal_subgraph(cfg, seed=0, num_blocks=500, max_instructions=1000)
    assert sorted(sample.selected_keys) == [(0, i) for i in range(4)]


def test_signal_sampler_include_library_ranks_all_functions_together():
    cfg = mixed_cfg()
    sample = pf.sample_signal_subgraph(cfg, seed=0, num_blocks=2, include_library=True)
    assert sample.selected_keys[0] == (0, 0)  # the library block that calls an import has the top score
    plain = mixed_cfg()
    plain.library_functions = set()
    for budget in (2, 5, 10):
        for seed in range(5):
            old = pf.sample_signal_subgraph(plain, seed=seed, num_blocks=budget)  # no library flags: the old behaviour
            new = pf.sample_signal_subgraph(cfg, seed=seed, num_blocks=budget, include_library=True)
            assert new.selected_keys == old.selected_keys


def test_signal_sampler_with_only_max_instructions_skips_library_blocks():
    cfg = mixed_cfg()
    sample = pf.sample_signal_subgraph(cfg, seed=0, max_instructions=6)
    assert library_blocks(cfg, sample) == [] and sample.selected_keys
    all_library = mixed_cfg()
    all_library.library_functions = {0, 1, 2, 3}
    assert pf.sample_signal_subgraph(all_library, seed=0, max_instructions=6).selected_keys


@pytest.mark.parametrize('sampler', [pf.sample_signal_subgraph, pf.sample_connected_subgraph])
def test_a_binary_of_library_code_only_still_gets_a_sample(sampler):
    cfg = mixed_cfg()
    cfg.library_functions = {0, 1, 2, 3}
    sample = sampler(cfg, seed=1, num_blocks=3)
    assert 1 <= len(sample.selected_keys) <= 3


def test_walk_sampler_never_starts_or_walks_into_a_library_function():
    cfg = mixed_cfg()
    starts = set()
    for seed in range(40):
        sample = pf.sample_connected_subgraph(cfg, seed=seed, num_blocks=3)
        assert library_blocks(cfg, sample) == []
        starts.add(sample.selected_keys[0][0])
    assert starts == {1, 2}
    with_library = {pf.sample_connected_subgraph(cfg, seed=seed, num_blocks=3, include_library=True).selected_keys[0][0]
                    for seed in range(40)}
    assert with_library == {0, 1, 2, 3}


@pytest.mark.parametrize('sampler', [pf.sample_signal_subgraph, pf.sample_connected_subgraph])
@pytest.mark.parametrize('include_library', [False, True])
def test_both_modes_are_deterministic_for_a_seed(sampler, include_library):
    cfg = mixed_cfg()
    runs = [sampler(cfg, seed=5, num_blocks=4, include_library=include_library) for _ in range(3)]
    assert runs[0].selected_keys == runs[1].selected_keys == runs[2].selected_keys
    assert runs[0].edges == runs[1].edges == runs[2].edges


def test_load_cfg_keeps_the_library_flag(tmp_path):
    import gzip
    import json
    header = {'record': 'header', 'schema': pf.SCHEMA, 'sha256': 'ab' * 32, 'imagebase': hex(IMAGEBASE)}

    def function(function_id, name, rva, **flags):
        block = {'id': 0, 'start_rva': hex(rva), 'end_rva': hex(rva + 2),
                 'instructions': [{'rva': hex(rva), 'mnemonic': 'mov'}, {'rva': hex(rva + 1), 'mnemonic': 'retn'}]}
        return {'record': 'function', 'id': function_id, 'name': name, 'rva': hex(rva), 'blocks': [block], 'edges': [], **flags}

    records = [header, function(0, 'lib', 0x1000, library=True, thunk=False), function(1, 'app', 0x2000, library=False),
               function(2, 'bare', 0x3000), {'record': 'footer', 'complete': True}]
    path = tmp_path / 'x.jsonl.gz'
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        handle.write('\n'.join(json.dumps(record) for record in records) + '\n')
    assert pf.load_cfg(path).library_functions == {0}


def test_the_signal_sampler_returns_a_union_of_functions_that_need_not_be_connected():
    cfg = mixed_cfg()
    cfg.library_functions = set()
    sample = pf.sample_signal_subgraph(cfg, seed=0, num_blocks=8)
    assert len({key[0] for key in sample.selected_keys}) > 1
    assert not pf.connected_keys(sample.selected_keys, sample.edges)
    assert 'not one connected subgraph' in pf.sample_signal_subgraph.__doc__


def test_missing_asm_count_is_the_unmatched_selected_instructions_except_duplicate_rvas():
    """After the endp fix the report does not count matched instructions. Instructions that share an RVA with another
    are never paired on purpose and are reported under duplicate_count instead."""
    twin = ('twin', 1, 0x1100, [(0, [(0x1012, 'retn')])])          # a second function with an instruction at 0x1012
    listing = LISTING + ['', 'sub_401100      proc near', '                retn', 'sub_401100      endp']
    cfg, report = aligned(listing, ENDP_CFG + [twin])
    instructions = [i for block in cfg.blocks.values() for i in block.instructions]
    unmatched = sum(not i.matched for i in instructions)
    shared = sum(1 for i in instructions if i.rva == 0x1012)
    assert report.duplicate_count == 1 and shared == 2
    assert report.missing_asm_count == 0
    assert unmatched == report.missing_asm_count + shared


# ---- num_blocks=0 sets no block budget -----------------------------------------------------------------------------

def all_library_cfg():
    cfg = mixed_cfg()
    cfg.library_functions = {0, 1, 2, 3}
    return cfg


def sample_fields(sample):
    return (sample.selected_keys, sample.edges, sample.original_blocks, sample.original_edges, sample.original_instructions)


UNCAPPED_CASES = [
    pytest.param(mixed_cfg, {}, id='mixed'),
    pytest.param(lambda: chain_cfg(120), {}, id='chain'),
    pytest.param(all_library_cfg, {}, id='all-library'),
    pytest.param(mixed_cfg, {'include_library': True}, id='include-library'),
    pytest.param(lambda: chain_cfg(120), {'max_instructions': 100}, id='instruction-cap'),
]


@pytest.mark.parametrize('sampler', [pf.sample_signal_subgraph, pf.sample_connected_subgraph])
@pytest.mark.parametrize('build, extra', UNCAPPED_CASES)
def test_num_blocks_zero_equals_a_budget_larger_than_the_binary(sampler, build, extra):
    cfg = build()
    for seed in range(6):
        uncapped = sampler(cfg, seed=seed, num_blocks=0, **extra)
        huge = sampler(cfg, seed=seed, num_blocks=10**9, **extra)
        assert uncapped.selected_keys and sample_fields(uncapped) == sample_fields(huge)


def test_uncapped_signal_keeps_every_non_library_block_and_only_those():
    cfg = mixed_cfg()
    for seed in range(4):
        sample = pf.sample_signal_subgraph(cfg, seed=seed, num_blocks=0)
        assert sorted(sample.selected_keys) == [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
    cfg = all_library_cfg()
    sample = pf.sample_signal_subgraph(cfg, seed=0, num_blocks=0)
    assert sorted(sample.selected_keys) == sorted(cfg.blocks)           # nothing else to pick: the library blocks


def test_uncapped_walk_takes_its_whole_connected_region():
    cfg = chain_cfg(120)
    sample = pf.sample_connected_subgraph(cfg, seed=0, num_blocks=0)
    assert sorted(sample.selected_keys) == [(0, i) for i in range(120)]


def test_uncapped_still_honours_max_instructions():
    cfg = chain_cfg(120)                                                 # 2 instructions per block
    for sampler in (pf.sample_signal_subgraph, pf.sample_connected_subgraph):
        sample = sampler(cfg, seed=0, num_blocks=0, max_instructions=20)
        assert len(sample.selected_keys) == 10


def test_a_sampler_still_needs_a_budget_and_refuses_a_negative_one():
    cfg = mixed_cfg()
    for sampler in (pf.sample_signal_subgraph, pf.sample_connected_subgraph):
        with pytest.raises(ValueError, match='set num_blocks or max_instructions'):
            sampler(cfg, seed=0)
    with pytest.raises(ValueError, match='num_blocks must be'):
        pf.sample_connected_subgraph(cfg, seed=0, num_blocks=-1)
