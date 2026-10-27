"""`--sample callwalk` of `preprocess_full.py`: a walk that crosses calls, on hand-built CFGs and a hand-written export.

Call edges only choose blocks. The saved graph is the intra-function CFG among the chosen blocks, as for `walk` and
`signal`, so no call edge, no new edge type and no change of `edge_attr`.
"""
import gzip
import json
import sys
from pathlib import Path

import pytest
import torch

STAGE = Path(__file__).resolve().parents[1] / 's02_preprocess'
sys.path.insert(0, str(STAGE))
import preprocess_full as pf  # noqa: E402
import tokenize_asm_gz as tok  # noqa: E402
from test_s02_preprocess import blocks_of, chain_cfg, library_blocks, make_cfg, mixed_cfg  # noqa: E402

import s02_collection as col  # noqa: E402

SEEDS = range(40)


def add_call(cfg, key, target, index=0):
    """Make instruction `index` of block `key` a call whose code reference is `target`."""
    instruction = cfg.blocks[key].instructions[index]
    instruction.mnemonic = 'call'
    instruction.is_call = True
    instruction.code_refs = [target]


def bridge_cfg():
    """A (0x1000): A0 -> A1. B (0x2000): B0. A0 calls B0. Without the call the two functions are two regions."""
    cfg = make_cfg(
        [('A', 0, 0x1000, blocks_of(0, 2, 0x1000)), ('B', 1, 0x2000, blocks_of(1, 1, 0x2000))],
        edges=[((0, 0), (0, 1))],
    )
    add_call(cfg, (0, 0), 0x2000)
    return cfg


def three_functions_cfg():
    """Three functions of two chained blocks each (2 instructions per block), no call between them."""
    functions = [(f'f{i}', i, 0x1000 * (i + 1), blocks_of(i, 2, 0x1000 * (i + 1))) for i in range(3)]
    return make_cfg(functions, edges=[((i, 0), (i, 1)) for i in range(3)])


# ---- (a) the walk crosses a call in both directions ----------------------------------------------------------------------------

def test_the_walk_crosses_a_call_into_the_callee_and_climbs_back_to_the_caller():
    cfg = bridge_cfg()
    starts = set()
    for seed in SEEDS:
        sample = pf.sample_callwalk_subgraph(cfg, seed=seed, num_blocks=3)
        assert sorted(sample.selected_keys) == [(0, 0), (0, 1), (1, 0)]
        starts.add(sample.selected_keys[0])
        if sample.selected_keys[0] == (1, 0):                       # started in the callee: the caller block comes next
            assert sample.selected_keys[1] == (0, 0)
        assert sample.sampling_info['regions'] == 1
        walked = pf.sample_connected_subgraph(cfg, seed=seed, num_blocks=3)
        assert len(walked.selected_keys) < 3                         # `walk` stays in one function
    assert starts == {(0, 0), (0, 1), (1, 0)}


def test_call_edges_are_caller_block_to_callee_entry_block_one_per_pair():
    cfg = bridge_cfg()
    add_call(cfg, (0, 0), 0x2000, index=1)                           # a second call from the same block to the same callee
    edges = pf.call_edges_of(cfg)
    assert len(edges) == 1
    edge = edges[0]
    assert (edge.src, edge.dst, edge.edge_type, edge.is_back_edge) == ((0, 0), (1, 0), 'call', False)
    assert edge.is_back_address is False                             # the callee starts above the caller block
    recursive = bridge_cfg()
    add_call(recursive, (1, 0), 0x1000)                              # B calls A: the target is below the caller block
    back = {(e.src, e.dst): e for e in pf.call_edges_of(recursive)}
    assert back[((1, 0), (0, 0))].is_back_address is True


def test_an_import_call_a_non_function_target_and_a_self_call_give_no_walk_neighbor():
    cfg = bridge_cfg()
    cfg.blocks[(0, 0)].instructions[0].code_refs = []                # a call through an import slot has no code reference
    add_call(cfg, (0, 1), 0x1004)                                    # a target inside a function, not a function RVA
    assert pf.call_edges_of(cfg) == []
    _edges, counts = pf._call_edge_scan(cfg)
    assert counts == {'call_targets': 0, 'unresolved_call_targets': 0, 'call_edges': 0}
    recursive = bridge_cfg()
    add_call(recursive, (0, 0), 0x1000)                              # A0 calls A: a self edge is no neighbor
    sample = pf.sample_callwalk_subgraph(recursive, seed=0, num_blocks=3)
    assert len(sample.selected_keys) == 3 and len(set(sample.selected_keys)) == 3


# ---- (b) library functions are never started in nor entered ---------------------------------------------------------------

def test_callwalk_never_starts_or_walks_into_a_library_function():
    cfg = mixed_cfg()
    add_call(cfg, (1, 0), 0x1000)                                    # app_b calls lib_a
    add_call(cfg, (0, 1), 0x3000)                                    # lib_a calls app_c
    add_call(cfg, (2, 0), 0x4000)                                    # app_c calls lib_d
    starts = set()
    for seed in SEEDS:
        sample = pf.sample_callwalk_subgraph(cfg, seed=seed, num_blocks=5)
        assert library_blocks(cfg, sample) == []
        assert len(sample.selected_keys) == 5                        # all of app_b and app_c, over restarts
        starts.add(sample.selected_keys[0][0])
    assert starts == {1, 2}
    with_library = {pf.sample_callwalk_subgraph(cfg, seed=seed, num_blocks=3, include_library=True).selected_keys[0][0]
                    for seed in SEEDS}
    assert with_library == {0, 1, 2, 3}


def test_a_binary_of_library_code_only_still_gets_a_callwalk_sample():
    cfg = mixed_cfg()
    cfg.library_functions = {0, 1, 2, 3}
    sample = pf.sample_callwalk_subgraph(cfg, seed=1, num_blocks=3)
    assert len(sample.selected_keys) == 3


# ---- (c) the start is the seeded random block of `walk` -----------------------------------------------------------------------

def test_the_start_is_the_seeded_random_block_and_not_the_entry_point():
    cfg = mixed_cfg()
    starts = set()
    for seed in SEEDS:
        sample = pf.sample_callwalk_subgraph(cfg, seed=seed, num_blocks=3)
        assert sample.selected_keys[0] == pf.sample_connected_subgraph(cfg, seed=seed, num_blocks=3).selected_keys[0]
        starts.add(sample.selected_keys[0])
    assert len(starts) > 1


def test_without_calls_and_inside_one_region_callwalk_is_walk():
    cfg = chain_cfg(120)
    for seed in range(6):
        walked = pf.sample_connected_subgraph(cfg, seed=seed, num_blocks=50)
        called = pf.sample_callwalk_subgraph(cfg, seed=seed, num_blocks=50)
        assert called.selected_keys == walked.selected_keys and called.edges == walked.edges
        assert called.sampling_info['regions'] == 1


# ---- (d) restarts --------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('budget, regions', [(1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (6, 3), (100, 3)])
def test_a_walk_below_the_budget_restarts_and_meets_it_whenever_the_binary_has_the_blocks(budget, regions):
    cfg = three_functions_cfg()
    for seed in range(10):
        sample = pf.sample_callwalk_subgraph(cfg, seed=seed, num_blocks=budget)
        assert len(sample.selected_keys) == len(set(sample.selected_keys)) == min(budget, 6)
        assert sample.sampling_info['regions'] == regions


def test_uncapped_takes_every_allowed_block_over_all_restarts():
    cfg = mixed_cfg()
    for seed in range(6):
        sample = pf.sample_callwalk_subgraph(cfg, seed=seed, num_blocks=0)
        assert sorted(sample.selected_keys) == [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
        assert sample.sampling_info['regions'] == 2
        assert sample.sampling_info == pf.sample_callwalk_subgraph(cfg, seed=seed, num_blocks=10**9).sampling_info
    huge = pf.sample_callwalk_subgraph(cfg, seed=3, num_blocks=10**9)
    assert huge.selected_keys == pf.sample_callwalk_subgraph(cfg, seed=3, num_blocks=0).selected_keys


def test_a_restart_block_that_would_break_max_instructions_is_passed_over():
    cfg = three_functions_cfg()                                      # 2 instructions per block
    for seed in range(10):
        sample = pf.sample_callwalk_subgraph(cfg, seed=seed, num_blocks=0, max_instructions=6)
        assert sum(cfg.blocks[key].instruction_count for key in sample.selected_keys) == 6
        assert len(sample.selected_keys) == 3


def test_callwalk_asks_for_a_budget_like_the_other_samplers():
    cfg = mixed_cfg()
    with pytest.raises(ValueError, match='set num_blocks or max_instructions'):
        pf.sample_callwalk_subgraph(cfg, seed=0)
    with pytest.raises(ValueError, match='num_blocks must be'):
        pf.sample_callwalk_subgraph(cfg, seed=0, num_blocks=-1)


# ---- (e) determinism -----------------------------------------------------------------------------------------------------------

def test_callwalk_is_deterministic_for_a_seed():
    cfg = bridge_cfg()
    runs = [pf.sample_callwalk_subgraph(cfg, seed=5, num_blocks=2) for _ in range(3)]
    assert runs[0].selected_keys == runs[1].selected_keys == runs[2].selected_keys
    assert runs[0].edges == runs[1].edges == runs[2].edges and runs[0].sampling_info == runs[1].sampling_info
    assert len({tuple(pf.sample_callwalk_subgraph(cfg, seed=seed, num_blocks=2).selected_keys) for seed in SEEDS}) > 1


# ---- (f) the saved edges are the intra-function CFG edges among the chosen blocks ---------------------------------------------

def test_saved_edges_are_the_induced_cfg_edges_and_no_call_edge():
    cfg = bridge_cfg()
    for budget in (2, 3):
        for seed in SEEDS:
            sample = pf.sample_callwalk_subgraph(cfg, seed=seed, num_blocks=budget)
            assert sample.edges == pf.induced_edges(cfg.edges, set(sample.selected_keys))
            assert all(edge.edge_type != 'call' for edge in sample.edges)


# ---- (g) load_cfg keeps the code references ----------------------------------------------------------------------------------

def header(imagebase=0x400000):
    return {'record': 'header', 'schema': pf.SCHEMA, 'sha256': 'ab' * 32, 'imagebase': hex(imagebase), 'pe': {'imports': []}}


def function(function_id, name, rva, blocks, edges=(), **flags):
    """blocks: [(block_id, [(rva, mnemonic, code_refs or None), ...])]; a block ends one byte after its last instruction."""
    records = []
    for block_id, rows in blocks:
        instructions = []
        for row_rva, mnemonic, refs in rows:
            item = {'rva': hex(row_rva), 'mnemonic': mnemonic, 'size': 1}
            if mnemonic == 'call':
                item['is_call'] = True
            if refs is not None:
                item['code_refs'] = refs
            instructions.append(item)
        records.append({'id': block_id, 'start_rva': hex(rows[0][0]), 'end_rva': hex(rows[-1][0] + 1),
                        'instructions': instructions})
    return {'record': 'function', 'id': function_id, 'name': name, 'rva': hex(rva), 'blocks': records,
            'edges': [list(edge) for edge in edges], **flags}


def write_export(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        handle.write('\n'.join(json.dumps(record) for record in [*records, {'record': 'footer', 'complete': True}]) + '\n')
    return path


def export_records(imagebase=0x400000):
    """f0 (0x1000): b0 calls f1, then b1. f1 (0x2000): b0. f2 (0x3000): b0 calls f3's address. f3 (0x4000): its only block
    starts at 0x4010 (a chunked function: the function RVA is not a block start)."""
    return [
        header(imagebase),
        function(0, 'sub_401000', 0x1000, [(0, [(0x1000, 'push', None), (0x1001, 'mov', None), (0x1003, 'call', [0x2000])]),
                                           (1, [(0x1008, 'pop', None), (0x1009, 'retn', None)])], edges=[(0, 1)]),
        function(1, 'sub_402000', 0x2000, [(0, [(0x2000, 'mov', []), (0x2001, 'retn', None)])]),
        function(2, 'sub_403000', 0x3000, [(0, [(0x3000, 'call', [0x4000, 0x3002]), (0x3001, 'retn', None)])]),
        function(3, 'sub_404000', 0x4000, [(0, [(0x4010, 'mov', None), (0x4011, 'retn', None)])]),
    ]


def test_load_cfg_keeps_the_code_references_of_every_instruction(tmp_path):
    cfg = pf.load_cfg(write_export(tmp_path / 'x.jsonl.gz', export_records()))
    refs = {instruction.rva: instruction.code_refs for block in cfg.blocks.values() for instruction in block.instructions}
    assert refs[0x1003] == [0x2000] and refs[0x3000] == [0x4000, 0x3002]
    assert refs[0x1000] == [] and refs[0x2000] == [] and refs[0x2001] == []


def test_load_cfg_skips_a_malformed_code_reference_and_converts_virtual_addresses(tmp_path):
    records = export_records()
    records[1]['blocks'][0]['instructions'][2]['code_refs'] = [0x2000, 'junk', None, True, 1.5]
    assert pf.load_cfg(write_export(tmp_path / 'a.jsonl.gz', records)).blocks[(0, 0)].instructions[2].code_refs == [0x2000]
    base = 0x400000                                                  # every address is a VA: all of them are converted
    shifted = export_records(base)
    for record in shifted[1:]:
        record['rva'] = hex(int(record['rva'], 16) + base)
        for block in record['blocks']:
            block['start_rva'] = hex(int(block['start_rva'], 16) + base)
            block['end_rva'] = hex(int(block['end_rva'], 16) + base)
            for instruction in block['instructions']:
                instruction['rva'] = int(instruction['rva'], 16) + base
                instruction['code_refs'] = [ref + base for ref in instruction.get('code_refs', [])]
    cfg = pf.load_cfg(write_export(tmp_path / 'b.jsonl.gz', shifted))
    assert cfg.blocks[(0, 0)].instructions[2].rva == 0x1003 and cfg.blocks[(0, 0)].instructions[2].code_refs == [0x2000]
    assert pf.call_edges_of(cfg)[0].dst == (1, 0)


# ---- a call target that is a function RVA but not a block start is counted ---------------------------------------------------

def test_a_function_rva_that_is_no_block_start_is_counted_as_unresolved(tmp_path):
    cfg = pf.load_cfg(write_export(tmp_path / 'x.jsonl.gz', export_records()))
    edges, counts = pf._call_edge_scan(cfg)
    assert [(e.src, e.dst) for e in edges] == [((0, 0), (1, 0))]
    assert counts == {'call_targets': 2, 'unresolved_call_targets': 1, 'call_edges': 1}   # 0x3002 is no function RVA
    sample = pf.sample_callwalk_subgraph(cfg, seed=0, num_blocks=0)
    assert sample.sampling_info == {'mode': 'callwalk', 'regions': 3, **counts}
    assert len(sample.selected_keys) == 5


# ---- the whole sample: manifest, payload, readers -------------------------------------------------------------------------------

def make_sample(tmp_path, mode, **kwargs):
    sha = 'ab' * 32
    asm = tmp_path / f'{sha}.asm.gz'
    with gzip.open(asm, 'wt', encoding='utf-8') as handle:
        handle.write(col.listing([]))
    cfg_path = write_export(tmp_path / f'{sha}.jsonl.gz', export_records())
    result = pf.process_sample(asm, cfg_path, label=1, seed=0, api_vocab=tok.ApiVocab(), num_blocks=8, sample_mode=mode,
                               file_id=sha, **kwargs)
    return result, pf.save_sample(result, tmp_path / f'out_{mode}')


def test_process_sample_callwalk_saves_only_cfg_edges_and_records_the_call_counts(tmp_path):
    result, (manifest_path, tensor_path) = make_sample(tmp_path, 'callwalk')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['sampling'] == {'sampled_blocks': 5, 'sampled_instructions': 11, 'mode': 'callwalk', 'regions': 3,
                                    'call_targets': 2, 'unresolved_call_targets': 1, 'call_edges': 1}
    assert [(e['src_function_id'], e['src_block_id'], e['dst_function_id'], e['dst_block_id'])
            for e in manifest['induced_edges']] == [(0, 0, 0, 1)]
    assert {e['edge_type'] for e in manifest['induced_edges']} <= set(pf.EDGE_TYPE_NAMES)
    assert manifest['edge_type_names'] == list(pf.EDGE_TYPE_NAMES) and len(pf.EDGE_TYPE_NAMES) == 7
    payload = torch.load(tensor_path, weights_only=False)
    assert payload['edge_index'].shape == (2, 1) and payload['edge_attr'].shape == (1, 7)
    assert payload['edge_type_names'] == list(pf.EDGE_TYPE_NAMES) and payload['graph_format_version'] == 3
    assert result['sample'].edges == pf.induced_edges(result['cfg'].edges, set(result['sample'].selected_keys))


@pytest.mark.parametrize('mode', ['walk', 'signal'])
def test_walk_and_signal_manifests_carry_no_new_sampling_field(tmp_path, mode):
    _result, (manifest_path, _tensor_path) = make_sample(tmp_path, mode)
    sampling = json.loads(manifest_path.read_text(encoding='utf-8'))['sampling']
    assert set(sampling) == {'sampled_blocks', 'sampled_instructions'}


def test_a_sample_mode_other_than_the_three_is_refused(tmp_path):
    with pytest.raises(ValueError, match="'walk', 'signal' or 'callwalk'"):
        make_sample(tmp_path, 'cross')


def test_fivechannel_reads_a_callwalk_payload(tmp_path):
    from s03a_transformer.five_channel import train as fc
    _result, (manifest_path, tensor_path) = make_sample(tmp_path, 'callwalk')
    sample_id = json.loads(manifest_path.read_text(encoding='utf-8'))['sample_id']
    binaries = fc.load_binaries(tensor_path.parent, {sample_id: 1}, 16)
    assert binaries[sample_id].n_instructions == 11 and binaries[sample_id].head.shape[1] == 5


def test_train_b_make_graph_reads_a_callwalk_payload(tmp_path):
    from experiments.team_pipelines import common as tp, train_b
    try:
        gl = tp.import_team('graph_full_loader')
    except tp.TeamSourceNotConfigured as error:
        pytest.skip(str(error))
    pytest.importorskip('torch_geometric')
    _result, (_manifest_path, tensor_path) = make_sample(tmp_path, 'callwalk')
    payload = train_b.slim(torch.load(tensor_path, weights_only=False))
    data = train_b.make_graph(None, gl, None, payload, None)
    assert data.x.shape == (5, 141) and data.edge_index.shape == (2, 1) and data.edge_attr.shape == (1, 8)


# ---- the adapters: build_b and match_acceptance ----------------------------------------------------------------------------------

def test_build_b_takes_callwalk_and_still_refuses_other_modes(monkeypatch):
    from experiments.team_pipelines import build_b

    class Parsed(Exception):
        pass

    def stop():
        raise Parsed

    monkeypatch.setattr(build_b, 'require_cache', stop)
    with pytest.raises(Parsed):
        build_b.main(['--sample', 'callwalk'])
    with pytest.raises(SystemExit) as refused:
        build_b.main(['--sample', 'cross'])
    assert refused.value.code == 2


def test_match_acceptance_rebuilds_with_the_sampler_of_the_build(tmp_path, monkeypatch):
    from experiments.team_pipelines import build_b, match_acceptance
    (tmp_path / 'settings.json').write_text(json.dumps({'sample': 'callwalk', 'num_blocks': 500, 'seed': 0}), encoding='utf-8')
    seen = []
    monkeypatch.setattr(build_b, 'main', lambda argv: seen.append(argv))
    match_acceptance.frozen_rebuild(tmp_path, 'repo')(['a' * 64], tmp_path / 'out')
    argv = seen[0]
    assert argv[argv.index('--sample') + 1] == 'callwalk' and argv[argv.index('--source') + 1] == 'repo'
