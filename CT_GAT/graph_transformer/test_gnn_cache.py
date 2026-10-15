"""Edge kind and DFS back-edge rules of cache v2, on synthetic blocks (no cache needed)."""
import numpy as np

from build_cache import EDGE_KINDS, FCB_EXTERN, FCB_INDJUMP, classify_edge_kind, dfs_back_edges, is_cond

KIND = {name: i for i, name in enumerate(EDGE_KINDS)}


def block(start, end, last='', type_=0):
    return {'start_rva': start, 'end_rva': end, 'type': type_, 'instructions': [{'mnemonic': last}] if last else []}


def test_is_cond():
    assert all(is_cond(m) for m in ('jz', 'jne', 'JA', 'jecxz', 'loop', 'loopne'))
    assert not any(is_cond(m) for m in ('jmp', 'call', 'retn', 'mov', ''))


def test_classify_edge_kind_synthetic():
    src = block(0x100, 0x110, 'jz')
    assert classify_edge_kind(src, block(0x110, 0x120)) == KIND['fallthrough']
    assert classify_edge_kind(src, block(0x200, 0x210)) == KIND['cond_taken']
    assert classify_edge_kind(block(0x100, 0x110, 'loop'), block(0x80, 0x90)) == KIND['cond_taken']
    assert classify_edge_kind(block(0x100, 0x110, 'jmp'), block(0x200, 0x210)) == KIND['jmp']
    assert classify_edge_kind(block(0x100, 0x110, 'jmp', FCB_INDJUMP), block(0x200, 0x210)) == KIND['indirect']
    assert classify_edge_kind(src, block(0x300, 0x310, type_=FCB_EXTERN)) == KIND['extern']
    assert classify_edge_kind(block(0x100, 0x110, 'mov'), block(0x200, 0x210)) == KIND['other']
    assert classify_edge_kind(block(0x100, 0x110), block(0x200, 0x210)) == KIND['other']
    # precedence: extern beats fallthrough, fallthrough beats cond_taken and jmp
    assert classify_edge_kind(src, block(0x110, 0x120, type_=FCB_EXTERN)) == KIND['extern']
    assert classify_edge_kind(block(0x100, 0x110, 'jmp'), block(0x110, 0x120)) == KIND['fallthrough']
    # a jmp-typed indjump block with a non-jmp tail is not indirect
    assert classify_edge_kind(block(0x100, 0x110, 'call', FCB_INDJUMP), block(0x200, 0x210)) == KIND['other']


def back(n, edges, entry=0):
    return dfs_back_edges(n, np.array(edges, dtype=np.int64).reshape(-1, 2), entry).tolist()


def test_dfs_back_edges():
    # loop: 0 -> 1 -> 2 -> 1, 2 -> 3
    assert back(4, [(0, 1), (1, 2), (2, 1), (2, 3)]) == [False, False, True, False]
    # forward jump into a lower-address block: 0 -> 2, 2 -> 1, 1 -> 3; addresses say 2 -> 1 is backward, DFS says no
    edges = [(0, 2), (2, 1), (1, 3)]
    assert back(4, edges) == [False, False, False]
    starts = np.array([0, 10, 20, 30])
    assert (starts[[d for _, d in edges]] <= starts[[s for s, _ in edges]]).tolist() == [False, True, False]
    # self-loop
    assert back(2, [(0, 0), (0, 1)]) == [True, False]
    # unreachable block 2 has its own cycle, found from the restart
    assert back(4, [(0, 1), (2, 3), (3, 2)]) == [False, False, True]
    # an unreachable block with an edge into the visited region is a cross edge, not a back edge
    assert back(3, [(0, 1), (2, 1)]) == [False, False]
    # entry is not the lowest address: the back edge is whichever closes the cycle back to the entry
    assert back(3, [(2, 0), (0, 1), (1, 2)], entry=2) == [False, False, True]
    assert back(3, [(2, 0), (0, 1), (1, 2)], entry=0) == [True, False, False]
    # parallel edges are each classified; no edges is fine
    assert back(2, [(0, 1), (0, 1), (1, 0)]) == [False, False, True]
    assert back(3, []) == []
