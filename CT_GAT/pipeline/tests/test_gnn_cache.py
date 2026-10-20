"""Edge kind and DFS back-edge rules of cache v2, and the append-only vocab, on synthetic data (no cache needed)."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from s02_preprocess.cache import build_cache
from s02_preprocess.cache.build_cache import EDGE_KINDS, FCB_EXTERN, FCB_INDJUMP, classify_edge_kind, dfs_back_edges, is_cond

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


def local_npz(path, mn):
    strings = np.array(sorted(set(mn)), dtype=np.str_)
    ids = np.searchsorted(strings, np.array(mn, dtype=np.str_)).astype(np.int32)
    empty = np.zeros(0, dtype=np.int32)
    build_cache.write_npz(path, {'mn': ids, 'mn_strings': strings, 'sig': empty, 'sig_strings': np.array([], dtype=np.str_),
                                 'api': empty, 'api_strings': np.array([], dtype=np.str_),
                                 'ids_kind': 'local', 'vocab_sha': ''})


# the nine behaviour categories of vocab.json['beh'] (an enumeration of names, not the rules that assign them): the
# test gives `finalize` this stand-in for token_mapping, which is not part of this package
BEHAVIOR_NAMES = ('NONE', 'CRYPTO_OPCODE', 'DECRYPT', 'ENCRYPT', 'KEY_SETUP', 'FILE_ENUM', 'FILE_READ', 'FILE_WRITE',
                  'FILE_DELETE')


def test_finalize_appends_and_keeps_earlier_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(build_cache, 'CACHE', tmp_path)
    monkeypatch.setattr(build_cache, 'behavior_rules',
                        lambda: SimpleNamespace(BEHAVIOR_MAP={n: i for i, n in enumerate(BEHAVIOR_NAMES)}))
    (tmp_path / 'samples').mkdir()
    a, b = {'sample_id': 'a'}, {'sample_id': 'b'}
    local_npz(build_cache.npz_path('a'), ['push', 'mov', 'mov'])
    build_cache.finalize([a])
    first = build_cache.npz_path('a').read_bytes()
    local_npz(build_cache.npz_path('b'), ['xor', 'add', 'mov'])
    sizes, sha, rewritten = build_cache.finalize([a, b])
    vocab = json.loads((tmp_path / 'vocab.json').read_text(encoding='utf-8'))
    assert vocab['mn'] == ['mov', 'push', 'add', 'xor'] and rewritten == 1
    assert build_cache.npz_path('a').read_bytes() == first
    with np.load(build_cache.npz_path('b')) as z:
        assert [vocab['mn'][i] for i in z['mn']] == ['xor', 'add', 'mov']
        assert z['mn_u'].tolist() == [0, 2, 3] and z['mn_strings'].tolist() == ['mov', 'add', 'xor']
        assert str(z['vocab_sha']) == sha
    generations = json.loads((tmp_path / 'vocab_generations.json').read_text(encoding='utf-8'))
    with np.load(build_cache.npz_path('a')) as z:
        assert [g['vocab_sha'] for g in generations] == [str(z['vocab_sha']), sha]
    assert [g['sizes']['mn'] for g in generations] == [2, 4]
    assert build_cache.finalize([a, b])[2] == 0


def hide_token_mapping(monkeypatch, then_real=False):
    """build_cache.importlib.import_module raises ImportError for token_mapping on the first call (and on every call unless
    `then_real`), as on a machine where the module is not on sys.path."""
    real, calls = build_cache.importlib.import_module, []

    def fake(name, *args):
        calls.append(name)
        if name == 'token_mapping' and not (then_real and len(calls) > 1):
            raise ImportError(name)
        return real(name, *args)
    monkeypatch.setattr(build_cache.importlib, 'import_module', fake)


def test_the_behaviour_rules_are_not_part_of_the_package(monkeypatch, tmp_path):
    """Without token_mapping on sys.path and without a configured folder, building says what is missing."""
    monkeypatch.setenv('GT_DATASETS', str(tmp_path / 'absent.json'))
    monkeypatch.delenv('GT_TOKEN_MAPPING_DIR', raising=False)
    hide_token_mapping(monkeypatch)
    with pytest.raises(ImportError, match='GT_TOKEN_MAPPING_DIR'):
        build_cache.behavior_rules()


def test_the_behaviour_rules_are_found_through_the_configured_folder(monkeypatch, tmp_path):
    (tmp_path / 'token_mapping.py').write_text('BEHAVIOR_MAP = {"NONE": 0}\n', encoding='utf-8')
    monkeypatch.setenv('GT_TOKEN_MAPPING_DIR', str(tmp_path))
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.delitem(sys.modules, 'token_mapping', raising=False)
    hide_token_mapping(monkeypatch, then_real=True)
    try:
        assert build_cache.behavior_rules().BEHAVIOR_MAP == {'NONE': 0}
    finally:
        sys.modules.pop('token_mapping', None)
