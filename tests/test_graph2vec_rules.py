"""Tests for graph2vec_pipeline/ and rules_pipeline/.

These run on synthetic .asm fixtures written into tmp_path, so they need
neither the 7.2GB disassembly tree nor the cohort CSVs. What they pin down is
the part of each prototype that a silent bug would make meaningless:

  * the disassembly parser's three line shapes (section header, instruction
    with a tab, `.skip` gap) and prefixed mnemonics (`rep movsd`)
  * basic-block cutting: leaders at branch targets and after terminators, no
    fall-through across an undecoded gap or a section boundary
  * the block cap actually binds
  * WL relabelling is deterministic across processes (it must be: the whole
    corpus is hashed once and cached)
  * the WL vocabulary is taken from TRAIN rows only
  * n-gram mining recovers a planted discriminative n-gram, and the Laplace
    prior keeps lift finite at zero counts
  * behaviour rules fire on the constructs they claim to detect, and density
    thresholds are fitted on train rows only
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph2vec_pipeline import cfg as C  # noqa: E402
from graph2vec_pipeline import embed as E  # noqa: E402
from rules_pipeline import behaviour_rules as B  # noqa: E402
from rules_pipeline import ngram_rules as N  # noqa: E402


def write_asm(path: Path, lines) -> Path:
    """`lines` are (addr, mnemonic, operands) or a raw '; section ...' string."""
    out = []
    for item in lines:
        if isinstance(item, str):
            out.append(item)
        else:
            a, m, o = item
            out.append(f"0x{a:08x}:  {m}\t{o}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def test_parse_line_shapes(tmp_path):
    p = write_asm(tmp_path / "a.asm", [
        "; section .text va=0x401000 size=64",
        (0x401000, "mov", "eax, ebx"),
        (0x401002, "rep movsd", "dword ptr es:[edi], dword ptr [esi]"),
        (0x401004, ".skip", "3 bytes"),
        (0x401007, "ret", ""),
        "; section .data va=0x402000 size=16",
        (0x402000, "nop", ""),
    ])
    a = C.parse_asm(p)
    assert a.insns_read == 4                    # .skip is not an instruction
    assert a.base == ["mov", "movsd", "ret", "nop"]   # prefix stripped
    assert a.gap_before == [False, False, True, False]
    assert a.sec == [0, 0, 0, 1]
    assert a.n_sections == 2


def test_parse_respects_max_insns(tmp_path):
    p = write_asm(tmp_path / "b.asm",
                  ["; section .text va=0x401000 size=1"] +
                  [(0x401000 + i, "nop", "") for i in range(50)])
    a = C.parse_asm(p, max_insns=10)
    assert a.insns_read == 10 and a.truncated is True


def test_direct_target_only_from_bare_immediate(tmp_path):
    p = write_asm(tmp_path / "c.asm", [
        "; section .text va=0x401000 size=32",
        (0x401000, "call", "0x401010"),
        (0x401005, "call", "dword ptr [0x405128]"),
        (0x401010, "ret", ""),
    ])
    a = C.parse_asm(p)
    assert a.target_imm == [0x401010, -1, -1]


# ---------------------------------------------------------------------------
# CFG
# ---------------------------------------------------------------------------
def _diamond(tmp_path) -> Path:
    #  0x1000 cmp / 0x1001 je 0x1005   -> two successors
    #  0x1002 mov / 0x1003 jmp 0x1006
    #  0x1005 xor
    #  0x1006 ret
    return write_asm(tmp_path / "d.asm", [
        "; section .text va=0x401000 size=32",
        (0x1000, "cmp", "eax, ebx"),
        (0x1001, "je", "0x1005"),
        (0x1002, "mov", "ecx, edx"),
        (0x1003, "jmp", "0x1006"),
        (0x1005, "xor", "eax, eax"),
        (0x1006, "ret", ""),
    ])


def test_cfg_diamond(tmp_path):
    g = C.cfg_from_file(_diamond(tmp_path))
    # leaders: 0 (start), 2 (fall-through after je), 4 (branch target),
    # 5 (target of jmp / after jmp)
    assert g.stats["n_blocks"] == 4
    assert set(g.edges) == {(0, 1), (0, 2), (1, 3), (2, 3)}
    assert g.stats["resolved_targets"] == 2


def test_no_fallthrough_across_gap_or_section(tmp_path):
    p = write_asm(tmp_path / "e.asm", [
        "; section .text va=0x401000 size=32",
        (0x1000, "mov", "eax, 1"),
        (0x1001, ".skip", "4 bytes"),
        (0x1005, "mov", "ebx, 2"),
        "; section .text2 va=0x402000 size=16",
        (0x2000, "mov", "ecx, 3"),
    ])
    g = C.cfg_from_file(p)
    assert g.stats["n_blocks"] == 3
    assert g.edges == []            # gap and section boundary both cut


def test_call_edges_and_return_stop(tmp_path):
    p = write_asm(tmp_path / "f.asm", [
        "; section .text va=0x401000 size=32",
        (0x1000, "call", "0x1003"),
        (0x1001, "mov", "eax, 1"),
        (0x1002, "ret", ""),
        (0x1003, "ret", ""),
    ])
    g = C.cfg_from_file(p, include_calls=True)
    assert g.stats["call_edges"] == 1
    assert (0, 1) in g.edges            # fall-through past the call
    g2 = C.cfg_from_file(p, include_calls=False)
    assert g2.stats["call_edges"] == 0


def test_block_cap_binds(tmp_path):
    lines = ["; section .text va=0x401000 size=4096"]
    for i in range(400):
        lines.append((0x1000 + 2 * i, "nop", ""))
        lines.append((0x1001 + 2 * i, "ret", ""))
    p = write_asm(tmp_path / "g.asm", lines)
    g = C.cfg_from_file(p, max_blocks=50)
    assert g.stats["n_blocks"] == 50
    assert g.stats["block_capped"] is True


def test_mnem_class_table():
    assert C.mnem_class("aesenc") == "CRYPTO"
    assert C.mnem_class("jne") == "COND"
    assert C.mnem_class("jmp") == "JMP"
    assert C.mnem_class("loopne") == "COND"
    assert C.mnem_class("rol") == "SHIFT"
    assert C.mnem_class("movsd", "xmm0, xmm1") == "SIMD"
    assert C.mnem_class("movsd", "dword ptr es:[edi], dword ptr [esi]") == "STRING"
    assert C.mnem_class("setne") == "SETCC"


# ---------------------------------------------------------------------------
# WL
# ---------------------------------------------------------------------------
def test_wl_is_deterministic_across_processes(tmp_path):
    p = _diamond(tmp_path)
    code = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {str(REPO_ROOT)!r})
        from graph2vec_pipeline.cfg import cfg_from_file, wl_labels
        d = wl_labels(cfg_from_file({str(p)!r}), 2)
        print(sum(sorted(d)) % (2**31), len(d))
    """)
    env_a = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, check=True).stdout.strip()
    env_b = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, check=True).stdout.strip()
    assert env_a == env_b and env_a != ""


def test_wl_depth_adds_labels(tmp_path):
    g = C.cfg_from_file(_diamond(tmp_path))
    d0, d1, d2 = (C.wl_labels(g, i) for i in (0, 1, 2))
    assert sum(d0.values()) == g.stats["n_blocks"]
    assert sum(d2.values()) > sum(d1.values()) > sum(d0.values())


def test_wl_separates_different_shapes(tmp_path):
    a = C.cfg_from_file(_diamond(tmp_path))
    p = write_asm(tmp_path / "chain.asm", [
        "; section .text va=0x401000 size=32",
        (0x1000, "mov", "eax, 1"),
        (0x1001, "jmp", "0x1002"),
        (0x1002, "ret", ""),
    ])
    b = C.cfg_from_file(p)
    assert set(C.wl_labels(a, 2)) != set(C.wl_labels(b, 2))


# ---------------------------------------------------------------------------
# tuned graph-construction options (block cap, cross-section targets, import
# thunks as extern nodes, edge labels, windowed sampling)
# ---------------------------------------------------------------------------
def _thunk_asm(tmp_path) -> Path:
    #  0x1000 push        |  0x1001 call [0x405128]  (import A)
    #  0x1002 call [0x405128]  same import -> same extern node
    #  0x1003 call [0x40512c]  a different import
    #  0x1004 ret
    return write_asm(tmp_path / "thunk.asm", [
        "; section .text va=0x401000 size=32",
        (0x1000, "push", "ebp"),
        (0x1001, "call", "dword ptr [0x405128]"),
        (0x1002, "call", "dword ptr [0x405128]"),
        (0x1003, "call", "dword ptr [0x40512c]"),
        (0x1004, "ret", ""),
    ])


def test_mem_operand_shapes():
    assert C._mem_operand("dword ptr [0x405128]") == (0x405128, False)
    assert C._mem_operand("qword ptr [rip + 0x1234]") == (0x1234, True)
    assert C._mem_operand("qword ptr [rip - 0x10]") == (-0x10, True)
    assert C._mem_operand("eax") == (-1, False)
    assert C._mem_operand("dword ptr [eax + ecx*4]") == (-1, False)


def test_rip_relative_thunk_resolves_against_next_instruction(tmp_path):
    p = write_asm(tmp_path / "rip.asm", [
        "; section .text va=0x401000 size=32",
        (0x1000, "call", "qword ptr [rip + 0x20]"),
        (0x1006, "ret", ""),
    ])
    a = C.parse_asm(p)
    # the displacement is relative to the address after the call, i.e. 0x1006
    assert a.mem_target == [0x1006 + 0x20, -1]


def test_extern_nodes_share_one_node_per_import(tmp_path):
    p = _thunk_asm(tmp_path)
    off = C.cfg_from_file(p, extern_nodes=False)
    on = C.cfg_from_file(p, extern_nodes=True)
    assert off.stats["extern_nodes"] == 0
    # two distinct import slots -> two extern nodes, three call sites
    assert on.stats["extern_nodes"] == 2
    # a call does not end a block here, but it does start one, so the three
    # call sites sit in three different blocks: three edges into two nodes.
    assert on.stats["extern_edges"] == 3
    ext_targets = [v for (u, v), t in zip(on.edges, on.etypes)
                   if t == C.E_EXTERN]
    assert len(ext_targets) == 3 and len(set(ext_targets)) == 2
    assert on.n_blocks == off.n_blocks
    assert len(on.labels) == on.n_blocks + 2
    assert on.labels[-1] == "EXTERN"
    # and they turn unresolved branch targets into resolved ones
    assert on.stats["unresolved_targets"] < off.stats["unresolved_targets"]


def test_extern_nodes_need_calls_enabled(tmp_path):
    g = C.cfg_from_file(_thunk_asm(tmp_path), extern_nodes=True,
                        include_calls=False)
    assert g.stats["extern_nodes"] == 0


def test_cross_section_targets(tmp_path):
    p = write_asm(tmp_path / "xs.asm", [
        "; section .text va=0x401000 size=16",
        (0x1000, "jmp", "0x2000"),
        "; section .itext va=0x402000 size=16",
        (0x2000, "ret", ""),
    ])
    off = C.cfg_from_file(p, cross_section=False)
    on = C.cfg_from_file(p, cross_section=True)
    assert off.edges == [] and off.stats["unresolved_targets"] == 1
    assert on.edges == [(0, 1)] and on.stats["resolved_targets"] == 1
    assert on.stats["xsec_targets"] == 1
    assert on.etypes == [C.E_BRANCH_XSEC]


def test_edge_kinds_are_labelled(tmp_path):
    p = write_asm(tmp_path / "kinds.asm", [
        "; section .text va=0x401000 size=32",
        (0x1000, "cmp", "eax, ebx"),
        (0x1001, "je", "0x1003"),
        (0x1002, "call", "0x1004"),
        (0x1003, "ret", ""),
        (0x1004, "ret", ""),
    ])
    g = C.cfg_from_file(p)
    kinds = set(g.etypes)
    assert {C.E_BRANCH, C.E_FALL, C.E_CALL} <= kinds


def test_node_attrs_align_with_labels(tmp_path):
    g = C.cfg_from_file(_thunk_asm(tmp_path), extern_nodes=True)
    for k in ("dom", "term", "szb", "flg", "blen", "seq"):
        assert len(g.attrs[k]) == len(g.labels)
    assert g.attrs["dom"][-1] == C.EXTERN_CLASS_ID


def test_windowed_parse_covers_the_whole_file(tmp_path):
    lines = ["; section .text va=0x401000 size=4096"]
    lines += [(0x1000 + i, "nop", "") for i in range(4000)]
    p = write_asm(tmp_path / "w.asm", lines)
    head = C.parse_asm(p, max_insns=400, windows=1)
    spread = C.parse_asm(p, max_insns=400, windows=4)
    assert head.insns_read == 400 and spread.insns_read == 400
    assert max(head.addr) < max(spread.addr)          # the head stops early
    assert spread.n_windows == 4
    # windows do not fall through into one another
    assert sum(spread.gap_before) >= 3
    assert len(set(spread.sec)) == 4


# ---------------------------------------------------------------------------
# vectorised WL (graph2vec_pipeline/wl.py)
# ---------------------------------------------------------------------------
def _batch_from(cfgs):
    """A wl.Batch straight from CFG objects, without touching the disk cache."""
    from graph2vec_pipeline import wl as W
    n_nodes = 0
    gid, dom, term, szb, flg, blen, seq, ext = [], [], [], [], [], [], [], []
    src, dst, et = [], [], []
    for gi, g in enumerate(cfgs):
        off = n_nodes
        for k, lst in (("dom", dom), ("term", term), ("szb", szb),
                       ("flg", flg), ("blen", blen), ("seq", seq)):
            lst.extend(g.attrs[k])
        gid.extend([gi] * len(g.labels))
        ext.extend([i >= g.n_blocks for i in range(len(g.labels))])
        for (u, v), t in zip(g.edges, g.etypes):
            src.append(off + u)
            dst.append(off + v)
            et.append(t)
        n_nodes += len(g.labels)
    a = np.asarray
    s64, d64 = a(src, np.int64), a(dst, np.int64)
    return W.Batch(
        n_graphs=len(cfgs), n_nodes=n_nodes, gid=a(gid, np.int32),
        dom=a(dom, np.uint8), term=a(term, np.uint8), szb=a(szb, np.uint8),
        flg=a(flg, np.uint8), blen=a(blen, np.uint16), seq=a(seq, np.uint32),
        is_extern=a(ext, bool), src=s64, dst=d64, etype=a(et, np.uint8),
        outdeg=np.bincount(s64, minlength=n_nodes).astype(np.int32),
        indeg=np.bincount(d64, minlength=n_nodes).astype(np.int32))


def _graphset_from(cfgs):
    """A `graph_cache.GraphSet` straight from CFG objects.

    The cache is built once at the widest construction and every narrower one
    is recovered by `wl.assemble` filtering it, so the filtering is what has to
    be tested -- not just the parser options it is supposed to reproduce.
    """
    from graph2vec_pipeline.graph_cache import GraphSet
    a = np.asarray
    cols = {k: [] for k in ("dom", "term", "szb", "flg", "blen", "seq")}
    esrc, edst, etype = [], [], []
    node_ptr, edge_ptr, nblk = [0], [0], []
    for g in cfgs:
        for k, lst in cols.items():
            lst.extend(g.attrs[k])
        for (u, v), t in zip(g.edges, g.etypes):
            esrc.append(u)
            edst.append(v)
            etype.append(t)
        node_ptr.append(node_ptr[-1] + len(g.labels))
        edge_ptr.append(edge_ptr[-1] + len(g.edges))
        nblk.append(g.n_blocks)
    shas = np.array([f"sha{i:02d}" for i in range(len(cfgs))], dtype="U64")
    types = dict(dom=np.uint8, term=np.uint8, szb=np.uint8, flg=np.uint8,
                 blen=np.uint16, seq=np.uint32)
    return GraphSet(
        shas=shas, index={str(s): i for i, s in enumerate(shas)},
        node_ptr=a(node_ptr, np.int64), edge_ptr=a(edge_ptr, np.int64),
        n_blocks=a(nblk, np.int64),
        esrc=a(esrc, np.int32), edst=a(edst, np.int32),
        etype=a(etype, np.uint8),
        **{k: a(v, t) for (k, v), t in zip(cols.items(), types.values())})


def _symmetric(tmp_path) -> Path:
    """A hand-built graph whose WL histogram has known multiplicities.

    Four blocks, and blocks 1 and 2 are identical in body *and* in context:

        0  [cmp; je 0x1004]      ->  1, 2
        1  [xor eax,eax; jmp]    ->  3
        2  [xor eax,eax; jmp]    ->  3
        3  [ret]

    so WL must keep 1 and 2 in one class at every depth. That is what makes
    the count vector non-trivial -- a graph with all-distinct labels would
    match any implementation.
    """
    return write_asm(tmp_path / "sym.asm", [
        "; section .text va=0x401000 size=64",
        (0x1000, "cmp", "eax, ebx"),
        (0x1001, "je", "0x1004"),
        (0x1002, "xor", "eax, eax"),
        (0x1003, "jmp", "0x1006"),
        (0x1004, "xor", "eax, eax"),
        (0x1005, "jmp", "0x1006"),
        (0x1006, "ret", ""),
    ])


def test_symmetric_fixture_has_the_blocks_and_edges_claimed(tmp_path):
    g = C.cfg_from_file(_symmetric(tmp_path))
    assert g.n_blocks == 4
    assert sorted(g.edges) == [(0, 1), (0, 2), (1, 3), (2, 3)]
    assert g.labels == ["CMP.cond.2", "JMP.jmp.2", "JMP.jmp.2", "RET.stop.1"]


def test_vectorised_wl_histogram_equals_cfg_wl_labels(tmp_path):
    """The two WL implementations must agree as *documents*.

    `cfg.wl_labels` hashes a sorted successor/predecessor tuple per node;
    `wl.wl_layers` sums mixed neighbour hashes instead. The label integers
    therefore differ, but the partition of nodes into WL classes -- and so the
    multiset of counts, which is all the document is -- must not.
    """
    from graph2vec_pipeline import wl as W
    g = C.cfg_from_file(_symmetric(tmp_path))
    b = _batch_from([g])
    for h in (0, 1, 2, 3):
        ref = sorted(C.wl_labels(g, h).values())
        _, cnt = np.unique(np.concatenate(W.wl_layers(b, h, "class", False)),
                           return_counts=True)
        assert sorted(cnt.tolist()) == ref, h
    # and the symmetry is really there: the two identical blocks stay merged
    assert sorted(C.wl_labels(g, 3).values()).count(2) == 4


def test_vectorised_wl_histogram_matches_on_a_graph_with_extern_nodes(tmp_path):
    from graph2vec_pipeline import wl as W
    g = C.cfg_from_file(_thunk_asm(tmp_path), extern_nodes=True)
    b = _batch_from([g])
    for h in (0, 2):
        ref = sorted(C.wl_labels(g, h).values())
        _, cnt = np.unique(np.concatenate(W.wl_layers(b, h, "class", False)),
                           return_counts=True)
        assert sorted(cnt.tolist()) == ref, h


# -- the construction options, as applied by wl.assemble to the cache -------
def test_assemble_block_cap_is_a_prefix_of_the_cached_graph(tmp_path):
    from graph2vec_pipeline import wl as W
    lines = ["; section .text va=0x401000 size=4096"]
    for i in range(40):
        lines.append((0x1000 + 2 * i, "nop", ""))
        lines.append((0x1001 + 2 * i, "ret", ""))
    g = C.cfg_from_file(write_asm(tmp_path / "many.asm", lines), max_blocks=40)
    gs = _graphset_from([g])
    assert W.assemble(gs, [0]).n_nodes == 40
    b = W.assemble(gs, [0], max_blocks=10)
    assert b.n_nodes == 10
    assert b.n_graphs == 1 and (b.gid == 0).all()


def test_assemble_edge_filters_reproduce_the_construction_flags(tmp_path):
    from graph2vec_pipeline import wl as W
    # one .asm carrying every edge kind the search can switch off
    p = write_asm(tmp_path / "all_kinds.asm", [
        "; section .text va=0x401000 size=32",
        (0x1000, "cmp", "eax, ebx"),
        (0x1001, "je", "0x1004"),
        (0x1002, "call", "0x1005"),
        (0x1003, "call", "dword ptr [0x405128]"),
        (0x1004, "jmp", "0x2000"),
        (0x1005, "ret", ""),
        "; section .itext va=0x402000 size=16",
        (0x2000, "ret", ""),
    ])
    g = C.cfg_from_file(p, include_calls=True, cross_section=True,
                        extern_nodes=True)
    kinds = set(g.etypes)
    assert {C.E_FALL, C.E_BRANCH, C.E_CALL, C.E_EXTERN,
            C.E_BRANCH_XSEC} <= kinds
    gs = _graphset_from([g])

    def kept(keep):
        return set(W.assemble(gs, [0], keep_edges=keep).etype.tolist())

    assert kept(W.ALL_EDGES) == kinds
    # calls off: no call and no extern edge survives
    no_calls = (C.E_FALL, C.E_BRANCH, C.E_BRANCH_XSEC)
    assert not ({C.E_CALL, C.E_CALL_XSEC, C.E_EXTERN} & kept(no_calls))
    # cross-section off
    same_sec = (C.E_FALL, C.E_BRANCH, C.E_CALL, C.E_EXTERN)
    assert C.E_BRANCH_XSEC not in kept(same_sec)
    # extern off drops the thunk nodes with their edges
    on = W.assemble(gs, [0], keep_edges=W.ALL_EDGES)
    off = W.assemble(gs, [0], keep_edges=tuple(
        k for k in W.ALL_EDGES if k != C.E_EXTERN))
    assert C.E_EXTERN not in set(off.etype.tolist())
    assert off.is_extern.sum() == 0 and on.is_extern.sum() >= 1
    assert off.n_nodes == on.n_nodes - int(on.is_extern.sum())


def test_assemble_matches_a_directly_built_batch(tmp_path):
    """Filtering the widest cache must give the same WL document as building
    the narrow construction from source -- that equivalence is the whole
    reason graph_cache stores one pass instead of one per option."""
    from graph2vec_pipeline import wl as W
    p = _thunk_asm(tmp_path)
    wide = C.cfg_from_file(p, include_calls=True, extern_nodes=True)
    narrow = C.cfg_from_file(p, include_calls=False, extern_nodes=False)
    keep = (C.E_FALL, C.E_BRANCH, C.E_BRANCH_XSEC)
    from_cache = W.assemble(_graphset_from([wide]), [0], keep_edges=keep)
    direct = _batch_from([narrow])
    assert from_cache.n_nodes == direct.n_nodes
    for h in (0, 2):
        a = np.unique(np.concatenate(W.wl_layers(from_cache, h, "class",
                                                 False)), return_counts=True)
        b = np.unique(np.concatenate(W.wl_layers(direct, h, "class", False)),
                      return_counts=True)
        assert a[0].tolist() == b[0].tolist()
        assert a[1].tolist() == b[1].tolist()


def test_node_labellings_are_ordered_by_granularity(tmp_path):
    """Each labelling the search can choose has to actually change the depth-0
    alphabet, and the coarser ones must not be finer than `class`."""
    from graph2vec_pipeline import wl as W
    b = _batch_from([C.cfg_from_file(_thunk_asm(tmp_path), extern_nodes=True),
                     C.cfg_from_file(_diamond(tmp_path)),
                     C.cfg_from_file(_symmetric(tmp_path))])
    n = {lb: len(np.unique(W.base_labels(b, lb))) for lb in W.LABELLINGS}
    assert set(n) == set(W.LABELLINGS)
    assert n["class"] >= n["class_noflag"] >= n["dom_term"]
    assert n["class"] >= n["lenbucket"]
    with pytest.raises(ValueError):
        W.base_labels(b, "not_a_labelling")


def test_vectorised_wl_is_deterministic_across_processes(tmp_path):
    p = _diamond(tmp_path)
    code = textwrap.dedent(f"""
        import sys, numpy as np
        sys.path.insert(0, {str(REPO_ROOT)!r})
        sys.path.insert(0, {str(REPO_ROOT / 'tests')!r})
        from test_graph2vec_rules import _batch_from
        from graph2vec_pipeline.cfg import cfg_from_file
        from graph2vec_pipeline import wl as W
        b = _batch_from([cfg_from_file({str(p)!r})])
        L = W.wl_layers(b, 3, "class", True)
        print([int(np.bitwise_xor.reduce(l) % (2**31)) for l in L])
    """)
    outs = [subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, check=True).stdout.strip()
            for _ in range(2)]
    assert outs[0] == outs[1] and outs[0] != ""


def test_vectorised_wl_depth_refines_and_separates_shapes(tmp_path):
    from graph2vec_pipeline import wl as W
    a = C.cfg_from_file(_diamond(tmp_path))
    chain = write_asm(tmp_path / "chain2.asm", [
        "; section .text va=0x401000 size=32",
        (0x1000, "mov", "eax, 1"),
        (0x1001, "jmp", "0x1002"),
        (0x1002, "ret", ""),
    ])
    b = _batch_from([a, C.cfg_from_file(chain)])
    L = W.wl_layers(b, 2, "class", False)
    assert len(L) == 3
    assert len(np.unique(L[1])) >= len(np.unique(L[0]))
    M = W.wl_documents(b, 2, "class", False, np.array([True, True]), min_df=1)
    assert M.shape[0] == 2
    assert (M[0] != M[1]).nnz > 0        # the two graphs differ as documents


def test_wl_edge_labels_change_the_document(tmp_path):
    from graph2vec_pipeline import wl as W
    b = _batch_from([C.cfg_from_file(_diamond(tmp_path))])
    off = W.wl_layers(b, 2, "class", False)[2]
    on = W.wl_layers(b, 2, "class", True)[2]
    assert not np.array_equal(off, on)


def test_layer_matrix_min_df_uses_train_rows_only():
    from graph2vec_pipeline import wl as W
    # graph 2 is the only one carrying label 0xAAAA and it is not in train
    gid = np.array([0, 0, 1, 2, 2], dtype=np.int32)
    lab = np.array([11, 22, 11, 11, 0xAAAA], dtype=np.uint64)
    train = np.array([True, True, False])
    M = W.layer_matrix(gid, lab, 3, train, min_df=2)
    assert M.shape == (3, 1)          # only label 11 reaches df>=2 on train
    assert M[2].sum() == 1.0


# -- reconstructing a searched configuration from its recorded row ----------
# final_eval.py scores the top-5 CV configurations post hoc, and those are
# recorded in cv_search.csv only as rendered strings plus flat columns.
def test_parse_model_key_round_trips_every_grid_shape():
    from graph2vec_pipeline.tune import Model, parse_model_key
    for m in (Model("LR", (("C", 1.0),)),
              Model("LR", (("C", 0.1), ("class_weight", "balanced"))),
              Model("LinearSVC", (("C", 0.01),)),
              Model("SVM-RBF", (("C", 10.0), ("gamma", "scale"))),
              Model("SVM-RBF", (("C", 100.0), ("gamma", 0.01),
                                ("class_weight", "balanced"))),
              Model("RF", (("n_estimators", 1000), ("max_depth", None),
                           ("min_samples_leaf", 3))),
              Model("MLP", (("hidden_layer_sizes", (256,)), ("alpha", 1e-4))),
              # the comma inside the tuple is why the split tracks brackets
              Model("MLP", (("hidden_layer_sizes", (256, 128)),
                            ("alpha", 1e-2))),
              Model("RF", (("n_estimators", 400),), True)):
        assert parse_model_key(m.key()) == m, m.key()
    with pytest.raises(ValueError):
        parse_model_key("LR")


def test_obj_from_row_round_trips_a_search_row():
    from graph2vec_pipeline.tune import Construction, Model, Rep, obj_from_row
    c = Construction(max_blocks=20_000, windows=4, calls=False,
                     cross_section=False, extern=False)
    r = Rep("wl_svd", "dom_term", h=1, edge_labels=False, norm="binary",
            min_df=3, with_size=False, dim=100)
    m = Model("MLP", (("hidden_layer_sizes", (256,)), ("alpha", 1e-4)))
    row = {"construction": c.key(), "representation": r.key(),
           "model": m.key(), "rep_kind": r.kind, "labelling": r.labelling,
           "h": r.h, "edge_labels": "False", "norm": r.norm,
           "min_df": r.min_df, "with_size": "False", "max_blocks": 20_000,
           "windows": 4, "calls": "False", "cross_section": "False",
           "extern": "False", "dim": 100}
    assert obj_from_row(row) == (c, r, m)
    # a row whose strings and columns disagree must not fit silently
    with pytest.raises(ValueError):
        obj_from_row({**row, "h": 3})


def test_size_features_track_the_graph(tmp_path):
    from graph2vec_pipeline import wl as W
    b = _batch_from([C.cfg_from_file(_diamond(tmp_path)),
                     C.cfg_from_file(_thunk_asm(tmp_path), extern_nodes=True)])
    S = W.size_features(b)
    assert S.shape == (2, len(W.SIZE_FEATURES))
    i_nodes = W.SIZE_FEATURES.index("n_nodes")
    i_ext = W.SIZE_FEATURES.index("n_extern")
    assert S[0, i_nodes] == 4 and S[0, i_ext] == 0
    assert S[1, i_ext] == 2


# ---------------------------------------------------------------------------
# embedding
# ---------------------------------------------------------------------------
def test_vocabulary_comes_from_train_only():
    # doc 2 (test) carries label 99 that no training doc has; it must not
    # become a feature.
    docs = [(np.array([1, 2], np.int64), np.array([1, 1], np.int32)),
            (np.array([1, 2], np.int64), np.array([2, 1], np.int32)),
            (np.array([1, 99], np.int64), np.array([1, 5], np.int32))]
    train = np.array([True, True, False])
    X, vocab, df = E.build_counts(docs, train, min_df=2)
    assert set(vocab.tolist()) == {1, 2}
    assert X.shape == (3, 2)
    assert X[2].sum() == 1.0          # only label 1 survives for the test doc


def test_min_df_prunes():
    docs = [(np.array([1], np.int64), np.array([1], np.int32)),
            (np.array([1, 7], np.int64), np.array([1, 1], np.int32))]
    train = np.array([True, True])
    _, vocab, _ = E.build_counts(docs, train, min_df=2)
    assert vocab.tolist() == [1]


def test_pvdbow_separates_two_clusters():
    from scipy import sparse
    rng = np.random.default_rng(0)
    # 40 docs: half use labels 0-9, half use labels 10-19
    rows, cols, vals = [], [], []
    for i in range(40):
        base = 0 if i < 20 else 10
        for c in range(base, base + 10):
            rows.append(i); cols.append(c); vals.append(3.0)
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(40, 20))
    pv = E.PVDBOW(dim=8, epochs=40, batch=256, seed=1, max_steps=400,
                  max_seconds=60)
    pv.fit(X)
    Z = E.l2norm(pv.Wd_train)
    within = float(np.mean(Z[:20] @ Z[:20].T))
    between = float(np.mean(Z[:20] @ Z[20:].T))
    assert within > between


# ---------------------------------------------------------------------------
# n-gram rules
# ---------------------------------------------------------------------------
def test_code_decode_roundtrip():
    inv = {1: "mov", 2: "push", 3: "call"}
    a = np.array([1, 2, 3], dtype=np.int64)
    codes = N._codes(a, 3)
    assert len(codes) == 1
    assert N.decode(int(codes[0]), 3, inv) == "mov push call"


def test_mining_recovers_a_planted_ngram():
    # ransomware docs all contain <7 8 9>; goodware docs never do.
    rng = np.random.default_rng(0)
    enc, y = [], []
    for i in range(60):
        seq = rng.integers(1, 6, 200)
        if i < 30:
            seq[50:53] = [7, 8, 9]
            y.append(1)
        else:
            y.append(0)
        enc.append(seq.astype(np.int64))
    y = np.array(y)
    idx = np.arange(60)
    mined = N.mine(enc, y, idx, min_support=10, max_n=3, verbose=False)
    codes, a, b = mined[3]
    planted = 7 * N.BASE ** 2 + 8 * N.BASE + 9
    j = int(np.searchsorted(codes, planted))
    assert codes[j] == planted
    assert a[j] == 30 and b[j] == 0
    sc = N.score_rules(a, b, 30, 30)
    assert sc["lift"][j] > 20              # strong, and finite thanks to alpha
    assert np.isfinite(sc["odds"][j])
    assert sc["recall_r"][j] == 1.0


def test_laplace_prior_keeps_lift_finite():
    a = np.array([5]); b = np.array([0])
    sc = N.score_rules(a, b, 100, 100)
    assert np.isfinite(sc["lift"][0]) and sc["lift"][0] > 1


def test_greedy_set_cover_prefers_coverage():
    # rule 0 hits positives 0-4, rule 1 hits 5-9, rule 2 duplicates rule 0.
    hits = np.zeros((12, 3), dtype=bool)
    hits[0:5, 0] = True
    hits[5:10, 1] = True
    hits[0:5, 2] = True
    y = np.array([1] * 10 + [0, 0])
    idx = np.arange(12)
    chosen, covered, total = N.greedy_set_cover(hits, y, idx,
                                               np.array([0, 2, 1]), 5, 0.8)
    assert chosen == [0, 1]                 # rule 2 adds nothing new
    assert covered == 10 and total == 10


def test_hit_matrix_matches_code_order():
    enc = [np.array([1, 2, 3], np.int64), np.array([4, 5], np.int64)]
    codes = {2: np.sort(np.array([1 * N.BASE + 2, 4 * N.BASE + 5],
                                 dtype=np.int64))}
    H = N.hit_matrix(enc, codes)
    assert H.shape == (2, 2)
    assert H[0].sum() == 1 and H[1].sum() == 1
    assert not (H[0] & H[1]).any()


# -- regression: the int16 mnemonic stream must produce int64 n-gram codes ---
# rules_pipeline/train_eval.MnemCorpus stores each file as int16 to fit the
# corpus in memory. `id * 2048` overflows int16 for any id >= 16, so before the
# cast in _codes() every rule whose leading mnemonic was not one of the first
# 15 distinct mnemonics ever seen -- and every 3- and 4-gram without exception
# -- was mined correctly but scored as firing on nothing. That is what made the
# whole "top by lift" table read 0 test hits.
@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64])
def test_codes_are_dtype_independent(dtype):
    a = np.array([210, 284, 17, 900, 3], dtype=dtype)      # ids well over 15
    ref = N._codes(np.array([210, 284, 17, 900, 3], dtype=np.int64), 3)
    got = N._codes(a, 3)
    assert got.dtype == np.int64
    assert got.tolist() == ref.tolist()
    assert (got > 0).all()                                  # no wraparound


def test_hit_matrix_fires_for_high_id_mnemonics():
    # `xorps movlpd` == ids (210, 284): the real rule that read 0 test hits.
    enc16 = [np.array([210, 284, 7], np.int16),
             np.array([6, 69, 6], np.int16),        # mov rol mov: low ids
             np.array([1, 2, 3], np.int16)]
    codes = {2: np.sort(np.array([210 * N.BASE + 284, 6 * N.BASE + 69],
                                 dtype=np.int64))}
    H = N.hit_matrix(enc16, codes)
    assert H[0].sum() == 1 and H[1].sum() == 1 and H[2].sum() == 0
    # and the int16 corpus must agree with an int64 one element for element
    enc64 = [a.astype(np.int64) for a in enc16]
    assert (H == N.hit_matrix(enc64, codes)).all()


def test_hit_matrix_agrees_with_mining_on_int16_corpus():
    """End to end: mine on int16 streams, then score them. The per-class hit
    counts from the hit matrix must equal the document frequencies the miner
    counted, for every surviving n-gram at every level."""
    rng = np.random.default_rng(7)
    enc, y = [], []
    for i in range(60):
        seq = rng.integers(1, 12, 300)
        if i < 30:
            seq[100:104] = [300, 401, 502, 603]   # all ids > 15, a 4-gram
            y.append(1)
        else:
            y.append(0)
        enc.append(seq.astype(np.int16))          # the corpus dtype
    y = np.array(y)
    idx = np.arange(60)
    mined = N.mine(enc, y, idx, min_support=10, max_n=4, verbose=False)
    codes, a, b = mined[4]
    planted = ((300 * N.BASE + 401) * N.BASE + 502) * N.BASE + 603
    j = int(np.searchsorted(codes, planted))
    assert codes[j] == planted and a[j] == 30 and b[j] == 0
    H = N.hit_matrix(enc, {4: codes})
    assert H[:, j].sum() == 30
    assert H[:30, j].all() and not H[30:, j].any()
    for k, (ck, ak, bk) in mined.items():
        Hk = N.hit_matrix(enc, {k: ck})
        assert (Hk[y == 1].sum(axis=0) == ak).all(), k
        assert (Hk[y == 0].sum(axis=0) == bk).all(), k


# ---------------------------------------------------------------------------
# behaviour rules
# ---------------------------------------------------------------------------
def test_behaviour_detects_aesni_and_constants(tmp_path):
    p = write_asm(tmp_path / "crypto.asm", [
        "; section .text va=0x401000 size=64",
        (0x1000, "aesenc", "xmm0, xmm1"),
        (0x1001, "pclmulqdq", "xmm0, xmm1, 0"),
        (0x1002, "mov", "eax, 0x67452301"),
        (0x1003, "rdtsc", ""),
        (0x1004, "cpuid", ""),
        (0x1005, "mov", "eax, dword ptr fs:[0x30]"),
        (0x1006, "ret", ""),
    ])
    f = B.extract(p)
    assert f["aesni"] == 1 and f["pclmulqdq"] == 1
    assert f["crypto_const"] == 1
    assert f["rdtsc"] == 1 and f["cpuid"] == 1 and f["peb_teb"] == 1


def test_behaviour_detects_tight_crypto_loop(tmp_path):
    lines = ["; section .text va=0x401000 size=128", (0x1000, "mov", "ecx, 16")]
    for i in range(6):
        lines.append((0x1001 + i, "xor", "eax, ebx"))
    lines.append((0x1007, "jnz", "0x1001"))
    p = write_asm(tmp_path / "loop.asm", lines)
    f = B.extract(p)
    assert f["tight_crypto_loop_n"] == 1
    assert f["backedge_n"] == 1


def test_behaviour_no_false_crypto_on_plain_code(tmp_path):
    p = write_asm(tmp_path / "plain.asm",
                  ["; section .text va=0x401000 size=64"] +
                  [(0x1000 + i, "mov", "eax, ebx") for i in range(20)] +
                  [(0x1020, "ret", "")])
    f = B.extract(p)
    for k in ("aesni", "pclmulqdq", "sha_ni", "crypto_const", "cpuid",
              "tight_crypto_loop_n"):
        assert f[k] == 0


def test_thresholds_are_fitted_on_train_rows_only():
    feats = [{"n_insns": 1000, "xor_n": v, "call_n": 0, "indirect_call_n": 0,
              "rotshift_n": 0, "tight_crypto_loop_n": 0, "arx_window_n": 0,
              "rep_string_n": 0, "scan_string_n": 0, "int3_n": 0,
              "simd_n": 0, "backedge_n": 0}
             for v in [10, 12, 90, 95, 10000, 0]]
    y = np.array([0, 0, 1, 1, 1, 0])
    train = np.array([0, 1, 2, 3])          # row 4 (xor_n=10000) is test
    rules = [r for r in B.RULES if r.name == "xor_dense"]
    B.fit_thresholds(rules, feats, y, train)
    t = rules[0].threshold
    assert t is not None
    # the threshold can only come from the training values' quantiles
    assert t <= 95.0


def test_rule_hits_shape():
    feats = [{k: 0 for k in B.FEATURES} for _ in range(3)]
    for f in feats:
        f["n_insns"] = 100
    feats[0]["aesni"] = 1
    for r in B.RULES:
        if r.kind == "density":
            r.threshold = 1e9
    H = B.rule_hits(B.RULES, feats)
    assert H.shape == (3, len(B.RULES))
    assert H[0, 0] and not H[1, 0]


# ---------------------------------------------------------------------------
# end to end on synthetic data
# ---------------------------------------------------------------------------
def test_cfg_to_embedding_end_to_end(tmp_path):
    docs = []
    for i in range(12):
        n = 6 + (i % 3)
        lines = ["; section .text va=0x401000 size=256"]
        for j in range(n):
            lines.append((0x1000 + 2 * j, "cmp", "eax, ebx"))
            lines.append((0x1001 + 2 * j, "je", f"0x{0x1000 + 2*((j+2) % n):x}"))
        p = write_asm(tmp_path / f"s{i}.asm", lines)
        docs.append(C.wl_labels(C.cfg_from_file(p), 2))
    arrs = [(np.array(sorted(d), dtype=np.int64),
             np.array([d[k] for k in sorted(d)], dtype=np.int32)) for d in docs]
    train = np.array([True] * 8 + [False] * 4)
    X, vocab, _ = E.build_counts(arrs, train, min_df=2)
    assert X.shape[0] == 12 and X.shape[1] > 0
    Xt, _ = E.tfidf_fit_transform(X, train)
    Z, _ = E.svd_fit_transform(Xt, train, dim=4, seed=0)
    assert Z.shape == (12, min(4, X.shape[1], 7))
    assert np.isfinite(Z).all()
