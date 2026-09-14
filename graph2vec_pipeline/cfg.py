#!/usr/bin/env python3
"""Control-flow graphs from the linear-sweep disassembly in Shared/Extract.

Input is one `.asm` file per sample, produced by the host extractor. Three
line shapes occur:

    ; section .text va=0x401000 size=11844
    0x00401000:  mov<TAB>al, byte ptr [0xf0004048]
    0x00403de8:  .skip<TAB>3 bytes

`.skip` marks bytes capstone could not decode. It is a hard basic-block
boundary: we do not know what control flow does across an undecoded gap, so no
fall-through edge is emitted over one.

What this is and is not
-----------------------
This is a *linear sweep* CFG, not a recursive-descent one. There is no function
recovery, so:

  * Blocks are cut out of the raw instruction stream, not out of functions.
    A "block" here is a maximal run of instructions between two leaders.
  * Only direct branch/call targets printed as a bare `0x...` immediate can be
    resolved. Indirect control flow (`call dword ptr [0x405128]`, `jmp eax`,
    vtable dispatch, SEH) contributes no edge. On this corpus 25-40% of call
    sites are indirect, so the graph is genuinely partial.
  * Data interleaved with code is decoded as code, which manufactures blocks.
    The `add byte ptr [eax], al` runs that open many of these files are zero
    padding, not instructions.

All of that is a property of the input, not a bug here; it is reported in
results/graph2vec/summary.md rather than hidden.

Caps
----
Reading stops at `max_insns` instruction lines (default 80_000) and the graph
keeps at most `max_blocks` blocks (default 5_000), whichever binds first.
Rationale: files run to 3.16M instructions (manifest.csv), the corpus is 7.2GB
of text, and the existing tokenization pipeline already truncates at 5_000
*instructions*, so a 5_000-*block* window (~30k instructions) is a strictly
wider view than the baseline it is being compared against. Truncation is
recorded per file (`truncated`, `insns_read`, `insns_total_seen`).
"""
from __future__ import annotations

import hashlib
import io
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Mnemonic -> coarse class. The class alphabet is deliberately small: node
# labels feed a Weisfeiler-Lehman relabelling, which explodes the alphabet
# again on its own. A large base alphabet just makes every h>=1 label unique.
# ---------------------------------------------------------------------------
_CLASS_TABLE = {
    "MOV": ("mov movzx movsx movsxd movabs lea xchg movbe cmpxchg8b "
            "cmpxchg16b").split(),
    "STACK": "push pop pusha pushal pushad popa popal popad pushf pushfd pushfq popf popfd popfq enter leave".split(),
    "ARITH": ("add sub adc sbb inc dec neg mul imul div idiv cdq cdqe cwd cwde "
              "cbw xadd das daa aaa aas aam aad").split(),
    "LOGIC": "and or xor not test".split(),
    "SHIFT": "shl shr sar sal rol ror rcl rcr shld shrd bt bts btr btc bswap bsf bsr".split(),
    "CMP": "cmp cmpxchg".split(),
    "JMP": ["jmp"],
    "CALL": ["call"],
    "RET": "ret retn retf iret iretd iretq".split(),
    "STRING": ("movsb movsw movsq stosb stosw stosd stosq lodsb lodsw lodsd "
               "lodsq scasb scasw scasd scasq cmpsb cmpsw cmpsd cmpsq insb "
               "insw insd outsb outsw outsd").split(),
    "CRYPTO": ("aesenc aesenclast aesdec aesdeclast aesimc aeskeygenassist "
               "pclmulqdq pclmullqlqdq crc32 sha1rnds4 sha1nexte sha1msg1 "
               "sha1msg2 sha256rnds2 sha256msg1 sha256msg2 rdrand rdseed").split(),
    "SYS": ("int int3 into syscall sysenter sysexit sysret cpuid rdtsc rdtscp "
            "rdmsr wrmsr in out hlt ud0 ud1 ud2 cli sti cld std clc stc cmc "
            "lgdt lidt lldt ltr sgdt sidt sldt str verr verw invd wbinvd "
            "xgetbv xsetbv arpl bound lar lsl clts").split(),
    "NOP": "nop pause fnop".split(),
}
MNEM_CLASS: dict[str, str] = {}
for _cls, _ms in _CLASS_TABLE.items():
    for _m in _ms:
        MNEM_CLASS[_m] = _cls

CLASSES = ("MOV", "STACK", "ARITH", "LOGIC", "SHIFT", "CMP", "COND", "JMP",
           "CALL", "RET", "STRING", "CRYPTO", "SYS", "NOP", "FLOAT", "SIMD",
           "SETCC", "CMOV", "OTHER")

_SIMD_PREFIXES = ("mov a", )  # unused; kept for readability of the rule below


def mnem_class(base: str, ops: str = "") -> str:
    """Coarse class of one mnemonic. `ops` disambiguates movsd/movss only."""
    c = MNEM_CLASS.get(base)
    if c is not None:
        return c
    if base.startswith("j"):          # jcc, jecxz, jrcxz (jmp caught above)
        return "COND"
    if base.startswith("loop"):
        return "COND"
    if base.startswith("set"):
        return "SETCC"
    if base.startswith("cmov"):
        return "CMOV"
    if base in ("movsd", "movss"):
        # SSE scalar move vs. string move: only the operands tell them apart.
        return "SIMD" if "xmm" in ops else "STRING"
    if base.startswith("f") and len(base) > 1:
        return "FLOAT"
    if base.startswith("v") or base.startswith("p") or base.startswith("xmm"):
        return "SIMD"
    if base.startswith(("movap", "movup", "movdq", "movnt", "unpck", "shuf",
                        "addp", "subp", "mulp", "divp", "xorp", "andp", "orp",
                        "cvt", "sqrt", "maxp", "minp", "rcpp", "rsqrt")):
        return "SIMD"
    return "OTHER"


# Control-flow behaviour of the *last* instruction in a block.
_TERM_STOP = frozenset("ret retn retf iret iretd iretq hlt ud0 ud1 ud2".split())
_TERM_CALL = frozenset(["call"])
_TERM_JMP = frozenset(["jmp"])


def term_kind(base: str) -> str | None:
    """'jmp' | 'cond' | 'call' | 'stop' | None (falls through)."""
    if base in _TERM_JMP:
        return "jmp"
    if base in _TERM_CALL:
        return "call"
    if base in _TERM_STOP:
        return "stop"
    if base.startswith("j") or base.startswith("loop"):
        return "cond"
    return None


def _size_bucket(n: int) -> str:
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    if n <= 4:
        return "4"
    if n <= 8:
        return "8"
    if n <= 16:
        return "16"
    return "N"


def _stable_hash(s: str) -> int:
    """Deterministic across processes, unlike hash(str)."""
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(),
                          "little", signed=True)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
class ParsedAsm:
    """Flat instruction arrays for one file, section boundaries kept.

    `mem_target` holds the *data* address an indirect branch reads its target
    from -- `call dword ptr [0x405128]` (x86 import thunk) and
    `call qword ptr [rip + 0x1234]` (x64 import thunk). It is -1 elsewhere.
    The rip-relative form is resolved against the address of the *following*
    instruction, which the linear sweep gives us for free, so x64 thunks are
    recoverable even though the operand is relative.
    """

    __slots__ = ("addr", "base", "ops", "sec", "gap_before", "n_sections",
                 "insns_read", "truncated", "target_imm", "mem_target",
                 "sec_names", "n_windows")

    def __init__(self):
        self.addr: list[int] = []
        self.base: list[str] = []
        self.ops: list[str] = []
        self.sec: list[int] = []
        self.gap_before: list[bool] = []
        self.target_imm: list[int] = []   # -1 when no direct immediate target
        self.mem_target: list[int] = []   # -1 when not an indirect [mem] branch
        self.sec_names: list[str] = []
        self.n_sections = 0
        self.insns_read = 0
        self.truncated = False
        self.n_windows = 1


def _mem_operand(ops: str) -> tuple[int, bool]:
    """(address, is_rip_relative) for `... ptr [0x..]` / `[rip +- 0x..]`.

    Returns (-1, False) for register-indirect, SIB and anything else: those
    cannot be resolved from a linear sweep and are left unresolved on purpose.
    """
    if not ops.endswith("]"):
        return -1, False
    lb = ops.rfind("[")
    if lb < 0:
        return -1, False
    inner = ops[lb + 1:-1]
    if inner.startswith("0x"):
        try:
            return int(inner, 16), False
        except ValueError:
            return -1, False
    if inner.startswith("rip"):
        rest = inner[3:].strip()
        if not rest:
            return 0, True
        sign = 1
        if rest[0] == "+":
            rest = rest[1:].strip()
        elif rest[0] == "-":
            sign, rest = -1, rest[1:].strip()
        else:
            return -1, False
        try:
            return sign * int(rest, 16), True
        except ValueError:
            return -1, False
    return -1, False


def _consume(fh, p, max_insns: int, sec_start: int, window: int) -> int:
    """Read instruction lines from an open handle into `p`. Returns count."""
    sec = sec_start
    pending_gap = window > 0       # a seek boundary is an undecoded gap
    addr_l, base_l, ops_l, sec_l, gap_l, tgt_l, mem_l = (
        p.addr, p.base, p.ops, p.sec, p.gap_before, p.target_imm, p.mem_target)
    rip_fix: list[int] = []
    n = 0
    for line in fh:
        if not line:
            continue
        c0 = line[0]
        if c0 == ";":
            sec += 1
            tail = line.split("section", 1)[-1].strip() if "section" in line else ""
            p.sec_names.append(tail.split(" ", 1)[0] if tail else "")
            pending_gap = False
            continue
        if c0 != "0":
            continue
        cut = line.find(":  ")
        if cut < 0:
            continue
        try:
            a = int(line[:cut], 16)
        except ValueError:
            continue
        rest = line[cut + 3:].rstrip("\n")
        t = rest.find("\t")
        if t < 0:
            mnem, ops = rest, ""
        else:
            mnem, ops = rest[:t], rest[t + 1:]
        if mnem == ".skip":
            pending_gap = True
            continue
        sp = mnem.rfind(" ")          # "rep movsd" -> base "movsd"
        base = mnem[sp + 1:] if sp >= 0 else mnem
        addr_l.append(a)
        base_l.append(base)
        ops_l.append(ops)
        sec_l.append(sec)
        gap_l.append(pending_gap)
        pending_gap = False
        # direct target immediate: the whole operand is one 0x... literal
        if ops[:2] == "0x" and " " not in ops and "," not in ops:
            try:
                tgt_l.append(int(ops, 16))
            except ValueError:
                tgt_l.append(-1)
        else:
            tgt_l.append(-1)
        # indirect branch through an absolute / rip-relative data slot
        if tgt_l[-1] < 0 and (base == "call" or base[0] == "j"):
            v, is_rip = _mem_operand(ops)
            if is_rip:
                rip_fix.append(len(mem_l))
            mem_l.append(v)
        else:
            mem_l.append(-1)
        n += 1
        if n >= max_insns:
            break
    # rip-relative displacements are relative to the NEXT instruction
    for i in rip_fix:
        mem_l[i] = (addr_l[i + 1] + mem_l[i]) if i + 1 < len(addr_l) else -1
    p.n_sections = max(p.n_sections, sec + 1)
    return n


def parse_asm(path: str | Path, max_insns: int = 80_000,
              windows: int = 1) -> ParsedAsm:
    """Parse a `.asm` dump into flat arrays.

    windows=1 (default) reads the first `max_insns` instruction lines.
    windows=k>1 reads `max_insns//k` instructions from each of k byte-offset
    windows spread evenly over the file, so a truncated view covers the whole
    binary instead of only its head. Each window is opened as a fresh section
    (no fall-through across a seek boundary, exactly as for a `.skip` gap),
    because the bytes in between were never decoded.
    """
    p = ParsedAsm()
    path = Path(path)
    if windows <= 1:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            n = _consume(fh, p, max_insns, -1, 0)
        p.insns_read = n
        p.truncated = n >= max_insns
        p.n_windows = 1
        return p

    size = path.stat().st_size
    per = max(1, max_insns // windows)
    total = 0
    truncated = False
    for w in range(windows):
        off = (size * w) // windows
        # binary handle, then a text wrapper: text-mode seek only accepts
        # cookies returned by tell(), and we want raw byte offsets.
        raw = open(path, "rb")
        try:
            if w:
                raw.seek(off)
                raw.readline()          # drop the partial line
            fh = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
            # each window gets a fresh section id, so nothing joins across a
            # seek boundary even when the window carries no `; section` line
            got = _consume(fh, p, per, (p.n_sections if w else -1), w)
            fh.detach()
        finally:
            raw.close()
        total += got
        truncated = truncated or got >= per
    p.insns_read = total
    p.truncated = truncated
    p.n_windows = windows
    return p


# ---------------------------------------------------------------------------
# CFG
# ---------------------------------------------------------------------------
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}
EXTERN_CLASS_ID = len(CLASSES)                    # synthetic import-thunk node
N_CLASS_IDS = len(CLASSES) + 1

TERM_IDX = {"fall": 0, "jmp": 1, "cond": 2, "call": 3, "stop": 4, "extern": 5}
SZB_IDX = {"1": 0, "2": 1, "4": 2, "8": 3, "16": 4, "N": 5}

# edge kinds. The fall-through / branch distinction is kept as an edge label
# (task 1); the *_XSEC variants record that the target lived in a different
# section, so a run can drop them again and get the old same-section-only rule.
E_FALL, E_BRANCH, E_BRANCH_XSEC, E_CALL, E_CALL_XSEC, E_EXTERN = range(6)
EDGE_NAMES = ("fall", "branch", "branch_xsec", "call", "call_xsec", "extern")

_SEQ_CAP = 8            # instructions of a block that enter its sequence hash


class CFG:
    __slots__ = ("labels", "edges", "stats", "etypes", "attrs", "n_blocks")

    def __init__(self, labels, edges, stats, etypes=None, attrs=None,
                 n_blocks=None):
        self.labels: list[str] = labels
        self.edges: list[tuple[int, int]] = edges
        self.stats: dict = stats
        # parallel to `edges`; one of the E_* codes
        self.etypes: list[int] = [] if etypes is None else etypes
        # per-node integer attributes, for the label-granularity experiments
        self.attrs: dict = {} if attrs is None else attrs
        # nodes [0, n_blocks) are basic blocks; [n_blocks, len(labels)) are
        # synthetic external (import-thunk) nodes
        self.n_blocks: int = len(labels) if n_blocks is None else n_blocks


def build_cfg(p: ParsedAsm, max_blocks: int = 5_000,
              include_calls: bool = True, cross_section: bool = False,
              extern_nodes: bool = False, max_externs: int = 2_000) -> CFG:
    """Basic blocks + fall-through/branch(/call) edges from a ParsedAsm.

    Options (all default to the original behaviour):

    cross_section   resolve a direct branch/call whose target lands in a
                    different section of the dump. The extractor only dumps
                    executable sections, so such a target is code; the old rule
                    threw it away and counted it unresolved.
    extern_nodes    give every distinct import thunk an `call dword ptr [0x..]`
                    or `call qword ptr [rip+0x..]` reads through) one synthetic
                    node, and an E_EXTERN edge from each call site. Call sites
                    that share an import then share a node, which is the only
                    call-graph structure a linear sweep can recover.
    """
    n = p.insns_read
    if n == 0:
        return CFG([], [], dict(n_blocks=0, n_edges=0, insns_used=0,
                                n_sections=p.n_sections, truncated=p.truncated,
                                block_capped=False, resolved_targets=0,
                                unresolved_targets=0, back_edges=0,
                                self_loops=0, call_edges=0, mean_block_len=0.0,
                                extern_nodes=0, extern_edges=0,
                                xsec_targets=0, n_components=0),
                   [], _empty_attrs(), 0)

    addr, base, ops, sec, gap, tgt = (p.addr, p.base, p.ops, p.sec,
                                      p.gap_before, p.target_imm)
    mem = p.mem_target
    addr2idx = {a: i for i, a in enumerate(addr)}

    kinds: list[str | None] = [term_kind(b) for b in base]

    # ---- leaders ----------------------------------------------------------
    leaders = {0}
    resolved = unresolved = xsec = 0
    tgt_idx: list[int] = [-1] * n
    tgt_xsec: list[bool] = [False] * n
    ext_of: dict[int, int] = {}          # data address -> extern node number
    ext_site: list[int] = [-1] * n       # instruction -> extern node number
    for i in range(n):
        k = kinds[i]
        if i and (sec[i] != sec[i - 1] or gap[i]):
            leaders.add(i)
        if k is None:
            continue
        if i + 1 < n:
            leaders.add(i + 1)           # fall-through leader after any term
        if k in ("jmp", "cond", "call"):
            t = tgt[i]
            if t >= 0:
                j = addr2idx.get(t)
                if j is None:
                    unresolved += 1
                elif sec[j] == sec[i]:
                    leaders.add(j)
                    tgt_idx[i] = j
                    resolved += 1
                elif cross_section:
                    leaders.add(j)
                    tgt_idx[i] = j
                    tgt_xsec[i] = True
                    resolved += 1
                    xsec += 1
                else:
                    unresolved += 1
            else:
                m = mem[i]
                if extern_nodes and m >= 0:
                    e = ext_of.get(m)
                    if e is None:
                        if len(ext_of) < max_externs:
                            e = len(ext_of)
                            ext_of[m] = e
                        else:
                            e = -1
                    if e >= 0:
                        ext_site[i] = e
                        resolved += 1
                    else:
                        unresolved += 1
                else:
                    unresolved += 1

    lead = sorted(leaders)
    if len(lead) > max_blocks:
        lead = lead[:max_blocks]
        capped = True
    else:
        capped = False
    blk_of = {}
    for b, i in enumerate(lead):
        blk_of[i] = b
    nb = len(lead)
    ends = [lead[b + 1] - 1 if b + 1 < nb else n - 1 for b in range(nb)]
    if capped:
        ends[-1] = min(ends[-1], lead[-1] + 64)   # do not absorb the tail

    # ---- node labels ------------------------------------------------------
    labels: list[str] = []
    a_dom: list[int] = []
    a_term: list[int] = []
    a_szb: list[int] = []
    a_flg: list[int] = []
    a_blen: list[int] = []
    a_seq: list[int] = []
    for b in range(nb):
        s, e = lead[b], ends[b]
        cnt: Counter = Counter()
        has_c = has_s = has_v = False
        h = 2166136261                       # FNV-1a over the class sequence
        for i in range(s, e + 1):
            c = mnem_class(base[i], ops[i])
            cnt[c] += 1
            if c == "CRYPTO":
                has_c = True
            elif c == "STRING":
                has_s = True
            elif c == "SIMD":
                has_v = True
            if i - s < _SEQ_CAP:
                h = ((h ^ CLASS_IDX[c]) * 16777619) & 0xFFFFFFFF
        dom = max(sorted(cnt), key=lambda c: cnt[c])
        tk = kinds[e] or "fall"
        szb = _size_bucket(e - s + 1)
        lab = f"{dom}.{tk}.{szb}"
        if has_c:
            lab += "+C"
        if has_s:
            lab += "+S"
        if has_v:
            lab += "+V"
        labels.append(lab)
        a_dom.append(CLASS_IDX[dom])
        a_term.append(TERM_IDX[tk])
        a_szb.append(SZB_IDX[szb])
        a_flg.append((1 if has_c else 0) | (2 if has_s else 0)
                     | (4 if has_v else 0))
        a_blen.append(min(e - s + 1, 65535))
        a_seq.append(h)

    # ---- edges ------------------------------------------------------------
    edges: dict[tuple[int, int], int] = {}
    back = self_loops = call_edges = extern_edges = 0

    def _add(u: int, v: int, et: int):
        nonlocal back, self_loops
        if v == u:
            self_loops += 1
        elif v < u:
            back += 1
        # a duplicated pair keeps the first (lowest-numbered) edge kind
        edges.setdefault((u, v), et)

    used_ext: dict[int, int] = {}            # extern number -> node index
    for b in range(nb):
        e = ends[b]
        k = kinds[e]
        nxt = e + 1
        fall_ok = (nxt < n and nxt in blk_of and sec[nxt] == sec[e]
                   and not gap[nxt])
        succ: list[tuple[int, int]] = []     # (instruction index, edge kind)
        if k == "stop":
            pass
        elif k == "jmp":
            if tgt_idx[e] >= 0:
                succ.append((tgt_idx[e],
                             E_BRANCH_XSEC if tgt_xsec[e] else E_BRANCH))
        elif k == "cond":
            if tgt_idx[e] >= 0:
                succ.append((tgt_idx[e],
                             E_BRANCH_XSEC if tgt_xsec[e] else E_BRANCH))
            if fall_ok:
                succ.append((nxt, E_FALL))
        elif k == "call":
            if fall_ok:
                succ.append((nxt, E_FALL))
            if include_calls and tgt_idx[e] >= 0:
                succ.append((tgt_idx[e],
                             E_CALL_XSEC if tgt_xsec[e] else E_CALL))
                call_edges += 1
        elif fall_ok:
            succ.append((nxt, E_FALL))
        for s_i, et in succ:
            tb = blk_of.get(s_i)
            if tb is None:
                continue
            _add(b, tb, et)
        if include_calls and extern_nodes:
            # every extern-reading branch anywhere inside the block, not only
            # its terminator: `call [0x..]` in the middle of a block is the
            # common case, since a call does not end a block here.
            for i in range(lead[b], ends[b] + 1):
                en = ext_site[i]
                if en < 0:
                    continue
                node = used_ext.get(en)
                if node is None:
                    node = nb + len(used_ext)
                    used_ext[en] = node
                if (b, node) not in edges:
                    extern_edges += 1
                edges.setdefault((b, node), E_EXTERN)

    for _ in range(len(used_ext)):
        labels.append("EXTERN")
        a_dom.append(EXTERN_CLASS_ID)
        a_term.append(TERM_IDX["extern"])
        a_szb.append(0)
        a_flg.append(0)
        a_blen.append(0)
        a_seq.append(0)

    mean_len = sum(ends[b] - lead[b] + 1 for b in range(nb)) / nb if nb else 0.0
    stats = dict(
        n_blocks=nb,
        n_edges=len(edges),
        insns_used=(ends[-1] + 1) if nb else 0,
        n_sections=p.n_sections,
        truncated=bool(p.truncated),
        block_capped=bool(capped),
        resolved_targets=resolved,
        unresolved_targets=unresolved,
        back_edges=back,
        self_loops=self_loops,
        call_edges=call_edges,
        extern_nodes=len(used_ext),
        extern_edges=extern_edges,
        xsec_targets=xsec,
        mean_block_len=round(mean_len, 3),
    )
    ekeys = list(edges)
    attrs = dict(dom=a_dom, term=a_term, szb=a_szb, flg=a_flg, blen=a_blen,
                 seq=a_seq)
    return CFG(labels, ekeys, stats, [edges[k] for k in ekeys], attrs, nb)


def _empty_attrs() -> dict:
    return dict(dom=[], term=[], szb=[], flg=[], blen=[], seq=[])


def cfg_from_file(path, max_insns: int = 80_000, max_blocks: int = 5_000,
                  include_calls: bool = True, windows: int = 1,
                  cross_section: bool = False, extern_nodes: bool = False,
                  max_externs: int = 2_000) -> CFG:
    return build_cfg(parse_asm(path, max_insns, windows=windows), max_blocks,
                     include_calls, cross_section=cross_section,
                     extern_nodes=extern_nodes, max_externs=max_externs)


# ---------------------------------------------------------------------------
# Weisfeiler-Lehman relabelling -> the "document" graph2vec consumes
# ---------------------------------------------------------------------------
def wl_labels(cfg: CFG, iterations: int = 2) -> Counter:
    """Multiset of WL subtree labels at depths 0..`iterations`.

    Directed WL: a node's context is its sorted successor labels and its
    sorted predecessor labels, kept apart. Labels are int64; depth-0 labels
    come from a stable string hash, deeper ones from Python's tuple hash,
    which is deterministic for integer contents (unlike str/bytes hashing).
    """
    nb = len(cfg.labels)
    doc: Counter = Counter()
    if nb == 0:
        return doc
    cur = [_stable_hash(l) for l in cfg.labels]
    doc.update(cur)
    if iterations <= 0:
        return doc
    succ: list[list[int]] = [[] for _ in range(nb)]
    pred: list[list[int]] = [[] for _ in range(nb)]
    for u, v in cfg.edges:
        succ[u].append(v)
        pred[v].append(u)
    for _ in range(iterations):
        nxt = [0] * nb
        for v in range(nb):
            nxt[v] = hash((cur[v],
                           tuple(sorted(cur[u] for u in succ[v])),
                           tuple(sorted(cur[u] for u in pred[v]))))
        cur = nxt
        doc.update(cur)
    return doc


def label_alphabet_size() -> int:
    return len(CLASSES) * 6 * 6 * 8
