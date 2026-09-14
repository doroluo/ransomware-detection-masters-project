#!/usr/bin/env python3
"""Hand-written crypto-loop / enumeration / anti-analysis signatures.

Each rule is a predicate over the instruction stream of one `.asm` file --
either "this instruction or constant occurs at all" or "this density exceeds a
threshold". Presence rules have no free parameter. Density rules get ONE
threshold, chosen on the TRAIN split by maximising that single rule's F1 over
the deciles of the training values; nothing is tuned on test.

The window is the first `max_insns` (default 80_000) decoded instructions, the
same window the graph2vec prototype sees, so the two prototypes are not reading
different amounts of each file.

These rules are deliberately the kind of thing a YARA/heuristic engine would
carry. The interesting result is not that they fire on ransomware, it is which
of them also fire on the Goodware_Balanced hard negatives (archivers,
encryption utilities, backup tools, hashing utilities) -- reported per rule in
results/rules/summary.md.
"""
from __future__ import annotations

import numpy as np

from graph2vec_pipeline.cfg import parse_asm, mnem_class

# ---------------------------------------------------------------------------
# Well-known cryptographic / hashing constants, as they appear in operand text.
# MD5+SHA-1 IV, SHA-1 round constants, SHA-256 IV and first K words, CRC-32
# polynomials (both bit orders), FNV-1, TEA/XXTEA delta, Blowfish/pi.
# ---------------------------------------------------------------------------
CRYPTO_CONSTS = frozenset("""
67452301 efcdab89 98badcfe 10325476 c3d2e1f0
5a827999 6ed9eba1 8f1bbcdc ca62c1d6
6a09e667 bb67ae85 3c6ef372 a54ff53a 510e527f 9b05688c 1f83d9ab 5be0cd19
428a2f98 71374491 b5c0fbcf e9b5dba5
edb88320 04c11db7 82f63b78 1edc6f41
01000193 811c9dc5 01000000013b
9e3779b9 61c88647 243f6a88 85a308d3
""".split())

_PEB_PAT = ("fs:[0x30]", "fs:[0x18]", "gs:[0x60]", "fs:[48]", "gs:[96]")

_AESNI = frozenset("aesenc aesenclast aesdec aesdeclast aesimc "
                   "aeskeygenassist".split())
_SHANI_PREFIX = ("sha1", "sha256")
_ROTSHIFT = frozenset("rol ror rcl rcr shl shr sar sal shld shrd".split())
_REPSTR = frozenset("movsb movsw movsd movsq stosb stosw stosd stosq".split())
_SCAN = frozenset("scasb scasw scasd scasq cmpsb cmpsw cmpsd cmpsq".split())

_CLS_ID = {c: i for i, c in enumerate(
    ["MOV", "STACK", "ARITH", "LOGIC", "SHIFT", "CMP", "COND", "JMP", "CALL",
     "RET", "STRING", "CRYPTO", "SYS", "NOP", "FLOAT", "SIMD", "SETCC",
     "CMOV", "OTHER"])}
_LOGIC_ID, _SHIFT_ID, _ARITH_ID, _SIMD_ID = (
    _CLS_ID["LOGIC"], _CLS_ID["SHIFT"], _CLS_ID["ARITH"], _CLS_ID["SIMD"])


def _find_crypto_const(ops: str) -> bool:
    i = ops.find("0x")
    while i >= 0:
        j = i + 2
        k = j
        while k < len(ops) and ops[k] in "0123456789abcdef":
            k += 1
        if k - j >= 8 and ops[j:k] in CRYPTO_CONSTS:
            return True
        i = ops.find("0x", k if k > i + 2 else i + 2)
    return False


# ---------------------------------------------------------------------------
FEATURES = [
    "n_insns", "aesni", "pclmulqdq", "sha_ni", "crc32", "rdrand",
    "crypto_const", "loop_instr", "cpuid", "rdtsc", "peb_teb", "syscall",
    "int_other", "xor_n", "rotshift_n", "tight_crypto_loop_n", "arx_window_n",
    "rep_string_n", "scan_string_n", "int3_n", "call_n", "indirect_call_n",
    "simd_n", "backedge_n", "branch_n",
]


def extract(path, max_insns: int = 80_000) -> dict:
    p = parse_asm(path, max_insns)
    n = p.insns_read
    f = {k: 0 for k in FEATURES}
    f["n_insns"] = n
    if n == 0:
        return f

    base, ops, addr, tgt = p.base, p.ops, p.addr, p.target_imm
    cls = np.empty(n, dtype=np.int8)
    cache: dict[str, int] = {}
    xor_n = rot_n = rep_n = scan_n = int3_n = call_n = ind_n = 0
    for i in range(n):
        b = base[i]
        c = cache.get(b, -1)
        if c < 0:
            c = _CLS_ID[mnem_class(b)]
            cache[b] = c
        if b == "movsd" or b == "movss":
            c = _SIMD_ID if "xmm" in ops[i] else _CLS_ID["STRING"]
        cls[i] = c
        o = ops[i]
        if b == "xor":
            xor_n += 1
        elif b in _ROTSHIFT:
            rot_n += 1
        elif b == "call":
            call_n += 1
            if not (o[:2] == "0x" and "," not in o and " " not in o):
                ind_n += 1
        elif b == "int3":
            int3_n += 1
        elif b in _REPSTR:
            rep_n += 1
        elif b in _SCAN:
            scan_n += 1
        elif b in _AESNI:
            f["aesni"] += 1
        elif b == "pclmulqdq":
            f["pclmulqdq"] += 1
        elif b.startswith(_SHANI_PREFIX):
            f["sha_ni"] += 1
        elif b == "crc32":
            f["crc32"] += 1
        elif b in ("rdrand", "rdseed"):
            f["rdrand"] += 1
        elif b.startswith("loop"):
            f["loop_instr"] += 1
        elif b == "cpuid":
            f["cpuid"] += 1
        elif b in ("rdtsc", "rdtscp"):
            f["rdtsc"] += 1
        elif b in ("syscall", "sysenter"):
            f["syscall"] += 1
        elif b == "int":
            f["int_other"] += 1
        if o and "0x" in o:
            if _find_crypto_const(o):
                f["crypto_const"] += 1
            if not f["peb_teb"]:
                for pat in _PEB_PAT:
                    if pat in o:
                        f["peb_teb"] += 1
                        break
        elif o and ("fs:" in o or "gs:" in o) and not f["peb_teb"]:
            for pat in _PEB_PAT:
                if pat in o:
                    f["peb_teb"] += 1
                    break

    f["xor_n"], f["rotshift_n"] = xor_n, rot_n
    f["rep_string_n"], f["scan_string_n"] = rep_n, scan_n
    f["int3_n"], f["call_n"], f["indirect_call_n"] = int3_n, call_n, ind_n
    f["simd_n"] = int((cls == _SIMD_ID).sum())

    # ---- windowed / control-flow patterns --------------------------------
    logic = (cls == _LOGIC_ID).astype(np.int32)
    shift = (cls == _SHIFT_ID).astype(np.int32)
    arith = (cls == _ARITH_ID).astype(np.int32)
    c_logic = np.concatenate([[0], np.cumsum(logic)])
    c_shift = np.concatenate([[0], np.cumsum(shift)])
    c_arith = np.concatenate([[0], np.cumsum(arith)])

    W = 16
    if n > W:
        lw = c_logic[W:] - c_logic[:-W]
        aw = c_arith[W:] - c_arith[:-W]
        sw = c_shift[W:] - c_shift[:-W]
        # ARX-shaped mixing: a 16-instruction window that is simultaneously
        # xor/and/or heavy and add/sub heavy, or logic+rotate heavy.
        f["arx_window_n"] = int((((lw >= 5) & (aw >= 5)) |
                                 ((lw >= 4) & (sw >= 4))).sum())

    # tight backward branches whose body is logic/shift heavy == crypto loop
    addr2idx = {a: i for i, a in enumerate(addr)}
    back = 0
    tight = 0
    branch = 0
    for i in range(n):
        b = base[i]
        if not (b[0] == "j" or b.startswith("loop")):
            continue
        branch += 1
        t = tgt[i]
        if t < 0:
            continue
        j = addr2idx.get(t)
        if j is None or j >= i:
            continue
        back += 1
        span = i - j
        if span <= 40:
            body_logic = int(c_logic[i + 1] - c_logic[j])
            body_shift = int(c_shift[i + 1] - c_shift[j])
            if body_logic + body_shift >= 3:
                tight += 1
    f["backedge_n"], f["tight_crypto_loop_n"], f["branch_n"] = back, tight, branch
    return f


# ---------------------------------------------------------------------------
# Rule definitions over the extracted feature dict
# ---------------------------------------------------------------------------
class Rule:
    def __init__(self, rid, name, kind, expr, description):
        self.rid, self.name, self.kind = rid, name, kind
        self.expr, self.description = expr, description
        self.threshold = None          # density rules only
        self.direction = "ransomware"  # set during fit

    def value(self, f: dict) -> float:
        return float(self.expr(f))

    def fires(self, f: dict) -> bool:
        v = self.value(f)
        if self.kind == "presence":
            return v > 0
        return v >= (self.threshold if self.threshold is not None else np.inf)


def _per1k(key):
    return lambda f: 1000.0 * f[key] / max(f["n_insns"], 1)


RULES: list[Rule] = [
    Rule("R01", "aesni_any", "presence", lambda f: f["aesni"],
         "AES-NI round instruction (aesenc/aesdec/aeskeygenassist) present"),
    Rule("R02", "pclmulqdq_any", "presence", lambda f: f["pclmulqdq"],
         "carry-less multiply present (AES-GCM / CRC / GF(2^n) math)"),
    Rule("R03", "sha_ni_any", "presence", lambda f: f["sha_ni"],
         "SHA extension instruction (sha1rnds4/sha256rnds2/...) present"),
    Rule("R04", "crc32_any", "presence", lambda f: f["crc32"],
         "SSE4.2 crc32 instruction present"),
    Rule("R05", "rdrand_any", "presence", lambda f: f["rdrand"],
         "hardware RNG (rdrand/rdseed) present -- key generation"),
    Rule("R06", "crypto_const_any", "presence", lambda f: f["crypto_const"],
         "MD5/SHA-1/SHA-256 IV, SHA-1 K, CRC-32 polynomial, FNV or TEA delta "
         "appears as an immediate"),
    Rule("R07", "loop_instr_any", "presence", lambda f: f["loop_instr"],
         "x86 `loop`/`loope`/`loopne` present (rare in modern compiler output)"),
    Rule("R08", "cpuid_any", "presence", lambda f: f["cpuid"],
         "cpuid present -- feature probe or VM/sandbox detection"),
    Rule("R09", "rdtsc_any", "presence", lambda f: f["rdtsc"],
         "rdtsc/rdtscp present -- timing-based anti-analysis or seeding"),
    Rule("R10", "peb_teb_any", "presence", lambda f: f["peb_teb"],
         "direct PEB/TEB access (fs:[0x30], fs:[0x18], gs:[0x60])"),
    Rule("R11", "syscall_any", "presence", lambda f: f["syscall"],
         "syscall/sysenter in user code -- ntdll bypass"),
    Rule("R12", "int_other_any", "presence", lambda f: f["int_other"],
         "software interrupt other than int3 (int 0x2d/0x2e anti-debug)"),
    Rule("R13", "xor_dense", "density", _per1k("xor_n"),
         "xor instructions per 1k decoded instructions"),
    Rule("R14", "rotshift_dense", "density", _per1k("rotshift_n"),
         "rol/ror/shl/shr/sar/shld/shrd per 1k -- bit-mixing chains"),
    Rule("R15", "tight_crypto_loop", "density", _per1k("tight_crypto_loop_n"),
         "backward branches spanning <=40 instructions over a logic/shift "
         "heavy body, per 1k -- the crypto inner-loop shape"),
    Rule("R16", "arx_window", "density", _per1k("arx_window_n"),
         "16-instruction windows that are both logic-heavy and arith- or "
         "rotate-heavy (ARX round shape), per 1k"),
    Rule("R17", "rep_string_dense", "density", _per1k("rep_string_n"),
         "rep movs/stos per 1k -- bulk buffer copy/fill"),
    Rule("R18", "scan_string_dense", "density", _per1k("scan_string_n"),
         "scas/cmps per 1k -- string scanning, path/extension enumeration"),
    Rule("R19", "int3_dense", "density", _per1k("int3_n"),
         "int3 per 1k -- alignment padding density, a linker/packer artefact"),
    Rule("R20", "call_dense", "density", _per1k("call_n"),
         "call instructions per 1k"),
    Rule("R21", "indirect_call_ratio", "density",
         lambda f: f["indirect_call_n"] / max(f["call_n"], 1),
         "fraction of calls with a non-immediate target (IAT/vtable/dynamic)"),
    Rule("R22", "simd_dense", "density", _per1k("simd_n"),
         "SSE/AVX instructions per 1k"),
    Rule("R23", "backedge_dense", "density", _per1k("backedge_n"),
         "resolved backward branches per 1k -- loop density"),
]


def fit_thresholds(rules, feats: list[dict], y: np.ndarray,
                   train_idx: np.ndarray):
    """One threshold per density rule, maximising that rule's own train F1."""
    n_r = int((y[train_idx] == 1).sum())
    for r in rules:
        if r.kind != "density":
            continue
        vals = np.array([r.value(feats[i]) for i in train_idx])
        cands = np.unique(np.quantile(vals, np.linspace(0.05, 0.95, 19)))
        best, best_f1 = None, -1.0
        for t in cands:
            pred = vals >= t
            tp = int(((y[train_idx] == 1) & pred).sum())
            fp = int(((y[train_idx] == 0) & pred).sum())
            if tp == 0:
                continue
            prec, rec = tp / (tp + fp), tp / max(n_r, 1)
            f1 = 2 * prec * rec / (prec + rec)
            if f1 > best_f1:
                best, best_f1 = float(t), f1
        r.threshold = best if best is not None else float(vals.max() + 1)
    return rules


def rule_hits(rules, feats: list[dict]) -> np.ndarray:
    return np.array([[r.fires(f) for r in rules] for f in feats], dtype=bool)
