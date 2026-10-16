"""Bounded intra-procedural explicit-register may-def-use from existing IDA exports.

Only register names are used internally to recover links; none are emitted as
features. This is NOT full x86 semantics, memory SSA, flag flow or interprocedural
taint analysis. Calls and unknown instructions are barriers (coverage is emitted).
Partial-register writes are weak updates to a canonical register family. This
overapproximates aliasing; address registers are absent in these exports and omitted.
"""
from collections import Counter, deque
import re

import numpy as np

EDGE_KIND_NAMES = ('explicit_register_may_def_use',)
ALIASES = {}
for family, names in enumerate((('rax', 'eax', 'ax', 'al', 'ah'), ('rbx', 'ebx', 'bx', 'bl', 'bh'),
                               ('rcx', 'ecx', 'cx', 'cl', 'ch'), ('rdx', 'edx', 'dx', 'dl', 'dh'),
                               ('rsi', 'esi', 'si', 'sil'), ('rdi', 'edi', 'di', 'dil'),
                               ('rbp', 'ebp', 'bp', 'bpl'), ('rsp', 'esp', 'sp', 'spl'))):
    for j, name in enumerate(names):
        ALIASES[name] = (family, j < 2)
for i in range(8, 16):
    for suffix in ('', 'd', 'w', 'b'):
        ALIASES[f'r{i}{suffix}'] = (i, suffix in ('', 'd'))


def semantics(insn):
    """(reads, writes[(family,strong)], barrier, reason). Omitted effects are documented."""
    m = insn['mnemonic'].lower().split()[-1]
    ops = insn.get('operands', [])
    regs = [ALIASES.get(o.get('register', '').lower()) if o.get('category') == 'register' else None for o in ops]
    if insn.get('is_call') or m in ('call', 'syscall', 'sysenter', 'int', 'int3'):
        return set(), [], True, 'call_or_system_barrier'
    if any(o.get('category') == 'register' and r is None for o, r in zip(ops, regs)):
        return set(), [], True, 'unsupported_barrier'
    # Branches/returns do not contribute flag or implicit stack-register flow.
    if m in ('nop', 'endbr32', 'endbr64', 'ret', 'retn', 'retf', 'jmp') or (m.startswith('j') and m != 'jmp'):
        return {r[0] for r in regs if r}, [], False, 'supported'
    if m in ('mov', 'movzx', 'movsx', 'movsxd', 'lea') and len(ops) == 2:
        reads = {r[0] for r in regs[1:] if r} if m != 'lea' else set()
        return reads, [regs[0]] if regs[0] else [], False, 'supported'
    if m in ('add', 'sub', 'and', 'or', 'xor', 'shl', 'shr', 'sar', 'sal', 'rol', 'ror') and len(ops) == 2:
        reads = {r[0] for r in regs if r}
        # Only identical operand registers (not merely same family) are zero idioms.
        if m in ('xor', 'sub') and regs[0] and ops[0].get('register') == ops[1].get('register'):
            reads = set()
        # Variable/zero shift counts can leave the destination unchanged.
        write = (regs[0][0], False) if regs[0] and m in ('shl', 'shr', 'sar', 'sal', 'rol', 'ror') else regs[0]
        return reads, [write] if write else [], False, 'supported'
    if m in ('inc', 'dec', 'neg', 'not') and len(ops) == 1:
        return {r[0] for r in regs if r}, [regs[0]] if regs[0] else [], False, 'supported'
    if m in ('cmp', 'test') and len(ops) == 2:
        return {r[0] for r in regs if r}, [], False, 'supported'
    if m == 'imul' and len(ops) in (2, 3) and regs[0]:
        return {r[0] for r in (regs if len(ops) == 2 else regs[1:]) if r}, [regs[0]], False, 'supported'
    # Conditional moves weakly define: old destination can still reach later uses.
    if re.fullmatch(r'cmov[a-z]+', m) and len(ops) == 2 and regs[0]:
        return {r[0] for r in regs if r}, [(regs[0][0], False)], False, 'supported'
    if re.fullmatch(r'set[a-z]+', m) and len(ops) == 1 and regs[0]:
        return set(), [(regs[0][0], False)], False, 'supported'
    return set(), [], True, 'unsupported_barrier'


def defuse_edges(blocks, cfg_edges, max_steps=1_000_000, max_edges=2_000_000):
    """Return sorted unique local block pairs and coverage. Loops use a fixed point.

    Summarize each block as GEN/KILL plus incoming-use registers. The bitsets hold
    defining BLOCK ids, not instruction ids. Self-block links are omitted. Resource
    exhaustion raises (no silently truncated or apparently complete output).
    """
    n = len(blocks)
    pred, succ = [[] for _ in blocks], [[] for _ in blocks]
    for src, dst in cfg_edges:
        pred[dst].append(src)
        succ[src].append(dst)
    gens, kills, incoming_uses = [], [], []
    coverage = Counter(instructions=0, supported=0, call_or_system_barrier=0,
                       unsupported_barrier=0, address_operands_omitted=0)
    for b, block in enumerate(blocks):
        gen, kill, uses = {}, set(), set()
        for insn in block['instructions']:
            coverage['instructions'] += 1
            coverage['address_operands_omitted'] += sum(o.get('category') in ('phrase', 'displacement') for o in insn.get('operands', []))
            reads, writes, barrier, reason = semantics(insn)
            coverage[reason] += 1
            uses.update(reads - kill)
            if barrier:
                gen.clear()
                kill.update(range(16))
            for reg, strong in writes:
                gen[reg] = 1 << b  # all definitions inside this block collapse to b
                if strong:
                    kill.add(reg)
        gens.append(gen)
        kills.append(kill)
        incoming_uses.append(uses)
    outs = [{} for _ in blocks]
    queue, queued = deque(range(n)), set(range(n))
    steps = 0
    while queue:
        b = queue.popleft()
        queued.remove(b)
        steps += 1
        if steps > max_steps:
            raise RuntimeError('def-use fixed-point step budget exceeded')
        new = dict(gens[b])
        for p in pred[b]:
            for reg, bits in outs[p].items():
                if reg not in kills[b]:
                    new[reg] = new.get(reg, 0) | bits
        if new != outs[b]:
            outs[b] = new
            for s in succ[b]:
                if s not in queued:
                    queue.append(s)
                    queued.add(s)
    edges = set()
    for b in range(n):
        bits = 0
        for p in pred[b]:
            for reg in incoming_uses[b]:
                bits |= outs[p].get(reg, 0)
        bits &= ~(1 << b)
        while bits:
            lowest = bits & -bits
            edges.add((lowest.bit_length() - 1, b))
            bits ^= lowest
            if len(edges) > max_edges:
                raise RuntimeError('def-use edge budget exceeded')
    coverage['fixed_point_steps'] = steps
    return np.array(sorted(edges), dtype=np.int32).reshape(-1, 2), dict(coverage)
