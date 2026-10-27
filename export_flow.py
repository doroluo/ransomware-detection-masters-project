import json
import os
import idaapi
import idautils
import idc
import ida_auto
import ida_funcs

ida_auto.auto_wait()

def mnemonics(start, end):
    ea = start
    out = []
    while ea < end and ea != idc.BADADDR:
        mnem = idc.print_insn_mnem(ea)
        if mnem:
            out.append(mnem.lower())
        nxt = idc.next_head(ea, end)
        if nxt == idc.BADADDR or nxt <= ea:
            break
        ea = nxt
    return out

functions = []
for fva in idautils.Functions():
    func = ida_funcs.get_func(fva)
    if func is None:
        continue
    chart = idaapi.FlowChart(func)
    blocks, edges = [], []
    for block in chart:
        blocks.append({
            "id": block.id,
            "start": block.start_ea,
            "opcodes": mnemonics(block.start_ea, block.end_ea),
        })
        for succ in block.succs():
            kind = "fallthrough" if succ.start_ea == block.end_ea else "branch"
            edges.append({"src": block.id, "dst": succ.id, "kind": kind})
    calls = []
    for head in idautils.FuncItems(fva):
        for xref in idautils.CodeRefsFrom(head, 0):
            callee = ida_funcs.get_func(xref)
            if callee and callee.start_ea != fva:
                calls.append({"at": head, "to": callee.start_ea})
    functions.append({
        "name": idc.get_func_name(fva),
        "start": fva,
        "blocks": blocks,
        "edges": edges,
        "calls": calls,
    })

idb = idc.get_idb_path()
out_path = os.path.splitext(idb)[0] + ".flow.json"
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump({"file": idc.get_input_file_path(), "functions": functions}, handle)
idc.qexit(0)