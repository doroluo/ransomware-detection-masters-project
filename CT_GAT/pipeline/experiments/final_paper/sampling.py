"""Sampling-only CFG reader; delegates selection to the original pinned samplers.

Retains block addresses/counts, library flags, CFG edges, and call/import signal.
Instruction text/operands and per-sample diagnostic manifests are never generated.
"""
import gzip
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as N
from s02_preprocess import preprocess_full as pp

try:
    import orjson
except ImportError:
    optional=Path(os.environ.get('LOCALAPPDATA','.'))/'Temp'/'codex-cfg-reader-deps'
    if optional.is_dir(): sys.path.insert(0,str(optional))
    try: import orjson
    except ImportError: orjson=None
json_loads=orjson.loads if orjson else json.loads


def compact_cfg(path,sid):
    blocks={}; edges=[]; fn_rvas={}; lib=set(); count=0; n_int=0; min_rva=None; header=None; complete=False
    with gzip.open(path,'rt',encoding='utf-8',errors='replace') as f:
        for line in f:
            if not line.strip(): continue
            rec=json_loads(line); kind=rec['record']
            if kind=='header':
                header=rec; assert rec['schema']==pp.SCHEMA and rec['sha256'].lower()==sid
            elif kind=='footer': complete=bool(rec.get('complete'))
            elif kind=='function':
                fid=int(rec['id']); name=str(rec['name']); fr=pp.parse_address(rec['rva'],'function rva')
                if name in fn_rvas: assert fn_rvas[name]==fr
                fn_rvas[name]=fr
                if rec.get('library'): lib.add(fid)
                for b in rec['blocks']:
                    key=(fid,int(b['id'])); assert key not in blocks; calls=[]; instructions=b['instructions']; count+=len(instructions)
                    for ins in instructions:
                        rva=ins.get('rva')
                        if isinstance(rva,int): n_int+=1; min_rva=rva if min_rva is None else min(min_rva,rva)
                        is_call=bool(ins.get('is_call')) or pp.tok._mnemonic(str(ins.get('mnemonic') or '')).lower()=='call'
                        import_ids=ins.get('import_ids') or []
                        if is_call or import_ids:
                            calls.append(N(is_call=is_call,import_ids=import_ids,code_refs=[v for v in ins.get('code_refs') or [] if isinstance(v,int) and not isinstance(v,bool)]))
                    blocks[key]=N(key=key,function_id=fid,function_name=name,block_id=key[1],start_rva=pp.parse_address(b['start_rva'],'start_rva'),instruction_count=len(instructions),instructions=calls)
                for edge in rec['edges']:
                    a,b,typ=pp._parse_edge(edge,name); edges.append(pp.CfgEdge(src=(fid,a),dst=(fid,b),edge_type=typ))
            del rec
    assert header is not None and complete and blocks
    base=pp._imagebase_of(header); convert=n_int>0 and base>0 and min_rva>=base
    if convert:
        fn_rvas={k:pp._as_rva(v,base,True) for k,v in fn_rvas.items()}
        for b in blocks.values():
            b.start_rva=pp._as_rva(b.start_rva,base,True)
            for i in b.instructions: i.code_refs=[pp._as_rva(v,base,True) for v in i.code_refs]
    return N(blocks=blocks,edges=edges,function_rvas=fn_rvas,library_functions=lib,instruction_count=count,file_id=sid)
