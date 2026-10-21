"""A tiny collection folder in the layout `preprocess_full.py --root` reads: `splits/{train,test}.csv`, one `.asm.gz` listing
and one `.jsonl.gz` CFG per sample. Each sample is one function that calls one import, so a run needs no real data."""
import csv
import gzip
import json
from pathlib import Path

SCHEMA = 'ida-cfg-research/2.1'
IMAGEBASE = 0x400000
CSV_FIELDS = ['sample_id', 'collection', 'label', 'architecture', 'asm_status', 'asm_path']


def listing(apis):
    """An IDA listing of `sub_401000`: a prologue, one `call ds:<api>` for each name, an epilogue."""
    lines = ['; Imagebase     : 400000', '; Imports from KERNEL32.dll']
    lines += [f'                extrn {api}:dword ; CODE XREF: sub_401000+3↑p' for api in apis]
    lines += ['', 'sub_401000      proc near', '                push    ebp', '                mov     ebp, esp']
    lines += [f'                call    ds:{api} ; KERNEL32' for api in apis]
    lines += ['                pop     ebp', '                retn', 'sub_401000      endp', '']
    return '\n'.join(lines)


def cfg_records(sample_id, apis, library=False):
    rows = [(0x1000, 'push'), (0x1001, 'mov')] + [(0x1003 + 6 * i, 'call') for i in range(len(apis))]
    end = 0x1003 + 6 * len(apis)
    rows += [(end, 'pop'), (end + 1, 'retn')]
    instructions = []
    for rva, mnemonic in rows:
        row = {'rva': hex(rva), 'mnemonic': mnemonic, 'size': 1}
        if mnemonic == 'call':
            row.update(is_call=True, import_ids=[(rva - 0x1003) // 6])
        instructions.append(row)
    block = {'id': 0, 'start_rva': hex(rows[0][0]), 'end_rva': hex(rows[-1][0] + 1), 'instructions': instructions}
    return [
        {'record': 'header', 'schema': SCHEMA, 'sha256': sample_id, 'imagebase': hex(IMAGEBASE),
         'pe': {'imports': [{'name': api, 'dll': 'KERNEL32.dll'} for api in apis]}},
        {'record': 'function', 'id': 0, 'name': 'sub_401000', 'rva': '0x1000', 'blocks': [block], 'edges': [],
         'library': library},
        {'record': 'footer', 'complete': True},
    ]


def write_gz(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8', newline='\n') as handle:
        handle.write(text)


def write_collection(root, samples, cfg_root=None, with_cfg_path=False, extra_columns=None, drop_columns=()):
    """samples: [(sample_id, split, label, [api, ...])], each optionally followed by a dict of extra CSV columns. The CFG sits next to the listing, or under `cfg_root` (same
    relative path) when it is given; `with_cfg_path` also puts it in a `cfg_path` column (relative to the root)."""
    root = Path(root)
    fields = [name for name in CSV_FIELDS + (['cfg_path'] if with_cfg_path else []) + list(extra_columns or [])
              if name not in drop_columns]
    rows = {'train': [], 'test': []}
    for sample_id, split, label, apis, *more in samples:
        asm_rel = f'samples/{split}/fam/{sample_id}.asm.gz'
        cfg_rel = f'samples/{split}/fam/{sample_id}.jsonl.gz'
        write_gz(root / asm_rel, listing(apis))
        write_gz((root if cfg_root is None else Path(cfg_root)) / cfg_rel,
                 '\n'.join(json.dumps(record) for record in cfg_records(sample_id, apis)) + '\n')
        row = {'sample_id': sample_id, 'collection': root.name, 'label': label, 'architecture': 'x86',
               'asm_status': 'ok', 'asm_path': asm_rel}
        if with_cfg_path:
            row['cfg_path'] = cfg_rel
        row.update(more[0] if more else {})
        rows[split].append(row)
    (root / 'splits').mkdir(parents=True, exist_ok=True)
    for split, split_rows in rows.items():
        with (root / 'splits' / f'{split}.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(split_rows)
    return root
