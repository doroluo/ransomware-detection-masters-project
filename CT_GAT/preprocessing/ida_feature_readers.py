"""Static IDA readers for Yanping operand vocabulary and header/unused imports.

python ida_feature_readers.py sample.jsonl.gz --out-dir features
No IDA runtime, binary execution, NumPy or Torch required. JSONL(.gz) is streamed.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

from cfg_viewer import records, SCHEMA
from token_mapping import TOKEN_MAP, API_MAP

STACK_REGISTERS = frozenset(('sp', 'esp', 'rsp', 'bp', 'ebp', 'rbp'))
ROLE_IDS = {'PADDING': 0, 'REG_DATA': 70, 'REG_STACK': 71, 'MEM_STACK_REF': 72,
            'MEM_GLOBAL_REF': 73, 'IMM_VALUE': 74, 'CODE_TARGET': 74,
            'MEM_UNKNOWN': 73, 'UNKNOWN_OPERAND': 73}


def operand_role(operand):
    """Semantic category, compatible token ID, confidence basis. Never emits register/width.

    IDA's processor-specific phrase/specflag values are deliberately not guessed:
    old exports omit explicit base/index register names and address-size mode.
    Legacy token 73 is a memory/fallback bucket, NOT a claim that unknown memory is global.
    Optional explicit address_registers from a future/augmented export can prove stack use.
    These are categories, not read/write roles: the export has no operand access flags.
    """
    if operand is None:
        return 'PADDING', 0, 'absent'
    category = operand.get('category')
    if category == 'register':
        register = operand.get('register', '').lower()
        role = 'REG_STACK' if register in STACK_REGISTERS else 'REG_DATA' if register else 'UNKNOWN_OPERAND'
        return role, ROLE_IDS[role], 'explicit_register_class' if register else 'missing_register'
    if category in ('immediate', 'code_target'):
        role = 'IMM_VALUE' if category == 'immediate' else 'CODE_TARGET'
        return role, ROLE_IDS[role], 'explicit_category'
    if category == 'memory':
        return 'MEM_GLOBAL_REF', 73, 'direct_memory_category'
    if category in ('phrase', 'displacement'):
        registers = operand.get('address_registers')
        if isinstance(registers, list) and registers and any(str(r).lower() in STACK_REGISTERS for r in registers):
            return 'MEM_STACK_REF', 72, 'explicit_address_registers'
        return 'MEM_UNKNOWN', 73, 'base_index_not_explicit'
    return 'UNKNOWN_OPERAND', 73, 'legacy_fallback'


def import_key(symbol):
    """Stable feature identity; name case normalized; ordinal and DLL retained."""
    name = symbol.get('name')
    if not name and symbol.get('ordinal') is None:
        raise ValueError('Import must have a name or ordinal')
    return str(symbol.get('dll', '')).lower() + '!' + (str(name).lower() if name else '#' + str(symbol['ordinal']))


def header_imports(header):
    """Complete normal AND delay import entries, preserving header indices and duplicates."""
    return [dict(index=i, key=import_key(s), dll=s.get('dll', ''), name=s.get('name') or None,
                 ordinal=s.get('ordinal'), delayed=bool(s.get('delayed', False)))
            for i, s in enumerate(header.get('pe', {}).get('imports', []))]


def read_header_imports(path):
    """Fast header-only reader; does not assert that the rest of the export is complete."""
    stream = records(path)
    try:
        header = next(stream)
        if header.get('record') != 'header' or header.get('schema') != SCHEMA:
            raise ValueError('Expected IDA CFG research/2.1 header')
        return {'sample_id': header['sha256'], 'imports': header_imports(header),
                'export_complete_verified': False}
    finally:
        stream.close()


def opcode_id(insn, imports):
    parts = insn['mnemonic'].upper().split()
    mnemonic = parts[-1] if parts else ''
    if mnemonic == 'CALL':
        for i in insn.get('import_ids', []):
            name = imports[i]['name'] or ''
            name = re.sub(r'@\d+$', '', name.lower())
            name = re.sub(r'^(?:__imp_)?_?', '', name)
            if name in API_MAP:
                return API_MAP[name]
        return TOKEN_MAP['CALL']
    return TOKEN_MAP['NOP_SLED'] if mnemonic == 'NOP' else TOKEN_MAP.get(mnemonic, TOKEN_MAP['UNKNOWN_OPCODE'])


def iter_operand_roles(path, include_orphans=False):
    """Instruction rows: cache-order functions, address-sorted blocks; all operand roles.

    tokens is [opcode/API, op1, op2] for Yanping's N x 3 input. Extra operands are
    retained separately, with a truncation count. IDs/indices are alignment metadata.
    Consume through exhaustion to validate the footer; CLI publishes only after that.
    Orphans are opt-in, have no function/block index, and do not join a CFG cache.
    """
    header, footer = None, None
    fn_index = block_index = insn_index = 0
    seen = set()
    imports = []
    for record in records(path):
        kind = record.get('record')
        if footer is not None:
            raise ValueError('Records after footer')
        if kind == 'header':
            if header is not None or record.get('schema') != SCHEMA:
                raise ValueError('Expected one IDA CFG research/2.1 header')
            header, imports = record, header_imports(record)
        elif kind in ('function', 'orphan'):
            if header is None:
                raise ValueError('Instruction before header')
            if kind == 'function':
                if record['id'] in seen:
                    raise ValueError('Duplicate function ID')
                seen.add(record['id'])
                blocks = sorted(record['blocks'], key=lambda b: b['start_rva'])
                if len({b['id'] for b in blocks}) != len(blocks):
                    raise ValueError('Duplicate block ID')
                groups = [(block_index + i, b['id'], b['instructions']) for i, b in enumerate(blocks)]
            else:
                if not include_orphans:
                    continue
                groups = [(None, None, [record['instruction']])]
            for global_block, local_block, instructions in groups:
                for local_insn, insn in enumerate(instructions):
                    for idx in insn.get('import_ids', []):
                        if type(idx) is not int or not 0 <= idx < len(imports):
                            raise ValueError('Instruction import index out of range')
                    roles = [operand_role(o) for o in insn.get('operands', [])]
                    first = [r[1] for r in roles[:2]]
                    first += [0] * (2 - len(first))
                    yield dict(record='instruction', fn_index=fn_index if kind == 'function' else None,
                        function_id=record.get('id') if kind == 'function' else None,
                        block_index=global_block, block_id=local_block, instruction_in_block=local_insn,
                        instruction_index=insn_index, tokens=[opcode_id(insn, imports), *first],
                        operand_roles=[r[0] for r in roles], operand_role_ids=[r[1] for r in roles],
                        role_basis=[r[2] for r in roles], extra_operands=max(0, len(roles) - 2),
                        orphan=kind == 'orphan')
                    insn_index += 1
            if kind == 'function':
                fn_index += 1
                block_index += len(blocks)
        elif kind == 'footer':
            footer = record
    if header is None or footer is None or footer.get('complete') is not True:
        raise ValueError('Incomplete export')
    expected = footer.get('counts', {})
    if expected.get('functions', fn_index) != fn_index or expected.get('blocks', block_index) != block_index:
        raise ValueError('Export footer count mismatch')
    n = expected.get('instructions')
    if n is not None and n + (expected.get('orphan_instructions', 0) if include_orphans else 0) != insn_index:
        raise ValueError('Export instruction count mismatch')


def read_import_features(path):
    """Header presence and observed-reference complement, including orphan instructions.

    An entry can be unreferenced even when a duplicate entry with the SAME symbol
    is referenced. Feature sets use symbol identity, so such a symbol is not
    falsely marked unused. Unreferenced is NOT runtime-unused.
    """
    header = footer = None
    imports, referenced, calls = [], set(), set()
    counts = Counter(functions=0, blocks=0, instructions=0, orphan_instructions=0)
    seen_functions = set()
    for r in records(path):
        if footer is not None:
            raise ValueError('Records after footer')
        kind = r.get('record')
        if kind == 'header':
            if header is not None or r.get('schema') != SCHEMA:
                raise ValueError('Expected one IDA CFG research/2.1 header')
            header, imports = r, header_imports(r)
        elif kind in ('function', 'orphan'):
            if header is None:
                raise ValueError('Instruction before header')
            if kind == 'function':
                if r['id'] in seen_functions:
                    raise ValueError('Duplicate function ID')
                seen_functions.add(r['id'])
                counts['functions'] += 1
                counts['blocks'] += len(r['blocks'])
                instructions = (i for b in r['blocks'] for i in b['instructions'])
            else:
                instructions = iter([r['instruction']])
            for insn in instructions:
                counts['instructions' if kind == 'function' else 'orphan_instructions'] += 1
                for idx in insn.get('import_ids', []):
                    if type(idx) is not int or not 0 <= idx < len(imports):
                        raise ValueError('Instruction import index out of range')
                    referenced.add(idx)
                    if insn.get('is_call'):
                        calls.add(idx)
        elif kind == 'footer':
            footer = r
    if header is None or footer is None or footer.get('complete') is not True:
        raise ValueError('Incomplete export')
    for key, value in counts.items():
        if footer.get('counts', {}).get(key, value) != value:
            raise ValueError('Export footer count mismatch: ' + key)
    all_keys = {entry['key'] for entry in imports}
    reference_keys = {imports[i]['key'] for i in referenced}
    return dict(schema='ida-import-features/1', sample_id=header['sha256'], export_complete_verified=True,
        imports=[dict(entry, observed_reference=entry['index'] in referenced,
                      observed_call_reference=entry['index'] in calls) for entry in imports],
        header_imports=sorted(all_keys), referenced_imports=sorted(reference_keys),
        unreferenced_imports=sorted(all_keys - reference_keys),
        unreferenced_entry_indices=sorted(set(range(len(imports))) - referenced),
        reference_scope='function and orphan instructions; import_ids; static references, not execution',
        limitations=['Unresolved/dynamic calls may be missing', 'Unreferenced does not mean runtime-unused'],
        counts=dict(counts))


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def build(path, out_dir, include_orphans=False):
    """Publish one new per-binary directory atomically; never edit source/cache/vocabulary."""
    source_hash = file_hash(path)
    header = read_header_imports(path)
    sid = header['sample_id']
    if not re.fullmatch(r'[0-9a-f]{64}', sid):
        raise ValueError('Invalid sample SHA-256')
    out = Path(out_dir) / sid
    if out.exists():
        raise FileExistsError(f'Output already exists: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = out.with_name(out.name + '.pending-' + uuid.uuid4().hex)
    stage.mkdir()
    summary = Counter()
    role_counts = Counter()
    with gzip.open(stage / 'operand_roles.jsonl.gz', 'wt', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(record='header', schema='ida-operand-roles/1', sample_id=sid,
            role_ids=ROLE_IDS, include_orphans=include_orphans,
            token_contract='Yanping [opcode_or_api, operand1, operand2]; IDs 0..255 and 0..75',
            memory_fallback='MEM_UNKNOWN and UNKNOWN_OPERAND map to existing fallback 73; not proven global')) + '\n')
        for row in iter_operand_roles(path, include_orphans):
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
            summary['instructions'] += 1
            summary['extra_operands'] += row['extra_operands']
            role_counts.update(row['operand_roles'])
        stream.write(json.dumps(dict(record='footer', complete=True, counts=dict(summary), role_counts=dict(role_counts))) + '\n')
    imports = read_import_features(path)
    (stage / 'imports.json').write_text(json.dumps(imports, indent=2), encoding='utf-8')
    if file_hash(path) != source_hash:
        raise ValueError('Source changed during extraction')
    manifest = dict(sample_id=sid, source=str(Path(path).resolve()), source_sha256=source_hash,
        reader_sha256=file_hash(__file__), token_mapping_sha256=file_hash(Path(__file__).with_name('token_mapping.py')),
        role_counts=dict(role_counts), counts=dict(summary), header_imports=len(imports['header_imports']),
        referenced_imports=len(imports['referenced_imports']), unreferenced_imports=len(imports['unreferenced_imports']))
    (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    os.rename(stage, out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', type=Path, nargs='+')
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--include-orphans', action='store_true')
    args = ap.parse_args()
    for source in args.input:
        print(build(source, args.out_dir, args.include_orphans))


if __name__ == '__main__':
    main()
