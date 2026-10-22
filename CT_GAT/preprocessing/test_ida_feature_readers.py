import gzip
import json

import pytest

import ida_feature_readers as readers


def operand(category, register='', **kw):
    return dict(category=category, register=register, width=4, **kw)


def instruction(mnemonic='mov', operands=None, imports=(), call=False):
    return dict(mnemonic=mnemonic, operands=operands or [], import_ids=list(imports), is_call=call)


def make_export(path):
    imports = [dict(dll='KERNEL32.dll', name='ReadFile'), dict(dll='kernel32.DLL', name='readfile', delayed=True),
               dict(dll='ADVAPI32.dll', name='RegOpenKeyW'), dict(dll='ordinal.dll', ordinal=7)]
    rows = [dict(record='header', schema=readers.SCHEMA, sha256='a' * 64, pe={'imports': imports}),
        dict(record='function', id=5, name='entry', rva=10, blocks=[
            dict(id=4, start_rva=20, instructions=[instruction('call', [operand('memory')], [0], True)]),
            dict(id=9, start_rva=10, instructions=[instruction('mov', [operand('register', 'eax'), operand('displacement', phrase=5)])])], edges=[[9, 4]]),
        dict(record='function', id=6, name='empty', rva=30, blocks=[], edges=[]),
        dict(record='orphan', instruction=instruction(imports=[3])),
        dict(record='footer', complete=True, counts=dict(functions=2, blocks=2, instructions=2, orphan_instructions=1))]
    write(path, rows)
    return rows


def write(path, rows):
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        stream.writelines(json.dumps(row) + '\n' for row in rows)


@pytest.mark.parametrize('op,role,token', [
    (operand('register', 'eax'), 'REG_DATA', 70), (operand('register', 'r9b'), 'REG_DATA', 70),
    (operand('register', 'rbp'), 'REG_STACK', 71), (operand('immediate'), 'IMM_VALUE', 74),
    (operand('code_target'), 'CODE_TARGET', 74), (operand('memory'), 'MEM_GLOBAL_REF', 73),
    (operand('displacement', phrase=5), 'MEM_UNKNOWN', 73),
    (operand('phrase', address_registers=['rsp']), 'MEM_STACK_REF', 72),
    (operand('processor_specific'), 'UNKNOWN_OPERAND', 73), (None, 'PADDING', 0)])
def test_operand_roles(op, role, token):
    assert readers.operand_role(op)[:2] == (role, token)


def test_operand_alignment_and_no_register_width_features(tmp_path):
    path = tmp_path / 'export.jsonl.gz'
    make_export(path)
    rows = list(readers.iter_operand_roles(path))
    assert len(rows) == 2
    assert rows[0]['block_id'] == 9 and rows[0]['block_index'] == 0
    assert rows[1]['block_id'] == 4 and rows[1]['block_index'] == 1
    assert rows[0]['tokens'] == [25, 70, 73]
    assert rows[1]['tokens'] == [143, 73, 0]
    text = json.dumps(rows)
    assert 'eax' not in text and 'width' not in text and 'phrase' not in text
    orphan_rows = list(readers.iter_operand_roles(path, include_orphans=True))
    assert len(orphan_rows) == 3 and orphan_rows[-1]['orphan']
    assert orphan_rows[-1]['block_index'] is None and orphan_rows[-1]['fn_index'] is None


def test_unused_imports_duplicate_symbols_delay_and_orphans(tmp_path):
    path = tmp_path / 'export.jsonl.gz'
    make_export(path)
    header = readers.read_header_imports(path)
    assert len(header['imports']) == 4 and not header['export_complete_verified']
    features = readers.read_import_features(path)
    assert features['unreferenced_entry_indices'] == [1, 2]
    assert features['unreferenced_imports'] == ['advapi32.dll!regopenkeyw']
    assert features['referenced_imports'] == ['kernel32.dll!readfile', 'ordinal.dll!#7']
    assert features['imports'][1]['delayed']
    assert features['imports'][3]['observed_reference'] and not features['imports'][3]['observed_call_reference']


@pytest.mark.parametrize('bad', ['incomplete', 'bad_import', 'bad_count', 'duplicate_function'])
def test_malformed_exports_rejected_by_both_readers(tmp_path, bad):
    path = tmp_path / 'export.jsonl.gz'
    rows = make_export(path)
    if bad == 'incomplete':
        rows.pop()
    elif bad == 'bad_import':
        rows[1]['blocks'][0]['instructions'][0]['import_ids'] = [99]
    elif bad == 'bad_count':
        rows[-1]['counts']['instructions'] = 100
    else:
        rows[2]['id'] = rows[1]['id']
    write(path, rows)
    with pytest.raises(ValueError):
        list(readers.iter_operand_roles(path))
    with pytest.raises(ValueError):
        readers.read_import_features(path)


def test_atomic_output_and_no_source_mutation(tmp_path):
    path = tmp_path / 'export.jsonl.gz'
    make_export(path)
    before = path.read_bytes()
    out = readers.build(path, tmp_path / 'features')
    assert path.read_bytes() == before
    assert json.loads((out / 'manifest.json').read_text())['counts']['instructions'] == 2
    with gzip.open(out / 'operand_roles.jsonl.gz', 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    assert rows[-1]['complete']
    with pytest.raises(FileExistsError):
        readers.build(path, tmp_path / 'features')


def test_extra_operand_preserved_and_no_api_substring_match(tmp_path):
    path = tmp_path / 'export.jsonl.gz'
    rows = make_export(path)
    rows[1]['blocks'][1]['instructions'][0]['operands'] = [operand('register', 'rax'), operand('register', 'rcx'), operand('immediate')]
    rows[0]['pe']['imports'][0]['name'] = 'NotReadFile'
    write(path, rows)
    result = list(readers.iter_operand_roles(path))
    assert result[0]['extra_operands'] == 1
    assert result[0]['operand_role_ids'] == [70, 70, 74]
    assert result[1]['tokens'][0] == readers.TOKEN_MAP['CALL']
