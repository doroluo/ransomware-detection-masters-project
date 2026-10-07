#!/usr/bin/env python3
"""Versioned, streaming static research features. Python 3.8; samples stay in VM."""
import argparse
import collections
import gzip
import hashlib
import json
import math
import mmap
from pathlib import Path
import re

SCHEMA = 'ida-cfg-research/2.1'
FIELDS = {
    'header': 'record schema sha256 extractor_sha256 arch input_kind size byte_histogram pe ida_version imagebase analysis_options',
    'function': 'record id rva name flags library thunk chunks blocks edges',
    'orphan': 'record instruction',
    'string': 'record rva file_offset encoding text',
    'footer': 'record counts graph_status complete static_limitations quality',
}


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pe_metadata(path):
    import pefile
    pe = None
    try:
        pe = pefile.PE(str(path), fast_load=True)
        pe.parse_data_directories(directories=[0, 1, 13])
        imports = []
        for attr in ('DIRECTORY_ENTRY_IMPORT', 'DIRECTORY_ENTRY_DELAY_IMPORT'):
            for library in getattr(pe, attr, []):
                for symbol in library.imports:
                    imports.append(dict(dll=library.dll.decode('utf-8', 'replace'),
                                        name=symbol.name.decode('utf-8', 'replace') if symbol.name else '',
                                        ordinal=symbol.ordinal, rva=int(symbol.address - pe.OPTIONAL_HEADER.ImageBase),
                                        delayed=attr.endswith('DELAY_IMPORT')))
        return dict(parse_status='ok', machine=pe.FILE_HEADER.Machine, subsystem=pe.OPTIONAL_HEADER.Subsystem,
                    timestamp=pe.FILE_HEADER.TimeDateStamp, is_dll=bool(pe.FILE_HEADER.Characteristics & 0x2000),
                    imagebase=hex(pe.OPTIONAL_HEADER.ImageBase), entry_rva=pe.OPTIONAL_HEADER.AddressOfEntryPoint,
                    sections=[dict(name=s.Name.rstrip(b'\0').decode('utf-8', 'replace'), rva=s.VirtualAddress,
                                   virtual_size=s.Misc_VirtualSize, raw_size=s.SizeOfRawData,
                                   raw_offset=s.PointerToRawData, characteristics=s.Characteristics,
                                   entropy=s.get_entropy()) for s in pe.sections], imports=imports,
                    exports=[dict(name=s.name.decode('utf-8', 'replace') if s.name else '', ordinal=s.ordinal,
                                  rva=s.address, forwarder=s.forwarder.decode('utf-8', 'replace') if s.forwarder else '')
                             for s in getattr(getattr(pe, 'DIRECTORY_ENTRY_EXPORT', None), 'symbols', [])])
    except Exception as error:
        return dict(parse_status=type(error).__name__)
    finally:
        if pe is not None:
            pe.close()


def histogram(path):
    counts = collections.Counter()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            counts.update(block)
    return [counts[i] for i in range(256)]


def file_strings(path):
    # Raw strings are inert JSON values, never evaluated/rendered as markup.
    # Minimum length is recorded, not a maximum or sample-size truncation.
    if not Path(path).stat().st_size:
        return
    with Path(path).open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        for pattern, encoding in [(rb'[\x20-\x7e]{4,}', 'ascii'), (rb'(?:[\x20-\x7e]\x00){4,}', 'utf-16le')]:
            for match in re.finditer(pattern, mapped):
                yield dict(record='string', rva=None, file_offset=match.start(), encoding=encoding,
                           text=match.group().decode(encoding))


def extract(binary, output, digest, arch, kind, asm_out=None):
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if hash_file(binary) != digest:
        raise ValueError('input_changed')
    pe = pe_metadata(binary)
    native = kind == 'candidate'
    opened = False
    counts = dict(functions=0, blocks=0, edges=0, instructions=0, calls=0, strings=0, orphan_instructions=0)
    quality = dict(decode_failures=0, unknown_mnemonics=0, unique_ida_code_instructions=0,
                   unique_ida_code_bytes=0, strings_min_length=4, string_scan='file_ascii_utf16le',
                   executable_raw_bytes=sum(s['raw_size'] for s in pe.get('sections', []) if s['characteristics'] & 0x20000000))
    imagebase = int(pe.get('imagebase', '0'), 16)
    ida_version = ''
    analysis_options = {}
    tmp = Path(str(output) + '.tmp')
    try:
        if native:
            import idapro
            import ida_auto
            import ida_bytes
            import ida_funcs
            import ida_gdl
            import ida_idp
            import ida_ida
            import ida_kernwin
            import ida_nalt
            import ida_ua
            import idautils
            import idc
            if idapro.open_database(str(binary), True) != 0:
                raise RuntimeError('ida_open_failed')
            opened = True
            if not ida_auto.auto_wait():
                raise RuntimeError('autoanalysis_incomplete')
            ida_version = ida_kernwin.get_kernel_version()
            imagebase = ida_nalt.get_imagebase()
            analysis_options = dict(af=int(ida_ida.inf_get_af()), af2=int(ida_ida.inf_get_af2()), processor=ida_ida.inf_get_procname())
            quality['string_scan'] = 'ida_ascii_utf16_min4_plus_file_ascii_utf16le'
        with gzip.open(str(tmp), 'wt', encoding='utf-8', compresslevel=6) as stream:
            def emit(value):
                stream.write(json.dumps(value, ensure_ascii=True, separators=(',', ':')) + '\n')
            emit(dict(record='header', schema=SCHEMA, sha256=digest, extractor_sha256=hash_file(__file__),
                      arch=arch or 'unknown', input_kind=kind, size=Path(binary).stat().st_size,
                      byte_histogram=histogram(binary), pe=pe, ida_version=ida_version, imagebase=hex(imagebase), analysis_options=analysis_options))
            if native:
                import_map = {imagebase + item['rva']: index for index, item in enumerate(pe.get('imports', []))}

                def instruction(ea):
                    insn = ida_ua.insn_t()
                    if not ida_ua.decode_insn(insn, ea):
                        quality['decode_failures'] += 1
                        return None
                    mnemonic = idc.print_insn_mnem(ea) or ''
                    if not mnemonic:
                        quality['unknown_mnemonics'] += 1
                    operands = []
                    for op in insn.ops:
                        if op.type == ida_ua.o_void:
                            break
                        category = {ida_ua.o_reg: 'register', ida_ua.o_imm: 'immediate', ida_ua.o_mem: 'memory',
                                    ida_ua.o_near: 'code_target', ida_ua.o_far: 'code_target',
                                    ida_ua.o_phrase: 'phrase', ida_ua.o_displ: 'displacement'}.get(op.type, 'processor_specific')
                        width = int(ida_ua.get_dtype_size(op.dtype))
                        # op_t has unions: read only fields appropriate to the category.
                        register = ida_idp.get_reg_name(op.reg, width) if category == 'register' and width > 0 else ''
                        value = hex(int(op.value)) if category == 'immediate' else None
                        target = int(op.addr) - imagebase if category in ('memory', 'code_target') else None
                        displacement = hex(int(op.addr)) if category == 'displacement' else None
                        operands.append(dict(type=int(op.type), category=category, dtype=int(op.dtype), width=width,
                                             register=register or '', immediate=value, target_rva=target,
                                             displacement=displacement,
                                             phrase=int(op.phrase) if category in ('phrase', 'displacement') else None,
                                             processor_flags=[int(op.specflag1), int(op.specflag2), int(op.specflag3), int(op.specflag4)]))
                    code_targets = list(idautils.CodeRefsFrom(ea, False))
                    data_targets = list(idautils.DataRefsFrom(ea))
                    is_call = bool(ida_idp.is_call_insn(insn))
                    import_ids = sorted(set(import_map[t] for t in code_targets + data_targets if t in import_map))
                    counts['calls'] += int(is_call)
                    return dict(rva=int(ea - imagebase), size=int(insn.size), mnemonic=mnemonic.lower(), operands=operands,
                                code_refs=[int(t - imagebase) for t in code_targets], data_refs=[int(t - imagebase) for t in data_targets],
                                is_call=is_call, import_ids=import_ids,
                                call_resolution=('import' if import_ids else 'code_reference' if code_targets else 'unresolved') if is_call else 'not_call')

                for fid, ea in enumerate(idautils.Functions()):
                    func = ida_funcs.get_func(ea)
                    # External placeholders and their edges are retained explicitly.
                    chart = ida_gdl.FlowChart(func, flags=0)
                    blocks = sorted(list(chart), key=lambda b: (b.start_ea, b.id))
                    ids = {b.id: i for i, b in enumerate(blocks)}
                    nodes, edges = [], []
                    for block in blocks:
                        external = block.type == ida_gdl.fcb_extern
                        instructions = []
                        if not external:
                            for address in idautils.Heads(block.start_ea, block.end_ea):
                                if ida_bytes.is_code(ida_bytes.get_full_flags(address)):
                                    item = instruction(address)
                                    if item is not None:
                                        instructions.append(item)
                        nodes.append(dict(id=ids[block.id], start_rva=int(block.start_ea - imagebase),
                                          end_rva=int(block.end_ea - imagebase), type=int(block.type), external=external,
                                          instructions=instructions))
                        edges.extend([ids[block.id], ids[s.id]] for s in block.succs() if s.id in ids)
                        counts['instructions'] += len(instructions)
                    edges = [list(e) for e in sorted(set(tuple(e) for e in edges))]
                    emit(dict(record='function', id=fid, rva=int(ea - imagebase), name=ida_funcs.get_func_name(ea),
                              flags=int(func.flags), library=bool(func.flags & ida_funcs.FUNC_LIB),
                              thunk=bool(func.flags & ida_funcs.FUNC_THUNK),
                              chunks=[[int(a - imagebase), int(b - imagebase)] for a, b in idautils.Chunks(ea)], blocks=nodes, edges=edges))
                    counts['functions'] += 1
                    counts['blocks'] += len(nodes)
                    counts['edges'] += len(edges)
                # Preserve recognized instructions outside recovered functions.
                for segment in idautils.Segments():
                    for address in idautils.Heads(segment, idc.get_segm_end(segment)):
                        if not ida_bytes.is_code(ida_bytes.get_full_flags(address)):
                            continue
                        quality['unique_ida_code_instructions'] += 1
                        quality['unique_ida_code_bytes'] += ida_bytes.get_item_size(address)
                        if ida_funcs.get_func(address) is None:
                            item = instruction(address)
                            if item is not None:
                                emit(dict(record='orphan', instruction=item))
                                counts['orphan_instructions'] += 1
                strings = idautils.Strings()
                strings.setup(strtypes=[ida_nalt.STRTYPE_C, ida_nalt.STRTYPE_C_16], minlen=4,
                              only_7bit=False, ignore_instructions=True)
                for item in strings:
                    emit(dict(record='string', rva=int(item.ea - imagebase), file_offset=None,
                              encoding='ida_type_' + str(item.strtype), text=str(item)))
                    counts['strings'] += 1
            # Also preserves file-offset strings in resources/overlays not loaded by IDA.
            for item in file_strings(binary):
                emit(item)
                counts['strings'] += 1
            graph_status = ('recovered' if counts['instructions'] else 'no_function_instructions') if native else 'metadata_only_' + kind
            quality['worker_max_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            emit(dict(record='footer', counts=counts, graph_status=graph_status, complete=True,
                      static_limitations=['no_execution', 'no_unpacking', 'indirect_targets_may_be_unresolved',
                                          'autoanalysis_is_not_ground_truth'], quality=quality))
        validate_file(tmp, digest)
        tmp.replace(output)
        if opened and asm_out is not None:
            # Export while this same analyzed IDA database is still open.
            from run_mendeley_asm import export_current
            export_current(asm_out, digest, hash_file(__file__))
    finally:
        if opened:
            idapro.close_database(False)


def exact(value, fields):
    if not isinstance(value, dict) or set(value) != set(fields.split()):
        raise ValueError('unexpected_fields')


def validate_stream(stream, digest):
    counts = dict(functions=0, blocks=0, edges=0, instructions=0, calls=0, strings=0, orphan_instructions=0)
    header, footer = None, None

    def insn(item):
        exact(item, 'rva size mnemonic operands code_refs data_refs is_call import_ids call_resolution')
        if type(item['rva']) is not int or type(item['size']) is not int or item['size'] <= 0 or not isinstance(item['mnemonic'], str):
            raise ValueError('invalid_instruction')
        if type(item['is_call']) is not bool:
            raise ValueError('invalid_call')
        for op in item['operands']:
            exact(op, 'type category dtype width register immediate target_rva displacement phrase processor_flags')
            if op['category'] not in ('register', 'immediate', 'memory', 'code_target', 'phrase', 'displacement', 'processor_specific'):
                raise ValueError('invalid_operand')
            for field in ('immediate', 'displacement'):
                if op[field] is not None and not re.fullmatch(r'0x[0-9a-f]+', op[field]):
                    raise ValueError('invalid_constant')
        for target in item['code_refs'] + item['data_refs'] + item['import_ids']:
            if type(target) is not int:
                raise ValueError('invalid_reference')
        counts['calls'] += int(item['is_call'])

    for line in stream:
        record = json.loads(line)
        kind = record.get('record')
        if kind not in FIELDS or footer is not None:
            raise ValueError('invalid_record_order')
        exact(record, FIELDS[kind])
        if header is None and kind != 'header':
            raise ValueError('header_required')
        if kind == 'header':
            if header is not None or record['schema'] != SCHEMA or record['sha256'] != digest or not re.fullmatch(r'[0-9a-f]{64}', digest):
                raise ValueError('identity_mismatch')
            if len(record['byte_histogram']) != 256 or sum(record['byte_histogram']) != record['size']:
                raise ValueError('invalid_byte_histogram')
            header = record
        elif kind == 'function':
            if record['id'] != counts['functions']:
                raise ValueError('function_id_mismatch')
            for bid, block in enumerate(record['blocks']):
                exact(block, 'id start_rva end_rva type external instructions')
                if block['id'] != bid or block['end_rva'] < block['start_rva'] or type(block['external']) is not bool:
                    raise ValueError('invalid_block')
                for item in block['instructions']:
                    insn(item)
                counts['instructions'] += len(block['instructions'])
            for edge in record['edges']:
                if len(edge) != 2 or any(type(v) is not int or v < 0 or v >= len(record['blocks']) for v in edge):
                    raise ValueError('dangling_edge')
            counts['functions'] += 1
            counts['blocks'] += len(record['blocks'])
            counts['edges'] += len(record['edges'])
        elif kind == 'orphan':
            insn(record['instruction'])
            counts['orphan_instructions'] += 1
        elif kind == 'string':
            if not isinstance(record['text'], str):
                raise ValueError('invalid_string')
            counts['strings'] += 1
        elif kind == 'footer':
            if record['complete'] is not True or record['counts'] != counts:
                raise ValueError('incomplete_or_counts_mismatch')
            footer = record
    if footer is None:
        raise ValueError('missing_completion_footer')
    return dict(header=header, footer=footer)


def validate_file(path, digest):
    with gzip.open(str(path), 'rt', encoding='utf-8') as stream:
        return validate_stream(stream, digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--arch', default='')
    parser.add_argument('--kind', required=True)
    parser.add_argument('--asm-out', type=Path)
    args = parser.parse_args()
    extract(args.binary, args.output, args.sha256, args.arch, args.kind, args.asm_out)


if __name__ == '__main__':
    main()
