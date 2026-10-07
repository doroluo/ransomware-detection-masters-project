"""Batch static CFG extraction. Run with the IDA idalib Python environment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def extract(binary, output, label, cohort, source, digest):
    # idapro must initialize idalib before any IDAPython module is imported.
    import idapro
    import ida_auto
    import ida_bytes
    import ida_funcs
    import ida_gdl
    import ida_idp
    import ida_kernwin
    import ida_lines
    import ida_ua
    import idautils
    import idc

    opened = False
    try:
        result = idapro.open_database(str(binary), run_auto_analysis=True)
        if result not in (None, 0):
            raise RuntimeError(f'IDA open_database failed: {result}')
        opened = True
        if not ida_auto.auto_wait():
            raise RuntimeError('IDA auto-analysis was interrupted')
        functions = []
        for address in idautils.Functions():
            function = ida_funcs.get_func(address)
            blocks = list(ida_gdl.FlowChart(function, flags=ida_gdl.FC_NOEXT))
            block_ids = {block.id for block in blocks}
            nodes, edges, calls = [], [], []
            for block in blocks:
                instructions = []
                for ea in idautils.Heads(block.start_ea, block.end_ea):
                    if not ida_bytes.is_code(ida_bytes.get_full_flags(ea)):
                        continue
                    insn = ida_ua.insn_t()
                    if not ida_ua.decode_insn(insn, ea):
                        continue
                    operands = []
                    for index, operand in enumerate(insn.ops):
                        if operand.type == ida_ua.o_void:
                            break
                        operands.append({'text': idc.print_operand(ea, index),
                                         'type': int(operand.type)})
                    instructions.append({
                        'address': hex(ea), 'size': insn.size,
                        'mnemonic': idc.print_insn_mnem(ea), 'operands': operands,
                        'asm': ida_lines.tag_remove(idc.generate_disasm_line(ea, 0) or ''),
                        'bytes': (ida_bytes.get_bytes(ea, insn.size) or b'').hex(),
                    })
                    if ida_idp.is_call_insn(insn):
                        targets = list(idautils.CodeRefsFrom(ea, False))
                        calls.append({'block_id': block.id, 'address': hex(ea),
                                      'targets': [hex(target) for target in targets],
                                      'resolution': 'code_refs' if targets else 'unresolved'})
                nodes.append({'id': block.id, 'start': hex(block.start_ea),
                              'end_exclusive': hex(block.end_ea),
                              'type': int(block.type), 'instructions': instructions})
                edges.extend([block.id, successor.id] for successor in block.succs()
                             if successor.id in block_ids)
            functions.append({'address': hex(address),
                              'name': ida_funcs.get_func_name(address),
                              'nodes': nodes, 'edges': sorted(edges), 'calls': calls})
        if not functions or not any(node['instructions'] for function in functions
                                    for node in function['nodes']):
            raise RuntimeError('IDA recovered no usable function instructions')
        atomic_json(output, {
            'schema_version': 1, 'sha256': digest, 'source_path': source,
            'label': label, 'cohort': cohort, 'ida_version': ida_kernwin.get_kernel_version(),
            'graph_kind': 'per_function_cfg',
            'edge_semantics': 'IDA successors; external blocks excluded; calls separate',
            'functions': functions,
        })
    finally:
        if opened:
            idapro.close_database(False)


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', nargs=3,
                        metavar=('CLASS', 'COHORT', 'FOLDER'),
                        help='repeatable: goodware/ransomware, cohort name, folder')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--timeout', type=positive, default=600, help='seconds per binary')
    parser.add_argument('--limit', type=positive, help='PE candidates per input folder')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--worker', nargs=6, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        binary, output, label, cohort, source, digest = args.worker
        extract(Path(binary), Path(output), int(label), cohort, source, digest)
        return 0
    if not args.input or args.out is None:
        parser.error('--input and --out are required')
    inputs = []
    for class_name, cohort, folder in args.input:
        if class_name not in ('goodware', 'ransomware'):
            parser.error('CLASS must be goodware or ransomware')
        if not cohort or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in cohort):
            parser.error('COHORT must contain only letters, digits, underscores or hyphens')
        root = Path(folder).resolve()
        if not root.is_dir():
            parser.error(f'input folder does not exist: {root}')
        if args.out.resolve() == root or root in args.out.resolve().parents:
            parser.error('output must be outside input folders')
        inputs.append((class_name, cohort, root))
    try:
        subprocess.run([sys.executable, '-c', 'import idapro'], check=True)
    except subprocess.CalledProcessError:
        parser.error('idapro unavailable: install/activate the IDA idalib Python binding')
    args.out.mkdir(parents=True, exist_ok=True)
    failures = 0
    manifest = args.out / 'manifest.csv'
    with manifest.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'source', 'class', 'cohort', 'sha256', 'status', 'output', 'log', 'error'])
        if manifest.stat().st_size == 0:
            writer.writeheader()
        for class_name, cohort, root in inputs:
            candidates = 0
            for binary in sorted(root.rglob('*')):
                if not binary.is_file():
                    continue
                row = dict(source=str(binary), **{'class': class_name}, cohort=cohort,
                           sha256='', status='', output='', log='', error='')
                try:
                    with binary.open('rb') as source:
                        if source.read(2) != b'MZ':
                            continue
                        if args.limit and candidates >= args.limit:
                            break
                        candidates += 1
                        source.seek(0)
                        digest = hashlib.file_digest(source, 'sha256').hexdigest()
                    folder = args.out / class_name / cohort
                    folder.mkdir(parents=True, exist_ok=True)
                    output = folder / f'{digest}.json'
                    log = folder / f'{digest}.log'
                    row.update(sha256=digest, output=str(output), log=str(log))
                    if output.exists() and not args.overwrite:
                        saved = json.loads(output.read_text(encoding='utf-8'))
                        if (saved.get('schema_version'), saved.get('sha256'), saved.get('label'), saved.get('cohort')) != (1, digest, int(class_name == 'ransomware'), cohort):
                            raise RuntimeError('existing output identity mismatch; use --overwrite')
                        row['status'] = 'skipped_existing'
                    else:
                        # Database sidecars are created beside a private input copy.
                        with tempfile.TemporaryDirectory(prefix='ida_cfg_') as scratch:
                            copied = Path(scratch) / 'sample.exe'
                            shutil.copyfile(binary, copied)
                            with copied.open('rb') as source:
                                if hashlib.file_digest(source, 'sha256').hexdigest() != digest:
                                    raise RuntimeError('input changed during copy')
                            with log.open('w', encoding='utf-8') as logging:
                                subprocess.run([
                                    sys.executable, str(Path(__file__).resolve()), '--worker',
                                    str(copied), str(output.resolve()),
                                    str(int(class_name == 'ransomware')), cohort, str(binary), digest,
                                ], stdout=logging, stderr=subprocess.STDOUT,
                                    timeout=args.timeout, check=True)
                        if not output.exists():
                            raise RuntimeError('worker produced no output')
                        row['status'] = 'ok'
                except Exception as error:
                    failures += 1
                    row.update(status='error', error=str(error))
                writer.writerow(row)
                stream.flush()
                print(f"{row['status']}: {binary}", flush=True)
    print(f'Manifest: {manifest}; failures: {failures}')
    return int(failures > 0)


if __name__ == '__main__':
    sys.exit(main())
