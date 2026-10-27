#!/usr/bin/env python3
"""IDA assembly listing export, called by extract_cfg_with_ida.py --asm-out inside the same IDA analysis.

export_current(out, sha256, cfg_extractor_sha256) writes out/asm/<sha256>.asm.gz (IDA OFILE_ASM, gzip) and
out/status/<sha256>.json; a failure is recorded in out/combined_errors/<sha256>.json and never invalidates the CFG.
The module keeps this name because the extractor imports it as run_mendeley_asm.
"""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, separators=(',', ':')), encoding='utf-8')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def text_digest(stream, target=None):
    """Validate plain-text bytes without interpreting assembly, strings or markup."""
    h, size = hashlib.sha256(), 0
    for block in iter(lambda: stream.read(1048576), b''):
        if b'\0' in block:
            raise ValueError('not_plain_text')
        h.update(block)
        size += len(block)
        if target is not None:
            target.write(block)
    if not size:
        raise ValueError('empty_assembly')
    return h.hexdigest(), size


def export_open_database(output, sha256):
    """Export the already analyzed database; never opens or closes a database."""
    import ida_ida
    import ida_kernwin
    import ida_loader
    import idc
    plain = Path(str(output) + '.asm.tmp')
    packed = Path(str(output) + '.tmp')
    try:
        lines = idc.gen_file(ida_loader.OFILE_ASM, str(plain), ida_ida.inf_get_min_ea(),
                             ida_ida.inf_get_max_ea(), ida_loader.GENFLG_ASMTYPE)
        if lines <= 0:
            raise RuntimeError('assembly_generation_failed')
        with plain.open('rb') as stream:
            checksum, size = text_digest(stream)
        with plain.open('rb') as source, gzip.open(str(packed), 'wb') as target:
            shutil.copyfileobj(source, target)
        packed.replace(output)
        metadata = dict(
            sha256=sha256, status='ok', asm_sha256=checksum, asm_bytes=size,
            compressed_sha256=digest(output), lines=lines,
            ida_version=ida_kernwin.get_kernel_version(), processor=ida_ida.inf_get_procname(),
            exporter_sha256=digest(__file__), format='IDA OFILE_ASM', static_only=True)
        write_json(Path(str(output) + '.meta.json'), metadata)
        return metadata
    finally:
        for path in (plain, packed):
            if path.exists():
                path.unlink()


def export_current(out, sha256, cfg_extractor_sha256):
    """Publish only complete ASM artifacts to the independent transfer controller."""
    out = Path(out)
    for folder in ('asm', 'status', 'combined_errors'):
        (out / folder).mkdir(exist_ok=True)
    status = out / 'status' / (sha256 + '.json')
    if status.exists() and json.loads(status.read_text()).get('status') in ('ok', 'deferred'):
        return
    started = time.monotonic()
    try:
        metadata = export_open_database(out / 'asm' / (sha256 + '.asm.gz'), sha256)
        metadata.update(analysis_mode='shared_cfg_database', cfg_extractor_sha256=cfg_extractor_sha256,
                        asm_export_seconds=round(time.monotonic() - started, 3), attempts=[])
        write_json(status, metadata)
    except Exception as error:
        # A valid CFG remains valid. The later backfill retries this missing ASM.
        write_json(out / 'combined_errors' / (sha256 + '.json'),
                          dict(sha256=sha256, reason=type(error).__name__ + ': ' + str(error)))
