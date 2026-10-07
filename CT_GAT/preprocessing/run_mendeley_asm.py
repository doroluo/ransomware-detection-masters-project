#!/usr/bin/env python3
"""Static IDA assembly companion; Python 3.8, Mendeley only, verified text transfer."""
import argparse
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time

import vm_ida_cfg as common
from run_research_cfg import process_memory, available_memory


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
        common.write_json(Path(str(output) + '.meta.json'), metadata)
        return metadata
    finally:
        for path in (plain, packed):
            if path.exists():
                path.unlink()


def defer_disk_limited_assembly(out, sha256):
    """Record a retryable exception and discard only incomplete listing files."""
    out = common.check_out(out)
    if not re.fullmatch(r'[0-9a-f]{64}', sha256):
        raise ValueError('invalid_sample_hash')
    status = out / 'status' / (sha256 + '.json')
    if status.exists() and json.loads(status.read_text()).get('status') == 'ok':
        return json.loads(status.read_text())
    partial_bytes = 0
    for suffix in ('.asm.gz.asm.tmp', '.asm.gz.tmp'):
        path = out / 'asm' / (sha256 + suffix)
        if path.exists():
            partial_bytes += path.stat().st_size
            path.unlink()
    state = dict(sha256=sha256, status='deferred', reason='assembly_disk_reserve',
                 incomplete_bytes_removed=partial_bytes, retry_requires_export_strategy_change=True)
    common.write_json(status, state)
    return state


def export_current(out, sha256, cfg_extractor_sha256):
    """Publish only complete ASM artifacts to the independent transfer controller."""
    out = common.check_out(out)
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
        common.write_json(status, metadata)
    except Exception as error:
        # A valid CFG remains valid. The later backfill retries this missing ASM.
        common.write_json(out / 'combined_errors' / (sha256 + '.json'),
                          dict(sha256=sha256, reason=type(error).__name__ + ': ' + str(error)))


def worker(args):
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if digest(args.binary) != args.sha256:
        raise ValueError('input_changed')
    import idapro
    import ida_auto
    opened = False
    try:
        if idapro.open_database(str(args.binary), True) != 0:
            raise RuntimeError('ida_open_failed')
        opened = True
        if not ida_auto.auto_wait():
            raise RuntimeError('autoanalysis_incomplete')
        export_open_database(args.output, args.sha256)
    finally:
        if opened:
            idapro.close_database(False)


def untransferred(out):
    return [s for s in (json.loads(p.read_text()) for p in (out / 'status').glob('*.json'))
            if not s.get('transferred')]


def run_job(row, out, timeout=3600, retry_timeout=14400):
    sha = row['sha256']
    saved = out / 'status' / (sha + '.json')
    if saved.exists():
        return json.loads(saved.read_text())
    result = dict(sha256=sha, status='failed', attempts=[])
    output = out / 'asm' / (sha + '.asm.gz')
    for budget in (timeout, retry_timeout):
        started, reason, peak = time.monotonic(), '', 0
        common.disk_guard(out)
        if shutil.disk_usage(str(out)).free < row['size'] + 3 * 1024**3:
            raise RuntimeError('insufficient_disk_for_private_sample_copy')
        with tempfile.TemporaryDirectory(prefix=sha + '_', dir=str(out / 'scratch')) as temp:
            binary = Path(temp) / (sha + '.pe')
            shutil.copyfile(row['source'], str(binary))
            binary.chmod(0o600)
            with (out / 'logs' / (sha + '_' + str(len(result['attempts']) + 1) + '.log')).open('w') as log:
                proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker',
                    '--binary', str(binary), '--output', str(output), '--sha256', sha],
                    cwd=temp, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while proc.poll() is None:
                    peak = max(peak, process_memory(proc.pid))
                    if time.monotonic() - started > budget:
                        reason = 'timeout'
                    elif peak > 6144 * 1024**2 or available_memory() < 768 * 1024**2:
                        reason = 'memory_pressure'
                    elif shutil.disk_usage(str(out)).free < 3 * 1024**3:
                        reason = 'disk_reserve'
                    if reason:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                        break
                    time.sleep(0.2)
                if not reason and proc.returncode:
                    reason = 'worker_exit_' + str(proc.returncode)
        result['attempts'].append(dict(seconds=round(time.monotonic() - started, 3),
                                       budget=budget, peak_rss_bytes=peak, reason=reason))
        if not reason:
            metadata = json.loads(Path(str(output) + '.meta.json').read_text())
            with gzip.open(str(output), 'rb') as stream:
                checksum, size = text_digest(stream)
            if checksum != metadata['asm_sha256'] or size != metadata['asm_bytes']:
                raise ValueError('assembly_verification_failed')
            result.update(metadata)
            result.pop('reason', None)
            break
        result['reason'] = reason
        if reason == 'disk_reserve' and Path(str(output) + '.asm.tmp').exists():
            state = defer_disk_limited_assembly(out, sha)
            if shutil.disk_usage(str(out)).free >= 3 * 1024**3:
                return state
        # A killed worker cannot run its finally block; discard only its derived partials.
        for suffix in ('', '.asm.tmp', '.tmp', '.meta.json'):
            partial = Path(str(output) + suffix)
            if partial.exists():
                partial.unlink()
        if reason == 'disk_reserve':
            raise RuntimeError(reason)
        if reason not in ('timeout', 'memory_pressure', 'worker_exit_-9'):
            break
    common.write_json(saved, result)
    return result


def summarize(out):
    rows = json.loads((out / 'inventory.json').read_text())
    states = {p.stem: json.loads(p.read_text()) for p in (out / 'status').glob('*.json')}
    manifest, groups = [], collections.defaultdict(collections.Counter)
    fields = 'sha256 group group_folder corpus original_split set label label_semantics family source relative_path size arch kind in_cohort exclude_reason'.split()
    for row in rows:
        status = states.get(row['sha256'], {})
        record = {k: row.get(k, '') for k in fields}
        record.update(status=status.get('status', 'pending'), reason=status.get('reason', ''),
                      asm_file=('groups/' + row['group_folder'] + '/' + row['sha256'] + '.asm') if status.get('status') == 'ok' else '',
                      asm_sha256=status.get('asm_sha256', ''), processor=status.get('processor', ''))
        manifest.append(record)
        groups[row['group']]['files'] += 1
        groups[row['group']][record['status']] += 1
    columns = fields + ['status', 'reason', 'asm_file', 'asm_sha256', 'processor']
    common.write_csv(out / 'manifest.csv', manifest, columns)
    for folder in set(r['group_folder'] for r in rows):
        path = out / 'groups' / folder / 'manifest.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        common.write_csv(path, [r for r in manifest if r['group_folder'] == folder], columns)
    result = dict(files=len(rows), unique_samples=len(set(r['sha256'] for r in rows)),
                  unique_statuses=dict(collections.Counter(r['status'] for r in states.values())), groups=groups)
    common.write_json(out / 'summary.json', result)
    return result


def finish_transfer(out, receipt, states):
    ack = out / 'acks' / (receipt['name'] + '.json')
    while not ack.exists():
        time.sleep(3)
    if json.loads(ack.read_text())['sha256'] != receipt['sha256']:
        raise ValueError('bad_acknowledgment')
    for state in states:
        state['transferred'] = receipt['name']
        common.write_json(out / 'status' / (state['sha256'] + '.json'), state)
        for suffix in ('.asm.gz', '.asm.gz.meta.json'):
            path = out / 'asm' / (state['sha256'] + suffix)
            if path.exists():
                path.unlink()
    (out / 'ready.json').unlink()
    (out / 'exports' / receipt['name']).unlink()


def ship(out, states, final=False):
    summarize(out)
    name = 'asm_{:06d}.tar'.format(len(list((out / 'acks').glob('*.json'))))
    index = dict(states=states, final=final)
    common.write_json(out / 'exports/index.json', index)
    package = out / 'exports' / name
    required = sum((out / 'asm' / (s['sha256'] + '.asm.gz')).stat().st_size
                   for s in states if s['status'] == 'ok') + (out / 'inventory.json').stat().st_size
    if shutil.disk_usage(str(out)).free < required + 3 * 1024**3:
        raise RuntimeError('insufficient_disk_to_package')
    with tarfile.open(str(package) + '.tmp', 'w') as archive:
        for file in ('inventory.json', 'policy.json'):
            archive.add(str(out / file), arcname=file, recursive=False)
        archive.add(str(out / 'exports/index.json'), arcname='index.json', recursive=False)
        for state in states:
            if state['status'] == 'ok':
                file = 'asm/' + state['sha256'] + '.asm.gz'
                archive.add(str(out / file), arcname=file, recursive=False)
    Path(str(package) + '.tmp').replace(package)
    receipt = dict(name=name, sha256=digest(package), bytes=package.stat().st_size,
                   expanded_bytes=sum(s.get('asm_bytes', 0) for s in states), final=final)
    common.write_json(out / 'ready.json', receipt)
    common.write_json(out / 'pipeline_state.json', dict(stage='waiting_for_verified_transfer', batch=name))
    finish_transfer(out, receipt, states)


def run(args):
    import fcntl
    out = common.check_out(args.out)
    for folder in ('asm', 'status', 'scratch', 'logs', 'exports', 'acks'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    lock = (out / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = [r for r in json.loads((args.cfg / 'inventory.json').read_text()) if r['corpus'] == 'mendeley']
    jobs = {r['sha256']: r for r in rows if r['sha256']}
    assert len(jobs) == 2670, len(jobs)
    common.write_json(out / 'inventory.json', rows)
    common.write_json(out / 'policy.json', dict(scope='mendeley_only', static_only=True,
        source='original on-disk sample, unchanged', asm_export_authorized=True,
        format='IDA OFILE_ASM', raw_executables_export=False, max_workers=1,
        no_line_or_address_caps=True, source_cfg=str(args.cfg), exporter_sha256=digest(__file__),
        combined_cfg_export=(args.cfg / 'combined_asm.json').exists()))
    if (out / 'ready.json').exists():
        receipt = json.loads((out / 'ready.json').read_text())
        with tarfile.open(str(out / 'exports' / receipt['name'])) as archive:
            index = json.load(archive.extractfile('index.json'))
        finish_transfer(out, receipt, index['states'])
    if args.preview_first and not (out / 'preview_complete.json').exists():
        preview = []
        for label in ('goodware', 'ransomware'):
            choices = []
            for row in jobs.values():
                status = args.cfg / 'status' / (row['sha256'] + '.json')
                if label in row['group_folder'] and row['kind'] == 'candidate' and status.exists():
                    if json.loads(status.read_text()).get('graph_status') == 'recovered':
                        choices.append(row)
            if not choices:
                choices = [r for r in jobs.values() if label in r['group_folder'] and r['kind'] == 'candidate' and r['size'] >= 4096]
            row = min(choices, key=lambda r: (r['size'], r['sha256']))
            common.write_json(out / 'pipeline_state.json', dict(stage='preview', sha256=row['sha256']))
            preview.append(run_job(row, out, 120, 240))
        ship(out, [s for s in preview if not s.get('transferred')])
        common.write_json(out / 'preview_complete.json', dict(states=preview))
    common.write_json(out / 'pipeline_state.json', dict(stage='waiting_for_mendeley_cfg', total=len(jobs)))
    while True:
        cfg_state = json.loads((args.cfg / 'pipeline_state.json').read_text())
        if cfg_state['stage'] == 'error':
            raise RuntimeError('CFG failed; ASM bulk pass not started')
        pending_shared = untransferred(out)
        if pending_shared:
            ship(out, pending_shared)
            common.write_json(out / 'pipeline_state.json', dict(stage='waiting_for_mendeley_cfg', total=len(jobs)))
        if cfg_state['stage'] == 'complete':
            break
        time.sleep(15)
    # Hold the original pipeline lock during bulk export to prevent overlapping IDA runs.
    cfg_lock = (args.cfg / 'pipeline.lock').open('a')
    fcntl.flock(cfg_lock, fcntl.LOCK_EX)
    pending, size = [], 0
    for index, row in enumerate(jobs.values()):
        state_path = out / 'status' / (row['sha256'] + '.json')
        if state_path.exists():
            previous = json.loads(state_path.read_text())
            # Preview uses shorter budgets. Retry its failures with the full budgets.
            if previous['status'] == 'failed' and previous.get('attempts', [{}])[0].get('budget', 3600) < 3600:
                state_path.unlink()
            elif previous.get('transferred'):
                continue
        common.write_json(out / 'pipeline_state.json', dict(stage='extracting', completed=index, total=len(jobs), sha256=row['sha256']))
        state = run_job(row, out)
        pending.append(state)
        size += state.get('asm_bytes', 0)
        if len(pending) >= 25 or size >= 256 * 1024**2:
            ship(out, pending)
            pending, size = [], 0
    ship(out, pending, final=True)
    common.write_json(out / 'pipeline_state.json', dict(stage='complete', summary=summarize(out)))


def accept(package, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for folder in ('status', 'asm'):
        (destination / folder).mkdir(exist_ok=True)
    with tarfile.open(str(package)) as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or any(not m.isfile() or not (
            m.name in ('inventory.json', 'policy.json', 'index.json') or
            re.fullmatch(r'asm/[0-9a-f]{64}\.asm\.gz', m.name)) for m in members):
            raise ValueError('unexpected_archive_member')
        index = json.load(archive.extractfile('index.json'))
        rows = json.load(archive.extractfile('inventory.json'))
        for row in rows:
            if row['corpus'] != 'mendeley' or not re.fullmatch(r'[0-9a-f]{64}', row['sha256']):
                raise ValueError('unexpected_inventory')
            if not re.fullmatch(r'mendeley/[a-zA-Z0-9_/-]+', row['group_folder']) or '..' in row['group_folder']:
                raise ValueError('unsafe_group_path')
        common.write_json(destination / 'inventory.json', rows)
        common.write_json(destination / 'policy.json', json.load(archive.extractfile('policy.json')))
        for state in index['states']:
            sha = state['sha256']
            if not re.fullmatch(r'[0-9a-f]{64}', sha) or sha not in {r['sha256'] for r in rows}:
                raise ValueError('unexpected_sample')
            if state['status'] == 'ok':
                path = destination / 'asm' / (sha + '.asm')
                temp = Path(str(path) + '.tmp')
                with archive.extractfile('asm/' + sha + '.asm.gz') as packed:
                    compressed = hashlib.sha256()
                    for block in iter(lambda: packed.read(1048576), b''):
                        compressed.update(block)
                if compressed.hexdigest() != state['compressed_sha256']:
                    raise ValueError('compressed_checksum_mismatch')
                with archive.extractfile('asm/' + sha + '.asm.gz') as packed, gzip.GzipFile(fileobj=packed) as source, temp.open('wb') as target:
                    checksum, size = text_digest(source, target)
                if checksum != state['asm_sha256'] or size != state['asm_bytes']:
                    raise ValueError('text_checksum_mismatch')
                temp.replace(path)
                for folder in set(r['group_folder'] for r in rows if r['sha256'] == sha):
                    target = destination / 'groups' / folder / path.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        try:
                            os.link(str(path), str(target))
                        except OSError:
                            shutil.copyfile(str(path), str(target))
            common.write_json(destination / 'status' / (sha + '.json'), state)
    summarize(destination)
    return index


def receive(args):
    dest = args.destination.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    key = str(Path.home() / '.ssh/vm_transfer')
    ssh = ['ssh', '-i', key, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-p', '2222', 'seed@127.0.0.1']
    scp = ['scp', '-i', key, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-P', '2222']
    remote = str(args.out).replace('\\', '/')
    if not re.fullmatch(r'/home/seed/work/out/[A-Za-z0-9_/-]+', remote):
        raise ValueError('unexpected_remote')
    while True:
        response = subprocess.run(ssh + ['if test -f ' + remote + '/ready.json; then cat ' + remote + '/ready.json; elif test -f ' + remote + '/pipeline_state.json; then cat ' + remote + '/pipeline_state.json; else echo "{}"; fi'], capture_output=True, text=True)
        if response.returncode:
            common.write_json(dest / 'transfer_status.json', dict(stage='connection_unavailable', pid=os.getpid()))
            time.sleep(30)
            continue
        receipt = json.loads(response.stdout)
        if receipt.get('stage') == 'error':
            raise RuntimeError(receipt.get('reason'))
        if 'name' not in receipt:
            common.write_json(dest / 'transfer_status.json', dict(stage=receipt.get('stage', 'waiting'), pid=os.getpid()))
            if receipt.get('stage') == 'complete':
                return
            time.sleep(15)
            continue
        name = receipt['name']
        if not re.fullmatch(r'asm_[0-9]{6}\.tar', name) or not re.fullmatch(r'[0-9a-f]{64}', receipt['sha256']):
            raise ValueError('unexpected_receipt')
        if shutil.disk_usage(str(dest)).free < 2 * receipt['bytes'] + 2 * receipt['expanded_bytes'] + 1024**3:
            raise RuntimeError('desktop_disk_reserve')
        package = dest / name
        common.write_json(dest / 'transfer_status.json', dict(stage='receiving', batch=name, pid=os.getpid()))
        subprocess.run(scp + ['seed@127.0.0.1:' + remote + '/exports/' + name, str(package)], check=True)
        if digest(package) != receipt['sha256']:
            raise ValueError('package_checksum_mismatch')
        index = accept(package, dest)
        ack = dest / 'receipts' / (name + '.json')
        ack.parent.mkdir(exist_ok=True)
        common.write_json(ack, dict(sha256=receipt['sha256'], states=index['states']))
        subprocess.run(scp + [str(ack), 'seed@127.0.0.1:' + remote + '/acks/' + name + '.json.tmp'], check=True)
        subprocess.run(ssh + ['mv ' + remote + '/acks/' + name + '.json.tmp ' + remote + '/acks/' + name + '.json'], check=True)
        package.unlink()
        common.write_json(dest / 'transfer_status.json', dict(stage='complete' if index['final'] else 'waiting', batch=name, pid=os.getpid()))
        if index['final']:
            return
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'worker', 'receive'])
    parser.add_argument('--out', type=Path, default=Path('/home/seed/work/out/ida_asm_mendeley_20261006'))
    parser.add_argument('--cfg', type=Path, default=Path('/home/seed/work/out/ida_cfg_quality_20261006/full'))
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--preview-first', action='store_true')
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sha256')
    args = parser.parse_args()
    try:
        dict(run=run, worker=worker, receive=receive)[args.command](args)
    except Exception as error:
        if args.command != 'worker':
            root = args.destination if args.command == 'receive' else args.out
            common.write_json(root / ('transfer_status.json' if args.command == 'receive' else 'pipeline_state.json'),
                              dict(stage='error', reason=str(error)))
        raise


if __name__ == '__main__':
    main()
