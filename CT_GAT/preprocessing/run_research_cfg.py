#!/usr/bin/env python3
"""Resumable quality-first extraction and acknowledged, incremental safe transfer."""
import argparse
import collections
import csv
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

import ida_cfg_research as rich
import vm_ida_cfg as common

META = ['inventory.json', 'cohort_membership.json', 'split_audit.json', 'manifest.csv', 'mendeley_samples.csv', 'summary.json', 'policy.json', 'source_labels.json']


def summary(out):
    rows = json.loads((out / 'inventory.json').read_text())
    statuses = {p.stem: json.loads(p.read_text()) for p in (out / 'status').glob('*.json')}
    groups = collections.defaultdict(collections.Counter)
    manifest = []
    for source in rows:
        row = dict(source)
        state = statuses.get(row['sha256'], {})
        row['status'] = state.get('status', 'pending' if row['kind'] not in ('dataset_support_file', 'symlink_not_followed', 'inventory_error') else row['kind'])
        row['graph_status'] = state.get('graph_status', '')
        row['reason'] = state.get('reason', '')
        row['feature_file'] = 'features/' + row['sha256'] + '.jsonl.gz' if row['status'] == 'ok' else ''
        for key in ('tag', 'arch', 'family', 'filename', 'label'):
            row['cohort_' + key] = '|'.join(sorted(set(r.get(key, '') for r in row.get('cohort_rows', []) if r['set'] == row['set'])))
        groups[row['group']]['files'] += 1
        groups[row['group']][row['status']] += 1
        if row['graph_status']:
            groups[row['group']][row['graph_status']] += 1
        manifest.append(row)
    fields = 'sha256 group group_folder corpus original_split set label label_semantics family source relative_path size arch kind cohort_arch cohort_tag cohort_family cohort_filename cohort_label in_cohort exclude_reason status graph_status reason feature_file'.split()
    common.write_csv(out / 'manifest.csv', manifest, fields)
    common.write_csv(out / 'mendeley_samples.csv', [r for r in manifest if r['corpus'] == 'mendeley'], fields)
    result = dict(files=len(rows), unique_input_hashes=len(set(r['sha256'] for r in rows if r['sha256'])),
                  groups=groups, unique_statuses=dict(collections.Counter(r['status'] for r in statuses.values())),
                  graph_statuses=dict(collections.Counter(r.get('graph_status', '') for r in statuses.values())),
                  failure_reasons=dict(collections.Counter(r.get('reason', '') for r in statuses.values() if r['status'] != 'ok')),
                  updated_unix=time.time())
    common.write_json(out / 'summary.json', result)
    if (out / 'identity_evidence.json').exists():
        from reconcile_cfg_dataset import reconcile
        reconcile(out)
    return result


def initialize(out, source, pilot):
    if (out / 'policy.json').exists():
        return
    for name in ('features', 'status', 'logs', 'scratch', 'exports', 'acks'):
        (out / name).mkdir(parents=True, exist_ok=True)
    for name in ('inventory.json', 'cohort_membership.json', 'split_audit.json'):
        shutil.copyfile(str(source / name), str(out / name))
    labels = Path.home() / 'ransomware/Ransomware_VirusShare/labels.csv'
    with labels.open(newline='', encoding='utf-8-sig') as stream:
        common.write_json(out / 'source_labels.json', list(csv.DictReader(stream)))
    rows = json.loads((out / 'inventory.json').read_text())
    jobs = {}
    for row in rows:
        if row['sha256'] and row['kind'] not in ('dataset_support_file', 'symlink_not_followed', 'inventory_error'):
            jobs.setdefault(row['sha256'], row)
    if pilot:
        selected = {}
        for row in sorted(jobs.values(), key=lambda r: (r['size'], r['sha256'])):
            key = (row['group'], row['arch'], row['kind'])
            if row['size'] >= 4096 and key not in selected:
                selected[key] = row
        jobs = {r['sha256']: r for r in selected.values()}
        common.write_json(out / 'inventory.json', [r for r in rows if r['sha256'] in jobs])
    common.write_json(out / 'jobs.json', list(jobs.values()))
    common.write_json(out / 'policy.json', dict(schema=rich.SCHEMA, extractor_sha256=rich.hash_file(rich.__file__),
                                               no_block_or_instruction_caps=True, workers=1, pilot=bool(pilot),
                                               raw_bytes_export=False, full_assembly_export=False,
                                               operand_constants=True, relative_addresses=True, api_names=True,
                                               symbols=True, raw_strings=True, byte_histograms=True))
    summary(out)


def process_memory(pid):
    # Samples cannot spawn processes: only the trusted IDA host is launched.
    # Include any IDA helper children when enforcing the process-tree RSS budget.
    children, rss = collections.defaultdict(list), {}
    for path in Path('/proc').glob('[0-9]*/status'):
        try:
            fields = dict(line.split(':', 1) for line in path.read_text().splitlines() if ':' in line)
            current = int(path.parent.name)
            children[int(fields['PPid'])].append(current)
            rss[current] = int(fields.get('VmRSS', '0 kB').split()[0]) * 1024
        except (OSError, ValueError, KeyError):
            continue
    pending, total = [pid], 0
    while pending:
        current = pending.pop()
        total += rss.get(current, 0)
        pending.extend(children.get(current, []))
    return total


def available_memory():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))) * 1024


def recover_cfg_after_asm_disk_limit(out, output, digest):
    control = out / 'combined_asm.json'
    if not control.exists():
        return False
    asm_out = common.check_out(Path(json.loads(control.read_text())['out']))
    if not (asm_out / 'asm' / (digest + '.asm.gz.asm.tmp')).exists():
        return False
    # A completed CFG must validate before its interrupted assembly is deferred.
    try:
        rich.validate_file(output, digest)
    except Exception:
        return False
    from run_mendeley_asm import defer_disk_limited_assembly
    defer_disk_limited_assembly(asm_out, digest)
    return shutil.disk_usage(str(out)).free >= 3 * 1024**3


def run_job(job, out, first_timeout, retry_timeout, rss_mib, retry_failed=False):
    digest = job['sha256']
    status_path = out / 'status' / (digest + '.json')
    if status_path.exists():
        saved = json.loads(status_path.read_text())
        if saved['status'] == 'ok' or not retry_failed:
            return saved
    output = out / 'features' / (digest + '.jsonl.gz')
    if output.exists():
        data = rich.validate_file(output, digest)
        saved = dict(sha256=digest, status='ok', graph_status=data['footer']['graph_status'], counts=data['footer']['counts'],
                     artifact_sha256=rich.hash_file(output), attempts=[])
        common.write_json(status_path, saved)
        return saved
    attempts = saved.get('attempts', []) if status_path.exists() else []
    for timeout in (first_timeout, retry_timeout):
        attempt = len(attempts) + 1
        common.disk_guard(out)
        if shutil.disk_usage(str(out)).free < job['size'] + 3 * 1024**3:
            raise RuntimeError('insufficient_disk_for_private_sample_copy')
        started, peak, reason = time.monotonic(), 0, ''
        with tempfile.TemporaryDirectory(prefix=digest + '_', dir=str(out / 'scratch')) as temp:
            binary = Path(temp) / (digest + '.pe')
            shutil.copyfile(job['source'], str(binary))
            binary.chmod(0o600)
            if rich.hash_file(binary) != digest:
                raise RuntimeError('input_changed')
            with (out / 'logs' / (digest + '_' + str(attempt) + '.log')).open('w') as log:
                command = [sys.executable, str(Path(rich.__file__).resolve()), '--binary', str(binary),
                           '--output', str(output), '--sha256', digest, '--arch', job['arch'], '--kind', job['kind']]
                combined = out / 'combined_asm.json'
                if combined.exists():
                    if job.get('corpus') != 'mendeley':
                        raise ValueError('combined_export_requires_mendeley')
                    asm_out = common.check_out(Path(json.loads(combined.read_text())['out']))
                    command += ['--asm-out', str(asm_out)]
                process = subprocess.Popen(command,
                                           cwd=temp, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while process.poll() is None:
                    rss = process_memory(process.pid)
                    peak = max(peak, rss)
                    available = available_memory()
                    if time.monotonic() - started > timeout:
                        reason = 'timeout'
                    elif rss > rss_mib * 1024**2 or available < 768 * 1024**2:
                        reason = 'resident_memory_pressure'
                    elif shutil.disk_usage(str(out)).free < 3 * 1024**3:
                        reason = 'disk_reserve'
                    if reason:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        break
                    time.sleep(0.2)
                if not reason and process.returncode:
                    reason = 'worker_exit_' + str(process.returncode)
        attempts.append(dict(attempt=attempt, seconds=round(time.monotonic() - started, 3), timeout=timeout,
                             peak_rss_bytes=peak, reason=reason))
        if reason == 'disk_reserve' and recover_cfg_after_asm_disk_limit(out, output, digest):
            attempts[-1]['asm_deferred'] = 'assembly_disk_reserve'
            reason = ''
        if not reason:
            try:
                data = rich.validate_file(output, digest)
                attempts[-1]['peak_rss_bytes'] = max(peak, data['footer']['quality'].get('worker_max_rss_bytes', 0))
                result = dict(sha256=digest, status='ok', graph_status=data['footer']['graph_status'],
                              counts=data['footer']['counts'], artifact_sha256=rich.hash_file(output), attempts=attempts)
                common.write_json(status_path, result)
                return result
            except Exception as error:
                reason = 'validation_' + type(error).__name__
                attempts[-1]['reason'] = reason
        for path in (output, Path(str(output) + '.tmp')):
            if path.exists():
                path.unlink()
        if reason == 'disk_reserve':
            raise RuntimeError('disk_reserve')
        if reason not in ('timeout', 'resident_memory_pressure', 'worker_exit_-9'):
            break
    result = dict(sha256=digest, status='failed', reason=reason, attempts=attempts)
    common.write_json(status_path, result)
    return result


def finish_transfer(out, receipt, states):
    stem = receipt['name'][:-4]
    acknowledgment = out / 'acks' / (stem + '.json')
    while not acknowledgment.exists():
        time.sleep(5)
    if json.loads(acknowledgment.read_text())['sha256'] != receipt['sha256']:
        raise RuntimeError('invalid_transfer_acknowledgment')
    for state in states:
        if state['status'] == 'ok':
            path = out / 'features' / (state['sha256'] + '.jsonl.gz')
            if path.exists():
                path.unlink()
        state['transferred_batch'] = stem
        common.write_json(out / 'status' / (state['sha256'] + '.json'), state)
    (out / 'ready.json').unlink()
    (out / 'exports' / receipt['name']).unlink()


def ship(out, states, include_metadata=False, final=False):
    sequence = len(list((out / 'acks').glob('*.json')))
    stem = 'batch_{:06d}'.format(sequence)
    package = out / 'exports' / (stem + '.tar')
    index = dict(states=states, final=final, metadata=include_metadata)
    common.write_json(out / 'exports/batch_index.json', index)
    members = [(out / 'exports/batch_index.json', 'batch_index.json')]
    if include_metadata:
        summary(out)
        members += [(out / name, name) for name in META]
    for state in states:
        if state['status'] == 'ok':
            name = 'features/' + state['sha256'] + '.jsonl.gz'
            rich.validate_file(out / name, state['sha256'])
            members.append((out / name, name))
    required = sum(path.stat().st_size for path, _ in members)
    if shutil.disk_usage(str(out)).free < required + 1024**3:
        raise RuntimeError('insufficient_disk_to_package')
    with tarfile.open(str(package) + '.tmp', 'w') as archive:
        for path, name in members:
            archive.add(str(path), arcname=name, recursive=False)
    Path(str(package) + '.tmp').replace(package)
    receipt = dict(name=package.name, sha256=rich.hash_file(package), bytes=package.stat().st_size, final=final)
    common.write_json(out / 'ready.json', receipt)
    common.write_json(out / 'pipeline_state.json', dict(stage='waiting_for_verified_transfer', batch=stem))
    # Only derived files belonging to this acknowledged batch are removed.
    # Original samples, inventory and per-hash status history remain in the VM.
    finish_transfer(out, receipt, states)


def run(args):
    import fcntl
    out = common.check_out(args.out)
    lock = (out / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        common.write_json(out / 'pipeline_state.json', dict(stage='waiting_for_inventory', pid=os.getpid()))
        deadline = time.monotonic() + 10800
        while not (args.source / 'summary.json').exists():
            if time.monotonic() > deadline:
                raise RuntimeError('inventory_wait_limit')
            time.sleep(10)
        initialize(out, args.source, args.pilot)
        policy = json.loads((out / 'policy.json').read_text())
        if policy['extractor_sha256'] != rich.hash_file(rich.__file__):
            raise RuntimeError('extractor_changed_use_new_output_directory')
        policy.update(first_timeout=args.timeout, retry_timeout=args.retry_timeout, rss_mib=args.rss_mib)
        common.write_json(out / 'policy.json', policy)
        if (out / 'ready.json').exists():
            receipt = json.loads((out / 'ready.json').read_text())
            with tarfile.open(str(out / 'exports' / receipt['name'])) as archive:
                interrupted = json.load(archive.extractfile('batch_index.json'))
            finish_transfer(out, receipt, interrupted['states'])
        if not list((out / 'acks').glob('*.json')):
            ship(out, [], include_metadata=True)
        pending, pending_bytes = [], 0
        jobs = json.loads((out / 'jobs.json').read_text())
        for i, job in enumerate(jobs):
            state_path = out / 'status' / (job['sha256'] + '.json')
            if state_path.exists():
                previous = json.loads(state_path.read_text())
                if previous.get('transferred_batch') and not (args.retry_failures and previous['status'] == 'failed'):
                    continue
            common.write_json(out / 'pipeline_state.json', dict(stage='extracting', completed=i, total=len(jobs), sha256=job['sha256']))
            state = run_job(job, out, args.timeout, args.retry_timeout, args.rss_mib, args.retry_failures)
            pending.append(state)
            if state['status'] == 'ok':
                pending_bytes += (out / 'features' / (state['sha256'] + '.jsonl.gz')).stat().st_size
            print(json.dumps(dict(completed=i + 1, total=len(jobs), **state)), flush=True)
            if len(pending) >= 100 or pending_bytes >= 256 * 1024**2:
                ship(out, pending)
                pending, pending_bytes = [], 0
                summary(out)
        ship(out, pending, include_metadata=True, final=True)
        common.write_json(out / 'pipeline_state.json', dict(stage='complete', summary=summary(out)))
    except Exception as error:
        common.write_json(out / 'pipeline_state.json', dict(stage='error', reason=str(error)))
        raise


def accept(package, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(package), 'r:') as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError('duplicate_member')
        for member in members:
            if not member.isfile() or (member.name not in META + ['batch_index.json'] and not re.fullmatch(r'features/[0-9a-f]{64}\.jsonl\.gz', member.name)):
                raise ValueError('unsafe_archive_member')
        index = json.load(archive.extractfile('batch_index.json'))
        for state in index['states']:
            if not re.fullmatch(r'[0-9a-f]{64}', state['sha256']):
                raise ValueError('invalid_status_hash')
        expected = {s['sha256']: s for s in index['states'] if s['status'] == 'ok'}
        actual = {Path(name).name[:64] for name in names if name.startswith('features/')}
        if set(expected) != actual:
            raise ValueError('artifact_index_mismatch')
        for digest, state in expected.items():
            name = 'features/' + digest + '.jsonl.gz'
            h = hashlib.sha256()
            with archive.extractfile(name) as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    h.update(chunk)
            if h.hexdigest() != state['artifact_sha256']:
                raise ValueError('artifact_checksum_mismatch')
            with archive.extractfile(name) as source, gzip.GzipFile(fileobj=source) as uncompressed:
                rich.validate_stream(uncompressed, digest)
        # Validate all graphs first; write only explicit members, never extractall.
        for member in members:
            if member.name == 'batch_index.json':
                continue
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open('wb') as stream:
                shutil.copyfileobj(source, stream, 1024 * 1024)
    (destination / 'status').mkdir(exist_ok=True)
    for state in index['states']:
        common.write_json(destination / 'status' / (state['sha256'] + '.json'), state)
    rows = json.loads((destination / 'inventory.json').read_text())
    for row in rows:
        if row['sha256'] not in expected:
            continue
        folder = row['group_folder']
        if not re.fullmatch(r'[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*', folder):
            raise ValueError('unsafe_group_path')
        source = destination / 'features' / (row['sha256'] + '.jsonl.gz')
        target = destination / 'groups' / folder / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            try:
                os.link(str(source), str(target))
            except OSError:
                shutil.copyfile(str(source), str(target))
    summary(destination)
    # Every original group, including metadata-only groups, gets a manifest.
    with (destination / 'manifest.csv').open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        fields, grouped = reader.fieldnames, collections.defaultdict(list)
        for row in reader:
            grouped[row['group_folder']].append(row)
    for folder, group_rows in grouped.items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*', folder):
            raise ValueError('unsafe_group_path')
        path = destination / 'groups' / folder / 'manifest.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        common.write_csv(path, group_rows, fields)
    return index


def receive(args):
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    key = str(Path.home() / '.ssh/vm_transfer')
    ssh = ['ssh', '-i', key, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-p', '2222', 'seed@127.0.0.1']
    remote = str(args.out).replace('\\', '/')
    if not re.fullmatch(r'/home/seed/work/out/[A-Za-z0-9_/-]+', remote):
        raise ValueError('unexpected_remote_directory')
    deadline, failures = time.monotonic() + 30 * 86400, 0
    state_path = destination / 'transfer_status.json'
    common.write_json(state_path, dict(state='waiting', pid=os.getpid()))
    while time.monotonic() < deadline:
        response = subprocess.run(ssh + ['if test -f ' + remote + '/ready.json; then cat ' + remote + '/ready.json; elif test -f ' + remote + '/pipeline_state.json; then cat ' + remote + '/pipeline_state.json; else echo "{}"; fi'], capture_output=True, text=True)
        if response.returncode:
            failures += 1
            if failures >= 60:
                raise RuntimeError('connection_unavailable_60_polls')
            time.sleep(30)
            continue
        failures = 0
        receipt = json.loads(response.stdout)
        if receipt.get('stage') == 'error':
            raise RuntimeError('VM pipeline: ' + receipt.get('reason', 'error'))
        if 'name' not in receipt:
            if receipt.get('stage') == 'complete':
                common.write_json(state_path, dict(state='complete', pid=os.getpid()))
                return
            time.sleep(15)
            continue
        name = receipt['name']
        if not re.fullmatch(r'batch_[0-9]{6}\.tar', name) or not re.fullmatch(r'[0-9a-f]{64}', receipt['sha256']):
            raise ValueError('unsafe_receipt')
        package = destination / name
        if shutil.disk_usage(str(destination)).free < 2 * receipt['bytes'] + 1024**3:
            raise RuntimeError('insufficient_desktop_space_for_verified_transfer')
        common.write_json(state_path, dict(state='receiving', batch=name, pid=os.getpid()))
        subprocess.run(['scp', '-i', key, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-P', '2222',
                        'seed@127.0.0.1:' + remote + '/exports/' + name, str(package)], check=True)
        if rich.hash_file(package) != receipt['sha256']:
            raise ValueError('package_checksum_mismatch')
        index = accept(package, destination)
        stem = name[:-4]
        receipts = destination / 'receipts'
        receipts.mkdir(exist_ok=True)
        ack = receipts / (stem + '.json')
        common.write_json(ack, dict(sha256=receipt['sha256'], states=index['states']))
        subprocess.run(['scp', '-i', key, '-o', 'BatchMode=yes', '-P', '2222', str(ack),
                        'seed@127.0.0.1:' + remote + '/acks/' + stem + '.json.tmp'], check=True)
        subprocess.run(ssh + ['mv ' + remote + '/acks/' + stem + '.json.tmp ' + remote + '/acks/' + stem + '.json'], check=True)
        package.unlink()
        common.write_json(state_path, dict(state='complete' if index['final'] else 'waiting', last_batch=stem, pid=os.getpid()))
        if index['final']:
            return
        time.sleep(10)
    raise RuntimeError('receiver_30_day_wait_limit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'receive'])
    parser.add_argument('--out', type=Path, default=Path('/home/seed/work/out/ida_cfg_quality_20261006'))
    parser.add_argument('--source', type=Path, default=Path('/home/seed/work/out/ida_cfg_20261006'))
    parser.add_argument('--destination', type=Path, default=Path(__file__).resolve().parents[2] / 'reports/ida_cfg_quality_20261006')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--retry-failures', action='store_true')
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--retry-timeout', type=int, default=14400)
    parser.add_argument('--rss-mib', type=int, default=6144)
    args = parser.parse_args()
    if min(args.timeout, args.retry_timeout, args.rss_mib) <= 0:
        parser.error('budgets must be positive')
    if args.command == 'run':
        run(args)
    else:
        try:
            receive(args)
        except Exception as error:
            common.write_json(args.destination / 'transfer_status.json', dict(state='error', reason=str(error)))
            raise


if __name__ == '__main__':
    main()
