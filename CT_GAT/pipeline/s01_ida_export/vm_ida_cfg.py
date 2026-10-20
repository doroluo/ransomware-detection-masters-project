#!/usr/bin/env python3
"""Python 3.8: static VM-only IDAPython extraction and safe derived-data export."""
import argparse
import collections
import concurrent.futures
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

SCHEMA = 'ida-cfg-safe/1'
HASH = re.compile(r'^[0-9a-f]{64}$')
TOKEN = re.compile(r'^[a-z][a-z0-9_.]{0,39}$')
ROOTS = [
    ('mendeley_good_train', 'goodware/Goodware_Mendeley/train/goodware', 'mendeley', 'good_train', '0'),
    ('mendeley_good_test', 'goodware/Goodware_Mendeley/test/goodware_test', 'mendeley', 'good_test', '0'),
    ('mendeley_mal_train', 'ransomware/Ransomware_Mendeley/train', 'mendeley', 'mal_train', '1'),
    ('mendeley_mal_test', 'ransomware/Ransomware_Mendeley/test', 'mendeley', 'mal_test', '1'),
    ('balanced', 'goodware/Goodware_Balanced', 'goodware_balanced', 'good_train', '0'),
    ('vs', 'ransomware/Ransomware_VS', 'vs', 'mal_train', '1'),
    ('virusshare', 'ransomware/Ransomware_VirusShare/files', 'virusshare', '', ''),
]
GROUP_FOLDERS = {
    'mendeley_good_train': 'mendeley/goodware/train',
    'mendeley_good_test': 'mendeley/goodware/test',
    'mendeley_mal_train': 'mendeley/ransomware/train',
    'mendeley_mal_test': 'mendeley/ransomware/test',
    'balanced': 'balanced/goodware', 'vs': 'vs/ransomware',
    'virusshare': 'virusshare',
}


def safe_component(text):
    value = re.sub(r'[^a-zA-Z0-9_-]', '_', text)[:80]
    return value or 'root'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, separators=(',', ':')), encoding='utf-8')
    tmp.replace(path)


def write_csv(path, rows, fields):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def check_out(out):
    out = Path(out).resolve()
    permitted = (Path.home() / 'work/out').resolve()
    if permitted not in out.parents:
        raise ValueError('VM output must be a task folder under ~/work/out/')
    out.mkdir(parents=True, exist_ok=True)
    return out


def disk_guard(out):
    if shutil.disk_usage(str(out)).free < 3 * 1024**3:
        raise RuntimeError('disk_reserve_below_3_GiB')


def pe_kind(path):
    import pefile
    try:
        with Path(path).open('rb') as stream:
            if stream.read(2) != b'MZ':
                return 'not_pe', ''
        pe = pefile.PE(str(path), fast_load=True)
        try:
            arch = {0x14c: 'x86', 0x8664: 'x64'}.get(pe.FILE_HEADER.Machine, '')
            if not arch:
                return 'unsupported_machine', ''
            dirs = pe.OPTIONAL_HEADER.DATA_DIRECTORY
            if len(dirs) > 14 and dirs[14].VirtualAddress:
                return 'managed_pe', arch
            if not any(s.Characteristics & 0x20000000 and s.SizeOfRawData for s in pe.sections):
                return 'no_executable_section', arch
            return 'candidate', arch
        finally:
            pe.close()
    except Exception:
        return 'invalid_pe', ''


def inventory(out, pilot=0):
    cohorts = collections.defaultdict(list)
    cohort_rows = []
    for path in sorted((out / 'cohorts').glob('cohort_*.csv')):
        with path.open(newline='', encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                cohorts[(row['corpus'], row['sha256'])].append(row)
                cohort_rows.append(dict(row, cohort_csv=path.name))
    write_json(out / 'cohort_membership.json', cohort_rows)
    labels = {}
    label_path = Path.home() / 'ransomware/Ransomware_VirusShare/labels.csv'
    with label_path.open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            labels[row['sha256']] = row
    rows, jobs, cache = [], {}, {}
    for group, relative, corpus, split, label in ROOTS:
        root = Path.home() / relative
        if not root.is_dir():
            raise RuntimeError('missing_dataset_root: ' + str(root))
        count, candidates = 0, 0
        paths = sorted(root.rglob('*'))
        if pilot:
            paths = sorted((p for p in paths if p.is_file() and p.stat().st_size >= 4096),
                           key=lambda p: (p.stat().st_size, str(p)))
        for path in paths:
            if not path.is_file():
                continue
            row = dict(group=group, corpus=corpus, set=split, label=label,
                       family=path.parent.name, source=str(path), sha256='', size=path.stat().st_size,
                       kind='', arch='', in_cohort='', exclude_reason='', cohort_rows=[])
            relative_path = path.relative_to(root)
            row['relative_path'] = str(relative_path)
            row['original_split'] = split
            row['group_folder'] = '/'.join([GROUP_FOLDERS[group]] + [safe_component(p) for p in relative_path.parent.parts])
            row['label_semantics'] = 'is_ransomware' if group == 'virusshare' else 'goodware_0_ransomware_1'
            if path.is_symlink() or root.resolve() not in path.resolve().parents:
                row['kind'] = 'symlink_not_followed'
            else:
                try:
                    digest = sha256(path)
                    row['sha256'] = digest
                    # Inventory helper binaries too, but do not analyze them as samples.
                    if any(part.startswith('.') for part in path.relative_to(root).parts):
                        row['kind'] = 'dataset_support_file'
                    else:
                        if digest not in cache:
                            cache[digest] = pe_kind(path)
                        row['kind'], row['arch'] = cache[digest]
                    matches = cohorts.get((corpus, digest), [])
                    row['cohort_rows'] = matches
                    applicable = [r for r in matches if r['set'] == split]
                    if applicable:
                        row['in_cohort'] = '|'.join(sorted(set(r['in_cohort'] for r in applicable)))
                        row['exclude_reason'] = '|'.join(sorted(set(r['exclude_reason'] for r in applicable)))
                    if group == 'virusshare':
                        external = labels.get(digest, {})
                        row['label'] = external.get('is_ransomware', '')
                        row['family'] = external.get('family', '')
                        row['label_source'] = 'VirusShare labels.csv' if external else 'unlabelled'
                    if row['kind'] == 'candidate':
                        candidates += 1
                        jobs.setdefault(digest, dict(sha256=digest, source=str(path), arch=row['arch'], groups=[]))
                        if group not in jobs[digest]['groups']:
                            jobs[digest]['groups'].append(group)
                except Exception as error:
                    row['kind'] = 'inventory_error'
                    row['error'] = type(error).__name__
            rows.append(row)
            count += 1
            if count % 500 == 0:
                print('inventory', group, count, flush=True)
            if pilot and candidates >= pilot:
                break
        print('inventory_complete', group, count, flush=True)
    write_json(out / 'inventory.json', rows)
    write_json(out / 'jobs.json', list(jobs.values()))
    by_hash = collections.defaultdict(list)
    for row in rows:
        if row['sha256']:
            by_hash[row['sha256']].append(row)
    conflicts = []
    for digest, members in by_hash.items():
        labels_seen = sorted(set(r['label'] for r in members if r['label'] != ''))
        splits = sorted(set(r['set'] for r in members if r['set']))
        train_test = any(s.endswith('train') for s in splits) and any(s.endswith('test') for s in splits)
        if len(labels_seen) > 1 or train_test:
            conflicts.append(dict(sha256=digest, labels=labels_seen, sets=splits,
                                  label_conflict=len(labels_seen) > 1, train_test_overlap=train_test))
    write_json(out / 'split_audit.json', conflicts)
    summarize(out)


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('invalid integer')


def keys(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected.split()):
        raise ValueError('unexpected schema fields')


def validate(data, digest):
    """Strict data-only allowlist, also used on the desktop before unpacking."""
    keys(data, 'schema sha256 ida_version arch truncated functions counts')
    if data['schema'] != SCHEMA or data['sha256'] != digest or not HASH.fullmatch(digest):
        raise ValueError('schema/hash mismatch')
    if not re.fullmatch(r'[0-9]+\.[0-9]+', data['ida_version']) or data['arch'] not in ('x86', 'x64'):
        raise ValueError('invalid producer metadata')
    if type(data['truncated']) is not bool or not isinstance(data['functions'], list):
        raise ValueError('invalid graph')
    counts = dict(functions=0, blocks=0, edges=0, instructions=0, calls=0)
    functions = data['functions']
    for fid, function in enumerate(functions):
        keys(function, 'id library thunk blocks edges calls')
        if function['id'] != fid or type(function['id']) is not int:
            raise ValueError('noncanonical function ID')
        if type(function['library']) is not bool or type(function['thunk']) is not bool:
            raise ValueError('invalid function flags')
        for bid, block in enumerate(function['blocks']):
            keys(block, 'id type mnemonics operand_types')
            if block['id'] != bid or type(block['id']) is not int:
                raise ValueError('noncanonical block ID')
            integer(block['type'])
            if len(block['mnemonics']) != len(block['operand_types']):
                raise ValueError('instruction alignment mismatch')
            for mnemonic, operands in zip(block['mnemonics'], block['operand_types']):
                if not isinstance(mnemonic, str) or not TOKEN.fullmatch(mnemonic):
                    raise ValueError('invalid mnemonic token')
                if not isinstance(operands, list) or len(operands) > 8:
                    raise ValueError('invalid operand categories')
                for operand in operands:
                    integer(operand, 1)
                    if operand > 20:
                        raise ValueError('invalid operand type')
            counts['instructions'] += len(block['mnemonics'])
        for edge in function['edges']:
            if not isinstance(edge, list) or len(edge) != 2:
                raise ValueError('invalid edge')
            for endpoint in edge:
                integer(endpoint)
                if endpoint >= len(function['blocks']):
                    raise ValueError('dangling edge')
        if len(set(tuple(e) for e in function['edges'])) != len(function['edges']):
            raise ValueError('duplicate edge')
        for call in function['calls']:
            keys(call, 'block instruction targets')
            integer(call['block'])
            integer(call['instruction'])
            if call['block'] >= len(function['blocks']) or call['instruction'] >= len(function['blocks'][call['block']]['mnemonics']):
                raise ValueError('invalid call site')
            for target in call['targets']:
                integer(target)
                if target >= len(functions):
                    raise ValueError('dangling call target')
        counts['functions'] += 1
        counts['blocks'] += len(function['blocks'])
        counts['edges'] += len(function['edges'])
        counts['calls'] += len(function['calls'])
    if counts['instructions'] == 0 or counts != data['counts']:
        raise ValueError('empty graph or incorrect counts')
    return counts


def load_graph(path, digest):
    with gzip.open(str(path), 'rt', encoding='utf-8') as stream:
        data = json.load(stream)
    validate(data, digest)
    return data


def worker(binary, output, digest, arch):
    # Applied before IDA is loaded. No sample is ever a subprocess executable.
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (2500 * 1024**2, 2500 * 1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1536 * 1024**2, 1536 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    import idapro  # Must initialize the kernel before importing IDAPython.
    import ida_auto
    import ida_bytes
    import ida_funcs
    import ida_gdl
    import ida_idp
    import ida_kernwin
    import ida_ua
    import idautils
    import idc
    if sha256(binary) != digest:
        raise ValueError('worker input hash mismatch')
    opened = False
    try:
        result = idapro.open_database(str(binary), True)
        if result != 0:
            raise RuntimeError('IDA open_database code: ' + str(result))
        opened = True
        if not ida_auto.auto_wait():
            raise RuntimeError('auto-analysis interrupted')
        addresses = list(idautils.Functions())
        ids = {ea: index for index, ea in enumerate(addresses)}
        functions = []
        total_blocks, total_insns, truncated = 0, 0, False
        for ea in addresses:
            if total_blocks >= 50000 or total_insns >= 1000000:
                truncated = True
                break
            func = ida_funcs.get_func(ea)
            chart = ida_gdl.FlowChart(func, flags=ida_gdl.FC_NOEXT)
            blocks = sorted(list(chart), key=lambda block: (block.start_ea, block.id))
            remaining = 50000 - total_blocks
            if len(blocks) > remaining:
                blocks = blocks[:remaining]
                truncated = True
            selected, nodes, edges, calls = {}, [], [], []
            for block in blocks:
                if total_insns >= 1000000:
                    truncated = True
                    break
                bid = len(nodes)
                selected[block.id] = bid
                mnemonics, operand_types = [], []
                for address in idautils.Heads(block.start_ea, block.end_ea):
                    if not ida_bytes.is_code(ida_bytes.get_full_flags(address)):
                        continue
                    if total_insns >= 1000000:
                        truncated = True
                        break
                    insn = ida_ua.insn_t()
                    if not ida_ua.decode_insn(insn, address):
                        continue
                    mnemonic = (idc.print_insn_mnem(address) or '').lower()
                    if not TOKEN.fullmatch(mnemonic):
                        mnemonic = 'unknown'
                    operands = []
                    for operand in insn.ops:
                        if operand.type == ida_ua.o_void:
                            break
                        operands.append(int(operand.type))
                    if ida_idp.is_call_insn(insn):
                        targets = sorted(set(ids[t] for t in idautils.CodeRefsFrom(address, False) if t in ids))
                        calls.append(dict(block=bid, instruction=len(mnemonics), targets=targets))
                    mnemonics.append(mnemonic)
                    operand_types.append(operands)
                    total_insns += 1
                nodes.append(dict(id=bid, type=int(block.type), mnemonics=mnemonics, operand_types=operand_types))
            for block in blocks:
                if block.id in selected:
                    edges.extend([selected[block.id], selected[s.id]] for s in block.succs() if s.id in selected)
            total_blocks += len(nodes)
            functions.append(dict(id=len(functions), library=bool(func.flags & ida_funcs.FUNC_LIB),
                                  thunk=bool(func.flags & ida_funcs.FUNC_THUNK), blocks=nodes,
                                  edges=sorted(set(tuple(e) for e in edges)), calls=calls))
        # Drop call targets outside a capped graph; unresolved and external calls have [].
        for function in functions:
            function['edges'] = [list(e) for e in function['edges']]
            for call in function['calls']:
                call['targets'] = [t for t in call['targets'] if t < len(functions)]
        counts = dict(functions=len(functions), blocks=total_blocks, instructions=total_insns,
                      edges=sum(len(f['edges']) for f in functions), calls=sum(len(f['calls']) for f in functions))
        data = dict(schema=SCHEMA, sha256=digest, ida_version=ida_kernwin.get_kernel_version(),
                    arch=arch, truncated=truncated, functions=functions, counts=counts)
        if not total_insns:
            raise RuntimeError('no_recovered_instructions')
        validate(data, digest)
        tmp = Path(str(output) + '.tmp')
        with gzip.open(str(tmp), 'wt', encoding='utf-8', compresslevel=6) as stream:
            json.dump(data, stream, separators=(',', ':'))
        tmp.replace(output)
    finally:
        if opened:
            idapro.close_database(False)


def run_one(job, out, timeout, retry):
    digest = job['sha256']
    output = out / 'cfg' / (digest + '.json.gz')
    status_path = out / 'status' / (digest + '.json')
    if status_path.exists() and not retry:
        previous = json.loads(status_path.read_text())
        if previous['status'] != 'ok':
            return previous
    if output.exists():
        try:
            data = load_graph(output, digest)
            result = dict(sha256=digest, status='ok', seconds=0, counts=data['counts'], truncated=data['truncated'])
            if status_path.exists():
                result = json.loads(status_path.read_text())
            else:
                write_json(status_path, result)
            return result
        except Exception:
            output.unlink()
    started = time.monotonic()
    result = dict(sha256=digest, status='error', reason='', seconds=0)
    try:
        disk_guard(out)
        with tempfile.TemporaryDirectory(prefix=digest + '_', dir=str(out / 'scratch')) as temp:
            binary = Path(temp) / (digest + '.pe')
            shutil.copyfile(job['source'], str(binary))
            binary.chmod(0o600)
            if sha256(binary) != digest:
                raise RuntimeError('input_changed')
            with (out / 'logs' / (digest + '.log')).open('w') as log:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker',
                                            '--binary', str(binary), '--output', str(output),
                                            '--sha256', digest, '--arch', job['arch']],
                                           cwd=temp, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise RuntimeError('timeout')
                if code:
                    error_path = Path(str(output) + '.error.json')
                    if error_path.exists():
                        error = json.loads(error_path.read_text())
                        error_path.unlink()
                        raise RuntimeError(error['reason'])
                    raise RuntimeError('worker_exit_' + str(code))
            data = load_graph(output, digest)
            result.update(status='ok', counts=data['counts'], truncated=data['truncated'])
    except Exception as error:
        result['reason'] = str(error) if isinstance(error, RuntimeError) else type(error).__name__
        if output.exists():
            output.unlink()
    result['seconds'] = round(time.monotonic() - started, 3)
    write_json(status_path, result)
    return result


def run(out, jobs_count, timeout, pilot, retry):
    import fcntl
    for name in ('cfg', 'status', 'logs', 'scratch'):
        (out / name).mkdir(exist_ok=True)
    lock = (out / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(out / 'run_state.json', dict(state='running', pid=os.getpid(), started_unix=time.time()))
    summarize(out)
    jobs = json.loads((out / 'jobs.json').read_text())
    if pilot:
        picked = {}
        for group, _, _, _, _ in ROOTS:
            group_jobs = [j for j in jobs if group in j['groups']]
            # Smallest first gives a fast compatibility gate for every group.
            for job in sorted(group_jobs, key=lambda j: Path(j['source']).stat().st_size)[:pilot]:
                picked[job['sha256']] = job
        jobs = list(picked.values())
    print('run_start unique_candidates={} workers={} timeout={}'.format(len(jobs), jobs_count, timeout), flush=True)
    # Bounded submission avoids queuing the whole corpus after disk reserve binds.
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs_count) as pool:
        iterator = iter(jobs)
        pending = {}
        for _ in range(jobs_count):
            job = next(iterator, None)
            if job:
                pending[pool.submit(run_one, job, out, timeout, retry)] = job
        completed = 0
        while pending:
            finished, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in finished:
                pending.pop(future)
                result = future.result()
                completed += 1
                print(json.dumps(dict(completed=completed, total=len(jobs), **result)), flush=True)
                disk_guard(out)
                job = next(iterator, None)
                if job:
                    pending[pool.submit(run_one, job, out, timeout, retry)] = job
            if completed % 25 == 0:
                summarize(out)
    summarize(out)
    write_json(out / 'run_state.json', dict(state='complete', pid=os.getpid(), finished_unix=time.time(), candidates=len(jobs)))
    print('run_complete', flush=True)


def summarize(out):
    rows = json.loads((out / 'inventory.json').read_text())
    statuses = {p.stem: json.loads(p.read_text()) for p in (out / 'status').glob('*.json')}
    groups = {}
    manifest = []
    for row in rows:
        cohort_matches = [r for r in row.get('cohort_rows', []) if r.get('set') == row['set']]
        for key in ('tag', 'arch', 'family', 'filename', 'label'):
            row['cohort_' + key] = '|'.join(sorted(set(r.get(key, '') for r in cohort_matches)))
        status = statuses.get(row['sha256'], {})
        state = status.get('status', 'pending') if row['kind'] == 'candidate' else row['kind']
        group = groups.setdefault(row['group'], collections.Counter())
        group['files'] += 1
        group[state] += 1
        if status.get('truncated'):
            group['truncated'] += 1
        manifest.append(dict(row, status=state, reason=status.get('reason', ''),
                             truncated=status.get('truncated', ''), seconds=status.get('seconds', ''),
                             cfg=('cfg/' + row['sha256'] + '.json.gz') if state == 'ok' else ''))
    hashes = [r['sha256'] for r in rows if r['sha256']]
    summary = dict(groups=groups, files=len(rows), unique_hashes=len(set(hashes)),
                   duplicate_file_occurrences=len(hashes) - len(set(hashes)),
                   unique_candidates=len(json.loads((out / 'jobs.json').read_text())),
                   unique_statuses=dict(collections.Counter(s['status'] for s in statuses.values())),
                   unique_truncated=sum(bool(s.get('truncated')) for s in statuses.values()),
                   failure_reasons=dict(collections.Counter(s.get('reason') for s in statuses.values() if s['status'] != 'ok')),
                   updated_unix=time.time())
    write_json(out / 'summary.json', summary)
    fields = 'sha256 group group_folder corpus original_split set label label_semantics family source relative_path size arch kind cohort_arch cohort_tag cohort_family cohort_filename cohort_label in_cohort exclude_reason status reason truncated seconds cfg'.split()
    write_csv(out / 'manifest.csv', manifest, fields)
    write_csv(out / 'mendeley_samples.csv', [r for r in manifest if r['corpus'] == 'mendeley'], fields)
    memberships = json.loads((out / 'cohort_membership.json').read_text()) if (out / 'cohort_membership.json').exists() else []
    observed = {(r['corpus'], r['sha256']) for r in rows}
    expected_by_csv = collections.defaultdict(list)
    for member in memberships:
        expected_by_csv[member['cohort_csv']].append(member)
    audit = dict(cohorts={name: dict(rows=len(members),
                                    matched_rows=sum((r['corpus'], r['sha256']) in observed for r in members),
                                    missing_sha256=sorted(set(r['sha256'] for r in members if (r['corpus'], r['sha256']) not in observed)))
                         for name, members in expected_by_csv.items()})
    reference = Path.home() / 'work/tools/mendeley_sha256.txt'
    if reference.exists():
        expected_hashes = {line.strip() for line in reference.read_text().splitlines() if HASH.fullmatch(line.strip())}
        present = set(hashes)
        audit['mendeley_hash_reference'] = dict(reference_rows=len(expected_hashes), matched_any_collection=len(expected_hashes & present),
                                                missing_sha256=sorted(expected_hashes - present))
    write_json(out / 'dataset_audit.json', audit)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def package(out):
    """Never archive logs, inputs, scratch, databases, or arbitrary source files."""
    summarize(out)
    members = ['manifest.csv', 'mendeley_samples.csv', 'summary.json', 'split_audit.json', 'inventory.json', 'cohort_membership.json', 'dataset_audit.json']
    for status_path in sorted((out / 'status').glob('*.json')):
        status = json.loads(status_path.read_text())
        if status['status'] == 'ok':
            digest = status_path.stem
            path = out / 'cfg' / (digest + '.json.gz')
            load_graph(path, digest)
            members.append('cfg/' + digest + '.json.gz')
    checksums = {name: sha256(out / name) for name in members}
    write_json(out / 'checksums.json', checksums)
    if shutil.disk_usage(str(out)).free < sum((out / name).stat().st_size for name in members) + 3 * 1024**3:
        raise RuntimeError('insufficient_disk_for_package_and_3_GiB_reserve')
    tmp = out / 'safe_cfg_export.tar.tmp'
    with tarfile.open(str(tmp), 'w') as archive:
        for name in members + ['checksums.json']:
            archive.add(str(out / name), arcname=name, recursive=False)
    target = out / 'safe_cfg_export.tar'
    tmp.replace(target)
    receipt = dict(package=str(target), sha256=sha256(target), graphs=len(members) - 7,
                   bytes=target.stat().st_size)
    write_json(out / 'transfer_ready.json', receipt)
    print(json.dumps(receipt), flush=True)


def verify_package(archive_path, destination):
    """Read explicit regular-file allowlist; do not use tar.extractall()."""
    destination = Path(destination)
    with tarfile.open(str(archive_path), 'r:') as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError('duplicate archive paths')
        allowed_meta = {'checksums.json', 'manifest.csv', 'mendeley_samples.csv', 'summary.json', 'split_audit.json', 'inventory.json', 'cohort_membership.json', 'dataset_audit.json'}
        for member in members:
            if not member.isfile() or (member.name not in allowed_meta and not re.fullmatch(r'cfg/[0-9a-f]{64}\.json\.gz', member.name)):
                raise ValueError('unsafe archive member')
            if member.size > 128 * 1024**2:
                raise ValueError('oversized archive member')
        checksums = json.load(archive.extractfile('checksums.json'))
        if set(checksums) != set(names) - {'checksums.json'}:
            raise ValueError('checksum index mismatch')
        # Validate all bytes before writing any artifact.
        for name, expected in checksums.items():
            raw = archive.extractfile(name).read()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError('checksum mismatch')
            if name.startswith('cfg/'):
                validate(json.loads(gzip.decompress(raw).decode('utf-8')), Path(name).name[:64])
        for member in members:
            path = destination / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.extractfile(member).read())
    # Materialize original group/split/family layout using links to validated graphs.
    # Hard links avoid duplicate storage; copy only if the filesystem disallows links.
    grouped = collections.defaultdict(list)
    with (destination / 'manifest.csv').open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        for row in reader:
            folder = row['group_folder']
            if not re.fullmatch(r'[a-zA-Z0-9_-]+(?:/[a-zA-Z0-9_-]+)*', folder):
                raise ValueError('unsafe group folder')
            grouped[folder].append(row)
            if row['status'] == 'ok':
                digest = row['sha256']
                if not HASH.fullmatch(digest):
                    raise ValueError('invalid manifest hash')
                source = destination / 'cfg' / (digest + '.json.gz')
                target = destination / 'groups' / folder / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    try:
                        os.link(str(source), str(target))
                    except OSError:
                        shutil.copyfile(str(source), str(target))
    for folder, rows in grouped.items():
        target = destination / 'groups' / folder / 'manifest.csv'
        target.parent.mkdir(parents=True, exist_ok=True)
        write_csv(target, rows, fields)
    print('verified_graphs', sum(n.startswith('cfg/') for n in names))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inventory', 'run', 'summary', 'package', 'verify', 'worker'])
    parser.add_argument('--out', type=Path, default=Path.home() / 'work/out/ida_cfg_20261006')
    parser.add_argument('--jobs', type=int, choices=[1, 2], default=2)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--pilot', type=int, default=0, help='smallest N native PEs per group; 0 runs all')
    parser.add_argument('--retry-failures', action='store_true')
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sha256')
    parser.add_argument('--arch')
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    if args.timeout <= 0 or args.pilot < 0:
        parser.error('timeout must be positive; pilot must be nonnegative')
    if args.command == 'worker':
        try:
            worker(args.binary, args.output, args.sha256, args.arch)
        except Exception as error:
            reason = 'no_recovered_instructions' if str(error) == 'no_recovered_instructions' else type(error).__name__
            write_json(Path(str(args.output) + '.error.json'), dict(reason=reason))
            raise
    elif args.command == 'verify':
        verify_package(args.archive, args.destination)
    else:
        out = check_out(args.out)
        if args.command == 'inventory':
            inventory(out, args.pilot)
        elif args.command == 'run':
            run(out, args.jobs, args.timeout, args.pilot, args.retry_failures)
        elif args.command == 'summary':
            summarize(out)
        elif args.command == 'package':
            package(out)


if __name__ == '__main__':
    main()
