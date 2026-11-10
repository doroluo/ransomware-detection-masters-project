"""Durable single-GPU paper campaign with bounded CPU preparation and checkpoint stops."""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import psutil

from experiments.model_grid import runner
from experiments.model_grid.data import digest
from experiments.model_grid.locking import exclusive

from .canonical import file_sha256
from .plan import counts, jobs_for, validate_plan
from .report import publish


PIPELINE = Path(__file__).resolve().parents[2]
DONE = {'complete', 'complete_with_resource_limits', 'complete_with_trial_failures'}
TERMINAL = DONE | {'failed', 'resource_infeasible', 'blocked', 'skipped'}


def source_snapshot():
    return dict(scientific=runner.source_hashes(),
        campaign={p.name: file_sha256(p) for p in Path(__file__).parent.glob('*.py')},
        utility_scripts={p.name: file_sha256(p) for p in Path(__file__).parent.glob('*.ps1')})


def apply_job_skips(config, state):
    requested = config.get('skip_jobs', {})
    if any(state['jobs'][name]['state'] == 'running' for name in requested):
        raise ValueError('Cannot skip a running job; stop it at a checkpoint first')
    for name, reason in requested.items():
        record = state['jobs'][name]
        if record['state'] in ('pending', 'skipped'):
            record.update(state='skipped', reason=reason)
            record.setdefault('skipped_utc', datetime.now(timezone.utc).isoformat())


def save_state(root, state):
    state['generation'] = state.get('generation', 0) + 1
    payload = json.dumps(state, sort_keys=True, allow_nan=False)
    journal = Path(root) / 'state.journal'; journal.mkdir(exist_ok=True)
    path = journal / f'{state["generation"]:012d}_{uuid.uuid4().hex}.json'
    with path.open('x', encoding='utf-8') as handle:
        json.dump(dict(payload=payload, sha256=hashlib.sha256(payload.encode()).hexdigest()), handle)
        handle.flush(); os.fsync(handle.fileno())
    try:
        runner.atomic_json(Path(root) / 'STATUS.json', state)
    except PermissionError:
        print('Status display locked; durable journal retained', flush=True)
    for old in sorted(journal.glob('*.json'))[:-8]:
        try: old.unlink()
        except OSError: pass


def load_state(root):
    root = Path(root); candidates = []
    for path in [root / 'STATUS.json', root / 'STATUS.json.tmp', *list((root / 'state.journal').glob('*.json'))]:
        try:
            value = json.loads(path.read_text())
            if 'payload' in value:
                if hashlib.sha256(value['payload'].encode()).hexdigest() != value['sha256']:
                    continue
                value = json.loads(value['payload'])
            if isinstance(value.get('jobs'), dict):
                candidates.append((value.get('generation', 0), path.stat().st_mtime_ns, value))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if candidates:
        return max(candidates, key=lambda v: v[:2])[2]
    if (root / 'STATUS.json').exists() or list((root / 'state.journal').glob('*.json')):
        raise RuntimeError('No valid campaign state; refusing to reset progress')
    return None


def owned_cache(path, root):
    path, base = Path(path).resolve(), (Path(root) / '_cache').resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError('Refusing to remove a path outside the campaign cache')
    marker = path / '_paper_owned_cache.json'
    if not marker.is_file():
        raise ValueError('Cache ownership marker missing')
    job = Path(json.loads(marker.read_text())['job']).resolve()
    if not job.is_relative_to(Path(root).resolve()) or job == Path(root).resolve():
        raise ValueError('Cache marker names an unrelated job')
    return path


def cleanup_cache(job, root):
    args = job['args']
    if '--cache' not in args:
        return
    path = Path(args[args.index('--cache') + 1])
    if path.exists():
        shutil.rmtree(owned_cache(path, root))


def prune_completed_checkpoints(job, root):
    """Keep final best weights and interrupted state; compact completed screening artifacts."""
    folder = Path(job['out']).resolve(); base = Path(root).resolve()
    if folder == base or not folder.is_relative_to(base):
        raise ValueError('Unowned checkpoint directory')
    chosen = []
    if job['module'].endswith('.selection'):
        selected = folder / 'selection.json'
        if not selected.exists():
            return
        chosen = [s for v in json.loads(selected.read_text())['specs'].values() for s in v]
    deleted_bytes, deleted = 0, []
    for metrics in folder.rglob('metrics.json'):
        config_path = metrics.parent / 'config.json'
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text()); result = json.loads(metrics.read_text())
        if result.get('fingerprint') != digest(config):
            raise ValueError('Refusing checkpoint cleanup for an invalid result contract')
        candidates = [metrics.parent / 'last.pt']
        if chosen and config['spec'] not in chosen:
            candidates.append(metrics.parent / 'best.pt')
        for path in candidates:
            if path.is_file():
                resolved = path.resolve()
                if not resolved.is_relative_to(folder):
                    raise ValueError('Checkpoint resolves outside the owned job')
                deleted_bytes += path.stat().st_size
                path.unlink(); deleted.append(str(path.relative_to(folder)))
    runner.atomic_json(folder / 'checkpoint_retention.json', dict(
        policy='All configurations, histories and predictions retained. Final best weights and interrupted checkpoints retained. '
               'Completed optimizer/resume states and unselected completed screening weights may be removed.',
        removed_files=deleted, removed_bytes=deleted_bytes))


def success_state(job):
    out = Path(job['out'])
    if job['success'] == 'four_build_summaries':
        for sampler in ('sig', 'callwalk'):
            for cap in (750, 1000):
                path = out / f'b_{sampler}{cap}_combined_v3' / 'build_summary.json'
                if not path.exists():
                    return None
                doc = json.loads(path.read_text())
                if doc.get('complete') is not True or doc.get('processed') != 5618:
                    return None
        return 'complete'
    if job['success'] == 'common_api.json':
        path = out / 'common_api.json'
        if path.exists():
            doc = json.loads(path.read_text())
            if doc.get('validation_or_test_used_for_fitting') is False:
                return 'complete'
        return None
    status = out / 'status.json'
    if not status.exists():
        return None
    value = json.loads(status.read_text()).get('state')
    plan_path = out / 'plan.json'
    if value in DONE and job['module'].endswith('.controlled') and plan_path.exists():
        from .data import protocol_hashes
        if json.loads(plan_path.read_text()).get('adapter_sources') != protocol_hashes():
            raise ValueError('Completed job uses a different scientific adapter version')
    if job['success'] == 'selection_complete' and value == 'selection_complete':
        return 'complete'
    return value if value in DONE else None


def matching_process(pid, job):
    try:
        proc = psutil.Process(pid)
        command = ' '.join(proc.cmdline())
        expected = job['module']
        return proc.is_running() and expected in command and str(job['out']) in command
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def adopted_launch(config, job):
    root = Path(config['root'])
    path = {'select_f0': root / 'selection_fold0_launch.json',
            'build_caps': root / 'preparation' / 'builds_launch.json',
            'vocabulary_f0': root / 'preparation' / 'vocabulary_fold0_launch.json'}.get(job['id'])
    if path and path.exists():
        doc = json.loads(path.read_text(encoding='utf-8-sig'))
        pid = doc['launcher_pid']
        if matching_process(pid, job):
            return pid
    return None


def discover_process(job):
    matches = []
    for proc in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline']):
        try:
            command = ' '.join(proc.info['cmdline'] or [])
            if proc.info['name'].lower() == 'python.exe' and job['module'] in command and str(job['out']) in command:
                matches.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    parents = {p['ppid'] for p in matches}
    leaves = [p['pid'] for p in matches if p['pid'] not in parents]
    if len(leaves) > 1:
        raise RuntimeError('Multiple independent writers found for one campaign job')
    return leaves[0] if leaves else None


def request_stop(job, tag):
    out = Path(job['out'])
    targets = [out / f'model{m}' / 'STOP' for m in (1, 2, 3)] if job['module'].endswith('.selection') else [out / 'STOP']
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open('x', encoding='utf-8') as handle:
                handle.write(tag)
        except FileExistsError:
            pass


def mirror_reports(config):
    destination = config.get('output_reports')
    if not destination or os.name != 'nt':
        return
    command = Path(__file__).with_name('mirror_reports.ps1')
    result = subprocess.run([config.get('powershell', 'powershell.exe'), '-NoProfile', '-NonInteractive', '-File', str(command),
        '-SourceRoot', str(Path(config['root']) / 'reports'), '-DestinationRoot', destination],
        capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError('Report mirror failed: ' + result.stderr[-1500:])


def write_plan_document(config, root):
    c = counts(config)
    lines = ['# Final-paper experiment plan', '',
        f'First checkpoint: {config["meeting_date"]}. Extended target: {config["extended_date"]}. Paper due: {config["paper_due"]}.',
        'The PC may train overnight. One GPU job runs at a time; at most two CPU preparation jobs overlap it.',
        f'The campaign requests a checkpoint stop after {config["maximum_hours"]} elapsed hours; an in-progress epoch may finish.', '',
        '## Primary search and confirmation', '',
        'Each of five grouped outer folds gets its own validation-only search. No outer-test score chooses a configuration.',
        'Models 1 and 3 train their position-aware instruction Transformers end to end. Model 2 is the 77-feature CFG GAT.',
        'Model 3 mean/CLS pooling is paired under otherwise identical hyperparameters and seeds.', '',
        '- Optimizer: five learning rates × four dropout values × three weight decays.',
        '- Model 1 architecture: width 32/64/128 × Transformer depth 1/2/4 × sequence limit 128/512/1024.',
        '- Model 2 architecture: GAT hidden 64/128/256 × depth 1/2/3/4 × heads 2/4/8.',
        '- Model 3 architecture: width 32/64/128 × Transformer depth 1/2/3 × GAT hidden 64/128/256 × GAT depth 1/2/3 × mean/CLS.',
        '- Model 3 instruction limit: 32/64/128/256 real instructions per block × paired pooling.',
        '- Additional attention-head and batch-size sweeps follow the selected architecture. These are staged grids, not one product of every axis.',
        f'- {c["staged_selection_entries_total"]} stage entries across five folds; exact compatible duplicates reuse recorded results.',
        '- Screening: 12 epochs, seed 0, validation AUC then macro-F1 at threshold 0.5.',
        '- Confirmation: seeds 0/1/2, up to 40 epochs, patience 8. Thresholds fit on validation only.', '',
        '## Controlled representation comparisons', '',
        '- Signal and callwalk at 250, 500, 750 and 1,000 blocks; shared uncapped eligible-block input.',
        '- Same 5,618 samples and 34 explicit exclusions, with identical split IDs across representations.',
        '- Semantic API IDs are mapped into one common vocabulary fitted on each fold training representations only.',
        '- Fold-selected primary recipes are transferred unchanged within this separate series. Results are not mixed with the original-vocabulary primary series.',
        '- Logical batch size stays fixed. Graph microbatches of one accumulate sample-weighted gradients before an optimizer step.',
        '- The math attention backend, CPU activation offload and Model 3 activation checkpointing are recorded and held fixed across compared graph sizes.',
        f'- Up to {c["representation_confirmations"]} seed/fold/representation confirmations; infeasible runs receive no accuracy score.', '',
        '## Ablations', '',
        '- Models 1/3: API identity removal, behavior-channel removal and deterministic within-input instruction shuffling.',
        '- Model 2: API-kind count removal, behavior-count removal and edge-free graph control.',
        '- Model 3 also receives the edge-free graph control. Self loops retained by the GAT remain part of that model.',
        '- All ablations use the canonical 500-block signal baseline and the same frozen recipe; they do not retune on test.',
        f'- Up to {c["ablation_confirmations"]} ablation confirmations.', '',
        '## Reporting and limits', '',
        'Report accuracy, ROC-AUC, PR-AUC, macro-F1, detection rate, FPR and confusion counts; preserve per-sample predictions.',
        'Per-fold seed variation and pooled out-of-fold metrics have separate labels. Five folds are never treated as fifteen independent datasets.',
        'OOF decisions use each fold validation-fitted threshold. Optional class-stratified family/identity bootstrap intervals are conditional on trained models.',
        'These folds already have historical experiments; this campaign does not create a fresh external holdout.',
        'The first week prioritizes full primary grids/confirmations and the first four representation comparisons. Larger caps, uncapped feasibility and ablations follow.',
        'Memory/numerical failures and unexecuted dependent cases remain visible. No samples are silently removed to make a trial fit.',
        'All configurations, histories, predictions and final best weights are retained. Completed screening optimizer states/unselected weights may be compacted; interrupted checkpoints remain resumable.',
        'Packed working caches are temporary and removable only inside the marked campaign cache directory. Original datasets and historical results are preserved.', '']
    if config.get('skip_jobs'):
        lines += ['## Schedule exclusions', '',
            f'{len(config["skip_jobs"])} jobs excluded from dispatch. '
            f'{c["skipped_representation_confirmations"]} representation confirmations removed from the original schedule.',
            *[f'- {name}: {reason}' for name, reason in sorted(config['skip_jobs'].items())], '']
    reports = root / 'reports'; reports.mkdir(parents=True, exist_ok=True)
    (reports / 'EXPERIMENT_PLAN.md').write_text('\n'.join(lines), encoding='utf-8')
    runner.atomic_json(reports / 'PLAN.json', dict(config=config, counts=c))


def run(config, once=False):
    root = Path(config['root']); root.mkdir(parents=True, exist_ok=True)
    jobs = jobs_for(config); validate_plan(config, jobs)
    expected_sources = config['sources']
    if source_snapshot() != expected_sources:
        raise ValueError('Campaign source changed; use a reviewed new campaign rather than mix implementations')
    by_id = {j['id']: j for j in jobs}
    state = load_state(root) or dict(
        state='running', jobs={j['id']: dict(state='pending', attempts=0) for j in jobs},
        started_utc=config['started_utc'], stop_tag='campaign:' + uuid.uuid4().hex)
    if set(state['jobs']) != set(by_id):
        raise ValueError('Campaign task graph changed')
    processes = {}
    for job in jobs:
        record = state['jobs'][job['id']]
        if record.get('state') == 'running' and matching_process(record.get('pid', 0), job):
            processes[job['id']] = None
            continue
        adopted = adopted_launch(config, job) or discover_process(job)
        if adopted:
            record.update(state='running', pid=adopted, adopted=True, started=time.time())
            processes[job['id']] = None
        elif success_state(job):
            record.update(state=success_state(job))
        elif record.get('state') == 'running':
            record.update(state='pending', note='Previous process exited before a complete marker; resume checkpoints')
    apply_job_skips(config, state)
    write_plan_document(config, root)
    last_report = 0.; stopping = False
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        while True:
            elapsed = (time.time() - config['started_epoch']) / 3600
            state['elapsed_hours'] = elapsed
            if (root / 'STOP').exists() or elapsed >= config['maximum_hours']:
                stopping = True
                state['state'] = 'stopping_at_checkpoint'
                for name in processes:
                    request_stop(by_id[name], state['stop_tag'])
            for name in list(processes):
                job, record = by_id[name], state['jobs'][name]
                proc = processes[name]
                live = proc.poll() is None if proc is not None else matching_process(record['pid'], job)
                if live:
                    continue
                code = proc.returncode if proc is not None else None
                del processes[name]
                complete = success_state(job)
                if complete:
                    record.update(state=complete, finished=time.time())
                    if job['module'].endswith(('.selection', '.controlled')):
                        prune_completed_checkpoints(job, root)
                    cleanup_cache(job, root)
                elif stopping:
                    record.update(state='pending', note='Checkpoint stop; rerun the same campaign to resume')
                elif code == 77 and record.get('attempts', 0) < 8:
                    record.update(state='pending', note='Retry remaining cases in a fresh process; failed cases are preserved')
                    cleanup_cache(job, root)
                elif code == 78 or (Path(job['out']) / 'preparation_failure.json').exists():
                    record.update(state='resource_infeasible', exit_code=code)
                    cleanup_cache(job, root)
                else:
                    record.update(state='failed', exit_code=code, note='Inspect job log; no result substituted')
            for job in jobs:
                record = state['jobs'][job['id']]
                if record['state'] != 'pending':
                    continue
                if any(state['jobs'][d]['state'] in TERMINAL - DONE for d in job['dependencies']):
                    record.update(state='blocked', note='Required job did not complete')
            if not stopping:
                for role, maximum in (('gpu', 1), ('cpu', 2)):
                    active = sum(by_id[name]['role'] == role for name in processes)
                    for job in jobs:
                        if active >= maximum:
                            break
                        record = state['jobs'][job['id']]
                        if job['role'] != role or record['state'] != 'pending':
                            continue
                        if not all(state['jobs'][d]['state'] in DONE for d in job['dependencies']):
                            continue
                        free = shutil.disk_usage(root).free / 1e9
                        if free < config['minimum_free_gb']:
                            stopping = True; state['state'] = 'storage_reserve_reached'
                            for name in processes:
                                request_stop(by_id[name], state['stop_tag'])
                            break
                        logs = root / 'logs'; logs.mkdir(exist_ok=True)
                        out = (logs / f'{job["id"]}.log').open('a', encoding='utf-8')
                        err = (logs / f'{job["id"]}.stderr.log').open('a', encoding='utf-8')
                        proc = subprocess.Popen([config['python'], '-u', '-m', job['module'], *job['args']],
                            cwd=config['pipeline'], stdout=out, stderr=err,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                        out.close(); err.close()
                        record.update(state='running', pid=proc.pid, started=time.time(),
                            attempts=record.get('attempts', 0) + 1, adopted=False)
                        processes[job['id']] = proc; active += 1
            checkpoint = elapsed >= config['checkpoint_hours'] and not state.get('meeting_checkpoint_published')
            finished = not processes and all(v['state'] in TERMINAL for v in state['jobs'].values())
            if finished:
                state['state'] = 'complete' if all(v['state'] in DONE | {'skipped'} for v in state['jobs'].values()) else 'finished_with_limits_or_failures'
            if stopping and not processes:
                state['state'] = 'stopped_at_budget' if elapsed >= config['maximum_hours'] else 'stopped'
            save_state(root, state)
            if time.monotonic() - last_report > config['report_interval_seconds'] or checkpoint or finished or stopping:
                try:
                    summary = publish(config, state, bootstrap=config['bootstrap_repeats'] if checkpoint or finished or (stopping and not processes) else 0)
                    state['last_report'] = summary; state.pop('report_error', None)
                    if checkpoint:
                        state['meeting_checkpoint_published'] = datetime.now(timezone.utc).isoformat()
                    mirror_reports(config)
                except Exception as exc:
                    state['report_error'] = str(exc)
                    print('Report error:', exc, flush=True)
                last_report = time.monotonic()
                save_state(root, state)
            if once or finished or (stopping and not processes):
                break
            time.sleep(20)
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--config', type=Path, help='Initialize from a reviewed JSON configuration')
    p.add_argument('--initialize-only', action='store_true')
    p.add_argument('--resume', action='store_true', help='Explicitly clear the campaign STOP and its own checkpoint-stop tags')
    p.add_argument('--once', action='store_true', help='One supervisory pass for integration validation; leaves workers running')
    a = p.parse_args(argv)
    a.root.mkdir(parents=True, exist_ok=True)
    target = a.root / 'CAMPAIGN.json'
    if a.config:
        config = json.loads(a.config.read_text(encoding='utf-8-sig'))
        config.update(root=str(a.root.resolve()), pipeline=str(PIPELINE), sources=source_snapshot())
        if target.exists() and json.loads(target.read_text()) != config:
            raise ValueError('Existing campaign configuration is immutable')
        validate_plan(config, jobs_for(config))
        runner.atomic_json(target, config)
        write_plan_document(config, a.root)
    elif not target.exists():
        p.error('Initialize a campaign with --config first')
    config = json.loads(target.read_text())
    os.environ.update(config.get('environment', {}))
    if a.initialize_only:
        print(json.dumps(counts(config), indent=2)); return
    with exclusive(a.root / 'campaign.lock'):
        if a.resume:
            saved = load_state(a.root)
            stop = a.root / 'STOP'
            if stop.exists(): stop.unlink()
            if saved:
                for job in jobs_for(config):
                    folder = Path(job['out'])
                    targets = [folder / f'model{m}' / 'STOP' for m in (1, 2, 3)] if job['module'].endswith('.selection') else [folder / 'STOP']
                    for target in targets:
                        if target.is_file() and target.read_text() == saved['stop_tag']:
                            target.unlink()
        run(config, once=a.once)


if __name__ == '__main__':
    main()
