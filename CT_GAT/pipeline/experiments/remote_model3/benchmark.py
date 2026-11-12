"""Time real Model 3 training steps before committing a remote GPU to full jobs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


PROFILES = ('original', 'resident', 'batched')


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def profile_spec(base, profile):
    spec = dict(base)
    if profile not in PROFILES:
        raise ValueError(f'Unknown profile {profile}')
    if profile != 'original':
        spec.update(activation_offload=False, activation_checkpoint=False)
    if profile == 'batched':
        spec['micro_batch_size'] = spec['batch_size']
    return spec


def select_training_rows(rows, count):
    """Use training examples near the median and at the large end of each class."""
    chosen = []
    for label in (0, 1):
        ordered = sorted((r for r in rows if r['label'] == label),
                         key=lambda r: (int(r.get('num_instructions') or 0), r['sample_id']))
        if len(ordered) < count // 2:
            raise ValueError('Too few training samples for the requested diagnostic')
        for index in range(count // 2):
            position = round((len(ordered) - 1) * (0.5 + 0.5 * index / max(1, count // 2 - 1)))
            row = ordered[position]
            if row in chosen:
                row = next(r for r in reversed(ordered) if r not in chosen)
            chosen.append(row)
    return chosen


def worker(args):
    progress = args.out / f'{args.worker}.progress.json'
    started = time.perf_counter()

    def emit(phase, **values):
        message = dict(profile=args.worker, phase=phase,
                       elapsed_seconds=time.perf_counter() - started,
                       utc=datetime.now(timezone.utc).isoformat(), **values)
        atomic_json(progress, message)
        print(json.dumps(message), flush=True)

    emit('importing_training_dependencies')
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from contextlib import nullcontext
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.final_paper.canonical import file_sha256, vocabulary_lookup
    from experiments.final_paper.controlled import controlled_spec
    from experiments.final_paper.trial import make_model
    from experiments.grid_study.coverage import DiskSamples
    from experiments.model_grid.data import Prepared, digest, manifest_for, tok
    from experiments.model_grid.models import collate, forward
    from experiments.model_grid.runner import batches

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU-enabled PyTorch is unavailable')
    environment = dict(
        python=sys.version, platform=platform.platform(), torch=torch.__version__,
        hip=torch.version.hip, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        threads=args.threads,
    )
    emit('environment', **environment)
    atomic_json(args.out / 'environment.json', environment)
    selection = json.loads((args.handoff_root / 'fold3/selection.json').read_text())
    vocabulary = json.loads((args.handoff_root / 'fold3/common_api.json').read_text())
    if selection['fold'] != 3 or selection.get('outer_test_tensors_loaded') is not False:
        raise ValueError('The diagnostic requires the frozen fold 3 selection')
    if digest({k: v for k, v in vocabulary.items() if k != 'fingerprint'}) != vocabulary['fingerprint']:
        raise ValueError('API vocabulary fingerprint mismatch')
    if vocabulary['validation_or_test_used_for_fitting'] is not False:
        raise ValueError('The vocabulary must be fitted only on training samples')
    if vocabulary['contract']['fold'] != 3 or vocabulary['contract']['dataset'] != 'combined':
        raise ValueError('The vocabulary belongs to another fold or dataset')
    spec = profile_spec(controlled_spec(selection['specs']['3'][0], 3), args.worker)
    build = args.build or args.handoff_root / 'data' / args.variant
    if (build / 'processed/cfg_subgraphs').is_dir():
        build = build / 'processed/cfg_subgraphs'
    settings_path = build / 'settings.json'
    if not settings_path.exists():
        settings_path = build / 'build_info/build_settings.json'
    api_path = build / 'api_vocab.json'
    if not api_path.exists():
        api_path = build.parent / 'api_vocab.json'
    settings = json.loads(settings_path.read_text())
    excluded = settings.get('excluded', {})
    training = [row for row in manifest_for(args.handoff_root / 'index.csv', 'combined', 3, 0)['train']
                if row['sample_id'] not in excluded]
    if [row['sample_id'] for row in training] != vocabulary['contract']['train_ids']:
        raise ValueError('The dataset training split differs from the fitted vocabulary')
    selected = select_training_rows(training, args.samples)
    lookup = vocabulary_lookup(json.loads(api_path.read_text()), vocabulary['names'])

    class Inputs(Prepared):
        def __init__(self):
            self.model, self.spec, self.coverage = 3, spec, {}
            self.seen = torch.arange(tok.FIRST_API_ID, tok.FIRST_API_ID + len(vocabulary['names']))
            self.api_size = tok.FIRST_API_ID + len(vocabulary['names'])
            self.samples = DiskSamples(args.out / (args.worker + '_cache'))
            self.hashes = {}
            for row in selected:
                path = build / (row['sample_id'] + '.pt')
                self.hashes[row['sample_id']] = file_sha256(path)
                payload = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
                if int(payload['label']) != row['label']:
                    raise ValueError('Dataset label mismatch')
                ids = payload['instruction_ids'].long().clone()
                if ids.numel() and (ids[:, 3].min() < 0 or ids[:, 3].max() >= len(lookup)):
                    raise ValueError('Dataset API ID is outside the vocabulary')
                ids[:, 3] = lookup[ids[:, 3]]
                payload['instruction_ids'] = ids
                sample = self.pack(payload, row)
                self.mask(sample)
                self.samples[row['sample_id']] = sample

    emit('loading_selected_training_samples', samples=args.samples)
    tick = time.perf_counter()
    data = Inputs()
    load_seconds = time.perf_counter() - tick
    emit('training_samples_loaded', seconds=load_seconds)
    torch.manual_seed(0)
    np.random.seed(0)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(0)
        torch.cuda.reset_peak_memory_stats()
    emit('initializing_model_on_device')
    network = make_model(3, data.api_size, spec, 0).to(device)
    emit('model_ready', parameter_device=str(next(network.parameters()).device))
    network.train()
    optimizer = torch.optim.AdamW(network.parameters(), lr=spec['lr'], weight_decay=spec['weight_decay'])
    counts = np.bincount([row['label'] for row in training], minlength=2)
    class_weight = torch.tensor(len(training) / (2 * counts), dtype=torch.float32, device=device)
    measurements = []

    def timed(phase, action, **detail):
        emit(phase, **detail)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        tick = time.perf_counter()
        value = action()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        measurements.append(dict(phase=phase, seconds=time.perf_counter() - tick, **detail))
        return value

    offload = torch.autograd.graph.save_on_cpu(pin_memory=True) if spec['activation_offload'] and device.type == 'cuda' else nullcontext()
    step_times = []
    with offload, sdpa_kernel(SDPBackend.MATH):
        for step in range(args.steps):
            part = [selected[(step * spec['batch_size'] + offset) % len(selected)]
                    for offset in range(spec['batch_size'])]
            optimizer.zero_grad(set_to_none=True)
            tick = time.perf_counter()
            for micro_index, micro in enumerate(batches(part, spec['micro_batch_size'])):
                graphs = timed('load_cached_graphs',
                               lambda: [data.samples[row['sample_id']] for row in micro],
                               step=step, microbatch=micro_index)
                batch, labels = timed(
                    'collate_and_transfer', lambda: collate(3, graphs, device),
                    step=step, microbatch=micro_index,
                )

                def loss_for_batch():
                    logits = forward(3, network, batch)
                    return (functional.cross_entropy(logits, labels, reduction='none') * class_weight[labels]).sum() / len(part)

                loss = timed('forward', loss_for_batch, step=step, microbatch=micro_index)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss during diagnostic')
                timed('backward', loss.backward, step=step, microbatch=micro_index)
                del loss, batch, labels, graphs
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)
            timed('optimizer_step', optimizer.step, step=step)
            step_times.append(time.perf_counter() - tick)
            emit('logical_batch_complete', step=step, graphs=len(part), seconds=step_times[-1])
    report = dict(
        state='complete', profile=args.worker, environment=environment, specification=spec,
        parameter_device=str(next(network.parameters()).device),
        gradient_device=str(next(p.grad for p in network.parameters() if p.grad is not None).device),
        selected_training_ids=[r['sample_id'] for r in selected],
        sample_sha256=data.hashes, selected_coverage=data.coverage, data_load_seconds=load_seconds,
        training_samples=len(training), logical_batch_size=spec['batch_size'],
        micro_batch_size=spec['micro_batch_size'], step_seconds=step_times,
        estimated_training_epoch_seconds=sum(step_times) / len(step_times) * len(training) / spec['batch_size'],
        peak_gpu_allocated_mb=torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else 0,
        measurements=measurements, held_out_samples_used=False,
        estimate_scope='Selected medium/large training examples with explicit phase synchronization. Validation and checkpoint time excluded.',
    )
    atomic_json(args.out / f'{args.worker}.json', report)
    emit('complete', step_seconds=step_times)


def terminate_worker_tree(process):
    """Stop only the timed-out benchmark process and its own descendants."""
    import psutil
    try:
        parent = psutil.Process(process.pid)
        owned = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return
    for child in reversed(owned):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(owned, timeout=3)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    process.wait(timeout=5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff-root', type=Path, default=Path.home() / 'model3_handoff')
    parser.add_argument('--variant', choices=('signal250', 'signal500', 'signal750', 'signal1000',
                                             'callwalk250', 'callwalk500', 'callwalk750', 'callwalk1000'),
                        default='signal250')
    parser.add_argument('--build', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--profiles', nargs='+', choices=PROFILES, default=list(PROFILES))
    parser.add_argument('--worker', choices=PROFILES, help=argparse.SUPPRESS)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args(argv)
    if args.samples < 4 or args.samples % 2 or min(args.steps, args.threads, args.timeout) <= 0:
        parser.error('Use an even sample count of at least four and positive steps, threads, and timeout')
    if len(set(args.profiles)) != len(args.profiles):
        parser.error('Profiles must be unique')
    args.out = args.out or args.handoff_root / 'throughput_check' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    args.out.mkdir(parents=True, exist_ok=True)
    if args.worker:
        try:
            worker(args)
            return 0
        except Exception as exc:
            atomic_json(args.out / f'{args.worker}.json',
                        dict(state='failed', profile=args.worker, error=type(exc).__name__, message=str(exc)))
            raise
    if any((args.out / f'{profile}.json').exists() for profile in args.profiles):
        parser.error('Use a new output folder to preserve the previous diagnostic reports')
    summary = []
    for profile in args.profiles:
        command = [
            sys.executable, '-u', '-m', 'experiments.remote_model3.benchmark',
            '--worker', profile, '--handoff-root', str(args.handoff_root),
            '--variant', args.variant, '--out', str(args.out), '--device', args.device,
            '--samples', str(args.samples), '--steps', str(args.steps), '--threads', str(args.threads),
        ]
        if args.build:
            command += ['--build', str(args.build)]
        print(f'Timing {profile}, timeout {args.timeout:g} seconds', flush=True)
        with (args.out / f'{profile}.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(
                command, cwd=Path(__file__).resolve().parents[2], stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                terminate_worker_tree(process)
                last_path = args.out / f'{profile}.progress.json'
                last = json.loads(last_path.read_text()) if last_path.exists() else None
                atomic_json(args.out / f'{profile}.json', dict(
                    state='timed_out', profile=profile, timeout_seconds=args.timeout, last_progress=last))
                code = 124
        report_path = args.out / f'{profile}.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else dict(
            state='worker_exited_without_report', profile=profile, exit_code=code)
        summary.append(report)
        print(json.dumps({key: report[key] for key in ('profile', 'state', 'step_seconds',
                                                       'parameter_device', 'gradient_device') if key in report}), flush=True)
    atomic_json(args.out / 'summary.json', summary)
    print(f'Return summary.json, environment.json, and the logs under {args.out}', flush=True)
    return 0 if any(r['state'] == 'complete' for r in summary) else 1


if __name__ == '__main__':
    raise SystemExit(main())
