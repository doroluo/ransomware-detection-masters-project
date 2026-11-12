"""Validation-only attention grids followed by frozen three-seed confirmations."""
import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.final_paper.controlled import resource_failure
from experiments.final_paper.data import ControlledPrepared, protocol_hashes
from experiments.final_paper.trial import run_trial
from experiments.grid_study import grids
from experiments.model_grid.data import digest
from experiments.model_grid.locking import exclusive
from experiments.model_grid.runner import atomic_json, source_hashes as paper_sources

from .models import candidates, source_hashes, use_attention_factory


@contextmanager
def deadline_stop(folder, deadline):
    marker = folder / 'STOP'
    timer = None
    if deadline:
        def request_stop():
            marker.write_text('Attention-study deadline reached', encoding='utf-8')
        timer = threading.Timer(max(0, deadline - time.time()), request_stop)
        timer.daemon = True
        timer.start()
    try:
        yield
    finally:
        if timer:
            timer.cancel()


def run(args):
    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.campaign / 'CAMPAIGN.json').read_text())
    selection_path = args.campaign / 'selection' / f'fold{args.fold}' / 'selection.json'
    vocabulary_path = args.campaign / 'vocab' / f'fold{args.fold}' / 'common_api.json'
    selection = json.loads(selection_path.read_text())
    if selection['fold'] != args.fold or selection.get('outer_test_tensors_loaded') is not False:
        raise ValueError('A matching validation-only primary selection is required')
    specs = candidates(args.model, selection['specs'][str(args.model)][0],
                       args.heads, args.attention_dropouts, args.width)
    if args.smoke:
        specs = [s for s in specs if s['transformer_heads'] == specs[0]['transformer_heads']
                 and s['gat_heads'] == specs[0]['gat_heads']
                 and s['attention_dropout'] == specs[0]['attention_dropout']]
    vocabulary = json.loads(vocabulary_path.read_text())
    if vocabulary['validation_or_test_used_for_fitting'] is not False:
        raise ValueError('The API vocabulary must be fitted on training inputs only')
    if digest({k: v for k, v in vocabulary.items() if k != 'fingerprint'}) != vocabulary['fingerprint']:
        raise ValueError('Vocabulary fingerprint mismatch')
    contract = vocabulary['contract']
    if contract['fold'] != args.fold or contract['dataset'] != 'combined' or contract['val_seed'] != 0:
        raise ValueError('The API vocabulary belongs to another fold or dataset')
    sources = source_hashes()
    original_sources = paper_sources()
    plan = dict(
        model=args.model, fold=args.fold, heads=args.heads, attention_dropouts=args.attention_dropouts,
        transformer_width=args.width, screen_epochs=args.screen_epochs, confirm_epochs=args.confirm_epochs,
        patience=args.patience, seeds=args.seeds, device=args.device, smoke=args.smoke,
        screen_seed=0, feature_dropout=specs[0]['dropout'], gat_head_policy='all_layers',
        selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        vocabulary_fingerprint=vocabulary['fingerprint'], sources=sources,
        paper_sources=original_sources, paper_adapter_sources=protocol_hashes(),
        planned_screening_entries=len(specs), created_from='Frozen primary recipe for this outer fold',
    )
    path = root / 'plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Attention study configuration changed. Use a new output folder')
    atomic_json(path, plan)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    if args.deadline_epoch and time.time() >= args.deadline_epoch:
        atomic_json(root / 'status.json', dict(state='budget_exhausted'))
        return 75
    if (root / 'STOP').exists():
        raise InterruptedError(f'Resume requires reviewing {root / "STOP"}')
    if shutil.disk_usage(root).free / 1e9 < args.minimum_free_gb:
        raise RuntimeError('Disk free space is below the campaign reserve')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU-enabled PyTorch is unavailable')
    build = Path(config['builds']['signal500'])
    cache = root / 'cache'
    cache.mkdir(exist_ok=True)
    atomic_json(cache / '_paper_owned_cache.json', dict(job=str(root.resolve())))
    data = ControlledPrepared(build, Path(config['index']), 'combined', args.fold, 0,
                              args.model, specs[0], cache, vocabulary, args.smoke)
    atomic_json(root / 'dataset.json', data.provenance)

    def trial(spec, seed, stage, epochs, evaluate_test):
        if source_hashes() != sources or paper_sources() != original_sources:
            raise ValueError('Source changed during the attention study')
        if args.deadline_epoch and time.time() >= args.deadline_epoch:
            raise InterruptedError('Attention study deadline reached')
        if shutil.disk_usage(root).free / 1e9 < args.minimum_free_gb:
            raise RuntimeError('Disk free space is below the campaign reserve')
        out = root / stage / grids.trial_id(spec, epochs, seed, evaluate_test)
        if (out / 'failure.json').exists():
            return None
        atomic_json(root / 'status.json', dict(
            state=stage, pooling=spec['pooling'], seed=seed,
            attention_dropout=spec['attention_dropout'],
            transformer_heads=spec['transformer_heads'], gat_heads=spec['gat_heads'],
            trial=str(out.resolve()),
        ))
        try:
            offload = torch.autograd.graph.save_on_cpu(pin_memory=True) if device.type == 'cuda' else nullcontext()
            with use_attention_factory(), offload, sdpa_kernel(SDPBackend.MATH):
                return run_trial(args.model, spec, seed, epochs, args.patience,
                                 data, out, device, original_sources, evaluate_test=evaluate_test)
        except (MemoryError, RuntimeError, FloatingPointError) as exc:
            if not resource_failure(exc) and not isinstance(exc, FloatingPointError):
                raise
            out.mkdir(parents=True, exist_ok=True)
            atomic_json(out / 'failure.json', dict(
                state='resource_failure' if resource_failure(exc) else 'numerical_failure',
                message=str(exc), spec=spec, seed=seed, no_score_assigned=True,
            ))
            # Exit so the supervisor retries remaining candidates in a fresh process.
            return 'retry'

    with deadline_stop(root, args.deadline_epoch):
        results = []
        for spec in specs:
            result = trial(spec, 0, 'screen', args.screen_epochs, False)
            if result == 'retry':
                return 77
            if result:
                results.append(result)
        choices = grids.choose(args.model, results)
        if not choices:
            raise RuntimeError('No eligible complete configuration or mean/CLS pair')
        frozen = dict(specs=choices, fold=args.fold, model=args.model,
                      outer_test_tensors_loaded=False,
                      selection_rule='Validation ROC-AUC then macro-F1, paired mean/CLS for Model 3')
        selected = root / 'selection.json'
        if selected.exists() and json.loads(selected.read_text()) != frozen:
            raise ValueError('Selection changed after being frozen')
        atomic_json(selected, frozen)
        for spec in choices:
            for seed in args.seeds:
                result = trial(spec, seed, 'confirm', args.confirm_epochs, not bool(args.smoke))
                if result == 'retry':
                    return 77
        atomic_json(root / 'status.json', dict(
            state='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
            failures=len(list(root.glob('*/*/failure.json'))),
            outer_test_evaluated=not bool(args.smoke),
        ))
    del data
    gc.collect()
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--campaign', type=Path, required=True)
    result.add_argument('--out', type=Path, required=True)
    result.add_argument('--fold', type=int, choices=range(5), required=True)
    result.add_argument('--model', type=int, choices=(1, 2, 3), required=True)
    result.add_argument('--heads', type=int, nargs='+', default=[2, 4, 8, 12])
    result.add_argument('--attention-dropouts', type=float, nargs='+', default=[0.0, 0.1, 0.2])
    result.add_argument('--width', type=int, default=96)
    result.add_argument('--screen-epochs', type=int, default=12)
    result.add_argument('--confirm-epochs', type=int, default=40)
    result.add_argument('--patience', type=int, default=8)
    result.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    result.add_argument('--threads', type=int, default=4)
    result.add_argument('--minimum-free-gb', type=float, default=35)
    result.add_argument('--deadline-epoch', type=float)
    result.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    result.add_argument('--smoke', type=int, default=0)
    result.add_argument('--plan-only', action='store_true')
    return result


def main(argv=None):
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    for name in ('heads', 'attention_dropouts', 'seeds'):
        values = getattr(args, name)
        if not values or len(set(values)) != len(values):
            arg_parser.error(f'{name} must be nonempty and unique')
    if any(h <= 0 for h in args.heads) or any(not 0 <= d < 1 for d in args.attention_dropouts):
        arg_parser.error('Head counts must be positive and attention dropout must be in [0, 1)')
    if any(s < 0 for s in args.seeds) or min(args.width, args.screen_epochs, args.confirm_epochs,
                                           args.patience, args.threads) < 1:
        arg_parser.error('Invalid training settings')
    if args.smoke and args.smoke < 4:
        arg_parser.error('A smoke subset needs at least four samples')
    if args.minimum_free_gb < 0:
        arg_parser.error('The free-space reserve cannot be negative')
    args.out.mkdir(parents=True, exist_ok=True)
    with exclusive(args.out / 'study.lock'):
        try:
            return run(args)
        except InterruptedError as exc:
            atomic_json(args.out / 'status.json', dict(state='stopped', message=str(exc)))
            return 75


if __name__ == '__main__':
    raise SystemExit(main())
