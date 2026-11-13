"""Frozen-recipe confirmations and controlled representation/ablation jobs."""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.model_grid import runner
from experiments.model_grid.data import Prepared, digest
from experiments.model_grid.locking import exclusive
from experiments.grid_study import grids, study

from .data import ABLATIONS, ControlledPrepared, protocol_hashes
from .trial import run_trial


def resource_failure(exc):
    if isinstance(exc, (torch.OutOfMemoryError, MemoryError)):
        return True
    text = str(exc).lower()
    return isinstance(exc, RuntimeError) and any(s in text for s in
        ('out of memory', 'cannot allocate memory', "can't allocate memory", 'not enough memory', 'defaultcpuallocator'))


def controlled_spec(spec, model, ablation='none'):
    return dict(spec, micro_batch_size=spec['batch_size'] if model == 1 else 1,
        attention_backend='math', activation_checkpoint=model == 3,
        activation_offload=True, ablation=ablation,
        paper_adapter_sources=protocol_hashes())


def execution_spec(spec, model, ablation, profile):
    result = controlled_spec(spec, model, ablation)
    if profile == 'gpu-resident':
        if model != 3:
            raise ValueError('GPU-resident profile is qualified only for Model 3')
        result.update(activation_checkpoint=False, activation_offload=False,
                      execution_profile=profile)
    elif profile != 'original':
        raise ValueError('Unknown execution profile')
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=int, choices=[1, 2, 3], required=True)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--vocabulary', type=Path)
    p.add_argument('--mode', choices=['primary', 'controlled'], default='controlled')
    p.add_argument('--ablation', default='none')
    p.add_argument('--dataset', default='combined')
    p.add_argument('--fold', type=int, choices=range(5), required=True)
    p.add_argument('--val-seed', type=int, default=0)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cache', type=Path)
    p.add_argument('--reuse-from', nargs='*', type=Path, default=[])
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--smoke', type=int, default=0)
    p.add_argument('--execution-profile', choices=['original', 'gpu-resident'], default='original')
    p.add_argument('--qualification', action='store_true', help='One full training/validation epoch, mean pooling, seed 0, no test evaluation')
    p.add_argument('--progress-seconds', type=float, default=0)
    a = p.parse_args(argv)
    if a.progress_seconds < 0:
        p.error('Progress interval must be nonnegative')
    if (a.execution_profile != 'original' or a.qualification) and (a.mode != 'controlled' or a.model != 3):
        p.error('Remote execution and qualification require controlled Model 3')
    if a.qualification and (a.epochs != 1 or a.seeds != [0] or a.smoke):
        p.error('Qualification requires --epochs 1 --seeds 0 and the full dataset (no --smoke)')
    evaluate_test = not bool(a.smoke) and not a.qualification
    if a.ablation != 'none' and a.ablation not in ABLATIONS[a.model]:
        p.error('Ablation is not defined for this model')
    if a.mode == 'primary' and (a.ablation != 'none' or a.vocabulary):
        p.error('Primary confirmation preserves the original 500-block scientific protocol')
    if a.mode == 'controlled' and (a.vocabulary is None or a.cache is None):
        p.error('Controlled jobs require a fitted vocabulary and an owned disk cache')
    if len(a.seeds) != len(set(a.seeds)) or any(s < 0 for s in a.seeds):
        p.error('Seeds must be unique and nonnegative')
    torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    selection = json.loads(a.selection.read_text())
    if selection['fold'] != a.fold or selection.get('val_seed', 0) != a.val_seed or selection.get('outer_test_tensors_loaded') is not False:
        raise ValueError('Selection must belong to this outer fold and use validation only')
    if a.mode == 'primary':
        settings = json.loads((a.build / 'settings.json').read_text())
        if settings.get('sample') != 'signal' or settings.get('num_blocks') != 500:
            raise ValueError('Primary confirmation requires the original signal 500-block representation')
    base_specs = selection['specs'][str(a.model)]
    if a.qualification:
        base_specs = [spec for spec in base_specs if spec['pooling'] == 'mean']
        if len(base_specs) != 1:
            raise ValueError('Qualification requires exactly one frozen mean-pooling recipe')
    sources = runner.source_hashes()
    vocabulary = None
    if a.mode == 'controlled':
        vocabulary = json.loads(a.vocabulary.read_text())
        if digest({k: v for k, v in vocabulary.items() if k != 'fingerprint'}) != vocabulary['fingerprint']:
            raise ValueError('Common vocabulary fingerprint mismatch')
        contract = vocabulary['contract']
        if contract['fold'] != a.fold or contract['val_seed'] != a.val_seed or contract['dataset'] != a.dataset:
            raise ValueError('Common vocabulary belongs to a different split')
        if vocabulary.get('validation_or_test_used_for_fitting') is not False:
            raise ValueError('Vocabulary must be fitted without validation/test payloads')
    a.out.mkdir(parents=True, exist_ok=True)
    plan = dict(mode=a.mode, model=a.model, fold=a.fold, ablation=a.ablation, seeds=a.seeds,
        epochs=a.epochs, patience=a.patience, device=a.device, smoke=a.smoke,
        build=str(a.build.resolve()), selection_sha256=hashlib.sha256(a.selection.read_bytes()).hexdigest(),
        scientific_sources=sources, adapter_sources=protocol_hashes(),
        common_vocabulary_fingerprint=vocabulary['fingerprint'] if vocabulary else None)
    if a.execution_profile != 'original' or a.qualification or a.progress_seconds:
        plan.update(execution_profile=a.execution_profile, qualification=a.qualification,
                    progress_seconds=a.progress_seconds, evaluate_test=evaluate_test)
    with exclusive(a.out / 'run.lock'):
        path = a.out / 'plan.json'
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError('Confirmation job plan changed')
        runner.atomic_json(path, plan)
        runner.atomic_json(a.out / 'status.json', dict(state='preparing', **plan))
        data = None
        records = []
        reuse = study.reuse_candidates(a.reuse_from) if a.mode == 'primary' else []
        try:
            first = base_specs[0] if a.mode == 'primary' else execution_spec(base_specs[0], a.model, a.ablation, a.execution_profile)
            if a.mode == 'primary':
                data = Prepared(a.build, a.index, a.dataset, a.fold, a.val_seed, a.model, first, a.smoke)
            else:
                a.cache.mkdir(parents=True, exist_ok=True)
                runner.atomic_json(a.cache / '_paper_owned_cache.json', dict(job=str(a.out.resolve())))
                data = ControlledPrepared(a.build, a.index, a.dataset, a.fold, a.val_seed, a.model,
                    first, a.cache, vocabulary, a.smoke)
            runner.atomic_json(a.out / 'dataset.json', data.provenance)
            coverage_path = a.out / 'coverage.json'
            previous_coverage = json.loads(coverage_path.read_text()) if coverage_path.exists() else {}
            previous_coverage.update(data.coverage)
            runner.atomic_json(coverage_path, previous_coverage)
            for base in base_specs:
                if protocol_hashes() != plan['adapter_sources']:
                    raise ValueError('Scientific adapter source changed during job')
                spec = base if a.mode == 'primary' else execution_spec(base, a.model, a.ablation, a.execution_profile)
                feasibility = a.out / f'feasibility_{spec["pooling"]}.json'
                if feasibility.exists():
                    continue
                for seed in a.seeds:
                    case = a.out / 'trials' / grids.trial_id(spec, a.epochs, seed, evaluate_test)
                    if (case / 'failure.json').exists():
                        continue
                    runner.atomic_json(a.out / 'status.json', dict(state='training', model=a.model,
                        fold=a.fold, pooling=spec['pooling'], seed=seed, trial=str(case.resolve()),
                        build=str(a.build.resolve()), ablation=a.ablation, completed=len(records)))
                    contract = dict(model=a.model, spec=spec, seed=seed, epochs=a.epochs, patience=a.patience,
                        dataset_fingerprint=data.fingerprint, source_hashes=sources, evaluate_test=evaluate_test,
                        torch_version=torch.__version__)
                    if a.mode == 'primary' and not (case / 'metrics.json').exists():
                        study.reuse_trial(reuse, contract, case)
                    try:
                        offload = torch.autograd.graph.save_on_cpu(pin_memory=True) if a.mode == 'controlled' and spec['activation_offload'] and device.type == 'cuda' else nullcontext()
                        attention = sdpa_kernel(SDPBackend.MATH) if a.mode == 'controlled' else nullcontext()
                        train = runner.run_trial if a.mode == 'primary' else run_trial
                        progress = dict(progress_seconds=a.progress_seconds) if a.mode == 'controlled' else {}
                        with offload, attention:
                            result = train(a.model, spec, seed, a.epochs, a.patience, data, case,
                                           device, sources, evaluate_test=evaluate_test, **progress)
                    except BaseException as exc:
                        numerical = isinstance(exc, FloatingPointError) or (isinstance(exc, RuntimeError)
                            and str(exc).startswith('The total norm of order ')
                            and 'is non-finite' in str(exc))
                        if not resource_failure(exc) and not numerical:
                            raise
                        doc = dict(state='numerical_failure' if numerical else 'resource_infeasible',
                            error=type(exc).__name__, message=str(exc),
                            pooling=spec['pooling'], seed=seed, spec=spec, not_a_valid_accuracy_result=True,
                            remaining_seeds='Retry other seeds after numerical failure; do not impute scores' if numerical
                                else 'Not executed after this recipe failed; no scores imputed')
                        runner.atomic_json(case / 'failure.json', doc)
                        if not numerical:
                            runner.atomic_json(feasibility, doc)
                        runner.atomic_json(a.out / 'status.json', {**doc, 'state': 'retry_in_fresh_process'})
                        return 77  # Fresh process before another pooling arm uses CUDA.
                    records.append(dict(path=str(case.resolve()), result=result))
                    hashes_path = a.out / 'payload_hashes.json'
                    hashes = json.loads(hashes_path.read_text()) if hashes_path.exists() else {}
                    if any(sid in hashes and hashes[sid] != value for sid, value in data.hashes.items()):
                        raise ValueError('Source payload changed between confirmations')
                    hashes.update(data.hashes)
                    runner.atomic_json(hashes_path, hashes)
                    previous_coverage.update(data.coverage)
                    runner.atomic_json(coverage_path, previous_coverage)
            runner.atomic_json(a.out / 'results.json', records)
            infeasible = [str(f) for f in a.out.glob('feasibility_*.json')]
            failures = [str(f) for f in (a.out / 'trials').glob('*/failure.json')]
            state = 'complete_with_resource_limits' if infeasible else 'complete_with_trial_failures' if failures else 'complete'
            runner.atomic_json(a.out / 'status.json', dict(state=state,
                completed=len(records), resource_limits=infeasible, model=a.model, fold=a.fold,
                trial_failures=failures, ablation=a.ablation, test_evaluated=evaluate_test))
        except BaseException as exc:
            runner.atomic_json(a.out / 'status.json', dict(state='stopped' if isinstance(exc, InterruptedError) else 'failed',
                error=str(exc), resource_failure=resource_failure(exc)))
            if resource_failure(exc):
                runner.atomic_json(a.out / 'preparation_failure.json', dict(error=str(exc),
                    not_a_valid_accuracy_result=True, remaining_cases='Not run because input preparation failed'))
                return 78
            raise
        finally:
            data = None
            gc.collect()
    return 0


if __name__ == '__main__':
    sys.exit(main())
