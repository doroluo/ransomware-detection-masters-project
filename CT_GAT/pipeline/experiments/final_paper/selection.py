"""Complete staged searches on validation only, without opening outer-test tensors."""
import hashlib
import json
from pathlib import Path
import shutil

import torch

from experiments.grid_study import grids, study
from experiments.grid_study.budget import Budget, BudgetExhausted
from experiments.model_grid import runner
from experiments.model_grid.data import digest
from experiments.model_grid.locking import exclusive


class FullStageBudget(Budget):
    def __init__(self, path, hours):
        super().__init__(path, hours, shares={k: 1.0 for k in ('optimizer', 'architecture', 'coverage', 'confirm')})


def reuse_metadata(paths, contract, out):
    """Reuse exact completed trials; retain checkpoint provenance without duplicating old weights."""
    for path in paths:
        config = path.parent / 'config.json'
        if not config.is_file():
            continue
        saved = json.loads(config.read_text())
        if not study.compatible(saved, contract):
            continue
        result = json.loads(path.read_text())
        if result.get('fingerprint') != digest(saved):
            continue
        out.mkdir(parents=True, exist_ok=True)
        for name in ('config.json', 'metrics.json', 'history.json', 'validation_predictions.csv'):
            src = path.parent / name
            if src.is_file() and src.resolve() != (out / name).resolve():
                shutil.copy2(src, out / name)
        runner.atomic_json(out / 'reused_from.json', dict(path=str(path.parent.resolve()),
            requested_contract=contract, original_fingerprint=result['fingerprint'],
            checkpoint_policy='Original checkpoints retained in source trial; no duplicate weight files.',
            rule='Exact model/data/seed/training/metric contract; CLI dispatcher may differ.'))
        return result
    return None


def head_cases(model, base):
    if model == 1:
        axes = dict(transformer_heads=[4, 2, 8])
    elif model == 2:
        axes = dict(gat_heads=[4, 2, 8])
    else:
        axes = dict(transformer_heads=[4, 2, 8], gat_heads=[4, 2, 8], pooling=['mean', 'cls'])
    return [s for s in grids.expand(base, axes) if s['width'] % s['transformer_heads'] == 0]


def batch_cases(model, base):
    axes = dict(batch_size={1: [32, 16, 64], 2: [16, 8, 32], 3: [8, 4, 2]}[model])
    if model == 3:
        axes['pooling'] = ['mean', 'cls']
    return grids.expand(base, axes)


class ValidationStudy(study.Study):
    def trial(self, model, spec, phase, stage, build, blocks, epochs, seed, test=False):
        if test:
            raise ValueError('This selection worker cannot evaluate the outer test set')
        return super().trial(model, spec, phase, stage, build, blocks, epochs, seed, test=False)

    def screen(self, model, phase, stage, specs):
        choices = super().screen(model, phase, stage, specs)
        planned = specs[:2] if self.a.smoke and model == 3 else specs[:1] if self.a.smoke else specs
        folder = self.root / f'model{model}' / f'{stage}_500'
        attempted = sum((folder / grids.trial_id(s, self.a.epochs, self.a.screen_seed, False) / 'metrics.json').exists()
            or (folder / grids.trial_id(s, self.a.epochs, self.a.screen_seed, False) / 'failure.json').exists()
            for s in planned)
        if attempted != len(planned):
            raise BudgetExhausted(f'{stage}: {attempted}/{len(planned)} cases attempted; selection not finalized')
        if not choices:
            raise RuntimeError(f'{stage}: no valid complete configuration/pooling pair')
        return choices

    def run(self):
        for model in self.a.models:
            choices = self.screen(model, 'optimizer', 'optimizer',
                grids.optimizer(model, self.a.learning_rates, self.a.dropouts, self.a.weight_decays))
            if not choices:
                raise RuntimeError(f'Model {model}: no eligible optimizer selection')
            self.choices[model] = self.optimizer_choices[model] = choices
        for model in self.a.models:
            self.choices[model] = self.screen(model, 'architecture', 'architecture',
                grids.architecture(model, self.choices[model][0]))
        for model in self.a.models:
            if model == 3:
                self.choices[model] = self.screen(model, 'coverage', 'instruction',
                    grids.instruction(model, self.choices[model][0]))
        if self.a.extended_axes:
            for model in self.a.models:
                self.choices[model] = self.screen(model, 'architecture', 'heads',
                    head_cases(model, self.choices[model][0]))
                self.choices[model] = self.screen(model, 'optimizer', 'batch',
                    batch_cases(model, self.choices[model][0]))
        selection = dict(specs=self.choices, block_cap=500, sampler='signal',
            fold=self.a.fold, val_seed=self.a.val_seed,
            rule='Staged validation AUC then macro-F1; Model 3 uses complete paired mean/CLS settings.',
            outer_test_tensors_loaded=False, extended_axes=self.a.extended_axes,
            optimizer_choices=self.optimizer_choices)
        runner.atomic_json(self.root / 'selection.json', selection)
        runner.atomic_json(self.root / 'status.json', dict(state='selection_complete',
            completed=len(self.records), outer_test_tensors_loaded=False,
            used_hours=sum(self.budget.doc['used_seconds'].values()) / 3600))


def main(argv=None):
    p = study.parser()
    p.description = __doc__
    p.add_argument('--extended-axes', action='store_true', help='Add attention-head and batch-size stages')
    args = p.parse_args(argv)
    if args.build_250 is not None or args.build_uncapped is not None:
        p.error('Selection uses the fixed 500-block build; other representations belong to controlled comparisons')
    if args.epochs < 1 or args.gpu_hours <= 0:
        p.error('Epoch and runtime budgets must be positive')
    if args.smoke:
        args.epochs = 2
    if args.plan_only:
        print(json.dumps({m: dict(optimizer=len(grids.optimizer(m, args.learning_rates, args.dropouts, args.weight_decays)),
            architecture=len(grids.architecture(m, grids.base_spec(m))),
            instruction=len(grids.instruction(m, grids.base_spec(m))),
            heads=len(head_cases(m, grids.base_spec(m))) if args.extended_axes else 0,
            batch=len(batch_cases(m, grids.base_spec(m))) if args.extended_axes else 0)
            for m in args.models}, indent=2))
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    plan = dict(arguments={k: str(v) if isinstance(v, Path) else [str(x) if isinstance(x, Path) else x for x in v]
        if isinstance(v, list) else v for k, v in vars(args).items()}, scientific_sources=runner.source_hashes(),
        selection_worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        coverage_experiments=False, outer_test_access=False,
        search_type='Staged Cartesian grids, not one Cartesian product of all axes')
    plan = json.loads(json.dumps(plan))
    with exclusive(args.out / 'study.lock'):
        path = args.out / 'plan.json'
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError('Selection plan changed; choose a new output directory')
        runner.atomic_json(path, plan)
        study.Budget = FullStageBudget
        study.reuse_trial = reuse_metadata
        try:
            ValidationStudy(args).run()
        except BaseException as exc:
            runner.atomic_json(args.out / 'status.json', dict(state='stopped' if isinstance(exc, InterruptedError) else 'failed',
                error=str(exc), outer_test_access=False))
            raise


if __name__ == '__main__':
    main()
