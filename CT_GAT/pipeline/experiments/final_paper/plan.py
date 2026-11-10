"""A finite, predeclared campaign with a first-week checkpoint and a 20-day ceiling."""
from pathlib import Path

from .data import ABLATIONS


REPRESENTATIONS = ('signal250', 'signal500', 'callwalk250', 'callwalk500',
                   'signal750', 'signal1000', 'callwalk750', 'callwalk1000', 'uncapped')


def jobs_for(config):
    root = Path(config['root'])
    builds = config['builds']
    index = config['index']
    jobs = []

    def add(name, role, module, args, out, dependencies=(), priority=0, success=None):
        jobs.append(dict(id=name, role=role, module=module, args=[str(a) for a in args],
            out=str(out), dependencies=list(dependencies), priority=priority, success=success))

    add('build_caps', 'cpu', 'experiments.final_paper.builds',
        ['--uncapped', builds['uncapped'], '--signal-500', builds['signal500'],
         '--callwalk-500', builds['callwalk500'], '--out-root', root / 'builds', '--workers', config['build_workers']],
        root / 'builds', priority=-100, success='four_build_summaries')
    for fold in config['folds']:
        selection = root / 'selection' / f'fold{fold}'
        add(f'select_f{fold}', 'gpu', 'experiments.final_paper.selection',
            ['--models', 1, 2, 3, '--build', builds['signal500'], '--index', index, '--dataset', 'combined',
             '--fold', fold, '--out', selection, '--gpu-hours', config['maximum_hours'], '--extended-axes',
             '--reuse-from', *config['reuse_from'], '--device', config['device'], '--threads', config['threads']],
            selection, priority=0 if fold == 0 else 100 + fold * 10, success='selection_complete')
        vocab = root / 'vocab' / f'fold{fold}' / 'common_api.json'
        fit_args = ['--build', *[f'{k}={builds[k]}' for k in ('signal500', 'callwalk500', 'uncapped')],
                    '--index', index, '--fold', fold, '--out', vocab]
        add(f'vocabulary_f{fold}', 'cpu', 'experiments.final_paper.canonical', fit_args, vocab.parent,
            dependencies=() if fold == 0 else (f'select_f{fold}',), priority=-50 + fold,
            success='common_api.json')
        for model in (1, 2, 3):
            common = ['--model', model, '--index', index, '--selection', selection / 'selection.json',
                      '--fold', fold, '--seeds', *config['seeds'], '--epochs', config['confirm_epochs'],
                      '--patience', config['patience'], '--device', config['device'], '--threads', config['threads']]
            out = root / 'primary' / f'fold{fold}' / f'model{model}'
            add(f'primary_f{fold}_m{model}', 'gpu', 'experiments.final_paper.controlled',
                [*common, '--mode', 'primary', '--build', builds['signal500'], '--out', out,
                 '--reuse-from', *config['reuse_from']], out, (f'select_f{fold}',),
                priority=10 + model if fold == 0 else 100 + fold * 10 + model,
                success='confirmation')
            for rep in REPRESENTATIONS:
                name = f'compare_{rep}_f{fold}_m{model}'
                out = root / 'comparisons' / rep / f'fold{fold}' / f'model{model}'
                deps = [f'select_f{fold}', f'vocabulary_f{fold}', f'cuda_control_smoke_m{model}']
                if rep.endswith(('750', '1000')):
                    deps.append('build_caps')
                if fold == 0 and rep in REPRESENTATIONS[:4]:
                    priority = 30 + REPRESENTATIONS.index(rep) * 3 + model
                elif fold == 0:
                    priority = 300 + REPRESENTATIONS.index(rep) * 3 + model
                else:
                    priority = 600 + fold * 100 + REPRESENTATIONS.index(rep) * 3 + model
                add(name, 'gpu', 'experiments.final_paper.controlled',
                    [*common, '--mode', 'controlled', '--build', builds[rep], '--vocabulary', vocab,
                     '--out', out, '--cache', root / '_cache' / name], out, deps, priority,
                    success='confirmation')
            for number, ablation in enumerate(ABLATIONS[model]):
                name = f'ablate_{ablation}_f{fold}_m{model}'
                out = root / 'ablations' / ablation / f'fold{fold}' / f'model{model}'
                add(name, 'gpu', 'experiments.final_paper.controlled',
                    [*common, '--mode', 'controlled', '--ablation', ablation, '--build', builds['signal500'],
                     '--vocabulary', vocab, '--out', out, '--cache', root / '_cache' / name], out,
                    [f'select_f{fold}', f'vocabulary_f{fold}', f'cuda_control_smoke_m{model}'],
                    priority=400 + model * 10 + number if fold == 0 else 2000 + fold * 100 + model * 10 + number,
                    success='confirmation')
    for model in (1, 2, 3):
        name = f'cuda_control_smoke_m{model}'
        out = root / 'validation' / 'controlled_cuda' / f'model{model}'
        add(name, 'gpu', 'experiments.final_paper.controlled',
            ['--model', model, '--mode', 'controlled', '--build', builds['signal250'], '--index', index,
             '--selection', root / 'validation' / 'control_stress_selection.json',
             '--vocabulary', root / 'vocab' / 'fold0' / 'common_api.json', '--fold', 0,
             '--out', out, '--cache', root / '_cache' / name, '--device', config['device'],
             '--threads', config['threads'], '--seeds', 0, '--epochs', 2, '--smoke', 8], out,
             dependencies=('select_f0', 'vocabulary_f0'), priority=20 + model, success='confirmation')
    return sorted(jobs, key=lambda j: (j['priority'], j['id']))


def validate_plan(config, jobs):
    names = {j['id'] for j in jobs}
    if len(names) != len(jobs):
        raise ValueError('Duplicate job identifiers')
    skipped = config.get('skip_jobs', {})
    if not isinstance(skipped, dict) or not set(skipped) <= names:
        raise ValueError('Schedule exclusions must name existing jobs')
    if any(not isinstance(reason, str) or not reason.strip() for reason in skipped.values()):
        raise ValueError('Every schedule exclusion requires a reason')
    if config['folds'] != [0, 1, 2, 3, 4] or len(config['seeds']) != 3:
        raise ValueError('Paper preset requires five outer folds and three declared seeds')
    if not 0 < config['checkpoint_hours'] <= config['maximum_hours'] <= 480:
        raise ValueError('Invalid checkpoint or maximum budget')
    for j in jobs:
        if not set(j['dependencies']) <= names or j['id'] in j['dependencies']:
            raise ValueError('Invalid dependencies')
    seen = set()
    while len(seen) < len(names):
        ready = {j['id'] for j in jobs if set(j['dependencies']) <= seen}
        if ready <= seen:
            raise ValueError('Dependency cycle')
        seen |= ready


def counts(config):
    folds, seeds = len(config['folds']), len(config['seeds'])
    skipped_cases = {}
    for kind in ('primary', 'compare', 'ablate'):
        skipped_cases[kind] = sum((2 if name.endswith('_m3') else 1) * seeds
            for name in config.get('skip_jobs', {}) if name.startswith(kind + '_'))
    return dict(staged_selection_entries_per_fold=509,
        staged_selection_entries_total=509 * folds,
        primary_confirmations=4 * folds * seeds - skipped_cases['primary'],
        original_representation_confirmations=4 * len(REPRESENTATIONS) * folds * seeds,
        skipped_representation_confirmations=skipped_cases['compare'],
        representation_confirmations=4 * len(REPRESENTATIONS) * folds * seeds - skipped_cases['compare'],
        ablation_confirmations=(3 + 3 + 4 * 2) * folds * seeds - skipped_cases['ablate'],
        note='Stage entries include exact reusable duplicates; failed/infeasible cases never receive fabricated metrics.')
