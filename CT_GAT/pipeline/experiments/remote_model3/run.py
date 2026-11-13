"""Run the frozen fold-3 job with the measured GPU-resident execution profile."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from experiments.final_paper import controlled
from experiments.final_paper.data import protocol_hashes
from experiments.model_grid import runner

VARIANTS = ('signal250', 'signal500', 'signal750', 'signal1000',
            'callwalk250', 'callwalk500', 'callwalk750', 'callwalk1000')
ABLATIONS = ('none', 'no_api_identity', 'no_behavior', 'shuffle_instructions', 'edge_free')


def qualification_passed(path, build, selection, vocabulary, ablation):
    """Require a full recorded epoch under the current source and frozen inputs."""
    try:
        plan = json.loads((path / 'plan.json').read_text())
        status = json.loads((path / 'status.json').read_text())
        expected = dict(execution_profile='gpu-resident', qualification=True, epochs=1,
                        seeds=[0], smoke=0, device='cuda', model=3, fold=3,
                        ablation=ablation, evaluate_test=False, build=str(build.resolve()),
                        scientific_sources=runner.source_hashes(), adapter_sources=protocol_hashes(),
                        selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),
                        common_vocabulary_fingerprint=json.loads(vocabulary.read_text())['fingerprint'])
        if any(plan.get(k) != v for k, v in expected.items()):
            return False
        if status.get('state') != 'complete' or status.get('completed') != 1 or status.get('test_evaluated') is not False:
            return False
        cases = list((path / 'trials').glob('*/metrics.json'))
        if len(cases) != 1:
            return False
        case = cases[0].parent
        result = json.loads(cases[0].read_text())
        config = json.loads((case / 'config.json').read_text())
        history = json.loads((case / 'history.json').read_text())
        return (len(history) == 1 and history[0]['epoch'] == 1 and
                result['epochs_run'] == 1 and result['device'] == 'cuda' and
                result['fingerprint'] == controlled.digest(config) and
                config['evaluate_test'] is False and 'test' not in result and
                result['spec']['pooling'] == 'mean' and result['seed'] == 0 and
                (case / 'last.pt').is_file())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff-root', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, default='signal250')
    parser.add_argument('--ablation', choices=ABLATIONS, default='none')
    parser.add_argument('--qualification', action='store_true')
    args = parser.parse_args(argv)
    if args.ablation != 'none' and args.variant != 'signal500':
        parser.error('Assigned ablations use signal500')
    root = args.handoff_root.resolve()
    selection, vocabulary, index = (root / p for p in ('fold3/selection.json', 'fold3/common_api.json', 'index.csv'))
    for path in (selection, vocabulary, index):
        if not path.is_file():
            parser.error(f'Missing handoff metadata: {path}')
    build = root / 'data' / args.variant
    if (build / 'processed/cfg_subgraphs').is_dir():
        build = build / 'processed/cfg_subgraphs'
        for source, target in ((build.parent / 'api_vocab.json', build / 'api_vocab.json'),
                               (build / 'build_info/build_settings.json', build / 'settings.json')):
            if not target.exists() and source.is_file():
                shutil.copyfile(source, target)
    for name in ('settings.json', 'api_vocab.json'):
        if not (build / name).is_file():
            parser.error(f'Prepared dataset metadata missing: {build / name}')
    job = f'{args.variant}_{args.ablation}'
    qualification = root / 'results_gpu_resident_v1' / 'qualification' / job
    qualified = qualification_passed(qualification, build, selection, vocabulary, args.ablation)
    if not args.qualification and not qualified:
        parser.error(f'Complete a full-epoch qualification first: run_model3.ps1 qualify {args.variant} {args.ablation}')
    if args.qualification and qualified:
        print(f'Full-epoch qualification already passed: {qualification}', flush=True)
        return 0
    out = qualification if args.qualification else root / 'results_gpu_resident_v1' / 'jobs' / job
    arguments = ['--model', '3', '--mode', 'controlled', '--dataset', 'combined',
                 '--build', str(build), '--index', str(index), '--selection', str(selection),
                 '--vocabulary', str(vocabulary), '--fold', '3', '--val-seed', '0',
                 '--ablation', args.ablation, '--device', 'cuda', '--threads', '4',
                 '--patience', '8', '--execution-profile', 'gpu-resident',
                 '--progress-seconds', '30', '--out', str(out),
                 '--cache', str(root / 'cache_gpu_resident_v1' / ('qualification' if args.qualification else 'jobs') / job)]
    if args.qualification:
        arguments += ['--qualification', '--epochs', '1', '--seeds', '0']
    else:
        arguments += ['--epochs', '40', '--seeds', '0', '1', '2']
    print(f'GPU-resident Model 3: {job}, qualification={args.qualification}, output={out}', flush=True)
    code = controlled.main(arguments)
    if args.qualification and code == 0 and not qualification_passed(qualification, build, selection, vocabulary, args.ablation):
        raise RuntimeError('Qualification did not complete a valid full epoch; full jobs remain blocked')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
