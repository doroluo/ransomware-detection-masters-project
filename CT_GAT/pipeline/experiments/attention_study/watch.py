"""Run the attention follow-up after the current machine's campaign queue finishes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.final_paper.campaign import TERMINAL, load_state
from experiments.model_grid.locking import exclusive
from experiments.model_grid.runner import atomic_json

from .models import source_hashes


def remaining_local(config, state):
    remote = set(config.get('remote_jobs', {}))
    return {name: record['state'] for name, record in state['jobs'].items()
            if name not in remote and record['state'] not in TERMINAL}


def export(args):
    command = [sys.executable, '-m', 'experiments.attention_study.report',
               '--campaign', str(args.campaign), '--study', str(args.out),
               '--out', str(args.out / 'reports' / 'records.json')]
    if args.xlsx_node:
        command += ['--xlsx-node', str(args.xlsx_node), '--xlsx-builder', str(args.xlsx_builder),
                    '--xlsx-out', str(args.out / 'reports' / 'attention_results.xlsx')]
    subprocess.run(command, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--heads', type=int, nargs='+', default=[2, 4, 8, 12])
    parser.add_argument('--attention-dropouts', type=float, nargs='+', default=[0.0, 0.1, 0.2])
    parser.add_argument('--width', type=int, default=96)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--xlsx-node', type=Path)
    parser.add_argument('--xlsx-builder', type=Path)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    if bool(args.xlsx_node) != bool(args.xlsx_builder):
        parser.error('Excel export requires both the Node executable and workbook builder')
    if not args.heads or len(set(args.heads)) != len(args.heads) or any(
            h < 1 or args.width % h for h in args.heads):
        parser.error('Head counts must be unique positive divisors of the fixed width')
    if not args.attention_dropouts or len(set(args.attention_dropouts)) != len(args.attention_dropouts) or any(
            not 0 <= value < 1 for value in args.attention_dropouts):
        parser.error('Attention dropout must contain unique probabilities in [0, 1)')
    args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.campaign / 'CAMPAIGN.json').read_text())
    deadline = config['started_epoch'] + config['maximum_hours'] * 3600
    plan = dict(
        campaign=str(args.campaign.resolve()), heads=args.heads, attention_dropouts=args.attention_dropouts,
        transformer_width=args.width, device=args.device, deadline_epoch=deadline,
        folds=config['folds'], models=[1, 2, 3], seeds=config['seeds'],
        screen_epochs=12, confirm_epochs=config['confirm_epochs'], patience=config['patience'],
        attention_sources=source_hashes(), start_condition='All local campaign jobs terminal',
        remote_assignments_remain_independent=True,
        xlsx_builder_sha256=hashlib.sha256(args.xlsx_builder.read_bytes()).hexdigest()
            if args.xlsx_builder else None,
    )
    plan_path = args.out / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Follow-up plan changed. Use a new output folder')
    atomic_json(plan_path, plan)
    with exclusive(args.out / 'attention_controller.lock'):
        while True:
            if time.time() >= deadline or (args.out / 'STOP').exists():
                atomic_json(args.out / 'status.json', dict(state='stopped_before_dispatch'))
                return 0
            config = json.loads((args.campaign / 'CAMPAIGN.json').read_text())
            state = load_state(args.campaign)
            if not state:
                raise ValueError('The original campaign state is missing')
            waiting = remaining_local(config, state)
            if waiting:
                atomic_json(args.out / 'status.json', dict(
                    state='waiting_for_local_queue', remaining=len(waiting),
                    active=[name for name, status in waiting.items() if status == 'running'],
                    checked_utc=datetime.now(timezone.utc).isoformat(),
                ))
                if args.once:
                    return 0
                time.sleep(20)
                continue
            queue_changed = False
            for fold in config['folds']:
                for model in (1, 2, 3):
                    if source_hashes() != plan['attention_sources']:
                        raise ValueError('Attention study sources changed after scheduling')
                    if args.xlsx_builder and hashlib.sha256(args.xlsx_builder.read_bytes()).hexdigest() != plan['xlsx_builder_sha256']:
                        raise ValueError('The Excel builder changed after scheduling')
                    output = args.out / f'fold{fold}' / f'model{model}'
                    status_path = output / 'status.json'
                    if status_path.exists() and json.loads(status_path.read_text())['state'] == 'complete':
                        continue
                    # Re-check the assignment before each worker, including any later local reassignment.
                    latest = json.loads((args.campaign / 'CAMPAIGN.json').read_text())
                    if remaining_local(latest, load_state(args.campaign)):
                        queue_changed = True
                        break
                    if time.time() >= deadline or (args.out / 'STOP').exists():
                        export(args)
                        return 0
                    command = [
                        sys.executable, '-u', '-m', 'experiments.attention_study.study',
                        '--campaign', str(args.campaign), '--out', str(output),
                        '--fold', str(fold), '--model', str(model), '--heads', *map(str, args.heads),
                        '--attention-dropouts', *map(str, args.attention_dropouts),
                        '--width', str(args.width), '--device', args.device,
                        '--seeds', *map(str, config['seeds']),
                        '--screen-epochs', '12', '--confirm-epochs', str(config['confirm_epochs']),
                        '--patience', str(config['patience']), '--threads', str(config['threads']),
                        '--minimum-free-gb', str(config['minimum_free_gb']),
                        '--deadline-epoch', str(deadline),
                    ]
                    logs = args.out / 'logs'
                    logs.mkdir(exist_ok=True)
                    while True:
                        previous_failures = len(list(output.glob('*/*/failure.json')))
                        with (logs / f'fold{fold}_model{model}.log').open('a', encoding='utf-8') as log:
                            worker = subprocess.Popen(
                                command, cwd=Path(__file__).resolve().parents[2], stdout=log,
                                stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                            )
                            atomic_json(args.out / 'status.json', dict(
                                state='running', fold=fold, model=model, pid=worker.pid,
                                started_utc=datetime.now(timezone.utc).isoformat(),
                            ))
                            while worker.poll() is None:
                                if (args.out / 'STOP').exists():
                                    output.mkdir(parents=True, exist_ok=True)
                                    (output / 'STOP').write_text('Attention study stop requested', encoding='utf-8')
                                time.sleep(20)
                        if worker.returncode != 77:
                            break
                        if len(list(output.glob('*/*/failure.json'))) <= previous_failures:
                            raise RuntimeError('A resource retry made no recorded progress')
                    export(args)
                    if worker.returncode:
                        atomic_json(args.out / 'status.json', dict(
                            state='stopped' if worker.returncode == 75 else 'failed',
                            fold=fold, model=model, exit_code=worker.returncode,
                        ))
                        return worker.returncode
                    if args.once:
                        return 0
                if queue_changed:
                    break
            if queue_changed:
                continue
            atomic_json(args.out / 'status.json', dict(
                state='complete', completed_utc=datetime.now(timezone.utc).isoformat()))
            export(args)
            return 0


if __name__ == '__main__':
    raise SystemExit(main())
