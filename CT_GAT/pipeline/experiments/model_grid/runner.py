"""Validation-only Cartesian screening, paired pooling selection, then seeded confirmation.

New result directories are deliberate: historical config.json files can contain fields
that were ignored by the old trainers. Never import their scores as matching trials.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score

from s04_evaluation.val_threshold import grid_threshold
from .data import DATASETS, Prepared, digest, manifest_for
from .models import collate, forward, make_model

ROOT = Path(__file__).resolve().parents[2]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def save_checkpoint(path, value):
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


def source_hashes():
    files = list(Path(__file__).parent.glob('*.py'))
    files += [ROOT / p for p in (
        'grid_search.py',
        's03a_transformer/five_channel/model.py', 's03b_cfg_feature/gat/gat_full_model.py',
        's03b_cfg_feature/gat/graph_full_loader.py', 's02_preprocess/preprocess_full.py',
        's02_preprocess/tokenize_asm_gz.py', 'experiments/team_pipelines/train_b.py',
        'common/config.py', 'common/manifests.py', 's04_evaluation/val_threshold.py')]
    # Text hashes survive Git's CRLF/LF conversion across Windows/Linux teammates.
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest()
            for p in sorted(files)}


def metrics(y, p, threshold=.5):
    y, p = np.asarray(y), np.asarray(p)
    if len(y) != len(p) or not np.isfinite(p).all() or set(y) != {0, 1}:
        raise ValueError('metrics require finite predictions and both classes')
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'n': len(y), 'auc': float(roc_auc_score(y, p)),
            'pr_auc': float(average_precision_score(y, p)),
            'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'accuracy': float((tp + tn) / len(y)), 'detection_rate': float(tp / (tp + fn)),
            'fpr': float(fp / (fp + tn)), 'threshold': float(threshold),
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}


def batches(ids, size):
    for start in range(0, len(ids), size):
        yield ids[start:start + size]


@torch.no_grad()
def predict(model_id, net, data, split, size, device):
    net.eval()
    ids = [r['sample_id'] for r in data.splits[split]]
    probs = []
    for part in batches(ids, size):
        batch, _ = collate(model_id, [data.samples[s] for s in part], device)
        probs.extend(forward(model_id, net, batch).softmax(1)[:, 1].cpu().tolist())
    labels = [data.rows[s]['label'] for s in ids]
    return ids, labels, probs


def write_predictions(path, data, predictions, split):
    ids, labels, probs = predictions
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['sample_id', 'split', 'label', 'prob', 'family', 'arch'])
        w.writeheader()
        for sid, label, prob in zip(ids, labels, probs):
            w.writerow(dict(sample_id=sid, split=split, label=label, prob=prob,
                            family=data.rows[sid]['family'], arch=data.rows[sid]['arch']))


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def cpu_state(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def run_trial(model_id, spec, seed, epochs, patience, data, out, device, sources, evaluate_test=False):
    out.mkdir(parents=True, exist_ok=True)
    contract = dict(model=model_id, spec=spec, seed=seed, epochs=epochs, patience=patience,
                    dataset_fingerprint=data.fingerprint, source_hashes=sources, evaluate_test=evaluate_test,
                    torch_version=torch.__version__)
    fingerprint = digest(contract)
    if (out/'config.json').exists():
        if json.loads((out/'config.json').read_text()) != contract:
            raise ValueError(f'run settings/input/source changed: choose a new output directory: {out}')
    else:
        atomic_json(out/'config.json', contract)
    if (out/'metrics.json').exists():
        return json.loads((out/'metrics.json').read_text())
    if source_hashes() != sources:
        raise ValueError('source changed during queue; refusing mixed implementations')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    net = make_model(model_id, data.api_size, spec, seed).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=spec['lr'], weight_decay=spec['weight_decay'])
    train_ids = [r['sample_id'] for r in data.splits['train']]
    counts = np.bincount([data.rows[s]['label'] for s in train_ids], minlength=2)
    class_weight = torch.tensor(len(train_ids) / (2 * counts), dtype=torch.float32, device=device)
    history, start_epoch, best_key, stale = [], 1, (-1., -1.), 0
    if (out/'last.pt').exists():
        last = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
        if last['fingerprint'] != fingerprint:
            raise ValueError('checkpoint contract mismatch')
        net.load_state_dict(last['model'])
        opt.load_state_dict(last['optimizer'])
        history, start_epoch, best_key, stale = last['history'], last['epoch'] + 1, tuple(last['best_key']), last['stale']
        restore_rng(last['rng'])
    start_time = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        if stale >= patience:
            break
        if (out.parent.parent/'STOP').exists():
            raise InterruptedError('STOP file: saved checkpoints can be resumed')
        net.train()
        order = np.random.default_rng([seed, epoch, 7919]).permutation(train_ids).tolist()
        tick = time.perf_counter()
        total = 0.
        for part in batches(order, spec['batch_size']):
            batch, y = collate(model_id, [data.samples[s] for s in part], device)
            opt.zero_grad(set_to_none=True)
            logits = forward(model_id, net, batch)
            loss = (F.cross_entropy(logits, y, reduction='none') * class_weight[y]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite loss in {out.name}, epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            total += float(loss.detach()) * len(part)
        val = predict(model_id, net, data, 'val', spec['batch_size'], device)
        score = metrics(val[1], val[2])
        key = (score['auc'], score['macro_f1'])
        row = dict(epoch=epoch, train_loss=total / len(order), seconds=time.perf_counter() - tick,
                   val_auc=score['auc'], val_macro_f1=score['macro_f1'])
        history.append(row)
        if key > best_key:
            best_key, stale = key, 0
            save_checkpoint(out/'best.pt', {'model': cpu_state(net), 'epoch': epoch, 'fingerprint': fingerprint})
        else:
            stale += 1
        save_checkpoint(out/'last.pt', dict(model=cpu_state(net), optimizer=opt.state_dict(), epoch=epoch,
                         history=history, best_key=best_key, stale=stale, rng=rng_state(), fingerprint=fingerprint))
        atomic_json(out/'history.json', history)
        print(json.dumps(dict(run=out.name, seed=seed, **row)), flush=True)
    best = torch.load(out/'best.pt', map_location='cpu', weights_only=False)
    net.load_state_dict(best['model'])
    val = predict(model_id, net, data, 'val', spec['batch_size'], device)
    threshold = grid_threshold(np.array(val[1]), np.array(val[2]))
    result = dict(spec=spec, seed=seed, best_epoch=best['epoch'], epochs_run=len(history),
                  validation=metrics(val[1], val[2], threshold), validation_fixed=metrics(val[1], val[2]),
                  epoch_seconds_mean=statistics.mean(r['seconds'] for r in history),
                  invocation_seconds=time.perf_counter() - start_time,
                  device=str(device), gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                  peak_gpu_mb=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0,
                  parameters=sum(p.numel() for p in net.parameters()), fingerprint=fingerprint)
    write_predictions(out/'validation_predictions.csv', data, val, 'val')
    if evaluate_test:
        data.load_test()
        test = predict(model_id, net, data, 'test', spec['batch_size'], device)
        result['test'] = metrics(test[1], test[2], threshold)
        write_predictions(out/'test_predictions.csv', data, test, 'test')
    atomic_json(out/'metrics.json', result)
    del net, opt
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def cases(args, model):
    base = dict(width=64, transformer_heads=4, transformer_layers=2, gat_hidden=128, gat_heads=4,
                gat_layers=2, sequence_length=args.sequence_length, block_length=args.block_length,
                position_initialization='normal_std_0.02',
                block_chunk=args.block_chunk, batch_size=args.batch_size or {1:32, 2:16, 3:8}[model],
                weight_decay=args.weight_decay if args.weight_decay is not None else (.01 if model == 1 else .0001))
    pools = ('mean', 'cls') if model == 3 else ('mean' if model == 1 else 'graph_mean_max',)
    return [dict(base, lr=lr, dropout=drop, pooling=pool)
            for lr, drop, pool in itertools.product(args.learning_rates, args.dropouts, pools)]


def case_name(spec):
    return f"lr{spec['lr']:g}_drop{spec['dropout']:g}_{spec['pooling']}"


def select_cases(model, results):
    """Model 3 picks a shared LR/dropout by average validation AUC, preserving a paired test."""
    score = lambda r: (r['validation_fixed']['auc'], r['validation_fixed']['macro_f1'])
    if model != 3:
        return [max(results, key=score)['spec']]
    groups = {}
    for r in results:
        groups.setdefault((r['spec']['lr'], r['spec']['dropout']), []).append(r)
    if any({r['spec']['pooling'] for r in pair} != {'mean', 'cls'} for pair in groups.values()):
        raise ValueError('incomplete mean/CLS pairs')
    best = max(groups.values(), key=lambda pair: tuple(statistics.mean(score(r)[i] for r in pair) for i in (0, 1)))
    return [r['spec'] for r in sorted(best, key=lambda r: r['spec']['pooling'], reverse=True)]


def summarize(out, model, screened, confirmed):
    lines = [f'# Model {model} grid search', '', 'Screening uses validation only; thresholds are fitted on validation.',
             'Confirmation uses a fixed grouped split across seeds. These are one-fold results, not five-fold CV.',
             'Model 3 dropout applies to both Transformer and GAT; Model 1 to Transformer/head; Model 2 to GAT.',
             'Metrics below use a validation-selected threshold; each checkpoint also records metrics at 0.5.', '',
             '| LR | Dropout | Pooling | Val AUC | Val detection | Val FPR | Seconds/epoch |',
             '|---|---|---|---|---|---|---|']
    records = []
    for r in screened:
        s, v = r['spec'], r['validation']
        lines.append(f"| {s['lr']:g} | {s['dropout']:g} | {s['pooling']} | {v['auc']:.4f} | "
                     f"{v['detection_rate']:.4f} | {v['fpr']:.4f} | {r['epoch_seconds_mean']:.2f} |")
        records.append(dict(stage='screen', seed=r['seed'], **s, **v, seconds_per_epoch=r['epoch_seconds_mean']))
    if confirmed:
        lines += ['', '## Held-out confirmation', '', 'Mean ± sample SD across seeds; not confidence intervals.', '',
                  '| Pooling | Seeds | AUC | Detection | FPR | Macro F1 |', '|---|---|---|---|---|---|']
        for pool in sorted({r['spec']['pooling'] for r in confirmed}):
            rr = [r for r in confirmed if r['spec']['pooling'] == pool]
            cells = []
            for k in ('auc', 'detection_rate', 'fpr', 'macro_f1'):
                vv = [r['test'][k] for r in rr]
                cells.append(f'{statistics.mean(vv):.4f}' + (f' ± {statistics.stdev(vv):.4f}' if len(vv) > 1 else ''))
            lines.append(f"| {pool} | {','.join(str(r['seed']) for r in rr)} | " + ' | '.join(cells) + ' |')
            records += [dict(stage='confirm_test', seed=r['seed'], **r['spec'], **r['test'],
                             seconds_per_epoch=r['epoch_seconds_mean']) for r in rr]
    (out/'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if records:
        with (out/'results.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for ax, metric in zip(axes, ('auc', 'detection_rate', 'fpr')):
            for pool, drop in sorted({(r['spec']['pooling'], r['spec']['dropout']) for r in screened}):
                rr = sorted([r for r in screened if r['spec']['pooling'] == pool and r['spec']['dropout'] == drop],
                            key=lambda r: r['spec']['lr'])
                ax.plot([r['spec']['lr'] for r in rr], [r['validation'][metric] for r in rr], 'o-',
                        label=f'{pool}, dropout {drop:g}')
            ax.set(xscale='log', xlabel='Learning rate', ylabel=f'Validation {metric}')
            ax.grid(alpha=.25)
        axes[0].legend(fontsize=7)
        fig.suptitle(f'Model {model}: validation grid (one seed; no seed error bars)')
        fig.tight_layout(); fig.savefig(out/'grid.png', dpi=160); plt.close(fig)
    except ImportError:
        pass


def parser(model):
    p = argparse.ArgumentParser(description=f'Model {model} Cartesian grid; run from CT_GAT/pipeline.')
    p.add_argument('--build', type=Path, required=True, help='Prepared pipeline-B .pt build; identical across machines')
    p.add_argument('--index', type=Path, required=True, help='CFG cache index.csv containing the committed folds')
    p.add_argument('--dataset', choices=DATASETS, default='mendeley-x86')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--folds', nargs='+', type=int, default=[0], choices=range(5))
    p.add_argument('--val-seed', type=int, default=0)
    p.add_argument('--screen-seed', type=int, default=0)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--learning-rates', nargs='+', type=float, default=[1e-4, 3e-4, 1e-3])
    p.add_argument('--dropouts', nargs='+', type=float, default=[.1, .3])
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--confirm-epochs', type=int, default=40)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--stage', choices=['screen', 'confirm', 'all'], default='all')
    p.add_argument('--sequence-length', type=int, default=128)
    p.add_argument('--block-length', type=int, default=64)
    p.add_argument('--block-chunk', type=int, default=1024)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--weight-decay', type=float)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--smoke', type=int, default=0, help='Tiny training/validation subsets, 2 epochs, no test access')
    p.add_argument('--benchmark', action='store_true', help='Full train/validation, one setting (both M3 pools), 2 epochs')
    p.add_argument('--plan-only', action='store_true', help='Print plan/split counts without loading tensors or training')
    return p


def main(model, argv=None):
    args = parser(model).parse_args(argv)
    for name in ('learning_rates', 'dropouts', 'folds'):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            raise ValueError(f'{name} must not contain duplicate values')
    if any(v <= 0 for v in args.learning_rates) or any(not 0 <= v < 1 for v in args.dropouts):
        raise ValueError('learning rates must be positive and dropout in [0,1)')
    for name in ('epochs', 'confirm_epochs', 'patience', 'sequence_length', 'block_length', 'block_chunk', 'threads'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError('batch size must be positive')
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds + [args.screen_seed, args.val_seed]):
        raise ValueError('seeds must be nonnegative and unique')
    specs = cases(args, model)
    if args.smoke or args.benchmark:
        specs = [s for s in specs if s['lr'] == args.learning_rates[0] and s['dropout'] == args.dropouts[0]]
        args.epochs, args.stage = 2, 'screen'
    if args.plan_only:
        for fold in args.folds:
            splits = manifest_for(args.index, args.dataset, fold, args.val_seed)
            print(json.dumps(dict(model=model, dataset=args.dataset, fold=fold, cases=specs,
                                  counts={s: len(r) for s,r in splits.items()}, stage=args.stage), indent=2))
        return
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; install a CUDA-enabled PyTorch build or use --device cpu')
    torch.set_num_threads(args.threads)
    sources = source_hashes()
    for fold in args.folds:
        out = args.out / f'fold{fold}'
        out.mkdir(parents=True, exist_ok=True)
        # An OS-held lock is released even after a crash, preventing two writers per fold.
        from .locking import exclusive
        with exclusive(out/'run.lock'):
            try:
                atomic_json(out/'status.json', dict(state='preparing', model=model, fold=fold))
                tick = time.perf_counter()
                data = Prepared(args.build, args.index, args.dataset, fold, args.val_seed, model, specs[0], args.smoke)
                prep_seconds = time.perf_counter() - tick
                plan = dict(model=model, specs=specs, epochs=args.epochs, confirm_epochs=args.confirm_epochs,
                            patience=args.patience, screen_seed=args.screen_seed, seeds=args.seeds,
                            dataset_fingerprint=data.fingerprint, source_hashes=sources,
                            torch_version=torch.__version__, smoke=args.smoke, benchmark=args.benchmark)
                if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text()) != plan:
                    raise ValueError('plan differs from saved plan; choose a new --out directory')
                atomic_json(out/'plan.json', plan)
                atomic_json(out/'dataset.json', dict(data.provenance, preparation_seconds=prep_seconds,
                                                    build_path=str(args.build.resolve()), index_path=str(args.index.resolve())))
                atomic_json(out/'coverage.json', data.coverage)
                print(json.dumps(dict(prepared=len(data.samples), seconds=prep_seconds, model=model, fold=fold)), flush=True)
                screened, confirmed = [], []
                for spec in specs:
                    trial = out/'screen'/case_name(spec)
                    if args.stage == 'confirm' and not (trial/'metrics.json').exists():
                        raise ValueError('confirmation requires a completed screen with this plan')
                    atomic_json(out/'status.json', dict(state='screening', case=case_name(spec), complete=len(screened), total=len(specs)))
                    screened.append(run_trial(model, spec, args.screen_seed, args.epochs, args.epochs + 1,
                                               data, trial, device, sources))
                    summarize(out, model, screened, confirmed)
                chosen = select_cases(model, screened)
                selection = dict(rule='shared LR/dropout by mean validation AUC across pooling arms' if model == 3 else
                                 'validation AUC, then macro-F1 at 0.5', specs=chosen,
                                 paired_pooling=model == 3, test_used_for_selection=False)
                atomic_json(out/'selection.json', selection)
                if args.stage != 'screen':
                    for spec, seed in itertools.product(chosen, args.seeds):
                        atomic_json(out/'status.json', dict(state='confirming', case=case_name(spec), seed=seed))
                        confirmed.append(run_trial(model, spec, seed, args.confirm_epochs, args.patience, data,
                                         out/'confirm'/f'{case_name(spec)}_seed{seed}', device, sources, evaluate_test=True))
                        summarize(out, model, screened, confirmed)
                atomic_json(out/'coverage.json', data.coverage)
                atomic_json(out/'payload_hashes.json', data.hashes)
                atomic_json(out/'status.json', dict(state='complete', stage=args.stage, screening_runs=len(screened),
                                                   confirmation_runs=len(confirmed), preparation_seconds=prep_seconds))
            except BaseException as exc:
                atomic_json(out/'status.json', dict(state='stopped' if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else 'failed',
                                                   error=str(exc)))
                raise


if __name__ == '__main__':
    raise SystemExit('Use python -m experiments.model_grid.model1 (or model2 / model3).')
