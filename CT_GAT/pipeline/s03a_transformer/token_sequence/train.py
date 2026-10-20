"""Train Model 1 on one fold of a block export.

python -m experiments.train --model tokenseq --protocol kfold --arch x86 --fold 0 --seed 0 [--seq-len 512 --max-windows 64]
[--export ours_cap0] [--device cuda|cpu] [--smoke N] [--epochs 40 --patience 8] [--window-budget 128]
python -m s03a_transformer.token_sequence.train ...        the same run, with the options of Model 1 only

Run from CT_GAT/pipeline. `common/registry.py` serves this loop as the model `tokenseq` (`token_sequence/spec.py`), so
`experiments.train` hands the run over to `run` below. The run directory is <runs>/kfold<f>_x86_tokenseq_s<seed>[-L<n>][-w<n>]
(GT_RUNS; RunConfig.name, model 'tokenseq'); it holds config.json, manifest.json, history.csv, best.pt (best validation macro-F1 at 0.5), best_auc.pt
(best validation AUC), predictions.csv (F1 checkpoint at 0.5), predictions_auc.csv (AUC checkpoint at its validation
threshold) and metrics.json, with the keys of a GNN run, so s04_evaluation/report.py, stats.py and gates.py read it like one.

Data: FoldSequences(export, fold, 'train' | 'val' | 'test'); the train / val / test split is the export's (carved with
val_seed 0, which tokenseq forces), the vocabulary is the fold's. Model: BinaryTokenSequence(Model1TokenSequence). Loss: cross
entropy with the class weight N / (2 n_class) of the training labels. AdamW (RunConfig lr, weight_decay), gradient clip
RunConfig.grad_clip. A batch is whole binaries whose windows add up to at most --window-budget (an upper bound: a binary
with more windows than the budget is alone in its batch; at most max_windows windows per binary are drawn in training), so
the peak memory is set by the budget, not by the binaries. Evaluation uses every window of every binary, in order, under
no_grad. Each epoch records the validation AUC and the macro-F1 at 0.5; the F1 checkpoint is the epoch of the best macro-F1,
the AUC checkpoint that of the best AUC, and training stops when the AUC has not improved for `patience` epochs. The AUC
checkpoint's threshold is the GRID maximiser of the validation macro-F1 (val_threshold_of). The test split is read once
after training, for the two checkpoints.

--smoke N trains on N train binaries (half of each class) for 2 epochs and evaluates on N validation and N test binaries,
writes into a temporary directory (never under the run directory), prints s/epoch and the peak memory, and projects the time of a full
fold run from the windows and positions of the whole split.
"""
import argparse
import csv
import json
import os
import random
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')      # before CUDA starts: deterministic cuBLAS for predict

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common.config import EXPORT, RUNS, RunConfig
from common.manifests import Manifest
from common.reader import BlockStore
from common.run_io import evaluate, write_run
from experiments.train import val_threshold_of

from .data import FoldSequences, collate
from .model import BinaryTokenSequence, Model1TokenSequence

WINDOW_BUDGET = 128      # windows per batch (all windows of the binaries of a batch)
EXPORT_NAME = 'ours_cap0'
SMOKE_EPOCHS = 2


def check_supported(cfg: RunConfig):
    """Model 1 trains `model='tokenseq'` configs only, and none of the GNN options applies to it (RunConfig refuses them at
    construction; this repeats the check for a config built some other way)."""
    if cfg.model != 'tokenseq':
        raise ValueError(f"token_sequence.train trains model='tokenseq', not {cfg.model!r}")
    cfg.check_tokenseq()


def resolve_export(export):
    """`export` is a directory, or the name of one under the export directory (<runs>/export, or GT_EXPORT)."""
    return Path(export) if Path(export).exists() else EXPORT / export


def check_export(store: BlockStore, cfg: RunConfig):
    """The export is the one `cfg` names: same protocol and arch, has the fold, and its fold manifest was carved with the
    validation seed tokenseq forces."""
    meta = store.meta
    if (meta['protocol'], meta['arch']) != (cfg.protocol, cfg.arch) or cfg.fold not in meta['folds']:
        raise ValueError(f"export {store.root} is {meta['protocol']} {meta['arch']} folds {meta['folds']}, the run is "
                         f'{cfg.protocol} {cfg.arch} fold {cfg.fold}')
    manifest = store.fold(cfg.fold)['manifest']
    if manifest.get('val_seed') != cfg.val_seed or manifest.get('fold') != cfg.fold:
        raise ValueError(f"fold {cfg.fold} of {store.root} has val_seed {manifest.get('val_seed')}, the run needs "
                         f'{cfg.val_seed}')


def export_manifest(store: BlockStore, fold: int) -> Manifest:
    """The train / val / test rows (binaries.json entries: sample_id, label, arch, family, fold, ...) of a fold of the export,
    in the export's order."""
    m = store.fold(fold)['manifest']
    rows = lambda ids: [dict(store.binaries[i]) for i in ids]
    return Manifest(rows(m['train']), rows(m['val']), rows(m['test']), list(m['val_families']))


def smoke_subset(manifest_rows, store: BlockStore, n: int):
    """The first n binaries of a split, half of each class (the other class fills in when one runs out); store indices."""
    by_label = {c: [store.index_of[r['sample_id']] for r in manifest_rows if r['label'] == c] for c in (0, 1)}
    take = {1: min(len(by_label[1]), (n + 1) // 2)}
    take[0] = min(len(by_label[0]), n - take[1])
    take[1] = min(len(by_label[1]), n - take[0])
    return by_label[1][:take[1]] + by_label[0][:take[0]]


def pack(sizes, order, budget):
    """Lists of indices from `order`, in order, each with at most `budget` windows in total (`sizes[i]` windows for item i);
    an item with more windows than the budget is alone."""
    batches, cur, total = [], [], 0
    for i in order:
        if cur and total + sizes[i] > budget:
            batches.append(cur)
            cur, total = [], 0
        cur.append(int(i))
        total += int(sizes[i])
    return batches + ([cur] if cur else [])


def build(ds: FoldSequences, idx):
    start = time.perf_counter()
    batch = collate([ds[i] for i in idx])
    batch['build_secs'] = time.perf_counter() - start
    return batch


def prefetch(ds: FoldSequences, plan, workers):
    """The collated batches of `plan` in order, built `workers` at a time on threads (the work is numpy; the draws are seeded
    by (seed, epoch, index), so the batches do not depend on the thread order)."""
    if workers <= 0:
        for idx in plan:
            yield build(ds, idx)
        return
    with ThreadPoolExecutor(workers) as pool:
        plan, pending = iter(plan), deque()
        for idx in plan:
            pending.append(pool.submit(build, ds, idx))
            if len(pending) > workers:
                break
        while pending:
            batch = pending.popleft().result()
            for idx in plan:
                pending.append(pool.submit(build, ds, idx))
                break
            yield batch


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def forward(net, batch):
    return net(batch['ids'], batch['kind'], batch['pad_mask'], batch['win2bin'], n_binaries=len(batch['labels']))


def autocast(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda')


@torch.no_grad()
def predict(net: BinaryTokenSequence, ds: FoldSequences, device, budget, workers):
    """(sample ids, P(ransomware) [n], sizes) of every binary of `ds` in its order, from all windows, deterministic. sizes
    is (windows, n_blocks_kept, n_blocks_total) per binary (the write_run layout)."""
    net.eval()
    net.window_chunk = budget
    sizes_of = [ds.n_windows(i) for i in range(len(ds))]
    ids, probs, sizes = [], [], []
    was = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)          # the per-binary pooling is an index_add, atomic (order-dependent) on CUDA
    try:
        for batch in prefetch(ds, pack(sizes_of, range(len(ds)), budget), workers):
            with autocast(device):
                logits = forward(net, to_device(batch, device))
            probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
            n_windows = np.bincount(batch['win2bin'].numpy(), minlength=len(batch['sample_ids']))
            for sid, w in zip(batch['sample_ids'], n_windows):
                info = ds.view.store.binaries[ds.view.store.index_of[sid]]
                sizes.append((int(w), info['n_blocks_kept'], info['n_blocks_total']))
            ids += batch['sample_ids']
    finally:
        torch.use_deterministic_algorithms(was)
        net.window_chunk = None
    return ids, (np.concatenate(probs) if probs else np.zeros(0)), sizes


def peak_ram_mb():
    try:
        import psutil
        info = psutil.Process().memory_info()
        return round(getattr(info, 'peak_wset', info.rss) / 2**20)
    except ImportError:
        return None


def projection(history, smoke_train, smoke_val, full_train, full_val, max_epochs, patience):
    """Seconds of one full fold run from the smoke epochs. Train and validation time per epoch is scaled from the smoke
    split to the whole split by windows (all the compute) and by positions (all the data preparation, which regroups
    every position of a binary, not only the windows drawn); the two are a lower and an upper estimate."""
    mean = lambda key: float(np.mean([h[key] for h in history]))
    out = {}
    for name, smoke, full, key in (('train', smoke_train, full_train, 'train_secs'), ('val', smoke_val, full_val, 'val_secs')):
        secs = mean(key)
        out[name] = {'smoke_windows': smoke['windows'], 'smoke_positions': smoke['positions'], 'full_windows': full['windows'],
                     'full_positions': full['positions'], 'smoke_secs': round(secs, 2),
                     'by_windows': round(secs * full['windows'] / max(1, smoke['windows']), 1),
                     'by_positions': round(secs * full['positions'] / max(1, smoke['positions']), 1)}
    lo = out['train']['by_windows'] + out['val']['by_windows']
    hi = out['train']['by_positions'] + out['val']['by_positions']
    out['epoch_secs'] = [round(lo, 1), round(hi, 1)]
    out['run_secs_at_max_epochs'] = [round(lo * max_epochs), round(hi * max_epochs)]
    out['max_epochs'], out['patience'] = max_epochs, patience
    return out


def split_size(ds: FoldSequences):
    return {'windows': int(sum(ds.n_windows(i) for i in range(len(ds)))), 'positions': int(ds.positions().sum()),
            'binaries': len(ds)}


def say(*args):
    print(*args, flush=True)


def run(cfg: RunConfig, export=EXPORT_NAME, device=None, smoke=None, window_budget=WINDOW_BUDGET, out_root=None, log=say):
    """Train and write one run directory (under RUNS, `out_root`, or a temporary directory for a smoke run); returns its
    metrics."""
    check_supported(cfg)
    if smoke is not None and out_root is None:
        with tempfile.TemporaryDirectory(prefix='tokenseq_smoke_') as tmp:
            return run(cfg, export, device, smoke, window_budget, Path(tmp), log)
    out_root = Path(out_root) if out_root is not None else RUNS
    if smoke is not None and RUNS.resolve() in (out_root / 'x').resolve().parents:
        raise ValueError(f'a smoke run never writes under {RUNS}: {out_root}')
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    full_epochs = cfg.epochs
    if smoke is not None:
        cfg = replace(cfg, epochs=SMOKE_EPOCHS)
    out = out_root / cfg.name
    out.mkdir(parents=True, exist_ok=True)

    store = BlockStore(resolve_export(export))
    check_export(store, cfg)
    manifest = export_manifest(store, cfg.fold)
    subsets = {}
    if smoke is not None:
        rows = {'train': manifest.train, 'val': manifest.val, 'test': manifest.test}
        subsets = {k: smoke_subset(v, store, smoke) for k, v in rows.items()}
        keep = {k: set(v) for k, v in subsets.items()}
        manifest = Manifest(*[[r for r in rows[k] if store.index_of[r['sample_id']] in keep[k]] for k in ('train', 'val', 'test')],
                            manifest.val_families)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    data = {s: FoldSequences(store, cfg.fold, s, L=cfg.seq_len, max_windows=cfg.max_windows, seed=cfg.seed,
                             subset=subsets.get(s)) for s in ('train', 'val', 'test')}
    train, val, test = data['train'], data['val'], data['test']
    if sorted(train.sample_ids()) != sorted(r['sample_id'] for r in manifest.train):
        raise AssertionError('the training binaries are not the manifest\'s')
    labels = np.array(train.labels())
    class_weight = torch.tensor([len(labels) / (2 * max(1, (labels == c).sum())) for c in (0, 1)], device=device)
    model = Model1TokenSequence(n_ids=train.n_ids, max_seq_len=cfg.seq_len)
    net = BinaryTokenSequence(model).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    cfg.save(out / 'config.json')
    log(f'{cfg.name}: train {len(train)} val {len(val)} test {len(test)} binaries, n_ids {train.n_ids}, {n_params} parameters, '
        f'window budget {window_budget}, device {device}' + (f', smoke {smoke}' if smoke is not None else ''))

    train_sizes = [train.n_windows(i) for i in range(len(train))]
    history, best_f1, best_f1_epoch, best_auc, best_auc_epoch = [], -1.0, 0, -1.0, 0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, cfg.epochs + 1):
        net.train()
        net.window_chunk = None
        train.set_epoch(epoch)
        order = np.random.default_rng((cfg.seed, epoch, 7919)).permutation(len(train))
        start, losses, windows, build_secs = time.perf_counter(), [], 0, 0.0
        for batch in prefetch(train, pack(train_sizes, order, window_budget), cfg.num_workers):
            windows += len(batch['ids'])
            build_secs += batch['build_secs']
            batch = to_device(batch, device)
            with autocast(device):
                logits = forward(net, batch)
            loss = (F.cross_entropy(logits.float(), batch['labels'], reduction='none') * class_weight[batch['labels']]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(loss.item())
        train_secs = time.perf_counter() - start
        ids, probs, _ = predict(net, val, device, window_budget, cfg.num_workers)
        scores = evaluate(manifest.val, ids, probs, cfg.threshold)
        row = {'epoch': epoch, 'loss': float(np.mean(losses)), 'train_secs': round(train_secs, 3),
               'val_secs': round(time.perf_counter() - start - train_secs, 3),
               'epoch_secs': round(time.perf_counter() - start, 3), 'train_windows': windows,
               'build_secs': round(build_secs, 3),
               'peak_gpu_mb': round(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == 'cuda' else 0,
               'peak_ram_mb': peak_ram_mb(), **{f'val_{k}': scores[k] for k in ('macro_f1', 'accuracy', 'auc', 'recall')}}
        history.append(row)
        log(json.dumps(row))
        if scores['macro_f1'] > best_f1:
            best_f1, best_f1_epoch = scores['macro_f1'], epoch
            torch.save(net.state_dict(), out / 'best.pt')
        if scores['auc'] > best_auc:
            best_auc, best_auc_epoch = scores['auc'], epoch
            torch.save(net.state_dict(), out / 'best_auc.pt')
        if epoch - best_auc_epoch >= cfg.patience:
            break
    with (out / 'history.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    by_rule, thresholds = {}, {}
    for rule, ckpt in (('f1', 'best.pt'), ('auc', 'best_auc.pt')):
        net.load_state_dict(torch.load(out / ckpt, map_location=device))
        by_rule[rule] = {'val': predict(net, val, device, window_budget, cfg.num_workers),
                         'test': predict(net, test, device, window_budget, cfg.num_workers)}
        thresholds[rule] = val_threshold_of(manifest.val, *by_rule[rule]['val'][:2])
    extra = {
        'best_epoch': best_f1_epoch, 'epochs_run': len(history), 'val_macro_f1': best_f1, 'auc_best_epoch': best_auc_epoch,
        'val_auc': best_auc, 'mean_train_epoch_secs': float(np.mean([h['train_secs'] for h in history])),
        'epoch_secs_mean': float(np.mean([h['epoch_secs'] for h in history])),
        'peak_gpu_mb': max(h['peak_gpu_mb'] for h in history), 'peak_ram_mb': peak_ram_mb(),
        'export': str(store.root), 'export_sha': store.export_sha, 'n_ids': train.n_ids, 'n_parameters': n_params,
        'window_budget': window_budget, 'device': device.type, 'smoke': smoke,
        'model': {'d_model': model.d_model, 'max_seq_len': model.max_seq_len,
                  'num_layers': len(model.transformer_encoder.layers)}}
    metrics = write_run(out, cfg, by_rule['f1'], by_rule['auc'], thresholds['f1'], thresholds['auc'], extra,
                        manifest=manifest)
    log(f"{cfg.name}: {metrics['epoch_secs_mean']:.1f} s/epoch over {len(history)} epochs (train "
        f"{metrics['mean_train_epoch_secs']:.1f}), peak GPU {metrics['peak_gpu_mb']} MB, peak RAM {metrics['peak_ram_mb']} MB")
    if smoke is not None:
        full = {s: split_size(FoldSequences(store, cfg.fold, s, L=cfg.seq_len, max_windows=cfg.max_windows))
                for s in ('train', 'val')}
        metrics['projection'] = projection(history, split_size(train), split_size(val), full['train'], full['val'],
                                           full_epochs, cfg.patience)
        log('projection ' + json.dumps(metrics['projection']))
    log(json.dumps({k: metrics[k] for k in ('run', 'best_epoch', 'val_macro_f1', 'test')}, indent=2))
    return metrics


def parse_config(argv=None):
    """(RunConfig, options) from a command line; options: export, device, smoke, window_budget."""
    ap = argparse.ArgumentParser(description='Train Model 1 (token-sequence transformer) on one fold of a block export')
    ap.add_argument('--protocol', choices=['kfold'], default='kfold')
    ap.add_argument('--arch', choices=['x86'], default='x86')
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--seq-len', type=int, default=512)
    ap.add_argument('--max-windows', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--export', default=EXPORT_NAME, help='export directory, or a name under <runs>/export (GT_EXPORT)')
    ap.add_argument('--device', choices=['cuda', 'cpu'], default=None, help='default: cuda when available')
    ap.add_argument('--smoke', type=int, default=None, help='N train binaries, 2 epochs, temporary directory, no RUNS')
    ap.add_argument('--window-budget', type=int, default=WINDOW_BUDGET, help='windows per batch (peak memory)')
    args = ap.parse_args(argv)
    try:
        cfg = RunConfig(protocol=args.protocol, fold=args.fold, arch=args.arch, seed=args.seed, model='tokenseq',
                        seq_len=args.seq_len, max_windows=args.max_windows, epochs=args.epochs, patience=args.patience)
    except ValueError as e:
        ap.error(str(e))
    if args.smoke is not None and args.smoke < 2:
        ap.error('--smoke needs at least 2 binaries')
    if args.window_budget < 1:
        ap.error('--window-budget is at least 1')
    return cfg, {'export': args.export, 'device': args.device, 'smoke': args.smoke, 'window_budget': args.window_budget}


def main(argv=None):
    cfg, opts = parse_config(argv)
    run(cfg, **opts)


if __name__ == '__main__':
    main()
