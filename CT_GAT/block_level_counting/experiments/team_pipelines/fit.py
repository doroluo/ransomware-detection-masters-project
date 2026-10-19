"""The teammates' training recipe (train_cfg.train: class-weighted cross-entropy, AdamW, early stopping on the validation
ransomware F1 at 0.5, best state restored) on OUR manifests, with a second selection rule kept next to it, written as a run
directory by `common.write_run`. Shared by train_a (BlockGAT) and train_b (FullBlockGAT).

What is theirs: the model, the optimiser, the loss weights (N / n_class of the training graphs), `_batches` (their
shuffling, `random.Random(seed)`), `predict_proba` and `metrics_at` from train_cfg, the stopping rule (the F1 state has
not improved for `patience` epochs) and the F1 state itself (ransomware-class F1 at 0.5, first epoch wins ties).
What is ours: the train / val / test ids (`common.fold_manifest`), the second state (best validation AUC, strictly
greater, tracked while the F1 rule keeps training) with its validation threshold (`val_threshold_of`), and the output.
"""
import csv
import json
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common.config import RUNS
from experiments.team_pipelines.common import (evaluate, fold_manifest, import_team, sample_rows, team_config,
                                               val_threshold_of, write_run)


def configure_threads():
    """Cap the CPU threads of this process at $TEAM_THREADS when set (team_runs.sh sets it so the CPU trainers leave the
    cores the pipeline-B build and the GPU queue's loaders use); returns the thread count torch will use."""
    n = os.environ.get('TEAM_THREADS')
    if n:
        torch.set_num_threads(int(n))
    return torch.get_num_threads()


def _state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def predict_rows(tc, model, ids, graphs, sizes, device, batch_size):
    """(ids, probs [n] float64, sizes) of `ids` in the given order; sizes[i] is the (n_segments_total, n_blocks_kept,
    n_blocks_total) triple that write_predictions prints."""
    probs, labels = tc.predict_proba(model, [graphs[i] for i in ids], device, batch_size)
    return list(ids), probs.numpy().astype(np.float64), [sizes[i] for i in ids], labels.numpy()


def fit(model, train_ids, val_ids, graphs, *, hp, seed, device, log=print):
    """Train `model` on graphs[train_ids]; returns a dict with the F1 state, the AUC state, the history and the timing.
    hp holds lr, weight_decay, batch_size, epochs, patience."""
    tc = import_team('train_cfg')
    rows = sample_rows()
    train = [graphs[i] for i in train_ids]
    labels = torch.tensor([int(g.y.item()) for g in train])
    counts = torch.bincount(labels, minlength=2).float().clamp(min=1)
    criterion = nn.CrossEntropyLoss(weight=(counts.sum() / counts).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    rng = random.Random(seed)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    val_rows = [rows[i] for i in val_ids]
    best_f1, best_f1_epoch, f1_state = -1.0, 0, None
    best_auc, best_auc_epoch, auc_state = -1.0, 0, None
    history, stale = [], 0
    for epoch in range(1, hp['epochs'] + 1):
        model.train()
        start, total_loss, seen = time.time(), 0.0, 0
        for batch in tc._batches(train, hp['batch_size'], rng):
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * batch.num_graphs
            seen += batch.num_graphs
        train_secs = time.time() - start
        probs, y = tc.predict_proba(model, [graphs[i] for i in val_ids], device, hp['batch_size'])
        val_f1 = tc.metrics_at(probs, y, 0.5)['f1']
        val = evaluate(val_rows, list(val_ids), probs.numpy().astype(np.float64), 0.5)
        row = {'epoch': epoch, 'loss': round(total_loss / max(seen, 1), 5), 'train_secs': round(train_secs, 1),
               'val_secs': round(time.time() - start - train_secs, 1), 'val_f1': round(val_f1, 5),
               'val_macro_f1': round(val['macro_f1'], 5), 'val_auc': round(val['auc'], 5), 'val_acc': round(val['accuracy'], 5),
               'peak_gpu_mb': round(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == 'cuda' else 0}
        history.append(row)
        log(json.dumps(row))
        if val_f1 > best_f1:
            best_f1, best_f1_epoch, f1_state, f1_macro = val_f1, epoch, _state(model), val['macro_f1']
            stale = 0
        else:
            stale += 1
        if val['auc'] > best_auc:
            best_auc, best_auc_epoch, auc_state = val['auc'], epoch, _state(model)
        if stale >= hp['patience']:
            log(f'early stop at epoch {epoch}')
            break
    return {'history': history, 'f1_state': f1_state, 'f1_epoch': best_f1_epoch, 'val_f1': best_f1,
            'val_macro_f1': f1_macro, 'auc_state': auc_state, 'auc_epoch': best_auc_epoch, 'val_auc': best_auc}


def stratified(ids, labels, n):
    """n of `ids` (in order), half of each class where the ids allow it, else filled from the other class."""
    ones = [i for i in ids if labels[i] == 1]
    zeros = [i for i in ids if labels[i] == 0]
    ones_part = ones[:n // 2]
    zeros_part = zeros[:n - len(ones_part)]
    return ones[:n - len(zeros_part)] + zeros_part


def smoke_ids(manifest, available, n):
    """(train, val) ids for a smoke run: n available training ids of both classes and the available validation ids. With
    fewer than 2 validation ids of one class, the validation set is instead max(2, n // 4) available training ids of
    each class, taken out of the training pool first."""
    labels = {r['sample_id']: r['label'] for r in manifest.train + manifest.val}
    pool = [r['sample_id'] for r in manifest.train if r['sample_id'] in available]
    val = [r['sample_id'] for r in manifest.val if r['sample_id'] in available]
    if sum(labels[i] for i in val) < 2 or sum(1 - labels[i] for i in val) < 2:
        k = max(2, n // 4)
        val = [i for i in pool if labels[i] == 1][:k] + [i for i in pool if labels[i] == 0][:k]
        pool = [i for i in pool if i not in set(val)]
    return stratified(pool, labels, n), val


def run_arm(*, arm, fold, seed, graphs, sizes, build_model, hp, device, smoke=None, smoke_split=None, extra=None,
            log=print):
    """Train one (arm, fold, seed). graphs / sizes map sample_id -> PyG Data / (n_segments, kept, total); build_model is
    called after torch.manual_seed(seed). Without `smoke` every id of the fold's manifest must be in graphs and the run is
    written to RUNS/<team_config name>. With smoke=N it trains on N available training ids for 2 epochs and writes only
    a temp directory (a mkdtemp, never under RUNS; smoke_split = (train ids, val ids) overrides the choice): returns that directory's path and the history."""
    tc = import_team('train_cfg')
    threads = configure_threads()
    manifest = fold_manifest(fold, seed)
    cfg = team_config(fold, seed, arm)
    random.seed(seed)
    torch.manual_seed(seed)
    model = build_model().to(device)
    if smoke:
        train_ids, val_ids = smoke_split or smoke_ids(manifest, set(graphs), smoke)
        hp = {**hp, 'epochs': 2}
        out = Path(tempfile.mkdtemp(prefix=f'team_smoke_{arm}_'))
        log(f'smoke {cfg.name}: train {len(train_ids)} val {len(val_ids)} (temp dir {out}, 2 epochs, device {device}, '
            f'{threads} torch threads)')
    else:
        need = [r['sample_id'] for r in manifest.train + manifest.val + manifest.test]
        missing = [i for i in need if i not in graphs]
        if missing:
            raise SystemExit(f'{len(missing)} of the fold\'s {len(need)} binaries have no graph, e.g. {missing[:3]}')
        train_ids = [r['sample_id'] for r in manifest.train]
        val_ids = [r['sample_id'] for r in manifest.val]
        out = RUNS / cfg.name
        log(f'{cfg.name}: train {len(train_ids)} val {len(val_ids)} test {len(manifest.test)} device {device}, '
            f'{threads} torch threads')
    result = fit(model, train_ids, val_ids, graphs, hp=hp, seed=seed, device=device, log=log)
    history = result['history']
    peak = max(h['peak_gpu_mb'] for h in history)
    mean_epoch = float(np.mean([h['train_secs'] + h['val_secs'] for h in history]))
    log(f"{cfg.name}: {mean_epoch:.1f} s/epoch over {len(history)} epochs, loss {history[0]['loss']} -> "
        f"{history[-1]['loss']}, best val F1 {result['val_f1']:.3f} (epoch {result['f1_epoch']}), best val AUC "
        f"{result['val_auc']:.3f} (epoch {result['auc_epoch']}), peak GPU {peak} MB")
    if smoke:
        out.mkdir(parents=True, exist_ok=True)
        write_history(out / 'history.csv', history)
        return {'dir': out, 'history': history, 'val_auc': result['val_auc'], 'peak_gpu_mb': peak,
                'device': str(device), 'epoch_secs_mean': mean_epoch, 'train_ids': train_ids, 'val_ids': val_ids}

    by_split, states = {}, {}
    for rule in ('f1', 'auc'):
        model.load_state_dict(result[f'{rule}_state'])
        by_split[rule] = {}
        for split, ids in (('val', [r['sample_id'] for r in manifest.val]),
                           ('test', [r['sample_id'] for r in manifest.test])):
            ids_out, probs, size, _ = predict_rows(tc, model, ids, graphs, sizes, device, hp['batch_size'])
            by_split[rule][split] = (ids_out, probs, size)
        states[rule] = result[f'{rule}_state']
    thr_f1 = val_threshold_of(manifest.val, *by_split['f1']['val'][:2])
    thr_auc = val_threshold_of(manifest.val, *by_split['auc']['val'][:2])
    kept = [s[1] for s in by_split['f1']['test'][2]]
    total = [s[2] for s in by_split['f1']['test'][2]]
    info = {'best_epoch': result['f1_epoch'], 'epochs_run': len(history), 'auc_best_epoch': result['auc_epoch'],
            'val_auc': result['val_auc'], 'val_macro_f1': result['val_macro_f1'], 'val_f1_ransomware': result['val_f1'],
            'peak_gpu_mb': peak, 'epoch_secs_mean': mean_epoch, 'device': str(device), 'torch_threads': threads,
            'mean_train_epoch_secs': float(np.mean([h['train_secs'] for h in history])),
            'n_blocks_kept': {'test_sum': int(sum(kept)), 'test_mean': float(np.mean(kept)), 'test_max': int(max(kept))},
            'n_blocks_total': {'test_sum': int(sum(total)), 'test_mean': float(np.mean(total)),
                               'test_max': int(max(total))},
            'recipe': {**hp, 'arm': arm, 'selection_f1': 'ransomware F1 at 0.5 (train_cfg.metrics_at), stop on it',
                       'selection_auc': 'validation AUC, tracked while training', 'device': str(device)},
            **(extra or {})}
    metrics = write_run(out, cfg, by_split['f1'], by_split['auc'], thr_f1, thr_auc, info)
    write_history(out / 'history.csv', history)
    torch.save(states['f1'], out / 'best.pt')
    torch.save(states['auc'], out / 'best_auc.pt')
    log(f"{cfg.name}: test AUC {metrics['test']['auc']:.4f} macro-F1 {metrics['test']['macro_f1']:.4f} (F1 state), "
        f"AUC state at val threshold {thr_auc:.2f}: AUC {metrics['selection']['auc']['test']['auc']:.4f}")
    return {'dir': out, 'metrics': metrics, 'history': history}


def write_history(path, history):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
