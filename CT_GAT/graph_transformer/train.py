"""One run: python train.py --protocol bundle --arch x86 --seed 0 [--no-graph] | --protocol kfold --fold 2
| --protocol kfold_v3 --fold 2 --arch all|matched (reads the combined cache unless GT_CACHE is set)
[--model gnn --control no_edges --encoder mean --untyped --fixed-epochs N --direction forward --drop-edges 0.2
--flip-edges 0.2] [--train-fraction 0.5]

Writes config.json (with the cache path used), manifest.json, vocab.json (sizes), history.csv, val_probs.csv, best.pt
(best validation macro-F1 at cfg.threshold), best_auc.pt (best validation AUC), predictions.csv (F1 checkpoint, pred at
cfg.threshold), predictions_auc.csv (AUC checkpoint, pred at its validation threshold) and metrics.json to
graph_transformer_runs/<run name>/.
"""
import argparse
import csv
import json
import random
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import DIRECTIONS, RUNS, RunConfig
from data import CACHE, TokenVocab, load_index, loader, make_manifest
from metrics import family_recall, floors, scores
from model import GraphTransformer
from model_gnn import HierGNN
from val_threshold import GRID, macro_f1


def parse_config(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', choices=['bundle', 'kfold', 'kfold_v3'], default='bundle')
    ap.add_argument('--fold', type=int)
    ap.add_argument('--arch', choices=['x86', 'all', 'matched'], default='x86')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-graph', action='store_true')
    ap.add_argument('--model', choices=['transformer', 'gnn'], default='transformer')
    ap.add_argument('--encoder', choices=['mean', 'tfidf'], default=None)
    ap.add_argument('--readout', choices=['fixed', 'gated'], default='fixed')
    ap.add_argument('--control', choices=['', 'no_edges', 'chain', 'shuffle', 'permute_types'], default='')
    ap.add_argument('--direction', choices=DIRECTIONS, default='both')
    ap.add_argument('--out', help='run directory name override (default: derived from the config)')
    done = {a.dest for a in ap._actions}
    for f in fields(RunConfig):
        if f.name in done:
            continue
        flag = '--' + f.name.replace('_', '-')
        if isinstance(f.default, bool):
            ap.add_argument(flag, action='store_true')
        else:
            ap.add_argument(flag, type=type(f.default), default=f.default)
    args = vars(ap.parse_args(argv))
    out = args.pop('out')
    try:
        return RunConfig(**args), out
    except ValueError as e:
        ap.error(str(e))


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def predict(model, data, device):
    model.eval()
    ids, probs, sizes = [], [], []
    for batch in data:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(to_device(batch, device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        ids += batch['sample_id']
        none = [None] * len(batch['sample_id'])
        sizes += zip(batch['n_segments_total'], batch.get('n_blocks_kept', none), batch.get('n_blocks_total', none))
    return ids, np.concatenate(probs), sizes


def evaluate(rows, ids, probs, threshold):
    by_id = {r['sample_id']: r for r in rows}
    y = np.array([by_id[i]['label'] for i in ids])
    arch = np.array([by_id[i]['arch'] for i in ids])
    return scores(y, probs, (probs >= threshold).astype(int), arch)


def check_supported(cfg: RunConfig):
    if cfg.stack_tfidf:
        raise NotImplementedError('--stack-tfidf needs out-of-fold TF-IDF logits (deferred)')
    if cfg.aux_recon > 0 or cfg.aux_edge > 0:
        raise NotImplementedError('--aux-recon / --aux-edge are deferred')


def build_model(cfg: RunConfig, vocab):
    return (HierGNN if cfg.model == 'gnn' else GraphTransformer)(cfg, vocab.total)


def should_stop(cfg: RunConfig, epoch, best_f1_epoch, best_auc_epoch):
    """Transformer: patience on val macro-F1 only. gnn: patience on the later of the F1 and AUC bests.
    fixed_epochs > 0 never stops early."""
    if cfg.fixed_epochs > 0:
        return False
    last_best = best_f1_epoch if cfg.model == 'transformer' else max(best_f1_epoch, best_auc_epoch)
    return epoch - last_best >= cfg.patience


def val_threshold_of(rows, ids, probs):
    """The GRID threshold with the highest validation macro-F1 (first on ties)."""
    by_id = {r['sample_id']: r['label'] for r in rows}
    y = np.array([by_id[i] for i in ids])
    return float(max(GRID, key=lambda g: macro_f1(y, probs, g)))


def write_predictions(path, rows, by_split, threshold):
    by_id = {r['sample_id']: r for r in rows}
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'split', 'label', 'prob', 'pred', 'arch', 'family', 'fold', 'n_segments_total',
                    'n_blocks_kept', 'n_blocks_total'])
        for split, (ids, probs, sizes) in by_split.items():
            for i, p, (n_seg, kept, total) in zip(ids, probs, sizes):
                r = by_id[i]
                w.writerow([i, split, r['label'], f'{p:.6f}', int(p >= threshold), r['arch'], r['family'], r['fold'],
                            n_seg, '' if kept is None else kept, '' if total is None else total])


def truncated_count(sizes):
    """Binaries whose kept blocks are fewer than their blocks; `sizes` are (n_segments_total, kept, total) triples
    (kept and total are None on the transformer path, which does not record them)."""
    return sum(kept is not None and kept < total for _, kept, total in sizes)


def run(cfg: RunConfig, out_name=None):
    check_supported(cfg)
    if cfg.cache and Path(cfg.cache) != CACHE:
        raise ValueError(f'config says cache {cfg.cache} but this process reads {CACHE}')
    cfg = replace(cfg, cache=str(CACHE))
    out = RUNS / (out_name or cfg.name)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / 'config.json')
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device('cuda')

    manifest = make_manifest(cfg, load_index())
    manifest.save(out / 'manifest.json')
    vocab = TokenVocab(manifest.train, cfg)
    (out / 'vocab.json').write_text(json.dumps(vocab.summary(), indent=2), encoding='utf-8')

    model = build_model(cfg, vocab).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    labels = np.array([r['label'] for r in manifest.train])
    class_weight = torch.tensor([len(labels) / (2 * max(1, (labels == c).sum())) for c in (0, 1)], device=device)
    train_data = loader(manifest.train, vocab, cfg, train=True)
    val_data = loader(manifest.val, vocab, cfg, train=False)
    print(f'{cfg.name}: train {len(manifest.train)} val {len(manifest.val)} test {len(manifest.test)} '
          f'val families {manifest.val_families} vocab {vocab.summary()}', flush=True)

    history, best_f1, best_f1_epoch, best_auc, best_auc_epoch = [], -1.0, 0, -1.0, 0
    with (out / 'val_probs.csv').open('w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(['epoch', 'sample_id', 'prob'])
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, (cfg.fixed_epochs or cfg.epochs) + 1):
        model.train()
        start, losses = time.time(), []
        for batch in train_data:
            batch = to_device(batch, device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(batch)
            loss = (F.binary_cross_entropy_with_logits(logits, batch['label'], reduction='none')
                    * class_weight[batch['label'].long()]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(loss.item())
        train_secs = time.time() - start
        ids, probs, _ = predict(model, val_data, device)
        val = evaluate(manifest.val, ids, probs, cfg.threshold)
        with (out / 'val_probs.csv').open('a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows((epoch, i, f'{p:.6f}') for i, p in zip(ids, probs))
        row = {'epoch': epoch, 'loss': float(np.mean(losses)), 'train_secs': round(train_secs, 1),
               'val_secs': round(time.time() - start - train_secs, 1), 'epoch_secs': round(time.time() - start, 1),
               'peak_gpu_mb': round(torch.cuda.max_memory_allocated(device) / 2**20),
               **{f'val_{k}': val[k] for k in ('macro_f1', 'accuracy', 'auc', 'recall')}}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val['macro_f1'] > best_f1:
            best_f1, best_f1_epoch = val['macro_f1'], epoch
            torch.save(model.state_dict(), out / 'best.pt')
        if val['auc'] > best_auc:
            best_auc, best_auc_epoch = val['auc'], epoch
            torch.save(model.state_dict(), out / 'best_auc.pt')
        if should_stop(cfg, epoch, best_f1_epoch, best_auc_epoch):
            break
    with (out / 'history.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    test_data = loader(manifest.test, vocab, cfg, train=False)
    rows = manifest.val + manifest.test
    by_id = {r['sample_id']: r for r in manifest.test}
    chosen = {}
    for rule, ckpt in (('f1', 'best.pt'), ('auc', 'best_auc.pt')):
        model.load_state_dict(torch.load(out / ckpt, map_location=device))
        by_split = {'val': predict(model, val_data, device), 'test': predict(model, test_data, device)}
        threshold_val = val_threshold_of(manifest.val, *by_split['val'][:2])
        chosen[rule] = (by_split, threshold_val)
        write_predictions(out / ('predictions.csv' if rule == 'f1' else 'predictions_auc.csv'), rows, by_split,
                          cfg.threshold if rule == 'f1' else threshold_val)

    def test_scores(rule, threshold):
        ids, probs, _ = chosen[rule][0]['test']
        return evaluate(manifest.test, ids, probs, threshold)

    def test_family_recall(rule, threshold):
        ids, probs, _ = chosen[rule][0]['test']
        return family_recall([by_id[i]['label'] for i in ids], (probs >= threshold).astype(int),
                             [by_id[i]['family'] for i in ids])

    ids = chosen['f1'][0]['test'][0]
    thr_f1, thr_auc = chosen['f1'][1], chosen['auc'][1]
    metrics = {
        'run': cfg.name, 'best_epoch': best_f1_epoch, 'epochs_run': len(history), 'val_macro_f1': best_f1,
        'truncated_test': truncated_count(chosen['f1'][0]['test'][2]),
        'test': test_scores('f1', cfg.threshold),
        'floors': floors(labels, [by_id[i]['label'] for i in ids], [by_id[i]['arch'] for i in ids]),
        'family_recall': test_family_recall('f1', cfg.threshold),
        'mean_train_epoch_secs': float(np.mean([h['train_secs'] for h in history])),
        'epoch_secs_mean': float(np.mean([h['epoch_secs'] for h in history])),
        'peak_gpu_mb': max(h['peak_gpu_mb'] for h in history),
        'selection': {
            'f1': {'best_epoch': best_f1_epoch, 'threshold_val': thr_f1,
                   'test_at_val_threshold': test_scores('f1', thr_f1)},
            'auc': {'best_epoch': best_auc_epoch, 'val_auc': best_auc, 'threshold_val': thr_auc,
                    'test': test_scores('auc', thr_auc), 'test_at_half': test_scores('auc', 0.5),
                    'family_recall': test_family_recall('auc', thr_auc)},
        },
    }
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(f"{cfg.name}: {metrics['epoch_secs_mean']:.1f} s/epoch over {len(history)} epochs "
          f"(train {metrics['mean_train_epoch_secs']:.1f}), peak {metrics['peak_gpu_mb']} MB", flush=True)
    print(json.dumps({k: metrics[k] for k in ('run', 'best_epoch', 'val_macro_f1', 'test')}, indent=2), flush=True)
    return metrics


if __name__ == '__main__':
    run(*parse_config())
