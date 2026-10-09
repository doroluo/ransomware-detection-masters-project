"""One run: python train.py --protocol bundle --arch x86 --seed 0 [--no-graph] | --protocol kfold --fold 2

Writes config.json, manifest.json, vocab.json (sizes), history.csv, best.pt, predictions.csv and metrics.json
to graph_transformer_runs/<run name>/.
"""
import argparse
import csv
import json
import random
import time
from dataclasses import fields

import numpy as np
import torch
import torch.nn.functional as F

from config import RUNS, RunConfig
from data import TokenVocab, load_index, loader, make_manifest
from metrics import family_recall, floors, scores
from model import GraphTransformer


def parse_config():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', choices=['bundle', 'kfold'], default='bundle')
    ap.add_argument('--fold', type=int)
    ap.add_argument('--arch', choices=['x86', 'all'], default='x86')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-graph', action='store_true')
    ap.add_argument('--out', help='run directory name override (default: derived from the config)')
    for f in fields(RunConfig):
        if f.name not in ('protocol', 'fold', 'arch', 'seed', 'no_graph'):
            ap.add_argument('--' + f.name.replace('_', '-'), type=type(f.default), default=f.default)
    args = vars(ap.parse_args())
    out = args.pop('out')
    return RunConfig(**args), out


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def predict(model, data, device):
    model.eval()
    ids, probs, n_seg = [], [], []
    for batch in data:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(to_device(batch, device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        ids += batch['sample_id']
        n_seg += batch['n_segments_total']
    return ids, np.concatenate(probs), n_seg


def evaluate(rows, ids, probs, threshold):
    by_id = {r['sample_id']: r for r in rows}
    y = np.array([by_id[i]['label'] for i in ids])
    arch = np.array([by_id[i]['arch'] for i in ids])
    return scores(y, probs, (probs >= threshold).astype(int), arch)


def run(cfg: RunConfig, out_name=None):
    out = RUNS / (out_name or cfg.name)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / 'config.json')
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device('cuda')

    manifest = make_manifest(cfg, load_index())
    manifest.save(out / 'manifest.json')
    vocab = TokenVocab(manifest.train, cfg.min_df)
    (out / 'vocab.json').write_text(json.dumps(vocab.summary(), indent=2), encoding='utf-8')

    model = GraphTransformer(cfg, vocab.total).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    labels = np.array([r['label'] for r in manifest.train])
    class_weight = torch.tensor([len(labels) / (2 * max(1, (labels == c).sum())) for c in (0, 1)], device=device)
    train_data = loader(manifest.train, vocab, cfg, train=True)
    val_data = loader(manifest.val, vocab, cfg, train=False)
    print(f'{cfg.name}: train {len(manifest.train)} val {len(manifest.val)} test {len(manifest.test)} '
          f'val families {manifest.val_families} vocab {vocab.summary()}', flush=True)

    history, best_f1, best_epoch = [], -1.0, 0
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, cfg.epochs + 1):
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
        row = {'epoch': epoch, 'loss': float(np.mean(losses)), 'train_secs': round(train_secs, 1),
               'val_secs': round(time.time() - start - train_secs, 1),
               'peak_gpu_mb': round(torch.cuda.max_memory_allocated(device) / 2**20),
               **{f'val_{k}': val[k] for k in ('macro_f1', 'accuracy', 'auc', 'recall')}}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val['macro_f1'] > best_f1:
            best_f1, best_epoch = val['macro_f1'], epoch
            torch.save(model.state_dict(), out / 'best.pt')
        elif epoch - best_epoch >= cfg.patience:
            break
    with (out / 'history.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    model.load_state_dict(torch.load(out / 'best.pt', map_location=device))
    test_data = loader(manifest.test, vocab, cfg, train=False)
    by_split = {'val': predict(model, val_data, device), 'test': predict(model, test_data, device)}
    rows = {r['sample_id']: r for r in manifest.val + manifest.test}
    with (out / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'split', 'label', 'prob', 'pred', 'arch', 'family', 'fold', 'n_segments_total'])
        for split, (ids, probs, n_seg) in by_split.items():
            for i, p, s in zip(ids, probs, n_seg):
                r = rows[i]
                w.writerow([i, split, r['label'], f'{p:.6f}', int(p >= cfg.threshold), r['arch'], r['family'], r['fold'], s])

    ids, probs, _ = by_split['test']
    test = evaluate(manifest.test, ids, probs, cfg.threshold)
    by_id = {r['sample_id']: r for r in manifest.test}
    metrics = {
        'run': cfg.name, 'best_epoch': best_epoch, 'epochs_run': len(history), 'val_macro_f1': best_f1,
        'test': test,
        'floors': floors(labels, [by_id[i]['label'] for i in ids], [by_id[i]['arch'] for i in ids]),
        'family_recall': family_recall([by_id[i]['label'] for i in ids], (probs >= cfg.threshold).astype(int),
                                       [by_id[i]['family'] for i in ids]),
        'mean_train_epoch_secs': float(np.mean([h['train_secs'] for h in history])),
        'peak_gpu_mb': max(h['peak_gpu_mb'] for h in history),
    }
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps({k: metrics[k] for k in ('run', 'best_epoch', 'val_macro_f1', 'test')}, indent=2), flush=True)
    return metrics


if __name__ == '__main__':
    run(*parse_config())
