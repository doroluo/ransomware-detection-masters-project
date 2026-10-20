"""One run: python train.py --protocol bundle --arch x86 --seed 0 | --protocol kfold --fold 2
| --protocol kfold_v3 --fold 2 --arch all|matched (reads the combined cache unless GT_CACHE is set)
[--control no_edges --encoder mean --untyped --fixed-epochs N --direction forward --drop-edges 0.2
--flip-edges 0.2] [--train-fraction 0.5] [--arch-balance] [--val-seed 0]
[--block-input vec --block-vec <export>_<name>] [--stack-tfidf] [--vocab yanping|hybrid] [--imports header]

Writes config.json (with the cache path used), manifest.json, vocab.json (sizes), history.csv, val_probs.csv, best.pt
(best validation macro-F1 at cfg.threshold), best_auc.pt (best validation AUC), predictions.csv (F1 checkpoint, pred at
cfg.threshold), predictions_auc.csv (AUC checkpoint, pred at its validation threshold) and metrics.json to
graph_transformer_runs/<run name>/.

--arch-balance weights each training sample's loss by N / (C n_cell) over the non-empty (label, arch) cells of the
training manifest instead of by its class weight; every metrics block gains `cell_balanced` (each cell weighs equally).

--val-seed k carves the validation set (and the train-fraction subsample) with seed k instead of --seed, so train, val and
test do not change with the seed; the run name gains `-vs<k>` when k differs from the seed.

--block-input vec replaces the token bag of step 5 by the precomputed block vectors <cache>/side/blockvec/<block-vec>/
fold<k>/ (transformer_paths/precompute_blockvec.py). `--control no_edges --block-input vec` is path 1 of the transformer
paths plan and the typed GNN with it is path 2. Fold runs of protocol kfold only.

--stack-tfidf feeds the out-of-fold TF-IDF logit of `baselines.py --stack-logits` to the head: it is read from
graph_transformer_runs/baselines/<baseline dir>/stack_logits.csv by sample id and standardised on the training rows.

--vocab yanping|hybrid (an experiment, not on the main path) swaps the instruction tokens of the block bag for the team's operand-role ids (<cache>/side/yanping/,
experiments/side_arrays/import_reader_side.py): yanping = the reader's three ids per instruction plus our api tokens,
hybrid = our mn and beh plus the reader's two operand ids plus api. The run name gains `-voc-<name>`.

--imports header adds an EmbeddingBag over every header import name (vocabulary and idf fitted on the training binaries,
min_df 2) and two log counts (log1p of the unreferenced and of the header imports) to the head input (<cache>/side/imports/,
pipeline/s10_unused_import_features); the run name gains `-imph`.

--select-on cell picks best_auc.pt by the validation cell-balanced AUC instead of the validation AUC, and the validation
threshold of the AUC checkpoint (threshold_val, predictions_auc.csv) by the validation cell-balanced macro-F1 instead of
the macro-F1. best.pt, the F1 checkpoint and its threshold are unchanged. As the gnn stops on the later of the F1 and
AUC bests, the AUC rule also moves the stopping epoch. metrics.json records the rule (`select_on`, `selection.auc.rule`).
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

from common.config import BLOCK_INPUTS, CACHE, DIRECTIONS, ENCODERS, IMPORTS, RUNS, VOCABS, RunConfig
from common.manifests import load_index, make_manifest
from evaluation.metrics import cell_balanced, cell_weights, family_recall, floors, scores
from evaluation.val_threshold import GRID, macro_f1
from pipeline.batching import loader
from pipeline.model import HierGNN
from pipeline.s03_tokenizer.vocab import TokenVocab
from pipeline.s10_unused_import_features.imports_feat import ImportVocab


AUC_RULES = {'overall': 'checkpoint: validation AUC; threshold: validation macro-F1',
             'cell': 'checkpoint: validation cell-balanced AUC; threshold: validation cell-balanced macro-F1'}


def parse_config(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', choices=['bundle', 'kfold', 'kfold_v3'], default='bundle')
    ap.add_argument('--fold', type=int)
    ap.add_argument('--arch', choices=['x86', 'all', 'matched'], default='x86')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--val-seed', type=int, default=None)
    ap.add_argument('--model', choices=['gnn'], default='gnn')
    ap.add_argument('--encoder', choices=ENCODERS, default=None)
    ap.add_argument('--readout', choices=['fixed', 'gated'], default='fixed')
    ap.add_argument('--control', choices=['', 'no_edges', 'chain', 'shuffle', 'permute_types'], default='')
    ap.add_argument('--direction', choices=DIRECTIONS, default='both')
    ap.add_argument('--block-input', choices=BLOCK_INPUTS, default='tokens')
    ap.add_argument('--vocab', choices=VOCABS, default='ours')
    ap.add_argument('--imports', choices=IMPORTS, default='')
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


def loss_weight_fn(cfg: RunConfig, rows, device):
    """batch -> per-sample loss weight. Default: the class weight N / (2 n_class). With cfg.arch_balance: N / (C n_cell)
    over the non-empty (label, arch) cells of `rows` (the training manifest), which replaces the class weight; both
    average 1 over `rows`."""
    labels = np.array([r['label'] for r in rows])
    if cfg.arch_balance:
        by_id = dict(zip((r['sample_id'] for r in rows), cell_weights(labels, [r['arch'] for r in rows])))
        return lambda batch: torch.tensor([by_id[i] for i in batch['sample_id']], dtype=torch.float32,
                                          device=device)
    class_weight = torch.tensor([len(labels) / (2 * max(1, (labels == c).sum())) for c in (0, 1)], device=device)
    return lambda batch: class_weight[batch['label'].long()]


def check_supported(cfg: RunConfig):
    if cfg.model == 'tokenseq':
        raise NotImplementedError("model 'tokenseq' trains with python -m token_sequence.train, not training/train.py")
    if cfg.aux_recon > 0 or cfg.aux_edge > 0:
        raise NotImplementedError('--aux-recon / --aux-edge are deferred')


def build_model(cfg: RunConfig, vocab, d_vec=None, n_imports=0):
    """The network of `cfg`. With block_input='vec' there is no vocabulary (`vocab` None) and `d_vec` is the size of
    the precomputed block vectors; `n_imports` is the size of the import vocabulary of an `imports` run."""
    return HierGNN(cfg, vocab.total if vocab else 0, d_vec, n_imports)


def open_block_vecs(cfg: RunConfig):
    """The BlockVecSource of a block_input='vec' run, checked against the run: same cache, protocol and arch as the export
    it was computed from."""
    from pipeline.s05_block_counting.block_vecs import BlockVecSource
    vecs = BlockVecSource(CACHE, cfg.block_vec, cfg.fold)
    meta = vecs.store_meta()
    if (meta['protocol'], meta['arch']) != (cfg.protocol, cfg.arch) or Path(meta['cache']) != Path(CACHE):
        raise ValueError(f"{cfg.block_vec} is for {meta['protocol']} {meta['arch']} on {meta['cache']}, this run is "
                         f'{cfg.protocol} {cfg.arch} on {CACHE}')
    return vecs


def stack_logits(cfg: RunConfig, manifest):
    """{sample_id: standardised out-of-fold TF-IDF logit} for train, val and test rows of `manifest`. The logits are
    read from RUNS/baselines/<baseline dir>/stack_logits.csv; the file must hold every row under the same role it has
    here (train rows out of fold, val and test rows from the full fit), and the standardisation uses the training rows
    only (mean and standard deviation)."""
    from evaluation.baselines import baseline_name, read_stack_logits
    raw = read_stack_logits(RUNS / 'baselines' / baseline_name(cfg) / 'stack_logits.csv', manifest)
    train = np.array([raw[r['sample_id']] for r in manifest.train])
    mean, std = float(train.mean()), float(max(train.std(), 1e-6))
    return {k: np.float32((v - mean) / std) for k, v in raw.items()}


def should_stop(cfg: RunConfig, epoch, best_f1_epoch, best_auc_epoch):
    """Patience on the later of the F1 and AUC bests. fixed_epochs > 0 never stops early."""
    if cfg.fixed_epochs > 0:
        return False
    last_best = max(best_f1_epoch, best_auc_epoch)
    return epoch - last_best >= cfg.patience


def val_auc_of(cfg: RunConfig, val):
    """The validation AUC that picks best_auc.pt: the plain one, or with select_on='cell' the cell-balanced one."""
    return val['cell_balanced']['auc'] if cfg.select_on == 'cell' else val['auc']


def val_threshold_of(rows, ids, probs, select_on='overall'):
    """The GRID threshold with the highest validation macro-F1 (first on ties); with select_on='cell' the highest
    cell-balanced macro-F1 (`metrics.cell_balanced`, each non-empty label x arch cell of `rows` weighs equally)."""
    by_id = {r['sample_id']: r for r in rows}
    y = np.array([by_id[i]['label'] for i in ids])
    if select_on == 'cell':
        arch = np.array([by_id[i]['arch'] for i in ids])
        return float(max(GRID, key=lambda g: cell_balanced(y, probs, (probs >= g).astype(int), arch)['macro_f1']))
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
    (kept and total are None when the batch carries no block counts)."""
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
    vecs = open_block_vecs(cfg) if cfg.block_input == 'vec' else None
    vocab = None if vecs else TokenVocab(manifest.train, cfg)
    (out / 'vocab.json').write_text(json.dumps(vocab.summary() if vocab else {'block_vec': cfg.block_vec, 'd': vecs.d},
                                               indent=2), encoding='utf-8')
    stack = stack_logits(cfg, manifest) if cfg.stack_tfidf else None
    imports = ImportVocab(manifest.train, cfg.min_df, cfg.imports) if cfg.imports else None
    if imports is not None:
        (out / 'imports_vocab.json').write_text(json.dumps(imports.summary(), indent=2), encoding='utf-8')

    model = build_model(cfg, vocab, vecs.d if vecs else None, len(imports) if imports is not None else 0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    labels = np.array([r['label'] for r in manifest.train])
    loss_weight = loss_weight_fn(cfg, manifest.train, device)
    train_data = loader(manifest.train, vocab, cfg, train=True, vecs=vecs, stack=stack, imports=imports)
    val_data = loader(manifest.val, vocab, cfg, train=False, vecs=vecs, stack=stack, imports=imports)
    print(f'{cfg.name}: train {len(manifest.train)} val {len(manifest.val)} test {len(manifest.test)} '
          f'val families {manifest.val_families} vocab {vocab.summary() if vocab else cfg.block_vec}', flush=True)

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
                    * loss_weight(batch)).mean()
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
        val_auc = val_auc_of(cfg, val)
        if val_auc > best_auc:
            best_auc, best_auc_epoch = val_auc, epoch
            torch.save(model.state_dict(), out / 'best_auc.pt')
        if should_stop(cfg, epoch, best_f1_epoch, best_auc_epoch):
            break
    with (out / 'history.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    test_data = loader(manifest.test, vocab, cfg, train=False, vecs=vecs, stack=stack, imports=imports)
    rows = manifest.val + manifest.test
    by_id = {r['sample_id']: r for r in manifest.test}
    chosen = {}
    for rule, ckpt in (('f1', 'best.pt'), ('auc', 'best_auc.pt')):
        model.load_state_dict(torch.load(out / ckpt, map_location=device))
        by_split = {'val': predict(model, val_data, device), 'test': predict(model, test_data, device)}
        threshold_val = val_threshold_of(manifest.val, *by_split['val'][:2],
                                         select_on=cfg.select_on if rule == 'auc' else 'overall')
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
        'run': cfg.name, 'select_on': cfg.select_on, 'best_epoch': best_f1_epoch, 'epochs_run': len(history),
        'val_macro_f1': best_f1,
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
            'auc': {'rule': AUC_RULES[cfg.select_on], 'best_epoch': best_auc_epoch, 'val_auc': best_auc,
                    'threshold_val': thr_auc,
                    'test': test_scores('auc', thr_auc), 'test_at_half': test_scores('auc', 0.5),
                    'family_recall': test_family_recall('auc', thr_auc)},
        },
    }
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(f"{cfg.name}: {metrics['epoch_secs_mean']:.1f} s/epoch over {len(history)} epochs "
          f"(train {metrics['mean_train_epoch_secs']:.1f}), peak {metrics['peak_gpu_mb']} MB", flush=True)
    print(json.dumps({k: metrics[k] for k in ('run', 'best_epoch', 'val_macro_f1', 'test')}, indent=2), flush=True)
    return metrics


def main():
    run(*parse_config())


if __name__ == '__main__':
    main()
