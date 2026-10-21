"""One run, from CT_GAT/pipeline: python -m experiments.train --protocol bundle --arch x86 --seed 0 | --protocol kfold --fold 2
| --protocol kfold_v3 --fold 2 --arch all|matched (the cache is the first dataset of datasets.json that lists the protocol,
`--cache <name>` picks another, GT_CACHE overrides) [--model gnn]
[--control no_edges --encoder mean --untyped --fixed-epochs N --direction forward --drop-edges 0.2
--flip-edges 0.2] [--train-fraction 0.5] [--arch-balance] [--val-seed 0]
[--block-input vec --block-vec <export>_<name>] [--stack-tfidf] [--vocab yanping|hybrid|noapi] [--imports header]

The loop is the same for every model; what a model builds and feeds is its `common.registry.ModelSpec` (`--model` names it).
The loop owns the class weights, AdamW, gradient clipping, patience, the two checkpoints, the validation thresholds and the
run directory (`common.run_io.write_run`).

Writes config.json (with the cache path used), manifest.json, vocab.json (sizes), history.csv, val_probs.csv, best.pt
(best validation macro-F1 at cfg.threshold), best_auc.pt (best validation AUC), predictions.csv (F1 checkpoint, pred at
cfg.threshold), predictions_auc.csv (AUC checkpoint, pred at its validation threshold) and metrics.json to
<runs>/<run name>/ (GT_RUNS, `runs_for` of common/paths.py).

--arch-balance weights each training sample's loss by N / (C n_cell) over the non-empty (label, arch) cells of the
training manifest instead of by its class weight; every metrics block gains `cell_balanced` (each cell weighs equally).

--val-seed k carves the validation set (and the train-fraction subsample) with seed k instead of --seed, so train, val and
test do not change with the seed; the run name gains `-vs<k>` when k differs from the seed.

--block-input vec replaces the token bag of step 5 by the precomputed block vectors <cache>/side/blockvec/<block-vec>/
fold<k>/ (s03c_cfg_transformer/precompute_blockvec.py; without that package the option stops with a message). `--control
no_edges --block-input vec` is path 1 of the transformer paths plan and the typed GNN with it is path 2. Fold runs of
protocol kfold only.

--stack-tfidf feeds the out-of-fold TF-IDF logit of `s04_evaluation/baselines.py --stack-logits` to the head: it is read from
<runs>/baselines/<baseline dir>/stack_logits.csv by sample id and standardised on the training rows.

--vocab yanping|hybrid (an experiment, not on the main path) swaps the instruction tokens of the block bag for the team's
operand-role ids (<cache>/side/yanping/, experiments/side_arrays/import_reader_side.py): yanping = the reader's three ids
per instruction plus our api tokens, hybrid = our mn and beh plus the reader's two operand ids plus api. The run name gains
`-voc-<name>`; noapi drops the api tokens.

--imports header adds an EmbeddingBag over every header import name (vocabulary and idf fitted on the training binaries,
min_df 2) and two log counts (log1p of the unreferenced and of the header imports) to the head input (<cache>/side/imports/,
s03b_cfg_feature/gnn/s10_unused_import_features); the run name gains `-imph`.

--select-on cell picks best_auc.pt by the validation cell-balanced AUC instead of the validation AUC, and the validation
threshold of the AUC checkpoint (threshold_val, predictions_auc.csv) by the validation cell-balanced macro-F1 instead of
the macro-F1. best.pt, the F1 checkpoint and its threshold are unchanged. As the gnn stops on the later of the F1 and
AUC bests, the AUC rule also moves the stopping epoch. metrics.json records the rule (`select_on`, `selection.auc.rule`).

--out names the run directory instead of the config's name (smoke runs).

--threshold-from oof (--model tokenseq|fivechannel; any model that gives `choose_thresholds` a refit can use it) chooses the
decision threshold of a run on out-of-fold predictions instead of the validation families (which saturate: validation AUC
near 1.0 within a few epochs, so the first threshold of a wide band wins and macro-F1 swings between folds). The training
binaries are split in 3 folds that keep a ransomware family or a goodware identity group whole, the model is refit on 2 of
them for the epoch count of each checkpoint, and the pooled held-out predictions give the GRID threshold with the highest
macro-F1. metrics.json records the rule (`threshold_rule`, `selection.auc.rule`); the run name gains `-thoof`. The default,
`--threshold-from val`, is the rule above and changes no name or result.

--model fivechannel trains Janice Park's five-channel Model 1 (s03a_transformer/five_channel) on the .pt files of a pipeline-B
build (experiments/team_pipelines/build_b.py) under our folds (`--protocol kfold --fold <k>`, x86). It takes --build <dir>
(default: the `team_b_build` source of datasets.json, or GT_TEAM_B_BUILD), --device and --smoke N, and --max-instructions N
(default 128, as she wrote it; 512 gives the run name suffix -L512). See that package's README.

--model tokenseq trains Model 1, the token-sequence transformer of s03a_transformer/token_sequence, on a fold of a block export
(`--protocol kfold --fold <k>`); its loop is its own and the registry hands the run over to it. It takes four more flags,
refused for other models: --export <name or directory> (default ours_cap0, under the export directory), --device cuda|cpu,
--smoke N (N binaries, 2 epochs, a temporary directory) and --window-budget N (windows per batch). See that package's README.
"""
import argparse
import csv
import json
import random
import shutil
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common.config import BLOCK_INPUTS, CACHE, DIRECTIONS, ENCODERS, IMPORTS, RUNS, THRESHOLD_FROM, VOCABS, RunConfig
from common.manifests import MissingCache, load_index, make_manifest, missing_cache_message, run_command
from common.paths import registry
from common.registry import MODELS, RunInfo, spec_for
from common.run_io import AUC_RULES, evaluate, truncated_count, write_predictions, write_run  # noqa: F401  (re-exported)
from s04_evaluation.metrics import cell_balanced, cell_weights
from s04_evaluation.val_threshold import GRID, macro_f1, oof_rule


# the flags of the models that own their loop and are not RunConfig fields, per model
MODEL_OPTIONS = {'tokenseq': ('export', 'device', 'smoke', 'window_budget'), 'fivechannel': ('build', 'device', 'smoke')}
ALL_OPTIONS = tuple(dict.fromkeys(o for options in MODEL_OPTIONS.values() for o in options))


def parse_args(argv=None):
    """(RunConfig, --out, options): `options` holds the flags of MODEL_OPTIONS[model] that were given, for the model that owns them."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', choices=['bundle', 'kfold', 'kfold_v3'], default='bundle')
    ap.add_argument('--fold', type=int)
    ap.add_argument('--arch', choices=['x86', 'all', 'matched'], default='x86')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--val-seed', type=int, default=None)
    ap.add_argument('--model', choices=list(MODELS), default='gnn')
    ap.add_argument('--encoder', choices=ENCODERS, default=None)
    ap.add_argument('--readout', choices=['fixed', 'gated'], default='fixed')
    ap.add_argument('--control', choices=['', 'no_edges', 'chain', 'shuffle', 'permute_types'], default='')
    ap.add_argument('--direction', choices=DIRECTIONS, default='both')
    ap.add_argument('--block-input', choices=BLOCK_INPUTS, default='tokens')
    ap.add_argument('--vocab', choices=VOCABS, default='ours')
    ap.add_argument('--imports', choices=IMPORTS, default='')
    ap.add_argument('--out', help='run directory name override (default: derived from the config)')
    ap.add_argument('--export', help="--model tokenseq: the block export, a directory or a name under the export directory")
    ap.add_argument('--device', choices=['cuda', 'cpu'], help='--model tokenseq|fivechannel: default cuda when available')
    ap.add_argument('--smoke', type=int, help='--model tokenseq|fivechannel: N train binaries, 2 epochs, temporary directory')
    ap.add_argument('--window-budget', type=int, help='--model tokenseq: windows per batch')
    ap.add_argument('--build', help='--model fivechannel: the pipeline-B build folder of <sample_id>.pt files (default: the '
                                    'team_b_build source of datasets.json, GT_TEAM_B_BUILD)')
    ap.add_argument('--threshold-from', choices=THRESHOLD_FROM, default='val',
                    help="--model tokenseq|fivechannel: choose the decision threshold on the validation families ('val') or on "
                         "out-of-fold predictions of grouped refits of the training binaries ('oof', run name suffix -thoof)")
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
    options = {k: v for k, v in ((k, args.pop(k)) for k in ALL_OPTIONS) if v is not None}
    try:
        cfg = RunConfig(**args)
    except ValueError as e:
        ap.error(str(e))
    foreign = [k for k in options if k not in MODEL_OPTIONS.get(cfg.model, ())]
    if foreign:
        owners = ' or '.join(sorted({m for k in foreign for m, o in MODEL_OPTIONS.items() if k in o}))
        ap.error(f"--{', --'.join(k.replace('_', '-') for k in foreign)} belong to --model {owners}")
    return cfg, out, options


def parse_config(argv=None):
    """(RunConfig, --out) of a command line; the flags that only `--model tokenseq` takes are in `parse_args`."""
    return parse_args(argv)[:2]


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def predict(spec, model, data, device):
    """(sample ids, probabilities, (n_segments_total, n_blocks_kept, n_blocks_total) per sample) over `data`."""
    model.eval()
    ids, probs, sizes = [], [], []
    for batch in data:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = spec.forward(model, to_device(batch, device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        ids += batch['sample_id']
        none = [None] * len(batch['sample_id'])
        sizes += zip(batch['n_segments_total'], batch.get('n_blocks_kept', none), batch.get('n_blocks_total', none))
    return ids, np.concatenate(probs), sizes


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


def choose_thresholds(cfg: RunConfig, manifest, val_by_rule, epochs, fit_predict=None):
    """({rule: threshold}, record) for the checkpoints of a run, rule being 'f1' and 'auc'.

    `val_by_rule[rule]` is the (sample ids, probabilities, sizes) of the validation binaries under that rule's checkpoint.
    cfg.threshold_from == 'val' (the default): the GRID threshold with the highest validation macro-F1 (`val_threshold_of`), and
    the record is None. 'oof': the out-of-fold rule of `s04_evaluation.val_threshold.oof_rule` over the training binaries,
    with `fit_predict(train_rows, held_rows, epochs, seed)` the model's refit (see there) and `epochs` = {rule: epoch count of
    that checkpoint}; the record (what metrics.json keeps) also holds the validation thresholds, for comparison."""
    val = {rule: val_threshold_of(manifest.val, *v[:2]) for rule, v in val_by_rule.items()}
    if cfg.threshold_from == 'val':
        return val, None
    if fit_predict is None:
        raise ValueError(f'model {cfg.model} gave no refit for threshold_from={cfg.threshold_from!r}')
    chosen, record = oof_rule(manifest.train, fit_predict, epochs, seed=cfg.seed)
    record['threshold_val'] = val
    return chosen, record


def check_requires(spec, cache, protocol=None):
    missing = [p for p in spec.requires if not (Path(cache) / p).exists()]
    if 'index.csv' in missing:
        raise MissingCache(missing_cache_message(cache, protocol))
    if missing:
        raise FileNotFoundError(f'model {spec.name} reads {", ".join(missing)} under {cache}, which is not there: point '
                                'GT_CACHE or datasets.json (--cache <name>) at the CFG cache')


def write_run_as(out_name, cfg, *args, **kw):
    """`write_run` into RUNS/<out_name>. write_run insists that the directory is named like the config; for a `--out` name
    the files are written under the config's name in a staging directory and moved."""
    if out_name is None or out_name == cfg.name:
        return write_run(RUNS / cfg.name, cfg, *args, **kw)
    stage = RUNS / f'.stage-{out_name}'
    shutil.rmtree(stage, ignore_errors=True)
    try:
        metrics = write_run(stage / cfg.name, cfg, *args, **kw)
        (RUNS / out_name).mkdir(parents=True, exist_ok=True)
        for path in (stage / cfg.name).iterdir():
            path.replace(RUNS / out_name / path.name)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return metrics


def run(cfg: RunConfig, out_name=None, **options):
    """One run. `options` are the flags of MODEL_OPTIONS[model], for a model that owns its loop (`spec.run`: `tokenseq`,
    `fivechannel`)."""
    spec = spec_for(cfg)
    spec.check(cfg)
    if hasattr(spec, 'run'):
        return spec.run(cfg, out_name, **options)
    if options:
        raise ValueError(f'model {spec.name} takes none of {", ".join(options)}')
    if cfg.cache and cfg.cache not in registry() and Path(cfg.cache) != CACHE:
        raise ValueError(f'config says cache {cfg.cache} but this process reads {CACHE}')
    check_requires(spec, CACHE, cfg.protocol)
    cfg = replace(cfg, cache=str(CACHE))
    out = RUNS / (out_name or cfg.name)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / 'config.json')
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device('cuda')

    manifest = make_manifest(cfg, load_index(protocol=cfg.protocol))
    manifest.save(out / 'manifest.json')
    info = RunInfo(manifest, out)
    model = spec.build(cfg, info).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_weight = loss_weight_fn(cfg, manifest.train, device)
    train_data = spec.loader(cfg, manifest.train, True)
    val_data = spec.loader(cfg, manifest.val, False)
    print(f'{cfg.name}: train {len(manifest.train)} val {len(manifest.val)} test {len(manifest.test)} '
          f'val families {manifest.val_families} vocab {info.summary.get("vocab")}', flush=True)

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
                logits = spec.forward(model, batch)
            loss = (F.binary_cross_entropy_with_logits(logits, batch['label'], reduction='none')
                    * loss_weight(batch)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(loss.item())
        train_secs = time.time() - start
        ids, probs, _ = predict(spec, model, val_data, device)
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

    test_data = spec.loader(cfg, manifest.test, False)
    chosen = {}
    for rule, ckpt in (('f1', 'best.pt'), ('auc', 'best_auc.pt')):
        model.load_state_dict(torch.load(out / ckpt, map_location=device))
        by_split = {'val': predict(spec, model, val_data, device), 'test': predict(spec, model, test_data, device)}
        threshold_val = val_threshold_of(manifest.val, *by_split['val'][:2],
                                         select_on=cfg.select_on if rule == 'auc' else 'overall')
        chosen[rule] = (by_split, threshold_val)

    extra = {'best_epoch': best_f1_epoch, 'epochs_run': len(history), 'val_macro_f1': best_f1,
             'auc_best_epoch': best_auc_epoch, 'val_auc': best_auc,
             'mean_train_epoch_secs': float(np.mean([h['train_secs'] for h in history])),
             'epoch_secs_mean': float(np.mean([h['epoch_secs'] for h in history])),
             'peak_gpu_mb': max(h['peak_gpu_mb'] for h in history)}
    metrics = write_run_as(out_name, cfg, chosen['f1'][0], chosen['auc'][0], chosen['f1'][1], chosen['auc'][1], extra,
                           manifest=manifest)
    print(f"{cfg.name}: {metrics['epoch_secs_mean']:.1f} s/epoch over {len(history)} epochs "
          f"(train {metrics['mean_train_epoch_secs']:.1f}), peak {metrics['peak_gpu_mb']} MB", flush=True)
    print(json.dumps({k: metrics[k] for k in ('run', 'best_epoch', 'val_macro_f1', 'test')}, indent=2), flush=True)
    return metrics


def main():
    cfg, out, options = parse_args()
    run(cfg, out, **options)


if __name__ == '__main__':
    run_command(main)
