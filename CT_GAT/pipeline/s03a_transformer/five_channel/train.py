"""Train Janice Park's five-channel Model 1 on one fold of a dataset of the team pipelines (default: our x86 kfold protocol).

python -m experiments.train --model fivechannel --protocol kfold --arch x86 --fold 0 --seed 0 [--max-instructions 512]
[--arch-balance] [--build DIR] [--device cuda|cpu] [--smoke N] [--epochs 40 --patience 8]
python -m s03a_transformer.five_channel.train ...          the same run, with the options of this model only

Run from CT_GAT/pipeline. `common/registry.py` serves this loop as the model `fivechannel` (`five_channel/spec.py`), so
`experiments.train` hands the run over to `run` below. The run directory is <runs>/kfold<f>_x86_fivechannel_s<seed>[-L<n>]
(GT_RUNS; RunConfig.name, model 'fivechannel'); it holds config.json, manifest.json, history.csv, best.pt (best validation
macro-F1 at 0.5), best_auc.pt (best validation AUC), predictions.csv, predictions_auc.csv and metrics.json, the files of every
other model, so s04_evaluation/report.py, stats.py and gates.py read it.

The dataset is a flag, as for the teammates' arms (`experiments/team_pipelines/train_b.py`): `--protocol kfold --arch x86` (the
default; 1,810 binaries), `--protocol kfold --arch all` (the 2,411 Mendeley binaries, x86 and x64) and `--protocol kfold_v3 --arch all|matched`
(the combined dataset, 5,652 binaries with a v3 fold; `matched`, 4,850, trains on the `all` build). The flag is the literal
`--protocol`, so that the cache is resolved from it. Run names: `kfold<f>_all_fivechannel[-L<n>][-ab]_s<seed>`,
`v3kfold<f>_matched_fivechannel..._s<seed>`. A run refuses a build made for another dataset, in both directions (the build's
settings.json names `protocol` and `arch`; a kfold x86 build records neither): `all` serves an `all` or `matched` run, `x86` an `x86`
one (`train_b.check_build_dataset`). `--arch-balance` (needs --arch all or matched; run name suffix -ab) weights each training
binary's loss by N / (C n_cell) over the non-empty (class, architecture) cells of the training manifest instead of N / (2 n_class)
of its class, and averages them as a weighted mean, sum(l_i w_i) / sum(w_i); metrics.json records `arch_balance` and the cell weights.
Vocabulary clash: `sig500v3` is the s02-fix *build* version, `kfold_v3` the fold *protocol*, `_v3.zip` the preprocessing version of
the zips.

Data: the folder of `<sample_id>.pt` files that `experiments/team_pipelines/build_b.py` writes (Yanping's stage-2 output; `--build`,
else the `team_b_build` source: GT_TEAM_B_BUILD, datasets.json, else `CT_GAT/processed/cfg_subgraphs`, where unzipping the
preprocessed zip at `CT_GAT/` or `build_b --source repo` puts it), read through `Model1Dataset`. The folds are ours
(`common.manifests.make_manifest` on the CFG cache index: the 5-fold x86 family holdout, the validation carve of the run seed,
the same train / val / test ids as the GNN and the teammates' arms of that seed); a binary the build left out on purpose
(`settings.json` `excluded`) is out of all three splits and recorded in metrics.json, as for the teammates' arms.

The API vocabulary of the build covers all the binaries of its dataset. Per fold it is made a training-set one again, with the logic of
`experiments/team_pipelines/train_b.py`: every API id (>= FIRST_API_ID) that no training binary of the fold uses becomes UNK_API
(`remap_api`, applied to the instructions the model reads).

Model: her `Model1TokenSequence` (d_model 64, 4 heads, 2 layers, `max_seq_len` = `--max-instructions`, 128 as she wrote it) on the
first `max_instructions` instructions of a binary, batched by her `model1_collate_fn`; a binary without an instruction is one
all-PAD tuple (an all-masked row would give NaN). Loss: cross entropy with the class weight N / (2 n_class) of the training
labels (cell weights under --arch-balance).
--per-block (run name suffix -pb) reads the first `max_instructions` instructions of every basic block (64, 128 or 512) instead of
the first of the binary: the same network encodes each block on its own (her embedder, positions and encoder, her masked mean
within the block, `block_transformer.encode_blocks`), the binary is the mean of its non-empty block vectors, and her `classifier`
gives the logits; no new parameter and no edge (the input of the end-to-end Model 3 without its GAT). Blocks are encoded in
groups of similar length of at most TOKENS_PER_CHUNK padded positions, which bounds the memory and does not change the result.
AdamW (RunConfig lr, weight_decay), gradient clip RunConfig.grad_clip, batches of BATCH_SIZE binaries, bf16 autocast on CUDA.
Each epoch records the validation AUC and the macro-F1 at 0.5; the F1 checkpoint is the epoch of the best macro-F1, the AUC
checkpoint that of the best AUC, and training stops when the AUC has not improved for `patience` epochs. The AUC checkpoint's
threshold is the GRID maximiser of the validation macro-F1 (`val_threshold_of`), or with --threshold-from oof the maximiser of the
macro-F1 of out-of-fold predictions on the training binaries (`refit_predict`, `experiments.train.choose_thresholds`, metrics.json
`threshold_rule`; the run name gains -thoof). The test split is read once after training.

--smoke N trains on N binaries (half of each class) for 2 epochs, evaluates on N validation and N test binaries, and writes into a
temporary directory (never under the run directory).
"""
import argparse
import csv
import json
import random
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from common.config import RUNS, THRESHOLD_FROM, RunConfig
from common.manifests import Manifest, load_index, make_manifest, run_command
from common.paths import MissingSource, source_for
from common.run_io import evaluate, write_run
from experiments.team_pipelines.build_b import resolve_dir
from experiments.team_pipelines.common import DATASET_OF, DEFAULT_ARCH
from experiments.team_pipelines.fit import cell_name
from experiments.team_pipelines.train_b import check_build_dataset, read_excluded, remap_api
from experiments.train import choose_thresholds
from s03c_cfg_transformer.block_transformer import encode_blocks, rank_in_block
from s04_evaluation.metrics import cell_weights

from .model import Model1Dataset, Model1TokenSequence, model1_collate_fn, tok

BATCH_SIZE = 32          # binaries per step (RunConfig.batch_size is the graph models' batch of graphs)
SMOKE_EPOCHS = 2
D_MODEL = 64             # her default
TOKENS_PER_CHUNK = 16384  # --per-block: padded instruction positions encoded at once
AUTOCAST_ON_CPU = False   # `experiments.predict` sets it to rescore a run that was trained under bf16 autocast on cuda with bf16 on cpu


class Binary(NamedTuple):
    """What a run keeps of one `.pt`: the instructions the model reads (the first max_instructions of the binary, or under
    --per-block those of every block, blocks in order), the API ids (>= FIRST_API_ID) used by the whole binary, its number of
    instructions and its label; under --per-block also the block of each kept instruction and the binary's number of blocks."""
    head: torch.Tensor
    api: torch.Tensor
    n_instructions: int
    label: int
    block: torch.Tensor | None = None
    n_blocks: int = 0


def say(*args):
    print(*args, flush=True)


FILL_BUILD = ('fill CT_GAT/processed/cfg_subgraphs: unzip Mendeley_x86_cfg_subgraphs_signal_500blocks_v3.zip at CT_GAT/, or build it '
              '(about 4 h on CPU) with python -m experiments.team_pipelines.build_b --source repo')
FILL_BUILD_OF = {
    ('kfold', 'x86'): FILL_BUILD,
    ('kfold', 'all'): ('fill CT_GAT/processed/cfg_subgraphs: unzip Mendeley_all_cfg_subgraphs_signal_500blocks_v3.zip at CT_GAT/, or '
                       'extend a copy of the x86 build (about 0.5 h on CPU) with python -m experiments.team_pipelines.build_b --source '
                       'repo --protocol kfold --arch all --extend --out <the copy>'),
    ('kfold_v3', 'all'): ('fill CT_GAT/processed/cfg_subgraphs: unzip combined_v3_cfg_subgraphs_signal_500blocks_v3.zip at CT_GAT/, or '
                          'build it (about 15 h on CPU) with python -m experiments.team_pipelines.build_b --source repo --protocol '
                          'kfold_v3 --arch all --max-instructions 4000000 --seeded-rows'),
}


def fill_build(protocol='kfold', arch='x86'):
    """How to fill the default build folder of a dataset (`matched` reads the all build)."""
    return FILL_BUILD_OF[protocol, 'all' if arch == 'matched' else arch]


def resolve_build(build=None, protocol='kfold', arch='x86') -> Path:
    """`build` (a leading RUNS/ is the run directory), else the `team_b_build` source: GT_TEAM_B_BUILD, datasets.json, else
    `CT_GAT/processed/cfg_subgraphs`. A folder that is missing or holds no `<sample_id>.pt` is an error; for the default folder
    it says how to fill it for the dataset (`protocol`, `arch`; `fill_build`)."""
    if build is None:
        try:
            path = source_for('team_b_build')
        except MissingSource as e:
            raise FileNotFoundError(f'--build was not given and {e}; or {fill_build(protocol, arch)}') from e
    else:
        path = resolve_dir(Path(build))
    hint = f': {fill_build(protocol, arch)}' if build is None else ''
    if not path.is_dir():
        raise FileNotFoundError(f'the pipeline-B build {path} is not a directory{hint}')
    if not any(p.name != 'channel_embedder.pt' for p in path.glob('*.pt')):
        raise FileNotFoundError(f'the pipeline-B build {path} holds no <sample_id>.pt file{hint}')
    return path


def build_done(build: Path):
    """The ids the build's checkpoint.json covers, or None when it has none (a folder of `.pt` files that is not a build)."""
    path = build / 'checkpoint.json'
    return set(json.loads(path.read_text(encoding='utf-8'))['done']) if path.is_file() else None


def per_block_head(ids, insn_to_block, n_blocks, block_len):
    """(instructions, block) of the first `block_len` instructions of every block, blocks in order, the order within a block kept."""
    order, block, position, _ = rank_in_block(insn_to_block, n_blocks)
    keep = position < block_len
    return ids[order][keep].clone(), block[keep].clone()


def load_binaries(build: Path, labels: dict, max_instructions: int, per_block: bool = False) -> dict:
    """{sample_id: Binary} of the ids in `labels` ({sample_id: label}), read with her dataset class
    (`Model1Dataset`) from the `<sample_id>.pt` files of `build`; under `per_block` the `.pt` is read whole for its
    `insn_to_block` and `block_features` (the blocks). A missing file, a tensor that is not [N, 5] or a label other
    than the fold's is an error. A label of `None` is the one the file holds (`experiments.predict` scores binaries whose
    label the fold does not name)."""
    dataset = Model1Dataset(build)
    where = {Path(f).stem: i for i, f in enumerate(dataset.files)}
    missing = sorted(set(labels) - set(where))
    if missing:
        raise FileNotFoundError(f'{len(missing)} of the {len(labels)} binaries of this fold have no <sample_id>.pt in {build}, '
                                f'e.g. {missing[:3]}')
    done = build_done(build)
    if done is not None and set(labels) - done:
        raise FileNotFoundError(f'{len(set(labels) - done)} binaries are not in the checkpoint.json of {build} (an unfinished '
                                f'build), e.g. {sorted(set(labels) - done)[:3]}')
    out = {}
    for sid, label in labels.items():
        item = dataset[where[sid]]
        ids = item['instruction_ids']
        if ids.dim() != 2 or ids.shape[1] != 5:
            raise ValueError(f'{sid}.pt: instruction_ids {tuple(ids.shape)}, expected [N, 5]')
        if label is None:
            label = int(item['label'])
        elif int(item['label']) != label:
            raise ValueError(f'{sid}.pt holds label {int(item["label"])}, the fold says {label}')
        api = torch.unique(ids[:, 3][ids[:, 3] >= tok.FIRST_API_ID])
        if per_block:
            sample = torch.load(dataset.files[where[sid]], weights_only=False)
            n_blocks = int(sample['block_features'].shape[0])
            head, block = per_block_head(ids, sample['insn_to_block'], n_blocks, max_instructions)
            out[sid] = Binary(head, api, int(len(ids)), label, block, n_blocks)
        else:
            head = ids[:max_instructions].clone() if len(ids) else torch.zeros((1, 5), dtype=torch.long)
            out[sid] = Binary(head, api, int(len(ids)), label)
    return out


def seen_api_ids(binaries: dict, train_ids) -> torch.Tensor:
    """Sorted tensor of the API ids (>= FIRST_API_ID) that occur in the training binaries (`train_b.train_api_ids`, from the
    per-binary sets that `load_binaries` keeps)."""
    parts = [binaries[i].api for i in train_ids]
    return torch.unique(torch.cat(parts)) if parts else torch.zeros(0, dtype=torch.long)


def make_items(binaries: dict, ids, seen: torch.Tensor) -> dict:
    """{sample_id: item for `model1_collate_fn` (or `collate_blocks` under --per-block)} with the API ids outside `seen` mapped
    to UNK_API."""
    items = {}
    for i in ids:
        b = binaries[i]
        items[i] = {'instruction_ids': remap_api(tok, b.head, seen), 'label': torch.tensor(b.label)}
        if b.block is not None:
            items[i].update(block=b.block, n_blocks=b.n_blocks)
    return items


def api_vocab_size(build: Path, binaries: dict) -> int:
    """Rows of the API embedding: the build's vocabulary (FIRST_API_ID + names), at least one more than the largest id read."""
    size = tok.FIRST_API_ID
    path = build / 'api_vocab.json'
    if path.is_file():
        size += len(tok.load_api_vocab(path).names)
    biggest = max((int(b.api.max()) for b in binaries.values() if len(b.api)), default=0)
    return max(size, biggest + 1)


def collate_blocks(items, ids):
    """One flat list of the kept instructions of the binaries `ids`: `insn_block` numbers the blocks of the batch, `block_owner`
    gives the binary (0..len(ids)-1) of every block."""
    heads, blocks, owners, offset = [], [], [], 0
    for g, i in enumerate(ids):
        item = items[i]
        heads.append(item['instruction_ids'])
        blocks.append(item['block'] + offset)
        owners.append(torch.full((item['n_blocks'],), g, dtype=torch.long))
        offset += item['n_blocks']
    return {'instruction_ids': torch.cat(heads), 'insn_block': torch.cat(blocks), 'block_owner': torch.cat(owners),
            'n_blocks': offset, 'label': torch.stack([items[i]['label'] for i in ids])}


def collate(items, ids, cfg):
    batch = collate_blocks(items, ids) if cfg.per_block else model1_collate_fn([items[i] for i in ids], max_instructions=cfg.max_instructions)
    batch['sample_ids'] = list(ids)
    return batch


def autocast(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda' or AUTOCAST_ON_CPU)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def per_block_logits(net, batch):
    """--per-block: her network on each block of the batch (`encode_blocks`, groups of blocks of similar length, each padded to
    its own longest block), the mean of the non-empty block vectors of each binary, and her `classifier`. A binary with no
    instruction gets the zero vector."""
    ids, n_blocks, owner = batch['instruction_ids'], batch['n_blocks'], batch['block_owner']
    n_binaries = len(batch['label'])
    order, block, position, lengths = rank_in_block(batch['insn_block'], n_blocks)
    ids = ids[order]
    filled = torch.nonzero(lengths > 0).flatten()
    filled = filled[torch.argsort(lengths[filled], stable=True)]
    sizes = lengths[filled].tolist()
    local = torch.full((n_blocks,), -1, dtype=torch.long, device=ids.device)
    picked, vectors, start = [], [], 0
    while start < len(filled):
        end = start + 1
        while end < len(filled) and (end + 1 - start) * sizes[end] <= TOKENS_PER_CHUNK:
            end += 1
        chunk, width = filled[start:end], sizes[end - 1]
        local.fill_(-1)
        local[chunk] = torch.arange(len(chunk), device=ids.device)
        sel = local[block] >= 0
        x = torch.zeros(len(chunk), width, ids.shape[1], dtype=ids.dtype, device=ids.device)
        x[local[block[sel]], position[sel]] = ids[sel]
        pad_mask = torch.arange(width, device=ids.device)[None, :] >= lengths[chunk][:, None]
        vectors.append(encode_blocks(net, x, pad_mask).float())
        picked.append(chunk)
        start = end
    out = torch.zeros(n_blocks, net.d_model, device=ids.device)
    if vectors:
        out = out.index_copy(0, torch.cat(picked), torch.cat(vectors))
    sums = torch.zeros(n_binaries, net.d_model, device=ids.device).index_add(0, owner, out)
    counts = torch.zeros(n_binaries, device=ids.device).index_add(0, owner, (lengths > 0).float())
    return net.classifier(sums / counts.clamp(min=1.0)[:, None])


def forward(net, batch):
    if 'insn_block' in batch:
        return per_block_logits(net, batch)
    return net(batch['instruction_ids'], padding_mask=batch['padding_mask'])


def train_epoch(net, opt, items, ids, class_weight, cfg, epoch, device, seed):
    """One pass over the binaries `ids` in a random order (seed, epoch); returns the mean batch loss. `class_weight` is the tensor
    of `class_weights` (the loss of a batch is the mean of the weighted losses) or, under --arch-balance, the {sample_id: weight} of
    `loss_weights` (the loss is the weighted mean sum(l_i w_i) / sum(w_i))."""
    net.train()
    order = np.random.default_rng((seed, epoch, 7919)).permutation(len(ids))
    losses = []
    for start in range(0, len(ids), BATCH_SIZE):
        batch = to_device(collate(items, [ids[i] for i in order[start:start + BATCH_SIZE]], cfg), device)
        with autocast(device):
            logits = forward(net, batch)
        per_binary = F.cross_entropy(logits.float(), batch['label'], reduction='none')
        if isinstance(class_weight, dict):
            weight = torch.tensor([class_weight[i] for i in batch['sample_ids']], dtype=torch.float32, device=device)
            loss = (per_binary * weight).sum() / weight.sum()
        else:
            loss = (per_binary * class_weight[batch['label']]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def predict(net, items, ids, binaries, cfg, device):
    """(sample ids, P(ransomware) [n] float64, sizes) of `ids` in order; sizes is (instructions, None, None) per binary."""
    net.eval()
    probs = []
    for start in range(0, len(ids), BATCH_SIZE):
        batch = to_device(collate(items, ids[start:start + BATCH_SIZE], cfg), device)
        with autocast(device):
            logits = forward(net, batch)
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    sizes = [(binaries[i].n_instructions, None, None) for i in ids]
    return list(ids), (np.concatenate(probs).astype(np.float64) if probs else np.zeros(0)), sizes


def class_weights(labels, device):
    labels = np.asarray(labels)
    return torch.tensor([len(labels) / (2 * max(1, (labels == c).sum())) for c in (0, 1)], device=device)


def loss_weights(cfg, rows, device):
    """The weights of the loss from the rows of the training manifest only: the class weights of `class_weights` (a tensor), or
    under cfg.arch_balance the weight N / (C n_cell) of every binary's (class, architecture) cell, {sample_id: weight}
    (`s04_evaluation.metrics.cell_weights`, as `experiments.train.loss_weight_fn`)."""
    if cfg.arch_balance:
        return dict(zip((r['sample_id'] for r in rows), cell_weights([r['label'] for r in rows], [r['arch'] for r in rows]).tolist()))
    return class_weights([r['label'] for r in rows], device)


def new_net(cfg, n_api, device):
    return Model1TokenSequence(api_vocab_size=n_api, d_model=D_MODEL, max_seq_len=cfg.max_instructions, num_classes=2).to(device)


def refit_predict(binaries, cfg, n_api, device):
    """The `fit_predict` of `val_threshold.oof_rule` for this model: a new model on the `train_rows` (binaries of the fold's
    training split), trained like the run for max(epochs) epochs with no early stopping, and the probabilities of `held_rows`
    after each epoch in `epochs`. As in a run, the API ids that no binary of `train_rows` uses are UNK_API for the held-out
    binaries too."""
    def fit_predict(train_rows, held_rows, epochs, seed):
        train_ids, held_ids = [r['sample_id'] for r in train_rows], [r['sample_id'] for r in held_rows]
        items = make_items(binaries, train_ids + held_ids, seen_api_ids(binaries, train_ids))
        torch.manual_seed(seed)
        net = new_net(cfg, n_api, device)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        weight = loss_weights(cfg, train_rows, device)
        out = {}
        for epoch in range(1, max(epochs) + 1):
            train_epoch(net, opt, items, train_ids, weight, cfg, epoch, device, seed)
            if epoch in epochs:
                out[epoch] = predict(net, items, held_ids, binaries, cfg, device)[1]
        return out
    return fit_predict


def half_of_each_class(rows, n):
    """The first n rows, half of each class (the other class fills in when one runs out)."""
    by_label = {c: [r for r in rows if r['label'] == c] for c in (0, 1)}
    take = {1: min(len(by_label[1]), (n + 1) // 2)}
    take[0] = min(len(by_label[0]), n - take[1])
    take[1] = min(len(by_label[1]), n - take[0])
    return by_label[1][:take[1]] + by_label[0][:take[0]]


def run(cfg: RunConfig, build=None, device=None, smoke=None, out_root=None, log=say):
    """Train and write one run directory (under RUNS, `out_root`, or a temporary directory for a smoke run); returns its
    metrics."""
    cfg.check_fivechannel()
    if smoke is not None and out_root is None:
        with tempfile.TemporaryDirectory(prefix='fivechannel_smoke_') as tmp:
            return run(cfg, build, device, smoke, Path(tmp), log)
    out_root = Path(out_root) if out_root is not None else RUNS
    if smoke is not None and RUNS.resolve() in (out_root / 'x').resolve().parents:
        raise ValueError(f'a smoke run never writes under {RUNS}: {out_root}')
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if smoke is not None:
        cfg = replace(cfg, epochs=SMOKE_EPOCHS)
    build = resolve_build(build, cfg.protocol, cfg.arch)
    check_build_dataset(build, cfg.protocol, cfg.arch)
    out = out_root / cfg.name
    out.mkdir(parents=True, exist_ok=True)

    manifest = make_manifest(cfg, load_index(protocol=cfg.protocol))
    excluded = read_excluded(build)
    keep = lambda rows: [r for r in rows if r['sample_id'] not in excluded]
    run_manifest = Manifest(keep(manifest.train), keep(manifest.val), keep(manifest.test), manifest.val_families)
    if smoke is not None:
        run_manifest = Manifest(*[half_of_each_class(rows, smoke) for rows in (run_manifest.train, run_manifest.val, run_manifest.test)],
                                run_manifest.val_families)
    splits = {'train': run_manifest.train, 'val': run_manifest.val, 'test': run_manifest.test}
    ids = {s: [r['sample_id'] for r in rows] for s, rows in splits.items()}
    labels = {r['sample_id']: r['label'] for rows in splits.values() for r in rows}
    binaries = load_binaries(build, labels, cfg.max_instructions, cfg.per_block)
    seen = seen_api_ids(binaries, ids['train'])
    every = set()
    for b in binaries.values():
        every.update(b.api.tolist())
    log(f'api ids in the fold\'s training binaries: {len(seen)}; unseen in val/test binaries mapped to UNK_API: '
        f'{len(every - set(seen.tolist()))} of {len(every)} ids used')
    items = make_items(binaries, list(labels), seen)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    n_api = api_vocab_size(build, binaries)
    net = new_net(cfg, n_api, device)
    n_params = sum(p.numel() for p in net.parameters())
    weight = loss_weights(cfg, splits['train'], device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    cfg.save(out / 'config.json')
    log(f'{cfg.name}: train {len(ids["train"])} val {len(ids["val"])} test {len(ids["test"])} binaries, {len(excluded)} excluded, '
        f'{n_params} parameters, max_instructions {cfg.max_instructions}' + (' per block' if cfg.per_block else '') + f', device {device}' + (f', smoke {smoke}' if smoke is not None else ''))

    history, best_f1, best_f1_epoch, best_auc, best_auc_epoch = [], -1.0, 0, -1.0, 0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, cfg.epochs + 1):
        start = time.perf_counter()
        loss = train_epoch(net, opt, items, ids['train'], weight, cfg, epoch, device, cfg.seed)
        train_secs = time.perf_counter() - start
        val_ids, probs, _ = predict(net, items, ids['val'], binaries, cfg, device)
        scores = evaluate(run_manifest.val, val_ids, probs, cfg.threshold)
        row = {'epoch': epoch, 'loss': loss, 'train_secs': round(train_secs, 3),
               'val_secs': round(time.perf_counter() - start - train_secs, 3), 'epoch_secs': round(time.perf_counter() - start, 3),
               'peak_gpu_mb': round(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == 'cuda' else 0,
               **{f'val_{k}': scores[k] for k in ('macro_f1', 'accuracy', 'auc', 'recall')}}
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

    by_rule = {}
    for rule, ckpt in (('f1', 'best.pt'), ('auc', 'best_auc.pt')):
        net.load_state_dict(torch.load(out / ckpt, map_location=device))
        by_rule[rule] = {s: predict(net, items, ids[s], binaries, cfg, device) for s in ('val', 'test')}
    thresholds, threshold_rule = choose_thresholds(
        cfg, run_manifest, {rule: by_rule[rule]['val'] for rule in by_rule}, {'f1': best_f1_epoch, 'auc': best_auc_epoch},
        refit_predict(binaries, cfg, n_api, device) if cfg.threshold_from == 'oof' else None)
    extra = {
        'best_epoch': best_f1_epoch, 'epochs_run': len(history), 'val_macro_f1': best_f1, 'auc_best_epoch': best_auc_epoch,
        'val_auc': best_auc, 'mean_train_epoch_secs': float(np.mean([h['train_secs'] for h in history])),
        'epoch_secs_mean': float(np.mean([h['epoch_secs'] for h in history])),
        'peak_gpu_mb': max(h['peak_gpu_mb'] for h in history), 'build_dir': str(build), 'n_parameters': n_params,
        'max_instructions': cfg.max_instructions, 'per_block': cfg.per_block, 'n_api_ids_train': int(len(seen)),
        'n_api_ids_unseen_mapped': len(every - set(seen.tolist())), 'device': device.type, 'smoke': smoke,
        'truncated_instructions': {s: sum(binaries[i].n_instructions > len(binaries[i].head) for i in ids[s]) for s in splits},
        'instructions_read': {s: sum(len(binaries[i].head) for i in ids[s]) / max(1, sum(binaries[i].n_instructions for i in ids[s]))
                              for s in splits},
        'model': {'d_model': D_MODEL, 'max_seq_len': cfg.max_instructions, 'num_layers': 2, 'batch_size': BATCH_SIZE,
                  'per_block': cfg.per_block}}
    if cfg.arch_balance:
        extra['arch_balance'] = True
        extra['cell_weights'] = {cell_name(r['label'], r['arch']): weight[r['sample_id']] for r in splits['train']}
    if threshold_rule is not None:
        extra['threshold_rule'] = threshold_rule
    reasons = '; '.join(sorted(set(excluded.values())))
    # a full run is written against the fold's own manifest with the excluded ids named (as the teammates' arms are); a smoke run
    # against its subset, which holds none of them
    metrics = write_run(out, cfg, by_rule['f1'], by_rule['auc'], thresholds['f1'], thresholds['auc'], extra,
                        excluded=sorted(excluded) if smoke is None else [], excluded_reason=reasons,
                        manifest=manifest if smoke is None else run_manifest)
    log(f"{cfg.name}: {metrics['epoch_secs_mean']:.1f} s/epoch over {len(history)} epochs (train "
        f"{metrics['mean_train_epoch_secs']:.1f}), peak GPU {metrics['peak_gpu_mb']} MB")
    log(json.dumps({k: metrics[k] for k in ('run', 'best_epoch', 'val_macro_f1', 'test')}, indent=2))
    return metrics


def parse_config(argv=None):
    """(RunConfig, options) from a command line; options: build, device, smoke."""
    ap = argparse.ArgumentParser(description="Train Janice Park's five-channel Model 1 on one fold of a pipeline-B build")
    ap.add_argument('--protocol', choices=sorted(DATASET_OF), default='kfold', help='the fold protocol, which picks the dataset: kfold '
                    '(Mendeley, mendeley_x86 cache) or kfold_v3 (combined_v3 cache); the literal flag also picks the cache')
    ap.add_argument('--arch', choices=['x86', 'all', 'matched'], default=None, help='the binaries to train on (default: x86 under '
                    'kfold, all under kfold_v3); matched (kfold_v3 only) trains on the all build')
    ap.add_argument('--arch-balance', action='store_true', help='weight the training binaries by (class, architecture) cell instead '
                    'of by class (needs --arch all or matched; run name suffix -ab)')
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-instructions', type=int, default=128,
                    help='instructions read per binary, or per basic block with --per-block (64, 128 or 512)')
    ap.add_argument('--per-block', action='store_true', help='read the first --max-instructions instructions of every block, encode '
                    'each block, average the block vectors (run name suffix -pb)')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--threshold-from', choices=THRESHOLD_FROM, default='val',
                    help='validation families (val) or out-of-fold refits of the training binaries (oof, name suffix -thoof)')
    ap.add_argument('--build', default=None, help='build folder of <sample_id>.pt files (default: the team_b_build source, CT_GAT/processed/cfg_subgraphs)')
    ap.add_argument('--device', choices=['cuda', 'cpu'], default=None, help='default: cuda when available')
    ap.add_argument('--smoke', type=int, default=None, help='N train binaries, 2 epochs, temporary directory, no RUNS')
    args = ap.parse_args(argv)
    try:
        cfg = RunConfig(protocol=args.protocol, fold=args.fold, arch=args.arch or DEFAULT_ARCH[args.protocol], seed=args.seed,
                        model='fivechannel', max_instructions=args.max_instructions, per_block=args.per_block,
                        epochs=args.epochs, patience=args.patience,
                        threshold_from=args.threshold_from, arch_balance=args.arch_balance)
    except ValueError as e:
        ap.error(str(e))
    if args.smoke is not None and args.smoke < 2:
        ap.error('--smoke needs at least 2 binaries')
    return cfg, {'build': args.build, 'device': args.device, 'smoke': args.smoke}


def main(argv=None):
    cfg, opts = parse_config(argv)
    run(cfg, **opts)


if __name__ == '__main__':
    run_command(main)
