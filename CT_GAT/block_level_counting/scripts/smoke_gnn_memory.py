"""Peak GPU memory of one bf16 forward + backward of the default gnn config.

python smoke_gnn_memory.py [--protocol kfold|kfold_v3] [--arch x86|all|matched] [--mask-blocks 0.15]

--protocol kfold_v3 reads the combined cache (GT_CACHE overrides it) and defaults to arch all; masking is applied as in
training. Two batches: the binary with the most blocks after the per-binary cap (MAX_BINARY_BLOCKS) alone, and a batch of
the largest capped binaries that together fit cfg.batch_blocks (at most cfg.batch_size of them). Prints
torch.cuda.max_memory_allocated for each and exits 1 when either exceeds LIMIT_GB.
"""
import argparse
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from common.config import CACHE, RunConfig
from common.manifests import load_index, load_sample, make_manifest
from pipeline.batching import collate, encode_binary
from pipeline.s04_select_subgraph.subgraph import MAX_BINARY_BLOCKS
from pipeline.s03_tokenizer.vocab import TokenVocab
from training.train import build_model, to_device

LIMIT_GB = 13


def make_batch(rows, vocab, cfg):
    items = []
    for r in rows:
        item = encode_binary(load_sample(r['sample_id']), vocab, cfg, np.random.default_rng(0), r['sample_id'])
        item['label'], item['sample_id'] = r['label'], r['sample_id']
        items.append(item)
    return collate(items)


def capped_blocks(r, cfg):
    """Blocks of a binary after the per-binary cap, the size the batch sampler uses."""
    return min(int(r['num_blocks']), cfg.budget or MAX_BINARY_BLOCKS)


def packed_rows(index, cfg):
    """Up to batch_size binaries filling batch_blocks: the largest, then the next-largest ones that still fit."""
    batch, total = [], 0
    for r in sorted(index, key=lambda r: -capped_blocks(r, cfg)):
        n = capped_blocks(r, cfg)
        if len(batch) < cfg.batch_size and total + n <= cfg.batch_blocks:
            batch.append(r)
            total += n
    return batch


def peak_gb(model, batch, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    model.zero_grad(set_to_none=True)
    batch = to_device(batch, device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(batch)
    F.binary_cross_entropy_with_logits(logits, batch['label']).backward()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 2**30, time.time() - start


def parse_config(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', choices=['kfold', 'kfold_v3'], default='kfold')
    ap.add_argument('--arch', choices=['x86', 'all', 'matched'])
    ap.add_argument('--mask-blocks', type=float, default=0.0)
    args = ap.parse_args(argv)
    arch = args.arch or ('all' if args.protocol == 'kfold_v3' else 'x86')
    return RunConfig(model='gnn', protocol=args.protocol, fold=0, arch=arch, mask_blocks=args.mask_blocks)


def main():
    cfg = parse_config()
    device = torch.device('cuda')
    index = load_index()
    vocab = TokenVocab(make_manifest(cfg, index).train, cfg)
    model = build_model(cfg, vocab).to(device)
    biggest = max(index, key=lambda r: capped_blocks(r, cfg))
    cases = {'largest binary alone': [biggest], f'packed to batch_blocks={cfg.batch_blocks}': packed_rows(index, cfg)}
    print(f'{cfg.name}: cache {CACHE}, mask_blocks {cfg.mask_blocks}', flush=True)
    too_big = False
    for name, rows in cases.items():
        batch = make_batch(rows, vocab, cfg)
        gb, secs = peak_gb(model, batch, device)
        print(f"{name}: {len(rows)} binaries, {sum(batch['n_blocks_kept'])} of "
              f"{sum(batch['n_blocks_total'])} blocks kept, "
              f"{len(batch['seg_len'])} functions, {len(batch['edge_src'])} edges, peak {gb:.2f} GB, {secs:.1f} s",
              flush=True)
        too_big |= gb > LIMIT_GB
    if too_big:
        print(f'peak above {LIMIT_GB} GB: stop and report', flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
