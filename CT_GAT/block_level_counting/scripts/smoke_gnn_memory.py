"""Peak GPU memory of one bf16 forward + backward of the default gnn config.

python smoke_gnn_memory.py

Two batches: the binary with the most blocks alone, and a batch of the largest binaries that together fit
cfg.batch_blocks (at most cfg.batch_size of them). Prints torch.cuda.max_memory_allocated
for each and exits 1 when either exceeds LIMIT_GB; nothing is capped.
"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from common.config import RunConfig
from common.manifests import load_index, load_sample, make_manifest
from pipeline.batching import collate, encode_binary
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


def packed_rows(index, cfg):
    """Up to batch_size binaries filling batch_blocks: the largest, then the next-largest ones that still fit."""
    batch, total = [], 0
    for r in sorted(index, key=lambda r: -int(r['num_blocks'])):
        n = int(r['num_blocks'])
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


def main():
    cfg = RunConfig(model='gnn', protocol='kfold', fold=0)
    device = torch.device('cuda')
    index = load_index()
    vocab = TokenVocab(make_manifest(cfg, index).train, cfg)
    model = build_model(cfg, vocab).to(device)
    biggest = max(index, key=lambda r: int(r['num_blocks']))
    cases = {'largest binary alone': [biggest], f'packed to batch_blocks={cfg.batch_blocks}': packed_rows(index, cfg)}
    too_big = False
    for name, rows in cases.items():
        batch = make_batch(rows, vocab, cfg)
        gb, secs = peak_gb(model, batch, device)
        print(f"{name}: {len(rows)} binaries, {sum(int(r['num_blocks']) for r in rows)} blocks, "
              f"{len(batch['seg_len'])} functions, {len(batch['edge_src'])} edges, peak {gb:.2f} GB, {secs:.1f} s",
              flush=True)
        too_big |= gb > LIMIT_GB
    if too_big:
        print(f'peak above {LIMIT_GB} GB: stop and report; nothing was capped', flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
