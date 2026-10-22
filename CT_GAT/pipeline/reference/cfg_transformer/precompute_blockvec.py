"""Frozen per-fold block vectors: python -m reference.cfg_transformer.precompute_blockvec --encoder-name ref
--export RUNS/export/ours_cap0 --folds 0 1 2 3 4

For each fold, loads RUNS/encoders/<encoder-name>/fold{k}/best.pt (after check_checkpoint, which refuses a checkpoint
that breaks the contract), embeds every DISTINCT block_key of the fold once (length-bucketed batches, no grad, bf16
autocast) and writes <cache>/side/blockvec/<export>_<name>/fold{k}/{keys.npy (sorted int64), vectors.npy (float16
[n, d]), meta.json}. Train a GNN on them with `train.py --block-input vec --block-vec <export>_<name>`.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common.config import EXPORT, RUNS
from reference.gnn.s05_block_counting.block_vecs import BlockVecs, vec_dir  # noqa: F401
from reference.cfg_transformer.block_export import sha_of_file
from reference.cfg_transformer.encoder_contract import check_checkpoint, load_encoder
from common.reader import BlockStore, FoldView

MAX_BATCH_TOKENS, MAX_BATCH_BLOCKS = 1 << 18, 8192


def length_buckets(lengths, max_tokens=MAX_BATCH_TOKENS, max_blocks=MAX_BATCH_BLOCKS):
    """Index batches over `lengths` sorted ascending: a batch closes when padded to its longest member it would pass
    max_tokens tokens or hold max_blocks blocks."""
    order = np.argsort(lengths, kind='stable')
    batches, start = [], 0
    for end in range(1, len(order) + 1):
        if end == len(order) or (end - start + 1) * lengths[order[end]] > max_tokens or end - start >= max_blocks:
            batches.append(order[start:end])
            start = end
    return batches


@torch.no_grad()
def embed_blocks(encoder, view: FoldView, blocks, device):
    """float16 [len(blocks), d_out]: `encoder` on the store blocks, length-bucketed, in bf16 autocast."""
    ptr = view.store.blk_tok_ptr
    lengths = np.minimum(np.asarray(ptr[blocks + 1]) - np.asarray(ptr[blocks]), encoder.max_len)
    out = np.empty((len(blocks), encoder.d_out), dtype=np.float16)
    for sel in length_buckets(np.maximum(lengths, 1)):
        tokens, mask = view.padded(blocks[sel], encoder.max_len)
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            vec = encoder(tokens.to(device), mask.to(device))
        out[sel] = vec.float().cpu().numpy().astype(np.float16)
    return out


def precompute(export_root, encoder_name, folds, enc_root=None, cache=None, device=None, log=print):
    """Write the block vectors of each fold; returns {fold: meta}. `enc_root` defaults to RUNS/encoders and `cache`
    to the cache the export was built from."""
    export_root = Path(export_root)
    store = BlockStore(export_root)
    cache = Path(cache or store.meta['cache'])
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    name = f'{export_root.name}_{encoder_name}'
    metas = {}
    for k in folds:
        start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        ckpt = Path(enc_root or RUNS / 'encoders') / encoder_name / f'fold{k}' / 'best.pt'
        check_checkpoint(ckpt, export_root, k)
        encoder = load_encoder(ckpt, device)
        view = FoldView(store, k, 'all')
        blocks, keys = view.distinct_blocks()
        vectors = embed_blocks(encoder, view, blocks, device)
        out = vec_dir(cache, name, k)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / 'keys.npy', keys)
        np.save(out / 'vectors.npy', vectors)
        manifest = store.fold(k)['manifest']
        secs = time.time() - start
        metas[k] = {'name': name, 'export': export_root.name, 'export_root': str(export_root),
                    'export_sha': store.export_sha, 'encoder': encoder_name, 'checkpoint': str(ckpt),
                    'checkpoint_sha': sha_of_file(ckpt), 'fold': k, 'manifest_sha': manifest['manifest_sha'],
                    'n_distinct': len(keys), 'n_blocks': manifest['n_blocks'], 'd': int(vectors.shape[1]),
                    'dtype': 'float16', 'autocast': 'bfloat16', 'device': device, 'max_len': encoder.max_len,
                    'secs': round(secs, 2), 'blocks_per_sec': round(len(keys) / max(secs, 1e-9)),
                    'peak_gpu_mb': round(torch.cuda.max_memory_allocated() / 2 ** 20) if device == 'cuda' else 0}
        (out / 'meta.json').write_text(json.dumps(metas[k], indent=1), encoding='utf-8')
        log(f"fold {k}: {len(keys)} distinct of {manifest['n_blocks']} blocks -> {out} ({vectors.shape[1]} dims, "
            f"{secs:.1f} s, {metas[k]['blocks_per_sec']} blocks/s on {device})")
    return metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder-name', required=True, help='RUNS/encoders/<name>/fold{k}/best.pt')
    ap.add_argument('--export', required=True, help='export root, or a name under RUNS/export')
    ap.add_argument('--folds', type=int, nargs='*', default=list(range(5)))
    ap.add_argument('--device')
    args = ap.parse_args()
    root = Path(args.export) if Path(args.export).exists() else EXPORT / args.export
    precompute(root, args.encoder_name, args.folds, device=args.device)


if __name__ == '__main__':
    main()
