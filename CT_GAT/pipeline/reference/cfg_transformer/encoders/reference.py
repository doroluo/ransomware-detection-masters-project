"""A tiny deterministic encoder for tests and smoke runs: embedding, masked mean pool, linear. Random weights from a
fixed seed; it is not trained and says nothing about what a transformer would learn.

python -m reference.cfg_transformer.encoders.reference --export RUNS/export/ours_cap0 --name ref --folds 0 1 2 3 4
writes RUNS/encoders/<name>/fold{k}/best.pt, a valid checkpoint (encoder_contract.py) per fold.
"""
import argparse
from pathlib import Path

import torch
from torch import nn

from common.config import EXPORT, RUNS
from reference.cfg_transformer.encoder_contract import save_checkpoint
from common.reader import BlockStore


class ReferenceEncoder(nn.Module):
    def __init__(self, n_ids, d_emb=32, d_out=64, max_len=192, seed=0):
        super().__init__()
        self.d_out, self.max_len = d_out, max_len
        self.embed = nn.Embedding(n_ids, d_emb, padding_idx=0)
        self.proj = nn.Linear(d_emb, d_out)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(n_ids, d_emb, generator=g))
            self.embed.weight[0] = 0
            self.proj.weight.copy_(torch.randn(d_out, d_emb, generator=g) / d_emb ** 0.5)
            self.proj.bias.zero_()

    def forward(self, tokens, mask):
        weight = mask.to(self.embed.weight.dtype)[..., None]
        pooled = (self.embed(tokens) * weight).sum(1) / weight.sum(1).clamp(min=1)
        return self.proj(pooled)


def write_reference_checkpoints(export_root, name, folds=range(5), out_root=None, **kwargs):
    """A ReferenceEncoder per fold with the fold's vocabulary size, saved with the fold's whole train and val sets as
    its training and selection ids; returns the paths. `out_root` defaults to RUNS/encoders."""
    store = BlockStore(export_root)
    paths = []
    for k in folds:
        manifest = store.fold(k)['manifest']
        ids = lambda split: [store.binaries[i]['sample_id'] for i in manifest[split]]
        init = {'n_ids': store.fold(k)['vocab']['n_ids'], **kwargs}
        path = Path(out_root or RUNS / 'encoders') / name / f'fold{k}' / 'best.pt'
        save_checkpoint(path, ReferenceEncoder(**init), init, export_root, k, ids('train'), ids('val'))
        paths.append(path)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', required=True, help='export root, or a name under RUNS/export')
    ap.add_argument('--name', default='ref')
    ap.add_argument('--folds', type=int, nargs='*', default=list(range(5)))
    args = ap.parse_args()
    root = Path(args.export) if Path(args.export).exists() else EXPORT / args.export
    for path in write_reference_checkpoints(root, args.name, args.folds):
        print(path)


if __name__ == '__main__':
    main()
