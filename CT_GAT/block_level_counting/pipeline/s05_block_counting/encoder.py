"""Block vectors: a weighted bag of tokens through an EmbeddingBag plus a Linear over the structural block features."""
from torch import nn


class BlockEncoder(nn.Module):
    """EmbeddingBag(tokens, per-token weights) + Linear(blk_feat). HierGNN adopts `tokens` and `blk_feat` under those
    names, so the checkpoint keys stay `tokens.*` and `blk_feat.*`."""

    def __init__(self, n_tokens, d):
        super().__init__()
        self.tokens = nn.EmbeddingBag(n_tokens, d, mode='sum')
        self.blk_feat = nn.Linear(6, d)

    def forward(self, batch, blk_feat):
        h = self.tokens(batch['tokens'], batch['tok_offsets'], per_sample_weights=batch['tok_weights'])
        return h + self.blk_feat(blk_feat)
