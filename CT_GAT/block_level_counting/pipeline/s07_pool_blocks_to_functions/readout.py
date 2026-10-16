"""Segment readouts: fixed (sum/mean/max projection) or gated attention. Step 9 reuses them for functions -> binary."""
import torch
import torch.nn.functional as F
from torch import nn

from pipeline.s06_gnn_block_cfg.layers import segment_softmax, segment_sum


class FixedReadout(nn.Module):
    """GELU(Linear(3d -> d)([sum / sqrt(n) ; mean ; max])) per segment."""

    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(3 * d, d)

    @staticmethod
    def pool(h, index, n):
        x = h.float()
        total = segment_sum(x, index, n)
        count = segment_sum(torch.ones_like(x[:, :1]), index, n).clamp(min=1)
        top = x.new_zeros(n, x.shape[1]).scatter_reduce(0, index[:, None].expand_as(x), x, 'amax', include_self=False)
        return torch.cat([total / count.sqrt(), total / count, top], 1)

    def forward(self, h, index, n):
        return F.gelu(self.proj(self.pool(h, index, n).to(h.dtype)))


class GatedReadout(nn.Module):
    """Attention pooling: score w.tanh(V h), softmax within each segment, weighted sum."""

    def __init__(self, d):
        super().__init__()
        self.v, self.w = nn.Linear(d, d), nn.Linear(d, 1, bias=False)

    def forward(self, h, index, n):
        alpha = segment_softmax(self.w(torch.tanh(self.v(h))).squeeze(-1), index, n)
        return segment_sum(alpha[:, None] * h.float(), index, n).to(h.dtype)


READOUTS = {'fixed': FixedReadout, 'gated': GatedReadout}
