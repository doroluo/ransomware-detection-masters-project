"""Directed, edge-typed GIN layer over a block CFG, and the scatter helpers it shares with the readouts."""
import torch
import torch.nn.functional as F
from torch import nn


def segment_softmax(score, index, n):
    """score [M] -> softmax within each of n segments (index [M]), float32. Unused segments stay finite."""
    score = score.float()
    top = score.new_zeros(n).scatter_reduce(0, index, score, 'amax', include_self=False)
    e = torch.exp(score - top[index])
    return e / score.new_zeros(n).index_add_(0, index, e)[index]


def segment_sum(x, index, n):
    return x.new_zeros(n, *x.shape[1:]).index_add_(0, index, x)


class DirGINELayer(nn.Module):
    """h + MLP(LN((1 + eps) h + W [m_in ; m_out])) with m_in = sum over in-edges of relu(h_src + E_in[type]) and
    m_out = sum over out-edges of relu(h_dst + E_out[type]); messages are accumulated in float32.
    `direction` 'forward' zeroes the m_out half (messages from predecessors only), 'backward' the m_in half, and
    'undirected' feeds m_in + m_out, both computed with the e_in bank, into both halves."""

    def __init__(self, d, n_types, dropout, direction='both'):
        super().__init__()
        self.direction = direction
        self.e_in, self.e_out = nn.Embedding(n_types, d), nn.Embedding(n_types, d)
        self.eps = nn.Parameter(torch.zeros(1))
        self.mix = nn.Linear(2 * d, d)
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, src, dst, etype):
        x = h.float()
        n = len(h)
        if self.direction == 'undirected':
            e_in = self.e_in(etype).float()
            m_in = m_out = (segment_sum(F.relu(x[src] + e_in), dst, n) + segment_sum(F.relu(x[dst] + e_in), src, n))
        else:
            m_in = m_out = x.new_zeros(x.shape)
            if self.direction != 'backward':
                m_in = segment_sum(F.relu(x[src] + self.e_in(etype).float()), dst, n)
            if self.direction != 'forward':
                m_out = segment_sum(F.relu(x[dst] + self.e_out(etype).float()), src, n)
        z = self.norm(((1 + self.eps) * x + self.mix(torch.cat([m_in, m_out], 1).to(h.dtype))).to(h.dtype))
        return h + self.drop(self.mlp(z))
