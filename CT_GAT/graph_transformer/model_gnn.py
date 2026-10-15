"""Edge-typed hierarchical message-passing network: blocks -> function vectors -> binary logit, plain torch.
`RunConfig.control == 'no_edges'`, `untyped` and `no_call_mp` are read only in `HierGNN.structure`."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from config import RunConfig

N_EDGE_TYPES = 12


def segment_softmax(score, index, n):
    """score [M] -> softmax within each of n segments (index [M]), float32. Unused segments stay finite."""
    score = score.float()
    top = score.new_zeros(n).scatter_reduce(0, index, score, 'amax', include_self=False)
    e = torch.exp(score - top[index])
    return e / score.new_zeros(n).index_add_(0, index, e)[index]


def segment_sum(x, index, n):
    return x.new_zeros(n, *x.shape[1:]).index_add_(0, index, x)


class FixedReadout(nn.Module):
    """GELU(Linear(3d -> d)([sum ; mean ; max])) per segment."""

    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(3 * d, d)

    def forward(self, h, index, n):
        x = h.float()
        total = segment_sum(x, index, n)
        mean = total / segment_sum(torch.ones_like(x[:, :1]), index, n).clamp(min=1)
        top = x.new_zeros(n, x.shape[1]).scatter_reduce(0, index[:, None].expand_as(x), x, 'amax', include_self=False)
        return F.gelu(self.proj(torch.cat([total, mean, top], 1).to(h.dtype)))


class GatedReadout(nn.Module):
    """Attention pooling: score w.tanh(V h), softmax within each segment, weighted sum."""

    def __init__(self, d):
        super().__init__()
        self.v, self.w = nn.Linear(d, d), nn.Linear(d, 1, bias=False)

    def forward(self, h, index, n):
        alpha = segment_softmax(self.w(torch.tanh(self.v(h))).squeeze(-1), index, n)
        return segment_sum(alpha[:, None] * h.float(), index, n).to(h.dtype)


class DirGINELayer(nn.Module):
    """h + MLP(LN((1 + eps) h + W [m_in ; m_out])) with m_in = sum over in-edges of relu(h_src + E_in[type]) and
    m_out = sum over out-edges of relu(h_dst + E_out[type]); messages are accumulated in float32."""

    def __init__(self, d, n_types, dropout):
        super().__init__()
        self.e_in, self.e_out = nn.Embedding(n_types, d), nn.Embedding(n_types, d)
        self.eps = nn.Parameter(torch.zeros(1))
        self.mix = nn.Linear(2 * d, d)
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, src, dst, etype):
        x = h.float()
        m_in = segment_sum(F.relu(x[src] + self.e_in(etype).float()), dst, len(h))
        m_out = segment_sum(F.relu(x[dst] + self.e_out(etype).float()), src, len(h))
        z = self.norm(((1 + self.eps) * x + self.mix(torch.cat([m_in, m_out], 1).to(h.dtype))).to(h.dtype))
        return h + self.drop(self.mlp(z))


@dataclass
class Structure:
    blk_feat: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    etype: torch.Tensor
    call_src: torch.Tensor
    call_dst: torch.Tensor


class HierGNN(nn.Module):
    def __init__(self, cfg: RunConfig, n_tokens):
        super().__init__()
        d = cfg.d_model
        readout = {'fixed': FixedReadout, 'gated': GatedReadout}[cfg.readout]
        self.cfg = cfg
        self.tokens = nn.EmbeddingBag(n_tokens, d, mode='sum')
        self.blk_feat = nn.Linear(6, d)
        self.fn_layers = nn.ModuleList(DirGINELayer(d, N_EDGE_TYPES, cfg.dropout) for _ in range(cfg.gnn_fn_layers))
        self.fn_readout, self.seg_feat = readout(d), nn.Linear(2, d)
        self.bin_layers = nn.ModuleList(DirGINELayer(d, 1, cfg.dropout) for _ in range(cfg.gnn_bin_layers))
        self.bin_readout = readout(d)
        self.head = nn.Sequential(nn.Linear(d + 5 + cfg.stack_tfidf, d), nn.GELU(), nn.Dropout(cfg.dropout),
                                  nn.Linear(d, 1))

    def structure(self, batch):
        """Every structural input of the network, with block edge endpoints made batch-global. no_edges empties both
        edge lists and zeroes the degree columns, untyped sets every type to 0, no_call_mp empties the call edges."""
        feat, etype = batch['blk_feat'], batch['edge_type']
        seg_start = torch.cumsum(batch['seg_len'], 0) - batch['seg_len']
        src = seg_start[batch['edge_seg']] + batch['edge_src']
        dst = seg_start[batch['edge_seg']] + batch['edge_dst']
        call_src, call_dst = batch['call_src'], batch['call_dst']
        if self.cfg.control == 'no_edges':
            feat = feat.clone()
            feat[:, 4:6] = 0
            src, dst, etype = src[:0], dst[:0], etype[:0]
            call_src, call_dst = call_src[:0], call_dst[:0]
        if self.cfg.untyped:
            etype = torch.zeros_like(etype)
        if self.cfg.no_call_mp:
            call_src, call_dst = call_src[:0], call_dst[:0]
        return Structure(feat, src, dst, etype, call_src, call_dst)

    def forward(self, batch):
        n_seg = len(batch['seg_len'])
        assert batch['n_fn'] == n_seg, f"one segment per function expected: {n_seg} segments, {batch['n_fn']} functions"
        st = self.structure(batch)
        h = self.tokens(batch['tokens'], batch['tok_offsets'], per_sample_weights=batch['tok_weights'])
        h = h + self.blk_feat(st.blk_feat)
        for layer in self.fn_layers:
            h = layer(h, st.src, st.dst, st.etype)
        f = self.fn_readout(h, batch['blk_seg'], n_seg) + self.seg_feat(batch['seg_feat']).to(h.dtype)
        call_type = torch.zeros_like(st.call_src)
        for layer in self.bin_layers:
            f = layer(f, st.call_src, st.call_dst, call_type)
        n_bin = len(batch['label'])
        z = [self.bin_readout(f, batch['seg_bin'], n_bin), batch['bin_feat']]
        if self.cfg.stack_tfidf:
            z.append(batch['stack_logit'][:, None])
        return self.head(torch.cat([t.to(z[0].dtype) for t in z], 1)).squeeze(-1).float()
