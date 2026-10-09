"""Hierarchical CFG graph transformer: blocks -> function segments (shortest-path attention bias) -> binary
(call-graph attention bias). `RunConfig.no_graph` is read only in `GraphTransformer.structure`."""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from config import RunConfig

LENGTH_BUCKETS = (8, 16, 32, 64, 128)
SPD_BUCKETS = 6
SPD_NONE = SPD_BUCKETS


class Attention(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x, bias):
        n, length, d = x.shape
        q, k, v = self.qkv(x).view(n, length, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype),
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(n, length, d))


class Layer(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias):
        x = x + self.drop(self.attn(self.norm1(x), bias))
        return x + self.drop(self.ffn(self.norm2(x)))


@torch.autocast('cuda', enabled=False)
def shortest_path_buckets(adj):
    """adj [n, L, L] bool, i -> j edges. Returns directed distance buckets 0, 1, 2, 3, 4+ and 5 = unreachable."""
    n, length, _ = adj.shape
    eye = torch.eye(length, dtype=torch.bool, device=adj.device).expand(n, -1, -1)
    w1 = adj | eye
    step = w1.half()
    w2 = torch.bmm(w1.half(), step) > 0
    w3 = torch.bmm(w2.half(), step) > 0
    reach = w3
    for _ in range(max(0, math.ceil(math.log2(max(length, 2))) - 1)):
        reach = torch.bmm(reach.half(), reach.half()) > 0
    out = torch.full_like(adj, 5, dtype=torch.long)
    out[reach] = 4
    out[w3] = 3
    out[w2] = 2
    out[w1] = 1
    out[eye] = 0
    return out


@dataclass
class Structure:
    blk_feat: torch.Tensor
    deg_in: torch.Tensor
    deg_out: torch.Tensor
    spd: dict
    call: torch.Tensor
    same_fn: torch.Tensor


class GraphTransformer(nn.Module):
    def __init__(self, cfg: RunConfig, n_tokens):
        super().__init__()
        d, h = cfg.d_model, cfg.heads
        self.cfg = cfg
        self.tokens = nn.EmbeddingBag(n_tokens, d, mode='mean')
        self.blk_feat = nn.Linear(6, d)
        self.deg_in = nn.Embedding(cfg.degree_clip + 2, d)
        self.deg_out = nn.Embedding(cfg.degree_clip + 2, d)
        self.spd_fwd = nn.Embedding(SPD_BUCKETS + 1, h)
        self.spd_bwd = nn.Embedding(SPD_BUCKETS + 1, h)
        self.fn_layers = nn.ModuleList(Layer(d, h, cfg.dropout) for _ in range(cfg.fn_layers))
        self.pool_norm, self.pool_score = nn.LayerNorm(d), nn.Linear(d, 1)
        self.seg_feat = nn.Linear(2, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.call_bias = nn.Parameter(torch.zeros(3, h))
        self.bin_layers = nn.ModuleList(Layer(d, h, cfg.dropout) for _ in range(cfg.bin_layers))
        self.head_norm, self.head = nn.LayerNorm(d), nn.Linear(d, 1)
        nn.init.normal_(self.cls, std=0.02)

    def length_groups(self, batch):
        """Segments grouped by padded length, so short functions are not padded to 128."""
        groups, low = {}, 0
        for cap in LENGTH_BUCKETS:
            cap = min(cap, self.cfg.max_segment_blocks)
            members = torch.nonzero((batch['seg_len'] > low) & (batch['seg_len'] <= cap)).flatten()
            if len(members):
                groups[cap] = members
            low = cap
            if cap == self.cfg.max_segment_blocks:
                break
        return groups

    def structure(self, batch, groups):
        """Every structural input of the network. With no_graph they become constants and the degree columns zero."""
        feat = batch['blk_feat']
        n_seg = len(batch['seg_len'])
        if self.cfg.no_graph:
            none_deg = torch.full_like(batch['deg_in'], self.cfg.degree_clip + 1)
            feat = feat.clone()
            feat[:, 4:6] = 0
            spd = {cap: torch.full((len(m), cap, cap), SPD_NONE, dtype=torch.long, device=feat.device)
                   for cap, m in groups.items()}
            return Structure(feat, none_deg, none_deg, spd, self.pairwise_zero(batch), self.pairwise_zero(batch))
        spd = {}
        row = torch.empty(n_seg, dtype=torch.long, device=feat.device)
        for cap, members in groups.items():
            row[members] = torch.arange(len(members), device=feat.device)
            in_group = torch.isin(batch['edge_seg'], members)
            adj = torch.zeros(len(members), cap, cap, dtype=torch.bool, device=feat.device)
            adj[row[batch['edge_seg'][in_group]], batch['edge_src'][in_group], batch['edge_dst'][in_group]] = True
            spd[cap] = shortest_path_buckets(adj)
        calls = torch.zeros(batch['n_fn'], batch['n_fn'], dtype=torch.bool, device=feat.device)
        calls[batch['call_src'], batch['call_dst']] = True
        fn = self.pad_segments(batch, batch['seg_fn'], fill=-1)
        valid = fn >= 0
        fc = fn.clamp(min=0)
        pair_valid = valid[:, :, None] & valid[:, None, :]
        call = calls[fc[:, :, None], fc[:, None, :]] & pair_valid
        same = (fn[:, :, None] == fn[:, None, :]) & pair_valid
        return Structure(feat, batch['deg_in'], batch['deg_out'], spd, call, same)

    def pairwise_zero(self, batch):
        n_bin, width = len(batch['label']), int(torch.bincount(batch['seg_bin']).max())
        return torch.zeros(n_bin, width, width, dtype=torch.bool, device=batch['seg_bin'].device)

    def pad_segments(self, batch, values, fill):
        """[S, ...] segment values -> [binaries, max segments, ...] padded with fill."""
        counts = torch.bincount(batch['seg_bin'], minlength=len(batch['label']))
        width = int(counts.max())
        slot = torch.arange(len(values), device=values.device) - (torch.cumsum(counts, 0) - counts)[batch['seg_bin']]
        out = torch.full((len(counts), width, *values.shape[1:]), fill, dtype=values.dtype, device=values.device)
        out[batch['seg_bin'], slot] = values
        return out

    def forward(self, batch):
        groups = self.length_groups(batch)
        st = self.structure(batch, groups)
        h = self.tokens(batch['tokens'], batch['tok_offsets']) + self.blk_feat(st.blk_feat)
        h = h + self.deg_in(st.deg_in) + self.deg_out(st.deg_out)

        seg_vec = torch.zeros(len(batch['seg_len']), h.shape[1], dtype=h.dtype, device=h.device)
        row = torch.empty(len(batch['seg_len']), dtype=torch.long, device=h.device)
        for cap, members in groups.items():
            row[members] = torch.arange(len(members), device=h.device)
            in_group = torch.isin(batch['blk_seg'], members)
            x = torch.zeros(len(members), cap, h.shape[1], dtype=h.dtype, device=h.device)
            x[row[batch['blk_seg'][in_group]], batch['blk_pos'][in_group]] = h[in_group]
            valid = torch.arange(cap, device=h.device)[None, :] < batch['seg_len'][members][:, None]
            bias = (self.spd_fwd(st.spd[cap]) + self.spd_bwd(st.spd[cap].transpose(1, 2))).permute(0, 3, 1, 2)
            bias = bias.masked_fill(~valid[:, None, None, :], float('-inf'))
            for layer in self.fn_layers:
                x = layer(x, bias)
            score = self.pool_score(self.pool_norm(x)).squeeze(-1).float().masked_fill(~valid, float('-inf'))
            seg_vec[members] = (torch.softmax(score, -1).unsqueeze(-1).to(x.dtype) * x).sum(1).to(h.dtype)

        seg_vec = seg_vec + self.seg_feat(batch['seg_feat']).to(seg_vec.dtype)
        x = self.pad_segments(batch, seg_vec, fill=0.0)
        n_bin, width = x.shape[:2]
        valid = self.pad_segments(batch, torch.ones_like(batch['seg_bin'], dtype=torch.bool), fill=False)
        x = torch.cat([self.cls.expand(n_bin, -1, -1).to(x.dtype), x], 1)
        rel = torch.stack([st.call, st.call.transpose(1, 2), st.same_fn], -1).to(self.call_bias.dtype)
        bias = torch.zeros(n_bin, self.cfg.heads, width + 1, width + 1, device=x.device)
        bias[:, :, 1:, 1:] = (rel @ self.call_bias).permute(0, 3, 1, 2)
        key_valid = torch.cat([torch.ones(n_bin, 1, dtype=torch.bool, device=x.device), valid], 1)
        bias = bias.masked_fill(~key_valid[:, None, None, :], float('-inf'))
        for layer in self.bin_layers:
            x = layer(x, bias)
        return self.head(self.head_norm(x[:, 0])).squeeze(-1).float()
