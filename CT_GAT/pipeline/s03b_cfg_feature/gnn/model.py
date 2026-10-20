"""Edge-typed hierarchical message-passing network: blocks -> function vectors -> binary logit, plain torch.
`RunConfig.control == 'no_edges'`, `untyped` and `no_call_mp` are read only in `HierGNN.structure`.
Stage order: s05 block vectors (token bags, or precomputed vectors with block_input='vec'), s06 block-CFG layers,
s07 blocks -> functions, s08 call-graph layers, s09 functions -> binary, s10 import bag (`imports`), s11 head. Attribute names are the checkpoint
keys."""
from dataclasses import dataclass

import torch
from torch import nn

from common.config import RunConfig
from s03b_cfg_feature.gnn.s05_block_counting.encoder import BlockEncoder, BlockVecEncoder
from s03b_cfg_feature.gnn.s06_gnn_block_cfg.edges import N_EDGE_TYPES
from s03b_cfg_feature.gnn.s06_gnn_block_cfg.layers import DirGINELayer
from s03b_cfg_feature.gnn.s07_pool_blocks_to_functions.readout import READOUTS
from s03b_cfg_feature.gnn.s08_gnn_call_graph.calls import call_layers, run_call_layers
from s03b_cfg_feature.gnn.s10_unused_import_features.imports_feat import ImportBag
from s03b_cfg_feature.gnn.s11_final_classifier.head import make_head


@dataclass
class Structure:
    blk_feat: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    etype: torch.Tensor
    call_src: torch.Tensor
    call_dst: torch.Tensor


class HierGNN(nn.Module):
    def __init__(self, cfg: RunConfig, n_tokens, d_vec=None, n_imports=0):
        super().__init__()
        d = cfg.d_model
        readout = READOUTS[cfg.readout]
        self.cfg = cfg
        if cfg.block_input == 'vec':
            if not d_vec:
                raise ValueError("block_input='vec' needs d_vec, the size of the precomputed block vectors")
            encoder = BlockVecEncoder(d_vec, d)
            self.vec, self.blk_feat, self.encode_blocks = encoder.vec, encoder.blk_feat, encoder.forward
        else:
            encoder = BlockEncoder(n_tokens, d)
            self.tokens, self.blk_feat, self.encode_blocks = encoder.tokens, encoder.blk_feat, encoder.forward
        self.fn_layers = nn.ModuleList(DirGINELayer(d, N_EDGE_TYPES, cfg.dropout, cfg.direction)
                                      for _ in range(cfg.gnn_fn_layers))
        self.fn_readout, self.seg_feat = readout(d), nn.Linear(2, d)
        self.bin_layers = call_layers(d, cfg)
        self.bin_readout = readout(d)
        if cfg.imports:
            self.imports = ImportBag(n_imports)
        self.head = make_head(d, cfg)

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
        h = self.encode_blocks(batch, st.blk_feat)
        for layer in self.fn_layers:
            h = layer(h, st.src, st.dst, st.etype)
        f = self.fn_readout(h, batch['blk_seg'], n_seg) + self.seg_feat(batch['seg_feat']).to(h.dtype)
        f = run_call_layers(self.bin_layers, f, st.call_src, st.call_dst)
        n_bin = len(batch['label'])
        z = [self.bin_readout(f, batch['seg_bin'], n_bin), batch['bin_feat']]
        if self.cfg.stack_tfidf:
            z.append(batch['stack_logit'][:, None])
        if self.cfg.imports:
            z += [self.imports(batch), batch['imp_feat']]
        return self.head(torch.cat([t.to(z[0].dtype) for t in z], 1)).squeeze(-1).float()
