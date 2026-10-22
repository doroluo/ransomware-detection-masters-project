"""Call edges between the kept functions, and the message-passing layers that run over them."""
import numpy as np
import torch
from torch import nn

from reference.gnn.s06_gnn_block_cfg.layers import DirGINELayer


def call_pairs(z, fn):
    """Unique (caller, callee) pairs among the kept functions `fn`, as indices into `fn`; self calls are dropped."""
    n_seg = len(fn)
    fn_local = np.full(len(z['fn_flags']), -1, dtype=np.int64)
    fn_local[fn] = np.arange(n_seg)
    c = z['call_edges'].astype(np.int64).reshape(-1, 2)
    cs, ct = fn_local[c[:, 0]], fn_local[c[:, 1]]
    ckeep = (cs >= 0) & (ct >= 0) & (cs != ct)
    pairs = np.unique(cs[ckeep] * n_seg + ct[ckeep])
    return pairs // max(n_seg, 1), pairs % max(n_seg, 1)


def call_layers(d, cfg):
    return nn.ModuleList(DirGINELayer(d, 1, cfg.dropout) for _ in range(cfg.gnn_bin_layers))


def run_call_layers(layers, f, call_src, call_dst):
    call_type = torch.zeros_like(call_src)
    for layer in layers:
        f = layer(f, call_src, call_dst, call_type)
    return f
