"""Binary-level features and the final MLP head."""
import numpy as np
from torch import nn

N_BIN_FEAT = 5


def bin_feat(z):
    """log1p of the full binary's counts: blocks, functions, instructions, CFG edges, call edges."""
    return np.log1p([len(z['blk_feat']), len(z['fn_flags']), len(z['mn']), len(z['cfg_edges']),
                     len(z['call_edges'].reshape(-1, 2))]).astype(np.float32)


def make_head(d, cfg):
    return nn.Sequential(nn.Linear(d + N_BIN_FEAT + cfg.stack_tfidf, d), nn.GELU(), nn.Dropout(cfg.dropout),
                         nn.Linear(d, 1))
