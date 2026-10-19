"""Binary-level features and the final MLP head."""
import numpy as np
from torch import nn

from pipeline.s10_unused_import_features.imports_feat import IMPORT_DIM

N_BIN_FEAT = 5
N_IMPORT_FEAT = 2


def bin_feat(z):
    """log1p of the full binary's counts: blocks, functions, instructions, CFG edges, call edges."""
    return np.log1p([len(z['blk_feat']), len(z['fn_flags']), len(z['mn']), len(z['cfg_edges']),
                     len(z['call_edges'].reshape(-1, 2))]).astype(np.float32)


def make_head(d, cfg):
    """Input: binary vector, bin_feat, and with `stack_tfidf` one logit, with `imports` the import bag (IMPORT_DIM) and
    its two log counts."""
    d_in = d + N_BIN_FEAT + cfg.stack_tfidf + (IMPORT_DIM + N_IMPORT_FEAT if cfg.imports else 0)
    return nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, 1))
