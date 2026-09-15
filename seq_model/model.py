#!/usr/bin/env python3
"""The sequence model: embedding -> 1-D conv stem -> yanping ViT blocks -> MIL.

Shape of the thing
------------------
    ids      (B, M, W)   M windows of W mnemonic token ids per file
    lens     (B, M)      how many of the W are real tokens
    win_mask (B, M)      which of the M windows exist for this file

    nn.Embedding(V, 64, padding_idx=0)          categorical, not ordinal
    ConvStem1d      64 -> 32 -> 64   stride 4    (the 1-D analogue of
    PatchMerging1d  64 -> 128        stride 2     CNN-ViT/model_train.py's
    PatchMerging1d  128 -> 256       stride 2     ConvStem / PatchMerging)
    -> (B*M, W/16, 256)
    6 x TransformerBlock(dim=256, heads=8, mlp_dim=512, dropout=0.25)
       IMPORTED from CNN-ViT/model_train.py, not reimplemented
    masked mean pool -> (B*M, 256)
    [optional] concat hashed bag-of-imports (2048) -> (B*M, 256+2048)
    mlp_head -> per-window logits -> MIL aggregate -> per-file logits

Why W/16 = 256 is not a coincidence: the image model's ViT saw exactly 256
patches, so with the default W = 4,096 the transformer stack here is the same
transformer stack, at the same sequence length, with the same relative
position-bias table. The only thing that changed is what a "patch" is made of.

Positional encoding
-------------------
None is added. `RobustRelativeAttention` already carries a LEARNED RELATIVE
position-bias table (`2*num_patches-1` x heads), which is the positional
signal the image model used, and it is the right one here: what matters in an
instruction stream is how far apart two blocks are, not where the window
happened to start in the file. `num_patches` is set to W//16, so the table
grows with the window length.

Padding is structurally unreadable
----------------------------------
The image encoder padded its canvas with a constant, and the constant told the
model the file's size. Here padding is never content:

  * `padding_idx=0` makes the pad embedding an exact zero vector;
  * every normalisation in the stem is POSITION-WISE (LayerNorm over channels
    at each position), never BatchNorm/GroupNorm over the length axis, so no
    statistic of the padded region can reach a real position;
  * the two squeeze-excitation gates average over VALID positions only;
  * the token mask is carried down through the stem by min-pooling with the
    same kernels and strides as the convolutions, so an output position is
    marked valid only when EVERY input it can see is a real token;
  * the transformer masks invalid positions out of the attention keys and the
    pooling averages over valid positions only.

The consequence is exact for any window holding at least `RECEPTIVE_FIELD`
real tokens, and `tests/test_seq_model.py` asserts it: writing arbitrary junk
into the padded region of such a window does not move the pooled
representation at all - not "by less than a tolerance", bit-for-bit.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
YANPING = REPO / "CNN-ViT" / "model_train.py"

STEM_STRIDE = 16          # ConvStem (4) x PatchMerging (2) x PatchMerging (2)
# One block position sees this many input tokens: 3 -> 7 -> 11 -> 19 through
# the two k=3,s=2 convolutions and the two k=2,s=2 merges. A window with at
# least this many real tokens has at least one block position whose receptive
# field contains no pad at all, which is what makes the padding guarantee
# exact; a shorter window keeps its first block position and that position
# necessarily mixes in the zero pad embedding. Two of the 3,846 cohort files
# are that short (2 and 12 mnemonics), and for them the mixed-in value is a
# constant zero, not a function of file size.
RECEPTIVE_FIELD = 19


# ---------------------------------------------------------------------------
# importing the yanping encoder blocks
# ---------------------------------------------------------------------------
def _torchvision_shim() -> None:
    """model_train.py's only torchvision use is `transforms.ToTensor()`, which
    it applies to PIL images this model never sees. The venv has no
    torchvision, so a stand-in is registered rather than COPYING the encoder
    blocks - copies go stale the moment anyone edits model_train.py."""
    if "torchvision" in sys.modules:
        return
    try:
        import torchvision  # noqa: F401
        return
    except ImportError:
        pass

    class _ToTensor:
        def __call__(self, img):
            import numpy as np
            a = np.asarray(img, dtype="float32") / 255.0
            if a.ndim == 2:
                a = a[None]
            return torch.from_numpy(a)

    transforms = types.ModuleType("torchvision.transforms")
    transforms.ToTensor = _ToTensor
    transforms.Compose = lambda fns: (lambda x: [x := f(x) for f in fns][-1])
    tv = types.ModuleType("torchvision")
    tv.transforms = transforms
    tv.__SHIM__ = True
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.transforms"] = transforms


_YP = None


def yanping():
    """CNN-ViT/model_train.py as a module (its __main__ block never runs)."""
    global _YP
    if _YP is None:
        _torchvision_shim()
        spec = importlib.util.spec_from_file_location(
            "yanping_model_train", YANPING)
        mod = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("yanping_model_train", mod)
        spec.loader.exec_module(mod)
        _YP = mod
    return _YP


# ---------------------------------------------------------------------------
# mask plumbing
# ---------------------------------------------------------------------------
def minpool_mask(mask: torch.Tensor, kernel: int, stride: int,
                 pad: int) -> torch.Tensor:
    """Propagate a (B, L) validity mask through a conv of this geometry.

    An output position is valid only if every input in its receptive field is.
    The convolution's own structural zero-padding is position-determined and
    identical for every sample, so the mask is replicate-padded rather than
    zero-padded: at the left edge it copies mask[0] (always valid), at the
    right edge mask[-1] (valid for a full window, invalid for a padded one).
    """
    m = mask.to(torch.float32).unsqueeze(1)           # (B, 1, L)
    if pad:
        m = F.pad(m, (pad, pad), mode="replicate")
    m = -F.max_pool1d(-m, kernel_size=kernel, stride=stride)
    return (m.squeeze(1) > 0.5)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Mean of `x` over `dim`, counting only positions where `mask` is True."""
    m = mask.unsqueeze(-1).to(x.dtype)
    s = (x * m).sum(dim=dim)
    n = m.sum(dim=dim).clamp(min=1e-6)
    return s / n


class PosLayerNorm(nn.Module):
    """LayerNorm over the channel axis of a (B, C, L) tensor, independently at
    every position. BatchNorm and GroupNorm both mix the length axis into their
    statistics, which is precisely how a constant pad leaks file size."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class MaskedSE1d(nn.Module):
    """`SqueezeExcitation` from CNN-ViT/model_train.py, 1-D and mask-aware.

    Same two-layer bottleneck gate and same reduction; the only change is that
    the global average pool runs over valid positions instead of over the whole
    (padded) length.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        r = max(1, channels // reduction)
        self.fc = nn.Sequential(nn.Linear(channels, r, bias=False), nn.GELU(),
                                nn.Linear(r, channels, bias=False), nn.Sigmoid())

    def forward(self, x, mask):
        m = mask.unsqueeze(1).to(x.dtype)                  # (B, 1, L)
        pooled = (x * m).sum(-1) / m.sum(-1).clamp(min=1e-6)
        return x * self.fc(pooled).unsqueeze(-1)


class ConvStem1d(nn.Module):
    """1-D analogue of model_train.ConvStem (two stride-2 convs + SE).

    Adapted from `CNN-ViT/model_train.py::ConvStem` (yanping): Conv2d -> Conv1d,
    BatchNorm2d -> position-wise LayerNorm, SqueezeExcitation -> MaskedSE1d.
    """

    def __init__(self, in_ch: int, mid: int = 32, out: int = 64):
        super().__init__()
        self.c1 = nn.Conv1d(in_ch, mid, kernel_size=3, stride=2, padding=1)
        self.n1 = PosLayerNorm(mid)
        self.c2 = nn.Conv1d(mid, out, kernel_size=3, stride=2, padding=1)
        self.n2 = PosLayerNorm(out)
        self.se = MaskedSE1d(out)
        self.act = nn.GELU()

    def forward(self, x, mask):
        mask = minpool_mask(mask, 3, 2, 1)
        x = self.act(self.n1(self.c1(x)))
        mask = minpool_mask(mask, 3, 2, 1)
        x = self.act(self.n2(self.c2(x)))
        return self.se(x, mask), mask


class PatchMerging1d(nn.Module):
    """1-D analogue of model_train.PatchMerging: a stride-2 reduction with an
    average-pool + 1x1 shortcut and an SE gate. Adapted from
    `CNN-ViT/model_train.py::PatchMerging` (yanping), with GroupNorm replaced
    by the position-wise LayerNorm for the reason given in the module
    docstring."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.reduction = nn.Conv1d(in_ch, out_ch, kernel_size=2, stride=2)
        self.norm = PosLayerNorm(out_ch)
        self.act = nn.GELU()
        self.shortcut = nn.Sequential(nn.AvgPool1d(kernel_size=2, stride=2),
                                      nn.Conv1d(in_ch, out_ch, kernel_size=1))
        self.se = MaskedSE1d(out_ch)

    def forward(self, x, mask):
        mask = minpool_mask(mask, 2, 2, 0)
        res = self.shortcut(x)
        x = self.act(self.norm(self.reduction(x)))
        return self.se(x + res, mask), mask


# ---------------------------------------------------------------------------
# encoder
# ---------------------------------------------------------------------------
class SeqEncoder(nn.Module):
    """Token ids -> (B, W/16, dim) contextualised block embeddings + mask."""

    def __init__(self, vocab_size: int, window: int = 4096, emb_dim: int = 64,
                 dim: int = 256, depth: int = 6, heads: int = 8,
                 mlp_dim: int = 512, dropout: float = 0.25,
                 drop_path_rate: float = 0.0):
        super().__init__()
        yp = yanping()
        if window % STEM_STRIDE:
            raise ValueError(f"window {window} must be a multiple of {STEM_STRIDE}")
        self.window = window
        self.num_patches = window // STEM_STRIDE
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.stem = ConvStem1d(emb_dim, 32, 64)
        self.merge1 = PatchMerging1d(64, dim // 2)
        self.merge2 = PatchMerging1d(dim // 2, dim)
        self.pos_drop = nn.Dropout(dropout)
        dpr = [float(x) for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            yp.TransformerBlock(dim=dim, num_patches=self.num_patches,
                                heads=heads, mlp_dim=mlp_dim, dropout=dropout,
                                drop_path=dpr[i])
            for i in range(depth)])

    def forward(self, ids: torch.Tensor, tok_mask: torch.Tensor):
        x = self.embed(ids).transpose(1, 2)               # (B, emb, W)
        x, m = self.stem(x, tok_mask)
        x, m = self.merge1(x, m)
        x, m = self.merge2(x, m)
        x = self.pos_drop(x.transpose(1, 2))              # (B, L, dim)
        # a stream shorter than the stem's receptive field would mask out
        # everything; keep its first block so softmax never sees an all -inf row
        empty = ~m.any(dim=1)
        if empty.any():
            m = m.clone()
            m[empty, 0] = True
        key_pad = ~m
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_pad)
        return x, m


class AttentionPool(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(),
                                   nn.Linear(hidden, 1))

    def forward(self, x, mask):
        s = self.score(x).squeeze(-1)
        s = s.masked_fill(~mask, float("-inf"))
        w = torch.softmax(s, dim=1).unsqueeze(-1)
        return (x * w).sum(dim=1)


class SeqTransformer(nn.Module):
    """Multiple-instance classifier over the windows of one file."""

    def __init__(self, vocab_size: int, window: int = 4096, emb_dim: int = 64,
                 dim: int = 256, depth: int = 6, heads: int = 8,
                 mlp_dim: int = 512, dropout: float = 0.25,
                 drop_path_rate: float = 0.0, num_classes: int = 2,
                 pooling: str = "masked_mean", mil: str = "mean_logits",
                 imports_dim: int = 0):
        super().__init__()
        self.encoder = SeqEncoder(vocab_size, window, emb_dim, dim, depth,
                                  heads, mlp_dim, dropout, drop_path_rate)
        if pooling not in ("masked_mean", "attention"):
            raise ValueError(pooling)
        if mil not in ("mean_logits", "attention"):
            raise ValueError(mil)
        self.pooling, self.mil = pooling, mil
        self.attn_pool = AttentionPool(dim) if pooling == "attention" else None
        self.imports_dim = int(imports_dim)
        head_in = dim + self.imports_dim
        # the image model's mlp_head, widened by the imports block when present
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(head_in), nn.Linear(head_in, mlp_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(mlp_dim, num_classes))
        self.mil_score = (nn.Sequential(nn.Linear(dim, 128), nn.Tanh(),
                                        nn.Linear(128, 1))
                          if mil == "attention" else None)

    # -- one window at a time ---------------------------------------------
    def window_features(self, ids, tok_mask):
        x, m = self.encoder(ids, tok_mask)
        if self.pooling == "attention":
            return self.attn_pool(x, m)
        return masked_mean(x, m, dim=1)

    def forward(self, ids, lens, win_mask, imports=None):
        """ids (B, M, W) long; lens (B, M) long; win_mask (B, M) bool;
        imports (B, imports_dim) float or None.

        Returns (file_logits (B, C), window_logits (B, M, C))."""
        B, M, W = ids.shape
        flat_ids = ids.reshape(B * M, W)
        flat_len = lens.reshape(B * M)
        keep = win_mask.reshape(B * M)
        idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        sel_ids = flat_ids.index_select(0, idx)
        pos = torch.arange(W, device=ids.device).unsqueeze(0)
        sel_mask = pos < flat_len.index_select(0, idx).unsqueeze(1)
        feats = self.window_features(sel_ids, sel_mask)          # (n_keep, dim)

        dim = feats.shape[-1]
        full = feats.new_zeros(B * M, dim)
        full = full.index_copy(0, idx, feats)
        wf = full.view(B, M, dim)

        if self.imports_dim:
            if imports is None:
                imports = wf.new_zeros(B, self.imports_dim)
            head_in = torch.cat(
                [wf, imports.unsqueeze(1).expand(B, M, self.imports_dim)], dim=-1)
        else:
            head_in = wf
        win_logits = self.mlp_head(head_in)                      # (B, M, C)

        if self.mil == "attention":
            s = self.mil_score(wf).squeeze(-1).masked_fill(~win_mask, float("-inf"))
            w = torch.softmax(s, dim=1).unsqueeze(-1)
            file_logits = (win_logits * w).sum(dim=1)
        else:
            file_logits = masked_mean(win_logits, win_mask, dim=1)
        return file_logits, win_logits


# ---------------------------------------------------------------------------
# masked-token pretraining head
# ---------------------------------------------------------------------------
class MaskedTokenModel(nn.Module):
    """Encoder + a per-token decoder, for the unsupervised pass.

    The stem collapses 16 tokens into one block embedding, so the decoder
    expands each block embedding back into 16 per-token vectors (one learned
    slice per within-block offset) and scores only the positions that were
    masked. Predicting every token would materialise a (B, W, V) logit tensor
    for no benefit.
    """

    def __init__(self, vocab_size: int, window: int = 4096, tok_dim: int = 64,
                 **enc):
        super().__init__()
        self.encoder = SeqEncoder(vocab_size, window=window, **enc)
        self.window = window
        self.tok_dim = tok_dim
        self.expand = nn.Linear(self.encoder.dim, STEM_STRIDE * tok_dim)
        self.norm = nn.LayerNorm(tok_dim)
        self.out = nn.Linear(tok_dim, vocab_size)

    def forward(self, ids, tok_mask, positions):
        """positions (B, P) token indices to score -> logits (B, P, V)."""
        x, _ = self.encoder(ids, tok_mask)                    # (B, L, dim)
        B, L, _ = x.shape
        t = self.expand(x).view(B, L * STEM_STRIDE, self.tok_dim)
        g = positions.unsqueeze(-1).expand(-1, -1, self.tok_dim)
        return self.out(self.norm(t.gather(1, g)))


# ---------------------------------------------------------------------------
def transfer_pretrained(model: SeqTransformer, ckpt: dict,
                        tokens: list[str]) -> dict:
    """Load a pretraining checkpoint into a fine-tuning model.

    The pretrained encoder was fitted with the corpus-wide vocabulary; the
    fine-tuning model's vocabulary was fitted on the training folds and is a
    (re-indexed) subset of it, so embedding rows are matched BY MNEMONIC
    STRING and every other parameter is loaded by name. Rows for mnemonics the
    training folds never produced are simply not transferred - those ids do
    not exist in this model.
    """
    sd = {k[len("encoder."):]: v for k, v in ckpt["state_dict"].items()
          if k.startswith("encoder.")}
    src_tokens = ckpt["tokens"]
    src_emb = sd.pop("embed.weight")
    tgt = model.encoder.state_dict()
    loadable = {k: v for k, v in sd.items()
                if k in tgt and tgt[k].shape == v.shape}
    skipped = sorted(set(sd) - set(loadable))
    emb = model.encoder.embed.weight.data
    src_index = {t: i for i, t in enumerate(src_tokens)}
    hit = 0
    for j, t in enumerate(tokens):
        i = src_index.get(t)
        if i is not None and src_emb.shape[1] == emb.shape[1]:
            emb[j] = src_emb[i]
            hit += 1
    model.encoder.load_state_dict(loadable, strict=False)
    return {"encoder_tensors_loaded": len(loadable),
            "encoder_tensors_skipped": skipped,
            "embedding_rows_transferred": hit,
            "embedding_rows_total": len(tokens)}
