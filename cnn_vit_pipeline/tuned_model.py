#!/usr/bin/env python3
"""A parameterised copy of model_train.HierarchicalMalwareNet.

`CNN-ViT/model_train.py` stays exactly as the teammate's branch has it, so
`train_eval.py` keeps importing the original and the baseline numbers in
`results/cnn_vit/` remain reproducible. This file is a deliberate copy of the
same architecture with the sizes that were hard-coded turned into arguments:

    width       ConvStem channels (w, 2w) then PatchMerging 2w -> 4w -> 8w.
                w = 32 gives the original 32/64/128/256, i.e. token dim 256.
    depth       number of TransformerBlocks (original 6)
    heads       attention heads (original 8)
    mlp_ratio   MLP hidden = mlp_ratio * dim (original 512/256 = 2)
    dropout     attention/MLP/pos dropout (train_eval used 0.15)
    drop_path   stochastic depth, linearly ramped over the blocks (original 0)

Everything else - the SE blocks, the residual patch merging, the relative
position bias, the masked attention and the mask-weighted global average pool
- is identical, line for line, to model_train.py. `assert_matches_original()`
checks at runtime that the default configuration produces the same parameter
shapes as the original class, so a future edit to model_train.py that changes
the architecture is caught rather than silently diverging.

The one fix relative to the original: a sample whose ViT mask is entirely zero
would make `masked_fill(-inf)` produce an all -inf softmax row and therefore
NaNs. The encoder never emits such a mask (every file has at least one
instruction), but the augmentation could in principle shift one off the
canvas, so the mask is clamped to keep at least the first patch alive.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
            nn.Linear(channels, hidden, bias=False), nn.GELU(),
            nn.Linear(hidden, channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        B, C, _, _ = x.shape
        return x * self.fc(x).view(B, C, 1, 1)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        r = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        r.floor_()
        return x.div(keep) * r


class ConvStem(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(width), nn.GELU(),
            nn.Conv2d(width, 2 * width, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2 * width), nn.GELU(),
            SqueezeExcitation(2 * width))

    def forward(self, x):
        return self.stem(x)


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.reduction = nn.Conv2d(in_channels, out_channels, 2, stride=2)
        self.norm = nn.GroupNorm(num_groups=min(32, out_channels),
                                 num_channels=out_channels)
        self.act = nn.GELU()
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.se = SqueezeExcitation(out_channels)

    def forward(self, x):
        res = self.shortcut(x)
        x = self.act(self.norm(self.reduction(x)))
        return self.se(x + res)


class RobustRelativeAttention(nn.Module):
    def __init__(self, dim, num_patches=256, heads=8, dropout=0.25):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.num_tokens = num_patches
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.num_tokens - 1), heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        coords = torch.arange(self.num_tokens)
        rel = coords[:, None] - coords[None, :] + (self.num_tokens - 1)
        self.register_buffer("relative_position_index", rel)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, C // self.heads) \
                            .permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].reshape(
            self.num_tokens, self.num_tokens, self.heads)
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2),
                                    torch.finfo(attn.dtype).min)
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.to_out(out))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_patches=256, heads=8, mlp_dim=512,
                 dropout=0.25, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RobustRelativeAttention(dim, num_patches, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, key_padding_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask=key_padding_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TunableMalwareNet(nn.Module):
    """model_train.HierarchicalMalwareNet with its sizes exposed."""

    def __init__(self, num_classes=2, dropout=0.15, drop_path_rate=0.0,
                 width=32, depth=6, heads=8, mlp_ratio=2.0):
        super().__init__()
        dim = 8 * width
        if dim % heads:
            raise ValueError(f"width*8={dim} not divisible by heads={heads}")
        self.dim = dim
        self.stage1_cnn = ConvStem(width)
        self.stage2_downsample = PatchMerging(2 * width, 4 * width)
        self.stage3_downsample = PatchMerging(4 * width, 8 * width)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # unused, as upstream
        self.pos_drop = nn.Dropout(p=dropout)
        dpr = torch.linspace(0, drop_path_rate, max(depth, 1)).tolist()
        self.vit_layers = nn.ModuleList([
            TransformerBlock(dim=dim, num_patches=256, heads=heads,
                             mlp_dim=int(mlp_ratio * dim), dropout=dropout,
                             drop_path=dpr[i]) for i in range(depth)])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * dim, num_classes))

    def forward(self, x, raw_patch_masks=None):
        x = self.stage3_downsample(self.stage2_downsample(self.stage1_cnn(x)))
        x = x.flatten(2).transpose(1, 2)               # (B, 256, dim)

        pad_mask, flat_masks = None, None
        if raw_patch_masks is not None:
            flat_masks = raw_patch_masks.flatten(start_dim=1).to(x.dtype)
            # keep at least one live patch so softmax never sees an all-min row
            dead = flat_masks.sum(dim=1) == 0
            if dead.any():
                flat_masks = flat_masks.clone()
                flat_masks[dead, 0] = 1.0
            pad_mask = flat_masks == 0

        x = self.pos_drop(x)
        for layer in self.vit_layers:
            x = layer(x, key_padding_mask=pad_mask)

        if flat_masks is not None:
            counts = flat_masks.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (x * flat_masks.unsqueeze(-1)).sum(dim=1) / counts
        else:
            pooled = x.mean(dim=1)
        return self.mlp_head(pooled)


def assert_matches_original(verbose: bool = True) -> bool:
    """Default config == model_train.HierarchicalMalwareNet, shape for shape."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for p in (root / "CNN-ViT", root):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        import torchvision  # noqa: F401
    except ImportError:
        from cnn_vit_pipeline.train_eval import _install_torchvision_shim
        _install_torchvision_shim()
    import model_train as mt

    a = mt.HierarchicalMalwareNet(num_classes=2, dropout=0.15)
    b = TunableMalwareNet(num_classes=2, dropout=0.15)
    sa = {k: tuple(v.shape) for k, v in a.state_dict().items()}
    sb = {k: tuple(v.shape) for k, v in b.state_dict().items()}
    same = sa == sb
    if verbose:
        if same:
            print(f"tuned_model default == model_train.HierarchicalMalwareNet "
                  f"({sum(p.numel() for p in a.parameters()):,} parameters, "
                  f"{len(sa)} tensors)")
        else:
            only_a = sorted(set(sa) - set(sb))
            only_b = sorted(set(sb) - set(sa))
            diff = [k for k in set(sa) & set(sb) if sa[k] != sb[k]]
            print(f"MISMATCH  only-original={only_a}  only-tuned={only_b}  "
                  f"different-shape={[(k, sa[k], sb[k]) for k in diff]}")
    return same


if __name__ == "__main__":
    raise SystemExit(0 if assert_matches_original() else 1)
