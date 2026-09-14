#!/usr/bin/env python3
"""Configurable CNN-ViT trainer, group-CV hyper-parameter search, final eval.

`train_eval.py` runs the *original* protocol and imports the *original* model
from `CNN-ViT/model_train.py`; it is what produced `results/cnn_vit/`. This
module is the tuning harness. It uses `tuned_model.TunableMalwareNet` (proved
shape-identical to the original at its default settings by
`tuned_model.assert_matches_original`) and `encode.py`'s token cache, so every
knob the brief asks about - optimisation, regularisation, capacity, loss,
sampling, input encoding, decision threshold, test-time augmentation - is a
command-line argument.

Protocol
--------
* Selection happens on the TRAIN split only. `search` runs k-fold *group*
  cross-validation over it: whole ransomware families and whole goodware
  projects are held out together, and inside each CV-train a further
  group-aware inner val fold is carved for early stopping, so the scored fold
  is never used to pick a checkpoint. Configurations are ranked by
  out-of-fold macro-F1.
* `final` then trains once per seed on the whole train split - early stopping
  on the same fixed group-aware val fold `cohort.add_val_fold` gives
  `train_eval.py` - and scores the test fold exactly once per seed.
* The decision threshold, when `--threshold` is not `argmax`, is fitted on the
  cross-validated out-of-fold train scores of that same seed. Test scores are
  never consulted while choosing anything.

Speed
-----
The whole corpus is ~160 MB of uint8 canvases, so it is decoded once into a
GPU-resident tensor and batched by index; there is no DataLoader and no PNG
decode in the training loop. That takes an epoch from ~4 s to a fraction of a
second and makes a few hundred configurations affordable.

    python cnn_vit_pipeline/tuned_train.py baseline --dataset mendeley
    python cnn_vit_pipeline/tuned_train.py search   --dataset mendeley --grid quick
    python cnn_vit_pipeline/tuned_train.py final    --dataset mendeley \\
        --variant mnem1 --lr 5e-4 --seeds 1,2,3,4,5
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT / "CNN-ViT", REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cnn_vit_pipeline import cohort as C           # noqa: E402
from cnn_vit_pipeline import encode as E           # noqa: E402
from cnn_vit_pipeline.tuned_model import TunableMalwareNet  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "cnn_vit" / "tuned"
MODEL_DIR = Path(os.environ.get("RANSOM_CNN_VIT_MODELS",
                                REPO_ROOT.parent / "cnn_vit_models")) / "tuned"

#: which cached asm trees make up each dataset, and the label they contribute
DATASET_TREES = {
    "mendeley": [("unified_mendeley", None)],
    "balanced": [("unified_goodware_balanced", 0), ("unified_mendeley", 1)],
}


# ---------------------------------------------------------------- config ---
@dataclass
class Config:
    dataset: str = "mendeley"
    # ---- input encoding
    variant: str = "head3"          # head3 stride3 mnem1 mnem1s crop3
    row_align: bool = False
    crops: int = 1                  # >1 only meaningful with variant=crop3
    tta: bool = False               # average the crop scores at eval time
    # ---- optimisation
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 16
    epochs: int = 80
    patience: int = 8
    warmup_epochs: int = 5
    schedule: str = "cosine"        # cosine onecycle plateau const
    grad_clip: float = 1.0
    label_smoothing: float = 0.15
    loss: str = "ce"                # ce weighted_ce focal
    focal_gamma: float = 2.0
    sampler: str = "class"          # natural class cell
    amp: str = "bf16"               # off bf16
    monitor: str = "val_loss"       # val_loss val_macro_f1
    # ---- capacity / regularisation
    dropout: float = 0.15
    drop_path: float = 0.0
    width: int = 32
    depth: int = 6
    heads: int = 8
    mlp_ratio: float = 2.0
    # ---- augmentation
    aug_p: float = 0.4
    aug_max_shift: float = 0.20
    # ---- decision rule
    threshold: str = "argmax"       # argmax oof_macro_f1 oof_bal_acc
    # ---- bookkeeping
    seed: int = 1337
    device: str = "cuda"
    folds: int = 3

    def tag(self) -> str:
        return (f"{self.dataset}-{self.variant}{'ra' if self.row_align else ''}"
                f"-lr{self.lr:g}-wd{self.weight_decay:g}-bs{self.batch_size}"
                f"-{self.schedule}-{self.loss}-{self.sampler}"
                f"-do{self.dropout:g}-dp{self.drop_path:g}"
                f"-w{self.width}d{self.depth}h{self.heads}")


def config_from_args(a, **over) -> Config:
    known = {f.name for f in fields(Config)}
    kw = {k: v for k, v in vars(a).items() if k in known and v is not None}
    kw.update(over)
    return Config(**kw)


# ------------------------------------------------------------------ data ---
_BUNDLE_CACHE: dict = {}


def _encoded_arrays(tree: str, variant: str, row_align: bool, crops: int):
    """Canvases for one asm tree, memo-ised on disk under the token cache."""
    key = f"enc_{variant}{'_ra' if row_align else ''}{'' if crops == 1 else f'_k{crops}'}"
    d = E.CACHE_ROOT / tree
    img_p, msk_p = d / f"{key}_img.npy", d / f"{key}_mask.npy"
    idx, _ = E.load_cache(tree)
    if img_p.exists() and msk_p.exists():
        return idx, np.load(img_p, mmap_mode="r"), np.load(msk_p, mmap_mode="r")
    t0 = time.time()
    idx, imgs, msks = E.encode_tree(tree, variant, row_align, crops=crops)
    np.save(img_p, imgs)
    np.save(msk_p, msks)
    print(f"  encoded {tree} as {key} in {time.time()-t0:.1f}s -> {img_p.name}")
    return idx, imgs, msks


class Bundle:
    """Everything one dataset needs, resident on the training device."""

    def __init__(self, cfg: Config):
        import torch
        self.device = torch.device(cfg.device)
        crops = cfg.crops if cfg.variant == "crop3" else 1
        self.crops = crops

        frames, img_parts, msk_parts = [], [], []
        for tree, label_filter in DATASET_TREES[cfg.dataset]:
            idx, imgs, msks = _encoded_arrays(tree, cfg.variant, cfg.row_align,
                                              crops)
            keep = np.ones(len(idx), dtype=bool) if label_filter is None \
                else (idx["label"].to_numpy() == label_filter)
            f = idx.loc[keep].copy()
            f["tree"] = tree
            frames.append(f)
            img_parts.append(np.asarray(imgs[keep]))
            msk_parts.append(np.asarray(msks[keep]))
        meta = pd.concat(frames, ignore_index=True)
        images = np.concatenate(img_parts, 0)
        masks = np.concatenate(msk_parts, 0)

        # ---- join to the shared cohort split, exactly as build_dataset does
        split = C.load_split(cfg.dataset)
        meta["sha256"] = meta["sha256"].str.strip().str.lower()
        dup = meta["sha256"].duplicated(keep="first")
        meta, images, masks = meta[~dup], images[~dup.to_numpy()], masks[~dup.to_numpy()]
        meta = meta.reset_index(drop=True)
        pos = {s: i for i, s in enumerate(meta["sha256"])}

        split = split[split["sha256"].isin(pos)].copy()
        take = split["sha256"].map(pos).to_numpy()
        joined = split.reset_index(drop=True)
        joined["tree_label"] = meta["label"].to_numpy()[take]
        bad = joined["tree_label"] != joined["label"]
        if bad.any():
            raise AssertionError(f"{int(bad.sum())} tree/cohort label mismatches")
        joined["asm_id"] = meta["asm_id"].to_numpy()[take]
        joined = C.add_val_fold(joined)

        self.meta = joined.reset_index(drop=True)
        self.images = torch.from_numpy(np.ascontiguousarray(images[take])).to(self.device)
        self.masks = torch.from_numpy(np.ascontiguousarray(masks[take])).to(self.device)
        self.labels = torch.from_numpy(self.meta["label"].to_numpy()).long().to(self.device)
        self.n = len(self.meta)

    def idx(self, fold: str) -> np.ndarray:
        return np.where(self.meta["fold"].to_numpy() == fold)[0]

    def split_idx(self, split: str) -> np.ndarray:
        return np.where(self.meta["split"].to_numpy() == split)[0]

    def summary(self) -> str:
        g = self.meta.groupby(["fold", "label"]).size().to_dict()
        return "  ".join(f"{k[0]}/{'good' if k[1]==0 else 'rans'}={v}"
                         for k, v in sorted(g.items()))


def get_bundle(cfg: Config) -> Bundle:
    key = (cfg.dataset, cfg.variant, cfg.row_align,
           cfg.crops if cfg.variant == "crop3" else 1, cfg.device)
    if key not in _BUNDLE_CACHE:
        _BUNDLE_CACHE.clear()          # one dataset/encoding resident at a time
        _BUNDLE_CACHE[key] = Bundle(cfg)
        print(f"  bundle {key[:4]}: {_BUNDLE_CACHE[key].n} samples  "
              f"{_BUNDLE_CACHE[key].summary()}")
    return _BUNDLE_CACHE[key]


# ---------------------------------------------------------- augmentation ---
def structural_shift(imgs, masks, p: float, max_ratio: float, gen):
    """model_train.MaskAwareStructuralShift, vectorised over the batch.

    Inserts a band of zero rows at a random height and pushes the rest of the
    canvas down, truncating at the bottom, updating the 16x16 patch mask the
    same way. Identical semantics to the original per-sample transform; the
    per-sample random draws are just made as one batched tensor.
    """
    import torch
    B, H, W = imgs.shape
    P = masks.shape[-1]
    dev = imgs.device
    hit = torch.rand(B, device=dev, generator=gen) <= p
    if not hit.any():
        return imgs, masks

    ratio = 0.05 + (max_ratio - 0.05) * torch.rand(B, device=dev, generator=gen)
    shift = (H * ratio).to(torch.long)
    lo, hi = int(H * 0.1), int(H * 0.8)
    insert = (lo + torch.rand(B, device=dev, generator=gen) * (hi - lo)).to(torch.long)
    shift = torch.where(hit, shift, torch.zeros_like(shift))

    rows = torch.arange(H, device=dev).unsqueeze(0).expand(B, H)
    src = rows - shift.unsqueeze(1)
    zero = (rows >= insert.unsqueeze(1)) & (src < insert.unsqueeze(1))
    src = torch.where(rows < insert.unsqueeze(1), rows, src).clamp(0, H - 1)
    out = torch.gather(imgs, 1, src.unsqueeze(-1).expand(B, H, W))
    out = out.masked_fill(zero.unsqueeze(-1), 0)

    pshift = torch.ceil(shift.float() / (H // P)).to(torch.long)
    pinsert = insert // (H // P)
    prows = torch.arange(P, device=dev).unsqueeze(0).expand(B, P)
    psrc = prows - pshift.unsqueeze(1)
    pzero = (prows >= pinsert.unsqueeze(1)) & (psrc < pinsert.unsqueeze(1))
    psrc = torch.where(prows < pinsert.unsqueeze(1), prows, psrc).clamp(0, P - 1)
    mout = torch.gather(masks, 1, psrc.unsqueeze(-1).expand(B, P, P))
    mout = mout.masked_fill(pzero.unsqueeze(-1), 0)
    return out, mout


# ----------------------------------------------------------------- losses ---
def make_criterion(cfg: Config, counts):
    import torch
    import torch.nn as nn
    if cfg.loss == "weighted_ce":
        w = torch.tensor(counts.sum() / (2.0 * np.clip(counts, 1, None)),
                         dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=w.to(cfg.device),
                                   label_smoothing=cfg.label_smoothing)
    if cfg.loss == "focal":
        gamma, ls = cfg.focal_gamma, cfg.label_smoothing
        alpha = torch.tensor(counts.sum() / (2.0 * np.clip(counts, 1, None)),
                             dtype=torch.float32).to(cfg.device)

        def focal(logits, target):
            logp = torch.log_softmax(logits.float(), dim=1)
            pt = logp.gather(1, target[:, None]).squeeze(1).exp()
            ce = nn.functional.cross_entropy(logits.float(), target,
                                             label_smoothing=ls,
                                             reduction="none")
            return (alpha[target] * (1 - pt).pow(gamma) * ce).mean()
        return focal
    import torch.nn as nn
    return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)


def sample_weights(cfg: Config, bundle: Bundle, idx: np.ndarray) -> np.ndarray:
    labels = bundle.meta["label"].to_numpy()[idx]
    if cfg.sampler == "natural":
        return np.ones(len(idx))
    if cfg.sampler == "class":
        cnt = np.bincount(labels, minlength=2).clip(1)
        return 1.0 / cnt[labels]
    # cell: balance the (label, arch) cells so bitness cannot stand in for class
    arch = bundle.meta["arch"].to_numpy()[idx]
    key = pd.Series([f"{l}|{a}" for l, a in zip(labels, arch)])
    cnt = key.value_counts().to_dict()
    return np.array([1.0 / cnt[k] for k in key])


# --------------------------------------------------------------- training ---
def seed_all(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _amp_ctx(cfg):
    import torch
    from contextlib import nullcontext
    if cfg.amp == "bf16" and cfg.device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def evaluate(model, bundle, idx, cfg, batch=128, crops_mean=True):
    """Probability of class 1 for every index, and the mean loss."""
    import torch
    model.eval()
    K = bundle.crops
    scores = np.zeros(len(idx), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            t = torch.as_tensor(b, device=bundle.device)
            reps = range(K) if (K > 1 and crops_mean and cfg.tta) else [0]
            acc = None
            for k in reps:
                im = (bundle.images[t, k] if K > 1 else bundle.images[t]).float().div(255.)
                mk = (bundle.masks[t, k] if K > 1 else bundle.masks[t]).float()
                with _amp_ctx(cfg):
                    out = model(im.unsqueeze(1), raw_patch_masks=mk)
                p = torch.softmax(out.float(), 1)[:, 1]
                acc = p if acc is None else acc + p
            scores[s:s + len(b)] = (acc / len(list(reps))).cpu().numpy()
    return scores


def eval_loss(model, bundle, idx, criterion, cfg, batch=128):
    import torch
    model.eval()
    tot, n = 0.0, 0
    K = bundle.crops
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            t = torch.as_tensor(b, device=bundle.device)
            im = (bundle.images[t, 0] if K > 1 else bundle.images[t]).float().div(255.)
            mk = (bundle.masks[t, 0] if K > 1 else bundle.masks[t]).float()
            y = bundle.labels[t]
            with _amp_ctx(cfg):
                out = model(im.unsqueeze(1), raw_patch_masks=mk)
            tot += float(criterion(out.float(), y)) * len(b)
            n += len(b)
    return tot / max(n, 1)


def fit(cfg: Config, bundle: Bundle, train_idx: np.ndarray, val_idx: np.ndarray,
        seed: int, ckpt: Path | None = None, verbose: bool = False):
    """Train one model. Returns (model, history, epoch_times)."""
    import torch
    seed_all(seed)
    gen = torch.Generator(device=bundle.device)
    gen.manual_seed(seed)

    model = TunableMalwareNet(num_classes=2, dropout=cfg.dropout,
                              drop_path_rate=cfg.drop_path, width=cfg.width,
                              depth=cfg.depth, heads=cfg.heads,
                              mlp_ratio=cfg.mlp_ratio).to(bundle.device)
    counts = np.bincount(bundle.meta["label"].to_numpy()[train_idx], minlength=2)
    criterion = make_criterion(cfg, counts)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    w = sample_weights(cfg, bundle, train_idx)
    wt = torch.as_tensor(w / w.sum(), dtype=torch.float32, device=bundle.device)
    n_steps = max(1, math.ceil(len(train_idx) / cfg.batch_size))

    sched = None
    if cfg.schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, cfg.epochs - cfg.warmup_epochs), eta_min=1e-6)
    elif cfg.schedule == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg.lr, total_steps=cfg.epochs * n_steps,
            pct_start=0.25, div_factor=10.0, final_div_factor=100.0)
    elif cfg.schedule == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=max(2, cfg.patience // 3))

    tidx = torch.as_tensor(train_idx, device=bundle.device)
    best, bad, history, times = math.inf, 0, [], []
    best_state = None
    K = bundle.crops

    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        if cfg.schedule in ("cosine", "const", "plateau") and epoch < cfg.warmup_epochs:
            for g in opt.param_groups:
                g["lr"] = cfg.lr * (epoch + 1) / max(cfg.warmup_epochs, 1)
        cur_lr = opt.param_groups[0]["lr"]

        pick = torch.multinomial(wt, len(train_idx), replacement=True,
                                 generator=gen)
        order = tidx[pick]
        running = 0.0
        for s in range(0, len(order), cfg.batch_size):
            t = order[s:s + cfg.batch_size]
            if K > 1:
                k = torch.randint(0, K, (len(t),), device=bundle.device,
                                  generator=gen)
                im = bundle.images[t, k].float().div(255.)
                mk = bundle.masks[t, k].float()
            else:
                im = bundle.images[t].float().div(255.)
                mk = bundle.masks[t].float()
            if cfg.aug_p > 0:
                im, mk = structural_shift(im, mk, cfg.aug_p,
                                          cfg.aug_max_shift, gen)
            y = bundle.labels[t]
            opt.zero_grad(set_to_none=True)
            with _amp_ctx(cfg):
                out = model(im.unsqueeze(1), raw_patch_masks=mk)
                loss = criterion(out.float(), y)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if cfg.schedule == "onecycle":
                sched.step()
            running += float(loss) * len(t)

        if cfg.schedule == "cosine" and epoch >= cfg.warmup_epochs:
            sched.step()

        vloss = eval_loss(model, bundle, val_idx, criterion, cfg)
        vscore = evaluate(model, bundle, val_idx, cfg)
        vy = bundle.meta["label"].to_numpy()[val_idx]
        vf1 = macro_f1(vy, (vscore >= 0.5).astype(int))
        if cfg.schedule == "plateau" and epoch >= cfg.warmup_epochs:
            sched.step(vloss)

        metric = vloss if cfg.monitor == "val_loss" else -vf1
        dt = time.time() - t0
        times.append(dt)
        history.append({"epoch": epoch + 1, "lr": cur_lr,
                        "train_loss": running / max(len(order), 1),
                        "val_loss": vloss, "val_macro_f1": vf1,
                        "seconds": round(dt, 3)})
        if verbose:
            print(f"    ep{epoch+1:03d} lr {cur_lr:.2e} train "
                  f"{history[-1]['train_loss']:.4f} val {vloss:.4f} "
                  f"f1 {vf1:.3f} {dt:.2f}s")

        if metric < best - 1e-4:
            best, bad = metric, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch >= cfg.warmup_epochs:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if ckpt:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, ckpt)
    return model, history, times


# ------------------------------------------------------------- CV + search --
def macro_f1(y, p) -> float:
    y, p = np.asarray(y).astype(int), np.asarray(p).astype(int)
    f1s = []
    for c in (0, 1):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f1s))


def bal_acc(y, p) -> float:
    y, p = np.asarray(y).astype(int), np.asarray(p).astype(int)
    out = []
    for c in (0, 1):
        m = y == c
        if m.any():
            out.append(float((p[m] == c).mean()))
    return float(np.mean(out)) if out else 0.0


def safe_auc(y, s):
    y = np.asarray(y).astype(int)
    if len(set(y.tolist())) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def best_threshold(y, s, objective="macro_f1") -> float:
    """Threshold maximising the objective on out-of-fold train scores."""
    f = macro_f1 if objective == "macro_f1" else bal_acc
    cand = np.unique(np.round(np.concatenate([[0.05, 0.5, 0.95], s]), 4))
    best_t, best_v = 0.5, -1.0
    for t in cand:
        v = f(y, (s >= t).astype(int))
        if v > best_v:
            best_t, best_v = float(t), v
    return best_t


def cv_folds(bundle: Bundle, n_splits: int, seed: int):
    """Group-aware, label-stratified folds over the TRAIN split only."""
    from sklearn.model_selection import StratifiedGroupKFold
    tr = bundle.split_idx("train")
    y = bundle.meta["label"].to_numpy()[tr]
    g = bundle.meta["family_or_group"].to_numpy()[tr]
    sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fit_i, held_i in sgk.split(np.zeros(len(tr)), y, g):
        yield tr[fit_i], tr[held_i]


def inner_val(bundle: Bundle, fit_idx: np.ndarray, seed: int):
    """Carve a group-aware early-stopping fold out of a CV-train part."""
    sub = bundle.meta.iloc[fit_idx].copy()
    sub["split"] = "train"
    sub = C.add_val_fold(sub, seed=seed)
    is_val = (sub["fold"] == "val").to_numpy()
    if is_val.sum() < 8 or (~is_val).sum() < 8:   # degenerate, fall back
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(fit_idx))
        cut = max(8, int(0.1 * len(fit_idx)))
        return fit_idx[perm[cut:]], fit_idx[perm[:cut]]
    return fit_idx[~is_val], fit_idx[is_val]


def run_cv(cfg: Config, bundle: Bundle, seed: int, verbose: bool = False):
    """k-fold group CV on train. Returns (row dict, oof y, oof score, indices).

    The reported `cv_macro_f1` is always an *unbiased* estimate of the
    configuration's own decision rule:

      threshold=argmax    -> macro-F1 of (score >= 0.5)
      threshold=oof_*     -> nested: for each fold the threshold is fitted on
                             the out-of-fold scores of the OTHER folds only,
                             so the fold being scored never contributed to the
                             threshold that scores it.

    `threshold_used` is the threshold refitted on all out-of-fold scores; that
    is the one the final model deploys, and it is still train-only.
    """
    tr = bundle.split_idx("train")
    oof = np.full(bundle.n, np.nan)
    which = np.full(bundle.n, -1)
    times, epochs = [], []
    for fi, (fit_i, held_i) in enumerate(cv_folds(bundle, cfg.folds, seed)):
        in_tr, in_val = inner_val(bundle, fit_i, seed + fi)
        model, hist, t = fit(cfg, bundle, in_tr, in_val, seed + fi,
                             verbose=verbose)
        oof[held_i] = evaluate(model, bundle, held_i, cfg)
        which[held_i] = fi
        times.extend(t)
        epochs.append(len(hist))
        del model
    m = ~np.isnan(oof)
    y = bundle.meta["label"].to_numpy()[m]
    s = oof[m]
    f = which[m]

    obj = "bal_acc" if cfg.threshold == "oof_bal_acc" else "macro_f1"
    thr_all = best_threshold(y, s, obj)
    nested_pred = (s >= 0.5).astype(int)
    nested_thr = []
    for fi in sorted(set(f.tolist())):
        other = f != fi
        t_fi = best_threshold(y[other], s[other], obj) if other.any() else 0.5
        nested_thr.append(t_fi)
        nested_pred[f == fi] = (s[f == fi] >= t_fi).astype(int)

    pred = (s >= 0.5).astype(int) if cfg.threshold == "argmax" else nested_pred
    thr = 0.5 if cfg.threshold == "argmax" else thr_all
    row = {
        "cv_macro_f1": macro_f1(y, pred),
        "cv_balanced_accuracy": bal_acc(y, pred),
        "cv_accuracy": float((pred == y).mean()),
        "cv_recall_goodware": float((pred[y == 0] == 0).mean()) if (y == 0).any() else 0.0,
        "cv_recall_ransomware": float((pred[y == 1] == 1).mean()) if (y == 1).any() else 0.0,
        "cv_macro_f1_at_0.5": macro_f1(y, (s >= 0.5).astype(int)),
        "cv_macro_f1_nested_thr": macro_f1(y, nested_pred),
        "cv_balanced_accuracy_nested_thr": bal_acc(y, nested_pred),
        "cv_auc": safe_auc(y, s),
        "threshold_used": thr,
        "threshold_refit_all_oof": thr_all,
        "threshold_nested": ";".join(f"{t:.3f}" for t in nested_thr),
        "cv_n": int(m.sum()),
        "epochs_mean": float(np.mean(epochs)),
        "seconds_per_epoch": float(np.median(times)),
        "seconds_total": float(np.sum(times)),
    }
    return row, y, s, np.where(m)[0]


# ---------------------------------------------------------------- search ---
def grid(name: str, dataset: str) -> list[dict]:
    """Named search stages. Each entry is a dict of Config overrides."""
    out: list[dict] = []

    if name.startswith("file:"):
        # an explicit list of override dicts, for refinement stages
        return json.loads(Path(name[5:]).read_text(encoding="utf-8"))

    if name == "baseline":
        return [{}]

    if name == "encoding":
        # stage 1: which input encoding, at the original optimisation settings
        for v in ("head3", "stride3", "mnem1", "mnem1s"):
            out.append({"variant": v})
        out.append({"variant": "head3", "row_align": True})
        out.append({"variant": "crop3", "crops": E.CROP_K, "tta": True})
        return out

    if name == "optim":
        for lr in (1e-4, 6e-4, 1e-3, 3e-3):
            out.append({"lr": lr})
        for bs in (32, 64):
            out.append({"batch_size": bs})
        for wd in (1e-3, 1e-1):
            out.append({"weight_decay": wd})
        for sch in ("onecycle", "plateau"):
            out.append({"schedule": sch})
        out.append({"epochs": 150, "patience": 20})
        out.append({"monitor": "val_macro_f1"})
        for ls in (0.0, 0.3):
            out.append({"label_smoothing": ls})
        return out

    if name == "loss":
        out.append({"loss": "weighted_ce", "sampler": "natural"})
        out.append({"loss": "focal", "sampler": "natural", "focal_gamma": 2.0})
        out.append({"loss": "focal", "sampler": "class", "focal_gamma": 2.0})
        out.append({"sampler": "natural"})
        # `cell` balances the (label, architecture) cells so the model cannot
        # win on the balanced dataset by reading bitness instead of code
        out.append({"sampler": "cell"})
        return out

    if name == "decision":
        out.append({"threshold": "oof_macro_f1"})
        out.append({"threshold": "oof_bal_acc"})
        return out

    if name == "capacity":
        for do in (0.0, 0.3):
            out.append({"dropout": do})
        out.append({"drop_path": 0.1})
        for d in (3, 8):
            out.append({"depth": d})
        out.append({"width": 16})
        out.append({"width": 16, "depth": 3, "dropout": 0.3})
        out.append({"aug_p": 0.0})
        return out

    raise ValueError(f"unknown grid {name!r}")


def search(a) -> int:
    base = config_from_args(a)
    out_dir = Path(a.out or (RESULTS_DIR / base.dataset))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cv_search.csv"

    stages = a.grid.split(",")
    todo: list[tuple[str, Config]] = []
    seen = set()
    for st in stages:
        # a `file:` stage is labelled by the file's stem, so cv_search.csv's
        # stage column stays readable ("combine", not the whole path)
        label = Path(st[5:]).stem if st.startswith("file:") else st
        for over in grid(st, base.dataset):
            cfg = replace(base, **over)
            key = json.dumps(asdict(cfg), sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            todo.append((label, cfg))

    if a.resume and csv_path.exists():
        done = pd.read_csv(csv_path)
        keys = {f.name for f in fields(Config)} & set(done.columns)
        have = set()
        for _, r in done.iterrows():
            have.add(json.dumps({k: _coerce(k, r[k]) for k in sorted(keys)},
                                sort_keys=True, default=str))
        before = len(todo)
        todo = [(st, c) for st, c in todo
                if json.dumps({k: getattr(c, k) for k in sorted(keys)},
                              sort_keys=True, default=str) not in have]
        print(f"resume: {before - len(todo)} of {before} configurations are "
              f"already in {csv_path.name}")

    print(f"=== search {base.dataset}: {len(todo)} configurations, "
          f"{base.folds}-fold group CV, seed {base.seed} ===", flush=True)
    rows = []
    t_start = time.time()
    for i, (stage, cfg) in enumerate(todo, 1):
        bundle = get_bundle(cfg)
        t0 = time.time()
        row, _, _, _ = run_cv(cfg, bundle, cfg.seed)
        rec = {"stage": stage, "config_id": f"{stage}_{i:03d}", **asdict(cfg),
               **row, "wall_seconds": round(time.time() - t0, 1)}
        rows.append(rec)
        print(f"[{i:3d}/{len(todo)}] {stage:9s} cvF1 {row['cv_macro_f1']:.4f} "
              f"bal {row['cv_balanced_accuracy']:.4f} auc {row['cv_auc']:.4f} "
              f"ep~{row['epochs_mean']:.0f} {time.time()-t0:.0f}s  "
              f"{cfg.tag()}", flush=True)
        _append_csv(csv_path, rec)
    print(f"\nsearch done in {(time.time()-t_start)/60:.1f} min -> {csv_path}")

    df = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False)
    print(df[["stage", "cv_macro_f1", "cv_balanced_accuracy", "cv_auc",
              "variant", "lr", "batch_size", "schedule", "loss", "sampler",
              "dropout", "depth", "width"]].head(10).to_string(index=False))
    return 0


def _coerce(name: str, value):
    """Read a cv_search.csv cell back into the Config field's own type."""
    t = {f.name: f.type for f in fields(Config)}[name]
    t = t if isinstance(t, str) else getattr(t, "__name__", str(t))
    if t == "bool":
        return bool(value) if not isinstance(value, str) else value == "True"
    if t == "int":
        return int(value)
    if t == "float":
        return float(value)
    return str(value)


def _append_csv(path: Path, rec: dict):
    exists = path.exists()
    old = None
    if exists:
        old = pd.read_csv(path)
        if list(old.columns) != list(rec.keys()):
            old = pd.concat([old, pd.DataFrame([rec])], ignore_index=True)
            old.to_csv(path, index=False)
            return
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rec.keys()))
        if not exists:
            w.writeheader()
        w.writerow(rec)


# ----------------------------------------------------------------- final ---
def final(a) -> int:
    cfg = config_from_args(a)
    seeds = [int(s) for s in str(a.seeds).split(",")]
    out_dir = Path(a.out or (RESULTS_DIR / cfg.dataset))
    (out_dir / "seeds").mkdir(parents=True, exist_ok=True)
    bundle = get_bundle(cfg)
    dev_name = device_name(cfg.device)

    tr_idx = bundle.idx("train")
    va_idx = bundle.idx("val")
    te_idx = bundle.idx("test")
    arch = bundle.meta["arch"].to_numpy()[te_idx]
    family = bundle.meta["family"].to_numpy()[te_idx]
    ids = bundle.meta["asm_id"].to_numpy()[te_idx]
    y_true = bundle.meta["label"].to_numpy()[te_idx]

    print(f"=== final {cfg.dataset} variant={cfg.variant} seeds={seeds} ===")
    print(f"    train {len(tr_idx)}  val {len(va_idx)}  test {len(te_idx)}")

    t_start = time.time()
    results, preds = [], []
    for seed in seeds:
        t0 = time.time()
        thr, oof_row = 0.5, {}
        if cfg.threshold != "argmax":
            # threshold from THIS seed's out-of-fold train scores only
            oof_row, oy, os_, _ = run_cv(cfg, bundle, seed)
            thr = oof_row["threshold_used"]
        # --ckpt-tag keeps a post-hoc run's weights out of the chosen
        # configuration's checkpoint directory
        ck_dir = cfg.dataset + (f"-{a.ckpt_tag}" if getattr(a, "ckpt_tag", None)
                                else "")
        ck = MODEL_DIR / ck_dir / f"seed{seed}" / "best_model.pth"
        model, hist, times = fit(cfg, bundle, tr_idx, va_idx, seed, ckpt=ck,
                                 verbose=a.verbose)
        score = evaluate(model, bundle, te_idx, cfg)
        y_pred = (score >= thr).astype(int)

        res = C.build_result(
            y_true, y_pred, score, arch, family,
            model="TunableMalwareNet (CNN-ViT, tuned)",
            seed=seed, threshold=thr, threshold_rule=cfg.threshold,
            epochs_run=len(hist), epochs_requested=cfg.epochs,
            seconds_per_epoch_mean=round(float(np.mean(times)), 3),
            seconds_per_epoch_median=round(float(np.median(times)), 3),
            train_seconds_total=round(float(np.sum(times)), 2),
            device=cfg.device, device_name=dev_name,
            cv_oof=oof_row or None,
        )
        results.append(res)
        preds.append(pd.DataFrame({"seed": seed, "file": ids, "arch": arch,
                                   "family": family, "y_true": y_true,
                                   "y_pred": y_pred, "score_ransomware": score}))
        sd = out_dir / "seeds" / f"seed{seed}"
        sd.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(hist).to_csv(sd / "training_log.csv", index=False)
        preds[-1].to_csv(sd / "predictions.csv", index=False)
        (sd / "metrics.json").write_text(json.dumps(res, indent=2, default=str),
                                         encoding="utf-8")
        print(f"  seed {seed}: macroF1 {res['macro_f1']:.4f}  bal "
              f"{res['balanced_accuracy']:.4f}  auc {res['roc_auc']:.4f}  "
              f"thr {thr:.3f}  ep {len(hist)}  {time.time()-t0:.0f}s", flush=True)
        del model

    agg = aggregate(results)
    pd.concat(preds).to_csv(out_dir / "predictions.csv", index=False)

    meta_all = bundle.meta
    cfgd = config_doc(cfg, seeds, dev_name, bundle,
                      ckpt_dir=MODEL_DIR / ck_dir)
    (out_dir / "config_used.yaml").write_text(_yaml(cfgd), encoding="utf-8")

    C.write_metrics(
        out_dir / "metrics.json",
        experiment=f"cnn_vit_tuned_{cfg.dataset}",
        description=(
            "Tuned CNN-ViT (TunableMalwareNet, shape-identical to "
            "model_train.HierarchicalMalwareNet at default settings) on the "
            "shared cohort split. Hyper-parameters and input encoding were "
            f"chosen by {cfg.folds}-fold group cross-validation over the TRAIN "
            "split only (families and goodware projects held out whole, with a "
            "further group-aware inner val fold for early stopping); the test "
            "fold was scored once per seed after selection. Encoding variant: "
            f"{cfg.variant}."),
        samples=C.split_summary(meta_all, "fold"),
        results=results,
        elapsed_seconds=time.time() - t_start,
        dataset=cfg.dataset,
        # top-level convenience fields: the headline number and its spread,
        # so a reader (or the comparison table) never has to average
        # `results` by hand. They are exactly seed_summary["macro_f1"].
        macro_f1=agg["macro_f1"]["mean"],
        macro_f1_sd=agg["macro_f1"]["sd"],
        n_seeds=len(seeds),
        seeds=list(seeds),
        majority_class_accuracy=results[0]["majority_class_accuracy"],
        seed_summary=agg,
        config=cfgd,
        predictions=("predictions.csv holds every seed; the per-seed copies "
                     "are under seeds/seed<N>/predictions.csv"),
    )
    print("\nseed summary (test fold, mean +/- sd over "
          f"{len(seeds)} seeds):")
    for k in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc",
              "recall_goodware", "recall_ransomware"):
        print(f"  {k:<20} {agg[k]['mean']:.4f} +/- {agg[k]['sd']:.4f}")
    print(f"wrote {out_dir/'metrics.json'}")
    return 0


def aggregate(results: list[dict]) -> dict:
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "macro_precision",
            "macro_recall", "roc_auc", "recall_goodware", "recall_ransomware",
            "precision_goodware", "precision_ransomware", "f1_goodware",
            "f1_ransomware", "false_positive_rate", "majority_class_accuracy"]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in results if r.get(k) is not None], dtype=float)
        if not len(v):
            continue
        out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                  "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}
    per_arch = {}
    for a in results[0].get("per_architecture", {}):
        per_arch[a] = {}
        for k in ("accuracy", "balanced_accuracy", "macro_f1",
                  "recall_goodware", "recall_ransomware"):
            v = np.array([r["per_architecture"][a][k] for r in results], dtype=float)
            per_arch[a][k] = {"mean": float(v.mean()),
                              "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
        per_arch[a]["n"] = results[0]["per_architecture"][a]["n"]
    out["per_architecture"] = per_arch
    fam = {}
    for f in results[0].get("per_family_recall", {}):
        v = np.array([r["per_family_recall"][f]["recall"] for r in results])
        fam[f] = {"mean": float(v.mean()),
                  "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                  "support": results[0]["per_family_recall"][f]["support"]}
    out["per_family_recall"] = fam
    return out


def device_name(device: str) -> str:
    import torch
    if device.startswith("cuda") and torch.cuda.is_available():
        i = int(device.split(":")[1]) if ":" in device else 0
        cap = torch.cuda.get_device_capability(i)
        return (f"{torch.cuda.get_device_name(i)} (sm_{cap[0]}{cap[1]}), "
                f"torch {torch.__version__}")
    import platform
    return f"CPU {platform.processor() or platform.machine()}, torch {torch.__version__}"


def config_doc(cfg: Config, seeds, dev_name, bundle, ckpt_dir=None) -> dict:
    d = asdict(cfg)
    return {
        "pipeline": "cnn_vit_tuned",
        "dataset": cfg.dataset,
        "seeds": list(seeds),
        "device_name": dev_name,
        "model": {"class": "TunableMalwareNet (cnn_vit_pipeline/tuned_model.py)",
                  "identical_to_original_at_defaults": True,
                  "width": cfg.width, "depth": cfg.depth, "heads": cfg.heads,
                  "mlp_ratio": cfg.mlp_ratio, "dropout": cfg.dropout,
                  "drop_path": cfg.drop_path,
                  "token_dim": 8 * cfg.width},
        "encoding": {"variant": cfg.variant, "row_align": cfg.row_align,
                     "crops": cfg.crops, "test_time_augmentation": cfg.tta,
                     "described_in": "cnn_vit_pipeline/encode.py"},
        "training": {k: d[k] for k in
                     ("lr", "weight_decay", "batch_size", "epochs", "patience",
                      "warmup_epochs", "schedule", "grad_clip",
                      "label_smoothing", "loss", "focal_gamma", "sampler",
                      "amp", "monitor", "aug_p", "aug_max_shift")},
        "decision_rule": {"threshold": cfg.threshold,
                          "fitted_on": "out-of-fold TRAIN scores from the same "
                                       "group CV, never on test"},
        "split": {"source": "cnn_vit_pipeline/cohort.py",
                  "val_fraction": C.VAL_FRACTION, "val_seed": C.VAL_SEED,
                  "cv_folds": cfg.folds,
                  "cv": "StratifiedGroupKFold over the train split"},
        "paths": {"checkpoints": str(ckpt_dir or (MODEL_DIR / cfg.dataset)),
                  "token_cache": str(E.CACHE_ROOT)},
    }


def _yaml(obj, indent=0):
    pad = "  " * indent
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                parts.append(f"{pad}{k}:\n{_yaml(v, indent+1)}")
            else:
                parts.append(f"{pad}{k}: {_scalar(v)}")
        return "\n".join(parts) + ("\n" if indent == 0 else "")
    if isinstance(obj, list):
        return "\n".join(f"{pad}- {_scalar(v)}" for v in obj)
    return f"{pad}{_scalar(obj)}"


def _scalar(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v)
    if s == "" or any(c in s for c in ":#{}[],&*?|<>=!%@`") or s != s.strip():
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return s


# ------------------------------------------------------------------- main ---
def baseline(a) -> int:
    """One CV run at the original settings - the number everything is judged against."""
    cfg = config_from_args(a)
    bundle = get_bundle(cfg)
    print(f"=== baseline CV {cfg.dataset} ({cfg.folds} folds) ===")
    row, y, s, _ = run_cv(cfg, bundle, cfg.seed, verbose=a.verbose)
    print(json.dumps(row, indent=2))
    return 0


def add_config_args(p):
    p.add_argument("--dataset", choices=C.DATASETS, default="mendeley")
    p.add_argument("--variant", choices=E.VARIANTS, default=None)
    p.add_argument("--row-align", action="store_true", default=None)
    p.add_argument("--crops", type=int, default=None)
    p.add_argument("--tta", action="store_true", default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--schedule", choices=["cosine", "onecycle", "plateau", "const"],
                   default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--label-smoothing", type=float, default=None)
    p.add_argument("--loss", choices=["ce", "weighted_ce", "focal"], default=None)
    p.add_argument("--focal-gamma", type=float, default=None)
    p.add_argument("--sampler", choices=["natural", "class", "cell"], default=None)
    p.add_argument("--amp", choices=["off", "bf16"], default=None)
    p.add_argument("--monitor", choices=["val_loss", "val_macro_f1"], default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--drop-path", type=float, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--mlp-ratio", type=float, default=None)
    p.add_argument("--aug-p", type=float, default=None)
    p.add_argument("--aug-max-shift", type=float, default=None)
    p.add_argument("--threshold", choices=["argmax", "oof_macro_f1", "oof_bal_acc"],
                   default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--verbose", action="store_true")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("baseline", "search", "final"):
        p = sub.add_parser(name)
        add_config_args(p)
        if name == "search":
            p.add_argument("--grid", default="encoding")
            p.add_argument("--resume", action="store_true",
                           help="skip configurations already in cv_search.csv")
        if name == "final":
            p.add_argument("--seeds", default="1,2,3,4,5")
            p.add_argument("--ckpt-tag", default=None,
                           help="suffix for the checkpoint directory, so a "
                                "post-hoc run does not overwrite the chosen "
                                "configuration's weights")
    a = ap.parse_args()
    if a.cmd == "baseline":
        return baseline(a)
    if a.cmd == "search":
        return search(a)
    return final(a)


if __name__ == "__main__":
    raise SystemExit(main())
