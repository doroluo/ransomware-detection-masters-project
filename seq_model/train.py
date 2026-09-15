#!/usr/bin/env python3
"""Datasets, samplers and the fine-tuning loop for the sequence model.

The recipe deliberately mirrors the untuned CNN-ViT recipe in
`cnn_vit_pipeline/train_eval.py` wherever the two are comparable - AdamW at
3e-4 with a short warmup and cosine decay, a weighted sampler, early stopping
on a group-aware val fold, argmax at 0.5, no threshold moving - so that the
family-holdout comparison between the two isolates the representation rather
than the optimiser. What differs is listed in `seq_model/config.yaml`.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from seq_model import data as D                       # noqa: E402


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
class FileWindowDataset(Dataset):
    """One item = one FILE, as up to N windows of W tokens (a MIL bag)."""

    def __init__(self, shas, labels, vocab: D.Vocab, window: int, n_windows: int,
                 cache_root=D.CACHE_ROOT, imports=None, imports_dim: int = 0):
        self.shas = list(shas)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.vocab = vocab
        self.W = int(window)
        self.N = int(n_windows)
        self.cache_root = Path(cache_root)
        self.imports = imports or {}
        self.imports_dim = int(imports_dim)

    def __len__(self):
        return len(self.shas)

    def __getitem__(self, i):
        sha = self.shas[i]
        w, lens = D.file_windows(sha, self.vocab, self.W, self.N, self.cache_root)
        item = {"ids": torch.from_numpy(w), "lens": torch.from_numpy(lens),
                "y": int(self.labels[i]), "index": i}
        if self.imports_dim:
            v = self.imports.get(sha)
            item["imports"] = torch.from_numpy(
                np.zeros(self.imports_dim, dtype=np.float32) if v is None
                else np.asarray(v, dtype=np.float32))
        return item


class Collate:
    """Pad every batch to the SAME (batch, n_windows, window) shape.

    Not cosmetic. Files have between 1 and N windows, so a collate that pads to
    the batch's own maximum emits a different tensor shape almost every step,
    and PyTorch's pinned-memory allocator caches a fresh block per shape and
    never releases it - the host working set walked past 11 GB and the machine
    started paging, which showed up as a 20x slowdown halfway through a sweep.
    A constant shape means one cached block, reused forever. The padded window
    slots cost nothing on the GPU either: `SeqTransformer.forward` selects the
    valid windows before the encoder runs.

    ids are int32 rather than int64: `nn.Embedding` takes either, and it halves
    both the pinned buffer and the host-to-device copy.
    """

    def __init__(self, n_windows: int, window: int):
        self.M, self.W = int(n_windows), int(window)

    def __call__(self, batch):
        M, W, B = self.M, self.W, len(batch)
        ids = torch.zeros(B, M, W, dtype=torch.int32)
        lens = torch.zeros(B, M, dtype=torch.long)
        win = torch.zeros(B, M, dtype=torch.bool)
        for i, b in enumerate(batch):
            k = b["ids"].shape[0]
            if k > M:
                raise ValueError(f"{k} windows exceeds the configured {M}")
            ids[i, :k] = b["ids"].to(torch.int32)
            lens[i, :k] = b["lens"]
            win[i, :k] = True
        out = {"ids": ids, "lens": lens, "win_mask": win,
               "y": torch.tensor([b["y"] for b in batch], dtype=torch.long),
               "index": torch.tensor([b["index"] for b in batch], dtype=torch.long)}
        if "imports" in batch[0]:
            out["imports"] = torch.stack([b["imports"] for b in batch])
        return out


def collate(batch):
    """Variable-shape collate, kept for tests and for one-off scoring."""
    M = max(b["ids"].shape[0] for b in batch)
    W = batch[0]["ids"].shape[1]
    B = len(batch)
    ids = torch.zeros(B, M, W, dtype=torch.long)
    lens = torch.zeros(B, M, dtype=torch.long)
    win = torch.zeros(B, M, dtype=torch.bool)
    for i, b in enumerate(batch):
        k = b["ids"].shape[0]
        ids[i, :k] = b["ids"]
        lens[i, :k] = b["lens"]
        win[i, :k] = True
    out = {"ids": ids, "lens": lens, "win_mask": win,
           "y": torch.tensor([b["y"] for b in batch], dtype=torch.long),
           "index": torch.tensor([b["index"] for b in batch], dtype=torch.long)}
    if "imports" in batch[0]:
        out["imports"] = torch.stack([b["imports"] for b in batch])
    return out


def make_sampler(labels, archs, arch_balance: bool, seed: int, n: int = None):
    w = (D.arch_balanced_weights(labels, archs) if arch_balance
         else D.class_balanced_weights(labels))
    g = torch.Generator().manual_seed(int(seed))
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                 num_samples=int(n or len(w)),
                                 replacement=True, generator=g)


def loader(ds, batch_size, sampler=None, shuffle=False, workers=0):
    return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                      shuffle=shuffle and sampler is None,
                      collate_fn=Collate(ds.N, ds.W),
                      # pin_memory is off deliberately. A sweep builds a fresh
                      # DataLoader per run and PyTorch's caching HOST allocator
                      # never returns pinned blocks to the OS, so committed
                      # memory in the parent climbed past 20 GB over ~30 runs
                      # and the machine thrashed (GPU utilisation fell to 1%
                      # and epochs went from 9 s to 170 s). The batches here
                      # are ~6 MB, so the unpinned copy costs far less than
                      # that. `run_family_holdout.py --max-runs` bounds the
                      # process lifetime as a second line of defence.
                      num_workers=workers, pin_memory=False,
                      persistent_workers=bool(workers),
                      prefetch_factor=4 if workers else None,
                      drop_last=False)


# ---------------------------------------------------------------------------
# metrics used for model selection only (never on a test fold)
# ---------------------------------------------------------------------------
def macro_f1(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    fs = []
    for c in (0, 1):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(fs))


# ---------------------------------------------------------------------------
# train / predict
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _amp(device):
    if device.type != "cuda":
        return torch.autocast("cpu", enabled=False)
    return torch.autocast("cuda", dtype=torch.bfloat16)


@torch.no_grad()
def predict(model, dl, device, n_total: int):
    """Scores (P(ransomware)) for every file in `dl`, indexed by dataset index."""
    model.eval()
    score = np.full(n_total, np.nan, dtype=np.float64)
    for b in dl:
        ids = b["ids"].to(device, non_blocking=True)
        lens = b["lens"].to(device, non_blocking=True)
        win = b["win_mask"].to(device, non_blocking=True)
        imp = b.get("imports")
        imp = None if imp is None else imp.to(device, non_blocking=True)
        with _amp(device):
            logits, _ = model(ids, lens, win, imp)
        p = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        score[b["index"].numpy()] = p
    return score


def train_model(model, train_ds, val_ds, cfg: dict, device, seed: int,
                train_labels, train_archs, log=print, tag: str = ""):
    """Fine-tune with early stopping on the val fold's macro-F1.

    Returns (best_state_dict, history). The val fold is group-aware and carved
    out of the TRAINING folds by `cnn_vit_pipeline.cohort.add_val_fold`; the
    test fold is never touched here.
    """
    t = cfg["training"]
    model.to(device)
    sampler = make_sampler(train_labels, train_archs, cfg["sampler"]["arch_balanced"],
                           seed, n=t.get("samples_per_epoch") or len(train_ds))
    dl_tr = loader(train_ds, t["batch_size"], sampler=sampler,
                   workers=t["num_workers"])
    # the val fold is re-scored every epoch, so its loader's worker processes
    # are spawned once and amortised over `max_epochs` passes
    dl_va = loader(val_ds, t["eval_batch_size"], workers=t.get("eval_workers", 0))
    y_val = np.asarray(val_ds.labels, dtype=int)

    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"],
                            weight_decay=t["weight_decay"])
    crit = nn.CrossEntropyLoss(label_smoothing=t.get("label_smoothing", 0.0))
    steps = max(1, math.ceil(len(sampler) / t["batch_size"]))
    warm = t["warmup_epochs"] * steps
    total = t["max_epochs"] * steps

    def lr_at(s):
        if s < warm:
            return (s + 1) / max(1, warm)
        q = (s - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, q)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(t["max_epochs"]):
        model.train()
        # the running loss is accumulated on the device: reading it every step
        # would force a host sync and stall the pipeline
        t0, tot_t, nb = time.time(), torch.zeros((), device=device), 0
        for b in dl_tr:
            ids = b["ids"].to(device, non_blocking=True)
            lens = b["lens"].to(device, non_blocking=True)
            win = b["win_mask"].to(device, non_blocking=True)
            y = b["y"].to(device, non_blocking=True)
            imp = b.get("imports")
            imp = None if imp is None else imp.to(device, non_blocking=True)
            with _amp(device):
                logits, _ = model(ids, lens, win, imp)
                loss = crit(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if t.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            opt.step()
            sched.step()
            tot_t += loss.detach()
            nb += 1
        tot = float(tot_t)
        sc = predict(model, dl_va, device, len(val_ds))
        f1 = macro_f1(y_val, (sc >= 0.5).astype(int))
        hist.append({"epoch": ep, "train_loss": tot / max(1, nb),
                     "val_macro_f1": round(f1, 6),
                     "seconds": round(time.time() - t0, 1)})
        log(f"    {tag}ep{ep}: loss {tot/max(1,nb):.4f} val_macroF1 {f1:.4f} "
            f"({time.time()-t0:.0f}s)")
        if f1 > best + 1e-6:
            best, bad = f1, 0
            best_state = {k: v.detach().to("cpu").clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= t["patience"]:
                log(f"    {tag}early stop at epoch {ep} (best {best:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_state, {"history": hist, "best_val_macro_f1": round(best, 6)}
