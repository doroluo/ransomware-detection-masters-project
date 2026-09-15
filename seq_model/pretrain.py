#!/usr/bin/env python3
"""One masked-token pretraining pass over every mnemonic stream in both corpora.

    "C:/Users/chaoa/Downloads/cnn_vit_venv/Scripts/python.exe" \
        seq_model/pretrain.py --minutes 45

What it does. 15% of the token positions of a window are corrupted (BERT's
80% `<mask>` / 10% random mnemonic / 10% unchanged) and the encoder has to name
the original mnemonic at those positions. No label is read, anywhere: the pool
is `data.all_corpus_shas()`, which is a directory listing.

THIS IS TRANSDUCTIVE, and the summary says so in those words. The pretraining
pass sees the unlabelled token streams of every file in the cohort, including
the files that will later be in a held-out test fold or a held-out family. It
never sees their labels, and the fine-tuning vocabulary is still fitted on the
training folds alone (`data.Vocab.fit`), but a reader who wants a strictly
inductive number should read the `no_pretrain` ablation instead - that is
exactly what it is there for.

Budget and resumability. The pass stops at `--minutes` (default 45) or
`--max-steps`, whichever comes first, and checkpoints every `--save-every`
steps to `seq_models/pretrained.pt`. Re-invoking resumes from that checkpoint's
step count and optimiser state, so an interrupted pass continues rather than
restarting; `--minutes` is then the budget for THAT invocation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from seq_model import data as D                                   # noqa: E402
from seq_model.model import MaskedTokenModel, STEM_STRIDE         # noqa: E402

DEFAULT_CKPT = D.WEIGHTS_ROOT / "pretrained.pt"


class MaskedWindowDataset(Dataset):
    """`n` random windows drawn from the whole corpus, deterministically.

    Item i is a pure function of (seed, i): which file, which offset, which
    positions are corrupted and how. Sampling is uniform over FILES rather
    than over tokens, so the 2 MB goodware binaries do not drown out the 20 kB
    ransomware ones.
    """

    def __init__(self, shas, lengths, vocab: D.Vocab, window: int, n: int,
                 seed: int = 0, mask_rate: float = 0.15,
                 cache_root=D.CACHE_ROOT):
        self.shas = list(shas)
        self.lengths = [int(lengths[s]) for s in self.shas]
        self.vocab = vocab
        self.W = int(window)
        self.n = int(n)
        self.seed = int(seed)
        self.mask_rate = float(mask_rate)
        self.cache_root = Path(cache_root)
        self.mask_id = vocab.size                 # one extra row, see build()
        self.n_real = vocab.size - D.N_SPECIAL

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i))
        k = int(rng.integers(len(self.shas)))
        L = self.lengths[k]
        off = 0 if L <= self.W else int(rng.integers(0, L - self.W + 1))
        arr = D.load_ids(self.shas[k], self.cache_root)
        piece = self.vocab.encode(np.asarray(arr[off:off + self.W]))
        n = len(piece)
        ids = np.zeros(self.W, dtype=np.int64)
        ids[:n] = piece
        target = ids.copy()

        n_mask = max(1, int(round(n * self.mask_rate)))
        pos = rng.choice(n, size=min(n_mask, n), replace=False)
        r = rng.random(len(pos))
        ids[pos[r < 0.8]] = self.mask_id
        rnd = pos[(r >= 0.8) & (r < 0.9)]
        ids[rnd] = rng.integers(D.N_SPECIAL, self.vocab.size, size=len(rnd))
        # the remaining 10% keep their original token
        pad_to = max(1, int(round(self.W * self.mask_rate)))
        sel = np.zeros(pad_to, dtype=np.int64)
        lab = np.full(pad_to, -100, dtype=np.int64)
        m = min(len(pos), pad_to)
        sel[:m] = pos[:m]
        lab[:m] = target[pos[:m]]
        return {"ids": torch.from_numpy(ids), "len": n,
                "pos": torch.from_numpy(sel), "lab": torch.from_numpy(lab)}


def collate(batch):
    return {"ids": torch.stack([b["ids"] for b in batch]),
            "len": torch.tensor([b["len"] for b in batch], dtype=torch.long),
            "pos": torch.stack([b["pos"] for b in batch]),
            "lab": torch.stack([b["lab"] for b in batch])}


def run(args) -> int:
    dev = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)

    shas = D.all_corpus_shas()
    lengths = D.cache_lengths(args.cache)
    vocab = D.Vocab.fit(shas, Path(args.cache))       # corpus-wide, unsupervised
    print(f"corpus {len(shas)} files, {sum(lengths[s] for s in shas):,} tokens, "
          f"model vocabulary {vocab.size} (+1 <mask>)", flush=True)

    model = MaskedTokenModel(vocab.size + 1, window=args.window,
                             dim=args.dim, depth=args.depth, heads=args.heads,
                             mlp_dim=args.mlp_dim, dropout=args.dropout,
                             emb_dim=args.emb_dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    ckpt_path = Path(args.out)
    step0 = 0
    if ckpt_path.exists() and not args.restart:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ck.get("tokens") == vocab.tokens and ck.get("window") == args.window:
            model.load_state_dict(ck["state_dict"])
            if "optimizer" in ck:
                opt.load_state_dict(ck["optimizer"])
            step0 = int(ck.get("step", 0))
            print(f"resumed from {ckpt_path} at step {step0}", flush=True)
        else:
            print(f"{ckpt_path} does not match this vocabulary/window; "
                  f"starting fresh", flush=True)

    ds = MaskedWindowDataset(shas, lengths, vocab, args.window,
                             n=args.max_steps * args.batch_size,
                             seed=args.seed, mask_rate=args.mask_rate,
                             cache_root=Path(args.cache))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, collate_fn=collate,
                    pin_memory=(dev.type == "cuda"),
                    persistent_workers=bool(args.workers),
                    prefetch_factor=4 if args.workers else None)
    crit = nn.CrossEntropyLoss(ignore_index=-100)

    def save(step, loss):
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = ckpt_path.with_suffix(".pt.tmp")
        torch.save({"state_dict": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "tokens": vocab.tokens, "lut": vocab.lut.tolist(),
                    "window": args.window, "step": step, "loss": loss,
                    "config": {"dim": args.dim, "depth": args.depth,
                               "heads": args.heads, "mlp_dim": args.mlp_dim,
                               "emb_dim": args.emb_dim, "dropout": args.dropout,
                               "mask_rate": args.mask_rate,
                               "batch_size": args.batch_size, "lr": args.lr,
                               "seed": args.seed,
                               "corpus_files": len(shas)}}, tmp)
        tmp.replace(ckpt_path)

    model.train()
    t_start = time.time()
    deadline = t_start + args.minutes * 60
    step, run_loss, run_n, run_acc = step0, 0.0, 0, 0.0
    last_loss = None
    W = args.window
    for b in dl:
        if time.time() > deadline or step >= args.max_steps:
            break
        ids = b["ids"].to(dev, non_blocking=True)
        pos = b["pos"].to(dev, non_blocking=True)
        lab = b["lab"].to(dev, non_blocking=True)
        n = b["len"].to(dev, non_blocking=True)
        tok_mask = torch.arange(W, device=dev).unsqueeze(0) < n.unsqueeze(1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
            logits = model(ids, tok_mask, pos)
            loss = crit(logits.float().reshape(-1, logits.shape[-1]),
                        lab.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        last_loss = float(loss.detach())
        run_loss += last_loss
        keep = lab.reshape(-1) != -100
        run_acc += float((logits.float().reshape(-1, logits.shape[-1]).argmax(-1)[keep]
                          == lab.reshape(-1)[keep]).float().mean())
        run_n += 1
        if step % args.log_every == 0:
            print(f"step {step}: loss {run_loss/run_n:.4f} acc "
                  f"{run_acc/run_n:.4f} ({time.time()-t_start:.0f}s, "
                  f"{(step-step0)*args.batch_size*W/1e6/(time.time()-t_start):.1f} "
                  f"Mtok/s)", flush=True)
            run_loss = run_acc = 0.0
            run_n = 0
        if step % args.save_every == 0:
            save(step, last_loss)
    save(step, last_loss)
    print(f"pretraining stopped at step {step} after "
          f"{time.time()-t_start:.0f}s; checkpoint {ckpt_path}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_CKPT))
    ap.add_argument("--cache", default=str(D.CACHE_ROOT))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--minutes", type=float, default=45.0)
    ap.add_argument("--max-steps", type=int, default=200000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--mask-rate", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--mlp-dim", type=int, default=512)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--restart", action="store_true")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
