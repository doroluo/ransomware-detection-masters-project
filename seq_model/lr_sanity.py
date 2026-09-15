#!/usr/bin/env python3
"""The one pre-registration check the protocol allows, and nothing else.

The brief permits "one quick sanity check of learning rate on fold 0's val
fold only". This script is that check, written down so the claim is auditable:

  * dataset mendeley, K-fold fold 0. The 2,006 training rows are split by
    `cnn_vit_pipeline.cohort.add_val_fold` into train and a group-aware val
    fold; fold 0's 503 TEST rows are never loaded, scored or looked at.
  * three learning rates around the CNN-ViT recipe's 3e-4, everything else
    exactly `seq_model/config.yaml`.
  * it also answers the second question the frozen config needs: does
    `max_epochs` reach the val plateau, or is the budget cutting training
    short? The recorded history shows the epoch at which each run peaked.

Output: `seq_model/lr_sanity.json`, quoted in the summary.

    "C:/Users/chaoa/Downloads/cnn_vit_venv/Scripts/python.exe" \
        seq_model/lr_sanity.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from family_holdout.common import Folds                            # noqa: E402
from seq_model import config as CFG                                # noqa: E402
from seq_model import data as D                                    # noqa: E402
from seq_model.run_family_holdout import build_model, val_split    # noqa: E402

OUT = HERE / "lr_sanity.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lrs", default="1e-4,3e-4,1e-3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-epochs", type=int, default=16,
                    help="deliberately above the config's budget, so the "
                         "plateau is visible rather than assumed")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    import torch
    from seq_model.train import FileWindowDataset, train_model
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")

    cfg = CFG.load()
    folds = Folds("mendeley")
    f, tr_all, te = folds.kfold()[0]
    tr, va, _ = val_split(folds, tr_all, te)
    print(f"fold 0: train {len(tr)}, val {len(va)}, test {len(te)} "
          f"(the test rows are not read by this script)", flush=True)

    cache = Path(cfg["paths"]["token_cache"])
    vocab = D.Vocab.fit(folds.sha[np.concatenate([tr, va])], cache,
                        cfg["vocabulary"]["min_count"])
    W, N = cfg["windows"]["window"], cfg["windows"]["n_windows"]
    mk = lambda i: FileWindowDataset(folds.sha[i], folds.y[i], vocab, W, N, cache)  # noqa: E731
    ds_tr, ds_va = mk(tr), mk(va)

    results = {}
    for lr in [float(x) for x in a.lrs.split(",")]:
        c = json.loads(json.dumps(cfg))
        c["training"]["lr"] = lr
        c["training"]["max_epochs"] = a.max_epochs
        c["training"]["patience"] = a.max_epochs      # no early stop: see the curve
        model, pre = build_model(c, vocab, print)
        t0 = time.time()
        _, hist = train_model(model, ds_tr, ds_va, c, device, seed=1,
                              train_labels=folds.y[tr], train_archs=folds.arch[tr],
                              log=print, tag=f"[lr={lr:g}] ")
        curve = [h["val_macro_f1"] for h in hist["history"]]
        results[f"{lr:g}"] = {
            "best_val_macro_f1": hist["best_val_macro_f1"],
            "best_epoch": int(np.argmax(curve)),
            "epochs_run": len(curve),
            "val_macro_f1_by_epoch": curve,
            "seconds": round(time.time() - t0, 1),
            "pretrained": pre}
        print(f"lr={lr:g}: best {hist['best_val_macro_f1']} at epoch "
              f"{np.argmax(curve)} ({time.time()-t0:.0f}s)", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    best = max(results, key=lambda k: results[k]["best_val_macro_f1"])
    doc = {"what": "learning-rate sanity check, mendeley fold 0 VAL fold only; "
                   "fold 0's test rows were never scored",
           "config": str(CFG.CONFIG_PATH), "device": str(device),
           "n_train": int(len(tr)), "n_val": int(len(va)),
           "max_epochs_probed": a.max_epochs,
           "config_max_epochs": cfg["training"]["max_epochs"],
           "config_lr": cfg["training"]["lr"],
           "results": results, "argmax_lr": best,
           "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    Path(a.out).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"wrote {a.out}; argmax lr = {best}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
