#!/usr/bin/env python3
"""Train and evaluate the CNN-ViT (HierarchicalMalwareNet) on the shared cohort.

The model, the dataset class and the training protocol are the originals from
model_train.py - imported, never copied, so they cannot drift. What changes is
the data: build_dataset.py lays out train/val/test according to the shared
cohort split instead of stratified_split.py's random per-class shuffle.

    python cnn_vit_pipeline/build_dataset.py --dataset mendeley --out DIR
    python cnn_vit_pipeline/train_eval.py --data DIR --epochs 80 --device cpu

    # plumbing check on the goodware images alone, with a clearly fake second
    # class; measures seconds/epoch on this CPU and writes nothing to results/
    python cnn_vit_pipeline/train_eval.py --smoke \
        --images ../cnn_vit_images/mendeley_goodware \
        --scratch /tmp/cnnvit_smoke --epochs 2 --max-samples 64

A note on importing model_train
-------------------------------
model_train.py has no module-level side effects (its old __main__ block is
gone), so importing it is safe, and it no longer needs torchvision: the one
ToTensor call is a plain uint8 -> float32 / 255 conversion now. The model and
dataset classes are imported from it rather than copied, so they cannot go
stale.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
# The yanping scripts (asm_parser.py, model_train.py) live under CNN-ViT/ since
# upstream moved them there; older checkouts had them at the repo root.
for _p in (REPO_ROOT / "CNN-ViT", REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cnn_vit_pipeline import cohort as C  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "cnn_vit"
VM_RUNBOOK = "vm_package/README.md"

# model_train.py's own settings, reproduced so this harness trains the way the
# original does. Anything not listed here comes from model_train itself
# (PATIENCE = 8, MIN_DELTA = 1e-4).
BATCH_SIZE = 16
BASE_LEARNING_RATE = 3e-4
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.15
DROPOUT = 0.15
NUM_CLASSES = 2

# ------------------------------------------------- importing model_train ----
def import_model_train():
    """Import model_train.py (the original model and dataset classes)."""
    import model_train
    return model_train


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


# ----------------------------------------------------------- data helpers ---
def load_manifest(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build the tree with "
            f"cnn_vit_pipeline/build_dataset.py so every image carries its "
            f"sha256, architecture and family.")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def subsample(dataset, max_samples: int, seed: int):
    """Deterministic, label-stratified subset, for smoke runs."""
    import torch
    if not max_samples or max_samples >= len(dataset):
        return dataset, list(range(len(dataset)))
    labels = np.array([s[1] for s in dataset.all_samples])
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        n = max(1, round(max_samples * len(idx) / len(labels)))
        keep.extend(rng.permutation(idx)[:n].tolist())
    keep.sort()
    return torch.utils.data.Subset(dataset, keep), keep


def make_loader(mt, dataset, indices, batch_size, shuffle_weighted, seed):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    base = dataset.dataset if hasattr(dataset, "dataset") else dataset
    if not shuffle_weighted:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                          num_workers=0)
    targets = torch.tensor([base.all_samples[i][1] for i in indices],
                           dtype=torch.long)
    counts = torch.bincount(targets, minlength=NUM_CLASSES).float().clamp(min=1)
    weights = (1.0 / counts)[targets]
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                    replacement=True, generator=g)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      num_workers=0)


# ---------------------------------------------------------------- the run ---
def train(mt, data_dir: Path, epochs: int, device_name: str, max_samples: int,
          batch_size: int, seed: int, ckpt_dir: Path, quiet: bool):
    """The original protocol: AdamW + warmup + cosine, weighted sampling,
    grad clipping, early stopping on val loss. Returns everything the caller
    needs to score the test set."""
    import torch
    import torch.nn as nn

    device = torch.device(device_name)
    sets, loaders, index = {}, {}, {}
    for fold in ("train", "val", "test"):
        folder = data_dir / fold
        ds = mt.MalwareMaskedDataset(
            base_folder=str(folder),
            transform=mt.MaskAwareStructuralShift(p=0.4, max_shift_ratio=0.20)
            if fold == "train" else None)
        sub, idx = subsample(ds, max_samples, seed)
        sets[fold], index[fold] = sub, idx
        loaders[fold] = make_loader(mt, sub, idx, batch_size,
                                    shuffle_weighted=(fold == "train"), seed=seed)
        print(f"  {fold:<5} {len(sub):5d} samples  classes {ds.class_names}")

    class_names = list(sets["train"].dataset.class_names) if hasattr(
        sets["train"], "dataset") else list(sets["train"].class_names)

    model = mt.HierarchicalMalwareNet(num_classes=NUM_CLASSES,
                                      dropout=DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - WARMUP_EPOCHS), eta_min=1e-6)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_model.pth"
    best_val, patience, epoch_times, history = float("inf"), 0, [], []

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running = 0.0
        if epoch < WARMUP_EPOCHS:
            lr = BASE_LEARNING_RATE * ((epoch + 1) / WARMUP_EPOCHS)
            for gp in optimizer.param_groups:
                gp["lr"] = lr
        cur_lr = optimizer.param_groups[0]["lr"]

        for images, masks, labels in loaders["train"]:
            images, masks, labels = (images.to(device), masks.to(device),
                                     labels.to(device))
            optimizer.zero_grad()
            out = model(images, raw_patch_masks=masks)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item() * images.size(0)
        if epoch >= WARMUP_EPOCHS:
            scheduler.step()

        train_loss = running / max(len(sets["train"]), 1)
        val_loss, val_acc = _quiet_eval(mt, model, loaders["val"], criterion,
                                        device, class_names, quiet)
        dt = time.time() - t0
        epoch_times.append(dt)
        history.append({"epoch": epoch + 1, "lr": cur_lr,
                        "train_loss": train_loss, "val_loss": val_loss,
                        "val_acc": val_acc, "seconds": round(dt, 2)})
        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] (LR {cur_lr:.6f}) "
              f"train {train_loss:.4f} | val {val_loss:.4f} | "
              f"val acc {val_acc:.2f}% | {dt:.1f}s")

        if val_loss < best_val - mt.MIN_DELTA:
            best_val, patience = val_loss, 0
            torch.save(model.state_dict(), best_path)
        elif epoch >= WARMUP_EPOCHS:
            patience += 1
            if patience >= mt.PATIENCE:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    return model, sets, index, loaders, class_names, epoch_times, history, device


def _quiet_eval(mt, model, loader, criterion, device, class_names, quiet):
    """model_train.evaluate_model prints a per-class table every epoch."""
    if not quiet:
        return mt.evaluate_model(model, loader, criterion, device, class_names,
                                 final_eval=False)
    buf, sys.stdout = sys.stdout, open(os.devnull, "w")
    try:
        return mt.evaluate_model(model, loader, criterion, device, class_names,
                                 final_eval=False)
    finally:
        sys.stdout.close()
        sys.stdout = buf


def predict(model, loader, device):
    import torch
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for images, masks, labels in loader:
            out = model(images.to(device), raw_patch_masks=masks.to(device))
            prob = torch.softmax(out, dim=1)[:, 1]
            y_true.extend(labels.tolist())
            y_pred.extend(out.argmax(1).cpu().tolist())
            y_score.extend(prob.cpu().tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_score)


def ordered_ids(dataset, indices):
    """asm_ids in DataLoader order, so predictions line up with the manifest."""
    base = dataset.dataset if hasattr(dataset, "dataset") else dataset
    return [Path(base.all_samples[i][0]).stem for i in indices]


def do_run(a) -> int:
    data_dir = Path(a.data)
    manifest = load_manifest(data_dir)
    by_id = manifest.set_index("asm_id")

    report_path = data_dir / "build_report.json"
    dataset_name = a.dataset
    if not dataset_name and report_path.exists():
        dataset_name = json.loads(report_path.read_text()).get("dataset")
    dataset_name = dataset_name or "unknown"

    n_test_classes = len({r["label"] for _, r in manifest.iterrows()
                          if r["fold"] == "test"})
    if n_test_classes < 2:
        print(f"\nERROR: the test fold in {data_dir} has {n_test_classes} "
              f"class(es). The ransomware and test-goodware images are built "
              f"from .asm trees that the VM has not delivered yet:", file=sys.stderr)
        for t in json.loads(report_path.read_text()).get("trees_missing", []) \
                if report_path.exists() else []:
            print(f"  missing image tree: {t['image_tree']}  "
                  f"(asm_output/{t['asm_tree']}, produced on the "
                  f"{t['produced_on']})", file=sys.stderr)
        print(f"See {VM_RUNBOOK} step 3, re-run build_dataset.py, then this. "
              f"No metrics are written; nothing is fabricated.", file=sys.stderr)
        print("Use --smoke to exercise the training path on goodware alone.",
              file=sys.stderr)
        return 2

    t_start = time.time()
    mt = import_model_train()
    seed_everything(a.seed)
    print(f"=== CNN-ViT train_eval  dataset={dataset_name} device={a.device} ===")
    model, sets, index, loaders, class_names, epoch_times, history, device = train(
        mt, data_dir, a.epochs, a.device, a.max_samples, a.batch_size, a.seed,
        Path(a.ckpt_dir or (data_dir / "checkpoints")), a.quiet)

    y_true, y_pred, y_score = predict(model, loaders["test"], device)
    ids = ordered_ids(sets["test"], index["test"])
    arch = [by_id.loc[i, "arch"] if i in by_id.index else "unknown" for i in ids]
    family = [by_id.loc[i, "family"] if i in by_id.index else "unknown" for i in ids]

    ckpt_dir = Path(a.ckpt_dir or (data_dir / "checkpoints"))
    result = C.build_result(
        y_true, y_pred, y_score, arch, family,
        model="HierarchicalMalwareNet (CNN-ViT)",
        epochs_run=len(epoch_times),
        epochs_requested=a.epochs,
        epoch_cap_reason=a.epoch_cap_reason or None,
        seconds_per_epoch_mean=round(float(np.mean(epoch_times)), 2),
        seconds_per_epoch_median=round(float(np.median(epoch_times)), 2),
        seconds_per_epoch_min=round(float(np.min(epoch_times)), 2),
        seconds_per_epoch_max=round(float(np.max(epoch_times)), 2),
        train_seconds_total=round(float(np.sum(epoch_times)), 2),
        device=a.device,
        device_name=device_name(a.device),
        batch_size=a.batch_size,
        seed=a.seed,
        class_names=class_names,
        n_train=len(sets["train"]),
        n_val=len(sets["val"]),
        n_test=len(sets["test"]),
        history=history,
    )
    print(f"\ntest acc {result['accuracy']:.4f}  "
          f"bal_acc {result['balanced_accuracy']:.4f}  "
          f"macro_f1 {result['macro_f1']:.4f}  "
          f"FPR {result['false_positive_rate']:.4f}")

    fold_df = manifest.assign(label=manifest["label"].astype(int))
    out = Path(a.out) if a.out else RESULTS_DIR / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    # ---- the three companion files the coordinator asked for -------------
    # splits.csv: the same five columns results/exp*/splits.csv uses, so the
    # three pipelines' membership can be diffed directly, plus sha256/arch/
    # family which the image pipeline needs and the text one does not have.
    splits = fold_df[["sha256", "asm_id", "source", "label", "family_or_group",
                      "fold", "split", "arch", "family", "image_tree"]].copy()
    splits = splits.rename(columns={"asm_id": "file", "family_or_group": "group"})
    splits.to_csv(out / "splits.csv", index=False)

    # training_log.csv: one row per epoch.
    log = pd.DataFrame(history)[["epoch", "lr", "train_loss", "val_loss",
                                 "val_acc", "seconds"]]
    log.to_csv(out / "training_log.csv", index=False)

    # test_predictions.csv: per-sample, so the per-family block can be redone
    # or checked without re-running training.
    pd.DataFrame({"file": ids, "arch": arch, "family": family,
                  "y_true": y_true, "y_pred": y_pred,
                  "score_ransomware": y_score}).to_csv(
        out / "test_predictions.csv", index=False)

    cfg = build_config(a, dataset_name, data_dir, ckpt_dir, mt, device_name(a.device))
    (out / "config_used.yaml").write_text(to_yaml(cfg), encoding="utf-8")

    path = C.write_metrics(
        out / "metrics.json",
        experiment=f"cnn_vit_{dataset_name}",
        description=("HierarchicalMalwareNet (CNN-ViT) from model_train.py on "
                     "the shared cohort split; .NET / packed / UPX / Thanos "
                     "excluded and the ransomware families kept disjoint, "
                     "unlike the original stratified_split.py protocol. Images "
                     "are rendered by the unmodified asm_parser.py from the "
                     "revised extractor's full disassembly (skip-data on), not "
                     "from asm_parse.py's linear sweep -- see asm_tool/README.md "
                     "section 6."),
        samples=C.split_summary(fold_df, "fold"),
        results=[result],
        elapsed_seconds=time.time() - t_start,
        dataset=dataset_name,
        data_dir=str(data_dir),
        config=cfg,
    )
    print(f"wrote {path}")
    print(f"wrote {out/'splits.csv'}, {out/'training_log.csv'}, "
          f"{out/'test_predictions.csv'}, {out/'config_used.yaml'}")
    print(f"model weights stay out of results/: {ckpt_dir}")
    return 0


# ------------------------------------------------------- run bookkeeping ----
def device_name(device: str) -> str:
    """A human-readable name for whatever we actually trained on."""
    import torch
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = int(device.split(":")[1]) if ":" in device else 0
        cap = torch.cuda.get_device_capability(idx)
        return (f"{torch.cuda.get_device_name(idx)} (sm_{cap[0]}{cap[1]}), "
                f"torch {torch.__version__}")
    import platform
    return f"CPU {platform.processor() or platform.machine()}, torch {torch.__version__}"


def build_config(a, dataset_name, data_dir, ckpt_dir, mt, dev_name) -> dict:
    """Everything needed to reproduce the run, in config_used.yaml's shape."""
    return {
        "pipeline": "cnn_vit",
        "dataset": dataset_name,
        "model": {
            "class": "HierarchicalMalwareNet",
            "defined_in": "model_train.py",
            "num_classes": NUM_CLASSES,
            "dropout": DROPOUT,
            "input": "256x256 uint8 token image + 16x16 ViT patch mask",
        },
        "images": {
            "renderer": "asm_parser.py (unmodified)",
            "square_resolution": 256,
            "vit_patch_size": 16,
            "asm_source": "asm_tool/unified_to_asm.py over the revised "
                          "extractor (skipdata on, .skip markers removed)",
        },
        "training": {
            "batch_size": a.batch_size,
            "base_learning_rate": BASE_LEARNING_RATE,
            "warmup_epochs": WARMUP_EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR after warmup, eta_min 1e-6",
            "grad_clip_max_norm": 1.0,
            "sampler": "WeightedRandomSampler (inverse class frequency)",
            "augmentation": "MaskAwareStructuralShift(p=0.4, "
                            "max_shift_ratio=0.20), train fold only",
            "early_stopping": {"monitor": "val_loss",
                               "patience": int(mt.PATIENCE),
                               "min_delta": float(mt.MIN_DELTA)},
            "max_epochs": a.epochs,
            "epoch_cap_reason": a.epoch_cap_reason or None,
            "seed": a.seed,
            "device": a.device,
            "device_name": dev_name,
        },
        "split": {
            "source": "cnn_vit_pipeline/cohort.py",
            "val_fraction": C.VAL_FRACTION,
            "val_seed": C.VAL_SEED,
            "val_is": "group-aware, carved out of train only",
        },
        "paths": {
            "data_dir": str(data_dir),
            "checkpoints": str(ckpt_dir),
            "results": str(Path(a.out) if a.out else RESULTS_DIR / dataset_name),
        },
    }


def to_yaml(obj, indent: int = 0) -> str:
    """Tiny YAML writer (PyYAML is not a dependency of this repo)."""
    pad = "  " * indent
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                out.append(f"{pad}{k}:\n{to_yaml(v, indent + 1)}")
            else:
                out.append(f"{pad}{k}: {_scalar(v)}")
        return "\n".join(out) + ("\n" if indent == 0 else "")
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
        return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'
    return s


# ------------------------------------------------------------ smoke test ----
FAKE_CLASS = "Class_1_FAKE_NOT_RANSOMWARE"


def build_smoke_tree(images: Path, scratch: Path, n: int, seed: int) -> Path:
    """A two-class tree from goodware images alone, for timing only.

    Half the goodware is relabelled into a class named FAKE_NOT_RANSOMWARE so
    nobody can mistake the output for a result. It never touches results/.
    """
    pngs = sorted(images.rglob("*.png"))
    if not pngs:
        raise SystemExit(f"no PNGs under {images}")
    rng = random.Random(seed)
    rng.shuffle(pngs)
    pngs = pngs[:max(8, n * 2)]
    if scratch.exists():
        shutil.rmtree(scratch)
    for fold in ("train", "val", "test"):
        for cls in ("Class_0_Goodware", FAKE_CLASS):
            (scratch / fold / cls).mkdir(parents=True, exist_ok=True)
    for i, png in enumerate(pngs):
        fold = "train" if i % 10 < 7 else ("val" if i % 10 < 8 else "test")
        cls = "Class_0_Goodware" if i % 2 == 0 else FAKE_CLASS
        dest = scratch / fold / cls
        shutil.copy2(png, dest / png.name)
        mask = png.with_name(png.stem + "_vit_mask.npy")
        if mask.exists():
            shutil.copy2(mask, dest / mask.name)
    return scratch


def do_smoke(a) -> int:
    print("=" * 72)
    print("SMOKE TEST - goodware images only, with a deliberately FAKE second")
    print("class. This measures seconds/epoch and proves the plumbing.")
    print("It is NOT a result and is never written under results/.")
    print("=" * 72)
    mt = import_model_train()
    seed_everything(a.seed)
    scratch = Path(a.scratch)
    build_smoke_tree(Path(a.images), scratch, a.max_samples or 64, a.seed)

    t0 = time.time()
    model, sets, index, loaders, class_names, epoch_times, history, device = train(
        mt, scratch, a.epochs, a.device, a.max_samples, a.batch_size, a.seed,
        scratch / "checkpoints", a.quiet)
    y_true, y_pred, y_score = predict(model, loaders["test"], device)

    n_train = len(sets["train"])
    per_epoch = float(np.median(epoch_times))
    print("\n--- SMOKE RESULT (timing only, labels are fake) ---")
    print(f"samples: train {n_train}, val {len(sets['val'])}, "
          f"test {len(sets['test'])}")
    print(f"epochs run: {len(epoch_times)}  "
          f"seconds/epoch: median {per_epoch:.2f}, mean "
          f"{np.mean(epoch_times):.2f}, min {min(epoch_times):.2f}, "
          f"max {max(epoch_times):.2f}")
    if n_train:
        print(f"per-sample train+val cost: {per_epoch / n_train * 1000:.1f} ms")
    print(f"total wall clock: {time.time() - t0:.1f}s on {a.device}")
    print(f"(fake-label test accuracy {np.mean(y_true == y_pred):.3f} - "
          f"meaningless by construction, the labels are arbitrary)")
    print(f"\nscratch tree: {scratch}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="tree built by build_dataset.py")
    ap.add_argument("--dataset", choices=C.DATASETS, default=None,
                    help="defaults to the value in DIR/build_report.json")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--max-samples", type=int, default=0,
                    help="cap each fold, stratified by label; for smoke runs")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--epoch-cap-reason", default="",
                    help="recorded verbatim in metrics.json and "
                         "config_used.yaml when --epochs is below the "
                         "protocol's 80 for reasons of wall clock")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--quiet", action="store_true", default=True)
    ap.add_argument("--verbose", dest="quiet", action="store_false")
    ap.add_argument("--smoke", action="store_true",
                    help="timing run on goodware images with a fake 2nd class")
    ap.add_argument("--images", help="--smoke: a Class_*/ image directory")
    ap.add_argument("--scratch", help="--smoke: where to build the fake tree")
    a = ap.parse_args()

    if a.smoke:
        if not a.images or not a.scratch:
            print("--smoke needs --images and --scratch", file=sys.stderr)
            return 2
        if "results" in Path(a.scratch).resolve().parts:
            print("--scratch must not be inside results/", file=sys.stderr)
            return 2
        return do_smoke(a)
    if not a.data:
        print("--data is required (or use --smoke)", file=sys.stderr)
        return 2
    return do_run(a)


if __name__ == "__main__":
    raise SystemExit(main())
