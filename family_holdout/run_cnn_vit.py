#!/usr/bin/env python3
"""
run_cnn_vit.py - family-holdout evaluation of the CNN-ViT encoder.

Runs the untuned CNN-ViT recipe (HierarchicalMalwareNet, exactly as
cnn_vit_pipeline/train_eval.py trains it: batch 16, AdamW 3e-4 with 5 warmup
epochs and cosine decay, weighted sampling, MaskAwareStructuralShift on train
only, early stopping on val loss with patience 8, max 80 epochs, argmax at
0.5) over the fold definition in family_holdout/folds.py. Nothing is tuned
here and no threshold is moved; the point is to replace the single 24/14
family split with a scheme in which every one of the 38 cohort families is
held out.

Two schemes, both read from results/family_holdout/folds_<dataset>.csv:

  K-fold  train on four folds, test on the fifth. A group-aware 10% val fold
          is carved out of the four training folds by cnn_vit_pipeline.cohort
          .add_val_fold, so no ransomware family and no goodware duplicate
          stream / source project straddles train and val either. Three seeds
          per fold; a fold's row is the mean over its seeds, and the pooled
          prediction for a sample is the argmax of its seed-mean score.

  LOFO    leave one family out: train on the other 37 families plus ALL
          goodware (again with a val fold carved out), test on the held-out
          family alone. One seed per family. There is no goodware in the test
          set, so this is a recall study only - never quote an FPR from it.

The model, the dataset class and the training loop are imported from
cnn_vit_pipeline/train_eval.py and CNN-ViT/model_train.py, never copied, so
this runner cannot drift from the pipeline it is supposed to be measuring.

Data. The images already exist, one PNG + _vit_mask.npy per sha256, under
cnn_vit_images/unified_mendeley (both classes) and
cnn_vit_images/unified_goodware_balanced (goodware). For each fold this script
hard-links them into a train/val/test/Class_* tree of the shape
build_dataset.py produces, so the imported trainer runs unchanged. Trees go
under cnn_vit_data/family_holdout/, weights under cnn_vit_models/, never into
results/.

Resumable. Every run writes models/.../run.json the moment it finishes; a
re-invocation skips any run whose run.json is already there and consistent
with the plan, so an interrupted sweep continues where it stopped. --report-
only rebuilds every CSV, metrics.json and the summary from those run.json
files without touching the GPU.

    python family_holdout/run_cnn_vit.py --dataset both --device cuda
    python family_holdout/run_cnn_vit.py --dataset both --report-only

Floors written next to every fold: `majority_floor` is the accuracy of always
predicting the larger class of that test fold, `x86_rule_floor` the accuracy
of "ransomware iff the PE machine field says x86" - the confound the image
features have to beat, and the same rule graph2vec_pipeline/final_eval.py
uses.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics as st
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnn_vit_pipeline import cohort as C          # noqa: E402
from cnn_vit_pipeline import train_eval as T      # noqa: E402

MODEL_NAME = "HierarchicalMalwareNet"
PIPELINE = "cnn_vit"
DATASETS = ("mendeley", "balanced")
CLASS_DIRS = {0: "Class_0_Goodware", 1: "Class_1_Ransomware"}
HEX64 = re.compile(r"[0-9a-f]{64}")

FOLDS_DIR = REPO_ROOT / "results" / "family_holdout"
OUT_ROOT = REPO_ROOT / "results" / "family_holdout"
DOWNLOADS = REPO_ROOT.parent
IMAGES_ROOT = Path(os.environ.get("RANSOM_CNN_VIT_IMAGES", DOWNLOADS / "cnn_vit_images"))
DATA_ROOT = Path(os.environ.get("RANSOM_FH_DATA", DOWNLOADS / "cnn_vit_data" / "family_holdout"))
MODELS_ROOT = Path(os.environ.get("RANSOM_FH_MODELS", DOWNLOADS / "cnn_vit_models" / "family_holdout"))

# Which image trees hold each dataset's samples. A tree can hold both classes;
# the class folder is cross-checked against the cohort label at link time.
IMAGE_TREES = {
    "mendeley": ("unified_mendeley",),
    "balanced": ("unified_goodware_balanced", "unified_mendeley"),
}

KFOLD_SEEDS = (1, 2, 3)
LOFO_SEEDS = (1,)
MAX_EPOCHS = 80


# --------------------------------------------------------------- the data ---
def load_fold_table(folds_dir: Path, dataset: str) -> pd.DataFrame:
    """results/family_holdout/folds_<dataset>.csv, typed, with the numeric
    fold column renamed to `kfold` so add_val_fold's train/val/test `fold`
    cannot collide with it."""
    path = folds_dir / f"folds_{dataset}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found; run family_holdout/folds.py first.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["label"] = df["label"].astype(int)
    df["kfold"] = df["fold"].astype(int)
    df = df.drop(columns=["fold"])
    df["family_or_group"] = df["group"]
    df["source"] = np.where(df["label"] == 1, f"{dataset}_ransomware", f"{dataset}_goodware")
    if df["sha256"].duplicated().any():
        raise AssertionError(f"duplicate sha256 in {path}")
    return df.reset_index(drop=True)


def image_index(images_root: Path, trees) -> dict:
    """sha256 -> {png, mask, tree_label, tree}.

    The PNG stem asm_parser.py produced carries the .asm path it was rendered
    from, so it is `<something>__<sha256>` (or the bare sha256 when the render
    was rooted at the family directory). Only the trailing 64 hex characters
    are used, and the result is checked for collisions.
    """
    idx: dict[str, dict] = {}
    for tree in trees:
        root = images_root / tree
        if not root.is_dir():
            raise SystemExit(f"image tree not found: {root}")
        for png in sorted(root.rglob("*.png")):
            cls = png.parent.name
            if not cls.startswith("Class_"):
                continue
            tail = png.stem.rsplit("__", 1)[-1]
            if not HEX64.fullmatch(tail):
                continue
            mask = png.with_name(png.stem + "_vit_mask.npy")
            entry = {"png": str(png), "mask": str(mask) if mask.exists() else "",
                     "tree_label": int(cls.split("_")[1]), "tree": tree}
            prev = idx.get(tail)
            if prev is not None and prev["png"] != entry["png"]:
                raise AssertionError(
                    f"sha256 {tail} rendered twice: {prev['png']} and {entry['png']}")
            idx[tail] = entry
    return idx


def plan_kfold(tbl: pd.DataFrame, k: int) -> pd.DataFrame:
    """train = the other four folds (minus a group-aware 10% val), test = fold k."""
    d = tbl.copy()
    d["split"] = np.where(d["kfold"] == k, "test", "train")
    return C.add_val_fold(d)


def plan_lofo(tbl: pd.DataFrame, family: str) -> pd.DataFrame:
    """train = every other family + all goodware (minus val), test = `family`."""
    d = tbl.copy()
    d["split"] = np.where((d["label"] == 1) & (d["family"] == family), "test", "train")
    if not (d["split"] == "test").any():
        raise SystemExit(f"no ransomware rows for family {family!r}")
    return C.add_val_fold(d)


def check_plan(plan: pd.DataFrame, scheme: str, key) -> None:
    """The invariants the whole study rests on, asserted every time a tree is
    built rather than only in the tests."""
    parts = {f: plan[plan["fold"] == f] for f in ("train", "val", "test")}
    shas = {f: set(p["sha256"]) for f, p in parts.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        both = shas[a] & shas[b]
        if both:
            raise AssertionError(f"{scheme} {key}: {len(both)} sha256 in both {a} and {b}")
    for a, b in (("train", "test"), ("val", "test"), ("train", "val")):
        ga = set(parts[a]["family_or_group"])
        gb = set(parts[b]["family_or_group"])
        if ga & gb:
            raise AssertionError(
                f"{scheme} {key}: group(s) in both {a} and {b}: {sorted(ga & gb)[:5]}")
    if len(parts["test"]) == 0 or len(parts["train"]) == 0 or len(parts["val"]) == 0:
        raise AssertionError(f"{scheme} {key}: an empty fold")
    if scheme == "lofo":
        fams = set(parts["test"]["family"])
        if fams != {key} or (parts["test"]["label"] == 0).any():
            raise AssertionError(f"lofo {key}: test set is {sorted(fams)}, not exactly the family")
    if not plan["fold"].isin(["train", "val", "test"]).all():
        raise AssertionError(f"{scheme} {key}: unknown fold value")
    if len(plan) != len(shas["train"] | shas["val"] | shas["test"]):
        raise AssertionError(f"{scheme} {key}: a sample lands in more than one fold")


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def materialise(plan: pd.DataFrame, index: dict, out_dir: Path) -> dict:
    """Hard-link the PNG/mask pairs into out_dir/{train,val,test}/Class_*.

    Destination stems are the sha256, so the trainer's `asm_id` (the file
    stem) is the sha256 and predictions join back to the fold table exactly.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for fold in ("train", "val", "test"):
        for cls in CLASS_DIRS.values():
            (out_dir / fold / cls).mkdir(parents=True, exist_ok=True)
    counts = Counter()
    for r in plan.itertuples(index=False):
        entry = index.get(r.sha256)
        if entry is None:
            raise AssertionError(f"no image for {r.sha256}")
        if entry["tree_label"] != int(r.label):
            raise AssertionError(
                f"{r.sha256}: image sits in class {entry['tree_label']} but the "
                f"cohort says {int(r.label)}")
        dest = out_dir / r.fold / CLASS_DIRS[int(r.label)]
        counts[link_or_copy(Path(entry["png"]), dest / f"{r.sha256}.png")] += 1
        if entry["mask"]:
            link_or_copy(Path(entry["mask"]), dest / f"{r.sha256}_vit_mask.npy")
        else:
            counts["mask_missing"] += 1
    man = plan.copy()
    man["asm_id"] = man["sha256"]
    man["image_tree"] = [index[s]["tree"] for s in man["sha256"]]
    man["png"] = [index[s]["png"] for s in man["sha256"]]
    man[["sha256", "asm_id", "label", "fold", "split", "family_or_group", "family",
         "arch", "source", "kfold", "image_tree", "png"]].to_csv(
        out_dir / "manifest.csv", index=False)
    return dict(counts)


# ---------------------------------------------------------------- one run ---
def run_dir(models_root, dataset: str, scheme: str, key, seed: int) -> Path:
    stem = f"fold{key}" if scheme == "kfold" else str(key)
    return Path(models_root) / dataset / scheme / stem / f"seed{seed}"


def load_run(path: Path, expect_n: int | None = None) -> dict | None:
    """A finished run, or None. A run.json whose test set no longer matches
    the plan is treated as absent rather than trusted."""
    f = path / "run.json"
    if not f.is_file():
        return None
    try:
        doc = json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not {"ids", "y_true", "y_pred", "y_score", "epochs_run"} <= set(doc):
        return None
    if expect_n is not None and len(doc["ids"]) != expect_n:
        print(f"  stale run.json in {path} ({len(doc['ids'])} != {expect_n} test "
              f"samples); re-running")
        return None
    return doc


def train_one(mt, data_dir: Path, ckpt_dir: Path, seed: int, device: str,
              epochs: int, batch_size: int, max_samples: int, keep_weights: bool) -> dict:
    """One training run of the untuned recipe, imported wholesale from
    cnn_vit_pipeline.train_eval."""
    import torch

    t0 = time.time()
    T.seed_everything(seed)
    model, sets, index, loaders, class_names, epoch_times, history, dev = T.train(
        mt, data_dir, epochs, device, max_samples, batch_size, seed, ckpt_dir,
        quiet=True)
    y_true, y_pred, y_score = T.predict(model, loaders["test"], dev)
    ids = T.ordered_ids(sets["test"], index["test"])
    doc = {
        "ids": ids,
        "y_true": [int(v) for v in y_true],
        "y_pred": [int(v) for v in y_pred],
        "y_score": [float(v) for v in y_score],
        "class_names": list(class_names),
        "epochs_run": len(epoch_times),
        "epoch_seconds_mean": round(float(np.mean(epoch_times)), 3),
        "epoch_seconds": [round(float(x), 3) for x in epoch_times],
        "best_val_loss": min(h["val_loss"] for h in history),
        "n_train": len(sets["train"]), "n_val": len(sets["val"]),
        "n_test": len(sets["test"]),
        "seed": seed, "device": device, "device_name": T.device_name(device),
        "wall_seconds": round(time.time() - t0, 2),
    }
    (ckpt_dir / "run.json").write_text(json.dumps(doc), encoding="utf-8")
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    if not keep_weights:
        (ckpt_dir / "best_model.pth").unlink(missing_ok=True)
    return doc


# ------------------------------------------------------------ the sweeps ---
def sweep(dataset: str, a, mt) -> None:
    """Every run this dataset needs, sequentially, skipping finished ones."""
    tbl = load_fold_table(Path(a.folds_dir), dataset)
    index = image_index(Path(a.images_root), IMAGE_TREES[dataset])
    families = sorted(tbl.loc[tbl["label"] == 1, "family"].unique())
    if a.families:
        families = [f for f in families if f in set(a.families.split(","))]
    folds = sorted(tbl["kfold"].unique()) if a.folds is None else \
        [int(x) for x in a.folds.split(",")]

    jobs = []
    if a.stage in ("kfold", "all"):
        jobs += [("kfold", k, s) for k in folds for s in a.kfold_seeds]
    if a.stage in ("lofo", "all"):
        jobs += [("lofo", f, s) for f in families for s in a.lofo_seeds]

    print(f"\n{'='*78}\n{dataset}: {len(jobs)} runs planned "
          f"({a.stage}); seeds kfold={list(a.kfold_seeds)} lofo={list(a.lofo_seeds)}\n{'='*78}")
    built: set = set()
    done = skipped = 0
    t_sweep = time.time()
    for i, (scheme, key, seed) in enumerate(jobs, 1):
        plan = plan_kfold(tbl, key) if scheme == "kfold" else plan_lofo(tbl, key)
        check_plan(plan, scheme, key)
        n_test = int((plan["fold"] == "test").sum())
        rdir = run_dir(a.models_root, dataset, scheme, key, seed)
        if load_run(rdir, n_test) is not None:
            skipped += 1
            print(f"[{i}/{len(jobs)}] {dataset} {scheme} {key} seed {seed}: "
                  f"already done, skipping")
            continue
        stem = f"fold{key}" if scheme == "kfold" else key
        data_dir = Path(a.data_root) / dataset / (
            f"fold{key}" if scheme == "kfold" else f"lofo/{key}")
        if (scheme, key) not in built:
            t0 = time.time()
            counts = materialise(plan, index, data_dir)
            built.add((scheme, key))
            print(f"  built {data_dir} in {time.time()-t0:.1f}s: {counts}")
        rdir.mkdir(parents=True, exist_ok=True)
        el = time.time() - t_sweep
        print(f"[{i}/{len(jobs)}] {dataset} {scheme} {stem} seed {seed}  "
              f"(train {int((plan['fold']=='train').sum())} / "
              f"val {int((plan['fold']=='val').sum())} / test {n_test}; "
              f"elapsed {el/60:.1f} min)")
        doc = train_one(mt, data_dir, rdir, seed, a.device, a.epochs,
                        a.batch_size, a.max_samples, a.keep_weights)
        done += 1
        print(f"    -> {doc['epochs_run']} epochs, {doc['wall_seconds']:.0f}s, "
              f"{doc['epoch_seconds_mean']:.2f}s/epoch")
        if scheme == "lofo" and not a.keep_lofo_trees:
            shutil.rmtree(data_dir, ignore_errors=True)
            built.discard((scheme, key))
    print(f"{dataset}: {done} runs trained, {skipped} skipped, "
          f"{(time.time()-t_sweep)/60:.1f} min")


# --------------------------------------------------------------- scoring ---
def _pstdev(v):
    """sample sd (ddof=1): the one definition every summary in this repo uses"""
    return float(st.stdev(v)) if len(v) > 1 else 0.0


def _mean(v):
    return float(st.mean(v)) if v else None


def collect(dataset: str, a):
    """Every finished run.json for this dataset, as tidy frames.

    Returns (tbl, kfold_runs, lofo_runs) where a run is
    (scheme, key, seed, DataFrame[sha256, y_true, y_pred, score], doc).
    """
    tbl = load_fold_table(Path(a.folds_dir), dataset)
    meta = tbl.set_index("sha256")
    kfold_runs, lofo_runs = [], []
    for k in sorted(tbl["kfold"].unique()):
        for s in a.kfold_seeds:
            doc = load_run(run_dir(a.models_root, dataset, "kfold", k, s),
                           int((tbl["kfold"] == k).sum()))
            if doc is None:
                continue
            df = pd.DataFrame({"sha256": doc["ids"], "y_true": doc["y_true"],
                               "y_pred": doc["y_pred"], "score": doc["y_score"]})
            kfold_runs.append(("kfold", int(k), int(s), df, doc))
    for f in sorted(tbl.loc[tbl["label"] == 1, "family"].unique()):
        for s in a.lofo_seeds:
            doc = load_run(run_dir(a.models_root, dataset, "lofo", f, s),
                           int((tbl["family"] == f).sum()))
            if doc is None:
                continue
            df = pd.DataFrame({"sha256": doc["ids"], "y_true": doc["y_true"],
                               "y_pred": doc["y_pred"], "score": doc["y_score"]})
            lofo_runs.append(("lofo", f, int(s), df, doc))
    return tbl, meta, kfold_runs, lofo_runs


def floors(y_true, arch) -> dict:
    """majority: always the larger test class. x86_rule: ransomware iff x86.
    Both as accuracy (the `majority` / `machine` columns of COMPARISON.md);
    the x86 rule's macro-F1 is carried alongside for the record."""
    y_true = np.asarray(y_true).astype(int)
    arch = np.asarray(arch, dtype=object)
    maj = int(np.bincount(y_true, minlength=2).argmax())
    m_maj = C._core_metrics(y_true, np.full(len(y_true), maj))
    m_x86 = C._core_metrics(y_true, (arch == "x86").astype(int))
    return {"majority_accuracy": m_maj["accuracy"],
            "majority_macro_f1": m_maj["macro_f1"],
            "x86_rule_accuracy": m_x86["accuracy"],
            "x86_rule_macro_f1": m_x86["macro_f1"],
            "x86_rule_recall_ransomware": m_x86["recall_ransomware"],
            "x86_rule_recall_goodware": m_x86["recall_goodware"]}


def _arch_cell(res: dict, arch: str, key: str, support_key: str):
    block = res.get("per_architecture", {}).get(arch)
    if not block or not block.get(support_key):
        return None
    return block[key]


def score_dataset(dataset: str, a) -> dict | None:
    """Everything results/family_holdout/<ds>/cnn_vit/<model>/ holds."""
    tbl, meta, kruns, lruns = collect(dataset, a)
    if not kruns:
        print(f"{dataset}: no finished K-fold runs; nothing to write")
        return None
    out = Path(a.out_root) / dataset / PIPELINE / MODEL_NAME
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---- pooled predictions from seed-averaged scores --------------------
    per_run = []
    for _, k, s, df, doc in kruns:
        d = df.copy()
        d["fold"] = k
        d["seed"] = s
        per_run.append(d)
    allk = pd.concat(per_run, ignore_index=True)
    pooled = (allk.groupby(["sha256", "fold"], as_index=False)
                  .agg(score=("score", "mean"), n_seeds=("score", "size")))
    pooled["label"] = pooled["sha256"].map(meta["label"])
    pooled["family"] = pooled["sha256"].map(meta["family"])
    pooled["arch"] = pooled["sha256"].map(meta["arch"])
    pooled["pred"] = (pooled["score"] >= 0.5).astype(int)
    pooled = pooled.sort_values(["fold", "label", "family", "sha256"]).reset_index(drop=True)
    pooled[["sha256", "label", "family", "arch", "fold", "score", "pred"]].round(
        {"score": 6}).to_csv(out / "predictions.csv", index=False)

    n_seeds = sorted(set(pooled["n_seeds"]))
    complete = len(pooled) == len(tbl) and n_seeds == [len(a.kfold_seeds)]

    pooled_result = C.build_result(
        pooled["label"].values, pooled["pred"].values, pooled["score"].values,
        pooled["arch"].values, pooled["family"].values,
        model=f"{MODEL_NAME} (CNN-ViT), untuned, pooled over 5 held-out folds, "
              f"scores averaged over {len(a.kfold_seeds)} seeds")

    # ---- per-fold rows ---------------------------------------------------
    fold_rows = []
    for k in sorted(allk["fold"].unique()):
        part = tbl[tbl["kfold"] == k]
        y_arch = part["arch"].values
        fl = floors(part["label"].values, y_arch)
        seed_res, epochs = [], []
        for _, kk, s, df, doc in kruns:
            if kk != k:
                continue
            d = df.copy()
            d["family"] = d["sha256"].map(meta["family"])
            d["arch"] = d["sha256"].map(meta["arch"])
            seed_res.append(C.build_result(d["y_true"].values, d["y_pred"].values,
                                           d["score"].values, d["arch"].values,
                                           d["family"].values))
            epochs.append(doc["epochs_run"])
        row = {"fold": int(k), "n_test": len(part),
               "n_good": int((part["label"] == 0).sum()),
               "n_rans": int((part["label"] == 1).sum())}
        for key in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc",
                    "recall_ransomware", "recall_goodware"):
            vals = [r[key] for r in seed_res if r[key] is not None]
            row[key] = _mean(vals)
        row["fpr"] = _mean([r["false_positive_rate"] for r in seed_res])
        for arch in ("x86", "x64"):
            for cls, sup in (("ransomware", "support_ransomware"),
                             ("goodware", "support_goodware")):
                vals = [_arch_cell(r, arch, f"recall_{cls}", sup) for r in seed_res]
                vals = [v for v in vals if v is not None]
                row[f"{arch}_recall_{cls}"] = _mean(vals)
        row["majority_floor"] = fl["majority_accuracy"]
        row["x86_rule_floor"] = fl["x86_rule_accuracy"]
        row["macro_f1_sd_over_seeds"] = _pstdev([r["macro_f1"] for r in seed_res])
        row["epochs_run_mean"] = _mean(epochs)
        row["n_seeds"] = len(seed_res)
        fold_rows.append(row)
    fold_df = pd.DataFrame(fold_rows)
    fold_cols = ["fold", "n_test", "n_good", "n_rans", "accuracy",
                 "balanced_accuracy", "macro_f1", "roc_auc", "recall_ransomware",
                 "recall_goodware", "fpr", "x86_recall_ransomware",
                 "x86_recall_goodware", "x64_recall_ransomware",
                 "x64_recall_goodware", "majority_floor", "x86_rule_floor",
                 "macro_f1_sd_over_seeds", "epochs_run_mean"]
    fold_df[fold_cols].round(6).to_csv(out / "fold_metrics.csv", index=False)

    # ---- LOFO ------------------------------------------------------------
    lofo_rows, lofo_recall = [], {}
    for _, fam, s, df, doc in lruns:
        d = df.copy()
        d["family"] = fam
        d["arch"] = d["sha256"].map(meta["arch"])
        lofo_rows.append(d)
    if lofo_rows:
        lofo = pd.concat(lofo_rows, ignore_index=True)
        lofo = (lofo.groupby(["sha256", "family"], as_index=False)
                    .agg(score=("score", "mean")))
        lofo["arch"] = lofo["sha256"].map(meta["arch"])
        lofo["pred"] = (lofo["score"] >= 0.5).astype(int)
        lofo = lofo.sort_values(["family", "sha256"]).reset_index(drop=True)
        lofo[["sha256", "family", "arch", "score", "pred"]].round(
            {"score": 6}).to_csv(out / "lofo_predictions.csv", index=False)
        lofo_recall = lofo.groupby("family")["pred"].mean().to_dict()
        lofo_arch = {
            arch: float(part["pred"].mean())
            for arch, part in lofo.groupby("arch")}
    else:
        lofo = pd.DataFrame(columns=["sha256", "family", "arch", "score", "pred"])
        lofo_arch = {}

    # ---- per family ------------------------------------------------------
    rans = tbl[tbl["label"] == 1]
    kf_recall = (pooled[pooled["label"] == 1].groupby("family")["pred"].mean().to_dict())
    fam_rows = []
    for fam, part in rans.groupby("family"):
        fam_rows.append({
            "family": fam, "n": len(part),
            "n_x64": int((part["arch"] == "x64").sum()),
            "fold": int(part["kfold"].iloc[0]),
            "recall_kfold": kf_recall.get(fam),
            "recall_lofo": lofo_recall.get(fam)})
    fam_df = pd.DataFrame(fam_rows).sort_values(["fold", "family"])
    fam_df.round(6).to_csv(out / "per_family.csv", index=False)

    # ---- metrics.json ----------------------------------------------------
    fold_mean, fold_sd = {}, {}
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc"):
        vals = [r[key] for r in fold_rows if r[key] is not None]
        fold_mean[key] = _mean(vals)
        fold_sd[key] = _pstdev(vals)

    epoch_secs = [doc["epoch_seconds_mean"] for *_, doc in kruns + lruns]
    wall = sum(doc["wall_seconds"] for *_, doc in kruns + lruns)
    dev_names = sorted({doc["device_name"] for *_, doc in kruns + lruns})
    pooled_floors = floors(pooled["label"].values, pooled["arch"].values)

    samples = {
        "total": int(len(tbl)),
        "goodware": int((tbl["label"] == 0).sum()),
        "ransomware": int((tbl["label"] == 1).sum()),
        "families": int(rans["family"].nunique()),
        "goodware_groups": int(tbl.loc[tbl["label"] == 0, "group"].nunique()),
        "arch": tbl["arch"].value_counts().to_dict(),
        "per_fold": {int(k): {"n": int(len(p)),
                              "goodware": int((p["label"] == 0).sum()),
                              "ransomware": int((p["label"] == 1).sum()),
                              "x64_ransomware": int(((p["label"] == 1) &
                                                     (p["arch"] == "x64")).sum()),
                              "families": sorted(p.loc[p["label"] == 1, "family"].unique())}
                     for k, p in tbl.groupby("kfold")},
        "kfold_predictions": int(len(pooled)),
        "every_sample_held_out_once": bool(complete),
        "seeds_per_sample": n_seeds,
    }
    cfg = build_config(dataset, a, dev_names)
    C.write_metrics(
        out / "metrics.json",
        experiment=f"family_holdout_cnn_vit_{dataset}",
        description=(
            f"CNN-ViT ({MODEL_NAME}) under family holdout: every one of the 38 "
            f"in-cohort ransomware families is held out exactly once in the "
            f"5-fold scheme of family_holdout/folds.py, and once more on its own "
            f"in LOFO. Untuned recipe, imported verbatim from "
            f"cnn_vit_pipeline/train_eval.py (batch 16, AdamW 3e-4 + 5 warmup "
            f"epochs + cosine, weighted sampling, MaskAwareStructuralShift on "
            f"train only, early stopping on val loss patience 8, max 80 epochs, "
            f"argmax at 0.5). No tuning, no threshold search. `results[0]` is "
            f"the pooled prediction: every sample scored once, in the fold where "
            f"it was held out, by the mean of {len(a.kfold_seeds)} seeds' scores. "
            f"fold_mean / fold_sd are over the five folds of the per-fold "
            f"seed-mean. LOFO has no goodware in its test sets, so it reports "
            f"recall only."),
        samples=samples,
        results=[pooled_result],
        elapsed_seconds=time.time() - t_start,
        dataset=dataset,
        pipeline=PIPELINE,
        model=MODEL_NAME,
        scheme="family_holdout_5fold + LOFO",
        fold_mean=fold_mean,
        fold_sd=fold_sd,
        fold_sd_kind="sample sd (ddof=1) over the 5 folds",
        lofo_mean_recall=(_mean(list(lofo_recall.values())) if lofo_recall else None),
        lofo_families_done=len(lofo_recall),
        lofo_pooled_recall=(float(lofo["pred"].mean()) if len(lofo) else None),
        lofo_recall_by_arch=lofo_arch,
        floors_pooled=pooled_floors,
        kfold_seeds=list(a.kfold_seeds),
        lofo_seeds=list(a.lofo_seeds),
        runs={"kfold": len(kruns), "lofo": len(lruns)},
        device=a.device,
        device_name=dev_names[0] if dev_names else None,
        seconds_per_epoch_mean=round(float(np.mean(epoch_secs)), 2) if epoch_secs else None,
        seconds_per_epoch_median=round(float(np.median(epoch_secs)), 2) if epoch_secs else None,
        train_seconds_total=round(wall, 1),
        epochs_run_mean=round(float(np.mean([doc["epochs_run"] for *_, doc in kruns + lruns])), 2),
        config=cfg,
    )
    (out / "config_used.yaml").write_text(T.to_yaml(cfg), encoding="utf-8")
    print(f"{dataset}: wrote {out}")
    if not complete:
        print(f"  NOTE incomplete: {len(pooled)}/{len(tbl)} samples have a "
              f"held-out prediction, seeds per sample {n_seeds}")
    return {"dataset": dataset, "out": out, "fold_df": fold_df,
            "pooled": pooled_result, "fold_mean": fold_mean, "fold_sd": fold_sd,
            "fam_df": fam_df, "lofo_recall": lofo_recall, "lofo_arch": lofo_arch,
            "samples": samples, "complete": complete,
            "lofo_pooled_recall": (float(lofo["pred"].mean()) if len(lofo) else None),
            "floors": pooled_floors, "n_runs": len(kruns) + len(lruns),
            "n_kfold_runs": len(kruns), "n_lofo_runs": len(lruns),
            "n_kfold_seeds": len(a.kfold_seeds), "n_lofo_seeds": len(a.lofo_seeds),
            "wall": wall, "epoch_secs": epoch_secs,
            "epochs": [doc["epochs_run"] for *_, doc in kruns + lruns]}


def build_config(dataset: str, a, dev_names) -> dict:
    return {
        "pipeline": PIPELINE,
        "study": "family_holdout",
        "dataset": dataset,
        "model": {"class": MODEL_NAME, "defined_in": "CNN-ViT/model_train.py",
                  "num_classes": T.NUM_CLASSES, "dropout": T.DROPOUT,
                  "input": "256x256 uint8 token image + 16x16 ViT patch mask"},
        "folds": {"source": "family_holdout/folds.py",
                  "file": f"results/family_holdout/folds_{dataset}.csv",
                  "k": 5,
                  "ransomware_groups": "family (assigned whole to a fold)",
                  "goodware_groups": ("mnemonic-stream sha256" if dataset == "mendeley"
                                      else "source project (entry_id)"),
                  "val": "group-aware 10% carved from train by "
                         "cnn_vit_pipeline.cohort.add_val_fold",
                  "val_fraction": C.VAL_FRACTION, "val_seed": C.VAL_SEED},
        "training": {"recipe": "untuned, imported from cnn_vit_pipeline/train_eval.py",
                     "batch_size": a.batch_size,
                     "base_learning_rate": T.BASE_LEARNING_RATE,
                     "warmup_epochs": T.WARMUP_EPOCHS,
                     "weight_decay": T.WEIGHT_DECAY,
                     "label_smoothing": T.LABEL_SMOOTHING,
                     "optimizer": "AdamW",
                     "scheduler": "CosineAnnealingLR after warmup, eta_min 1e-6",
                     "grad_clip_max_norm": 1.0,
                     "sampler": "WeightedRandomSampler (inverse class frequency)",
                     "augmentation": "MaskAwareStructuralShift(p=0.4, "
                                     "max_shift_ratio=0.20), train fold only",
                     "early_stopping": {"monitor": "val_loss", "patience": 8,
                                        "min_delta": 0.0001},
                     "max_epochs": a.epochs,
                     "decision": "argmax (score >= 0.5); no threshold search",
                     "kfold_seeds": list(a.kfold_seeds),
                     "lofo_seeds": list(a.lofo_seeds),
                     "device": a.device,
                     "device_name": dev_names[0] if dev_names else None},
        "aggregation": {"per_fold": "mean over seeds of the per-seed metric",
                        "pooled": "argmax of the seed-mean score, over the union "
                                  "of the five held-out folds",
                        "fold_sd": "sample sd (ddof=1) over the five folds"},
        "floors": {"majority": "accuracy of always predicting the larger test class",
                   "x86_rule": "accuracy of predicting ransomware iff arch == x86"},
        "paths": {"images": str(Path(a.images_root)),
                  "image_trees": list(IMAGE_TREES[dataset]),
                  "data": str(Path(a.data_root) / dataset),
                  "weights": str(Path(a.models_root) / dataset),
                  "results": str(Path(a.out_root) / dataset / PIPELINE / MODEL_NAME)},
    }


# --------------------------------------------------------------- summary ---
def f(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.{nd}f}"


FIXED_SPLIT = {"mendeley": (0.628, 0.018), "balanced": (0.502, 0.113)}


def write_summary(results: list, out_root: Path) -> Path:
    n_seeds = results[0]["n_kfold_seeds"] if results else 3
    n_lofo = results[0]["n_lofo_seeds"] if results else 1
    L = ["# CNN-ViT under family holdout", "",
         f"`{MODEL_NAME}` (CNN-ViT), untuned recipe, evaluated with every one of the 38 in-cohort ransomware",
         "families held out in turn. Fold definition: `family_holdout/folds.py` (K = 5, families assigned whole,",
         "goodware assigned by duplicate-stream group on Mendeley and by source project on Goodware_Balanced).",
         f"{n_seeds} seed(s) per fold; a fold's number is the mean over its seeds, `pooled` scores the union of the",
         "five held-out folds using the seed-mean score at argmax. LOFO trains on the other 37 families plus all",
         f"goodware and tests on the held-out family alone - {n_lofo} seed, recall only (no goodware in its test set).",
         "",
         "Floors are per-fold accuracies: `majority` = always predict the larger class of that fold, `x86 rule` =",
         "predict ransomware iff the PE machine field says x86. Nothing here is tuned and no threshold is moved.",
         "",
         f"Fold mean and pooled are not the same quantity. A fold's number is one model's; the pooled column",
         f"scores every file once from the mean of {n_seeds} seeds' scores, which is a {n_seeds}-seed ensemble and",
         "is reliably the higher of the two. The fold mean +/- sd is the like-for-like comparison with the",
         "single-model fixed-split number; the pooled column is what the per-family and per-architecture",
         "breakdowns below are computed from, because every file has exactly one held-out prediction there.",
         ""]
    for r in results:
        ds = r["dataset"]
        fm, fsd = r["fold_mean"], r["fold_sd"]
        p = r["pooled"]
        fix_mu, fix_sd = FIXED_SPLIT[ds]
        L += [f"## {ds}", "",
              f"{r['samples']['total']} files: {r['samples']['ransomware']} ransomware in "
              f"{r['samples']['families']} families, {r['samples']['goodware']} goodware in "
              f"{r['samples']['goodware_groups']} groups. "
              f"{r['n_runs']} training runs ({r['n_kfold_runs']} K-fold, "
              f"{r['n_lofo_runs']} LOFO).", "",
              "| metric | fold mean +/- sd (5 folds) | pooled | fixed split (5 seeds) |",
              "|---|---|---|---|",
              f"| macro-F1 | {f(fm['macro_f1'])} +/- {f(fsd['macro_f1'])} | {f(p['macro_f1'])} | {fix_mu:.3f} +/- {fix_sd:.3f} |",
              f"| balanced accuracy | {f(fm['balanced_accuracy'])} +/- {f(fsd['balanced_accuracy'])} | {f(p['balanced_accuracy'])} | - |",
              f"| accuracy | {f(fm['accuracy'])} +/- {f(fsd['accuracy'])} | {f(p['accuracy'])} | - |",
              f"| ROC AUC | {f(fm['roc_auc'])} +/- {f(fsd['roc_auc'])} | {f(p['roc_auc'])} | - |",
              ""]
        fd = r["fold_df"]
        L += ["Per fold:", "",
              "| fold | n | good | rans | macro-F1 (seed mean) | sd over seeds | bal-acc | AUC | recall R / G | FPR | "
              "majority floor | x86-rule floor | epochs |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, row in fd.iterrows():
            L.append(
                f"| {int(row['fold'])} | {int(row['n_test'])} | {int(row['n_good'])} | {int(row['n_rans'])} | "
                f"{f(row['macro_f1'])} | {f(row['macro_f1_sd_over_seeds'])} | {f(row['balanced_accuracy'])} | "
                f"{f(row['roc_auc'])} | {f(row['recall_ransomware'], 2)} / {f(row['recall_goodware'], 2)} | "
                f"{f(row['fpr'], 2)} | {f(row['majority_floor'])} | {f(row['x86_rule_floor'])} | "
                f"{f(row['epochs_run_mean'], 1)} |")
        fl = r["floors"]
        d_fixed = fm["macro_f1"] - fix_mu
        d_x86 = p["macro_f1"] - fl["x86_rule_macro_f1"]
        ratio = (abs(d_fixed) / fsd["macro_f1"]) if fsd["macro_f1"] else float("inf")
        L += ["",
              f"Pooled floors over all {r['samples']['total']} held-out predictions: majority "
              f"{fl['majority_accuracy']:.3f} accuracy (macro-F1 {fl['majority_macro_f1']:.3f}), x86 rule "
              f"{fl['x86_rule_accuracy']:.3f} accuracy (macro-F1 {fl['x86_rule_macro_f1']:.3f}, recall "
              f"{fl['x86_rule_recall_ransomware']:.2f} ransomware / {fl['x86_rule_recall_goodware']:.2f} "
              f"goodware).",
              "",
              f"Against the fixed split: the fold mean is {d_fixed:+.3f} macro-F1 from the single 24/14 split's "
              f"{fix_mu:.3f}, which is {ratio:.1f}x the fold sd "
              f"({fsd['macro_f1']:.3f}) - "
              + ("the one split was not representative of the cohort's families."
                 if abs(d_fixed) > fsd["macro_f1"] else
                 "the one split sat inside the spread over held-out family sets."),
              "",
              (f"Against the architecture shortcut: pooled macro-F1 {p['macro_f1']:.3f} is {d_x86:+.3f} against the "
               f"x86 rule's {fl['x86_rule_macro_f1']:.3f}. "
               + ("**Below the floor** - a classifier that reads nothing but the PE machine field scores higher "
                  "than the image encoder here, so nothing in this dataset's number demonstrates that the token "
                  "images carry the detection."
                  if d_x86 < 0 else
                  "The encoder clears the floor, so it is reading more than the machine field.")), ""]
        pa = p.get("per_architecture", {})
        L += ["Per architecture, pooled:", "",
              "| arch | n | good | rans | accuracy | macro-F1 | recall ransomware | recall goodware | FPR | LOFO recall |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for arch in sorted(pa):
            b = pa[arch]
            L.append(f"| {arch} | {b['n']} | {b['support_goodware']} | {b['support_ransomware']} | "
                     f"{f(b['accuracy'])} | {f(b['macro_f1'])} | {f(b['recall_ransomware'])} | "
                     f"{f(b['recall_goodware'])} | {f(b['false_positive_rate'])} | "
                     f"{f(r['lofo_arch'].get(arch))} |")
        L.append("")
        lof = r["lofo_recall"]
        if lof:
            L.append(f"LOFO mean recall over {len(lof)} families: "
                     f"{_mean(list(lof.values())):.3f} unweighted (each family counts once); "
                     f"file-weighted {f(r['lofo_pooled_recall'])}. See `lofo_predictions.csv`.")
            L.append("")
        fam = r["fam_df"].copy()
        fam["sort"] = fam["recall_kfold"].fillna(-1)
        fam = fam.sort_values(["sort", "family"])
        L += ["Per-family recall (K-fold held-out, LOFO):", "",
              "| family | n | x64 | fold | K-fold recall | LOFO recall |",
              "|---|---|---|---|---|---|"]
        for _, row in fam.iterrows():
            L.append(f"| {row['family']} | {int(row['n'])} | {int(row['n_x64'])} | {int(row['fold'])} | "
                     f"{f(row['recall_kfold'], 2)} | {f(row['recall_lofo'], 2)} |")
        L += ["", "Sorted by K-fold recall; the families at the top are the ones this encoder never sees.", ""]
        both = fam[(fam["recall_kfold"] < 0.25) & (fam["recall_lofo"] < 0.25)]
        zero = fam[fam["recall_kfold"] == 0]
        half = fam[fam["recall_kfold"] < 0.5]
        gap = fam.assign(d=(fam["recall_lofo"] - fam["recall_kfold"]).abs()) \
                 .sort_values("d", ascending=False).head(3)
        L += [f"{len(half)} of {len(fam)} families are below 0.5 K-fold recall and {len(zero)} are at zero "
              f"({', '.join(zero['family']) or 'none'}). "
              f"{len(both)} are missed by both schemes ({', '.join(both['family']) or 'none'}) - those are the "
              f"ones neither more training data nor a different fold rescues.",
              "",
              "LOFO is one seed per family, so a large K-fold/LOFO disagreement is not by itself evidence that "
              "training-set size mattered: " +
              "; ".join(f"{r['family']} {f(r['recall_kfold'], 2)} -> {f(r['recall_lofo'], 2)}"
                        for _, r in gap.iterrows()) +
              " are the three widest here, and the per-fold seed sd above is already up to "
              f"{fd['macro_f1_sd_over_seeds'].max():.3f} macro-F1 on three seeds of the same fold.", ""]
        secs = r["epoch_secs"]
        L += [f"Runtime: {r['n_runs']} runs, {r['wall']/60:.1f} min of training wall clock, "
              f"{np.mean(secs):.2f} s/epoch mean, {np.mean(r['epochs']):.1f} epochs per run "
              f"(early stopping, patience 8 of a possible {MAX_EPOCHS}).", ""]

    L += ["## Reading this against the fixed split", "",
          "The fixed 24/14 family split gave this encoder 0.63 +/- 0.02 macro-F1 on Mendeley and 0.50 +/- 0.11 on",
          "Goodware_Balanced (5 seeds, `results/COMPARISON.md`). Those sds are over seeds on one split; the fold sd",
          "above is over five different choices of held-out families, which is the quantity that says whether the",
          "single split was lucky.", "",
          "Caveat on the fold sd: x64 ransomware is concentrated in fold 2 (59 of 114 x64 ransomware files; Hive",
          "alone is 43), so part of the spread across folds measures architecture mix rather than family",
          "difficulty. The per-architecture table above is the check on that.", "",
          "Produced by `family_holdout/run_cnn_vit.py`; per-fold numbers in `fold_metrics.csv`, every held-out",
          "prediction in `predictions.csv`, LOFO in `lofo_predictions.csv`.", ""]
    path = out_root / "summary_cnn_vit.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path}")
    return path


# ------------------------------------------------------------------ main ---
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="both", choices=(*DATASETS, "both"))
    ap.add_argument("--stage", default="all", choices=("kfold", "lofo", "all"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=T.BATCH_SIZE)
    ap.add_argument("--kfold-seeds", default=",".join(str(s) for s in KFOLD_SEEDS))
    ap.add_argument("--lofo-seeds", default=",".join(str(s) for s in LOFO_SEEDS))
    ap.add_argument("--folds-dir", default=str(FOLDS_DIR))
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--images-root", default=str(IMAGES_ROOT))
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--models-root", default=str(MODELS_ROOT))
    ap.add_argument("--folds", default=None, help="subset of K-folds, e.g. 0,1")
    ap.add_argument("--families", default=None, help="subset of LOFO families")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--keep-weights", action="store_true", default=True)
    ap.add_argument("--no-keep-weights", dest="keep_weights", action="store_false")
    ap.add_argument("--keep-lofo-trees", action="store_true", default=False,
                    help="keep the 38 LOFO image trees instead of removing each "
                         "one once its run has finished")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the CSVs, metrics.json and the summary from "
                         "the run.json files already on disk; no training")
    ap.add_argument("--no-summary", action="store_true")
    a = ap.parse_args(argv)

    a.kfold_seeds = tuple(int(s) for s in a.kfold_seeds.split(",") if s != "")
    a.lofo_seeds = tuple(int(s) for s in a.lofo_seeds.split(",") if s != "")

    datasets = list(DATASETS) if a.dataset == "both" else [a.dataset]
    if not a.report_only:
        mt = T.import_model_train()
        for ds in datasets:
            sweep(ds, a, mt)

    results = [r for r in (score_dataset(ds, a) for ds in datasets) if r]
    if results and not a.no_summary:
        write_summary(results, Path(a.out_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
