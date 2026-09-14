#!/usr/bin/env python3
"""Materialise a CNN-ViT train/val/test tree that obeys the shared cohort split.

The original yanping protocol built its splits with stratified_split.py, which
shuffles each class folder at random. That puts the same ransomware family -
often the same binary under two names - on both sides, and it keeps .NET,
packed and UPX samples that the other pipelines exclude. This script instead
takes the split from cnn_vit_pipeline/cohort.py, the same one the tokenization
and EMBER harnesses use, so the three sets of numbers describe the same
samples.

What it does
------------
1. For each image tree, map every PNG back to a sha256. The PNG stem is
   asm_parser.asm_id() of the .asm file, and asm_output/<tree>/asm_manifest.csv
   carries sha256 alongside that .asm path, so the join is exact, not by name.
2. Apply the shared split, then carve a fixed-seed 10% val out of train,
   stratified by label and group-aware (no ransomware family and no goodware
   source project appears in both train and val).
3. Hard-link (or copy) each PNG and its _vit_mask.npy into
   DIR/{train,val,test}/Class_{0_Goodware,1_Ransomware}/.

Everything that does not map, and every cohort row with no image yet, is
reported rather than dropped in silence.

    python cnn_vit_pipeline/build_dataset.py --dataset mendeley --out DIR
    python cnn_vit_pipeline/build_dataset.py --dataset balanced --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import asm_parser  # noqa: E402  (imported, never modified; import is side-effect free)
from cnn_vit_pipeline import cohort as C  # noqa: E402

ASM_ROOT = Path(os.environ.get("RANSOM_ASM_OUTPUT", REPO_ROOT.parent / "asm_output"))
IMAGE_ROOT = Path(os.environ.get("RANSOM_CNN_VIT_IMAGES",
                                 REPO_ROOT.parent / "cnn_vit_images"))
VM_RUNBOOK = "vm_package/README.md"

CLASS_DIRS = {0: "Class_0_Goodware", 1: "Class_1_Ransomware"}

# (image tree, asm tree, label, produced on)
TREES = {
    "mendeley": [
        ("mendeley_goodware", "mendeley_goodware", 0, "host"),
        ("mendeley_goodware_test", "mendeley_goodware_test", 0, "VM"),
        ("mendeley_ransomware_train", "mendeley_ransomware_train", 1, "VM"),
        ("mendeley_ransomware_test", "mendeley_ransomware_test", 1, "VM"),
    ],
    "balanced": [
        ("goodware_balanced", "goodware_balanced", 0, "host"),
        ("mendeley_ransomware_train", "mendeley_ransomware_train", 1, "VM"),
        ("mendeley_ransomware_test", "mendeley_ransomware_test", 1, "VM"),
    ],
}


# ---------------------------------------------------------------- mapping ---
def index_tree(image_tree: str, asm_tree: str) -> tuple[pd.DataFrame, list[dict]]:
    """Every PNG under an image tree, keyed by sha256.

    Returns (frame, unmapped). `unmapped` holds PNGs whose asm_id is not in the
    tree's asm_manifest.csv - they would be silent losses otherwise.
    """
    img_dir = IMAGE_ROOT / image_tree
    asm_dir = ASM_ROOT / asm_tree
    manifest = asm_dir / "asm_manifest.csv"
    if not img_dir.is_dir() or not manifest.exists():
        return pd.DataFrame(), []

    man = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    id_to_sha, id_to_status = {}, {}
    for _, r in man.iterrows():
        if not r["asm_path"]:
            continue  # extraction produced no .asm for this input
        key = asm_parser.asm_id(asm_dir / r["asm_path"], asm_dir)
        id_to_sha[key] = r["sha256"].strip().lower()
        id_to_status[key] = r.get("status", "")

    rows, unmapped = [], []
    for png in sorted(img_dir.rglob("*.png")):
        stem = png.stem
        sha = id_to_sha.get(stem)
        if sha is None:
            unmapped.append({"image_tree": image_tree, "png": str(png),
                             "asm_id": stem, "reason": "asm_id not in asm_manifest"})
            continue
        mask = png.with_name(png.stem + "_vit_mask.npy")
        rows.append({
            "sha256": sha,
            "asm_id": stem,
            "png": str(png),
            "mask": str(mask) if mask.exists() else "",
            "image_tree": image_tree,
            "asm_status": id_to_status.get(stem, ""),
        })
    return pd.DataFrame(rows), unmapped


def index_dataset(dataset: str):
    frames, unmapped, present, missing = [], [], [], []
    for image_tree, asm_tree, label, origin in TREES[dataset]:
        df, un = index_tree(image_tree, asm_tree)
        unmapped.extend(un)
        if len(df):
            df["tree_label"] = label
            frames.append(df)
            present.append((image_tree, len(df)))
        else:
            missing.append((image_tree, asm_tree, origin))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return frame, unmapped, present, missing


# ------------------------------------------------------------ materialise ---
def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "link":
        try:
            os.link(src, dst)
            return "link"
        except OSError:
            pass  # different volume, or a filesystem without hard links
    shutil.copy2(src, dst)
    return "copy"


def materialise(df: pd.DataFrame, out: Path, mode: str) -> dict:
    counts = {"link": 0, "copy": 0, "mask_missing": 0}
    for fold in ("train", "val", "test"):
        for label, cls in CLASS_DIRS.items():
            (out / fold / cls).mkdir(parents=True, exist_ok=True)
    for _, r in df.iterrows():
        cls = CLASS_DIRS[int(r["label"])]
        dest = out / r["fold"] / cls
        counts[link_or_copy(Path(r["png"]), dest / f"{r['asm_id']}.png", mode)] += 1
        if r["mask"]:
            link_or_copy(Path(r["mask"]), dest / f"{r['asm_id']}_vit_mask.npy", mode)
        else:
            counts["mask_missing"] += 1
    return counts


# ------------------------------------------------------------------- main ---
def build(dataset: str, out: Path, mode: str, clean: bool) -> int:
    split = C.load_split(dataset)
    frame, unmapped, present, missing = index_dataset(dataset)

    print(f"=== build_dataset --dataset {dataset} ===")
    print(f"image trees under {IMAGE_ROOT}")
    for tree, n in present:
        print(f"  present  {tree:<28} {n:5d} PNGs")
    for tree, asm_tree, origin in missing:
        print(f"  MISSING  {tree:<28}   (images built from "
              f"asm_output/{asm_tree}, produced on the {origin})")

    if frame.empty:
        print("\nno images at all; nothing to build.", file=sys.stderr)
        return 1

    dup = frame["sha256"].duplicated(keep="first")
    dup_rows = frame.loc[dup, ["sha256", "png"]].to_dict("records")
    frame = frame.loc[~dup].reset_index(drop=True)
    print(f"\n{len(frame)} PNGs with a unique sha256 "
          f"({len(dup_rows)} duplicate-sha PNGs collapsed, "
          f"{len(unmapped)} unmapped to any sha256)")

    joined = frame.merge(split, on="sha256", how="inner")
    mismatch = joined[joined["tree_label"] != joined["label"]]
    if len(mismatch):
        print(f"ERROR: {len(mismatch)} images whose tree label disagrees with "
              f"the cohort label; refusing to build.", file=sys.stderr)
        print(mismatch[["sha256", "png", "tree_label", "label"]].head().to_string(),
              file=sys.stderr)
        return 3

    joined = C.add_val_fold(joined)
    print(f"joined to the shared split: {len(joined)}/{len(split)} cohort rows "
          f"have an image")

    gap = C.explain_missing(split, set(frame["sha256"]),
                            on_disk_names=set(_tree_filenames(dataset)))
    print("\nfold counts:")
    print(joined.groupby(["fold", "label"]).size().to_string())
    print("\nper-arch:")
    print(joined.groupby(["fold", "arch"]).size().to_string())

    print(f"\ncohort rows with no image yet: {gap['cohort_rows_missing']}")
    for k, v in sorted(gap["missing_by_split_source"].items()):
        print(f"  {k:<30} {v}")
    if gap["sha_mismatch_on_disk"]:
        print(f"  NOTE {gap['sha_mismatch_on_disk']} of those ARE in an "
              f"extracted tree but hash differently than the cohort CSV "
              f"(Mendeley goodware: cohort holds the UPX-unpacked sha256, "
              f"asm_parse ran on the packed original).")
        print(f"       e.g. {', '.join(gap['sha_mismatch_examples'][:4])}")

    if clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    counts = materialise(joined, out, mode)
    print(f"\nmaterialised into {out}: {counts['link']} hard-linked, "
          f"{counts['copy']} copied, {counts['mask_missing']} without a mask")

    cols = ["sha256", "asm_id", "label", "fold", "split", "family_or_group",
            "family", "arch", "source", "image_tree", "asm_status", "png", "mask"]
    joined[cols].to_csv(out / "manifest.csv", index=False)
    if unmapped:
        pd.DataFrame(unmapped).to_csv(out / "unmapped.csv", index=False)
    if dup_rows:
        pd.DataFrame(dup_rows).to_csv(out / "duplicate_sha_pngs.csv", index=False)

    report = {
        "dataset": dataset,
        "out": str(out),
        "image_root": str(IMAGE_ROOT),
        "asm_root": str(ASM_ROOT),
        "trees_present": {t: n for t, n in present},
        "trees_missing": [{"image_tree": t, "asm_tree": a, "produced_on": o}
                          for t, a, o in missing],
        "vm_runbook": VM_RUNBOOK,
        "pngs_indexed": int(len(frame) + len(dup_rows)),
        "pngs_unmapped_to_sha256": len(unmapped),
        "duplicate_sha_pngs_collapsed": len(dup_rows),
        "samples": C.split_summary(joined, "fold"),
        "coverage": gap,
        "materialise": counts,
    }
    (out / "build_report.json").write_text(json.dumps(report, indent=2, default=str),
                                           encoding="utf-8")
    print(f"wrote {out/'manifest.csv'} and {out/'build_report.json'}")

    if missing:
        print(f"\nINCOMPLETE: {len(missing)} image tree(s) still to come from "
              f"the VM; see {VM_RUNBOOK} step 3, then re-run this command.")
        print("The tree written above holds only the classes that exist today.")
    return 0


def _tree_filenames(dataset: str) -> set[str]:
    names: set[str] = set()
    for _, asm_tree, _, _ in TREES[dataset]:
        man = ASM_ROOT / asm_tree / "asm_manifest.csv"
        if man.exists():
            df = pd.read_csv(man, dtype=str, keep_default_na=False)
            names |= set(df["rel_path"].str.rsplit("/", n=1).str[-1])
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=C.DATASETS, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["link", "copy"], default="link",
                    help="hard-link (default) or copy the PNG/mask pairs")
    ap.add_argument("--clean", action="store_true",
                    help="remove --out first")
    ap.add_argument("--images-root", default=None)
    ap.add_argument("--asm-root", default=None)
    a = ap.parse_args()

    global IMAGE_ROOT, ASM_ROOT
    if a.images_root:
        IMAGE_ROOT = Path(a.images_root)
    if a.asm_root:
        ASM_ROOT = Path(a.asm_root)
    return build(a.dataset, Path(a.out), a.mode, a.clean)


if __name__ == "__main__":
    raise SystemExit(main())
