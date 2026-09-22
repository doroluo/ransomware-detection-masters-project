#!/usr/bin/env python3
"""EMBER-branch LightGBM, re-run on the shared cohort so it is comparable.

The EMBER branch's own headline numbers came from a random 80/20 split over
everything that parsed - .NET assemblies, packed samples and UPX binaries
included - so they cannot be put next to the tokenization pipeline's numbers.
This harness keeps the branch's features and the branch's model and changes
only which samples go where.

    --variant cohort   join the npz rows to the shared cohort split by sha256
                       (cnn_vit_pipeline/cohort.py), train on its train split,
                       evaluate on its family-disjoint test split.
    --variant branch   the branch's own protocol: no cohort filter, every row
                       that parsed, train_test_split(test_size=0.2,
                       random_state=42, stratify=y).

Running both makes the gap between them visible, which is the point.

Features come from ember_pipeline/extract_features.py (537 dims from the
EMBER branch's extractor, unchanged). Each variant reports two models:
`full` (all 537) and `pruned` (the branch's brittle-feature prune, dropping
the byte histogram + byte-entropy histogram at 0:512 and the 4 string stats
at the end, leaving 21 structural dims).

    python ember_pipeline/train_eval.py --dataset mendeley --variant cohort
    python ember_pipeline/train_eval.py --dataset balanced --variant cohort
    python ember_pipeline/train_eval.py --dataset mendeley --variant branch
    python ember_pipeline/train_eval.py --dataset mendeley --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnn_vit_pipeline import cohort as C  # noqa: E402
from ember_pipeline.ember_extractor import prune_brittle_features  # noqa: E402

FEATURES_DIR = Path(os.environ.get("RANSOM_EMBER_FEATURES",
                                   REPO_ROOT.parent / "ember_features"))
RESULTS_DIR = REPO_ROOT / "results" / "ember"
VM_RUNBOOK = "vm_package/README.md"

# Exactly the EMBER branch's train_and_defend.py settings.
LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbose": -1,
    "seed": 42,
    "bagging_seed": 42,
    "feature_fraction_seed": 42,
    "deterministic": True,
    "force_row_wise": True,
}
NUM_BOOST_ROUND = 100

# npz basenames each dataset needs. label is only used for reporting.
NEEDED = {
    "mendeley": ["mendeley_good_train", "mendeley_good_test",
                 "mendeley_mal_train", "mendeley_mal_test"],
    "balanced": ["goodware_balanced", "mendeley_mal_train", "mendeley_mal_test"],
}
# Which of those the host can produce itself; the rest come from the VM.
HOST_PRODUCIBLE = {"mendeley_good_train", "goodware_balanced"}


# ---------------------------------------------------------------- loading ---
def npz_path(name: str) -> Path:
    return FEATURES_DIR / f"{name}.npz"


def missing_npz(dataset: str) -> list[str]:
    return [n for n in NEEDED[dataset] if not npz_path(n).exists()]


def _fail_missing(dataset: str, missing: list[str]) -> None:
    vm_side = [m for m in missing if m not in HOST_PRODUCIBLE]
    print(f"\nERROR: cannot run --dataset {dataset}: "
          f"{len(missing)} feature file(s) missing under {FEATURES_DIR}",
          file=sys.stderr)
    for m in missing:
        origin = "host" if m in HOST_PRODUCIBLE else "VM"
        print(f"  missing: {npz_path(m)}   (produced on the {origin})",
              file=sys.stderr)
    if vm_side:
        print(f"\nThe ransomware and test-goodware vectors are extracted inside "
              f"the VM; see {VM_RUNBOOK} step 4, then copy *.npz and "
              f"*.manifest.csv back to {FEATURES_DIR}.", file=sys.stderr)
        print("Nothing is fabricated in their absence.", file=sys.stderr)
    print("\nUse --dry-run to exercise the split and the loader on the feature "
          "files that do exist.", file=sys.stderr)


def load_npz(name: str) -> pd.DataFrame:
    """One npz -> a DataFrame with an object column holding each feature row."""
    d = np.load(npz_path(name), allow_pickle=False)
    X = d["X"].astype(np.float32)
    df = pd.DataFrame({
        "sha256": [s.lower() for s in d["sha256"].tolist()],
        "rel_path": d["rel_path"].tolist(),
        "label_npz": d["label"].astype(int).tolist(),
        "npz": name,
        "row": np.arange(len(X)),
    })
    return df, X


def load_dataset_rows(dataset: str, require_all: bool = True):
    """Stack every available npz for a dataset. Returns (frame, X, missing)."""
    missing = missing_npz(dataset)
    if missing and require_all:
        return None, None, missing
    frames, mats, offset = [], [], 0
    for name in NEEDED[dataset]:
        if name in missing:
            continue
        df, X = load_npz(name)
        df["row"] = df["row"] + offset
        offset += len(X)
        frames.append(df)
        mats.append(X)
    if not frames:
        return None, None, missing
    frame = pd.concat(frames, ignore_index=True)
    X = np.vstack(mats)
    return frame, X, missing


def dedupe_by_sha(frame: pd.DataFrame):
    """The corpus stores some identical bytes under several filenames.

    The cohort keeps one row per sha256 (the others are tag:dup), so the
    feature table has to collapse the same way. First occurrence wins;
    the rest are reported, never silently dropped.
    """
    dup_mask = frame["sha256"].duplicated(keep="first")
    dropped = frame.loc[dup_mask, ["sha256", "rel_path", "npz"]].to_dict("records")
    return frame.loc[~dup_mask].reset_index(drop=True), dropped


# --------------------------------------------------------------- metadata ---
def arch_family_lookup() -> pd.DataFrame:
    """sha256 -> arch/family for every row of both cohort CSVs, in-cohort or not.

    The `branch` variant scores samples the cohort excludes, and they still
    need an architecture and a family for the breakdowns.
    """
    parts = []
    for path in (C.COHORT_MENDELEY, C.COHORT_BALANCED):
        if path.exists():
            parts.append(C._read_cohort(path)[["sha256", "arch", "family", "label"]])
    if not parts:
        return pd.DataFrame(columns=["sha256", "arch", "family", "label"])
    allrows = pd.concat(parts, ignore_index=True)
    return allrows.drop_duplicates("sha256").set_index("sha256")


def attach_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    meta = arch_family_lookup()
    out = frame.copy()
    out["arch"] = out["sha256"].map(meta["arch"]).fillna("unknown")
    fam = out["sha256"].map(meta["family"])
    # fall back to the npz rel_path, whose first component is the family folder
    fallback = out["rel_path"].str.split("/", n=1).str[0].where(
        out["rel_path"].str.contains("/"), "root")
    out["family"] = fam.fillna(fallback)
    return out


# --------------------------------------------------------------- training ---
def train_lightgbm(Xtr, ytr, Xte):
    import lightgbm as lgb
    dtrain = lgb.Dataset(Xtr, label=ytr, free_raw_data=False)
    booster = lgb.train(LGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)
    score = booster.predict(Xte, num_iteration=booster.best_iteration or NUM_BOOST_ROUND)
    return booster, np.asarray(score, dtype=float)


def run_model(tag, Xtr, ytr, Xte, yte, arch, family, prune):
    X_tr = prune_brittle_features(Xtr) if prune else Xtr
    X_te = prune_brittle_features(Xte) if prune else Xte
    t0 = time.time()
    booster, score = train_lightgbm(X_tr, ytr, X_te)
    pred = (score >= 0.5).astype(int)
    return C.build_result(
        yte, pred, score, arch, family,
        model="LightGBM",
        feature_set=tag,
        n_features=int(X_tr.shape[1]),
        params={**LGB_PARAMS, "num_boost_round": NUM_BOOST_ROUND},
        n_train=int(len(ytr)),
        n_test=int(len(yte)),
        train_seconds=round(time.time() - t0, 2),
    )


# ------------------------------------------------------------------ modes ---
def do_dry_run(dataset: str) -> int:
    """Prove the loader and the split without needing the VM's npz files."""
    print(f"=== DRY RUN: dataset={dataset} ===")
    split = C.load_split(dataset)
    print(f"shared split: {len(split)} samples "
          f"({int((split['split']=='train').sum())} train / "
          f"{int((split['split']=='test').sum())} test)")

    miss = missing_npz(dataset)
    print(f"\nfeature files under {FEATURES_DIR}")
    for name in NEEDED[dataset]:
        p = npz_path(name)
        if p.exists():
            n = int(np.load(p)["X"].shape[0])
            print(f"  present  {name:<22} {n:5d} rows")
        else:
            origin = "host" if name in HOST_PRODUCIBLE else "VM"
            print(f"  MISSING  {name:<22}   (from the {origin})")

    frame, X, _ = load_dataset_rows(dataset, require_all=False)
    if frame is None:
        print("\nno feature files at all; nothing to join.")
        return 1
    frame, dup = dedupe_by_sha(frame)
    print(f"\nloaded {len(frame)} unique-sha256 vectors, {X.shape[1]} dims "
          f"({len(dup)} duplicate-sha rows collapsed)")

    joined = frame.merge(split, on="sha256", how="inner")
    print(f"joined to the shared split: {len(joined)} rows")
    print(joined.groupby(["split", "source"]).size().to_string())
    print("\nper-arch:")
    print(joined.groupby(["split", "arch"]).size().to_string())

    unmatched = frame[~frame["sha256"].isin(set(split["sha256"]))]
    print(f"\nnpz rows not in the cohort split: {len(unmatched)} "
          f"(excluded by the cohort filter, or belong to the other dataset)")

    gap = C.explain_missing(split, set(frame["sha256"]),
                            on_disk_names=set(_manifest_names(dataset)))
    print(f"\ncoverage so far: {gap['cohort_rows_covered']}/{gap['cohort_rows_total']}")
    print("cohort rows still awaiting features:")
    for k, v in sorted(gap["missing_by_split_source"].items()):
        print(f"  {k:<30} {v}")
    if gap["sha_mismatch_on_disk"]:
        print(f"\n  NOTE {gap['sha_mismatch_on_disk']} of those ARE on disk but "
              f"hash differently than the cohort CSV records.")
        print(f"       e.g. {', '.join(gap['sha_mismatch_examples'][:4])}")
        print(f"       (Mendeley goodware: the cohort holds the UPX-unpacked "
              f"sha256, the extractor ran on the packed original.)")
    if gap["absent_from_host_tree"]:
        if miss:
            print(f"  NOTE {gap['absent_from_host_tree']} cohort filenames are "
                  f"not in any extracted tree yet - dominated by the "
                  f"{len(miss)} feature file(s) still missing, above.")
        else:
            print(f"  NOTE {gap['absent_from_host_tree']} cohort filenames are "
                  f"not in the extracted tree at all.")

    if miss:
        print(f"\nfull run still blocked on: {', '.join(miss)}  (see {VM_RUNBOOK})")
    print("\nDRY RUN OK")
    return 0


def _manifest_names(dataset: str) -> set[str]:
    """Filenames every available extractor manifest actually saw."""
    names: set[str] = set()
    for name in NEEDED[dataset]:
        man = FEATURES_DIR / f"{name}.manifest.csv"
        if man.exists():
            df = pd.read_csv(man, dtype=str, keep_default_na=False)
            names |= set(df["rel_path"].str.rsplit("/", n=1).str[-1])
    return names


def do_smoke(dataset: str, out_dir: Path) -> int:
    """Prove the LightGBM + metrics path end-to-end without any ransomware.

    NOT a result. The available goodware is pseudo-labelled by architecture
    (x86 -> 0, x64 -> 1) so there is something real to learn, and the whole
    train/prune/evaluate/write chain runs exactly as it will on the real data.
    Output goes wherever --out points; never to results/.
    """
    print("=== SMOKE TEST: pseudo-labels (x86=0 / x64=1), NOT a result ===")
    frame, X, _ = load_dataset_rows(dataset, require_all=False)
    if frame is None:
        print("no feature files available", file=sys.stderr)
        return 1
    frame, _ = dedupe_by_sha(frame)
    X = X[frame["row"].to_numpy()]
    frame = attach_metadata(frame.reset_index(drop=True))
    keep = frame["arch"].isin(["x86", "x64"]).to_numpy()
    frame, X = frame[keep].reset_index(drop=True), X[keep]

    y = (frame["arch"] == "x64").to_numpy().astype(int)
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2,
                              random_state=42, stratify=y)
    is_train = np.zeros(len(y), bool)
    is_train[tr] = True
    print(f"pseudo-task: train {is_train.sum()} / test {(~is_train).sum()}, "
          f"{X.shape[1]} dims")

    t0 = time.time()
    results = [
        run_model("full", X[is_train], y[is_train], X[~is_train], y[~is_train],
                  frame.loc[~is_train, "arch"], frame.loc[~is_train, "family"],
                  prune=False),
        run_model("pruned", X[is_train], y[is_train], X[~is_train], y[~is_train],
                  frame.loc[~is_train, "arch"], frame.loc[~is_train, "family"],
                  prune=True),
    ]
    for r in results:
        print(f"  {r['feature_set']:<7} ({r['n_features']:3d} dims)  "
              f"acc {r['accuracy']:.4f}  macro_f1 {r['macro_f1']:.4f}  "
              f"auc {round(r['roc_auc'], 4)}  "
              f"train {r['train_seconds']}s")
    frame = frame.assign(label=y, split=np.where(is_train, "train", "test"),
                         source="smoke_pseudo", family_or_group=frame["family"])
    out_dir.mkdir(parents=True, exist_ok=True)
    C.write_metrics(out_dir / "metrics.json",
                    experiment=f"ember_{dataset}_SMOKE",
                    description="SMOKE TEST ONLY - pseudo-labels x86/x64 on "
                                "goodware, proves the code path, not a result.",
                    samples=C.split_summary(frame, "split"),
                    results=results, elapsed_seconds=time.time() - t0,
                    smoke_test=True)
    print(f"\nwrote {out_dir / 'metrics.json'} (smoke output, not a result)")
    print("SMOKE OK")
    return 0


def do_run(dataset: str, variant: str, out_dir: Path) -> int:
    t0 = time.time()
    frame, X, miss = load_dataset_rows(dataset, require_all=True)
    if miss:
        _fail_missing(dataset, miss)
        return 2

    frame, dup = dedupe_by_sha(frame)
    X = X[frame["row"].to_numpy()]
    frame = frame.reset_index(drop=True)
    frame["row"] = np.arange(len(frame))

    if variant == "cohort":
        split = C.load_split(dataset)
        joined = frame.merge(split, on="sha256", how="inner")
        dropped = len(frame) - len(joined)
        idx = joined["row"].to_numpy()
        Xj = X[idx]
        y = joined["label"].to_numpy()
        is_train = (joined["split"] == "train").to_numpy()
        meta = joined
        description = (f"EMBER branch features + LightGBM on the shared cohort "
                       f"({dataset}); family-disjoint split, .NET/packed/UPX/"
                       f"Thanos excluded. {dropped} extracted vectors dropped by "
                       f"the cohort filter.")
        samples = C.split_summary(joined, "split")
        samples["dropped_by_cohort_filter"] = int(dropped)
        samples["cohort_rows_without_features"] = int(
            len(split) - len(joined))
    else:
        from sklearn.model_selection import train_test_split
        meta = attach_metadata(frame)
        y = meta["label_npz"].to_numpy()
        Xj = X
        tr_idx, te_idx = train_test_split(
            np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
        is_train = np.zeros(len(y), bool)
        is_train[tr_idx] = True
        meta = meta.assign(
            split=np.where(is_train, "train", "test"),
            label=y,
            source=meta["npz"],
            family_or_group=meta["family"])
        description = (f"EMBER branch's own protocol on {dataset}: no cohort "
                       f"filter, every vector that parsed, "
                       f"train_test_split(test_size=0.2, random_state=42, "
                       f"stratify=y). Reported for comparison only - the split "
                       f"is random, so families and near-duplicates straddle it.")
        samples = C.split_summary(meta, "split")

    ytr, yte = y[is_train], y[~is_train]
    Xtr, Xte = Xj[is_train], Xj[~is_train]
    arch = meta.loc[~is_train, "arch"].to_numpy()
    family = meta.loc[~is_train, "family"].to_numpy()

    print(f"train {len(ytr)} (good {int((ytr==0).sum())} / "
          f"rans {int((ytr==1).sum())})   "
          f"test {len(yte)} (good {int((yte==0).sum())} / "
          f"rans {int((yte==1).sum())})")
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        print("ERROR: a split has only one class; refusing to train.",
              file=sys.stderr)
        return 3

    results = [
        run_model("full", Xtr, ytr, Xte, yte, arch, family, prune=False),
        run_model("pruned", Xtr, ytr, Xte, yte, arch, family, prune=True),
    ]
    for r in results:
        print(f"  {r['feature_set']:<7} ({r['n_features']:3d} dims)  "
              f"acc {r['accuracy']:.4f}  bal_acc {r['balanced_accuracy']:.4f}  "
              f"macro_f1 {r['macro_f1']:.4f}  "
              f"auc {r['roc_auc'] if r['roc_auc'] is None else round(r['roc_auc'],4)}  "
              f"FPR {r['false_positive_rate']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_cols = [c for c in ("sha256", "rel_path", "npz", "label", "split",
                             "arch", "family", "source") if c in meta.columns]
    meta[meta_cols].to_csv(out_dir / "split_used.csv", index=False)
    if dup:
        pd.DataFrame(dup).to_csv(out_dir / "duplicate_sha_rows.csv", index=False)

    path = C.write_metrics(
        out_dir / "metrics.json",
        experiment=f"ember_{dataset}_{variant}",
        description=description,
        samples=samples,
        results=results,
        elapsed_seconds=time.time() - t0,
        dataset=dataset,
        variant=variant,
        features="EMBER branch extractor, 537 dims "
                 "(256 byte histogram + 256 byte-entropy histogram + "
                 "21 structural + 4 string)",
        feature_files=[str(npz_path(n)) for n in NEEDED[dataset]],
        duplicate_sha_rows_collapsed=len(dup),
    )
    print(f"\nwrote {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=C.DATASETS, required=True)
    ap.add_argument("--variant", choices=["cohort", "branch"], default="cohort")
    ap.add_argument("--dry-run", action="store_true",
                    help="load whatever npz exist, apply the split, print counts, "
                         "train nothing")
    ap.add_argument("--smoke", action="store_true",
                    help="prove the LightGBM + metrics path on goodware alone "
                         "using pseudo-labels; requires --out, never a result")
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    global FEATURES_DIR
    if a.features_dir:
        FEATURES_DIR = Path(a.features_dir)

    if a.dry_run:
        return do_dry_run(a.dataset)
    if a.smoke:
        if not a.out:
            print("--smoke needs --out (a scratch directory); smoke output must "
                  "never land in results/.", file=sys.stderr)
            return 2
        return do_smoke(a.dataset, Path(a.out))
    out = Path(a.out) if a.out else RESULTS_DIR / a.dataset / a.variant
    return do_run(a.dataset, a.variant, out)


if __name__ == "__main__":
    raise SystemExit(main())
