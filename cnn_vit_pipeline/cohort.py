#!/usr/bin/env python3
"""Shared cohort + split logic, and the shared metrics.json schema writer.

This is the ONE place where "which samples, which split" is decided. Both
harnesses import it:

    from cnn_vit_pipeline.cohort import load_split, add_val_fold
    from cnn_vit_pipeline.cohort import build_result, write_metrics

(The brief asked for exactly one shared module. The metrics schema lives here
too rather than in a second file, so the EMBER and CNN-ViT harnesses cannot
drift apart on either the split or the JSON shape.)

Why this exists
---------------
The original EMBER and CNN-ViT runs used random splits over everything that
parsed, including .NET assemblies, packed samples and UPX binaries, so their
headline numbers are not comparable with the tokenization pipeline's. This
module reproduces the tokenization pipeline's cohort and split exactly.

Two datasets
------------
mendeley  ransomware and goodware both from the Mendeley corpus.
          train = cohort rows with set good_train + mal_train
          test  = cohort rows with set good_test  + mal_test
          This IS the paper's family-disjoint split (24 train / 14 test
          ransomware families once the cohort filter removes Night Sky and
          Thanos). It is never re-split.

balanced  ransomware exactly as in `mendeley`; goodware from the
          Goodware_Balanced corpus, group-split by source project. The
          membership is NOT invented here: it is read back from the
          tokenization pipeline's own split (results/expB/splits.csv, snapshot
          kept at results/cnn_vit/expB_splits_snapshot.csv), mapped from its
          `file` column to sha256 through LLM_Features_Balanced/
          opcode_manifest.csv, and then intersected with cohort_balanced.

Cohort membership
-----------------
in_cohort == 1 in the cohort CSVs, i.e. tag is `plain` or `upx_unpacked` and
the family is not Thanos. .NET, entropy-flagged packed, UPX that would not
unpack, no-code, odd-arch, broken and duplicate samples are out.

Groups
------
`family_or_group` is what must not straddle a split boundary:
  * ransomware        -> family (avaddon, conti, ...)
  * mendeley goodware -> "file:<sha256>", a singleton group (this mirrors
                         results/expA/splits.csv, which uses "file:<name>")
  * balanced goodware -> the source project, taken verbatim from expB's
                         `group` column so our groups are literally theirs.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Locations. Every one can be overridden with an environment variable so the
# harnesses stay runnable if the data moves.
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
_DOWNLOADS = REPO_ROOT.parent

SHARED_DIR = Path(os.environ.get("RANSOM_SHARED_DIR", _DOWNLOADS / "asm and mm" / "Shared"))
COHORT_MENDELEY = SHARED_DIR / "cohort_mendeley.csv"
COHORT_BALANCED = SHARED_DIR / "cohort_balanced.csv"

BALANCED_MANIFEST = Path(os.environ.get(
    "RANSOM_BALANCED_MANIFEST", _DOWNLOADS / "LLM_Features_Balanced" / "opcode_manifest.csv"))

# The snapshot is preferred: the tokenization worker may be rewriting
# results/expB/splits.csv concurrently, so we froze a copy on first read.
EXPB_SPLITS_SNAPSHOT = REPO_ROOT / "results" / "cnn_vit" / "expB_splits_snapshot.csv"
EXPB_SPLITS_LIVE = REPO_ROOT / "results" / "expB" / "splits.csv"

DATASETS = ("mendeley", "balanced")

VAL_FRACTION = 0.10
VAL_SEED = 1337

CLASS_NAMES = ("goodware", "ransomware")  # index == label


# ----------------------------------------------------------------------------
# Cohort CSV loading
# ----------------------------------------------------------------------------
def _read_cohort(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"cohort file not found: {path}\n"
            f"Set RANSOM_SHARED_DIR to the directory holding cohort_mendeley.csv "
            f"and cohort_balanced.csv.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    need = {"corpus", "sha256", "set", "label", "family", "filename", "arch", "tag",
            "in_cohort", "exclude_reason"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df["sha256"] = df["sha256"].str.strip().str.lower()
    df["in_cohort"] = df["in_cohort"].str.strip()
    df["label"] = df["label"].astype(int)
    return df


def load_cohort(name: str, in_cohort_only: bool = True) -> pd.DataFrame:
    """Raw cohort table. name is 'mendeley' or 'balanced'."""
    path = {"mendeley": COHORT_MENDELEY, "balanced": COHORT_BALANCED}[name]
    df = _read_cohort(path)
    if in_cohort_only:
        df = df[df["in_cohort"] == "1"].copy()
    return df.reset_index(drop=True)


def _read_expb_splits() -> tuple[pd.DataFrame, Path]:
    path = EXPB_SPLITS_SNAPSHOT if EXPB_SPLITS_SNAPSHOT.exists() else EXPB_SPLITS_LIVE
    if not path.exists():
        raise FileNotFoundError(
            f"neither {EXPB_SPLITS_SNAPSHOT} nor {EXPB_SPLITS_LIVE} exists; the "
            f"balanced dataset reuses the tokenization pipeline's expB split.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return df, path


# ----------------------------------------------------------------------------
# The split
# ----------------------------------------------------------------------------
_SET_TO_SPLIT = {"good_train": "train", "mal_train": "train",
                 "good_test": "test", "mal_test": "test"}

_COLUMNS = ["sha256", "label", "family_or_group", "split", "arch", "source",
            "filename", "family"]


def _mendeley_frame() -> pd.DataFrame:
    coh = load_cohort("mendeley")
    coh = coh[coh["set"].isin(_SET_TO_SPLIT)].copy()
    coh["split"] = coh["set"].map(_SET_TO_SPLIT)
    coh["source"] = np.where(coh["label"] == 1, "mendeley_ransomware", "mendeley_goodware")
    coh["family_or_group"] = np.where(
        coh["label"] == 1, coh["family"], "file:" + coh["sha256"])
    return coh[_COLUMNS].reset_index(drop=True)


def balanced_goodware_audit() -> dict:
    """How expB's balanced goodware survives the cohort filter.

    Returns a dict with the join and survival counts, and the surviving rows.
    Reported by build_dataset/train_eval so the coordinator can see the
    attrition rather than having to trust a number.
    """
    splits, splits_path = _read_expb_splits()
    good = splits[splits["source"] == "balanced_goodware"].copy()

    man = pd.read_csv(BALANCED_MANIFEST, dtype=str, keep_default_na=False,
                      encoding="utf-8-sig")
    # expB's `file` is "<bucket>_<filename>.txt", which is exactly the manifest's
    # txt_file (a filename starting with "_" therefore yields a double underscore,
    # e.g. everyday__cvpcb.dll.txt -- that is a real name, not a typo).
    txt2sha = dict(zip(man["txt_file"], man["sha256"].str.strip().str.lower()))

    good["sha256"] = good["file"].map(txt2sha)
    unmapped = good[good["sha256"].isna()]
    mapped = good[good["sha256"].notna()].copy()

    coh_all = _read_cohort(COHORT_BALANCED)
    coh_by_sha = coh_all.set_index("sha256")
    in_cohort = set(coh_all.loc[coh_all["in_cohort"] == "1", "sha256"])

    mapped["in_cohort"] = mapped["sha256"].isin(in_cohort)
    mapped["not_in_cohort_csv"] = ~mapped["sha256"].isin(set(coh_all["sha256"]))

    def _reason(sha):
        if sha in coh_by_sha.index:
            return coh_by_sha.loc[sha, "exclude_reason"] or "(kept)"
        return "not_in_cohort_csv"

    dropped = mapped[~mapped["in_cohort"]].copy()
    dropped["exclude_reason"] = dropped["sha256"].map(_reason)

    kept = mapped[mapped["in_cohort"]].copy()
    kept = kept.join(coh_by_sha[["arch", "family", "filename"]], on="sha256")
    kept["label"] = 0
    kept["source"] = "balanced_goodware"
    kept["family_or_group"] = kept["group"]  # expB's own project groups, verbatim

    return {
        "expb_splits_path": str(splits_path),
        "expb_balanced_goodware_rows": int(len(good)),
        "expb_train": int((good["split"] == "train").sum()),
        "expb_test": int((good["split"] == "test").sum()),
        "unmapped_to_sha256": sorted(unmapped["file"].tolist()),
        "survived_train": int((kept["split"] == "train").sum()),
        "survived_test": int((kept["split"] == "test").sum()),
        "dropped_by_cohort": int(len(dropped)),
        "dropped_reasons": dropped["exclude_reason"].value_counts().to_dict(),
        "dropped_files": dropped[["file", "split", "exclude_reason"]]
                         .to_dict(orient="records"),
        "_kept": kept[_COLUMNS].reset_index(drop=True),
    }


def _balanced_frame() -> pd.DataFrame:
    mend = _mendeley_frame()
    rans = mend[mend["label"] == 1]  # ransomware exactly as in dataset 1
    good = balanced_goodware_audit()["_kept"]
    return pd.concat([rans, good], ignore_index=True)[_COLUMNS]


def load_split(dataset: str) -> pd.DataFrame:
    """The shared split.

    Returns a DataFrame with columns
        sha256, label, family_or_group, split, arch, source, filename, family
    one row per sample, `split` in {train, test}.
    """
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    df = _mendeley_frame() if dataset == "mendeley" else _balanced_frame()
    df = df.sort_values(["split", "label", "sha256"]).reset_index(drop=True)
    dup = df["sha256"].duplicated()
    if dup.any():
        raise AssertionError(f"duplicate sha256 in {dataset} split: "
                             f"{df.loc[dup, 'sha256'].tolist()[:5]}")
    return df


# ----------------------------------------------------------------------------
# train -> train/val, fixed seed, stratified by label, group-aware
# ----------------------------------------------------------------------------
def add_val_fold(df: pd.DataFrame, val_frac: float = VAL_FRACTION,
                 seed: int = VAL_SEED) -> pd.DataFrame:
    """Add a `fold` column with train/val/test.

    val is carved out of train only. Whole groups move together, so no family
    and no source project appears in both train and val. Groups are drawn per
    class so both classes are represented in val.
    """
    out = df.copy()
    out["fold"] = out["split"]
    train = out[out["split"] == "train"]

    val_groups: set[str] = set()
    for label in sorted(train["label"].unique()):
        part = train[train["label"] == label]
        target = int(round(len(part) * val_frac))
        sizes = part.groupby("family_or_group").size().to_dict()
        groups = sorted(sizes)
        rng = random.Random(f"{seed}:{label}")
        rng.shuffle(groups)
        taken = 0
        for g in groups:
            if taken >= target:
                break
            # never take the last remaining group of a class
            if len(val_groups & set(sizes)) + 1 >= len(sizes):
                break
            val_groups.add(g)
            taken += sizes[g]

    mask = (out["split"] == "train") & out["family_or_group"].isin(val_groups)
    out.loc[mask, "fold"] = "val"
    return out


# ----------------------------------------------------------------------------
# Metrics, in the schema of results/expA/metrics.json plus the two extra
# breakdowns the brief asks for (per-architecture, per-family recall).
# ----------------------------------------------------------------------------
def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def _core_metrics(y_true, y_pred, y_score=None) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    n = len(y_true)

    rec_g = _safe_div(tn, tn + fp)
    rec_r = _safe_div(tp, tp + fn)
    pre_g = _safe_div(tn, tn + fn)
    pre_r = _safe_div(tp, tp + fp)
    f1_g = _safe_div(2 * pre_g * rec_g, pre_g + rec_g)
    f1_r = _safe_div(2 * pre_r * rec_r, pre_r + rec_r)
    sup_g, sup_r = tn + fp, tp + fn

    present = [r for r, s in ((rec_g, sup_g), (rec_r, sup_r)) if s]
    maj = max(sup_g, sup_r)

    out = {
        "accuracy": _safe_div(tn + tp, n),
        "balanced_accuracy": float(np.mean(present)) if present else 0.0,
        "majority_class_accuracy": _safe_div(maj, n),
        "precision_goodware": pre_g,
        "recall_goodware": rec_g,
        "f1_goodware": f1_g,
        "support_goodware": sup_g,
        "precision_ransomware": pre_r,
        "recall_ransomware": rec_r,
        "f1_ransomware": f1_r,
        "support_ransomware": sup_r,
        "macro_precision": float(np.mean([pre_g, pre_r])),
        "macro_recall": float(np.mean([rec_g, rec_r])),
        "macro_f1": float(np.mean([f1_g, f1_r])),
        "false_positive_rate": _safe_div(fp, tn + fp),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "roc_auc": None,
    }
    if y_score is not None and sup_g and sup_r:
        from sklearn.metrics import roc_auc_score
        out["roc_auc"] = float(roc_auc_score(y_true, np.asarray(y_score, dtype=float)))
    return out


def build_result(y_true, y_pred, y_score, arch, family, **extra) -> dict:
    """One entry of metrics.json["results"].

    arch and family are per-test-sample arrays aligned with y_true, taken from
    the cohort CSVs. They drive the two extra breakdowns:
      per_architecture  -- the whole metric block again, for x86 and for x64
      per_family_recall -- recall per ransomware test family
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score = None if y_score is None else np.asarray(y_score, dtype=float)
    arch = np.asarray(arch, dtype=object)
    family = np.asarray(family, dtype=object)

    res = _core_metrics(y_true, y_pred, y_score)

    per_arch = {}
    for a in sorted({str(x) for x in arch}):
        m = arch == a
        per_arch[a] = _core_metrics(y_true[m], y_pred[m],
                                    None if y_score is None else y_score[m])
        per_arch[a]["n"] = int(m.sum())
    res["per_architecture"] = per_arch
    # expA reports a goodware-only view of the same thing; keep the key so the
    # three pipelines' metrics.json stay directly diffable.
    # Aliases in the exact shape llm_features_pipeline writes, so metrics.json
    # files from every pipeline carry the same generic keys.
    res["per_arch"] = {
        a: {"n": per_arch[a]["n"],
            "support_goodware": per_arch[a]["support_goodware"],
            "support_ransomware": per_arch[a]["support_ransomware"],
            "recall_goodware": per_arch[a]["recall_goodware"],
            "recall_ransomware": per_arch[a]["recall_ransomware"],
            "accuracy": per_arch[a]["accuracy"],
            "macro_f1": per_arch[a]["macro_f1"]}
        for a in per_arch}
    res["goodware_recall_by_arch"] = {
        a: {"n": int(((arch == a) & (y_true == 0)).sum()),
            "recall_goodware": per_arch[a]["recall_goodware"]}
        for a in per_arch if int(((arch == a) & (y_true == 0)).sum())}

    per_family = {}
    for f in sorted({str(x) for x, t in zip(family, y_true) if t == 1}):
        m = (family == f) & (y_true == 1)
        correct = int((y_pred[m] == 1).sum())
        per_family[f] = {"recall": _safe_div(correct, int(m.sum())),
                         "correct": correct, "support": int(m.sum())}
    res["per_family_recall"] = per_family

    res["ransomware_recall_by_family"] = {
        f: {"n": v["support"], "recall": v["recall"], "detected": v["correct"]}
        for f, v in per_family.items()}

    res.update(extra)
    return res


def explain_missing(split_df: pd.DataFrame, have_shas, on_disk_names=()) -> dict:
    """Why do cohort rows have no extracted artifact?

    Both host extractors key their output by the sha256 of the file as it sits
    on disk. Where that disagrees with the cohort CSV the row cannot be joined,
    and the two causes look identical unless they are separated:

      absent_from_host_tree  the filename is not in the extracted tree at all
      sha_mismatch_on_disk   the filename IS there but hashes differently. On
                             the Mendeley goodware tree this is the UPX case:
                             the cohort recorded the unpacked sha256, the host
                             extractors ran on the packed original.

    `on_disk_names` is the set of filenames the extractor actually saw.
    """
    have_shas = set(have_shas)
    on_disk_names = set(on_disk_names)
    want = split_df[~split_df["sha256"].isin(have_shas)]
    mismatch = want[want["filename"].isin(on_disk_names)]
    absent = want[~want["filename"].isin(on_disk_names)]
    return {
        "cohort_rows_total": int(len(split_df)),
        "cohort_rows_covered": int(len(split_df) - len(want)),
        "cohort_rows_missing": int(len(want)),
        "sha_mismatch_on_disk": int(len(mismatch)),
        "absent_from_host_tree": int(len(absent)),
        "missing_by_split_source": {
            f"{s}/{src}": int(n) for (s, src), n in
            want.groupby(["split", "source"]).size().items()},
        "sha_mismatch_examples": mismatch["filename"].tolist()[:20],
        "absent_examples": absent["filename"].tolist()[:20],
    }


def split_summary(df: pd.DataFrame, fold_col: str = "split") -> dict:
    """The metrics.json["samples"] block."""
    out = {"total": int(len(df))}
    for fold in ("train", "val", "test"):
        part = df[df[fold_col] == fold]
        if not len(part):
            continue
        out[fold] = {
            "n": int(len(part)),
            "goodware": int((part["label"] == 0).sum()),
            "ransomware": int((part["label"] == 1).sum()),
            "sources": part["source"].value_counts().to_dict(),
            "groups": int(part["family_or_group"].nunique()),
            "arch": part["arch"].value_counts().to_dict(),
        }
    folds = [f for f in ("train", "val", "test") if f in out]
    overlap = {}
    for i, a in enumerate(folds):
        for b in folds[i + 1:]:
            ga = set(df.loc[df[fold_col] == a, "family_or_group"])
            gb = set(df.loc[df[fold_col] == b, "family_or_group"])
            overlap[f"{a}|{b}"] = sorted(ga & gb)
    out["group_overlap"] = overlap

    good = df[df["label"] == 0]
    out["goodware_arch"] = {
        f: good.loc[good[fold_col] == f, "arch"].value_counts().to_dict()
        for f in folds}
    out["test_goodware_arch"] = out["goodware_arch"].get("test", {})
    return out


def write_metrics(path, experiment: str, description: str, samples: dict,
                  results: list, elapsed_seconds: float, **extra) -> Path:
    """Write metrics.json in the schema of results/expA/metrics.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "experiment": experiment,
        "description": description,
        "task": "binary: 0=goodware, 1=ransomware",
        "samples": samples,
        "results": results,
        "elapsed_seconds": round(float(elapsed_seconds), 2),
    }
    doc.update(extra)
    path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    return path


# ----------------------------------------------------------------------------
def _cli() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="print the shared cohort split")
    ap.add_argument("--dataset", choices=DATASETS, default="mendeley")
    ap.add_argument("--with-val", action="store_true")
    ap.add_argument("--audit-balanced", action="store_true")
    a = ap.parse_args()

    df = load_split(a.dataset)
    if a.with_val:
        df = add_val_fold(df)
    col = "fold" if a.with_val else "split"
    print(json.dumps(split_summary(df, col), indent=2, default=str))

    fam = df[df["label"] == 1]
    tr = set(fam.loc[fam["split"] == "train", "family_or_group"])
    te = set(fam.loc[fam["split"] == "test", "family_or_group"])
    print(f"\nransomware families: {len(tr)} train, {len(te)} test, "
          f"overlap {sorted(tr & te) or 'none'}")

    if a.audit_balanced:
        au = balanced_goodware_audit()
        au.pop("_kept")
        au["dropped_files"] = au["dropped_files"][:20]
        print("\nbalanced goodware audit:")
        print(json.dumps(au, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
