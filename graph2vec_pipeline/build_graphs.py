#!/usr/bin/env python3
"""Extract CFGs for every sample in the shared cohort and cache WL documents.

    python graph2vec_pipeline/build_graphs.py            # both trees
    python graph2vec_pipeline/build_graphs.py --limit 50 # smoke test

Writes, per disassembly tree:
    <cache>/wl_<tree>_i<iters>_b<blocks>_n<insns>.npz
        shas     (U64)   one row per sample, sorted
        indptr   (int64) row boundaries into labels/counts
        labels   (int64) WL subtree label hashes, per row sorted ascending
        counts   (int32) multiplicity of each label in that graph
    results/graph2vec/graph_stats.csv
        one row per sample: block/edge counts, truncation, class, arch

The cache lives outside the repo (RANSOM_G2V_CACHE, default the session
scratchpad) because it is ~400MB and is pure derived data.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from graph2vec_pipeline.cfg import cfg_from_file, wl_labels  # noqa: E402

SHARED = Path(os.environ.get(
    "RANSOM_SHARED_DIR", REPO.parent / "asm and mm" / "Shared"))
# the extraction tree of every registered corpus (family_holdout.common), plus
# the pipeline's historical name for the balanced-goodware tree, which its
# cache files are keyed by
from family_holdout.common import TREE_OF_CORPUS  # noqa: E402
TREES = {**TREE_OF_CORPUS, "balanced_goodware": TREE_OF_CORPUS["balanced"]}
DEFAULT_CACHE = Path(os.environ.get(
    "RANSOM_G2V_CACHE",
    Path(os.environ.get("TEMP", "/tmp")) / "ransom_g2v_cache"))

MAX_BLOCKS = 5_000
MAX_INSNS = 80_000
WL_ITERS = 2


def cache_path(cache_dir: Path, tree: str, iters: int, blocks: int,
               insns: int) -> Path:
    return cache_dir / f"wl_{tree}_i{iters}_b{blocks}_n{insns}.npz"


def _needed() -> pd.DataFrame:
    """Union of both datasets' samples, with the tree each one lives in."""
    from cnn_vit_pipeline.cohort import load_split
    frames = []
    for ds in ("mendeley", "balanced"):
        d = load_split(ds).copy()
        d["dataset"] = ds
        frames.append(d)
    allrows = pd.concat(frames, ignore_index=True)
    allrows["tree"] = np.where(
        allrows["source"] == "balanced_goodware", "balanced_goodware",
        "mendeley")
    keep = allrows.drop_duplicates("sha256")[
        ["sha256", "tree", "label", "arch", "family", "source", "filename"]]
    return keep.sort_values("sha256").reset_index(drop=True)


def build(tree: str, shas: list[str], iters: int, blocks: int, insns: int,
          cache_dir: Path, progress: int = 200):
    root = TREES[tree] / "asm"
    labels_parts, counts_parts = [], []
    indptr = [0]
    stats_rows = []
    t0 = time.time()
    for k, sha in enumerate(shas, 1):
        path = root / f"{sha}.asm"
        try:
            cfg = cfg_from_file(path, max_insns=insns, max_blocks=blocks)
            doc = wl_labels(cfg, iters)
        except Exception as exc:                      # pragma: no cover
            print(f"  !! {sha[:12]}: {type(exc).__name__}: {exc}")
            cfg, doc = None, {}
        if doc:
            arr = np.fromiter(doc.keys(), dtype=np.int64, count=len(doc))
            cnt = np.fromiter(doc.values(), dtype=np.int32, count=len(doc))
            order = np.argsort(arr, kind="stable")
            labels_parts.append(arr[order])
            counts_parts.append(cnt[order])
        indptr.append(indptr[-1] + len(doc))
        row = {"sha256": sha, "tree": tree}
        row.update(cfg.stats if cfg is not None else {})
        row["wl_doc_size"] = len(doc)
        stats_rows.append(row)
        if k % progress == 0:
            rate = k / (time.time() - t0)
            print(f"  {tree}: {k}/{len(shas)}  {rate:.1f} files/s", flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_path(cache_dir, tree, iters, blocks, insns)
    np.savez(
        out,
        shas=np.array(shas, dtype="U64"),
        indptr=np.array(indptr, dtype=np.int64),
        labels=(np.concatenate(labels_parts) if labels_parts
                else np.empty(0, np.int64)),
        counts=(np.concatenate(counts_parts) if counts_parts
                else np.empty(0, np.int32)),
    )
    print(f"  wrote {out}  ({out.stat().st_size/1e6:.0f} MB, "
          f"{time.time()-t0:.0f}s)")
    return pd.DataFrame(stats_rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--iters", type=int, default=WL_ITERS)
    ap.add_argument("--max-blocks", type=int, default=MAX_BLOCKS)
    ap.add_argument("--max-insns", type=int, default=MAX_INSNS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cache_dir = Path(a.cache_dir)
    need = _needed()
    print(f"{len(need)} unique samples across both datasets")
    print(need["tree"].value_counts().to_string())

    stats = []
    for tree in ("mendeley", "balanced_goodware"):
        shas = need.loc[need["tree"] == tree, "sha256"].tolist()
        if a.limit:
            shas = shas[:a.limit]
        cp = cache_path(cache_dir, tree, a.iters, a.max_blocks, a.max_insns)
        if cp.exists() and not a.force and not a.limit:
            print(f"  {tree}: cache present, skipping ({cp})")
            continue
        print(f"building {tree}: {len(shas)} files")
        stats.append(build(tree, shas, a.iters, a.max_blocks, a.max_insns,
                           cache_dir))

    if stats:
        df = pd.concat(stats, ignore_index=True)
        df = df.merge(need[["sha256", "label", "arch", "family", "source"]],
                      on="sha256", how="left")
        outdir = REPO / "results" / "graph2vec"
        outdir.mkdir(parents=True, exist_ok=True)
        dest = outdir / ("graph_stats.csv" if not a.limit
                         else "graph_stats_smoke.csv")
        df.to_csv(dest, index=False)
        print(f"wrote {dest}  ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
