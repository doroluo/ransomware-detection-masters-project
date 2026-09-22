#!/usr/bin/env python3
"""One cached pass over the disassembly, rich enough to replay every option.

`build_graphs.py` caches *WL documents*: one WL hyper-parameter per pass over
7.2GB of text. That is the wrong unit for a hyper-parameter search, because
`h`, the node-label granularity, the block cap, whether call edges count and
whether edge labels enter the relabelling are all things we want to vary and
none of them needs the disassembly again.

So this module caches the **graph** instead: per node, the integer attributes
every labelling scheme is built from; per edge, its endpoints and its kind.
Everything downstream (`wl.py`) is numpy over those arrays.

Layout, one .npz per (tree, construction):

    shas      (U64)    one row per sample, sorted
    node_ptr  (int64)  N+1 offsets into the node arrays
    edge_ptr  (int64)  N+1 offsets into the edge arrays
    n_blocks  (int32)  nodes [0, n_blocks) are basic blocks, the rest are
                       synthetic extern (import-thunk) nodes
    dom term szb flg   (uint8)  per node
    blen               (uint16) instructions in the block
    seq                (uint32) FNV-1a of the block's first 8 mnemonic classes
    esrc edst          (int32)  graph-local node indices
    etype              (uint8)  one of cfg.E_*

A construction is named by its parameters, e.g.
`g_mendeley_b20000_n80000_w1_xs1_ex1.npz`. `include_calls` is NOT baked in:
call edges carry their own edge kind, so a run can drop them by filtering.
Likewise a smaller block cap is obtained by keeping the first K blocks, and
"no cross-section targets" by dropping the *_XSEC edge kinds.

    python graph2vec_pipeline/graph_cache.py --max-blocks 20000
    python graph2vec_pipeline/graph_cache.py --windows 4 --max-blocks 20000
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
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from graph2vec_pipeline.cfg import cfg_from_file  # noqa: E402

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

STAT_COLS = ["n_blocks", "n_edges", "insns_used", "n_sections", "truncated",
             "block_capped", "resolved_targets", "unresolved_targets",
             "back_edges", "self_loops", "call_edges", "extern_nodes",
             "extern_edges", "xsec_targets", "mean_block_len"]


def cache_name(tree: str, blocks: int, insns: int, windows: int,
               xsec: bool, extern: bool) -> str:
    return (f"g_{tree}_b{blocks}_n{insns}_w{windows}"
            f"_xs{int(xsec)}_ex{int(extern)}.npz")


def cache_file(cache_dir: Path, tree: str, blocks: int, insns: int,
               windows: int, xsec: bool, extern: bool) -> Path:
    return Path(cache_dir) / cache_name(tree, blocks, insns, windows, xsec,
                                        extern)


def needed_samples() -> pd.DataFrame:
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


class GraphSet:
    """A loaded cache. Views are numpy slices; nothing is copied per graph."""

    __slots__ = ("shas", "index", "node_ptr", "edge_ptr", "n_blocks", "dom",
                 "term", "szb", "flg", "blen", "seq", "esrc", "edst", "etype")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def rows(self, shas) -> np.ndarray:
        return np.array([self.index[s] for s in shas], dtype=np.int64)


def load_graphs(paths) -> GraphSet:
    """Concatenate one or more cache files into a single GraphSet."""
    parts = [np.load(Path(p), allow_pickle=False) for p in paths]
    shas = np.concatenate([z["shas"] for z in parts])
    n_blocks = np.concatenate([z["n_blocks"] for z in parts])
    node_ptr = [0]
    edge_ptr = [0]
    for z in parts:
        np_off, ep_off = node_ptr[-1], edge_ptr[-1]
        node_ptr.extend((z["node_ptr"][1:] + np_off).tolist())
        edge_ptr.extend((z["edge_ptr"][1:] + ep_off).tolist())
    cat = {k: np.concatenate([z[k] for z in parts])
           for k in ("dom", "term", "szb", "flg", "blen", "seq",
                     "esrc", "edst", "etype")}
    return GraphSet(shas=shas,
                    index={str(s): i for i, s in enumerate(shas)},
                    node_ptr=np.array(node_ptr, dtype=np.int64),
                    edge_ptr=np.array(edge_ptr, dtype=np.int64),
                    n_blocks=n_blocks.astype(np.int64), **cat)


def build_tree(tree: str, shas: list[str], blocks: int, insns: int,
               windows: int, xsec: bool, extern: bool, out: Path,
               progress: int = 250) -> pd.DataFrame:
    root = TREES[tree] / "asm"
    dom, term, szb, flg, blen, seq = [], [], [], [], [], []
    esrc, edst, etype = [], [], []
    node_ptr, edge_ptr, nblk = [0], [0], []
    stats_rows = []
    t0 = time.time()
    for k, sha in enumerate(shas, 1):
        path = root / f"{sha}.asm"
        try:
            g = cfg_from_file(path, max_insns=insns, max_blocks=blocks,
                              include_calls=True, windows=windows,
                              cross_section=xsec, extern_nodes=extern)
        except Exception as exc:                      # pragma: no cover
            print(f"  !! {sha[:12]}: {type(exc).__name__}: {exc}", flush=True)
            g = None
        if g is None or not g.labels:
            node_ptr.append(node_ptr[-1])
            edge_ptr.append(edge_ptr[-1])
            nblk.append(0)
            stats_rows.append({"sha256": sha, "tree": tree})
            continue
        a = g.attrs
        dom.append(np.asarray(a["dom"], dtype=np.uint8))
        term.append(np.asarray(a["term"], dtype=np.uint8))
        szb.append(np.asarray(a["szb"], dtype=np.uint8))
        flg.append(np.asarray(a["flg"], dtype=np.uint8))
        blen.append(np.asarray(a["blen"], dtype=np.uint16))
        seq.append(np.asarray(a["seq"], dtype=np.uint32))
        if g.edges:
            e = np.asarray(g.edges, dtype=np.int32)
            esrc.append(e[:, 0])
            edst.append(e[:, 1])
            etype.append(np.asarray(g.etypes, dtype=np.uint8))
        node_ptr.append(node_ptr[-1] + len(g.labels))
        edge_ptr.append(edge_ptr[-1] + len(g.edges))
        nblk.append(g.n_blocks)
        row = {"sha256": sha, "tree": tree}
        row.update(g.stats)
        stats_rows.append(row)
        if k % progress == 0:
            print(f"  {tree}: {k}/{len(shas)}  "
                  f"{k/(time.time()-t0):.1f} files/s  "
                  f"{node_ptr[-1]/1e6:.1f}M nodes", flush=True)

    def _cat(parts, dtype):
        return (np.concatenate(parts) if parts
                else np.empty(0, dtype)).astype(dtype, copy=False)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             shas=np.array(shas, dtype="U64"),
             node_ptr=np.array(node_ptr, dtype=np.int64),
             edge_ptr=np.array(edge_ptr, dtype=np.int64),
             n_blocks=np.array(nblk, dtype=np.int32),
             dom=_cat(dom, np.uint8), term=_cat(term, np.uint8),
             szb=_cat(szb, np.uint8), flg=_cat(flg, np.uint8),
             blen=_cat(blen, np.uint16), seq=_cat(seq, np.uint32),
             esrc=_cat(esrc, np.int32), edst=_cat(edst, np.int32),
             etype=_cat(etype, np.uint8))
    print(f"  wrote {out.name}  ({out.stat().st_size/1e6:.0f} MB, "
          f"{node_ptr[-1]/1e6:.1f}M nodes, {edge_ptr[-1]/1e6:.1f}M edges, "
          f"{time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(stats_rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--max-blocks", type=int, default=20_000)
    ap.add_argument("--max-insns", type=int, default=80_000)
    ap.add_argument("--windows", type=int, default=1)
    ap.add_argument("--no-cross-section", action="store_true")
    ap.add_argument("--no-extern", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stats-out", default="")
    a = ap.parse_args()

    xsec, extern = not a.no_cross_section, not a.no_extern
    cache_dir = Path(a.cache_dir)
    need = needed_samples()
    print(f"{len(need)} unique samples; "
          f"blocks={a.max_blocks} insns={a.max_insns} windows={a.windows} "
          f"cross_section={xsec} extern={extern}")

    stats = []
    for tree in ("mendeley", "balanced_goodware"):
        shas = need.loc[need["tree"] == tree, "sha256"].tolist()
        if a.limit:
            shas = shas[:a.limit]
        out = cache_file(cache_dir, tree, a.max_blocks, a.max_insns,
                         a.windows, xsec, extern)
        if out.exists() and not a.force and not a.limit:
            print(f"  {tree}: cache present, skipping ({out.name})")
            continue
        print(f"building {tree}: {len(shas)} files -> {out.name}", flush=True)
        stats.append(build_tree(tree, shas, a.max_blocks, a.max_insns,
                                a.windows, xsec, extern, out))

    if stats and a.stats_out:
        df = pd.concat(stats, ignore_index=True)
        df = df.merge(need[["sha256", "label", "arch", "family", "source"]],
                      on="sha256", how="left")
        Path(a.stats_out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.stats_out, index=False)
        print(f"wrote {a.stats_out}  ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
