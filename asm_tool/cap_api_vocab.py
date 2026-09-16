#!/usr/bin/env python3
"""
cap_api_vocab.py - `mn_api_top/`: the API-inlined stream with rare API names
collapsed, so the sequence model's vocabulary stays small.

    python asm_tool/cap_api_vocab.py --min-df 5

`mn_api/` carries about 20,000 distinct `mnemonic:dll!func` tokens on the
Mendeley cohort alone and more on Goodware_Balanced (C++ mangled exports).
That is fine for TF-IDF, and too many for the masked-token pretraining head
(a softmax over the vocabulary at every one of 4,096 positions) and for the
token cache's 16-bit global ids. Here every API token whose DOCUMENT
frequency over the cohort files is below --min-df is rewritten to
`<mnemonic>:api_rare`, so the instruction still says "this calls an import"
and only the name is dropped. Everything else is copied unchanged; token
count and line structure are identical to `mn_api/`.

The document frequency is counted over all cohort files of both corpora
without reading a label - the same unsupervised, transductive footing as the
pretraining pass, and recorded as such in the stats.

Output
    <shared>/Extract/mn_api_top/<sha256>.txt
    <shared>/Extract_Goodware_Balanced/mn_api_top/<sha256>.txt
    <shared>/mn_api_top_vocab.csv      token, doc_freq, kept
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from asm_tool.rewrite_streams import TREES, cohort_files  # noqa: E402

SRC, DST = "mn_api", "mn_api_top"
RARE = "api_rare"


def _df_one(path: str) -> set:
    return {t for t in Path(path).read_text(encoding="utf-8", errors="replace").split() if ":" in t}


def _rewrite_one(args) -> tuple:
    src, dst, keep = args
    src, dst = Path(src), Path(dst)
    out_lines, n_api, n_rare = [], 0, 0
    for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
        toks = line.split()
        for i, t in enumerate(toks):
            if ":" in t:
                n_api += 1
                if t not in keep:
                    toks[i] = t.split(":", 1)[0] + ":" + RARE
                    n_rare += 1
        out_lines.append(" ".join(toks))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return src.stem, n_api, n_rare


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    t0 = time.time()
    files = [(sha, tree) for sha, tree in cohort_files()]
    paths = [str(TREES[tree] / SRC / f"{sha}.txt") for sha, tree in files]
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} {SRC} files missing, e.g. {missing[:2]}; run rewrite_streams.py first")
    df: collections.Counter = collections.Counter()
    with Pool(a.workers) as pool:
        for s in pool.imap_unordered(_df_one, paths, chunksize=8):
            df.update(s)
    keep = frozenset(t for t, n in df.items() if n >= a.min_df)
    print(f"{len(files)} files; {len(df)} distinct API tokens, {len(keep)} kept at doc-freq >= {a.min_df} "
          f"({time.time()-t0:.0f}s)", flush=True)
    shared = TREES["mendeley"].parent
    with (shared / f"{DST}_vocab.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["token", "doc_freq", "kept"])
        for t, n in sorted(df.items(), key=lambda x: (-x[1], x[0])):
            w.writerow([t, n, int(t in keep)])
    tasks = [(p, str(TREES[tree] / DST / f"{sha}.txt"), keep) for p, (sha, tree) in zip(paths, files)]
    n_api = n_rare = 0
    with Pool(a.workers) as pool:
        for k, (_, na, nr) in enumerate(pool.imap_unordered(_rewrite_one, tasks, chunksize=8), 1):
            n_api += na; n_rare += nr
            if k % 500 == 0:
                print(f"  {k}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"done: {n_rare:,} of {n_api:,} API occurrences collapsed to {RARE} "
          f"({100*n_rare/max(n_api,1):.1f}%), {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
