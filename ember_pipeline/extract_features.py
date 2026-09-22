#!/usr/bin/env python3
"""
extract_features.py - EMBER-branch static features, one row per PE, keyed by SHA-256.

Uses the EMBER branch's extractor unchanged (ember_extractor.extract_features_single,
537 dims: 256 byte histogram + 256 byte-entropy histogram + 21 structural + 4 string)
so results stay comparable with that branch. What changes is bookkeeping: every
file gets a manifest row (ok or error), features are keyed by sha256 so the
cohort filter and the family split can be applied afterwards, and nothing is
executed - files are read as bytes and parsed with LIEF.

    python ember_pipeline/extract_features.py --in DIR --out features.npz --label 0 \
        [--set NAME] [--exclude _upx_packed --exclude flagged] [--sanitize]

Output: <out>.npz with X (n,537) float32, sha256 (n,), rel_path (n,), label (n,)
        <out>.manifest.csv with one row per input file: sha256, rel_path, family,
        set, label, size, status
"""
from __future__ import annotations
import argparse, csv, hashlib, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ember_pipeline.ember_extractor import extract_features_single  # noqa: E402

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True, help="path without extension, e.g. features/mendeley_good_train")
    ap.add_argument("--label", type=int, required=True)
    ap.add_argument("--set", default="")
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--sanitize", action="store_true", help="strip overlay bytes (EMBER branch default is off)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.inp); excl = set(a.exclude)
    files, rows = [], []
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if not d.startswith(".") and d not in excl)
        for fn in sorted(fns):
            p = Path(dp) / fn
            rel = p.relative_to(root).as_posix()
            fam = rel.split("/", 1)[0] if "/" in rel else "root"
            try:
                with p.open("rb") as fh:
                    magic = fh.read(2)
            except OSError as e:
                rows.append({"sha256": "", "rel_path": rel, "family": fam, "set": a.set,
                             "label": a.label, "size": 0, "status": f"read_error:{type(e).__name__}"})
                continue
            if magic != b"MZ":
                rows.append({"sha256": "", "rel_path": rel, "family": fam, "set": a.set,
                             "label": a.label, "size": p.stat().st_size, "status": "not_pe"})
                continue
            files.append(p)
    if a.limit: files = files[:a.limit]
    print(f"{len(files)} PEs under {root}")
    X, shas, rels, labels = [], [], [], []
    t0 = time.time()
    for i, p in enumerate(files, 1):
        rel = p.relative_to(root).as_posix()
        fam = rel.split("/", 1)[0] if "/" in rel else "root"     # family = top-level folder
        sha = sha256_file(p)
        row = {"sha256": sha, "rel_path": rel, "family": fam, "set": a.set, "label": a.label,
               "size": p.stat().st_size, "status": "ok"}
        try:
            vec = extract_features_single(str(p), sanitize=a.sanitize)
            if vec is None or len(vec) != 537:
                row["status"] = "error:no_vector" if vec is None else f"error:dim{len(vec)}"
            else:
                X.append(np.asarray(vec, dtype=np.float32)); shas.append(sha); rels.append(rel); labels.append(a.label)
        except Exception as e:
            row["status"] = f"error:{type(e).__name__}:{str(e)[:60]}"
        rows.append(row)
        if i % 100 == 0 or i == len(files):
            print(f"  {i}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
    if not rows:
        sys.exit(f"nothing under {root}: wrong --in path?")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out) + ".npz", X=np.vstack(X) if X else np.zeros((0, 537), np.float32),
                        sha256=np.array(shas), rel_path=np.array(rels), label=np.array(labels, np.int32))
    with open(str(out) + ".manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    err = sum(r["status"] != "ok" for r in rows)
    print(f"wrote {len(X)} vectors -> {out}.npz ; errors {err}/{len(rows)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
