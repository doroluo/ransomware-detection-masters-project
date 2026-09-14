#!/usr/bin/env python3
"""
mn_to_features.py - turn an extract_unified.py output folder (mn/ + manifest.csv)
into the flat per-sample text layout that LLM_Features / the tokenization
pipeline consume, applying the cohort filter.

This is the "revised" feature variant:

    traditional  extract.py / asm_parse.py   linear sweep, stops at the first
                 (LLM_Features)               undecodable byte, full instruction
                                              per line ("mov eax, ebx")
    revised      extract_unified.py           skip-data sweep, uncapped,
                 (this script)                MNEMONIC ONLY per line ("mov")

Mnemonic-only is deliberate: operands carry register widths and address
sizes that leak the architecture (rbp / r8 / rip-relative), and the
ransomware side is 96% x86 while the goodware sides are 44-75% x64.

Cohort filter (see extract-opcode/WRITEUP.md section 5-6): a sample is kept
iff its manifest tag is `plain` or `upx_unpacked` and its family is not on
the exclude list. .NET, entropy-flagged (packed), UPX-not-unpacked, no-code,
odd-architecture, broken and duplicate rows are dropped. decoded_ratio is
NOT used - random bytes decode as x86 at ~0.99, so it cannot detect packing.

Output naming follows extract.py exactly: `<family>_<filename>.txt`, with
family = "root" for flat goodware folders and the bucket for Goodware_Balanced,
so files drop straight into the layouts the existing pipeline reads.

    python asm_tool/mn_to_features.py \
        --extract "C:/Users/chaoa/Downloads/asm and mm/Shared/Extract" \
        --out     "C:/Users/chaoa/Downloads/LLM_Features_Revised/Features_Extraction" \
        --exclude-family thanos

    python asm_tool/mn_to_features.py \
        --extract "C:/Users/chaoa/Downloads/asm and mm/Shared/Extract_Goodware_Balanced" \
        --out     "C:/Users/chaoa/Downloads/LLM_Features_Revised_Balanced" \
        --set-dir good_all

By default each manifest `set` (good_train, mal_test, ...) becomes a
sub-folder of --out. --set-dir NAME puts every kept sample in one folder
instead (Goodware_Balanced has no train/test split of its own).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

KEEP_TAGS = {"plain", "upx_unpacked"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", required=True,
                    help="extract_unified.py output folder (contains mn/ and manifest.csv)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set-dir", default="",
                    help="write every kept sample into this one sub-folder instead of per-set folders")
    ap.add_argument("--exclude-family", action="append", default=[],
                    help="drop this family entirely (repeatable), e.g. thanos")
    ap.add_argument("--keep-tags", default=",".join(sorted(KEEP_TAGS)),
                    help="comma-separated manifest tags to keep")
    ap.add_argument("--manifest-out", default="",
                    help="where to write the conversion manifest (default <out>/revised_manifest.csv)")
    args = ap.parse_args()

    src = Path(args.extract)
    manifest = src / "manifest.csv"
    mn_dir = src / "mn"
    if not manifest.is_file() or not mn_dir.is_dir():
        sys.exit(f"{src} must contain manifest.csv and mn/")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keep_tags = {t.strip() for t in args.keep_tags.split(",") if t.strip()}
    excl = {f.lower() for f in args.exclude_family}

    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    written: dict[str, str] = {}
    out_rows = []
    reasons = Counter()
    per_set = Counter()

    for r in rows:
        dest_set = args.set_dir or r["set"]
        fam = r["family"] if r["family"] else "root"
        txt_name = f"{fam.replace('/', '_')}_{r['filename']}.txt"
        rec = {"txt_file": txt_name, "set": r["set"], "out_dir": dest_set, "label": r["label"],
               "family": fam, "sha256": r["sha256"], "filename": r["filename"],
               "arch": r["arch"], "tag": r["tag"], "n_insns": r["n_insns"],
               "kept": 0, "reason": ""}

        if r["tag"] not in keep_tags:
            rec["reason"] = f"tag:{r['tag']}"
        elif fam.lower() in excl:
            rec["reason"] = f"family:{fam}"
        elif not r["mn_file"]:
            rec["reason"] = "no_mn_file"
        else:
            mn_path = src / r["mn_file"]
            if not mn_path.is_file():
                rec["reason"] = "mn_missing_on_disk"
            else:
                key = f"{dest_set}/{txt_name}"
                if key in written:
                    sys.exit(f"output name collision: {key}\n  {written[key]}\n  {r['sha256']}")
                written[key] = r["sha256"]
                mnems = mn_path.read_text(encoding="utf-8").split()
                if not mnems:
                    rec["reason"] = "empty_mn"
                else:
                    d = out / dest_set
                    d.mkdir(parents=True, exist_ok=True)
                    (d / txt_name).write_text("\n".join(mnems) + "\n", encoding="utf-8", newline="\n")
                    rec["kept"] = 1
                    rec["n_insns"] = len(mnems)
                    per_set[dest_set] += 1
        if not rec["kept"]:
            reasons[rec["reason"]] += 1
        out_rows.append(rec)

    mpath = Path(args.manifest_out) if args.manifest_out else out / "revised_manifest.csv"
    with mpath.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    print(f"{len(rows)} manifest rows -> {sum(per_set.values())} feature files under {out}")
    for k, v in sorted(per_set.items()):
        print(f"  {k:<12} {v}")
    print("dropped:", dict(sorted(reasons.items())))
    print(f"manifest: {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
