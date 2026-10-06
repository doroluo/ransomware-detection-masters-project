#!/usr/bin/env python3
"""
render_vit_images.py

Renders tokenized opcode sequences as the image pair the CNN-ViT classifier
loads: a 256×256 grayscale PNG of token ids, and a 16×16 ViT patch mask.

The CNN-ViT's MalwareMaskedDataset looks one directory down for class
folders and reads `<sha256>.png` plus `<sha256>_vit_mask.npy` beside it.
This script writes:

    <out>/train/goodware/<sha256>.png
    <out>/train/goodware/<sha256>_vit_mask.npy
    <out>/train/ransomware/...
    <out>/val/...
    <out>/test/...

Point that loader's data root at `<out>`. Class folders sort alphabetically,
so goodware is class 0 and ransomware is class 1, matching the label column
in index.csv. The CNN-ViT training script was written for 9 malware families,
so its classification head has to be set to 2 classes for this dataset.

Token ids have to fit in an 8-bit image. Id 0 is padding, id 1 is an unknown
token, and ids 2..255 are the most common training tokens of whichever
tokenizer produced the pickle. Anything rarer collapses to unknown. The
ranking uses the train split only, same rule as the GIN vocabulary.

Padding is 0, not the texture noise the BIG 2015 parser adds. The mask is 0
on padded patches, which is what tells the ViT those patches are not code.

Usage:

    python render_vit_images.py --tokens data/tokens/sw/records.pkl
    python render_vit_images.py --tokens data/tokens/bpe/records.pkl --out data/vit_images/bpe
"""

import argparse
import json
import pickle
import struct
import sys
import zlib
from collections import Counter
from pathlib import Path


SQUARE_RESOLUTION = 256
TOTAL_TOKEN_CAPACITY = SQUARE_RESOLUTION * SQUARE_RESOLUTION
VIT_PATCH_SIZE = 16
PAD_ID = 0
UNK_ID = 1
MAX_TOKEN_ID = 255


def load_records(path):
    token_path = Path(path)
    if not token_path.is_file():
        raise SystemExit(f"{token_path} not found. Run tokenize_opcodes.py first.")
    with open(token_path, "rb") as handle:
        payload = pickle.load(handle)
    rows = payload.get("rows")
    if not rows:
        raise SystemExit(f"{token_path} has no token rows")
    return payload


def build_id_map(rows):
    """
    Maps the most common train tokens onto ids 2..255.
    Frequency first, then the token string, so ties do not depend on row order.
    """
    counts = Counter()
    for row in rows:
        if row["split"] != "train":
            continue
        counts.update(row["tokens"])
    if not counts:
        raise SystemExit("The token file has no training rows to build image ids from.")

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    id_map = {}
    next_id = UNK_ID + 1
    for token, _count in ranked:
        if next_id > MAX_TOKEN_ID:
            break
        id_map[token] = next_id
        next_id += 1
    return id_map


def ids_and_mask_length(tokens, id_map):
    """Integer ids, truncated to the canvas, and how many of them are real."""
    ids = [id_map.get(token, UNK_ID) for token in tokens]
    if len(ids) > TOTAL_TOKEN_CAPACITY:
        ids = ids[:TOTAL_TOKEN_CAPACITY]
    return ids, len(ids)


def patch_mask(real_tokens):
    """16×16 mask: a patch is active if it contains at least one real token."""
    side = SQUARE_RESOLUTION // VIT_PATCH_SIZE
    mask = [[0] * side for _ in range(side)]
    for index in range(real_tokens):
        row = index // SQUARE_RESOLUTION
        col = index % SQUARE_RESOLUTION
        mask[row // VIT_PATCH_SIZE][col // VIT_PATCH_SIZE] = 1
    return mask


def write_png(path, pixels):
    """Writes one 8-bit grayscale PNG. `pixels` is row-major, length 256*256."""
    def chunk(tag, data):
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    raw = bytearray()
    for row in range(SQUARE_RESOLUTION):
        start = row * SQUARE_RESOLUTION
        raw.append(0)  # filter: None
        raw.extend(pixels[start:start + SQUARE_RESOLUTION])
    ihdr = struct.pack(">IIBBBBB", SQUARE_RESOLUTION, SQUARE_RESOLUTION, 8, 0, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b"")
    path.write_bytes(png)


def save_mask(path, mask):
    import numpy as np

    array = np.array(mask, dtype=np.uint8)
    np.save(path, array)


def render(rows, id_map, out_dir):
    import csv

    out_dir = Path(out_dir)
    manifest = []
    total = len(rows)
    for i, row in enumerate(rows, start=1):
        label_name = row["label_name"]
        class_dir = out_dir / row["split"] / label_name
        class_dir.mkdir(parents=True, exist_ok=True)

        ids, real = ids_and_mask_length(row["tokens"], id_map)
        pixels = bytearray(TOTAL_TOKEN_CAPACITY)
        pixels[:real] = bytes(ids)

        image_path = class_dir / f"{row['sha256']}.png"
        mask_path = class_dir / f"{row['sha256']}_vit_mask.npy"
        write_png(image_path, pixels)
        save_mask(mask_path, patch_mask(real))
        manifest.append({
            "sha256": row["sha256"],
            "split": row["split"],
            "label": row["label"],
            "label_name": label_name,
            "family": row["family"],
            "n_tokens": len(row["tokens"]),
            "n_painted": real,
            "image": str(image_path),
            "mask": str(mask_path),
        })
        if i % 250 == 0 or i == total:
            print(f"  {i}/{total} images written", flush=True)

    manifest_path = out_dir / "images.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    return manifest_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Render tokenized opcode sequences as 256x256 CNN-ViT images and masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tokens", required=True,
                        help="records.pkl written by tokenize_opcodes.py")
    parser.add_argument("--out", default="",
                        help="image root (default: data/vit_images/<method>)")
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        import numpy  # noqa: F401
    except ImportError as exc:
        print("Rendering masks needs numpy. Install it with: pip install numpy", file=sys.stderr)
        return 1

    payload = load_records(args.tokens)
    rows = payload["rows"]
    id_map = build_id_map(rows)
    out_dir = Path(args.out) if args.out else Path("data/vit_images") / payload.get("method", "tokens")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "id_map.json", "w", encoding="utf-8") as handle:
        json.dump({"pad": PAD_ID, "unk": UNK_ID, "tokens": id_map}, handle)

    dropped = 0
    train_tokens = Counter()
    for row in rows:
        if row["split"] == "train":
            train_tokens.update(row["tokens"])
    if train_tokens:
        dropped = max(0, len(train_tokens) - len(id_map))

    print(f"Image vocabulary: {len(id_map)} tokens mapped to ids 2..{MAX_TOKEN_ID} "
          f"({dropped} rarer train tokens collapse to unknown)")
    print(f"Rendering {len(rows)} images into {out_dir}...")
    manifest_path = render(rows, id_map, out_dir)
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
