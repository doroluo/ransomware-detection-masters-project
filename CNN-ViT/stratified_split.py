import os
import random
import shutil
from collections import defaultdict


def stratified_split(source_root, output_root, val_size=0.10, test_size=0.10, seed=42):
    """
    Split class folders into train / val / test (default 80/10/10),
    keeping class balance. Copies each .png and its matching _vit_mask.npy.
    """
    by_class = defaultdict(list)

    for folder in sorted(os.listdir(source_root)):
        folder_path = os.path.join(source_root, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file.lower().endswith(".png"):
                by_class[folder].append(os.path.join(folder_path, file))

    classes = list(by_class.keys())
    total = sum(len(v) for v in by_class.values())
    print(f"Found {total} images in {len(classes)} classes: {classes}")
    if total == 0:
        print("Error: no .png files found.")
        return

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}

    for folder, paths in by_class.items():
        paths = paths[:]
        rng.shuffle(paths)
        n = len(paths)
        n_test = max(1, round(n * test_size)) if n >= 3 else 0
        n_val = max(1, round(n * val_size)) if n >= 3 else 0
        # keep at least one in train when possible
        if n_test + n_val >= n:
            n_test = max(0, n // 10)
            n_val = max(0, n // 10)

        test_paths = paths[:n_test]
        val_paths = paths[n_test:n_test + n_val]
        train_paths = paths[n_test + n_val:]

        splits["test"].extend(test_paths)
        splits["val"].extend(val_paths)
        splits["train"].extend(train_paths)
        print(f"  {folder}: train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}")

    for split_name, path_list in splits.items():
        for src_path in path_list:
            class_name = os.path.basename(os.path.dirname(src_path))
            dest_dir = os.path.join(output_root, split_name, class_name)
            os.makedirs(dest_dir, exist_ok=True)

            shutil.copy2(src_path, os.path.join(dest_dir, os.path.basename(src_path)))

            mask_src = src_path[:-4] + "_vit_mask.npy"  # foo.png -> foo_vit_mask.npy
            if os.path.exists(mask_src):
                shutil.copy2(mask_src, os.path.join(dest_dir, os.path.basename(mask_src)))

    print(
        f"Train: {len(splits['train'])}  "
        f"Val: {len(splits['val'])}  "
        f"Test: {len(splits['test'])}"
    )
    print(f"Wrote splits to: {output_root}")


if __name__ == "__main__":
    SOURCE = "/home/yl/quarantine/extract/training_dataset_sorted"
    OUTPUT = "/home/yl/quarantine/extract/evaluation_dataset_split"

    stratified_split(SOURCE, OUTPUT, val_size=0.10, test_size=0.10, seed=42)
