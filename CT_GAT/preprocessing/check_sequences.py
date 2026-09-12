from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQUENCES_DIR = PROJECT_ROOT / "processed" / "sequences"

sequences_files = sorted(SEQUENCES_DIR.glob("*.pt"))

print("Number of sequences files:", len(sequences_files))

if not sequences_files:
    raise FileNotFoundError("No sequences files found.")

sample_path = sequences_files[0]
sample = torch.load(sample_path, weights_only=False)

print("\nFile:", sample_path.name)
print("File ID:", sample["file_id"])
print("Label:", sample["label"])
print("Sequence shape:", sample["sequence"].shape)
print("Sequence type:", sample["sequence"].dtype)

print("\nFirst 10 instructions:")
print(sample["sequence"][:10])

print("\nLast 10 instructions:")
print(sample["sequence"][-10:])