from pathlib import Path
import torch
import pandas as pd

from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEQUENCE_DIR = PROJECT_ROOT / "processed" / "sequences"
SPLIT_DIR = PROJECT_ROOT / "processed" / "splits"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)


records = []

for path in sorted(SEQUENCE_DIR.glob("*.pt")):
    sample = torch.load(path, weights_only=False)

    records.append({
        "file_id": sample["file_id"],
        "file_path": str(path),
        "label": int(sample["label"]),
    })

df = pd.DataFrame(records)

print("Usable samples:", len(df))
print("Labels:")
print(df["label"].value_counts())


# 70% training, 30% temporary set
train_df, temporary_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"],
)

# Split temporary set into 15% validation and 15% testing
validation_df, test_df = train_test_split(
    temporary_df,
    test_size=0.50,
    random_state=42,
    stratify=temporary_df["label"],
)


train_df.to_csv(SPLIT_DIR / "train.csv", index=False)
validation_df.to_csv(SPLIT_DIR / "validation.csv", index=False)
test_df.to_csv(SPLIT_DIR / "test.csv", index=False)


print("\nSplit sizes:")
print("Training:", len(train_df))
print("Validation:", len(validation_df))
print("Testing:", len(test_df))

print("\nTraining labels:")
print(train_df["label"].value_counts())

print("\nValidation labels:")
print(validation_df["label"].value_counts())

print("\nTesting labels:")
print(test_df["label"].value_counts())

