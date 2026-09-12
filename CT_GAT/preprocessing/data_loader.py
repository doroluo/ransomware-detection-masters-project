from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "processed" / "splits"
MAX_LEN = 2048


class OpcodeDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = torch.load(row["file_path"], weights_only=True)
        sequence = data["sequence"].long()
        length = min(sequence.shape[0], MAX_LEN)
        sequence = sequence[:MAX_LEN]

        if sequence.shape[0] < MAX_LEN:
            padding = torch.zeros(MAX_LEN - sequence.shape[0], sequence.shape[1], dtype=torch.long)
            sequence = torch.cat([sequence, padding], dim=0)

        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return sequence, label, length

train_dataset = OpcodeDataset(csv_path=str(SPLITS_DIR / "train.csv"))
val_dataset = OpcodeDataset(csv_path=str(SPLITS_DIR / "validation.csv"))
test_dataset = OpcodeDataset(csv_path=str(SPLITS_DIR / "test.csv"))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

if __name__ == "__main__":
    sequences, labels, lengths = next(iter(train_loader))
    print("Sequence shape:", sequences.shape)
    print("Label shape:", labels.shape)
    print("Lengths:", lengths)
    print("Labels:", labels)
