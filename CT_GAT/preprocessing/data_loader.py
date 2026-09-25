"""PyTorch Dataset / DataLoader for opcode sequences.

Baseline loader: contiguous windows spaced across the *full* ASM (stride /
even coverage), budget MAX_WINDOWS * WINDOW_LEN = 2048. No behavior heuristic.
Behavior loader: same budget, but windows chosen by densest crypto/file regions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from token_mapping import (
    BEHAVIOR_MAP,
    HEAD_BEHAVIOR_IDS,
    SUSPICIOUS_BEHAVIOR_IDS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "processed" / "splits"
MAX_LEN = 2048
WINDOW_LEN = 512
MAX_WINDOWS = 4


def _ensure_channels(sequence: torch.Tensor) -> torch.Tensor:
    """Accept legacy [N, 3] tensors by padding a NONE behavior channel."""
    if sequence.ndim != 2:
        raise ValueError(f"Expected [N, C] sequence, got {tuple(sequence.shape)}")
    if sequence.shape[1] == 4:
        return sequence
    if sequence.shape[1] == 3:
        none = torch.full(
            (sequence.shape[0], 1),
            BEHAVIOR_MAP["NONE"],
            dtype=sequence.dtype,
        )
        return torch.cat([sequence, none], dim=1)
    raise ValueError(f"Unsupported channel count: {sequence.shape[1]}")


def _pad_window(chunk: torch.Tensor, window_len: int) -> tuple[torch.Tensor, int]:
    length = int(chunk.shape[0])
    if length < window_len:
        pad = torch.zeros(window_len - length, chunk.shape[1], dtype=torch.long)
        chunk = torch.cat([chunk, pad], dim=0)
    return chunk, length


def pick_stride_windows(
    sequence: torch.Tensor,
    window_len: int = WINDOW_LEN,
    max_windows: int = MAX_WINDOWS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contiguous windows spaced evenly across the full sequence.

    Same 2048 budget as the behavior path, but *no* suspicious-token scoring —
    just stride/coverage so the baseline sees early, middle, and late code.

    For a file of length N:
      starts ≈ linspace(0, N - window_len, max_windows)
    so the gap between windows is the adaptive stride.

    Returns windows [W, L, C], lengths [W].
    """
    sequence = sequence.long()
    n = int(sequence.shape[0])
    channels = int(sequence.shape[1])

    if n == 0:
        starts: list[int] = []
    elif n <= window_len or max_windows <= 1:
        starts = [0]
    else:
        max_start = n - window_len
        starts = [
            int(round(i * max_start / (max_windows - 1)))
            for i in range(max_windows)
        ]
        deduped: list[int] = []
        for start in starts:
            if not deduped or start != deduped[-1]:
                deduped.append(start)
        starts = deduped

    windows = []
    lengths = []
    for start in starts[:max_windows]:
        chunk, length = _pad_window(sequence[start : start + window_len], window_len)
        windows.append(chunk)
        lengths.append(length)

    while len(windows) < max_windows:
        windows.append(torch.zeros(window_len, channels, dtype=torch.long))
        lengths.append(0)

    return (
        torch.stack(windows, dim=0),
        torch.tensor(lengths, dtype=torch.long),
    )


def pick_behavior_windows(
    sequence: torch.Tensor,
    window_len: int = WINDOW_LEN,
    max_windows: int = MAX_WINDOWS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scan the *entire* sequence, then keep up to ``max_windows`` chunks.

    Budget is ``max_windows * window_len`` (default 4 x 512 = 2048).

    Steps:
      1. Mark every crypto/file-tagged instruction in the full file.
      2. Score every possible aligned window by how many tagged ops it covers.
      3. Greedily take the highest-scoring non-overlapping windows.
      4. If nothing is tagged, fall back to stride coverage (same as baseline).

    Returns windows [W, L, 4], lengths [W], behavior targets [W, 4].
    """
    sequence = _ensure_channels(sequence.long())
    n = int(sequence.shape[0])
    behaviors = sequence[:, 3]
    suspicious = torch.zeros(n, dtype=torch.bool)
    for behavior_id in SUSPICIOUS_BEHAVIOR_IDS:
        suspicious |= behaviors == behavior_id

    step = max(1, window_len // 4)
    candidates: list[tuple[int, int]] = []
    if n == 0:
        starts: list[int] = []
    elif not bool(suspicious.any()):
        stride_windows, stride_lengths = pick_stride_windows(
            sequence, window_len=window_len, max_windows=max_windows
        )
        targets = torch.zeros(max_windows, len(HEAD_BEHAVIOR_IDS), dtype=torch.float)
        return stride_windows, stride_lengths, targets
    else:
        sus_prefix = torch.zeros(n + 1, dtype=torch.long)
        sus_prefix[1:] = suspicious.long().cumsum(0)
        max_start = 0 if n <= window_len else n - window_len
        for start in range(0, max_start + 1, step):
            end = min(start + window_len, n)
            score = int(sus_prefix[end] - sus_prefix[start])
            if score > 0:
                candidates.append((score, start))
        for index in suspicious.nonzero(as_tuple=False).flatten().tolist():
            start = max(0, int(index) - window_len // 4)
            if n > window_len:
                start = min(start, n - window_len)
            else:
                start = 0
            end = min(start + window_len, n)
            score = int(sus_prefix[end] - sus_prefix[start])
            candidates.append((score, start))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        starts = []
        for score, start in candidates:
            if any(abs(start - kept) < window_len for kept in starts):
                continue
            starts.append(start)
            if len(starts) >= max_windows:
                break
        if not starts:
            starts = [0]

    windows = []
    lengths = []
    targets = []
    head_ids = list(HEAD_BEHAVIOR_IDS)
    for start in starts[:max_windows]:
        chunk, length = _pad_window(sequence[start : start + window_len], window_len)
        windows.append(chunk)
        lengths.append(length)
        present = chunk[:length, 3]
        target = torch.zeros(len(head_ids), dtype=torch.float)
        for i, behavior_id in enumerate(head_ids):
            target[i] = float((present == behavior_id).any())
        targets.append(target)

    while len(windows) < max_windows:
        windows.append(torch.zeros(window_len, sequence.shape[1], dtype=torch.long))
        lengths.append(0)
        targets.append(torch.zeros(len(HEAD_BEHAVIOR_IDS), dtype=torch.float))

    return (
        torch.stack(windows, dim=0),
        torch.tensor(lengths, dtype=torch.long),
        torch.stack(targets, dim=0),
    )


class OpcodeDataset(Dataset):
    """Baseline: contiguous stride windows across the full file (no behavior filter)."""

    def __init__(
        self,
        csv_path,
        window_len: int = WINDOW_LEN,
        max_windows: int = MAX_WINDOWS,
    ):
        self.df = pd.read_csv(csv_path)
        self.window_len = window_len
        self.max_windows = max_windows

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = torch.load(row["file_path"], weights_only=True)
        sequence = _ensure_channels(data["sequence"].long())[:, :3]
        windows, lengths = pick_stride_windows(
            sequence,
            window_len=self.window_len,
            max_windows=self.max_windows,
        )
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return windows, label, lengths


class BehaviorOpcodeDataset(Dataset):
    """Behavior path: windows chosen by densest crypto/file regions."""

    def __init__(self, csv_path, window_len: int = WINDOW_LEN, max_windows: int = MAX_WINDOWS):
        self.df = pd.read_csv(csv_path)
        self.window_len = window_len
        self.max_windows = max_windows

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = torch.load(row["file_path"], weights_only=True)
        sequence = _ensure_channels(data["sequence"].long())
        windows, lengths, behavior_targets = pick_behavior_windows(
            sequence,
            window_len=self.window_len,
            max_windows=self.max_windows,
        )
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        file_id = str(data.get("file_id", row.get("file_id", "")))
        return windows, label, lengths, behavior_targets, file_id


train_dataset = OpcodeDataset(csv_path=str(SPLITS_DIR / "train.csv"))
val_dataset = OpcodeDataset(csv_path=str(SPLITS_DIR / "validation.csv"))
test_dataset = OpcodeDataset(csv_path=str(SPLITS_DIR / "test.csv"))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

behavior_train_dataset = BehaviorOpcodeDataset(csv_path=str(SPLITS_DIR / "train.csv"))
behavior_val_dataset = BehaviorOpcodeDataset(csv_path=str(SPLITS_DIR / "validation.csv"))
behavior_test_dataset = BehaviorOpcodeDataset(csv_path=str(SPLITS_DIR / "test.csv"))

behavior_train_loader = DataLoader(behavior_train_dataset, batch_size=4, shuffle=True)
behavior_val_loader = DataLoader(behavior_val_dataset, batch_size=4, shuffle=False)
behavior_test_loader = DataLoader(behavior_test_dataset, batch_size=4, shuffle=False)


if __name__ == "__main__":
    sequences, labels, lengths = next(iter(train_loader))
    print("Baseline windows shape:", sequences.shape)
    print("Label shape:", labels.shape)
    print("Lengths:", lengths)

    windows, labels_b, win_lengths, targets, file_ids = next(iter(behavior_train_loader))
    print("Behavior windows shape:", windows.shape)
    print("Window lengths:", win_lengths[0])
    print("Behavior targets[0]:", targets[0])
    print("File ids:", file_ids)
