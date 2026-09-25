"""Opcode transformers for binary malware classification.

OpcodeTransformer — baseline: contiguous stride windows across the full ASM
(4x512=2048), mean-pool window embeddings, classify. No behavior channel.
BehaviorOpcodeTransformer — densest crypto/file windows, per-window malware
scores (max), and a behavior head (encrypt/decrypt/file read/write).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from token_mapping import BEHAVIOR_VOCAB_SIZE, HEAD_BEHAVIOR_NAMES

# From token_mapping.py: opcode/API IDs go up to 255; operand category IDs up to 75.
OPCODE_VOCAB_SIZE = 256
OPERAND_VOCAB_SIZE = 76
MAX_LEN = 2048
WINDOW_LEN = 512


class OpcodeTransformer(nn.Module):
    """Baseline: embed [opcode, op1, op2] over stride windows, mean-pool, classify.

    Accepts either:
      - legacy [B, L, 3] with lengths [B]
      - stride windows [B, W, L, 3] with lengths [B, W]
    """

    def __init__(
        self,
        opcode_vocab_size=OPCODE_VOCAB_SIZE,
        operand_vocab_size=OPERAND_VOCAB_SIZE,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_len=WINDOW_LEN,
        num_classes=2,
    ):
        super().__init__()

        self.max_len = max_len

        self.opcode_embedding = nn.Embedding(
            opcode_vocab_size, d_model, padding_idx=0
        )
        self.operand_embedding = nn.Embedding(
            operand_vocab_size, d_model, padding_idx=0
        )
        self.position_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def _encode_tokens(self, sequences, lengths):
        """sequences [B, L, 3], lengths [B] → pooled [B, D]."""
        opcode_ids = sequences[:, :, 0]
        operand1_ids = sequences[:, :, 1]
        operand2_ids = sequences[:, :, 2]

        x = (
            self.opcode_embedding(opcode_ids)
            + self.operand_embedding(operand1_ids)
            + self.operand_embedding(operand2_ids)
        )

        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.position_embedding(positions)

        padding_mask = positions.expand(batch_size, -1) >= lengths.unsqueeze(1)
        empty = lengths <= 0
        if empty.any():
            padding_mask = padding_mask.clone()
            padding_mask[empty] = True
            padding_mask[empty, 0] = False

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        valid = (~padding_mask).unsqueeze(-1).float()
        pooled = (x * valid).sum(dim=1) / lengths.clamp(min=1).unsqueeze(1).float()
        return pooled

    def forward(self, sequences, lengths):
        # Windowed baseline: [B, W, L, 3], lengths [B, W]
        if sequences.ndim == 4:
            batch_size, num_windows, seq_len, channels = sequences.shape
            flat = sequences.reshape(batch_size * num_windows, seq_len, channels)
            flat_lengths = lengths.reshape(batch_size * num_windows)
            pooled = self._encode_tokens(flat, flat_lengths)
            pooled = pooled.reshape(batch_size, num_windows, -1)
            window_mask = (lengths > 0).unsqueeze(-1).float()
            denom = window_mask.sum(dim=1).clamp(min=1.0)
            graph = (pooled * window_mask).sum(dim=1) / denom
            return self.classifier(graph)

        # Legacy single sequence: [B, L, 3], lengths [B]
        return self.classifier(self._encode_tokens(sequences, lengths))


class BehaviorOpcodeTransformer(nn.Module):
    """Windowed transformer with category channel + behavior head.

    Input windows: [B, W, L, 4]
    Returns:
      logits:           [B, 2]           — max over window malware scores
      window_logits:    [B, W, 2]        — per-window malware scores
      behavior_logits:  [B, W, 4]        — encrypt/decrypt/file_read/file_write
    """

    def __init__(
        self,
        opcode_vocab_size=OPCODE_VOCAB_SIZE,
        operand_vocab_size=OPERAND_VOCAB_SIZE,
        behavior_vocab_size=BEHAVIOR_VOCAB_SIZE,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_len=WINDOW_LEN,
        num_classes=2,
        num_behaviors=len(HEAD_BEHAVIOR_NAMES),
    ):
        super().__init__()
        self.max_len = max_len
        self.num_behaviors = num_behaviors

        self.opcode_embedding = nn.Embedding(
            opcode_vocab_size, d_model, padding_idx=0
        )
        self.operand_embedding = nn.Embedding(
            operand_vocab_size, d_model, padding_idx=0
        )
        self.behavior_embedding = nn.Embedding(
            behavior_vocab_size, d_model, padding_idx=0
        )
        self.position_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.window_classifier = nn.Linear(d_model, num_classes)
        self.behavior_head = nn.Linear(d_model, num_behaviors)

    def encode_windows(self, windows, lengths):
        """Encode [B, W, L, 4] → pooled [B, W, D], masks [B, W]."""
        batch_size, num_windows, seq_len, _ = windows.shape
        flat = windows.reshape(batch_size * num_windows, seq_len, -1)
        flat_lengths = lengths.reshape(batch_size * num_windows)

        x = (
            self.opcode_embedding(flat[:, :, 0])
            + self.operand_embedding(flat[:, :, 1])
            + self.operand_embedding(flat[:, :, 2])
            + self.behavior_embedding(flat[:, :, 3])
        )
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.position_embedding(positions)

        padding_mask = positions.expand(x.shape[0], -1) >= flat_lengths.unsqueeze(1)
        empty = flat_lengths <= 0
        if empty.any():
            padding_mask = padding_mask.clone()
            padding_mask[empty] = True
            padding_mask[empty, 0] = False

        x = self.transformer(x, src_key_padding_mask=padding_mask)
        valid = (~padding_mask).unsqueeze(-1).float()
        denom = flat_lengths.clamp(min=1).unsqueeze(1).float()
        pooled = (x * valid).sum(dim=1) / denom
        pooled = pooled.reshape(batch_size, num_windows, -1)
        window_mask = lengths > 0
        return pooled, window_mask

    def forward(self, windows, lengths):
        pooled, window_mask = self.encode_windows(windows, lengths)
        window_logits = self.window_classifier(pooled)
        behavior_logits = self.behavior_head(pooled)

        masked = window_logits.clone()
        masked[~window_mask] = -1e9
        ransomware_scores = masked[:, :, 1]
        best_index = ransomware_scores.argmax(dim=1)
        batch_indices = torch.arange(windows.shape[0], device=windows.device)
        logits = window_logits[batch_indices, best_index]
        all_empty = ~window_mask.any(dim=1)
        if all_empty.any():
            logits = logits.clone()
            logits[all_empty] = 0.0
        return logits, window_logits, behavior_logits


if __name__ == "__main__":
    from data_loader import behavior_train_loader, train_loader

    sequences, labels, lengths = next(iter(train_loader))
    baseline = OpcodeTransformer()
    print("Baseline input:", tuple(sequences.shape), "logits:", baseline(sequences, lengths).shape)

    windows, labels_b, win_lengths, targets, _ = next(iter(behavior_train_loader))
    model = BehaviorOpcodeTransformer()
    logits, window_logits, behavior_logits = model(windows, win_lengths)
    print("Behavior logits:", logits.shape)
    print("Window logits:", window_logits.shape)
    print("Behavior head:", behavior_logits.shape, HEAD_BEHAVIOR_NAMES)