import torch
import torch.nn as nn

# From token_mapping.py: opcode/API IDs go up to 255; operand category IDs up to 75.
OPCODE_VOCAB_SIZE = 256
OPERAND_VOCAB_SIZE = 76
MAX_LEN = 2048


class OpcodeTransformer(nn.Module):
    def __init__(
        self,
        opcode_vocab_size=OPCODE_VOCAB_SIZE,
        operand_vocab_size=OPERAND_VOCAB_SIZE,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_len=MAX_LEN,
        num_classes=2,
    ):
        super().__init__()

        self.max_len = max_len

        # sequences[:, :, 0] = opcode or API id
        self.opcode_embedding = nn.Embedding(
            opcode_vocab_size, d_model, padding_idx=0
        )
        # sequences[:, :, 1] and [:, :, 2] = operand category ids
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

    def forward(self, sequences, lengths):
        # sequences: [B, L, 3] = [opcode_or_api, operand1, operand2]
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

        # True where position is padding
        padding_mask = positions.expand(batch_size, -1) >= lengths.unsqueeze(1)

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Mean-pool over real (non-pad) tokens
        valid = (~padding_mask).unsqueeze(-1).float()
        x = (x * valid).sum(dim=1) / lengths.clamp(min=1).unsqueeze(1).float()

        return self.classifier(x)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
    from data_loader import train_loader

    sequences, labels, lengths = next(iter(train_loader))
    model = OpcodeTransformer()
    logits = model(sequences, lengths)
    print("Input:", sequences.shape)
    print("Logits:", logits.shape)
    print("Labels:", labels)
