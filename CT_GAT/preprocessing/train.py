from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from data_loader import train_loader, val_loader
from transformer_model import OpcodeTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "processed" / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_transformer.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
LR = 1e-3


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Vocab sizes default from transformer_model / token_mapping.py (256 / 76)
    model = OpcodeTransformer().to(DEVICE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR)

    best_validation_loss = float("inf")

    print(f"Device: {DEVICE}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    for epoch in range(EPOCHS):
        model.train()
        total_training_loss = 0.0

        for sequences, labels, lengths in train_loader:
            sequences = sequences.to(DEVICE)
            labels = labels.to(DEVICE)
            lengths = lengths.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(sequences, lengths)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            total_training_loss += loss.item()

        model.eval()
        total_validation_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for sequences, labels, lengths in val_loader:
                sequences = sequences.to(DEVICE)
                labels = labels.to(DEVICE)
                lengths = lengths.to(DEVICE)

                outputs = model(sequences, lengths)
                loss = loss_function(outputs, labels)

                total_validation_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        average_training_loss = total_training_loss / max(len(train_loader), 1)
        average_validation_loss = total_validation_loss / max(len(val_loader), 1)
        validation_accuracy = correct / max(total, 1)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {average_training_loss:.4f} | "
            f"Validation Loss: {average_validation_loss:.4f} | "
            f"Validation Accuracy: {validation_accuracy:.4f}"
        )

        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"Saved best model -> {CHECKPOINT_PATH}")

    print("Training complete.")


if __name__ == "__main__":
    main()
