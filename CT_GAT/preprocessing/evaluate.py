from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from data_loader import test_loader
from device import get_device
from transformer_model import OpcodeTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "processed" / "checkpoints" / "best_transformer.pt"

def main():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run train.py first."
        )

    device = get_device()
    model = OpcodeTransformer()
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    )
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels, lengths in test_loader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)

            outputs = model(sequences, lengths)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_predictions)

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=["Benign", "Ransomware"],
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_predictions))


if __name__ == "__main__":
    main()