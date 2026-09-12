from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from data_loader import test_loader
from transformer_model import OpcodeTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "processed" / "checkpoints" / "best_transformer.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run train.py first."
        )

    model = OpcodeTransformer().to(DEVICE)
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels, lengths in test_loader:
            sequences = sequences.to(DEVICE)
            lengths = lengths.to(DEVICE)

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
