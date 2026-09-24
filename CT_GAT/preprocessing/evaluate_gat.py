from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from device import get_device
from gat_model import OpcodeGAT
from graph_loader import test_loader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "processed" / "checkpoints" / "best_gat.pt"


def resolve_device():
    device = get_device()
    if device.type == "privateuseone":
        print("DirectML does not support PyG scatter/GATConv; using CPU for GAT.")
        return torch.device("cpu")
    return device


def main():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run train_gat.py first."
        )
    if len(test_loader.dataset) == 0:
        raise FileNotFoundError(
            "No test graphs found. Run: python build_pyg_graphs.py"
        )

    device = resolve_device()
    model = OpcodeGAT()
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    )
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)
            predictions = outputs.argmax(dim=1)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())

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
