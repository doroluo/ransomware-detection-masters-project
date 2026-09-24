from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from device import get_device
from gat_model import OpcodeGAT
from graph_loader import train_loader, val_loader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "processed" / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_gat.pt"

EPOCHS = 10
LR = 1e-3


def main():
    if len(train_loader.dataset) == 0:
        raise FileNotFoundError(
            "No training graphs found. Run: python build_pyg_graphs.py"
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    if device.type == "privateuseone":
        print("DirectML does not support PyG scatter/GATConv; using CPU for GAT.")
        device = torch.device("cpu")

    model = OpcodeGAT().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR)

    best_validation_loss = float("inf")

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"Train graphs: {len(train_loader.dataset)} | Val graphs: {len(val_loader.dataset)}")

    for epoch in range(EPOCHS):
        model.train()
        total_training_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = loss_function(outputs, batch.y)
            loss.backward()
            optimizer.step()
            total_training_loss += loss.item()

        model.eval()
        total_validation_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch)
                loss = loss_function(outputs, batch.y)
                total_validation_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                correct += (predictions == batch.y).sum().item()
                total += batch.y.numel()

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
            torch.save(
                {key: value.detach().cpu() for key, value in model.state_dict().items()},
                CHECKPOINT_PATH,
            )
            print(f"Saved best model -> {CHECKPOINT_PATH}")

    print("Training complete.")


if __name__ == "__main__":
    main()
