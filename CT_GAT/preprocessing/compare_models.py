"""Train / evaluate baseline vs behavior-aware Transformer and GAT.

Same split CSVs. Reports ransomware F1 for each model, then explains one
ransomware sample with top windows (Transformer) and top blocks + read→write
path (GAT).

Usage:
    python compare_models.py
    python compare_models.py --epochs 5 --skip-train
    python compare_models.py --sample-id <file_id>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch_geometric.loader import DataLoader as GeoDataLoader

from build_graphs import find_read_to_write_paths
from build_pyg_graphs import build_tokenized_cfg
from data_loader import (
    behavior_test_loader,
    behavior_train_loader,
    behavior_val_loader,
    test_loader as tf_test_loader,
    train_loader as tf_train_loader,
    val_loader as tf_val_loader,
    BehaviorOpcodeDataset,
    SPLITS_DIR,
)
from device import get_device
from gat_model import (
    BehaviorOpcodeGAT,
    OpcodeGAT,
    block_behavior_targets,
    top_blocks_report,
)
from graph_loader import OpcodeGraphDataset, load_graph, GRAPH_DIR
from token_mapping import HEAD_BEHAVIOR_IDS, HEAD_BEHAVIOR_NAMES
from transformer_model import BehaviorOpcodeTransformer, OpcodeTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "processed" / "checkpoints"
LR = 1e-3
# Keep binary CE dominant; behavior aux is only a light regularizer.
BEHAVIOR_AUX_WEIGHT = 0.1
GAT_BEHAVIOR_PATIENCE = 3


def _resolve_gat_device():
    device = get_device()
    if device.type == "privateuseone":
        print("DirectML does not support PyG scatter/GATConv; using CPU for GAT.")
        return torch.device("cpu")
    return device


def train_transformer_baseline(device, epochs: int) -> Path:
    path = CHECKPOINT_DIR / "best_transformer.pt"
    model = OpcodeTransformer().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR)
    best = float("inf")
    for epoch in range(epochs):
        model.train()
        for sequences, labels, lengths in tf_train_loader:
            sequences, labels, lengths = (
                sequences.to(device),
                labels.to(device),
                lengths.to(device),
            )
            optimizer.zero_grad()
            loss = loss_fn(model(sequences, lengths), labels)
            loss.backward()
            optimizer.step()
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for sequences, labels, lengths in tf_val_loader:
                sequences, labels, lengths = (
                    sequences.to(device),
                    labels.to(device),
                    lengths.to(device),
                )
                total_loss += loss_fn(model(sequences, lengths), labels).item()
        avg = total_loss / max(len(tf_val_loader), 1)
        print(f"[TF baseline] epoch {epoch + 1}/{epochs} val_loss={avg:.4f}")
        if avg < best:
            best = avg
            torch.save(
                {k: v.detach().cpu() for k, v in model.state_dict().items()},
                path,
            )
    return path


def train_transformer_behavior(device, epochs: int) -> Path:
    path = CHECKPOINT_DIR / "best_transformer_behavior.pt"
    model = BehaviorOpcodeTransformer().to(device)
    cls_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LR)
    best = float("inf")
    for epoch in range(epochs):
        model.train()
        for windows, labels, lengths, behavior_targets, _ in behavior_train_loader:
            windows = windows.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            behavior_targets = behavior_targets.to(device)
            optimizer.zero_grad()
            logits, _, behavior_logits = model(windows, lengths)
            # Supervise behavior head on real (non-empty) windows only.
            mask = lengths > 0
            loss = cls_loss(logits, labels)
            if mask.any():
                loss = loss + bce_loss(
                    behavior_logits[mask], behavior_targets[mask]
                )
            loss.backward()
            optimizer.step()
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for windows, labels, lengths, behavior_targets, _ in behavior_val_loader:
                windows = windows.to(device)
                labels = labels.to(device)
                lengths = lengths.to(device)
                behavior_targets = behavior_targets.to(device)
                logits, _, behavior_logits = model(windows, lengths)
                mask = lengths > 0
                loss = cls_loss(logits, labels)
                if mask.any():
                    loss = loss + bce_loss(
                        behavior_logits[mask], behavior_targets[mask]
                    )
                total_loss += loss.item()
        avg = total_loss / max(len(behavior_val_loader), 1)
        print(f"[TF behavior] epoch {epoch + 1}/{epochs} val_loss={avg:.4f}")
        if avg < best:
            best = avg
            torch.save(
                {k: v.detach().cpu() for k, v in model.state_dict().items()},
                path,
            )
    return path


def train_gat_baseline(device, epochs: int) -> Path:
    path = CHECKPOINT_DIR / "best_gat.pt"
    train_ds = OpcodeGraphDataset(SPLITS_DIR / "train.csv")
    val_ds = OpcodeGraphDataset(SPLITS_DIR / "validation.csv")
    if len(train_ds) == 0:
        raise FileNotFoundError("No training graphs. Run build_pyg_graphs.py --overwrite")
    train_loader = GeoDataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = GeoDataLoader(val_ds, batch_size=8, shuffle=False)
    model = OpcodeGAT().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR)
    best = float("inf")
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch.y)
            loss.backward()
            optimizer.step()
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                total_loss += loss_fn(model(batch), batch.y).item()
        avg = total_loss / max(len(val_loader), 1)
        print(f"[GAT baseline] epoch {epoch + 1}/{epochs} val_loss={avg:.4f}")
        if avg < best:
            best = avg
            torch.save(
                {k: v.detach().cpu() for k, v in model.state_dict().items()},
                path,
            )
    return path


def train_gat_behavior(device, epochs: int) -> Path:
    path = CHECKPOINT_DIR / "best_gat_behavior.pt"
    train_ds = OpcodeGraphDataset(SPLITS_DIR / "train.csv")
    val_ds = OpcodeGraphDataset(SPLITS_DIR / "validation.csv")
    if len(train_ds) == 0:
        raise FileNotFoundError("No training graphs. Run build_pyg_graphs.py --overwrite")
    train_loader = GeoDataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = GeoDataLoader(val_ds, batch_size=8, shuffle=False)
    model = BehaviorOpcodeGAT().to(device)
    cls_loss = nn.CrossEntropyLoss()
    # pos_weight softens the all-negative majority inside tagged blocks.
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LR)
    best = float("inf")
    stale = 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, block_logits, behavior_logits = model(batch)
            behavior_targets = block_behavior_targets(batch, HEAD_BEHAVIOR_IDS)
            # Only supervise blocks that actually carry crypto/file signal.
            interesting = (
                (behavior_targets.sum(dim=1) > 0)
                | (batch.block_crypto_count > 0)
                | (batch.block_file_count > 0)
            )
            loss = cls_loss(logits, batch.y)
            if interesting.any():
                # Calibrate explanation heads only where crypto/file signal exists.
                block_labels = batch.y[batch.batch]
                loss = loss + BEHAVIOR_AUX_WEIGHT * cls_loss(
                    block_logits[interesting], block_labels[interesting]
                )
                loss = loss + BEHAVIOR_AUX_WEIGHT * bce_loss(
                    behavior_logits[interesting],
                    behavior_targets[interesting],
                )
            loss.backward()
            optimizer.step()
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits, _, _ = model(batch)
                total_loss += cls_loss(logits, batch.y).item()
        avg = total_loss / max(len(val_loader), 1)
        print(f"[GAT behavior] epoch {epoch + 1}/{epochs} val_loss={avg:.4f}")
        if avg < best:
            best = avg
            stale = 0
            torch.save(
                {k: v.detach().cpu() for k, v in model.state_dict().items()},
                path,
            )
        else:
            stale += 1
            if stale >= GAT_BEHAVIOR_PATIENCE:
                print(
                    f"[GAT behavior] early stop at epoch {epoch + 1} "
                    f"(best val_loss={best:.4f})"
                )
                break
    return path


def eval_transformer_baseline(device, path: Path) -> float:
    model = OpcodeTransformer()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    preds, labels = [], []
    with torch.no_grad():
        for sequences, y, lengths in tf_test_loader:
            out = model(sequences.to(device), lengths.to(device))
            preds.extend(out.argmax(1).cpu().tolist())
            labels.extend(y.tolist())
    return _ransomware_f1(labels, preds)


def eval_transformer_behavior(device, path: Path) -> float:
    model = BehaviorOpcodeTransformer()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    preds, labels = [], []
    with torch.no_grad():
        for windows, y, lengths, _, _ in behavior_test_loader:
            logits, _, _ = model(windows.to(device), lengths.to(device))
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(y.tolist())
    return _ransomware_f1(labels, preds)


def eval_gat(device, path: Path, behavior: bool) -> float:
    test_ds = OpcodeGraphDataset(SPLITS_DIR / "test.csv")
    loader = GeoDataLoader(test_ds, batch_size=8, shuffle=False)
    model = BehaviorOpcodeGAT() if behavior else OpcodeGAT()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            logits = out[0] if behavior else out
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(batch.y.cpu().tolist())
    return _ransomware_f1(labels, preds)


def _ransomware_f1(labels, preds) -> float:
    print(
        classification_report(
            labels,
            preds,
            target_names=["Benign", "Ransomware"],
            zero_division=0,
        )
    )
    return float(f1_score(labels, preds, pos_label=1, zero_division=0))


def pick_ransomware_sample(sample_id: str | None) -> tuple[str, Path]:
    import pandas as pd

    df = pd.read_csv(SPLITS_DIR / "test.csv")
    ransomware = df[df["label"] == 1]
    if ransomware.empty:
        raise FileNotFoundError("No ransomware rows in test.csv")
    if sample_id:
        hit = ransomware[ransomware["file_id"].astype(str) == sample_id]
        if hit.empty:
            raise FileNotFoundError(f"Sample {sample_id} not in test ransomware set")
        row = hit.iloc[0]
    else:
        row = ransomware.iloc[0]
    file_id = str(row["file_id"])
    graph_path = GRAPH_DIR / f"{file_id}.pt"
    return file_id, graph_path


def explain_sample(device, file_id: str, graph_path: Path, tf_ckpt: Path, gat_ckpt: Path):
    print("\n=== Explanation for", file_id, "===")

    # Transformer windows
    ds = BehaviorOpcodeDataset(SPLITS_DIR / "test.csv")
    sample = None
    for i in range(len(ds)):
        windows, _, lengths, _, fid = ds[i]
        if str(fid) == file_id:
            sample = (windows, lengths)
            break
    if sample is None:
        print(f"No behavior windows found for {file_id} in test split")
    else:
        windows, lengths = sample
        model = BehaviorOpcodeTransformer()
        model.load_state_dict(torch.load(tf_ckpt, map_location="cpu", weights_only=True))
        model = model.to(device).eval()
        with torch.no_grad():
            _, window_logits, behavior_logits = model(
                windows.unsqueeze(0).to(device),
                lengths.unsqueeze(0).to(device),
            )
        window_prob = torch.softmax(window_logits[0], dim=-1)[:, 1]
        behavior_prob = torch.sigmoid(behavior_logits[0])
        order = window_prob.argsort(descending=True)
        print("\nTop transformer windows:")
        shown = 0
        for w in order.tolist():
            if int(lengths[w]) == 0:
                continue
            scores = ", ".join(
                f"{name.lower()}={float(behavior_prob[w, i]):.3f}"
                for i, name in enumerate(HEAD_BEHAVIOR_NAMES)
            )
            print(
                f"  #{shown + 1} window={w} len={int(lengths[w])} "
                f"ransomware={float(window_prob[w]):.3f} | {scores}"
            )
            shown += 1
            if shown >= 5:
                break

    # GAT blocks + read→write path
    if not graph_path.is_file():
        print(f"Missing graph for explanation: {graph_path}")
        return
    data = next(iter(GeoDataLoader([load_graph(graph_path)], batch_size=1))).to(device)
    gat = BehaviorOpcodeGAT()
    gat.load_state_dict(torch.load(gat_ckpt, map_location="cpu", weights_only=True))
    gat = gat.to(device).eval()
    with torch.no_grad():
        _, block_logits, behavior_logits = gat(data)
    print("\nTop GAT blocks:")
    for row in top_blocks_report(data.cpu(), block_logits.cpu(), behavior_logits.cpu()):
        print(
            f"  B{row['block_id']}: ransomware={row['ransomware_score']:.3f} "
            f"crypto={row['crypto_count']} file={row['file_count']} | "
            f"encrypt={row['encrypt']:.3f} decrypt={row['decrypt']:.3f} "
            f"file_read={row['file_read']:.3f} file_write={row['file_write']:.3f}"
        )

    payload = torch.load(graph_path, map_location="cpu", weights_only=False)
    asm_path = Path(
        payload.get("asm_path", PROJECT_ROOT / "data" / "asm" / f"{file_id}.asm")
    )
    if asm_path.is_file():
        cfg = build_tokenized_cfg(asm_path)
        paths = find_read_to_write_paths(cfg)
        print("\nRead → write paths:")
        if not paths:
            print("  (none found)")
        for path in paths[:5]:
            print("  " + " -> ".join(f"B{b}" for b in path))


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs behavior models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--sample-id", type=str, default=None)
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    gat_device = _resolve_gat_device()

    tf_base = CHECKPOINT_DIR / "best_transformer.pt"
    tf_beh = CHECKPOINT_DIR / "best_transformer_behavior.pt"
    gat_base = CHECKPOINT_DIR / "best_gat.pt"
    gat_beh = CHECKPOINT_DIR / "best_gat_behavior.pt"

    if not args.skip_train:
        print("=== Training ===")
        train_transformer_baseline(device, args.epochs)
        train_transformer_behavior(device, args.epochs)
        train_gat_baseline(gat_device, args.epochs)
        train_gat_behavior(gat_device, args.epochs)

    for path in (tf_base, tf_beh, gat_base, gat_beh):
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint {path}; run without --skip-train")

    print("\n=== Test ransomware F1 ===")
    results = {
        "transformer_baseline": eval_transformer_baseline(device, tf_base),
        "transformer_behavior": eval_transformer_behavior(device, tf_beh),
        "gat_baseline": eval_gat(gat_device, gat_base, behavior=False),
        "gat_behavior": eval_gat(gat_device, gat_beh, behavior=True),
    }
    print("\nSummary (ransomware F1):")
    for name, score in results.items():
        print(f"  {name}: {score:.4f}")

    file_id, graph_path = pick_ransomware_sample(args.sample_id)
    explain_sample(device, file_id, graph_path, tf_beh, gat_beh)


if __name__ == "__main__":
    main()
