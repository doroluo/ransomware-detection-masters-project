"""Train ``FullBlockGAT`` on the rich subgraphs from ``preprocess_full.py``.

Pipeline: ``preprocess_full.py`` -> ``processed/cfg_subgraphs/*.pt`` ->
``graph_full_loader.get_loaders`` -> ``gat_full_model.FullBlockGAT``.

The harness mirrors the root ``train_cfg.py``: a class-weighted loss, early
stopping on validation ransomware F1, a decision threshold tuned on validation,
and held-out test metrics (accuracy, precision, recall, F1, ROC-AUC) at that
threshold. Checkpoint and a metrics JSON are written next to each other.

    cd CT_GAT/preprocessing
    python train_full_gat.py --epochs 40
    python train_full_gat.py --feature-mode embeddings --block-dropout 0.15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from gat_full_model import FullBlockGAT
from graph_full_loader import GRAPH_DIR, get_loaders

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "processed" / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_full_gat.pt"


@torch.no_grad()
def predict_proba(model, loader, device):
    """Return (ransomware probabilities, labels) over a loader."""
    model.eval()
    probs, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        probs.append(model(batch).softmax(dim=1)[:, 1].cpu())
        labels.append(batch.y.view(-1).cpu())
    if not probs:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(probs), torch.cat(labels)


def roc_auc(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """ROC-AUC via the Mann-Whitney statistic (no sklearn dependency)."""
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = probs.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, probs.numel() + 1, dtype=torch.float64)
    # Average ranks for ties so the statistic is exact on duplicate scores.
    sorted_probs = probs[order]
    index = 0
    while index < sorted_probs.numel():
        end = index
        while end + 1 < sorted_probs.numel() and sorted_probs[end + 1] == sorted_probs[index]:
            end += 1
        if end > index:
            average = (ranks[order[index]] + ranks[order[end]]) / 2.0
            ranks[order[index : end + 1]] = average
        index = end + 1
    rank_sum_pos = float(ranks[positives].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def metrics_at(probs, labels, threshold):
    preds = probs >= threshold
    positive = labels == 1
    tp = int((preds & positive).sum())
    fp = int((preds & ~positive).sum())
    fn = int((~preds & positive).sum())
    tn = int((~preds & ~positive).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = max(int(labels.numel()), 1)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": roc_auc(probs, labels) if labels.numel() else 0.0,
        "support": int(labels.numel()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def best_threshold(probs, labels):
    chosen = metrics_at(probs, labels, 0.5)
    for step in range(5, 96):
        candidate = metrics_at(probs, labels, step / 100)
        if candidate["f1"] > chosen["f1"]:
            chosen = candidate
    return chosen


def class_weights(loader, device):
    counts = torch.zeros(2)
    for batch in loader:
        labels = batch.y.view(-1)
        counts += torch.bincount(labels, minlength=2).float()
    counts = counts.clamp(min=1.0)
    return (counts.sum() / counts).to(device)


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_loader, val_loader, test_loader, in_dim, edge_dim = get_loaders(
        graph_dir=args.graphs,
        feature_mode=args.feature_mode,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        scale=not args.raw_features,
    )
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    if n_train < 2:
        raise SystemExit(
            f"Need at least 2 training graphs, found {n_train}. Run preprocess_full.py first."
        )

    model = FullBlockGAT(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden=args.hidden,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
        block_dropout=args.block_dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_loader, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"train {n_train}  val {n_val}  test {n_test}  "
        f"in_dim {in_dim}  edge_dim {edge_dim}  device {device}"
    )

    best_f1 = -1.0
    best_state = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * batch.num_graphs
            seen += batch.num_graphs
        val_probs, val_labels = predict_proba(model, val_loader, device)
        val = metrics_at(val_probs, val_labels, 0.5)
        print(
            f"epoch {epoch:02d}  loss {total_loss / max(seen, 1):.4f}  "
            f"val_f1 {val['f1']:.3f}  val_acc {val['accuracy']:.3f}  val_auc {val['auc']:.3f}"
        )
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stop at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_probs, val_labels = predict_proba(model, val_loader, device)
    val_at = best_threshold(val_probs, val_labels)
    test_metrics = None
    if n_test:
        test_probs, test_labels = predict_proba(model, test_loader, device)
        test_metrics = metrics_at(test_probs, test_labels, val_at["threshold"])

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "in_dim": in_dim,
            "edge_dim": edge_dim,
            "feature_mode": args.feature_mode,
            "hidden": args.hidden,
            "heads": args.heads,
            "layers": args.layers,
            "dropout": args.dropout,
            "block_dropout": args.block_dropout,
            "threshold": val_at["threshold"],
            "val_metrics": val_at,
            "test_metrics": test_metrics,
        },
        checkpoint,
    )
    checkpoint.with_suffix(".json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "train": n_train, "val": n_val, "test": n_test,
                "in_dim": in_dim, "edge_dim": edge_dim,
                "feature_mode": args.feature_mode,
                "val_metrics": val_at,
                "test_metrics": test_metrics,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Saved {checkpoint} (val F1 {val_at['f1']:.3f} at threshold {val_at['threshold']:.2f})"
    )
    if test_metrics:
        print(
            "Test ransomware  "
            f"P {test_metrics['precision']:.3f}  R {test_metrics['recall']:.3f}  "
            f"F1 {test_metrics['f1']:.3f}  acc {test_metrics['accuracy']:.3f}  "
            f"AUC {test_metrics['auc']:.3f}  n {test_metrics['support']}  "
            f"tp {test_metrics['tp']} fp {test_metrics['fp']} "
            f"fn {test_metrics['fn']} tn {test_metrics['tn']}"
        )
    else:
        print("No test graphs; skipped the held-out score.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", default=str(GRAPH_DIR))
    parser.add_argument("--feature-mode", choices=("both", "embeddings", "features"), default="both")
    parser.add_argument(
        "--raw-features",
        action="store_true",
        help="Keep raw block counts. By default sizes are log-scaled and counts become rates.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--block-dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
