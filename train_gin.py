#!/usr/bin/env python3
"""
train_gin.py

Trains the GIN in gin_model.py on the opcode transition graphs built by
build_graphs.py, using the train / val / test assignment from split_dataset.py.

The validation split picks the epoch and the decision threshold; the test split
is read exactly once, at the end, with whatever the validation split chose. Any
other order silently tunes on test and reports a number you cannot defend.

Reported per run: accuracy, precision, recall, F1 and ROC-AUC, plus a confusion
matrix and per-family recall, because an average over 25 ransomware families
can hide a family the model never detects.

Usage:

    python train_gin.py --graphs data/graphs/graphs.pt
    python train_gin.py --epochs 100 --layers 5 --hidden-dim 128
    python train_gin.py --seeds 5          # repeat and report mean +/- std
"""

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from gin_model import GIN

SPLITS = ("train", "val", "test")


class GraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]


def collate(batch):
    """Pads a list of graphs into dense [B, N, ...] tensors."""
    batch_size = len(batch)
    max_nodes = max(g["node_ids"].size(0) for g in batch)
    num_features = batch[0]["x"].size(1)

    node_ids = torch.zeros(batch_size, max_nodes, dtype=torch.long)
    features = torch.zeros(batch_size, max_nodes, num_features)
    adjacency = torch.zeros(batch_size, max_nodes, max_nodes)
    mask = torch.zeros(batch_size, max_nodes)
    labels = torch.zeros(batch_size, dtype=torch.long)

    for i, graph in enumerate(batch):
        n = graph["node_ids"].size(0)
        node_ids[i, :n] = graph["node_ids"]
        features[i, :n] = graph["x"]
        mask[i, :n] = 1.0
        labels[i] = graph["label"]
        if graph["edge_index"].numel():
            source, target = graph["edge_index"]
            adjacency[i, source, target] = graph["edge_weight"]

    return node_ids, features, adjacency, mask, labels


def binary_metrics(labels, probabilities, threshold=0.5):
    """Accuracy / precision / recall / F1 at a threshold, plus ROC-AUC."""
    predictions = [1 if p >= threshold else 0 for p in probabilities]
    tp = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 0)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels, probabilities) if len(set(labels)) > 1 else float("nan")
    except ImportError:
        auc = float("nan")

    return {
        "accuracy": (tp + tn) / len(labels) if labels else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def best_threshold(labels, probabilities):
    """Threshold maximising F1 on the validation split."""
    candidates = sorted(set(probabilities))
    best, best_f1 = 0.5, -1.0
    for threshold in candidates:
        f1 = binary_metrics(labels, probabilities, threshold)["f1"]
        if f1 > best_f1:
            best, best_f1 = threshold, f1
    return best


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for node_ids, features, adjacency, mask, y in loader:
        logits = model(node_ids.to(device), features.to(device),
                       adjacency.to(device), mask.to(device))
        probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
        labels.extend(y.tolist())
    return labels, probabilities


def run_once(args, payload, seed, quiet=False):
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device(args.device)
    graphs = payload["graphs"]
    by_split = defaultdict(list)
    for graph in graphs:
        by_split[graph["split"]].append(graph)

    loaders = {
        split: DataLoader(
            GraphDataset(by_split[split]),
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
        )
        for split in SPLITS
    }

    model = GIN(
        vocab_size=len(payload["vocab"]),
        num_features=graphs[0]["x"].size(1),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
        adjacency_norm=args.adjacency_norm,
        direction=args.direction,
        pooling=args.pooling,
    ).to(device)

    # Mild class imbalance (roughly 730 goodware to 665 ransomware in train),
    # weighted so the loss does not drift toward the majority class.
    counts = Counter(g["label"] for g in by_split["train"])
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (2 * counts.get(c, 1)) for c in (0, 1)], dtype=torch.float, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_state, best_score, best_epoch, waited = None, -1.0, 0, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, seen = 0.0, 0
        for node_ids, features, adjacency, mask, y in loaders["train"]:
            optimizer.zero_grad()
            logits = model(node_ids.to(device), features.to(device),
                           adjacency.to(device), mask.to(device))
            loss = criterion(logits, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            epoch_loss += loss.item() * y.size(0)
            seen += y.size(0)

        val_labels, val_probabilities = evaluate(model, loaders["val"], device)
        val = binary_metrics(val_labels, val_probabilities)
        score = val["auc"] if args.select_on == "auc" and val["auc"] == val["auc"] else val["f1"]
        scheduler.step(score)

        if score > best_score:
            best_score, best_epoch, waited = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1

        if not quiet and (epoch % args.log_every == 0 or epoch == 1):
            print(f"  epoch {epoch:>3}  loss {epoch_loss/seen:.4f}  "
                  f"val f1 {val['f1']:.4f}  val auc {val['auc']:.4f}"
                  f"{'  *' if best_epoch == epoch else ''}", flush=True)

        if waited >= args.patience:
            if not quiet:
                print(f"  early stop at epoch {epoch} (no val gain for {args.patience})")
            break

    model.load_state_dict(best_state)

    # The validation split chooses the threshold, before test is ever touched.
    val_labels, val_probabilities = evaluate(model, loaders["val"], device)
    threshold = best_threshold(val_labels, val_probabilities) if args.tune_threshold else 0.5

    test_labels, test_probabilities = evaluate(model, loaders["test"], device)
    test = binary_metrics(test_labels, test_probabilities, threshold)
    val = binary_metrics(val_labels, val_probabilities, threshold)

    return {
        "model": model,
        "seed": seed,
        "best_epoch": best_epoch,
        "threshold": threshold,
        "val": val,
        "test": test,
        "test_graphs": by_split["test"],
        "test_probabilities": test_probabilities,
        "test_labels": test_labels,
    }


def print_report(result):
    test = result["test"]
    print("\n=== Test set ===")
    print(f"threshold (chosen on val) : {result['threshold']:.4f}")
    print(f"accuracy  {test['accuracy']:.4f}")
    print(f"precision {test['precision']:.4f}")
    print(f"recall    {test['recall']:.4f}")
    print(f"f1        {test['f1']:.4f}")
    print(f"auc       {test['auc']:.4f}")
    print("\nconfusion matrix")
    print("                 predicted good   predicted ransom")
    print(f"actual good      {test['tn']:>14}   {test['fp']:>16}")
    print(f"actual ransom    {test['fn']:>14}   {test['tp']:>16}")

    per_family = defaultdict(lambda: [0, 0])
    for graph, probability in zip(result["test_graphs"], result["test_probabilities"]):
        if graph["label"] != 1:
            continue
        per_family[graph["family"]][1] += 1
        if probability >= result["threshold"]:
            per_family[graph["family"]][0] += 1

    if per_family:
        print("\nper-family recall (test)")
        for family, (hit, total) in sorted(per_family.items(), key=lambda kv: kv[1][0] / kv[1][1]):
            print(f"  {family:<15} {hit:>3}/{total:<3}  {hit/total:.2f}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a GIN on opcode transition graphs for ransomware detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--graphs", default="data/graphs/graphs.pt")
    parser.add_argument("--out", default="runs/gin",
                        help="directory for the checkpoint, metrics and predictions")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=2.0)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--adjacency-norm", choices=["row", "log", "none"], default="row")
    parser.add_argument("--direction", choices=["both", "in", "out"], default="both")
    parser.add_argument("--pooling", choices=["sum", "mean", "max"], default="sum")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--select-on", choices=["f1", "auc"], default="auc")
    parser.add_argument("--tune-threshold", action="store_true",
                        help="pick the decision threshold on val instead of using 0.5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=1,
                        help="train this many times with consecutive seeds and average")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args(argv)

    graphs_path = Path(args.graphs)
    if not graphs_path.is_file():
        print(f"{graphs_path} not found. Run build_graphs.py first.", file=sys.stderr)
        return 1

    payload = torch.load(graphs_path, weights_only=False)
    counts = Counter((g["split"], g["label"]) for g in payload["graphs"])
    print(f"Loaded {len(payload['graphs'])} graphs, vocabulary {len(payload['vocab'])}")
    for split in SPLITS:
        print(f"  {split:<6} goodware {counts[(split, 0)]:<5} ransomware {counts[(split, 1)]}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i in range(args.seeds):
        seed = args.seed + i
        print(f"\n--- run {i+1}/{args.seeds} (seed {seed}) ---")
        started = time.time()
        result = run_once(args, payload, seed, quiet=args.seeds > 1)
        print(f"  best epoch {result['best_epoch']}, {time.time()-started:.1f}s, "
              f"test f1 {result['test']['f1']:.4f}, test auc {result['test']['auc']:.4f}")
        results.append(result)

    best = max(results, key=lambda r: r["val"]["f1"])
    print_report(best)

    if len(results) > 1:
        print(f"\n=== Across {len(results)} seeds ===")
        for metric in ("accuracy", "precision", "recall", "f1", "auc"):
            values = [r["test"][metric] for r in results]
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            print(f"{metric:<10} {mean:.4f} +/- {std:.4f}")

    torch.save({"state_dict": best["model"].state_dict(),
                "vocab": payload["vocab"],
                "args": vars(args),
                "threshold": best["threshold"]},
               out_dir / "model.pt")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump({
            "args": vars(args),
            "runs": [{"seed": r["seed"], "best_epoch": r["best_epoch"],
                      "threshold": r["threshold"],
                      "val": {k: v for k, v in r["val"].items()},
                      "test": {k: v for k, v in r["test"].items()}} for r in results],
        }, handle, indent=2)

    with open(out_dir / "test_predictions.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sha256", "family", "label", "probability_ransomware", "predicted"])
        for graph, probability in zip(best["test_graphs"], best["test_probabilities"]):
            writer.writerow([graph["sha256"], graph["family"], graph["label"],
                             f"{probability:.6f}", int(probability >= best["threshold"])])

    print(f"\nWrote model.pt, metrics.json and test_predictions.csv to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
