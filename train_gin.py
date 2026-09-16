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


METRIC_LABELS = [
    ("accuracy", "Accuracy", "of all samples, how many were called correctly"),
    ("precision", "Precision", "of the samples flagged as ransomware, how many were"),
    ("recall", "Recall", "of the actual ransomware, how much was caught"),
    ("f1", "F1", "harmonic mean of precision and recall"),
    ("auc", "ROC-AUC", "ranking quality, independent of the threshold"),
]


def as_percent(value):
    return "n/a" if value != value else f"{value * 100:.2f}%"


def summarize(values):
    """mean / std / min / max for one metric across seeds."""
    clean = [v for v in values if v == v]
    if not clean:
        return {"mean": None, "std": None, "min": None, "max": None}
    mean = sum(clean) / len(clean)
    std = (sum((v - mean) ** 2 for v in clean) / len(clean)) ** 0.5
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(min(clean), 4),
        "max": round(max(clean), 4),
    }


def per_family_recall(result):
    """How much of each ransomware family the model caught, worst family first."""
    tally = defaultdict(lambda: [0, 0])
    for graph, probability in zip(result["test_graphs"], result["test_probabilities"]):
        if graph["label"] != 1:
            continue
        tally[graph["family"]][1] += 1
        if probability >= result["threshold"]:
            tally[graph["family"]][0] += 1

    return {
        family: {"detected": hit, "total": total, "recall": round(hit / total, 4)}
        for family, (hit, total) in sorted(tally.items(), key=lambda kv: (kv[1][0] / kv[1][1], kv[0]))
    }


def round_metrics(metrics):
    """Trims 16-digit floats down to something a human can scan."""
    return {
        key: (round(value, 4) if isinstance(value, float) else value)
        for key, value in metrics.items()
    }


def print_report(result, results, families):
    test = result["test"]
    total = test["tp"] + test["tn"] + test["fp"] + test["fn"]

    print("\n" + "=" * 62)
    print(f"TEST RESULTS   (best of {len(results)} seed"
          f"{'s' if len(results) > 1 else ''}, {total} held-out samples)")
    print("=" * 62)

    spread = {key: summarize([r["test"][key] for r in results]) for key, _, _ in METRIC_LABELS}
    show_spread = len(results) > 1

    header = f"{'':<11}{'best run':>10}"
    if show_spread:
        header += f"{'mean':>10}{'std dev':>10}"
    print(header)
    for key, label, explanation in METRIC_LABELS:
        line = f"{label:<11}{as_percent(test[key]):>10}"
        if show_spread:
            mean, std = spread[key]["mean"], spread[key]["std"]
            line += f"{as_percent(mean) if mean is not None else 'n/a':>10}"
            line += f"{as_percent(std) if std is not None else 'n/a':>10}"
        print(f"{line}   {explanation}")

    print(f"\nDecision threshold {result['threshold']:.4f}, chosen on validation "
          f"(best epoch {result['best_epoch']}).")

    print("\nConfusion matrix")
    print(f"  {'':<22}{'called goodware':>17}{'called ransomware':>19}")
    print(f"  {'really goodware':<22}{test['tn']:>17}{test['fp']:>19}"
          f"   <- {test['fp']} false alarm{'s' if test['fp'] != 1 else ''}")
    print(f"  {'really ransomware':<22}{test['fn']:>17}{test['tp']:>19}"
          f"   <- {test['fn']} missed")

    if families:
        print("\nPer-family recall, worst first")
        for family, stats in families.items():
            bar = "#" * round(stats["recall"] * 20)
            print(f"  {family:<15}{stats['detected']:>3}/{stats['total']:<4}"
                  f"{as_percent(stats['recall']):>8}  {bar}")
        perfect = sum(1 for s in families.values() if s["recall"] == 1.0)
        print(f"  ({perfect} of {len(families)} families fully detected)")


def write_report(path, metrics):
    """A markdown summary meant to be read, not parsed."""
    summary = metrics["summary"]
    best = summary["best_run"]
    lines = [
        "# GIN ransomware detection - results",
        "",
        f"Generated {metrics['created']} from `{metrics['dataset']['graphs']}`.",
        "",
        "## Test set",
        "",
        f"Best of {summary['seeds']} seed(s), evaluated once on "
        f"{metrics['dataset']['splits']['test']['total']} held-out samples.",
        "",
    ]

    if summary["seeds"] > 1:
        lines += ["| Metric | Best run | Mean | Std dev |", "| --- | --- | --- | --- |"]
        for key, label, _ in METRIC_LABELS:
            spread = summary["spread"][key]
            lines.append(f"| {label} | {as_percent(best['test'][key])} | "
                         f"{as_percent(spread['mean'])} | {as_percent(spread['std'])} |")
    else:
        lines += ["| Metric | Value |", "| --- | --- |"]
        for key, label, _ in METRIC_LABELS:
            lines.append(f"| {label} | {as_percent(best['test'][key])} |")

    test = best["test"]
    lines += [
        "",
        f"Decision threshold {best['threshold']:.4f}, chosen on the validation split "
        f"(best epoch {best['best_epoch']}).",
        "",
        "## Confusion matrix",
        "",
        "| | Called goodware | Called ransomware |",
        "| --- | --- | --- |",
        f"| **Really goodware** | {test['tn']} | {test['fp']} (false alarms) |",
        f"| **Really ransomware** | {test['fn']} (missed) | {test['tp']} |",
        "",
    ]

    if metrics["per_family_recall"]:
        lines += ["## Per-family recall (worst first)", "",
                  "| Family | Detected | Recall |", "| --- | --- | --- |"]
        for family, stats in metrics["per_family_recall"].items():
            lines.append(f"| {family} | {stats['detected']}/{stats['total']} | "
                         f"{as_percent(stats['recall'])} |")
        lines.append("")

    config = metrics["config"]
    lines += [
        "## Configuration",
        "",
        f"{config['layers']} GIN layers, hidden dim {config['hidden_dim']}, "
        f"embedding dim {config['embedding_dim']}, dropout {config['dropout']}, "
        f"{config['pooling']} pooling, {config['adjacency_norm']} adjacency "
        f"normalisation, {config['direction']} edge direction.",
        "",
        f"Trained with AdamW at lr {config['lr']}, batch size {config['batch_size']}, "
        f"up to {config['epochs']} epochs, early stopping after "
        f"{config['patience']} epochs without a validation gain.",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


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
    families = per_family_recall(best)
    print_report(best, results, families)

    # Ordered so the answer comes first: what the run scored, then the spread
    # across seeds, then the per-seed detail, rather than 160 lines of floats.
    metrics = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "graphs": str(graphs_path),
            "vocab_size": len(payload["vocab"]),
            "splits": {
                split: {
                    "total": counts[(split, 0)] + counts[(split, 1)],
                    "goodware": counts[(split, 0)],
                    "ransomware": counts[(split, 1)],
                }
                for split in SPLITS
            },
        },
        "summary": {
            "seeds": len(results),
            "spread": {
                key: summarize([r["test"][key] for r in results])
                for key, _, _ in METRIC_LABELS
            },
            "best_run": {
                "seed": best["seed"],
                "best_epoch": best["best_epoch"],
                "threshold": round(best["threshold"], 4),
                "test": round_metrics(best["test"]),
                "val": round_metrics(best["val"]),
            },
        },
        "per_family_recall": families,
        "config": vars(args),
        "runs": [
            {
                "seed": r["seed"],
                "best_epoch": r["best_epoch"],
                "threshold": round(r["threshold"], 4),
                "val": round_metrics(r["val"]),
                "test": round_metrics(r["test"]),
            }
            for r in results
        ],
    }

    torch.save({"state_dict": best["model"].state_dict(),
                "vocab": payload["vocab"],
                "args": vars(args),
                "threshold": best["threshold"]},
               out_dir / "model.pt")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    write_report(out_dir / "report.md", metrics)

    with open(out_dir / "test_predictions.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sha256", "family", "label", "probability_ransomware", "predicted"])
        for graph, probability in zip(best["test_graphs"], best["test_probabilities"]):
            writer.writerow([graph["sha256"], graph["family"], graph["label"],
                             f"{probability:.6f}", int(probability >= best["threshold"])])

    print(f"\nWrote report.md, metrics.json, model.pt and test_predictions.csv to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
