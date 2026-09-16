#!/usr/bin/env python3
"""
baseline.py

Bag-of-opcodes baselines on the same splits the GIN uses, so the GIN's score
can be read as "better than counting opcodes" rather than just "high".

Features are opcode unigram and bigram frequencies (TF-IDF weighted), which
throw away all graph structure and keep only how often each opcode and each
adjacent pair occurs. If logistic regression on those matches the GIN, the
graph structure is not contributing and the simpler model is the better answer.

The vocabulary is fit on the training split only, matching build_graphs.py.

Usage:

    python baseline.py --index data/opcodes_dataset/index.csv
    python baseline.py --model random-forest --ngram 1
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")


def load_sequences(index_path, max_instructions):
    import csv

    with open(index_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    data = defaultdict(lambda: {"text": [], "y": [], "family": []})
    for row in rows:
        try:
            with open(row["opcode_file"], encoding="utf-8") as handle:
                opcodes = [line for line in handle.read().split("\n") if line]
        except OSError:
            continue
        if not opcodes:
            continue
        if max_instructions:
            opcodes = opcodes[:max_instructions]
        bucket = data[row["split"]]
        # The vectoriser tokenises on whitespace, so the sequence becomes a
        # single space-separated "sentence" of mnemonics.
        bucket["text"].append(" ".join(opcodes))
        bucket["y"].append(int(row["label"]))
        bucket["family"].append(row["family"])
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Bag-of-opcodes baseline for comparison against the GIN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--index", default="data/opcodes_dataset/index.csv")
    parser.add_argument("--model", choices=["logreg", "random-forest", "both"], default="both")
    parser.add_argument("--ngram", type=int, default=2, help="use 1..N grams of opcodes")
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--max-instructions", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)

    index_path = Path(args.index)
    if not index_path.is_file():
        print(f"{index_path} not found. Run extract_opcodes.py first.", file=sys.stderr)
        return 1

    data = load_sequences(index_path, args.max_instructions)
    for split in SPLITS:
        print(f"  {split:<6} {len(data[split]['y'])} samples")

    vectoriser = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"\S+",
        ngram_range=(1, args.ngram),
        max_features=args.max_features,
        sublinear_tf=True,
    )
    x_train = vectoriser.fit_transform(data["train"]["text"])
    x_test = vectoriser.transform(data["test"]["text"])
    print(f"\nFeature matrix: {x_train.shape[0]} x {x_train.shape[1]} "
          f"(1..{args.ngram}-grams)")

    models = []
    if args.model in ("logreg", "both"):
        models.append(("logistic regression",
                       LogisticRegression(max_iter=2000, class_weight="balanced",
                                          random_state=args.seed)))
    if args.model in ("random-forest", "both"):
        models.append(("random forest",
                       RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                              n_jobs=-1, random_state=args.seed)))

    y_train, y_test = data["train"]["y"], data["test"]["y"]
    for name, model in models:
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        print(f"\n=== {name} (test) ===")
        print(f"accuracy  {accuracy_score(y_test, predictions):.4f}")
        print(f"precision {precision_score(y_test, predictions, zero_division=0):.4f}")
        print(f"recall    {recall_score(y_test, predictions, zero_division=0):.4f}")
        print(f"f1        {f1_score(y_test, predictions, zero_division=0):.4f}")
        print(f"auc       {roc_auc_score(y_test, probabilities):.4f}")
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        print(f"confusion tn {tn}  fp {fp}  fn {fn}  tp {tp}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
