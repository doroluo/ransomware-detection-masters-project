#!/usr/bin/env python3
"""
tokenize_opcodes.py

Turns the opcode sequences from extract_opcodes.py into the tokenizations
from Tokenization-Testing-for-Malware-Data, without that repo's dataset.

The tokenization project reads a dataframe column named Opcodes and writes
train/test pickles. Two things in it do not fit this dataset, and are fixed
here:

  * Its main() loads seven hardcoded family CSVs and draws a new 70/30 split.
    This script reads index.csv, so the ransomware train / val / test
    assignment is the one every other model uses.
  * BPE, WordPiece, Unigram, and SentencePiece encoding is sliced at rows
    2450 and 4900, which silently drops samples on any other corpus size,
    and val is never encoded. Every row of every split is encoded here.

The vocabulary is fit on the training split only. Val and test are encoded
with that vocabulary. Opcodes outside a closed vocabulary (single words,
word pairs) are dropped. Subword tokenizers map them to <UNK>.

Methods, matching the tokenization project:

    sw    most common opcodes (default 31)
    wp    most common adjacent pairs, written as mov_add (default 30)
    bpe   byte-pair encoding (default vocab 1000)
    wpc   WordPiece (default vocab 500)
    uni   Unigram (default vocab 500)
    spc   SentencePiece BPE (default vocab 500)

Subword methods need the `tokenizers` package. sw and wp do not.

Usage:

    python tokenize_opcodes.py --index data/opcodes_dataset/index.csv
    python tokenize_opcodes.py --methods sw,wp --limit 20
    python tokenize_opcodes.py --methods bpe --vocab-size 1000
"""

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

from opcode_sequences import SPLITS, apply_limit, load_sequences, opcode_tokens, read_index


DEFAULT_VOCAB = {
    "sw": 31,
    "wp": 30,
    "bpe": 1000,
    "wpc": 500,
    "uni": 500,
    "spc": 500,
}
SUBWORD = ("bpe", "wpc", "uni", "spc")
SPECIAL_TOKENS = ["<UNK>", "<SEP>", "<MASK>", "<CLS>"]
UNK_TOKEN = "<UNK>"


def vocab_size_for(method, override):
    size = override if override else DEFAULT_VOCAB[method]
    if method in SUBWORD and size <= len(SPECIAL_TOKENS):
        raise SystemExit(
            f"{method} vocab size must be larger than the {len(SPECIAL_TOKENS)} special tokens"
        )
    return size


def keep_frequent(loaded, key_of, size):
    """
    Counts `key_of(mnemonics)` on the train split and returns the `size`
    most common keys, ties broken alphabetically so the list is stable.
    """
    counts = Counter()
    for row, mnemonics in loaded:
        if row["split"] != "train":
            continue
        counts.update(key_of(mnemonics))
    if not counts:
        raise SystemExit("The train split has no opcodes to build a vocabulary from.")
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _count in ranked[:size]]


def unigram_keys(mnemonics):
    tokens = []
    for line in mnemonics:
        tokens.extend(opcode_tokens(line))
    return tokens


def bigram_keys(mnemonics):
    opcodes = unigram_keys(mnemonics)
    return [f"{first}_{second}" for first, second in zip(opcodes, opcodes[1:])]


def filter_tokens(loaded, keys_of, kept):
    allowed = set(kept)
    rows = []
    for row, mnemonics in loaded:
        tokens = [token for token in keys_of(mnemonics) if token in allowed]
        rows.append(record(row, tokens))
    return rows


def record(row, tokens):
    return {
        "sha256": row["sha256"],
        "split": row["split"],
        "label": row["label"],
        "label_name": row["label_name"],
        "family": row["family"],
        "tokens": tokens,
    }


def require_tokenizers():
    try:
        import tokenizers  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Subword tokenization needs the `tokenizers` package. "
            "Install it with: pip install tokenizers"
        ) from exc


def train_subword(method, train_texts, size):
    """Fits one Hugging Face tokenizer on the training opcode strings."""
    from tokenizers import SentencePieceBPETokenizer, Tokenizer
    from tokenizers.models import BPE, Unigram, WordPiece
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer

    if method == "spc":
        # The default unknown token is "<unk>". It has to match a special
        # token actually present in the trained vocabulary, or encode() fails.
        tokenizer = SentencePieceBPETokenizer(unk_token=UNK_TOKEN)
        tokenizer.train_from_iterator(
            train_texts,
            vocab_size=size,
            special_tokens=SPECIAL_TOKENS,
            show_progress=False,
        )
        return tokenizer

    if method == "bpe":
        tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
        trainer = BpeTrainer(special_tokens=SPECIAL_TOKENS, vocab_size=size, show_progress=False)
    elif method == "wpc":
        tokenizer = Tokenizer(WordPiece(unk_token=UNK_TOKEN))
        trainer = WordPieceTrainer(special_tokens=SPECIAL_TOKENS, vocab_size=size, show_progress=False)
    elif method == "uni":
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(
            unk_token=UNK_TOKEN,
            special_tokens=SPECIAL_TOKENS,
            vocab_size=size,
            show_progress=False,
        )
    else:
        raise SystemExit(f"unknown subword method {method}")

    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(train_texts, trainer=trainer)
    return tokenizer


def encode_subword(tokenizer, loaded):
    rows = []
    total = len(loaded)
    for i, (row, mnemonics) in enumerate(loaded, start=1):
        text = " ".join(unigram_keys(mnemonics))
        tokens = list(tokenizer.encode(text).tokens) if text else []
        rows.append(record(row, tokens))
        if i % 250 == 0 or i == total:
            print(f"  encoded {i}/{total}", flush=True)
    return rows


def surface_vocab(tokenizer):
    """Tokenizer vocabulary as token strings, lowest id first."""
    vocab = tokenizer.get_vocab()
    return [token for token, _idx in sorted(vocab.items(), key=lambda item: item[1])]


def write_method(out_dir, method, size, max_instructions, vocab, rows, tokenizer=None):
    method_dir = Path(out_dir) / method
    method_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "vocab_size": size,
        "max_instructions": max_instructions,
        "vocab": vocab,
        "rows": rows,
    }
    with open(method_dir / "records.pkl", "wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    with open(method_dir / "vocab.json", "w", encoding="utf-8") as handle:
        json.dump(vocab, handle)
    if tokenizer is not None:
        tokenizer.save(str(method_dir / "tokenizer.json"))

    by_split = Counter(row["split"] for row in rows)
    lengths = [len(row["tokens"]) for row in rows] or [0]
    print(f"\n{method}: vocab {len(vocab)}, "
          f"tokens avg/max {sum(lengths) / len(lengths):.0f}/{max(lengths)}")
    print("  " + ", ".join(f"{split}={by_split[split]}" for split in SPLITS))
    print(f"  wrote {method_dir / 'records.pkl'}")


def run_closed_vocab(method, loaded, size, out_dir, max_instructions):
    keys_of = unigram_keys if method == "sw" else bigram_keys
    print(f"\nFitting {method} on the train split (top {size})...")
    vocab = keep_frequent(loaded, keys_of, size)
    rows = filter_tokens(loaded, keys_of, vocab)
    write_method(out_dir, method, size, max_instructions, vocab, rows)


def run_subword(method, loaded, size, out_dir, max_instructions):
    print(f"\nFitting {method} on the train split (vocab {size})...")
    # Materialized on purpose: Unigram and SentencePiece walk the corpus
    # more than once, and a generator would be empty on the second pass.
    train_texts = [
        " ".join(unigram_keys(mnemonics))
        for row, mnemonics in loaded
        if row["split"] == "train"
    ]
    train_texts = [text for text in train_texts if text]
    try:
        tokenizer = train_subword(method, train_texts, size)
    except Exception as exc:
        raise SystemExit(f"{method} training failed: {exc}") from exc
    print(f"Encoding {method} on every split...")
    rows = encode_subword(tokenizer, loaded)
    write_method(
        out_dir, method, size, max_instructions, surface_vocab(tokenizer), rows, tokenizer
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Tokenize opcode sequences, fitting the vocabulary on the train split only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--index", default="data/opcodes_dataset/index.csv")
    parser.add_argument("--out", default="data/tokens",
                        help="directory to write one subdirectory per method into")
    parser.add_argument("--methods", default="sw,wp,bpe,wpc,uni,spc",
                        help="comma-separated subset of sw,wp,bpe,wpc,uni,spc")
    parser.add_argument("--vocab-size", type=int, default=0,
                        help="override the per-method default vocabulary size (0 = defaults)")
    parser.add_argument("--max-instructions", type=int, default=0,
                        help="truncate each sequence before tokenizing (0 = no limit)")
    parser.add_argument("--limit", type=int, default=0,
                        help="keep only this many samples from each split (0 = all)")
    args = parser.parse_args(argv)

    methods = [part.strip().lower() for part in args.methods.split(",") if part.strip()]
    unknown = [method for method in methods if method not in DEFAULT_VOCAB]
    if not methods or unknown:
        parser.error(
            "--methods must be a comma-separated list drawn from "
            + ",".join(DEFAULT_VOCAB)
        )
    args.methods = methods
    return args


def main(argv=None):
    args = parse_args(argv)
    if any(method in SUBWORD for method in args.methods):
        require_tokenizers()

    rows = apply_limit(read_index(args.index), args.limit)
    print(f"Reading {len(rows)} opcode files...")
    loaded, missing = load_sequences(rows, args.max_instructions)
    if missing:
        print(f"Skipped {len(missing)} samples with missing or empty opcode files.")
    if not any(row["split"] == "train" for row, _mnemonics in loaded):
        print("No training samples were loaded.", file=sys.stderr)
        return 1

    for method in args.methods:
        size = vocab_size_for(method, args.vocab_size)
        if method in SUBWORD:
            run_subword(method, loaded, size, args.out, args.max_instructions)
        else:
            run_closed_vocab(method, loaded, size, args.out, args.max_instructions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
