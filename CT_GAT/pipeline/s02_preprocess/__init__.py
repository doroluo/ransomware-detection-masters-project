"""Stage 2, preprocess. Two independent paths live here.

The flat scripts `preprocess_full.py` and `tokenize_asm_gz.py` (asm listing -> token rows) are the transformer path; they are
run as scripts from this folder and are not imported through this package.

The jsonl-only path reads the CFG export (`.jsonl.gz`) directly: `cache` turns each export into one npz per binary, and `cache/folds.py`
assigns the family-holdout folds. See README.md, section "jsonl-only path (cache, folds)".
"""
