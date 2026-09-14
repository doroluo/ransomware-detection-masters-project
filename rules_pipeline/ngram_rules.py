#!/usr/bin/env python3
"""Discriminative mnemonic n-gram rules, mined on TRAIN only.

Input is the mnemonic-only stream (`mn/<sha256>.txt`, one executable section
per line, space separated). A *rule* is "the n-gram <m1 m2 ... mk> occurs
anywhere in this file's first `max_mnems` mnemonics" -- presence, not count,
so a rule is readable and a rule set is auditable.

Mining is level-wise (Apriori over sequences): 2-grams are counted in full,
then a (k+1)-gram is only counted if both of its k-gram sub-windows survived
the support cut. Support is *document* frequency inside the training split.

Scoring uses a Jeffreys (alpha=0.5) prior so an n-gram seen in 3 of 796
ransomware and 0 of 1003 goodware does not get infinite lift:

    P(g|R) = (a + a0) / (N_R + 2*a0)        a0 = 0.5
    lift   = P(g|R) / P(g|G)
    odds   = [P(g|R)/(1-P(g|R))] / [P(g|G)/(1-P(g|G))]

Both directions are mined: ransomware-leaning rules and goodware-leaning ones.
"""
from __future__ import annotations

import numpy as np

OOV = 0
BASE = 2048          # vocabulary slots; 4-gram code fits in int64 (2048^4)
ALPHA = 0.5


# ---------------------------------------------------------------------------
def read_mnemonics(path, max_mnems: int) -> list[str]:
    out: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            for tok in line.split():
                out.append(tok)
                if len(out) >= max_mnems:
                    return out
    return out


def build_vocab(seqs, train_mask, max_size: int = BASE - 1) -> dict[str, int]:
    from collections import Counter
    c: Counter = Counter()
    for s, keep in zip(seqs, train_mask):
        if keep:
            c.update(s)
    vocab = {}
    for i, (m, _) in enumerate(c.most_common(max_size), start=1):
        vocab[m] = i
    return vocab


def encode(seqs, vocab) -> list[np.ndarray]:
    return [np.fromiter((vocab.get(t, OOV) for t in s), dtype=np.int64,
                        count=len(s)) for s in seqs]


# ---------------------------------------------------------------------------
def _codes(a: np.ndarray, k: int) -> np.ndarray:
    """int64 code for every k-gram window of the encoded sequence.

    The int64 cast is load-bearing, not defensive. Callers store the mnemonic
    stream as int16 to keep a 2,500-file corpus in memory (see
    rules_pipeline/train_eval.MnemCorpus); `c * BASE` on an int16 array stays
    int16 under NEP 50 and wraps silently, so every k-gram whose leading token
    id exceeds 15 gets a code that does not match the int64 code the miner
    computed for the same k-gram -- and the rule then never fires. Casting here
    rather than at each call site means no caller can reintroduce it.
    """
    a = np.asarray(a, dtype=np.int64)
    n = len(a) - k + 1
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    c = a[:n].copy()
    for j in range(1, k):
        c = c * BASE + a[j:j + n]
    return c


def _doc_unique(a: np.ndarray, k: int) -> np.ndarray:
    return np.unique(_codes(a, k))


def _merge_counts(codes_a, cnt_a, codes_b, cnt_b):
    """Sum two (sorted codes, counts) tables."""
    if codes_a is None:
        return codes_b, cnt_b
    codes = np.union1d(codes_a, codes_b)
    out = np.zeros(len(codes), dtype=np.int64)
    out[np.searchsorted(codes, codes_a)] += cnt_a
    out[np.searchsorted(codes, codes_b)] += cnt_b
    return codes, out


def _doc_grams(a: np.ndarray, k: int, prev_frequent) -> np.ndarray:
    u = _doc_unique(a, k)
    if prev_frequent is not None and len(u):
        # Apriori downward closure: both (k-1)-sub-windows must be frequent.
        pre = u // BASE
        suf = u % (BASE ** (k - 1))
        u = u[np.isin(pre, prev_frequent) & np.isin(suf, prev_frequent)]
    return u


def mine(encoded: list[np.ndarray], y: np.ndarray, train_idx: np.ndarray,
         min_support: int = 40, max_n: int = 4, chunk: int = 256,
         verbose=True):
    """Level-wise mining over the TRAIN split.

    Returns {k: (codes, a, b)} where a and b are the training document
    frequencies of each surviving k-gram in ransomware and in goodware.
    Counting is chunked so no level ever materialises every document's gram
    list at once.
    """
    n_tr = len(train_idx)
    out = {}
    prev_frequent = None
    for k in range(2, max_n + 1):
        codes_all, df_all = None, None
        for s in range(0, n_tr, chunk):
            part = [_doc_grams(encoded[i].astype(np.int64), k, prev_frequent)
                    for i in train_idx[s:s + chunk]]
            flat = np.concatenate(part) if part else np.empty(0, np.int64)
            c, d = np.unique(flat, return_counts=True)
            codes_all, df_all = _merge_counts(codes_all, df_all, c, d)
        if codes_all is None or not len(codes_all):
            break
        codes = codes_all[df_all >= min_support]
        if not len(codes):
            if verbose:
                print(f"  {k}-grams: none reach support {min_support} "
                      f"({len(codes_all):,} candidates counted)")
            break
        a = np.zeros(len(codes), dtype=np.int64)   # ransomware doc freq
        b = np.zeros(len(codes), dtype=np.int64)   # goodware doc freq
        for i in train_idx:
            u = _doc_grams(encoded[i].astype(np.int64), k, prev_frequent)
            j = np.clip(np.searchsorted(codes, u), 0, len(codes) - 1)
            hit = j[codes[j] == u]
            if y[i] == 1:
                np.add.at(a, hit, 1)
            else:
                np.add.at(b, hit, 1)
        out[k] = (codes, a, b)
        prev_frequent = codes
        if verbose:
            print(f"  {k}-grams: {len(codes):,} with df>={min_support} "
                  f"(of {len(codes_all):,} candidates, {n_tr} train docs)")
    return out


def score_rules(a, b, n_r, n_g):
    pr = (a + ALPHA) / (n_r + 2 * ALPHA)
    pg = (b + ALPHA) / (n_g + 2 * ALPHA)
    lift_r = pr / pg
    odds_r = (pr / (1 - pr)) / (pg / (1 - pg))
    # single-rule F1 on train, treating "n-gram present" as "predict ransomware"
    prec = a / np.maximum(a + b, 1)
    rec = a / max(n_r, 1)
    f1_r = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-9), 0.0)
    # the mirror rule: present => predict goodware
    prec_g = b / np.maximum(a + b, 1)
    rec_g = b / max(n_g, 1)
    f1_g = np.where(prec_g + rec_g > 0,
                    2 * prec_g * rec_g / np.maximum(prec_g + rec_g, 1e-9), 0.0)
    return dict(p_r=pr, p_g=pg, lift=lift_r, odds=odds_r,
                precision_r=prec, recall_r=rec, f1_r=f1_r,
                precision_g=prec_g, recall_g=rec_g, f1_g=f1_g)


def decode(code: int, k: int, inv_vocab: dict[int, str]) -> str:
    toks = []
    for _ in range(k):
        toks.append(inv_vocab.get(int(code % BASE), "?"))
        code //= BASE
    return " ".join(reversed(toks))


# ---------------------------------------------------------------------------
def hit_matrix(encoded, rule_codes_by_k) -> np.ndarray:
    """Boolean (n_docs, n_rules): does rule r fire on doc i?"""
    cols = []
    for k in sorted(rule_codes_by_k):
        codes = rule_codes_by_k[k]
        if not len(codes):
            continue
        M = np.zeros((len(encoded), len(codes)), dtype=bool)
        for i, a in enumerate(encoded):
            u = _doc_unique(a, k)
            j = np.searchsorted(codes, u)
            jc = np.clip(j, 0, len(codes) - 1)
            M[i, jc[codes[jc] == u]] = True
        cols.append(M)
    return np.hstack(cols) if cols else np.zeros((len(encoded), 0), dtype=bool)


def greedy_set_cover(hits: np.ndarray, y: np.ndarray, idx: np.ndarray,
                     order: np.ndarray, k_rules: int, min_precision: float):
    """Pick rules that cover new positives, keeping per-rule train precision."""
    pos = idx[y[idx] == 1]
    uncovered = set(pos.tolist())
    chosen = []
    for r in order:
        if len(chosen) >= k_rules or not uncovered:
            break
        fired = idx[hits[idx, r]]
        if len(fired) == 0:
            continue
        prec = float((y[fired] == 1).mean())
        if prec < min_precision:
            continue
        new = uncovered & set(fired[y[fired] == 1].tolist())
        if len(new) < 1:
            continue
        chosen.append(int(r))
        uncovered -= new
    return chosen, len(pos) - len(uncovered), len(pos)
