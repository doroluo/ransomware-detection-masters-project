#!/usr/bin/env python3
"""Mnemonic streams -> token windows, for the sequence model.

What the image encoder did to this data, and what this module does instead
-------------------------------------------------------------------------
`CNN-ViT/asm_parser.py` turned a mnemonic stream into a 256x256 grayscale
image: token id -> pixel intensity, row-major, the canvas covering the first
~21,845 instructions of the file and everything past that thrown away, with
the unused tail of the canvas filled with a constant. Three things went wrong
and all three are data-side, not model-side:

  * a categorical id became an ordinal intensity, so "the distance between
    `push` and `pop`" was a number the convolution could act on;
  * only the first ~22% of the cohort's instructions were ever looked at;
  * the constant pad was a free, perfectly clean channel for file size.

Here a stream is a 1-D sequence of token ids consumed by `nn.Embedding`
(nothing ordinal survives), windows are strided across the WHOLE stream, and
padding is carried in an attention mask that the model is structurally unable
to read as content (see `model.py`).

The token cache
---------------
Parsing 4.1 GB of whitespace-separated text on every run is not affordable, so
each `<sha256>.txt` is cached once as a `uint16` `.npy` of token ids that
`numpy.memmap` can slice a window out of without reading the file. The cache
is built in two phases so that it can be produced in parallel and still carry
a single, deterministic, global id space:

  phase A (parallel)  worker splits the text, assigns LOCAL ids in order of
                      first appearance, writes `tok_local/<sha>.npy` plus
                      `tok_local/<sha>.vocab.txt`;
  phase B (serial)    union of all local vocabularies -> the global vocabulary
                      (sorted, so it is a pure function of the corpus), then
                      every file is remapped into `tok/<sha>.npy`.

Both phases skip work that is already on disk, so an interrupted build
continues where it stopped.

The GLOBAL id space is a property of the corpus, not of a training split - it
is just "which strings exist". The MODEL vocabulary is a different object: see
`Vocab.fit`, which is given training shas only and maps every global id it did
not see in training to `<unk>`. That is the fold-safe object the model uses.

A file's stream is the concatenation of its lines (one executable section per
line) with no separator token. Sections are an artefact of how capstone was
driven, not a semantic boundary the classifier is meant to key on, and a
separator would be a constant the model could count - the exact failure mode
this rewrite exists to remove.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------------------
# where things live
# ---------------------------------------------------------------------------
SHARED = Path(os.environ.get(
    "RANSOM_SHARED", r"C:/Users/chaoa/Downloads/asm and mm/Shared"))
MN_DIRS = (SHARED / "Extract" / "mn",
           SHARED / "Extract_Goodware_Balanced" / "mn")
WEIGHTS_ROOT = Path(os.environ.get(
    "RANSOM_SEQ_MODELS", r"C:/Users/chaoa/Downloads/seq_models"))
CACHE_ROOT = Path(os.environ.get("RANSOM_SEQ_CACHE", WEIGHTS_ROOT / "token_cache"))

PAD_ID = 0
UNK_ID = 1
N_SPECIAL = 2


def mn_path(sha: str) -> Path:
    for d in MN_DIRS:
        p = d / f"{sha}.txt"
        if p.exists():
            return p
    raise FileNotFoundError(f"no mnemonic stream for {sha} under {MN_DIRS}")


def all_corpus_shas() -> list[str]:
    """Every sha256 with a mnemonic stream in either corpus, sorted.

    This is the pretraining pool: labels are never consulted to build it.
    """
    out: set[str] = set()
    for d in MN_DIRS:
        if not d.is_dir():
            raise FileNotFoundError(d)
        for p in d.glob("*.txt"):
            out.add(p.stem)
    return sorted(out)


# ---------------------------------------------------------------------------
# phase A: text -> local ids
# ---------------------------------------------------------------------------
def _encode_one(args) -> tuple[str, int]:
    sha, cache_root = args
    cache_root = Path(cache_root)
    loc = cache_root / "tok_local"
    npy, voc = loc / f"{sha}.npy", loc / f"{sha}.vocab.txt"
    if npy.exists() and voc.exists():
        return sha, -1
    text = mn_path(sha).read_text(encoding="utf-8", errors="replace")
    toks = text.split()
    del text
    local = sorted(set(toks))                 # deterministic, ~hundreds of strings
    if len(local) > 65535:
        raise ValueError(f"{sha}: {len(local)} distinct mnemonics exceeds uint16")
    m = {t: i for i, t in enumerate(local)}
    ids = np.fromiter((m[t] for t in toks), dtype=np.uint16, count=len(toks))
    del toks
    loc.mkdir(parents=True, exist_ok=True)
    tmp = loc / f"{sha}.npy.tmp"
    with tmp.open("wb") as fh:
        np.save(fh, ids)
    os.replace(tmp, npy)
    voc.write_text("\n".join(local), encoding="utf-8")
    return sha, len(ids)


def build_cache(shas=None, cache_root: Path = CACHE_ROOT, workers: int = 8,
                log=print) -> dict:
    """Build (or finish) the token cache. Safe to re-run; skips finished files."""
    cache_root = Path(cache_root)
    shas = list(all_corpus_shas() if shas is None else shas)
    loc, tok = cache_root / "tok_local", cache_root / "tok"
    loc.mkdir(parents=True, exist_ok=True)
    tok.mkdir(parents=True, exist_ok=True)

    todo = [s for s in shas
            if not (tok / f"{s}.npy").exists()
            and not ((loc / f"{s}.npy").exists() and (loc / f"{s}.vocab.txt").exists())]
    log(f"phase A: {len(shas)} files, {len(todo)} to encode")
    if todo:
        t0 = time.time()
        if workers > 1:
            import multiprocessing as mp
            with mp.Pool(workers) as pool:
                for k, _ in enumerate(pool.imap_unordered(
                        _encode_one, [(s, str(cache_root)) for s in todo],
                        chunksize=4), 1):
                    if k % 100 == 0:
                        log(f"  encoded {k}/{len(todo)} ({time.time()-t0:.0f}s)")
        else:
            for k, s in enumerate(todo, 1):
                _encode_one((s, str(cache_root)))
                if k % 100 == 0:
                    log(f"  encoded {k}/{len(todo)} ({time.time()-t0:.0f}s)")
        log(f"phase A done in {time.time()-t0:.0f}s")

    # -- phase B: global vocabulary, then remap -----------------------------
    meta_path = cache_root / "cache_meta.json"
    pending = [s for s in shas if not (tok / f"{s}.npy").exists()]
    if pending:
        strings: set[str] = set()
        for s in shas:
            v = loc / f"{s}.vocab.txt"
            if v.exists():
                strings.update(x for x in v.read_text(encoding="utf-8").split("\n") if x)
        if meta_path.exists():
            strings.update(json.loads(meta_path.read_text(encoding="utf-8"))["vocab"])
        vocab = sorted(strings)
        meta_path.write_text(json.dumps(
            {"vocab": vocab, "n_files": len(shas),
             "note": "global id space = index into this sorted list; the MODEL "
                     "vocabulary is refit per training split, see Vocab.fit"},
            indent=1), encoding="utf-8")
        gid = {t: i for i, t in enumerate(vocab)}
        log(f"phase B: global vocabulary {len(vocab)} mnemonics; "
            f"remapping {len(pending)} files")
        t0 = time.time()
        for k, s in enumerate(pending, 1):
            lv = [x for x in (loc / f"{s}.vocab.txt").read_text(
                encoding="utf-8").split("\n") if x]
            lut = np.array([gid[x] for x in lv], dtype=np.uint16)
            arr = np.load(loc / f"{s}.npy")
            out = lut[arr] if len(arr) else arr.astype(np.uint16)
            tmp = tok / f"{s}.npy.tmp"
            with tmp.open("wb") as fh:
                np.save(fh, out)
            os.replace(tmp, tok / f"{s}.npy")
            if k % 200 == 0:
                log(f"  remapped {k}/{len(pending)} ({time.time()-t0:.0f}s)")
        log(f"phase B done in {time.time()-t0:.0f}s")
    else:
        log("phase B: nothing to remap")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    lens = {}
    len_path = cache_root / "lengths.json"
    if len_path.exists():
        lens = json.loads(len_path.read_text(encoding="utf-8"))
    missing = [s for s in shas if s not in lens]
    if missing:
        for s in missing:
            lens[s] = int(np.load(tok / f"{s}.npy", mmap_mode="r").shape[0])
        len_path.write_text(json.dumps(lens), encoding="utf-8")
    log(f"cache ready: {len(shas)} files, {len(meta['vocab'])} global mnemonics, "
        f"{sum(lens[s] for s in shas):,} tokens")
    return meta


def cache_lengths(cache_root: Path = CACHE_ROOT) -> dict:
    return json.loads((Path(cache_root) / "lengths.json").read_text(encoding="utf-8"))


def global_vocab(cache_root: Path = CACHE_ROOT) -> list[str]:
    return json.loads(
        (Path(cache_root) / "cache_meta.json").read_text(encoding="utf-8"))["vocab"]


def load_ids(sha: str, cache_root: Path = CACHE_ROOT) -> np.ndarray:
    return np.load(Path(cache_root) / "tok" / f"{sha}.npy", mmap_mode="r")


_COUNTS = {}


def token_counts(shas, cache_root: Path = CACHE_ROOT, n_vocab: int = None):
    """Per-mnemonic occurrence counts over `shas`, from a cached per-file table.

    `Vocab.fit` is called once per fold x seed x family (about 250 times over
    the whole study) and a naive fit reads the 833 M-token training stream to
    find out which of 1,418 strings occur in it. The per-file histogram is
    tiny (1,418 uint32 per file), so it is computed once and summed instead.
    """
    cache_root = Path(cache_root)
    key = str(cache_root)
    if key not in _COUNTS:
        vocab = global_vocab(cache_root)
        mat_p, idx_p = cache_root / "counts.npy", cache_root / "counts_index.json"
        order, mat = [], None
        if mat_p.exists() and idx_p.exists():
            order = json.loads(idx_p.read_text(encoding="utf-8"))
            mat = np.load(mat_p)
            if mat.shape[1] != len(vocab):
                order, mat = [], None
        have = {s: i for i, s in enumerate(order)}
        missing = sorted({s for s in shas if s not in have})
        if missing:
            all_shas = sorted(set(order) | set(missing) | {
                p.stem for p in (cache_root / "tok").glob("*.npy")})
            new = np.zeros((len(all_shas), len(vocab)), dtype=np.uint32)
            for i, s in enumerate(all_shas):
                if s in have:
                    new[i] = mat[have[s]]
                else:
                    a = np.asarray(load_ids(s, cache_root))
                    if len(a):
                        new[i] = np.bincount(a.astype(np.int64),
                                             minlength=len(vocab))
            np.save(mat_p, new)
            idx_p.write_text(json.dumps(all_shas), encoding="utf-8")
            order, mat = all_shas, new
        _COUNTS[key] = ({s: i for i, s in enumerate(order)}, mat)
    have, mat = _COUNTS[key]
    n = mat.shape[1] if n_vocab is None else n_vocab
    rows = [have[s] for s in shas if s in have]
    out = mat[rows].sum(axis=0, dtype=np.int64)
    # a sha that arrived after the table was memoised in this process: count it
    # directly rather than silently dropping it from the vocabulary fit
    for s in shas:
        if s not in have:
            a = np.asarray(load_ids(s, cache_root))
            if len(a):
                out += np.bincount(a.astype(np.int64), minlength=mat.shape[1])
    return out[:n]


# ---------------------------------------------------------------------------
# the model vocabulary - fit on training files only
# ---------------------------------------------------------------------------
class Vocab:
    """Global mnemonic id -> model id, with everything unseen in training
    collapsed onto `<unk>`.

    id 0 is `<pad>` and id 1 is `<unk>`; both are reserved and neither is ever
    produced by a training mnemonic. `lut` is applied with a single fancy-index
    per window, so the fold-safety costs nothing at read time.
    """

    def __init__(self, lut: np.ndarray, tokens: list[str]):
        self.lut = np.asarray(lut, dtype=np.int64)
        self.tokens = list(tokens)            # model id -> string
        self.size = len(self.tokens)

    @classmethod
    def fit(cls, train_shas, cache_root: Path = CACHE_ROOT,
            min_count: int = 1) -> "Vocab":
        """Which mnemonics exist is decided by the TRAINING files alone."""
        vocab = global_vocab(cache_root)
        counts = token_counts(train_shas, cache_root, len(vocab))
        keep = np.flatnonzero(counts >= max(1, min_count))
        lut = np.full(len(vocab), UNK_ID, dtype=np.int64)
        lut[keep] = np.arange(N_SPECIAL, N_SPECIAL + len(keep))
        tokens = ["<pad>", "<unk>"] + [vocab[i] for i in keep]
        return cls(lut, tokens)

    def to_dict(self) -> dict:
        return {"tokens": self.tokens, "lut": self.lut.tolist()}

    @classmethod
    def from_dict(cls, d) -> "Vocab":
        return cls(np.asarray(d["lut"], dtype=np.int64), d["tokens"])

    def encode(self, global_ids: np.ndarray) -> np.ndarray:
        g = np.asarray(global_ids, dtype=np.int64)
        return self.lut[g] if len(g) else g


# ---------------------------------------------------------------------------
# multiple-instance windowing over the WHOLE stream
# ---------------------------------------------------------------------------
def window_offsets(n_tokens: int, W: int, N: int) -> np.ndarray:
    """Start offsets of up to `N` windows of `W` tokens over `n_tokens`.

    * a stream no longer than one window is one (padded) window;
    * otherwise `ceil(n/W)` windows would tile it, and when that many are
      allowed the offsets are spread evenly from 0 to n-W, so consecutive
      starts are at most W apart and the union of the windows is the WHOLE
      stream - first window AND last window, never just the head;
    * when `ceil(n/W) > N` the same even spread is used with N windows, which
      leaves regular gaps rather than truncating the file at token N*W.
    """
    if W <= 0 or N <= 0:
        raise ValueError("W and N must be positive")
    if n_tokens <= W:
        return np.zeros(1, dtype=np.int64)
    need = -(-n_tokens // W)                       # ceil
    k = min(N, need)
    if k == 1:
        return np.zeros(1, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, n_tokens - W, k)).astype(np.int64))


def file_windows(sha: str, vocab: Vocab, W: int, N: int,
                 cache_root: Path = CACHE_ROOT):
    """(windows int64 [n_win, W] padded with PAD_ID, lengths int64 [n_win])."""
    arr = load_ids(sha, cache_root)
    n = int(arr.shape[0])
    if n == 0:
        w = np.full((1, W), PAD_ID, dtype=np.int64)
        w[0, 0] = UNK_ID                    # never hand the model an empty window
        return w, np.ones(1, dtype=np.int64)
    offs = window_offsets(n, W, N)
    out = np.full((len(offs), W), PAD_ID, dtype=np.int64)
    lens = np.zeros(len(offs), dtype=np.int64)
    for i, o in enumerate(offs):
        piece = vocab.encode(np.asarray(arr[o:o + W]))
        out[i, :len(piece)] = piece
        lens[i] = len(piece)
    return out, lens


# ---------------------------------------------------------------------------
# imports side-input (interface only; see run_family_holdout.py --imports)
# ---------------------------------------------------------------------------
def hash_imports(names, dim: int = 2048) -> np.ndarray:
    """Signed feature hashing of a bag of import names into `dim` floats.

    Deterministic across processes and interpreters: Python's `hash()` is
    salted per process, so the digest of the lowercased name is used instead.
    The sign trick keeps collisions from systematically inflating the norm.
    L2-normalised so a 500-import binary and a 5-import one are comparable.
    """
    import hashlib
    v = np.zeros(dim, dtype=np.float32)
    for name in names:
        h = hashlib.blake2b(str(name).strip().lower().encode("utf-8"),
                            digest_size=8).digest()
        k = int.from_bytes(h, "little")
        v[k % dim] += 1.0 if (k >> 63) & 1 else -1.0
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def load_imports(path, dim: int = 2048) -> dict:
    """`{sha256: [import names]}` -> `{sha256: hashed vector}`."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: hash_imports(v, dim) for k, v in doc.items()}


# ---------------------------------------------------------------------------
# architecture-balanced batches
# ---------------------------------------------------------------------------
def arch_balanced_weights(labels, archs) -> np.ndarray:
    """Sampling weight per file so that each (label, arch) cell is equally
    likely to supply the next file in a batch.

    The cohort is ~95% x86 and the x64 ransomware is 114 files concentrated in
    one fold, so uniform batches show the model roughly one x64 ransomware
    sample per two batches. Weighting by 1/|cell| makes the four cells equal in
    expectation, which is what "architecture-balanced" has to mean when the
    cells cannot be made equal by construction.
    """
    labels = np.asarray(labels, dtype=int)
    archs = np.asarray(archs, dtype=object)
    w = np.zeros(len(labels), dtype=np.float64)
    cells = {(int(l), str(a)) for l, a in zip(labels, archs)}
    for (l, a) in cells:
        m = (labels == l) & (archs == a)
        w[m] = 1.0 / float(m.sum())
    return w / w.sum()


def class_balanced_weights(labels) -> np.ndarray:
    """The ablation's fallback: balance the two classes only, ignore arch.

    Not "no sampler at all" - the untuned CNN-ViT recipe this replaces uses a
    class-weighted sampler too, so turning arch balancing off has to leave the
    class balancing in or the ablation measures two changes at once.
    """
    labels = np.asarray(labels, dtype=int)
    w = np.zeros(len(labels), dtype=np.float64)
    for l in np.unique(labels):
        m = labels == l
        w[m] = 1.0 / float(m.sum())
    return w / w.sum()


def main() -> int:
    ap = argparse.ArgumentParser(description="build the mnemonic token cache")
    ap.add_argument("--cache", default=str(CACHE_ROOT))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    build_cache(cache_root=Path(a.cache), workers=a.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
