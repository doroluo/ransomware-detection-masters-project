# token_sequence: Model 1, a transformer over the token sequence of a binary

Model 1 classifies a binary from the instruction and API tokens of its basic blocks, read as one sequence. It reads a
block export (`transformer_paths/EXPORT_SCHEMA.md`) in the fold vocabulary of one fold; it needs neither the CFG cache nor
the graph.

```python
from token_sequence import FoldSequences, collate, Model1TokenSequence, BinaryTokenSequence

train = FoldSequences(export_root, fold=0, split='train', L=512, max_windows=64, seed=0)
net = BinaryTokenSequence(Model1TokenSequence(n_ids=train.n_ids))
batch = collate([train[i] for i in range(8)])
logits = net(batch['ids'], batch['kind'], batch['pad_mask'], batch['win2bin'])      # [8, 2]
```

## Input

A block of the export is the token sequence `[mn_1..mn_n | sig_1..sig_n | beh_1..beh_n | api_1..api_m]` in fold ids (PAD 0,
CLS 1, SEP 2, MASK 3, then the vocabulary fitted on the training binaries with one UNK per kind), cut at instruction
boundaries when longer than 192 tokens.

## Sequence rule

`instruction_stream(fold_view, binary)` lists the positions of a binary in store block order. For each block it regroups
the kind-major layout into its `n` instruction triplets `(mn_i, sig_i, beh_i)`, then appends its `m` API ids, each as an API
position. The counts of the three instruction kinds must agree in every block and the kinds must follow the layout above;
otherwise `regroup` raises `ValueError`. The result is `ids [N, 3]` (an API id sits in slot 0, the other slots are 0) and
`kind [N]` (0 instruction, 1 API). Blocks are concatenated without separators; the empty blocks (about 2%) add nothing.

## Windows

`windows(stream, L=512, train=False, max_windows=64, rng=None)` cuts the stream into contiguous windows of `L` positions,
ignoring block boundaries. The last window is padded (`pad_mask` True on padding, as in `nn.Transformer`).

* Evaluation (`train=False`) returns every window in order: `ceil(N / L)` of them, the same on every call.
* Training returns at most `max_windows` windows. A random start offset in `[0, L)` is drawn from `rng`, the windows after
  it are numbered, and if there are more than `max_windows` of them a random subset is kept in stream order.

`collate` stacks the windows of several binaries: `ids [W, L, 3]`, `kind [W, L]`, `pad_mask [W, L]`, `win2bin [W]` (the
binary of each window), `labels [n]`, `sample_ids`.

`FoldSequences(export_root, fold, split)` iterates the binaries of one split through `FoldView`. It is built for
`'train'`, `'val'` or `'test'` and holds a view of that split only, so code given the training object cannot read a
validation or test id. `'train'` draws random windows (seeded by `(seed, epoch, index)`; call `set_epoch`); `'val'` and
`'test'` return all windows.

## Model

* `InstructionEmbedder(n_ids, d_model)`: one `nn.Embedding(n_ids, d, padding_idx=0)` over the fold vocabulary. An
  instruction position embeds `E[mn] + E[sig] + E[beh]`, an API position `E[api]`, plus a 2-way kind embedding.
* `Model1TokenSequence`: learned positional embedding, `nn.TransformerEncoder` (d_model 64, 4 heads, 2 layers, dropout 0.1,
  `max_seq_len` 512), masked mean pool over the real positions, two-layer MLP head. `encode(ids, kind, pad_mask)` returns the
  window vector `[B, d]`; `forward` returns one logit pair per window. A window longer than `max_seq_len` raises. A window
  without any token is encoded as zeros.
* `BinaryTokenSequence(model)`: encodes every window, averages the window vectors of each binary over its non-empty windows
  and applies the head, giving logits `[n_binaries, 2]`. `window_chunk` bounds how many windows are encoded at once.

## Training

```bash
python -m token_sequence.train --protocol kfold --arch x86 --fold 0 --seed 0 [--seq-len 512 --max-windows 64]
    [--export ours_cap0] [--device cuda|cpu] [--smoke N] [--epochs 40 --patience 8] [--window-budget 128]
bash scripts/model1_runs.sh [--dry]          # seeds 0 1 2 x folds 0-4, then report.py and gates.py --stage model1
```

`train.py` trains `BinaryTokenSequence(Model1TokenSequence)` on one fold of the export (`ours_cap0` under `RUNS/export`, or
`GT_EXPORT`). The run is a `RunConfig` with `model='tokenseq'`: name `kfold<f>_x86_tokenseq_s<seed>`, with `-L<n>` / `-w<n>`
only for a non-default `seq_len` / `max_windows`. Its `val_seed` is forced to 0, the carve of the export's fold manifests, and
is never printed in the name; none of the GNN options (controls, masking, vectors, weightings, ...) is accepted.

* Loss: cross entropy with the class weight `N / (2 n_class)` of the training labels. AdamW lr 3e-4, weight decay 0.01,
  gradient clip 1.0, bf16 autocast on CUDA.
* Batches: whole binaries, in a per-epoch random order, whose windows add up to at most `--window-budget` (a binary with
  more windows than that is alone), so the peak GPU memory is set by the budget. Evaluation runs every window of every
  binary in order under `no_grad`, in chunks of the same size.
* Each epoch records the validation AUC and the macro-F1 at 0.5 in `history.csv`. `best.pt` is the epoch of the best macro-F1
  (predictions at 0.5), `best_auc.pt` that of the best AUC (predictions at the `val_threshold` maximiser of the validation
  macro-F1). Training stops when the AUC has not improved for `patience` epochs. The test split is read once, after training.
* The run directory is written by `evaluation/run_io.write_run` with the export's manifest, so `metrics.json`, `predictions.csv`
  and `predictions_auc.csv` have the layout of a GNN run, and `evaluation/stats.py`, `report.py` and `gates.py --stage model1`
  read it.
* `--smoke N`: N training binaries (half of each class), N validation and N test binaries, 2 epochs, a temporary directory
  (never under `RUNS`). It prints s/epoch, peak GPU and RAM, and projects the time of a full fold run from the windows and
  positions of the whole split.

Tests: `python -m pytest tests/test_model1_trainer.py tests/test_harness_prep.py`.

## Credit

The encoder is Janice Park's `Model1TokenSequence` (`src/models/token_sequence.py` on `origin/TokenSeqModel`): positional
embedding, transformer encoder, masked mean pool and MLP head are hers. Changed here: the five-channel embedder is replaced by
the single-table `InstructionEmbedder`, `max_seq_len` defaults to 512, `encode` is separate from the head, empty windows are
handled, and `BinaryTokenSequence` pools windows into binaries.

## Tests

`python -m pytest tests/test_token_sequence.py` (CPU, on the small export built from `tests/tiny_cache.py`).
