# Rules for the per-block transformer encoder

You train an encoder that turns one basic block into a vector. We freeze it, embed every distinct block once per fold,
and feed the vectors to our GNN (path 2) and to the same network without edges (path 1). Path 3 is our token-counting
model. All three see the same blocks and the same folds, so the comparison only holds if the rules below hold.

The data is a block export (`EXPORT_SCHEMA.md`), for example `RUNS/export/ours_cap0`, with
`RUNS = C:/Users/chaoa/Downloads/graph_transformer_runs`. Read it with `transformer_paths/reader.py`.

1. **One checkpoint per fold.** Fold k of the export is the test fold k of our 5-fold family-holdout protocol. Train a
   separate encoder for each of the five folds.
2. **Use only your fold's data in the right way.**
   - Train on the `train` binaries of `fold{k}` only (`FoldView(store, k, 'train')`).
   - Select checkpoints, epochs and hyperparameters on the `val` binaries of `fold{k}` only.
   - Never read, embed or otherwise touch the `test` binaries of `fold{k}`, and do not use another fold's checkpoint or
     vocabulary for fold k. The test families of fold k are the training families of the other folds, so a shared
     encoder would leak.
   - Use the fold vocabulary (`lut.npy`, ids 0..3 are PAD, CLS, SEP, MASK). It is cut at training document frequency,
     so it carries no information from the test binaries.
3. **The encoder is a pure function of one block's token sequence.** `forward(tokens [B, L], mask [B, L]) -> [B, d_out]`
   with `tokens` long fold-vocabulary ids, `mask` bool (True on tokens, False on padding), and the encoder having the
   attributes `d_out` and `max_len` (the longest sequence it accepts; blocks are at most 192 tokens, so use
   `max_len >= 192`). The vector of a block must not depend on the other blocks of the batch, on the padding length, on
   the binary the block came from, or on its label. Empty blocks (no token, about 2% of them) are in the data and need a
   finite vector too. Tokens and `tok_kind` are all it may use. It must be deterministic in
   `eval()` mode (no dropout, no sampling), because we embed each distinct sequence once and reuse the vector for every
   copy of it.
4. **Save to `RUNS/encoders/<name>/fold{k}/best.pt`**, with `<name>` the name of your model. The file is the checkpoint
   dict of `transformer_paths/encoder_contract.py` (`save_checkpoint` writes it):
   `module`, `class`, `init_kwargs` (the encoder is rebuilt as `module.class(**init_kwargs)`, so the class must be
   importable and `state_dict` must load into it), `state_dict`, `export_sha`, `fold`, `manifest_sha`, `vocab_sha`,
   `train_ids` and `val_ids` (the sample ids you trained and selected on) with `train_ids_sha` and `val_ids_sha`,
   `d_out`, `max_len`. The shas tie the checkpoint to the export, fold, vocabulary and ids it was made from.
5. **Run `check_checkpoint` before handing a checkpoint over**, once per fold:

   ```
   python -m transformer_paths.encoder_contract --ckpt RUNS/encoders/<name>/fold0/best.pt --export ours_cap0 --fold 0
   ```

   It loads the checkpoint and checks that the shas match the export; that `train_ids` are in the fold's `train` and
   `val_ids` in its `val`, and that no test id is in either; that the output on real blocks has shape `[B, d_out]` and is
   finite; that two eval passes agree; and that a block alone, in a batch, and with extra padding gives the same vector.
   `precompute_blockvec.py` runs the same check and refuses a checkpoint that fails it.

`transformer_paths/reference_encoder.py` is a minimal example of a conforming encoder (embedding, masked mean, linear)
and of how to write the checkpoint.

We then run `python -m transformer_paths.precompute_blockvec --encoder-name <name> --export <export> --folds 0 1 2 3 4`
and train the GNN arms on the vectors.

## The CodeBERT encoder (`codebert_encoder.py`)

`CodeBertBlockEncoder` is an encoder of this contract that needs no training: the frozen `microsoft/codebert-base`, read
from the local Hugging Face cache (`HF_HOME`, `local_files_only`), on a text built from our own vocabulary strings. The block
text is the signature strings (`xor r,r`) and then the API reference strings (`kernel32.dll!CreateFileW`) of the block,
exactly as the cache `vocab.json` spells them, in sequence order and joined with ` ; `. A token that the fold vocabulary maps
to its kind's UNK id, and the specials CLS, SEP and MASK, give the placeholder `<unk>`; PAD, the mnemonic tokens and the
behaviour tags are not put in the text (the signature already carries the mnemonic). An empty block is the empty text. No operand
reaches the model, so no register name or address can appear in the text (`tests/test_codebert_encoder.py` checks 2,000
blocks). The text is cut at 256 subwords, the hidden state of the last layer is averaged over the non-pad subwords, and the
vector has 768 float32 numbers (computed in fp16 on CUDA, fp32 on CPU; two batch shapes agree to about 3e-4).

The weights are not in `state_dict`, so a checkpoint is about 0.1 MB; `init_kwargs` carries the model revision and the sha256 of
`model.safetensors`, and loading refuses other weights. To rebuild the checkpoints and the vectors:

```bash
export HF_HOME=C:/Users/chaoa/Downloads/hf_home HF_HUB_OFFLINE=1
$PY -m transformer_paths.codebert_encoder --write-checkpoints --export ours_cap0 --name codebert
$PY -m transformer_paths.encoder_contract --ckpt RUNS/encoders/codebert/fold0/best.pt --export ours_cap0 --fold 0   # per fold
$PY -m transformer_paths.precompute_blockvec --encoder-name codebert --export ours_cap0 --folds 0 1 2 3 4
```

The vectors land in `<cache_v2>/side/blockvec/ours_cap0_codebert/fold{k}/`; `train.py --block-input vec --block-vec
ours_cap0_codebert` reads them (path 2; `--control no_edges` for path 1), and `queue_gnn.sh --stage paths_codebert` runs both
on seeds 0 1 2. The encoder is not fitted to our data, so the validation split may follow `--seed` (no `--val-seed`).
