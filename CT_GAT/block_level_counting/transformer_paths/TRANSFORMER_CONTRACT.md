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
