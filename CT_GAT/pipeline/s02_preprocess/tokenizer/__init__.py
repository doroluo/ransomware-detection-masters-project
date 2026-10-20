"""Tokenizer: `vocab.TokenVocab` maps global token ids to run ids (document frequency >= min_df over the training
binaries, UNK = 0, one [MASK] id per kind, idf); `block_tokens.token_table` lays out [mn | sig | beh | api] per block.
With `vocab` 'noapi' the blocks hold mn, sig and beh only (the api-token ablation; header imports are a separate switch).
With `vocab` 'yanping' the instruction kinds are the reader's three ids (y0 y1 y2) and with 'hybrid' mn, beh, y1, y2; both
read <cache>/side/yanping/, which `experiments/side_arrays/import_reader_side.py` (not in this package) writes from the team's operand-role reader
output (an experiment, not on the main path).
Out: batch keys `tokens`, `tok_offsets`. RunConfig: min_df, vocab."""
