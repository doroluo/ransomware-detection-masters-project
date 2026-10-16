"""Step 3, tokenizer: `vocab.TokenVocab` maps global token ids to run ids (document frequency >= min_df over the training
binaries, UNK = 0, one [MASK] id per kind, idf); `block_tokens.token_table` lays out [mn | sig | beh | api] per block.
Out: batch keys `tokens`, `tok_offsets`. RunConfig: min_df."""
