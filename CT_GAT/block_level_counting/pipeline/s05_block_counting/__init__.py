"""Step 5, block-level counting: a block is a weighted bag of its tokens (`weights.token_weights`: tf-idf or equal weight)
summed by an EmbeddingBag, plus a Linear of the six structural block features (`encoder.BlockEncoder`).
In: tokens, tok_offsets, tok_weights, blk_feat. Out: one vector per block. RunConfig: encoder, mask_blocks, mask_tokens, d_model."""
