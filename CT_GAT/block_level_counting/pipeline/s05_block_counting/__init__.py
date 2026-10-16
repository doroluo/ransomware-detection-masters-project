"""Step 5, block-level counting: a block is a weighted bag of its tokens (`weights.token_weights`: tf-idf, equal weight,
or the raw / presence / rawlog count weightings) summed by an EmbeddingBag, plus a Linear of the six structural block
features (`encoder.BlockEncoder`).
In: tokens, tok_offsets, tok_weights, blk_feat. Out: one vector per block.
With `block_input='vec'` the bag is replaced by precomputed block vectors from a frozen per-block encoder
(`block_vecs.BlockVecs`, `block_vecs.BlockVecSource`; written by transformer_paths/precompute_blockvec.py): a
Linear(d_vec -> d) of the vector plus the same Linear of the structural features (`encoder.BlockVecEncoder`).
In: blk_vec, blk_feat.
RunConfig: encoder, mask_blocks, mask_tokens, d_model, block_input, block_vec."""
