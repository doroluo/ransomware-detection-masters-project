"""Step 4, select subgraph: the whole binary, empty functions removed, capped at MAX_BINARY_BLOCKS (500,000) blocks by
`subgraph.cap_functions`; in training `batching.keep_functions` first drops functions at random. Out: `seg_len`, `seg_feat`,
`blk_seg`, `blk_pos`, `n_fn`. RunConfig: drop_functions, drop_library, batch_blocks. `segment_table` and
`select_segments` serve the legacy transformer only."""
