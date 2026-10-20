"""Step 4, select subgraph: whole functions only. `subgraph.select_subgraph` keeps the functions of a binary that fit in
a block budget (default MAX_BINARY_BLOCKS, 500,000) by the policy 'cap' (`cap_functions` order, the only policy); in
training `batching.keep_functions` first drops functions at random. Out: `seg_len`, `seg_feat`, `blk_seg`, `blk_pos`,
`n_fn`. RunConfig: subgraph, budget, drop_functions, drop_library, batch_blocks.
`subgraph.chain_reduction` measures the blocks a chain merge could remove (no model change)."""
