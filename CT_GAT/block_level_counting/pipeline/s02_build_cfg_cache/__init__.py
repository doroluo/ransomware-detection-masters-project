"""Step 2, build the CFG cache: each IDA export becomes `<cache>/samples/<sample_id>.npz` (mn, sig, beh, api ids,
blk_insn_ptr, blk_api_ptr, blk_feat, fn_blk_ptr, fn_flags, cfg_edges, call_edges, edge_kind, edge_back), with index.csv and
vocab.json beside it. `build_cache` writes, `check_cache` verifies. RunConfig: none; GT_CACHE picks the cache."""
