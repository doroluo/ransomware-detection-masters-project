# Step 9: pool functions to binary

No module of its own. `HierGNN.bin_readout` is built from the same readout classes as step 7
(`pipeline/s07_pool_blocks_to_functions/readout.py`, chosen by `RunConfig.readout`), applied over `seg_bin`, the
binary index of each function. The output is one vector per binary, which step 11 reads together with `bin_feat`.
