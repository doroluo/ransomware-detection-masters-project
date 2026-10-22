"""Step 7, pool blocks to functions: `readout.FixedReadout` (sum / sqrt n, mean, max) or `GatedReadout` (attention) over the
blocks of each function (`blk_seg`), plus a Linear of `seg_feat`. Out: one vector per function. RunConfig: readout."""
