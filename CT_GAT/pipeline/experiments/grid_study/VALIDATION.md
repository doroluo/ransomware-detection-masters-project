# Validation, 2026-10-28

- Combined model-grid and expanded-grid tests: **20 passed**. Expanded tests rerun with a full study-resume check: **7 passed**. Coverage includes case IDs, complete pooling pairs, budget persistence/STOP cleanup, exact reuse contracts, induced edges/instruction ownership, and disk/resident data equality with no test loading during preparation.
- All three expanded paths completed a real-data CPU smoke (8 train / 8 validation samples), including optimizer, architecture reuse, Model 3 instruction pooling, and 500-block coverage. Smoke scores are not accuracy evidence.
- Fresh assembly/CFG reconstruction of sample `027d99aaaa6803a07d07ce0ba1fa66964388129d3b26dcf8621a3310692b0a61` exactly matched the derived 250-block build: 2,674 instructions, five-channel tokens, owners, match flags, RVAs, block keys, handcrafted features, directed edges, edge attributes and back-edge flags.
- RTX 5080 finite-loss/backprop checks at width 128: Model 1 depth 4 / sequence limit 1024 / batch 32 peaked at 803 MiB on the selected cohort. Model 3 Transformer depth 3 / GAT hidden 256 and depth 3 / block limit 256 / batch 8 peaked at 11,490 MiB (mean) and 11,585 MiB (CLS). These are tested batches, not memory guarantees for every graph.
- The 250-block combined build completed with 5,618 tensors and the same 34 explicit exclusions as the 500-block build. Uncapped combined assembly/CFG preprocessing is running separately. Complete-dataset uncapped training is pending, not yet validated end to end.
- The narrow study completed 6 Model 1 and 12 Model 3 screens. Model 1 three-seed confirmation is complete; Model 3 confirmation was checkpoint-paused because the expanded grid supersedes that selection. Narrow cases are reused only after exact input/training contract checks.

No frozen-Transformer Model 3 arm is included. The expanded 30-hour search and its final confirmation remain ongoing work. Per-stage files report planned/completed counts, partial grids and failures.
