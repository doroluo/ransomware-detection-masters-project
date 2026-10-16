"""Transformer paths: a block dataset export for teammates who train a per-block transformer encoder, the checkpoint
contract they hand back, and the precomputed block vectors that go into the GNN (path 2) and its no-edge control
(path 1). Path 3 is the token-counting model of `pipeline/`. See EXPORT_SCHEMA.md and TRANSFORMER_CONTRACT.md.

block_export.py: the export. reader.py: reads it back (BlockStore, FoldView, collate_blocks). encoder_contract.py: the
BlockEncoder protocol, the checkpoint format and check_checkpoint. reference_encoder.py: a tiny encoder for tests and
smoke runs. precompute_blockvec.py: one vector per distinct block and fold from a frozen encoder.
"""
