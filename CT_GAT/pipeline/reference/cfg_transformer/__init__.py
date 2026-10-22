"""The frozen-encoder path to Model 3 (reference): a block dataset export for teammates who train a per-block transformer
encoder, the checkpoint contract they hand back, and the precomputed block vectors that go into the GNN (path 2) and its
no-edge control (path 1). See EXPORT_SCHEMA.md and TRANSFORMER_CONTRACT.md.

block_export.py: the export (read back through common/reader.py). encoder_contract.py: the BlockEncoder protocol, the
checkpoint format and check_checkpoint. encoders/: the CodeBERT encoder and a tiny reference encoder. precompute_blockvec.py:
one vector per distinct block and fold from a frozen encoder.
"""
