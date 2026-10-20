"""Stage 3b, the CFG feature model: a hierarchical, edge-typed graph neural network over the block CFG and the call graph.
`gnn` holds the model and its data path, one subpackage per step of the plan; `gnn.spec.GnnSpec` is the model's entry in the
registry (`common/registry.py`). See README.md."""
