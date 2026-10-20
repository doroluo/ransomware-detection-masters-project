"""Step 6, GNN over the block CFG: `gnn_fn_layers` directed, edge-typed GIN layers (`layers.DirGINELayer`) over the 12
edge types of `edges.block_edges`, with the controls and edge augmentations of `edges`. In: block vectors, edge_seg,
edge_src, edge_dst, edge_type. Out: block vectors. RunConfig: gnn_fn_layers, direction, control, untyped, drop_edges, flip_edges."""
