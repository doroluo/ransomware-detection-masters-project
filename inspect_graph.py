import os
import sys
import torch


def inspect_graph(file_path: str):
    if not os.path.exists(file_path):
        print(f"\nError: File '{file_path}' does not exist.")
        return

    # Load saved PyG Data object
    data = torch.load(file_path, weights_only=False)

    print(f"\n=== Graph File: {file_path} ===")
    print(data)

    print("\n=== Detailed Tensor Specs ===")
    print(f"Node Features (x)      : {data.x.shape}")
    
    # Handle both edge_index_cfg and standard edge_index
    edge_tensor = getattr(data, 'edge_index_cfg', getattr(data, 'edge_index', None))
    print(f"CFG Edges (edge_index) : {edge_tensor.shape if edge_tensor is not None else 'N/A'}")
    
    if hasattr(data, 'func_batch'):
        print(f"Function Mapping       : {data.func_batch.shape}")
        
    label = data.y.item()
    label_str = "Ransomware (1)" if label == 1 else "Goodware (0)"
    print(f"Target Label (y)       : {label} ({label_str})")
    if getattr(data, "sample_id", None):
        print(f"Sample id              : {data.sample_id}")
    if getattr(data, "family", None):
        print(f"Family                 : {data.family}")
    if getattr(data, "architecture", None):
        print(f"Architecture           : {data.architecture}")
    if getattr(data, "num_blocks_total", None):
        print(f"Blocks kept / total    : {data.num_nodes} / {data.num_blocks_total}")

    # Inspect first 3 basic blocks across first 5 embedding dimensions
    print("\nSample Node Features (First 3 blocks, 5 dims):")
    print(data.x[:3, :5])


if __name__ == "__main__":
    # Option 1: Passed as command-line argument (e.g., python inspect_graph.py path/to/file.pt)
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        # Option 2: Prompt interactively if no argument was passed
        default_path = "processed_graphs/train/graph_0.pt"
        user_input = input(f"Enter path to .pt file [Press Enter for default: {default_path}]: ").strip()
        target_path = user_input if user_input else default_path

    inspect_graph(target_path)