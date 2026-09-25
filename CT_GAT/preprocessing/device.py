import torch


def get_device():
    """Pick CUDA, DirectML (AMD/Intel on Windows), or CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f"Device: {device} ({name})")
        return device

    try:
        import torch_directml

        device = torch_directml.device()
        name = torch_directml.device_name(0).replace("\x00", "").strip()
        print(f"Device: {device} ({name})")
        return device
    except ImportError:
        pass

    print("Device: cpu")
    print(
        "GPU not used: this PyTorch build has no CUDA, and torch-directml is not installed."
    )
    print(
        "NVIDIA: pip install torch --index-url https://download.pytorch.org/whl/cu124"
    )
    print(
        "AMD on Windows: use Python 3.11 and pip install torch-directml"
    )
    return torch.device("cpu")