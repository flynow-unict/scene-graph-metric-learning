"""Selezione del device: CUDA (NVIDIA o AMD ROCm), MPS o CPU."""
import torch


def get_device(verbose: bool = True) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        backend = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
        name, backend = "Apple Silicon", "MPS"
    else:
        device = torch.device("cpu")
        name, backend = "CPU", "CPU"

    if verbose:
        print(f"[device] {backend} | {device} | {name}")
    return device
