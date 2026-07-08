"""Pick the best available torch device: cuda, then mps, then cpu."""

import torch


def get_best_device() -> str:
    """Best available device string for inference."""
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"
