"""Pick the best available torch device: cuda, then mps, then cpu."""

import torch


def get_best_device() -> str:
    """Best available device string for inference."""
    # is_available() can stay True when CUDA_VISIBLE_DEVICES is empty and device_count is 0.
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"
