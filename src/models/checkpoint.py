"""Save and load BccModel weights with architecture metadata."""

import torch

from src.models.bcc_model import BccModel

_BACKBONE = "efficientnet_b4"
_N_OUTPUTS = 1


def save_checkpoint(model, path, *, input_size=380, loss="BCEWithLogitsLoss"):
    """Save the state_dict plus primitives-only architecture metadata."""
    metadata = {
        "model_name": "bcc",
        "backbone": _BACKBONE,
        "n_outputs": _N_OUTPUTS,
        "dropout_p": 0.4,
        "in_features": model.backbone.classifier[1].in_features,
        "input_size": input_size,
        "loss": loss,
    }
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def load_checkpoint(
    path, *, map_location: str | torch.device = "cpu"
) -> BccModel:
    """Rebuild a BccModel from a checkpoint, refusing architecture mismatches."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    metadata = checkpoint["metadata"]
    if metadata["backbone"] != _BACKBONE:
        raise ValueError(
            f"backbone mismatch: checkpoint is {metadata['backbone']!r}, "
            f"this repo builds {_BACKBONE!r}"
        )
    if metadata["n_outputs"] != _N_OUTPUTS:
        raise ValueError(
            f"n_outputs mismatch: checkpoint has {metadata['n_outputs']}, "
            f"this repo builds {_N_OUTPUTS}"
        )
    model = BccModel()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
