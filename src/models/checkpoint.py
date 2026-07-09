"""Save and load BccModel weights with architecture metadata and a padding-mismatch guard."""

import torch

from src.models.bcc_model import BccModel

_BACKBONE = "efficientnet_b4"
_N_OUTPUTS = 1


def save_checkpoint(model, path, *, input_size=380, loss="BCEWithLogitsLoss"):
    """Save the state_dict plus primitives-only metadata that pins the padding mode."""
    metadata = {
        "model_name": "bcc",
        "backbone": _BACKBONE,
        "padding_mode": model.padding_mode,
        "n_outputs": _N_OUTPUTS,
        "dropout_p": 0.4,
        "in_features": model.backbone.classifier[1].in_features,
        "input_size": input_size,
        "loss": loss,
    }
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def load_checkpoint(
    path, *, expected_padding_mode: str | None = None, map_location: str | torch.device = "cpu"
) -> BccModel:
    """Rebuild a BccModel from a checkpoint, refusing padding or architecture mismatches."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    metadata = checkpoint["metadata"]
    padding_mode = metadata["padding_mode"]
    # zeros-pad and reflect-pad checkpoints share identical state_dict shapes,
    # so load_state_dict cannot catch a padding mismatch - this is the only guard.
    if expected_padding_mode is not None and padding_mode != expected_padding_mode:
        raise ValueError(
            f"padding_mode mismatch: checkpoint is {padding_mode!r}, "
            f"expected {expected_padding_mode!r}"
        )
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
    model = BccModel(padding_mode=padding_mode)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
