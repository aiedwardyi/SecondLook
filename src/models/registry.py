"""Catalog of the shipped before- and after-correction BCC weights."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    path: str
    sha: str | None
    label: str
    padding_mode: str
    backbone: str


_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"

_REGISTRY = {
    "before": ModelSpec(
        path=str(_WEIGHTS_DIR / "bcc_before.pth"),
        sha=None,
        label="before",
        padding_mode="zeros",
        backbone="efficientnet_b4",
    ),
    "after": ModelSpec(
        path=str(_WEIGHTS_DIR / "bcc_after.pth"),
        sha=None,
        label="after",
        padding_mode="reflect",
        backbone="efficientnet_b4",
    ),
}


def resolve(model_id) -> ModelSpec:
    """Look up a ModelSpec by id ('before' or 'after')."""
    try:
        return _REGISTRY[model_id]
    except KeyError:
        valid = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown model id {model_id!r}; valid ids: {valid}") from None
