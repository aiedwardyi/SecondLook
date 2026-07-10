"""Attention metrics from a Grad-CAM map."""

from __future__ import annotations

from typing import Any

import numpy as np

_DEFAULT_TOPK_FRAC = 0.05
_DEFAULT_CORNER_FRAC = 0.15
_DEFAULT_EDGE_FRAC = 0.08


def _as_2d_float(cam: np.ndarray) -> np.ndarray:
    arr = np.asarray(cam, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"cam must be 2-D HxW; got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("cam is empty")
    return arr


def _total_mass(cam: np.ndarray) -> float:
    return float(np.clip(cam, 0.0, None).sum())


def topk_mass(cam: np.ndarray, frac: float = _DEFAULT_TOPK_FRAC) -> float:
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"frac must be in (0, 1]; got {frac}")
    arr = _as_2d_float(cam)
    flat = np.clip(arr, 0.0, None).ravel()
    total = float(flat.sum())
    if total <= 0.0:
        return 0.0
    k = max(1, int(round(frac * flat.size)))
    if k >= flat.size:
        return 1.0
    part = np.argpartition(flat, -k)[-k:]
    return float(flat[part].sum() / total)


def corner_ratio(cam: np.ndarray, corner_frac: float = _DEFAULT_CORNER_FRAC) -> float:
    if not (0.0 < corner_frac <= 0.5):
        raise ValueError(f"corner_frac must be in (0, 0.5]; got {corner_frac}")
    arr = _as_2d_float(cam)
    h, w = arr.shape
    total = _total_mass(arr)
    if total <= 0.0:
        return 0.0
    ch = max(1, int(round(h * corner_frac)))
    cw = max(1, int(round(w * corner_frac)))
    mass = (
        arr[:ch, :cw].sum()
        + arr[:ch, w - cw :].sum()
        + arr[h - ch :, :cw].sum()
        + arr[h - ch :, w - cw :].sum()
    )
    return float(mass / total)


def edge_ratio(cam: np.ndarray, edge_frac: float = _DEFAULT_EDGE_FRAC) -> float:
    if not (0.0 < edge_frac <= 0.5):
        raise ValueError(f"edge_frac must be in (0, 0.5]; got {edge_frac}")
    arr = _as_2d_float(cam)
    h, w = arr.shape
    total = _total_mass(arr)
    if total <= 0.0:
        return 0.0
    band = max(1, int(round(min(h, w) * edge_frac)))
    mask = np.zeros_like(arr, dtype=bool)
    mask[:band, :] = True
    mask[-band:, :] = True
    mask[:, :band] = True
    mask[:, -band:] = True
    return float(np.clip(arr, 0.0, None)[mask].sum() / total)


def compute_heatmap_metrics(
    cam: np.ndarray,
    *,
    topk_frac: float = _DEFAULT_TOPK_FRAC,
    corner_frac: float = _DEFAULT_CORNER_FRAC,
    edge_frac: float = _DEFAULT_EDGE_FRAC,
) -> dict[str, Any]:
    return {
        "topk_mass": topk_mass(cam, frac=topk_frac),
        "corner_ratio": corner_ratio(cam, corner_frac=corner_frac),
        "edge_ratio": edge_ratio(cam, edge_frac=edge_frac),
        "topk_frac": topk_frac,
        "corner_frac": corner_frac,
        "edge_frac": edge_frac,
    }
