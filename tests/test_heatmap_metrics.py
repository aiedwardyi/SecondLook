"""Heatmap metric tests."""

import numpy as np
import pytest

from src.trust.heatmap_metrics import (
    compute_heatmap_metrics,
    corner_ratio,
    edge_ratio,
    topk_mass,
)


def _blank(n: int = 64) -> np.ndarray:
    return np.zeros((n, n), dtype=np.float64)


def test_topk_mass_tight_center_higher_than_scatter():
    tight = _blank()
    tight[28:36, 28:36] = 1.0
    scatter = _blank()
    rng = np.random.default_rng(0)
    scatter += rng.random(scatter.shape) * 0.05
    assert topk_mass(tight) > topk_mass(scatter)


def test_corner_ratio_tr_blob_high():
    cam = _blank()
    cam[0:8, -8:] = 1.0
    assert corner_ratio(cam) > 0.9


def test_corner_ratio_center_blob_low():
    cam = _blank()
    cam[28:36, 28:36] = 1.0
    assert corner_ratio(cam) < 0.05


def test_edge_ratio_border_vs_center():
    border = _blank()
    border[0:3, :] = 1.0
    center = _blank()
    center[28:36, 28:36] = 1.0
    assert edge_ratio(border) > edge_ratio(center)


def test_zero_map_returns_zero():
    cam = _blank()
    m = compute_heatmap_metrics(cam)
    assert m["topk_mass"] == 0.0
    assert m["corner_ratio"] == 0.0
    assert m["edge_ratio"] == 0.0


def test_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        topk_mass(np.zeros((4, 4, 3)))


def test_corner_before_higher_than_tissue_after():
    before = _blank(128)
    before[0:16, -16:] = 1.0
    after = _blank(128)
    after[48:80, 40:88] = 1.0
    b = compute_heatmap_metrics(before)
    a = compute_heatmap_metrics(after)
    assert b["corner_ratio"] > a["corner_ratio"]
    assert a["topk_mass"] >= b["topk_mass"] * 0.5
