"""Operating-point metrics and the three-tier verdict bucketing."""

import math
import warnings

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import HIGH_THRESH, LOW_THRESH


def _aggregate_by_group(y_true, y_prob, groups):
    """Per-group max() aggregation of (y_true, y_prob); validates length alignment when grouped."""
    if groups is None:
        return list(y_true), list(y_prob)
    if not (len(y_true) == len(y_prob) == len(groups)):
        raise ValueError(
            f"y_true, y_prob, and groups must have the same length; "
            f"got {len(y_true)}, {len(y_prob)}, {len(groups)}"
        )
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    groups_arr = np.asarray(groups)
    unique_groups = np.unique(groups_arr)
    y_true_agg = []
    y_prob_agg = []
    for g in unique_groups:
        mask = groups_arr == g
        y_true_agg.append(int(y_true_arr[mask].max()))
        y_prob_agg.append(float(y_prob_arr[mask].max()))
    return y_true_agg, y_prob_agg


def compute_auroc(y_true, y_prob, groups=None) -> float:
    """Threshold-independent ranking quality (AUROC); NaN for single-class y_true."""
    y_true_agg, y_prob_agg = _aggregate_by_group(y_true, y_prob, groups)
    if len(set(y_true_agg)) < 2:
        warnings.warn(
            "compute_auroc undefined for single-class y_true; returning nan",
            UndefinedMetricWarning,
            stacklevel=2,
        )
        return float("nan")
    return float(roc_auc_score(y_true_agg, y_prob_agg))


def sens_at_spec(y_true, y_prob, target_spec: float = 0.95, groups=None) -> float:
    """Sensitivity at the least stringent threshold that still meets the specificity floor."""
    if not (0.0 <= target_spec <= 1.0):
        raise ValueError(f"target_spec must be in [0, 1]; got {target_spec}")
    y_true_agg, y_prob_agg = _aggregate_by_group(y_true, y_prob, groups)
    if len(set(y_true_agg)) < 2:
        warnings.warn(
            "sens_at_spec undefined for single-class y_true; returning nan",
            UndefinedMetricWarning,
            stacklevel=2,
        )
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true_agg, y_prob_agg)
    spec = 1.0 - fpr
    mask = spec >= target_spec
    if not mask.any():
        warnings.warn(
            "No threshold meets target_spec; returning nan",
            UndefinedMetricWarning,
            stacklevel=2,
        )
        return float("nan")
    last_idx = int(np.where(mask)[0][-1])
    return float(tpr[last_idx])


def spec_at_sens(y_true, y_prob, target_sens: float = 0.95, groups=None) -> float:
    """Specificity at the most stringent threshold meeting the sensitivity floor."""
    y_true_agg, y_prob_agg = _aggregate_by_group(y_true, y_prob, groups)
    threshold = find_threshold_at_sens(y_true_agg, y_prob_agg, target_sens=target_sens)
    if math.isnan(threshold):
        return float("nan")
    y_true_arr = np.asarray(y_true_agg)
    y_pred = (np.asarray(y_prob_agg) >= threshold).astype(int)
    tn = int(((y_pred == 0) & (y_true_arr == 0)).sum())
    fp = int(((y_pred == 1) & (y_true_arr == 0)).sum())
    if tn + fp == 0:
        return float("nan")
    return float(tn / (tn + fp))


def ppv_at_sens(y_true, y_prob, target_sens: float = 0.95, groups=None) -> float:
    """Positive predictive value at the most stringent threshold meeting the sensitivity floor."""
    y_true_agg, y_prob_agg = _aggregate_by_group(y_true, y_prob, groups)
    threshold = find_threshold_at_sens(y_true_agg, y_prob_agg, target_sens=target_sens)
    if math.isnan(threshold):
        return float("nan")
    y_true_arr = np.asarray(y_true_agg)
    y_pred = (np.asarray(y_prob_agg) >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true_arr == 1)).sum())
    fp = int(((y_pred == 1) & (y_true_arr == 0)).sum())
    if tp + fp == 0:
        warnings.warn(
            "ppv_at_sens undefined: no positive predictions at chosen threshold; returning nan",
            UndefinedMetricWarning,
            stacklevel=2,
        )
        return float("nan")
    return float(tp / (tp + fp))


def find_threshold_at_sens(
    y_true, y_prob, target_sens: float = 0.95, groups=None
) -> float:
    """Highest threshold where sensitivity meets the target floor; NaN for single-class y_true."""
    if not (0.0 <= target_sens <= 1.0):
        raise ValueError(f"target_sens must be in [0, 1]; got {target_sens}")
    y_true_agg, y_prob_agg = _aggregate_by_group(y_true, y_prob, groups)
    if len(set(y_true_agg)) < 2:
        warnings.warn(
            "find_threshold_at_sens undefined for single-class y_true; returning nan",
            UndefinedMetricWarning,
            stacklevel=2,
        )
        return float("nan")
    _, tpr, thresholds = roc_curve(y_true_agg, y_prob_agg)
    mask = tpr >= target_sens
    if not mask.any():
        warnings.warn(
            "No threshold reaches target_sens; returning nan",
            UndefinedMetricWarning,
            stacklevel=2,
        )
        return float("nan")
    first_idx = int(np.where(mask)[0][0])
    return float(thresholds[first_idx])


def compute_three_tier(
    y_probs: list[float],
    high_thresh: float = HIGH_THRESH,
    low_thresh: float = LOW_THRESH,
) -> list[str]:
    """Three-tier verdict per probability: p>=high POSITIVE, p<=low NEGATIVE, else UNCERTAIN."""
    high_thresh = round(high_thresh, 4)
    low_thresh = round(low_thresh, 4)
    labels = []
    for p in y_probs:
        p = round(p, 4)
        if p >= high_thresh:
            labels.append("POSITIVE")
        elif p <= low_thresh:
            labels.append("NEGATIVE")
        else:
            labels.append("UNCERTAIN")
    return labels
