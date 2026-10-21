"""Calibration of predicted probabilities, overall and by group."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from fairaudit.errors import AuditInputError
from fairaudit.metrics import as_binary, as_groups, as_scores, check_lengths


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """One reliability-diagram bin: mean predicted probability vs observed positive rate."""

    lower: float
    upper: float
    count: int
    mean_score: float
    observed_rate: float


def calibration_bins(
    y_true: ArrayLike, y_score: ArrayLike, n_bins: int = 10
) -> list[CalibrationBin]:
    """Uniform-width reliability bins; empty bins are omitted.

    Binning matches ``sklearn.calibration.calibration_curve(strategy="uniform")``.
    """
    if n_bins < 1:
        raise AuditInputError("n_bins must be >= 1")
    y, s = as_binary(y_true, "y_true"), as_scores(y_score)
    check_lengths(y_true=y, y_score=s)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.searchsorted(edges[1:-1], s)
    bins = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            bins.append(
                CalibrationBin(
                    lower=float(edges[b]),
                    upper=float(edges[b + 1]),
                    count=int(mask.sum()),
                    mean_score=float(s[mask].mean()),
                    observed_rate=float(y[mask].mean()),
                )
            )
    return bins


def expected_calibration_error(bins: list[CalibrationBin]) -> float:
    """Count-weighted mean absolute gap between predicted and observed rates."""
    total = sum(b.count for b in bins)
    if total == 0:
        return math.nan
    return sum(b.count * abs(b.observed_rate - b.mean_score) for b in bins) / total


def brier_score(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Mean squared error between probabilities and outcomes (lower is better)."""
    y, s = as_binary(y_true, "y_true"), as_scores(y_score)
    check_lengths(y_true=y, y_score=s)
    return float(np.mean((s - y) ** 2)) if y.size else math.nan


def calibration_by_group(
    y_true: ArrayLike, y_score: ArrayLike, groups: ArrayLike, n_bins: int = 10
) -> dict[str, list[CalibrationBin]]:
    """Reliability bins for each group."""
    y, s, g = as_binary(y_true, "y_true"), as_scores(y_score), as_groups(groups)
    check_lengths(y_true=y, y_score=s, groups=g)
    return {str(v): calibration_bins(y[g == v], s[g == v], n_bins) for v in np.unique(g)}
