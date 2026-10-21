"""Group-fairness metrics for binary classifiers.

Inputs are 1-D array-likes: ``y_true`` / ``y_pred`` are binary (0/1 or bool) and ``groups``
holds the value of one sensitive attribute for each sample (compared as strings).

Disparity metrics follow fairlearn's conventions so results are directly comparable:
differences are ``max - min`` over groups and ratios are ``min / max``. Both are symmetric and
work for any number of groups. Undefined quantities (e.g. a TPR for a group without positives)
are ``nan`` and are ignored when aggregating over groups.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fairaudit.errors import AuditInputError

BinaryArray = NDArray[np.int8]


def as_binary(values: ArrayLike, name: str = "values") -> BinaryArray:
    """Validate a 1-D binary array (0/1 or bool) and return it as ``int8``."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise AuditInputError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.dtype == np.bool_:
        return arr.astype(np.int8)
    valid = np.isin(arr, (0, 1))
    if not valid.all():
        bad = np.unique(arr[~valid].astype(str))[:5].tolist()
        raise AuditInputError(f"{name} must be binary (0/1); found values {bad}")
    return arr.astype(np.int8)


def as_scores(values: ArrayLike, name: str = "y_score") -> NDArray[np.float64]:
    """Validate a 1-D array of probabilities in ``[0, 1]``."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise AuditInputError(f"{name} must be 1-D, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise AuditInputError(f"{name} contains missing or infinite values")
    if arr.size and (arr.min() < 0.0 or arr.max() > 1.0):
        raise AuditInputError(f"{name} must contain probabilities in [0, 1]")
    return arr


def as_groups(values: ArrayLike, name: str = "groups") -> NDArray[np.str_]:
    """Return sensitive-attribute values as a 1-D array of strings."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise AuditInputError(f"{name} must be 1-D, got shape {arr.shape}")
    return arr.astype(str)


def check_lengths(**arrays: NDArray[np.generic]) -> int:
    """Ensure all arrays have the same length and return it."""
    lengths = {name: len(arr) for name, arr in arrays.items()}
    if len(set(lengths.values())) > 1:
        raise AuditInputError(f"Inputs must have the same length, got {lengths}")
    return next(iter(lengths.values()), 0)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    """Confusion-matrix counts with the usual rates derived from them."""

    tp: int
    fp: int
    tn: int
    fn: int

    @classmethod
    def from_arrays(cls, y_true: BinaryArray, y_pred: BinaryArray) -> "ConfusionCounts":
        positive, predicted = y_true == 1, y_pred == 1
        return cls(
            tp=int(np.sum(positive & predicted)),
            fp=int(np.sum(~positive & predicted)),
            tn=int(np.sum(~positive & ~predicted)),
            fn=int(np.sum(positive & ~predicted)),
        )

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def base_rate(self) -> float:
        """Share of actual positives, P(Y=1)."""
        return _ratio(self.tp + self.fn, self.n)

    @property
    def selection_rate(self) -> float:
        """Share of predicted positives, P(Ŷ=1)."""
        return _ratio(self.tp + self.fp, self.n)

    @property
    def tpr(self) -> float:
        """True-positive rate (recall, sensitivity), P(Ŷ=1 | Y=1)."""
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def fpr(self) -> float:
        """False-positive rate, P(Ŷ=1 | Y=0)."""
        return _ratio(self.fp, self.fp + self.tn)

    @property
    def fnr(self) -> float:
        """False-negative rate (miss rate), P(Ŷ=0 | Y=1)."""
        return _ratio(self.fn, self.tp + self.fn)

    @property
    def tnr(self) -> float:
        """True-negative rate (specificity), P(Ŷ=0 | Y=0)."""
        return _ratio(self.tn, self.fp + self.tn)

    @property
    def ppv(self) -> float:
        """Positive predictive value (precision), P(Y=1 | Ŷ=1)."""
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def accuracy(self) -> float:
        return _ratio(self.tp + self.tn, self.n)

    @property
    def balanced_accuracy(self) -> float:
        return (self.tpr + self.tnr) / 2


def confusion_by_group(
    y_true: ArrayLike, y_pred: ArrayLike, groups: ArrayLike
) -> dict[str, ConfusionCounts]:
    """Confusion counts for each group, keyed by group value (sorted)."""
    yt, yp, g = as_binary(y_true, "y_true"), as_binary(y_pred, "y_pred"), as_groups(groups)
    check_lengths(y_true=yt, y_pred=yp, groups=g)
    return {str(v): ConfusionCounts.from_arrays(yt[g == v], yp[g == v]) for v in np.unique(g)}


def spread(values: Iterable[float]) -> float:
    """``max - min`` over the defined values (``nan`` if fewer than two)."""
    finite = [v for v in values if not math.isnan(v)]
    return max(finite) - min(finite) if len(finite) >= 2 else math.nan


def min_max_ratio(values: Iterable[float]) -> float:
    """``min / max`` over the defined values (``nan`` if fewer than two or max is 0)."""
    finite = [v for v in values if not math.isnan(v)]
    if len(finite) < 2 or max(finite) == 0:
        return math.nan
    return min(finite) / max(finite)


def _selection_rates(y_pred: ArrayLike, groups: ArrayLike) -> list[float]:
    yp, g = as_binary(y_pred, "y_pred"), as_groups(groups)
    check_lengths(y_pred=yp, groups=g)
    return [float(yp[g == v].mean()) for v in np.unique(g)]


def selection_rate(y_pred: ArrayLike) -> float:
    """Share of positive predictions."""
    yp = as_binary(y_pred, "y_pred")
    return float(yp.mean()) if yp.size else math.nan


def demographic_parity_difference(y_pred: ArrayLike, groups: ArrayLike) -> float:
    """Largest gap in selection rate between two groups (0 = parity)."""
    return spread(_selection_rates(y_pred, groups))


def disparate_impact_ratio(y_pred: ArrayLike, groups: ArrayLike) -> float:
    """Lowest selection rate divided by the highest (1 = parity; < 0.8 fails the 80% rule)."""
    return min_max_ratio(_selection_rates(y_pred, groups))


def base_rate_difference(y_true: ArrayLike, groups: ArrayLike) -> float:
    """Largest gap in the rate of actual positives: disparity already present in the labels."""
    return spread(_selection_rates(y_true, groups))


def equal_opportunity_difference(y_true: ArrayLike, y_pred: ArrayLike, groups: ArrayLike) -> float:
    """Largest gap in true-positive rate between two groups."""
    return spread(c.tpr for c in confusion_by_group(y_true, y_pred, groups).values())


def false_positive_rate_difference(
    y_true: ArrayLike, y_pred: ArrayLike, groups: ArrayLike
) -> float:
    """Largest gap in false-positive rate between two groups."""
    return spread(c.fpr for c in confusion_by_group(y_true, y_pred, groups).values())


def equalized_odds_difference(y_true: ArrayLike, y_pred: ArrayLike, groups: ArrayLike) -> float:
    """Worst of the TPR gap and the FPR gap (fairlearn's ``equalized_odds_difference``)."""
    counts = confusion_by_group(y_true, y_pred, groups).values()
    gaps = [spread(c.tpr for c in counts), spread(c.fpr for c in counts)]
    defined = [gap for gap in gaps if not math.isnan(gap)]
    return max(defined) if defined else math.nan


def predictive_parity_difference(y_true: ArrayLike, y_pred: ArrayLike, groups: ArrayLike) -> float:
    """Largest gap in positive predictive value (precision) between two groups."""
    return spread(c.ppv for c in confusion_by_group(y_true, y_pred, groups).values())


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the ROC curve (Mann-Whitney U statistic, ties count one half)."""
    y = as_binary(y_true, "y_true")
    s = np.asarray(y_score, dtype=np.float64)
    check_lengths(y_true=y, y_score=s)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    average_rank = np.cumsum(counts) - (counts - 1) / 2  # 1-based average rank of each value
    rank_sum = float(average_rank[inverse][y == 1].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
