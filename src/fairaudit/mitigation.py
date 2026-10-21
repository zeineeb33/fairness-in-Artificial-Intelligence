"""Model-agnostic bias-mitigation strategies (numpy only).

* :func:`reweighing_weights`: pre-processing, Kamiran & Calders (2012).
* :class:`GroupThresholdOptimizer`: post-processing with one decision threshold per group,
  in the spirit of Hardt et al. (2016) but deterministic (no randomised thresholds).

Feature-level strategies (removing the sensitive attribute, PCA representation) act on a
model's inputs; they are implemented in the case-study pipeline (:mod:`fairaudit.casestudy`).
"""

from typing import Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fairaudit.errors import AuditInputError
from fairaudit.metrics import as_binary, as_groups, as_scores, check_lengths

Constraint = Literal["demographic_parity", "equal_opportunity"]


def reweighing_weights(y_true: ArrayLike, groups: ArrayLike) -> NDArray[np.float64]:
    """Kamiran-Calders sample weights ``w(s, y) = P(s) * P(y) / P(s, y)``.

    Under these weights the label is statistically independent of the group, so a model
    trained with them cannot simply learn the group's base rate. The weights average to 1.
    """
    y, g = as_binary(y_true, "y_true"), as_groups(groups)
    check_lengths(y_true=y, groups=g)
    weights = np.ones(len(y), dtype=np.float64)
    for value in np.unique(g):
        in_group = g == value
        for label in (0, 1):
            cell = in_group & (y == label)
            if cell.any():
                weights[cell] = in_group.mean() * (y == label).mean() / cell.mean()
    return weights


class GroupThresholdOptimizer:
    """Choose one threshold per group so that a fairness constraint holds approximately.

    For every candidate target rate ``t`` on a grid, each group gets the score quantile that
    makes its rate equal to ``t``: the selection rate for ``demographic_parity``, the
    true-positive rate for ``equal_opportunity``. The ``t`` with the best overall accuracy
    is kept. Fit it on held-out data (not the model's training set).
    """

    def __init__(self, constraint: Constraint = "demographic_parity", grid_size: int = 99) -> None:
        if constraint not in ("demographic_parity", "equal_opportunity"):
            raise ValueError(f"Unknown constraint: {constraint}")
        if grid_size < 2:
            raise ValueError("grid_size must be >= 2")
        self.constraint = constraint
        self.grid_size = grid_size
        self.thresholds_: dict[str, float] = {}
        self.target_rate_: float | None = None
        self.accuracy_: float | None = None

    def fit(self, y_score: ArrayLike, y_true: ArrayLike, groups: ArrayLike) -> Self:
        s, y, g = as_scores(y_score), as_binary(y_true, "y_true"), as_groups(groups)
        check_lengths(y_score=s, y_true=y, groups=g)
        references = {}
        for value in np.unique(g):
            mask = g == value
            reference = s[mask] if self.constraint == "demographic_parity" else s[mask & (y == 1)]
            if reference.size == 0:
                raise AuditInputError(f"Group '{value}' has no positive samples to calibrate on")
            references[str(value)] = reference

        best: tuple[float, float, dict[str, float]] | None = None
        for rate in np.linspace(0.01, 0.99, self.grid_size):
            thresholds = {k: float(np.quantile(ref, 1.0 - rate)) for k, ref in references.items()}
            accuracy = float(np.mean(self._apply(s, g, thresholds) == y))
            if best is None or accuracy > best[0]:
                best = (accuracy, float(rate), thresholds)
        assert best is not None
        self.accuracy_, self.target_rate_, self.thresholds_ = best
        return self

    def predict(self, y_score: ArrayLike, groups: ArrayLike) -> NDArray[np.int8]:
        if not self.thresholds_:
            raise RuntimeError("GroupThresholdOptimizer is not fitted")
        s, g = as_scores(y_score), as_groups(groups)
        check_lengths(y_score=s, groups=g)
        if unknown := set(np.unique(g)) - set(self.thresholds_):
            raise AuditInputError(f"Groups unseen during fit: {sorted(unknown)}")
        return self._apply(s, g, self.thresholds_)

    @staticmethod
    def _apply(
        s: NDArray[np.float64], g: NDArray[np.str_], thresholds: dict[str, float]
    ) -> NDArray[np.int8]:
        values, inverse = np.unique(g, return_inverse=True)
        cutoffs = np.array([thresholds.get(str(v), np.inf) for v in values])[inverse]
        return (s >= cutoffs).astype(np.int8)
