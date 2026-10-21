"""Metrics checked against small examples computed by hand."""

import math

import numpy as np
import pytest

from fairaudit import metrics as m
from fairaudit.calibration import (
    brier_score,
    calibration_bins,
    calibration_by_group,
    expected_calibration_error,
)
from fairaudit.errors import AuditInputError

# Group A: tp=2 fn=1 fp=1 tn=2 | Group B: tp=1 fn=1 fp=0 tn=2
Y_TRUE = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0]
Y_PRED = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0]
GROUPS = ["A"] * 6 + ["B"] * 4


def test_confusion_by_group_counts_and_rates() -> None:
    counts = m.confusion_by_group(Y_TRUE, Y_PRED, GROUPS)
    a, b = counts["A"], counts["B"]
    assert (a.tp, a.fn, a.fp, a.tn) == (2, 1, 1, 2)
    assert (b.tp, b.fn, b.fp, b.tn) == (1, 1, 0, 2)
    assert a.selection_rate == pytest.approx(0.5)
    assert b.selection_rate == pytest.approx(0.25)
    assert a.base_rate == b.base_rate == pytest.approx(0.5)
    assert a.tpr == pytest.approx(2 / 3)
    assert b.tpr == pytest.approx(0.5)
    assert a.fpr == pytest.approx(1 / 3)
    assert b.fpr == 0.0
    assert a.fnr == pytest.approx(1 / 3)
    assert a.ppv == pytest.approx(2 / 3)
    assert b.ppv == 1.0
    assert a.accuracy == pytest.approx(4 / 6)
    assert b.balanced_accuracy == pytest.approx((0.5 + 1.0) / 2)


def test_group_disparities() -> None:
    assert m.demographic_parity_difference(Y_PRED, GROUPS) == pytest.approx(0.25)
    assert m.disparate_impact_ratio(Y_PRED, GROUPS) == pytest.approx(0.5)
    assert m.equal_opportunity_difference(Y_TRUE, Y_PRED, GROUPS) == pytest.approx(1 / 6)
    assert m.false_positive_rate_difference(Y_TRUE, Y_PRED, GROUPS) == pytest.approx(1 / 3)
    assert m.equalized_odds_difference(Y_TRUE, Y_PRED, GROUPS) == pytest.approx(1 / 3)
    assert m.predictive_parity_difference(Y_TRUE, Y_PRED, GROUPS) == pytest.approx(1 / 3)
    assert m.base_rate_difference(Y_TRUE, GROUPS) == 0.0
    assert m.selection_rate(Y_PRED) == pytest.approx(0.4)


def test_more_than_two_groups_uses_extremes() -> None:
    y_pred = [1, 1, 0, 0, 1, 0, 0, 0, 0]
    groups = ["a", "a", "a", "b", "b", "b", "c", "c", "c"]  # rates 2/3, 1/3, 0
    assert m.demographic_parity_difference(y_pred, groups) == pytest.approx(2 / 3)
    assert m.disparate_impact_ratio(y_pred, groups) == 0.0


def test_undefined_rates_are_nan_and_ignored() -> None:
    # Group "b" has no actual positives, so its TPR is undefined.
    y_true = [1, 0, 1, 0, 0, 0]
    y_pred = [1, 0, 0, 1, 0, 1]
    groups = ["a", "a", "a", "b", "b", "b"]
    counts = m.confusion_by_group(y_true, y_pred, groups)
    assert math.isnan(counts["b"].tpr)
    assert math.isnan(m.equal_opportunity_difference(y_true, y_pred, groups))
    assert m.equalized_odds_difference(y_true, y_pred, groups) == pytest.approx(2 / 3)
    # All-zero selection rates: the ratio is undefined rather than "fair".
    assert math.isnan(m.disparate_impact_ratio([0, 0, 0, 0], ["a", "a", "b", "b"]))
    # A single group has no disparity.
    assert math.isnan(m.demographic_parity_difference([1, 0], ["a", "a"]))


def test_roc_auc_with_and_without_ties() -> None:
    assert m.roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert m.roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert math.isnan(m.roc_auc([1, 1], [0.2, 0.9]))


def test_calibration_hand_example() -> None:
    y_true, y_score = [0, 1, 1, 0], [0.1, 0.15, 0.8, 0.9]
    bins = calibration_bins(y_true, y_score, n_bins=2)
    assert [(b.count, b.mean_score, b.observed_rate) for b in bins] == [
        (2, pytest.approx(0.125), 0.5),
        (2, pytest.approx(0.85), 0.5),
    ]
    assert expected_calibration_error(bins) == pytest.approx(0.3625)
    assert brier_score(y_true, y_score) == pytest.approx(0.395625)
    by_group = calibration_by_group(y_true, y_score, ["a", "a", "b", "b"], n_bins=2)
    assert [len(v) for v in by_group.values()] == [1, 1]
    assert math.isnan(expected_calibration_error([]))


@pytest.mark.parametrize(
    ("y_true", "y_pred", "match"),
    [
        ([0, 2], [0, 1], "binary"),
        ([0, 1], [0, 1, 1], "same length"),
        ([[0, 1]], [[0, 1]], "1-D"),
    ],
)
def test_invalid_inputs(y_true: list[int], y_pred: list[int], match: str) -> None:
    with pytest.raises(AuditInputError, match=match):
        m.confusion_by_group(y_true, y_pred, ["a"] * len(np.ravel(y_pred)))


def test_invalid_scores() -> None:
    with pytest.raises(AuditInputError, match=r"\[0, 1\]"):
        m.as_scores([0.2, 1.5])
    with pytest.raises(AuditInputError, match="missing"):
        m.as_scores([0.2, float("nan")])
    with pytest.raises(AuditInputError, match="n_bins"):
        calibration_bins([0, 1], [0.1, 0.9], n_bins=0)


def test_booleans_are_accepted() -> None:
    assert m.as_binary([True, False]).tolist() == [1, 0]
