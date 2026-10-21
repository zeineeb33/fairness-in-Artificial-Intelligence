"""Cross-check fairaudit against reference implementations (fairlearn, scikit-learn)."""

import numpy as np
import pytest
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

from fairaudit import metrics as m
from fairaudit.calibration import brier_score, calibration_bins


@pytest.fixture(params=[0, 1, 2])
def sample(request: pytest.FixtureRequest) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(request.param)
    n = 2_000
    groups = rng.choice(["a", "b", "c"], size=n, p=[0.5, 0.3, 0.2])
    y_true = rng.integers(0, 2, size=n)
    # Group-dependent scores so that the metrics are not trivially zero.
    shift = np.select([groups == "a", groups == "b"], [0.15, -0.1], default=0.0)
    y_score = np.clip(0.5 * y_true + 0.25 + shift + rng.normal(0, 0.2, size=n), 0, 1)
    return y_true, y_score, groups


def test_group_metrics_match_fairlearn(sample: tuple[np.ndarray, ...]) -> None:
    y_true, y_score, groups = sample
    y_pred = (y_score >= 0.5).astype(int)
    kw = {"sensitive_features": groups}
    assert m.demographic_parity_difference(y_pred, groups) == pytest.approx(
        demographic_parity_difference(y_true, y_pred, **kw)
    )
    assert m.disparate_impact_ratio(y_pred, groups) == pytest.approx(
        demographic_parity_ratio(y_true, y_pred, **kw)
    )
    assert m.equal_opportunity_difference(y_true, y_pred, groups) == pytest.approx(
        equal_opportunity_difference(y_true, y_pred, **kw)
    )
    assert m.equalized_odds_difference(y_true, y_pred, groups) == pytest.approx(
        equalized_odds_difference(y_true, y_pred, **kw)
    )

    frame = MetricFrame(
        metrics={"sr": selection_rate, "tpr": true_positive_rate, "fpr": false_positive_rate},
        y_true=y_true,
        y_pred=y_pred,
        **kw,
    )
    counts = m.confusion_by_group(y_true, y_pred, groups)
    for group, row in frame.by_group.iterrows():
        c = counts[str(group)]
        assert (c.selection_rate, c.tpr, c.fpr) == pytest.approx(
            (row["sr"], row["tpr"], row["fpr"])
        )


def test_score_metrics_match_sklearn(sample: tuple[np.ndarray, ...]) -> None:
    y_true, y_score, _ = sample
    assert m.roc_auc(y_true, y_score) == pytest.approx(roc_auc_score(y_true, y_score))
    assert m.roc_auc(y_true, np.round(y_score, 1)) == pytest.approx(  # heavy ties
        roc_auc_score(y_true, np.round(y_score, 1))
    )
    assert brier_score(y_true, y_score) == pytest.approx(brier_score_loss(y_true, y_score))

    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="uniform")
    bins = calibration_bins(y_true, y_score, n_bins=10)
    assert [b.observed_rate for b in bins] == pytest.approx(prob_true.tolist())
    assert [b.mean_score for b in bins] == pytest.approx(prob_pred.tolist())
