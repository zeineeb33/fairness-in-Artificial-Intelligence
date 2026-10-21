import numpy as np
import pytest

from fairaudit.errors import AuditInputError
from fairaudit.metrics import confusion_by_group
from fairaudit.mitigation import GroupThresholdOptimizer, reweighing_weights


def test_reweighing_hand_example() -> None:
    # P(a) = P(b) = 1/2, P(y=1) = 1/2; P(a, 1) = P(b, 0) = 3/8, P(a, 0) = P(b, 1) = 1/8.
    y = [1, 1, 1, 0, 1, 0, 0, 0]
    g = ["a"] * 4 + ["b"] * 4
    expected = [2 / 3, 2 / 3, 2 / 3, 2, 2, 2 / 3, 2 / 3, 2 / 3]
    np.testing.assert_allclose(reweighing_weights(y, g), expected)


def test_reweighing_makes_label_independent_of_group() -> None:
    rng = np.random.default_rng(0)
    g = rng.choice(["a", "b", "c"], size=5_000, p=[0.6, 0.3, 0.1])
    y = (rng.random(5_000) < np.select([g == "a", g == "b"], [0.7, 0.4], 0.2)).astype(int)
    w = reweighing_weights(y, g)
    weighted_rates = [np.average(y[g == v], weights=w[g == v]) for v in ("a", "b", "c")]
    np.testing.assert_allclose(weighted_rates, y.mean())
    assert w.mean() == pytest.approx(1.0)


@pytest.fixture
def biased_scores() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    g = rng.choice(["a", "b"], size=4_000)
    y = rng.integers(0, 2, size=4_000)
    shift = np.where(g == "a", 0.2, -0.1)  # group "a" gets inflated scores
    s = np.clip(0.4 * y + 0.3 + shift + rng.normal(0, 0.15, size=4_000), 0, 1)
    return s, y, g


@pytest.mark.parametrize(
    ("constraint", "rate"),
    [("demographic_parity", "selection_rate"), ("equal_opportunity", "tpr")],
)
def test_group_thresholds_equalise_the_constrained_rate(
    biased_scores: tuple[np.ndarray, ...], constraint: str, rate: str
) -> None:
    s, y, g = biased_scores
    before = confusion_by_group(y, (s >= 0.5).astype(int), g)
    optimizer = GroupThresholdOptimizer(constraint).fit(s, y, g)  # type: ignore[arg-type]
    after = confusion_by_group(y, optimizer.predict(s, g), g)
    gap_before = abs(getattr(before["a"], rate) - getattr(before["b"], rate))
    gap_after = abs(getattr(after["a"], rate) - getattr(after["b"], rate))
    assert gap_before > 0.2
    assert gap_after < 0.02
    assert optimizer.thresholds_["a"] > optimizer.thresholds_["b"]
    assert optimizer.accuracy_ is not None
    assert optimizer.target_rate_ is not None


def test_group_thresholds_errors(biased_scores: tuple[np.ndarray, ...]) -> None:
    s, y, g = biased_scores
    with pytest.raises(ValueError, match="Unknown constraint"):
        GroupThresholdOptimizer("bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="grid_size"):
        GroupThresholdOptimizer(grid_size=1)
    with pytest.raises(RuntimeError, match="not fitted"):
        GroupThresholdOptimizer().predict(s, g)
    optimizer = GroupThresholdOptimizer().fit(s, y, g)
    with pytest.raises(AuditInputError, match="unseen"):
        optimizer.predict([0.5], ["z"])
    with pytest.raises(AuditInputError, match="no positive"):
        GroupThresholdOptimizer("equal_opportunity").fit([0.2, 0.8], [0, 1], ["a", "b"])
