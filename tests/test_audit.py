import pandas as pd
import pytest
from pydantic import ValidationError

from fairaudit.audit import MISSING_GROUP, ColumnMapping, audit, audit_dataframe
from fairaudit.errors import AuditInputError
from fairaudit.report import AuditPolicy, AuditReport

Y_TRUE = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0]
Y_PRED = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0]
GROUPS = ["A"] * 6 + ["B"] * 4


def test_audit_report_values_and_findings() -> None:
    report = audit(Y_TRUE, Y_PRED, sensitive={"group": GROUPS})
    attr = report.attribute("group")
    assert report.n_samples == 10
    assert report.threshold is None
    assert not report.has_scores
    assert report.overall.accuracy == pytest.approx(0.7)
    assert [g.group for g in attr.groups] == ["A", "B"]
    assert [g.selection_rate for g in attr.groups] == [0.5, 0.25]
    assert attr.disparate_impact_ratio == pytest.approx(0.5)
    assert attr.passes_four_fifths_rule is False
    assert attr.equalized_odds_difference == pytest.approx(1 / 3, abs=1e-6)
    text = " ".join(attr.findings)
    for keyword in ("Disparate impact", "Equal opportunity", "False positives", "Small groups"):
        assert keyword in text
    assert "Bias amplification" in text  # labels are balanced, predictions are not
    assert all(f.startswith("[group]") for f in report.findings)
    with pytest.raises(KeyError):
        report.attribute("missing")


def test_fair_predictions_produce_no_findings() -> None:
    y = [1, 0] * 50
    groups = ["a"] * 50 + ["b"] * 50
    report = audit(y, y, sensitive={"g": groups}, policy=AuditPolicy(min_group_size=10))
    attr = report.attribute("g")
    assert attr.passes_four_fifths_rule is True
    assert attr.findings == []


def test_scores_enable_calibration_and_threshold() -> None:
    y_score = [0.9, 0.8, 0.4, 0.7, 0.2, 0.1, 0.6, 0.3, 0.2, 0.1]
    report = audit(Y_TRUE, sensitive={"group": GROUPS}, y_score=y_score, threshold=0.5)
    assert report.threshold == 0.5
    assert report.has_scores
    assert report.overall.roc_auc is not None
    assert [g.selection_rate for g in report.attribute("group").groups] == [0.5, 0.25]
    group_a = report.attribute("group").groups[0]
    assert group_a.ece is not None
    assert sum(b.count for b in group_a.calibration) == group_a.n


def test_json_round_trip() -> None:
    report = audit(Y_TRUE, Y_PRED, sensitive={"group": GROUPS, "other": ["x", "y"] * 5})
    assert AuditReport.model_validate_json(report.model_dump_json()) == report


def test_single_class_labels_warn_and_yield_undefined_rates() -> None:
    report = audit([1, 1, 1, 1], [1, 0, 1, 1], sensitive={"g": ["a", "a", "b", "b"]})
    assert report.warnings
    assert report.overall.fpr is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sensitive": {}}, "sensitive attribute"),
        ({"sensitive": {"g": GROUPS}, "y_pred": None}, "predictions, scores"),
        ({"sensitive": {"g": GROUPS[:5]}}, "same length"),
        (
            {"sensitive": {"g": GROUPS}, "y_pred": None, "y_score": [0.5] * 10, "threshold": 2},
            "threshold",
        ),
    ],
)
def test_audit_input_errors(kwargs: dict[str, object], match: str) -> None:
    kwargs.setdefault("y_pred", Y_PRED)
    with pytest.raises(AuditInputError, match=match):
        audit(Y_TRUE, **kwargs)  # type: ignore[arg-type]


def test_empty_input_is_rejected() -> None:
    with pytest.raises(AuditInputError, match="empty"):
        audit([], [], sensitive={"g": []})


def test_too_many_groups_is_rejected() -> None:
    with pytest.raises(AuditInputError, match="bin continuous"):
        audit([0, 1] * 30, [0, 1] * 30, sensitive={"age": list(range(60))})


def test_audit_dataframe_coerces_columns() -> None:
    df = pd.DataFrame(
        {
            "outcome": ["sick", "healthy", "sick", "healthy"] * 10,
            "flagged": ["yes", "no", "no", "no"] * 10,
            "proba": [0.9, 0.2, 0.4, 0.1] * 10,
            "sex": ["F", "M", None, "M"] * 10,
        }
    )
    mapping = ColumnMapping(
        label="outcome",
        prediction="flagged",
        score="proba",
        sensitive=["sex"],
        positive_label="sick",
    )
    report = audit_dataframe(df, mapping)
    groups = {g.group: g for g in report.attribute("sex").groups}
    assert set(groups) == {"F", "M", MISSING_GROUP}
    assert groups["F"].tpr == 1.0
    assert groups[MISSING_GROUP].tpr == 0.0
    assert any("missing" in w for w in report.warnings)
    assert report.has_scores


def test_audit_dataframe_from_scores_only() -> None:
    df = pd.DataFrame({"y": [1, 0, 1, 0], "s": [0.8, 0.3, 0.6, 0.7], "g": ["a", "a", "b", "b"]})
    report = audit_dataframe(
        df, ColumnMapping(label="y", score="s", sensitive=["g"]), threshold=0.65
    )
    assert [g.selection_rate for g in report.attribute("g").groups] == [0.5, 0.5]


@pytest.mark.parametrize(
    ("data", "mapping", "match"),
    [
        ({"y": [0, 1]}, {"label": "y", "prediction": "p", "sensitive": ["g"]}, "not found"),
        (
            {"y": [0, 2], "p": [0, 1], "g": ["a", "b"]},
            {"label": "y", "prediction": "p", "sensitive": ["g"]},
            "binary",
        ),
        (
            {"y": ["a", "b"], "p": [0, 1], "g": ["a", "b"]},
            {"label": "y", "prediction": "p", "sensitive": ["g"], "positive_label": "z"},
            "does not occur",
        ),
        (
            {"y": [0, None], "p": [0, 1], "g": ["a", "b"]},
            {"label": "y", "prediction": "p", "sensitive": ["g"]},
            "missing",
        ),
        (
            {"y": [0, 1], "s": ["0.2", "x"], "g": ["a", "b"]},
            {"label": "y", "score": "s", "sensitive": ["g"]},
            "non-numeric",
        ),
        (
            {"y": [0, 1], "s": [0.2, 3.0], "g": ["a", "b"]},
            {"label": "y", "score": "s", "sensitive": ["g"]},
            "probabilities",
        ),
    ],
)
def test_audit_dataframe_errors(
    data: dict[str, list[object]], mapping: dict[str, object], match: str
) -> None:
    with pytest.raises(AuditInputError, match=match):
        audit_dataframe(pd.DataFrame(data), ColumnMapping.model_validate(mapping))


@pytest.mark.parametrize(
    ("mapping", "match"),
    [
        ({"label": "y", "sensitive": ["g"]}, "prediction column, a score column"),
        ({"label": "y", "prediction": "p", "sensitive": ["g", "g"]}, "distinct"),
        ({"label": "y", "prediction": "p", "sensitive": ["y"]}, "both as outcome"),
        ({"label": "y", "prediction": "p", "sensitive": []}, "at least 1"),
    ],
)
def test_column_mapping_validation(mapping: dict[str, object], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        ColumnMapping.model_validate(mapping)
