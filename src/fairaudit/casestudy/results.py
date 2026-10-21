"""Typed results of the case study (importable without scikit-learn, e.g. by the API)."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from fairaudit.casestudy.config import StudyConfig
from fairaudit.casestudy.data import CleaningStats
from fairaudit.report import AuditReport

RESULTS_FILE = "results.json"
REPORT_FILE = "report.md"
MODEL_CARD_FILE = "model_card.md"
PREDICTIONS_FILE = "test_predictions.csv"
BASELINE = "logreg"  # experiment every mitigation is compared with


class SplitSummary(BaseModel):
    n_rows: int
    n_patients: int
    positive_rate: float


class GroupBaseRate(BaseModel):
    group: str
    n: int
    share: float
    base_rate: float


class ExperimentResult(BaseModel):
    name: str
    title: str
    description: str
    stage: str = Field(description="none, pre-processing, in-processing or post-processing.")
    report: AuditReport
    details: dict[str, Any] = Field(default_factory=dict)


class ComparisonRow(BaseModel):
    experiment: str
    attribute: str
    accuracy: float | None
    balanced_accuracy: float | None
    roc_auc: float | None
    disparate_impact_ratio: float | None
    demographic_parity_difference: float | None
    equal_opportunity_difference: float | None
    equalized_odds_difference: float | None


class StudyResults(BaseModel):
    fairaudit_version: str
    config: StudyConfig
    dataset: CleaningStats
    splits: dict[str, SplitSummary]
    label_base_rates: dict[str, list[GroupBaseRate]] = Field(
        description="Positive-label rate per group on the full cleaned dataset."
    )
    experiments: list[ExperimentResult]
    comparison: list[ComparisonRow]

    def experiment(self, name: str) -> ExperimentResult:
        for experiment in self.experiments:
            if experiment.name == name:
                return experiment
        raise KeyError(name)

    def rows_for(self, attribute: str) -> list[ComparisonRow]:
        return [row for row in self.comparison if row.attribute == attribute]

    @classmethod
    def load(cls, directory: Path) -> "StudyResults":
        return cls.model_validate_json((directory / RESULTS_FILE).read_text(encoding="utf-8"))


def recommended_experiment(results: StudyResults) -> ExperimentResult | None:
    """Most accurate experiment that passes the four-fifths rule on the target attribute."""
    target = results.config.target_attribute
    passing = [e for e in results.experiments if e.report.attribute(target).passes_four_fifths_rule]
    return max(passing, key=lambda e: e.report.overall.accuracy or 0.0, default=None)
