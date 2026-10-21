"""Typed audit-report schema, shared by the library, the CLI and the API."""

from pydantic import BaseModel, Field, computed_field

Rate = (
    float | None
)  # ``None`` when the quantity is undefined (e.g. TPR of a group without positives)


class AuditPolicy(BaseModel):
    """Thresholds that turn metric values into human-readable findings."""

    min_disparate_impact: float = Field(
        default=0.8, gt=0, le=1, description="Four-fifths rule: minimum acceptable ratio."
    )
    max_difference: float = Field(
        default=0.1, gt=0, le=1, description="Maximum acceptable gap in a group rate (TPR, FPR…)."
    )
    max_calibration_gap: float = Field(
        default=0.05, gt=0, le=1, description="Maximum acceptable gap in per-group ECE."
    )
    amplification_margin: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Flag when the selection-rate gap exceeds the label base-rate gap by this.",
    )
    min_group_size: int = Field(
        default=30, ge=1, description="Groups smaller than this are flagged as unreliable."
    )
    max_groups: int = Field(
        default=50, ge=2, description="Reject attributes with more groups (bin them first)."
    )
    calibration_bins: int = Field(default=10, ge=2, le=50)


class CalibrationBinReport(BaseModel):
    lower: float
    upper: float
    count: int
    mean_score: float
    observed_rate: float


class GroupMetrics(BaseModel):
    """Metrics for one group of one sensitive attribute."""

    group: str
    n: int
    share: float = Field(description="Fraction of all audited samples in this group.")
    base_rate: Rate = Field(description="Share of actual positives, P(Y=1).")
    selection_rate: Rate = Field(description="Share of predicted positives, P(Ŷ=1).")
    tpr: Rate = Field(description="True-positive rate, P(Ŷ=1 | Y=1).")
    fpr: Rate = Field(description="False-positive rate, P(Ŷ=1 | Y=0).")
    fnr: Rate = Field(description="False-negative (miss) rate, P(Ŷ=0 | Y=1).")
    ppv: Rate = Field(description="Positive predictive value, P(Y=1 | Ŷ=1).")
    accuracy: Rate
    ece: Rate = Field(default=None, description="Expected calibration error (needs scores).")
    brier: Rate = Field(default=None, description="Brier score (needs scores).")
    calibration: list[CalibrationBinReport] = Field(default_factory=list)
    small_sample: bool = False


class AttributeReport(BaseModel):
    """Fairness summary for one sensitive attribute."""

    attribute: str
    groups: list[GroupMetrics]
    demographic_parity_difference: Rate = Field(description="max - min selection rate.")
    disparate_impact_ratio: Rate = Field(description="min / max selection rate.")
    passes_four_fifths_rule: bool | None
    equal_opportunity_difference: Rate = Field(description="max - min TPR.")
    false_positive_rate_difference: Rate = Field(description="max - min FPR.")
    equalized_odds_difference: Rate = Field(description="max(TPR gap, FPR gap).")
    predictive_parity_difference: Rate = Field(description="max - min PPV.")
    base_rate_difference: Rate = Field(description="max - min P(Y=1): disparity in the labels.")
    calibration_gap: Rate = Field(default=None, description="max - min per-group ECE.")
    findings: list[str] = Field(default_factory=list)


class OverallMetrics(BaseModel):
    n: int
    base_rate: Rate
    selection_rate: Rate
    accuracy: Rate
    balanced_accuracy: Rate
    tpr: Rate
    fpr: Rate
    ppv: Rate
    roc_auc: Rate = None
    ece: Rate = None
    brier: Rate = None


class AuditReport(BaseModel):
    """Result of :func:`fairaudit.audit.audit`."""

    fairaudit_version: str
    n_samples: int
    threshold: float | None = Field(
        description="Threshold used to binarise scores; null when predictions were supplied."
    )
    has_scores: bool
    policy: AuditPolicy
    overall: OverallMetrics
    attributes: list[AttributeReport]
    warnings: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def findings(self) -> list[str]:
        """Every attribute's findings, prefixed with the attribute name."""
        return [f"[{a.attribute}] {f}" for a in self.attributes for f in a.findings]

    def attribute(self, name: str) -> AttributeReport:
        for report in self.attributes:
            if report.attribute == name:
                return report
        raise KeyError(name)
