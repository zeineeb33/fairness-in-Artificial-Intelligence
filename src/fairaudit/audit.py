"""Build an :class:`~fairaudit.report.AuditReport` from labels, predictions and attributes."""

import logging
import math
from collections.abc import Mapping
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, Field, model_validator

from fairaudit import __version__
from fairaudit import metrics as m
from fairaudit.calibration import brier_score, calibration_bins, expected_calibration_error
from fairaudit.errors import AuditInputError
from fairaudit.report import (
    AttributeReport,
    AuditPolicy,
    AuditReport,
    CalibrationBinReport,
    GroupMetrics,
    OverallMetrics,
)

logger = logging.getLogger(__name__)

MISSING_GROUP = "(missing)"
_TRUE = frozenset({"1", "true", "yes", "y", "t"})
_FALSE = frozenset({"0", "false", "no", "n", "f"})


def _num(value: float) -> float | None:
    """JSON-friendly float: ``nan`` becomes ``None``; values are rounded to 6 decimals."""
    return None if math.isnan(value) else round(float(value), 6)


def audit(
    y_true: ArrayLike,
    y_pred: ArrayLike | None = None,
    *,
    sensitive: Mapping[str, ArrayLike],
    y_score: ArrayLike | None = None,
    threshold: float = 0.5,
    policy: AuditPolicy | None = None,
) -> AuditReport:
    """Audit binary predictions for group fairness.

    Args:
        y_true: ground-truth labels (0/1).
        y_pred: binary predictions. If omitted, they are derived as ``y_score >= threshold``.
        sensitive: mapping of attribute name to the attribute value of every sample.
        y_score: predicted probabilities in ``[0, 1]``; enables AUC and calibration metrics.
        threshold: decision threshold used only when ``y_pred`` is omitted.
        policy: thresholds used to generate findings.
    """
    policy = policy or AuditPolicy()
    if not sensitive:
        raise AuditInputError("At least one sensitive attribute is required")
    yt = m.as_binary(y_true, "y_true")
    scores = m.as_scores(y_score) if y_score is not None else None
    if y_pred is not None:
        yp, used_threshold = m.as_binary(y_pred, "y_pred"), None
    elif scores is not None:
        if not 0.0 <= threshold <= 1.0:
            raise AuditInputError("threshold must be in [0, 1]")
        yp, used_threshold = (scores >= threshold).astype(np.int8), threshold
    else:
        raise AuditInputError("Provide predictions, scores, or both")

    groups = {name: m.as_groups(values, name) for name, values in sensitive.items()}
    arrays: dict[str, NDArray[np.generic]] = {"y_true": yt, "y_pred": yp, **groups}
    if scores is not None:
        arrays["y_score"] = scores
    n = m.check_lengths(**arrays)
    if n == 0:
        raise AuditInputError("Nothing to audit: inputs are empty")

    counts = m.ConfusionCounts.from_arrays(yt, yp)
    overall = OverallMetrics(
        n=n,
        base_rate=_num(counts.base_rate),
        selection_rate=_num(counts.selection_rate),
        accuracy=_num(counts.accuracy),
        balanced_accuracy=_num(counts.balanced_accuracy),
        tpr=_num(counts.tpr),
        fpr=_num(counts.fpr),
        ppv=_num(counts.ppv),
    )
    if scores is not None:
        overall.roc_auc = _num(m.roc_auc(yt, scores))
        overall.ece = _num(expected_calibration_error(calibration_bins(yt, scores)))
        overall.brier = _num(brier_score(yt, scores))

    warnings = []
    if counts.tp + counts.fn in (0, n):
        warnings.append("Labels contain a single class: TPR/FPR-based metrics are undefined.")
    attributes = [_attribute_report(name, g, yt, yp, scores, policy) for name, g in groups.items()]
    report = AuditReport(
        fairaudit_version=__version__,
        n_samples=n,
        threshold=used_threshold,
        has_scores=scores is not None,
        policy=policy,
        overall=overall,
        attributes=attributes,
        warnings=warnings,
    )
    logger.debug("Audited %d samples over %d attribute(s)", n, len(attributes))
    return report


def _attribute_report(
    name: str,
    g: NDArray[np.str_],
    yt: m.BinaryArray,
    yp: m.BinaryArray,
    scores: NDArray[np.float64] | None,
    policy: AuditPolicy,
) -> AttributeReport:
    values = np.unique(g)
    if len(values) > policy.max_groups:
        raise AuditInputError(
            f"Attribute '{name}' has {len(values)} distinct values (max {policy.max_groups}); "
            "bin continuous attributes before auditing"
        )
    counts: list[m.ConfusionCounts] = []
    group_metrics: list[GroupMetrics] = []
    for value in values:
        mask = g == value
        c = m.ConfusionCounts.from_arrays(yt[mask], yp[mask])
        counts.append(c)
        gm = GroupMetrics(
            group=str(value),
            n=c.n,
            share=round(c.n / len(g), 6),
            base_rate=_num(c.base_rate),
            selection_rate=_num(c.selection_rate),
            tpr=_num(c.tpr),
            fpr=_num(c.fpr),
            fnr=_num(c.fnr),
            ppv=_num(c.ppv),
            accuracy=_num(c.accuracy),
            small_sample=c.n < policy.min_group_size,
        )
        if scores is not None:
            bins = calibration_bins(yt[mask], scores[mask], policy.calibration_bins)
            gm.ece = _num(expected_calibration_error(bins))
            gm.brier = _num(brier_score(yt[mask], scores[mask]))
            gm.calibration = [
                CalibrationBinReport(
                    lower=b.lower,
                    upper=b.upper,
                    count=b.count,
                    mean_score=round(b.mean_score, 6),
                    observed_rate=round(b.observed_rate, 6),
                )
                for b in bins
            ]
        group_metrics.append(gm)

    tpr_gap = m.spread(c.tpr for c in counts)
    fpr_gap = m.spread(c.fpr for c in counts)
    odds_gaps = [gap for gap in (tpr_gap, fpr_gap) if not math.isnan(gap)]
    di_ratio = m.min_max_ratio(c.selection_rate for c in counts)
    report = AttributeReport(
        attribute=name,
        groups=group_metrics,
        demographic_parity_difference=_num(m.spread(c.selection_rate for c in counts)),
        disparate_impact_ratio=_num(di_ratio),
        passes_four_fifths_rule=None
        if math.isnan(di_ratio)
        else bool(di_ratio >= policy.min_disparate_impact),
        equal_opportunity_difference=_num(tpr_gap),
        false_positive_rate_difference=_num(fpr_gap),
        equalized_odds_difference=_num(max(odds_gaps)) if odds_gaps else None,
        predictive_parity_difference=_num(m.spread(c.ppv for c in counts)),
        base_rate_difference=_num(m.spread(c.base_rate for c in counts)),
        calibration_gap=_num(m.spread(gm.ece for gm in group_metrics if gm.ece is not None))
        if scores is not None
        else None,
    )
    report.findings = _findings(report, policy)
    return report


def _extremes(report: AttributeReport, field: str) -> tuple[GroupMetrics, GroupMetrics]:
    defined = [g for g in report.groups if getattr(g, field) is not None]
    return (
        min(defined, key=lambda g: getattr(g, field)),
        max(defined, key=lambda g: getattr(g, field)),
    )


def _findings(r: AttributeReport, policy: AuditPolicy) -> list[str]:
    """Plain-English findings for the metrics that breach the policy."""
    findings = []
    if r.disparate_impact_ratio is not None and not r.passes_four_fifths_rule:
        lo, hi = _extremes(r, "selection_rate")
        findings.append(
            f"Disparate impact: '{lo.group}' is predicted positive {lo.selection_rate:.1%} of the "
            f"time vs {hi.selection_rate:.1%} for '{hi.group}' (ratio "
            f"{r.disparate_impact_ratio:.2f} < {policy.min_disparate_impact:.2f})."
        )
    if (r.equal_opportunity_difference or 0) > policy.max_difference:
        lo, hi = _extremes(r, "tpr")
        findings.append(
            f"Equal opportunity: true-positive rate is {lo.tpr:.1%} for '{lo.group}' vs "
            f"{hi.tpr:.1%} for '{hi.group}'; actual positives in '{lo.group}' are missed "
            "more often."
        )
    if (r.false_positive_rate_difference or 0) > policy.max_difference:
        lo, hi = _extremes(r, "fpr")
        findings.append(
            f"False positives: false-positive rate is {hi.fpr:.1%} for '{hi.group}' vs "
            f"{lo.fpr:.1%} for '{lo.group}'."
        )
    if (r.predictive_parity_difference or 0) > policy.max_difference:
        lo, hi = _extremes(r, "ppv")
        findings.append(
            f"Predictive parity: a positive prediction is correct {hi.ppv:.1%} of the time for "
            f"'{hi.group}' vs {lo.ppv:.1%} for '{lo.group}'."
        )
    dp, base = r.demographic_parity_difference, r.base_rate_difference
    if dp is not None and base is not None and dp - base > policy.amplification_margin:
        findings.append(
            f"Bias amplification: the gap in predicted positive rates ({dp:.1%}) is larger than "
            f"the gap in actual positive rates ({base:.1%})."
        )
    if r.calibration_gap is not None and r.calibration_gap > policy.max_calibration_gap:
        lo, hi = _extremes(r, "ece")
        findings.append(
            f"Calibration: expected calibration error is {hi.ece:.3f} for '{hi.group}' vs "
            f"{lo.ece:.3f} for '{lo.group}'; the same score means different risks across groups."
        )
    small = [g.group for g in r.groups if g.small_sample]
    if small:
        findings.append(
            f"Small groups (< {policy.min_group_size} samples): {', '.join(small)}; "
            "their metrics are unreliable."
        )
    return findings


class ColumnMapping(BaseModel):
    """Which columns of a tabular dataset hold labels, predictions and sensitive attributes."""

    label: str = Field(description="Ground-truth outcome column.")
    prediction: str | None = Field(default=None, description="Binary prediction column.")
    score: str | None = Field(default=None, description="Predicted probability column, in [0, 1].")
    sensitive: list[str] = Field(min_length=1, description="Sensitive-attribute columns.")
    positive_label: str | None = Field(
        default=None,
        description="Value denoting the positive class in non-binary label/prediction columns.",
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.prediction is None and self.score is None:
            raise ValueError("map a prediction column, a score column, or both")
        if len(set(self.sensitive)) != len(self.sensitive):
            raise ValueError("sensitive columns must be distinct")
        outcome = {c for c in (self.label, self.prediction, self.score) if c is not None}
        if overlap := outcome & set(self.sensitive):
            raise ValueError(f"columns used both as outcome and sensitive: {sorted(overlap)}")
        return self

    @property
    def columns(self) -> list[str]:
        outcome = [c for c in (self.label, self.prediction, self.score) if c is not None]
        return [*outcome, *self.sensitive]


def to_binary(series: pd.Series, name: str, positive_label: str | None = None) -> m.BinaryArray:
    """Coerce a column to 0/1.

    ``positive_label`` wins when it occurs in the column; otherwise numeric 0/1, booleans and
    true/false-like strings are accepted.
    """
    if n_missing := int(series.isna().sum()):
        raise AuditInputError(f"Column '{name}' has {n_missing} missing value(s)")
    text = series.astype(str).str.strip()
    if positive_label is not None and (is_positive := text == positive_label).any():
        return is_positive.to_numpy().astype(np.int8)
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy().astype(np.int8)
    if pd.api.types.is_numeric_dtype(series) and series.isin([0, 1]).all():
        return series.to_numpy().astype(np.int8)
    lowered = text.str.lower()
    if lowered.isin(_TRUE | _FALSE).all():
        return lowered.isin(_TRUE).to_numpy().astype(np.int8)
    examples = sorted(text.unique().tolist())[:5]
    hint = (
        f"positive_label '{positive_label}' does not occur in it"
        if positive_label is not None
        else "set positive_label to choose the positive class"
    )
    raise AuditInputError(
        f"Column '{name}' must be binary (0/1 or true/false); found values {examples}; {hint}."
    )


def to_scores(series: pd.Series, name: str) -> NDArray[np.float64]:
    values = pd.to_numeric(series, errors="coerce")
    if n_bad := int(values.isna().sum()):
        raise AuditInputError(f"Column '{name}' has {n_bad} missing or non-numeric value(s)")
    return m.as_scores(values.to_numpy(dtype=np.float64), name=f"Column '{name}'")


def to_groups(series: pd.Series) -> NDArray[np.str_]:
    return series.astype("string").str.strip().fillna(MISSING_GROUP).to_numpy(dtype=str)


def audit_dataframe(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    *,
    threshold: float = 0.5,
    policy: AuditPolicy | None = None,
) -> AuditReport:
    """Audit a DataFrame according to a :class:`ColumnMapping`."""
    if missing := [c for c in mapping.columns if c not in df.columns]:
        raise AuditInputError(f"Columns not found: {missing}; available: {list(df.columns)}")
    sensitive = {c: to_groups(df[c]) for c in mapping.sensitive}
    report = audit(
        to_binary(df[mapping.label], mapping.label, mapping.positive_label),
        to_binary(df[mapping.prediction], mapping.prediction, mapping.positive_label)
        if mapping.prediction
        else None,
        sensitive=sensitive,
        y_score=to_scores(df[mapping.score], mapping.score) if mapping.score else None,
        threshold=threshold,
        policy=policy,
    )
    for column, groups in sensitive.items():
        if n_missing := int((groups == MISSING_GROUP).sum()):
            report.warnings.append(
                f"'{column}' has {n_missing} missing value(s), audited as group '{MISSING_GROUP}'."
            )
    return report
