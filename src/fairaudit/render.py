"""Render audit reports, study reports and model cards as Markdown."""

import math

from jinja2 import Environment, PackageLoader, StrictUndefined

from fairaudit.casestudy.results import BASELINE, StudyResults, recommended_experiment
from fairaudit.report import AuditReport


def _pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _md(text: object) -> str:
    """Escape characters that would break a Markdown table cell."""
    return str(text).replace("|", r"\|").replace("\n", " ")


_env = Environment(
    loader=PackageLoader("fairaudit", "templates"),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
    autoescape=False,  # Markdown output; table cells go through the `md` filter
)
_env.filters.update(pct=_pct, num=_num, md=_md)


def render_audit_markdown(report: AuditReport, title: str = "Fairness audit report") -> str:
    return _env.get_template("audit_report.md.j2").render(report=report, title=title)


def _study_context(results: StudyResults) -> dict[str, object]:
    rows: dict[str, dict[str, object]] = {}
    for row in results.comparison:
        rows.setdefault(row.experiment, {})[row.attribute] = row
    baseline = results.experiment(BASELINE)
    accuracy = baseline.report.overall.accuracy or 0.0
    return {
        "results": results,
        "rows": rows,
        "baseline": baseline,
        "recommended": recommended_experiment(results),
        # Two binomial standard errors of the baseline accuracy on the test set.
        "noise": 2 * math.sqrt(accuracy * (1 - accuracy) / baseline.report.n_samples),
    }


def render_study_markdown(results: StudyResults) -> str:
    return _env.get_template("study_report.md.j2").render(**_study_context(results))


def render_model_card(results: StudyResults) -> str:
    return _env.get_template("model_card.md.j2").render(**_study_context(results))
