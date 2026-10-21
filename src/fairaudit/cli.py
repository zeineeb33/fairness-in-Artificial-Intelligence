"""Command-line interface: ``fairaudit --help``."""

import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from pydantic import ValidationError

from fairaudit import __version__
from fairaudit.audit import ColumnMapping, audit_dataframe
from fairaudit.config import get_settings
from fairaudit.errors import AuditInputError
from fairaudit.log import configure_logging

app = typer.Typer(help="Group-fairness audit toolkit.", no_args_is_help=True)
logger = logging.getLogger("fairaudit.cli")


@app.callback()
def main(
    log_level: Annotated[str, typer.Option(help="DEBUG, INFO, WARNING or ERROR.")] = "INFO",
) -> None:
    """Audit binary classifiers for group fairness."""
    configure_logging(log_level)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def audit(
    csv_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="CSV with one row per sample.")
    ],
    label: Annotated[str, typer.Option(help="Ground-truth column.")],
    sensitive: Annotated[
        list[str], typer.Option("--sensitive", "-s", help="Sensitive-attribute column (repeat).")
    ],
    prediction: Annotated[str | None, typer.Option(help="Binary prediction column.")] = None,
    score: Annotated[str | None, typer.Option(help="Predicted-probability column.")] = None,
    positive_label: Annotated[
        str | None, typer.Option(help="Positive-class value for non-binary columns.")
    ] = None,
    threshold: Annotated[float, typer.Option(help="Threshold applied to scores.")] = 0.5,
    output_dir: Annotated[Path, typer.Option(help="Where to write the report.")] = Path(
        "artifacts/audit"
    ),
    fail_on_findings: Annotated[
        bool, typer.Option(help="Exit with code 1 if any finding is raised (CI gate).")
    ] = False,
) -> None:
    """Audit predictions stored in a CSV; writes report.json and report.md."""
    from fairaudit.render import render_audit_markdown

    try:
        mapping = ColumnMapping(
            label=label,
            prediction=prediction,
            score=score,
            sensitive=sensitive,
            positive_label=positive_label,
        )
        report = audit_dataframe(pd.read_csv(csv_path), mapping, threshold=threshold)
    except (AuditInputError, ValidationError) as exc:
        typer.echo(f"Invalid input: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(
        render_audit_markdown(report, title=f"Fairness audit: {csv_path.name}"), encoding="utf-8"
    )
    logger.info("Wrote %s and %s", output_dir / "report.json", output_dir / "report.md")
    for finding in report.findings:
        typer.echo(f"- {finding}")
    if not report.findings:
        typer.echo("No metric breaches the policy.")
    if fail_on_findings and report.findings:
        raise typer.Exit(code=1)


@app.command()
def train(
    config: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Study configuration (YAML).")
    ] = Path("configs/chestxray.yaml"),
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Output directory [default: $FAIRAUDIT_ARTIFACTS_DIR/chestxray]."),
    ] = None,
) -> None:
    """Run the ChestX-ray14 case study: train baseline and mitigated models, audit, report."""
    from fairaudit.casestudy.config import StudyConfig

    try:
        from fairaudit.casestudy.study import run_study, write_outputs
    except ImportError as exc:  # pragma: no cover - depends on the installed extras
        typer.echo("The case study needs scikit-learn: install fairaudit[study].", err=True)
        raise typer.Exit(code=2) from exc

    study_config = StudyConfig.from_yaml(config)
    run = run_study(study_config)
    for path in write_outputs(run, output_dir or get_settings().case_study_dir):
        logger.info("Wrote %s", path)

    target = study_config.target_attribute
    typer.echo(f"\n{'experiment':<16} {'accuracy':>8} {'roc_auc':>8} {'DI ' + target:>14}")
    for row in run.results.rows_for(target):
        typer.echo(
            f"{row.experiment:<16} {row.accuracy or 0:>8.4f} {row.roc_auc or 0:>8.4f} "
            f"{row.disparate_impact_ratio or 0:>14.3f}"
        )
