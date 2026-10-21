import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from fairaudit import __version__
from fairaudit.casestudy.config import StudyConfig
from fairaudit.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


@pytest.fixture
def predictions_csv(tmp_path: Path) -> Path:
    path = tmp_path / "preds.csv"
    pd.DataFrame(
        {
            "y": [1, 0, 1, 0] * 20,
            "p": [1, 0, 0, 0] * 20,
            "s": [0.8, 0.3, 0.4, 0.1] * 20,
            "sex": ["F", "F", "M", "M"] * 20,
        }
    ).to_csv(path, index=False)
    return path


def test_audit_command_writes_reports(predictions_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    args = ["audit", str(predictions_csv), "--label", "y", "--prediction", "p", "--score", "s"]
    result = runner.invoke(app, [*args, "-s", "sex", "--output-dir", str(out)])
    assert result.exit_code == 0, result.output
    report = json.loads((out / "report.json").read_text())
    assert report["attributes"][0]["attribute"] == "sex"
    assert (out / "report.md").read_text().startswith("# Fairness audit: preds.csv")
    assert "Disparate impact" in result.stdout

    gated = runner.invoke(app, [*args, "-s", "sex", "--output-dir", str(out), "--fail-on-findings"])
    assert gated.exit_code == 1


def test_audit_command_rejects_bad_columns(predictions_csv: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["audit", str(predictions_csv), "--label", "nope", "--prediction", "p", "-s", "sex"]
    )
    assert result.exit_code == 2
    assert "Columns not found" in result.output


def test_train_command(small_config: StudyConfig, tmp_path: Path) -> None:
    config_path = tmp_path / "study.yaml"
    config_path.write_text(f"data_path: {small_config.data_path}\n")
    out = tmp_path / "study"
    result = runner.invoke(app, ["train", "--config", str(config_path), "--output-dir", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "results.json").exists()
    assert "reweighing" in result.stdout
