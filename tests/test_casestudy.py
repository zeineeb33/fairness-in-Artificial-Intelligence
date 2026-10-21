from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from fairaudit.casestudy.config import StudyConfig
from fairaudit.casestudy.data import load_metadata, patient_split
from fairaudit.casestudy.results import (
    MODEL_CARD_FILE,
    PREDICTIONS_FILE,
    REPORT_FILE,
    StudyResults,
    recommended_experiment,
)

EXPERIMENTS = [
    "decision_tree",
    "logreg",
    "reweighing",
    "unawareness",
    "pca",
    "thresholds_dp",
    "thresholds_eo",
]


def test_load_metadata(small_metadata_csv: Path) -> None:
    df, stats = load_metadata(small_metadata_csv)
    assert stats.n_rows_raw == 8_001
    assert stats.n_dropped_implausible_age == 1
    assert stats.n_rows == len(df) == 8_000
    assert df["age"].max() <= 100
    assert set(df["label"].unique()) == {0, 1}
    assert set(df["age_50_70"]) == {"50-70", "other"}
    assert (df.loc[df["age"].between(50, 69), "age_50_70"] == "50-70").all()
    assert (df.loc[df["age"] == 70, "age_group"] == "70+").all()
    assert len(stats.sha256) == 64


def test_load_metadata_rejects_other_csvs(tmp_path: Path) -> None:
    path = tmp_path / "other.csv"
    pd.DataFrame({"a": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing ChestX-ray14 columns"):
        load_metadata(path)


def test_patient_split_is_disjoint_and_reproducible(small_metadata_csv: Path) -> None:
    df, _ = load_metadata(small_metadata_csv)
    splits = patient_split(df, test_size=0.2, val_size=0.2, seed=7)
    patients = {name: set(part["patient_id"]) for name, part in splits.items()}
    assert not patients["train"] & patients["test"]
    assert not patients["train"] & patients["val"]
    assert not patients["val"] & patients["test"]
    assert sum(len(part) for part in splits.values()) == len(df)
    again = patient_split(df, test_size=0.2, val_size=0.2, seed=7)
    assert again["test"].equals(splits["test"])


def test_study_outputs(study_dir: Path) -> None:
    results = StudyResults.load(study_dir)
    assert [e.name for e in results.experiments] == EXPERIMENTS
    target = results.config.target_attribute
    assert len(results.rows_for(target)) == len(EXPERIMENTS)
    assert len(results.comparison) == len(EXPERIMENTS) * len(results.config.audit_attributes)
    assert results.dataset.n_dropped_implausible_age == 1
    for name in (REPORT_FILE, MODEL_CARD_FILE, PREDICTIONS_FILE):
        assert (study_dir / name).stat().st_size > 0
    assert "not a clinical tool" in (study_dir / REPORT_FILE).read_text()
    predictions = pd.read_csv(study_dir / PREDICTIONS_FILE)
    assert len(predictions) == results.splits["test"].n_rows
    assert {"label", "score_logreg", "pred_thresholds_eo"} <= set(predictions.columns)
    thresholds = results.experiment("thresholds_dp").details["thresholds"]
    assert set(thresholds) == {"50-70", "other"}
    with pytest.raises(KeyError):
        results.experiment("missing")


def test_recommended_experiment(study_dir: Path) -> None:
    results = StudyResults.load(study_dir)
    best = recommended_experiment(results)
    if best is not None:
        assert best.report.attribute(results.config.target_attribute).passes_four_fifths_rule


def test_config_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "study.yaml"
    path.write_text("seed: 3\ntarget_attribute: gender\npolicy:\n  min_group_size: 5\n")
    config = StudyConfig.from_yaml(path)
    assert (config.seed, config.target_attribute, config.policy.min_group_size) == (3, "gender", 5)
    repo_config = Path(__file__).resolve().parents[1] / "configs" / "chestxray.yaml"
    assert StudyConfig.from_yaml(repo_config) == StudyConfig()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"test_size": 0.6, "val_size": 0.5}, "must be < 1"),
        ({"target_attribute": "gender", "audit_attributes": ["age_50_70"]}, "one of audit"),
        ({"unknown_key": 1}, "Extra inputs"),
    ],
)
def test_config_validation(overrides: dict[str, object], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        StudyConfig.model_validate(overrides)
