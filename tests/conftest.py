from pathlib import Path

import pandas as pd
import pytest

from fairaudit.casestudy.config import StudyConfig

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "JEMLI_ZEINEB.csv"


@pytest.fixture(scope="session")
def small_metadata_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """First 8,000 rows of the real metadata plus one implausible-age row."""
    df = pd.read_csv(DATA_FILE, nrows=8_000)
    bad = df.iloc[[0]].assign(**{"Patient Age": 412, "Patient ID": 999_999})
    path = tmp_path_factory.mktemp("data") / "metadata_small.csv"
    pd.concat([df, bad]).to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def small_config(small_metadata_csv: Path) -> StudyConfig:
    return StudyConfig(data_path=small_metadata_csv)


@pytest.fixture(scope="session")
def study_dir(small_config: StudyConfig, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Case-study outputs (results.json, report.md…) generated on the small dataset."""
    from fairaudit.casestudy.study import run_study, write_outputs

    directory = tmp_path_factory.mktemp("artifacts") / "chestxray"
    write_outputs(run_study(small_config), directory)
    return directory
