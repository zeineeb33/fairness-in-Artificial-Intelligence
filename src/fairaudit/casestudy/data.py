"""Load and prepare the NIH ChestX-ray14 metadata."""

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)

RAW_COLUMNS = {
    "Finding Labels": "finding_labels",
    "Follow-up #": "follow_up",
    "Patient ID": "patient_id",
    "Patient Age": "age",
    "Patient Gender": "gender",
    "View Position": "view_position",
    "OriginalImage[Width": "image_width",
    "Height]": "image_height",
    "OriginalImagePixelSpacing[x": "pixel_spacing_x",
    "y]": "pixel_spacing_y",
}
NUMERIC_FEATURES = (
    "follow_up",
    "age",
    "image_width",
    "image_height",
    "pixel_spacing_x",
    "pixel_spacing_y",
)
CATEGORICAL_FEATURES = ("gender", "view_position", "age_group")
FEATURES = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)

# Same age bins as the original notebook (left-closed intervals).
AGE_BINS = (0, 30, 50, 70, np.inf)
AGE_LABELS = ("0-30", "30-50", "50-70", "70+")

# Feature columns that directly encode each sensitive attribute ("unawareness" drops them).
ATTRIBUTE_FEATURES: dict[str, tuple[str, ...]] = {
    "gender": ("gender",),
    "age_50_70": ("age", "age_group"),
    "age_group": ("age", "age_group"),
    "view_position": ("view_position",),
}


class CleaningStats(BaseModel):
    source: str
    sha256: str
    n_rows_raw: int
    n_rows: int
    n_dropped_implausible_age: int
    n_patients: int
    positive_rate: float


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(path: Path, max_age: int = 100) -> tuple[pd.DataFrame, CleaningStats]:
    """Read the ChestX-ray14 metadata CSV and derive the label and sensitive attributes.

    * ``label`` is 1 for any pathology and 0 for "No Finding" (as in the notebook);
    * ages above ``max_age`` (known data-entry errors, e.g. 412) are dropped;
    * ``age_group`` uses the notebook's bins and ``age_50_70`` contrasts 50-70 with the rest.
    """
    raw = pd.read_csv(path)
    if missing := sorted(set(RAW_COLUMNS) - set(raw.columns)):
        raise ValueError(f"{path} is missing ChestX-ray14 columns: {missing}")
    df = raw[list(RAW_COLUMNS)].rename(columns=RAW_COLUMNS)
    implausible = df["age"] > max_age
    if n_dropped := int(implausible.sum()):
        logger.info("Dropping %d row(s) with age > %d", n_dropped, max_age)
    df = df.loc[~implausible].reset_index(drop=True)

    df["label"] = (df.pop("finding_labels") != "No Finding").astype(np.int8)
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
    df["age_50_70"] = np.where(df["age_group"] == "50-70", "50-70", "other")
    stats = CleaningStats(
        source=path.name,
        sha256=file_sha256(path),
        n_rows_raw=len(raw),
        n_rows=len(df),
        n_dropped_implausible_age=n_dropped,
        n_patients=int(df["patient_id"].nunique()),
        positive_rate=round(float(df["label"].mean()), 6),
    )
    return df, stats


def patient_split(
    df: pd.DataFrame, test_size: float, val_size: float, seed: int
) -> dict[str, pd.DataFrame]:
    """Split by patient so that no patient's images appear in two splits."""
    rng = np.random.default_rng(seed)
    patients = rng.permutation(df["patient_id"].unique())
    n_test = round(len(patients) * test_size)
    n_val = round(len(patients) * val_size)
    assignment = pd.Series("train", index=patients)
    assignment.iloc[:n_test] = "test"
    assignment.iloc[n_test : n_test + n_val] = "val"
    split = df["patient_id"].map(assignment)
    return {name: df.loc[split == name].reset_index(drop=True) for name in ("train", "val", "test")}
