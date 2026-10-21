"""Case-study configuration, loaded from YAML (see ``configs/chestxray.yaml``)."""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fairaudit.report import AuditPolicy

Attribute = Literal["gender", "age_50_70", "age_group", "view_position"]
ALL_ATTRIBUTES: tuple[Attribute, ...] = ("gender", "age_50_70", "age_group", "view_position")


class StudyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_path: Path = Path("data/JEMLI_ZEINEB.csv")
    seed: int = 42
    test_size: float = Field(default=0.2, gt=0, lt=1, description="Share of patients for test.")
    val_size: float = Field(
        default=0.2, gt=0, lt=1, description="Share of patients for threshold tuning."
    )
    max_age: int = Field(default=100, description="Rows with a larger age are data-entry errors.")
    target_attribute: Attribute = Field(
        default="age_50_70", description="Attribute the mitigations try to make fair."
    )
    audit_attributes: list[Attribute] = Field(default_factory=lambda: list(ALL_ATTRIBUTES))
    threshold: float = Field(default=0.5, ge=0, le=1)
    tree_max_depth: int = Field(default=3, ge=1)
    pca_components: int = Field(default=5, ge=1)
    policy: AuditPolicy = Field(default_factory=AuditPolicy)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.test_size + self.val_size >= 1:
            raise ValueError("test_size + val_size must be < 1")
        if self.target_attribute not in self.audit_attributes:
            raise ValueError("target_attribute must be one of audit_attributes")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "StudyConfig":
        with path.open(encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh) or {})
