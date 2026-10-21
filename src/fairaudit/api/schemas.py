"""Request/response models of the HTTP API (the audit report itself is ``AuditReport``)."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from fairaudit.audit import ColumnMapping
from fairaudit.report import AuditPolicy


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    case_study_available: bool = Field(description="Whether case-study artifacts were loaded.")


class AuditRequest(BaseModel):
    """Audit rows sent as JSON records (one object per sample)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "records": [
                        {"label": 1, "score": 0.81, "sex": "F"},
                        {"label": 0, "score": 0.35, "sex": "F"},
                        {"label": 1, "score": 0.62, "sex": "M"},
                        {"label": 0, "score": 0.12, "sex": "M"},
                    ],
                    "mapping": {"label": "label", "score": "score", "sensitive": ["sex"]},
                    "threshold": 0.5,
                }
            ]
        }
    )

    records: list[dict[str, Any]] = Field(min_length=1)
    mapping: ColumnMapping
    threshold: float = Field(default=0.5, ge=0, le=1, description="Applied to the score column.")
    policy: AuditPolicy = Field(default_factory=AuditPolicy)


class ErrorResponse(BaseModel):
    detail: str
