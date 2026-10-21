"""Runtime settings, read from environment variables (prefix ``FAIRAUDIT_``)."""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service settings. ``PORT`` is honoured directly because PaaS platforms (Render) set it."""

    model_config = SettingsConfigDict(env_prefix="FAIRAUDIT_", env_file=".env", extra="ignore")

    artifacts_dir: Path = Field(
        default=Path("artifacts"), description="Directory holding case-study outputs."
    )
    host: str = "0.0.0.0"  # bind all interfaces inside the container
    port: int = Field(default=8000, validation_alias=AliasChoices("PORT", "FAIRAUDIT_PORT"))
    log_level: str = "INFO"
    max_upload_mb: float = Field(default=10.0, gt=0, description="Max size of an uploaded CSV.")
    max_rows: int = Field(default=200_000, gt=0, description="Max rows accepted per audit.")

    @property
    def case_study_dir(self) -> Path:
        return self.artifacts_dir / "chestxray"


@lru_cache
def get_settings() -> Settings:
    return Settings()
