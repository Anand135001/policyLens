from functools import lru_cache
from typing import Literal
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[4]

class Settings(BaseSettings):
    app_name: str = "PolicyLens API"
    app_version: str = "0.1.0"
    debug: bool = False

    environment: Literal[
        "development",
        "testing",
        "staging",
        "production",
    ] = "development"

    model_config = SettingsConfigDict(
        env_file= PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="POLICYLENS_",
        extra="ignore",
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()