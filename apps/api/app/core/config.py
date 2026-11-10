from functools import lru_cache
from typing import Literal
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr


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

    postgres_host: str = Field(
        default="127.0.0.1",
        validation_alias="POSTGRES_HOST",
    )
    postgres_port: int = Field(
        default=5432,
        validation_alias="POSTGRES_PORT",
    )
    postgres_db: str = Field(
        default="policylens",
        validation_alias="POSTGRES_DB",
    )
    postgres_user: str = Field(
        default="policylens",
        validation_alias="POSTGRES_USER",
    )
    postgres_password: SecretStr | None = Field(
        default=None,
        validation_alias="POSTGRES_PASSWORD",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()