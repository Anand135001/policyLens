
import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_settings_read_environment_variables(monkeypatch):
    monkeypatch.setenv("POLICYLENS_DEBUG", "true")
    monkeypatch.setenv("POLICYLENS_ENVIRONMENT", "testing")

    settings = Settings(_env_file=None)

    assert settings.debug is True
    assert settings.environment == "testing"


def test_settings_reject_invalid_environment(monkeypatch):
    monkeypatch.setenv("POLICYLENS_ENVIRONMENT", "invalid")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
