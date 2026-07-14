"""Contract cấu hình additive cho migration control-plane của agent."""

import pytest
from pydantic import ValidationError

from app.config import Settings


@pytest.mark.golden
def test_agent_control_plane_flags_default_to_legacy(monkeypatch):
    for name in (
        "AGENT_TURN_ADMISSION_ENABLED",
        "AGENT_POLICY_RESOLVER_MODE",
        "AGENT_COMMAND_HANDLERS_ENABLED",
        "AGENT_EXECUTION_MODE",
        "AGENT_CHECKPOINT_HISTORY_ENABLED",
        "AGENT_TRACE_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)

    assert settings.agent_turn_admission_enabled is False
    assert settings.agent_policy_resolver_mode == "legacy"
    assert settings.agent_command_handlers_enabled is False
    assert settings.agent_execution_mode == "inline"
    assert settings.agent_checkpoint_history_enabled is False
    assert settings.agent_trace_enabled is False


@pytest.mark.golden
def test_agent_control_plane_flags_accept_valid_environment_overrides(monkeypatch):
    monkeypatch.setenv("AGENT_TURN_ADMISSION_ENABLED", "true")
    monkeypatch.setenv("AGENT_POLICY_RESOLVER_MODE", "shadow")
    monkeypatch.setenv("AGENT_COMMAND_HANDLERS_ENABLED", "true")
    monkeypatch.setenv("AGENT_EXECUTION_MODE", "durable")
    monkeypatch.setenv("AGENT_CHECKPOINT_HISTORY_ENABLED", "true")
    monkeypatch.setenv("AGENT_TRACE_ENABLED", "true")

    settings = Settings(_env_file=None)

    assert settings.agent_turn_admission_enabled is True
    assert settings.agent_policy_resolver_mode == "shadow"
    assert settings.agent_command_handlers_enabled is True
    assert settings.agent_execution_mode == "durable"
    assert settings.agent_checkpoint_history_enabled is True
    assert settings.agent_trace_enabled is True


@pytest.mark.golden
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("AGENT_POLICY_RESOLVER_MODE", "automatic"),
        ("AGENT_EXECUTION_MODE", "background"),
    ],
)
def test_agent_control_plane_modes_reject_invalid_environment_values(monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
