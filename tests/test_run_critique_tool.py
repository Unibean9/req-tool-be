"""Tests for the in-loop run_critique tool and its gating (spec §6.6, §5.5)."""

import pytest

from app.graphs.agent_tools import (
    CRITIQUE_ROUNDS_MAX,
    NOTE_STEP_LIMIT,
    _run_critique_impl,
    get_available_tools,
)
from app.graphs.critique import CRITIQUE_MODES, _invoke_judge
from tests.test_graph_nodes import _state


def _tool_names(state):
    return {t.name for t in get_available_tools(state)}


def test_run_critique_is_available_when_draft_body_loaded():
    state = _state(artifact_type="goal")
    state["draft_body"] = "## Mục tiêu\n- Tăng giữ chân 30%."
    assert "run_critique" in _tool_names(state)


def test_run_critique_not_available_without_draft():
    state = _state(artifact_type="goal")
    assert "run_critique" not in _tool_names(state)


def test_run_critique_gated_after_max_rounds():
    state = _state(artifact_type="goal")
    state["working_draft"] = "draft"
    state["critique_rounds"] = CRITIQUE_ROUNDS_MAX
    assert "run_critique" not in _tool_names(state)


def test_write_draft_always_available_regardless_of_critique_cap():
    state = _state(artifact_type="goal")
    state["working_draft"] = "draft"
    state["critique_rounds"] = CRITIQUE_ROUNDS_MAX
    assert "write_draft" in _tool_names(state)


def test_ask_user_always_available_regardless_of_caps():
    state = _state(artifact_type="goal")
    state["working_draft"] = "draft"
    state["critique_rounds"] = CRITIQUE_ROUNDS_MAX
    state["messages"] = [
        type("M", (), {"tool_calls": [{"name": "critique_note"}]})() for _ in range(NOTE_STEP_LIMIT)
    ]
    assert "ask_user" in _tool_names(state)


@pytest.mark.asyncio
async def test_run_critique_increments_critique_rounds():
    state = _state(artifact_type="goal")
    state["working_draft"] = "draft body"
    state["critique_rounds"] = 1
    config = {"configurable": {"llm_client": None}}

    command = await _run_critique_impl("draft", "completeness", state, config, "call_1")

    assert command.update["critique_rounds"] == 2


@pytest.mark.asyncio
async def test_run_critique_updates_quality_report():
    state = _state(artifact_type="goal")
    state["working_draft"] = "draft body"
    config = {"configurable": {"llm_client": None}}

    command = await _run_critique_impl("draft", "completeness", state, config, "call_1")

    assert command.update["quality_report"] is not None
    assert command.update["quality_report"]["mode"] == "completeness"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", CRITIQUE_MODES)
async def test_run_critique_mode_enum_accepted(mode):
    result = await _invoke_judge("some body", mode, llm_client=None)
    assert result["mode"] == mode


@pytest.mark.asyncio
async def test_run_critique_unknown_mode_defaults_gracefully():
    result = await _invoke_judge("some body", "unknown", llm_client=None)
    assert result["mode"] == "completeness"


@pytest.mark.asyncio
async def test_run_critique_degrades_when_no_llm_client():
    result = await _invoke_judge("body", "clarity", llm_client=None)
    assert set(result) == {"mode", "score", "findings", "suggestions"}
    assert result["score"] == 0.0
    assert result["suggestions"] == ["no_llm_client"]


@pytest.mark.asyncio
async def test_run_critique_invokes_llm_when_present():
    """With an LLM client, the judge returns the normalized report shape from the model output."""
    from unittest.mock import AsyncMock

    client = AsyncMock()
    client.generate = AsyncMock(return_value=(
        {"score": 0.8, "findings": ["thiếu metric"], "suggestions": ["thêm KPI"]}, None
    ))

    result = await _invoke_judge("body", "completeness", llm_client=client)

    assert result["score"] == 0.8
    assert result["findings"] == ["thiếu metric"]
