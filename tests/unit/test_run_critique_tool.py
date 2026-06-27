"""Tests for the in-loop run_critique tool and its gating (spec §6.6, §5.5)."""

import pytest

from app.graphs.agent_tools import (
    CRITIQUE_ROUNDS_MAX,
    _run_critique_impl,
    get_available_tools,
)
from app.graphs.critique import CRITIQUE_MODES, _invoke_judge
from app.graphs.decision_graph import create_node, render_view
from tests.integration.test_graph_nodes import _state


def _tool_names(state):
    # Every gating test in this file exercises the artifact phase (post-confirm_intent).
    return {t.name for t in get_available_tools({**state, "user_confirmed": True})}


def _draft_state(statement: str = "Increase retention by 30%.") -> dict:
    state = _state(artifact_type="brd")
    state["decision_nodes"] = {
        "N1": create_node(
            kind="objective",
            statement=statement,
            origin={"source": "test"},
            status="confirmed",
        )
    }
    return state


def _draft_body(state: dict) -> str:
    return render_view(state["decision_nodes"], state["artifact_type"])


def test_run_critique_is_available_when_decision_graph_has_view():
    state = _draft_state()
    assert "run_critique" in _tool_names(state)


def test_run_critique_ignores_focused_artifact_body_without_graph():
    state = _state(artifact_type="brd")
    state["focused_artifact_id"] = "00000000-0000-0000-0000-000000000001"
    state["draft_body"] = "## Goal\n- Increase retention by 30%."
    assert "run_critique" not in _tool_names(state)


def test_run_critique_not_available_without_draft():
    state = _state(artifact_type="goal")
    assert "run_critique" not in _tool_names(state)


def test_run_critique_gated_after_max_rounds():
    state = _draft_state()
    state["critique_rounds"] = CRITIQUE_ROUNDS_MAX
    assert "run_critique" not in _tool_names(state)


@pytest.mark.asyncio
async def test_run_critique_bypass_without_draft_returns_tool_error():
    state = _state(artifact_type="goal")
    config = {"configurable": {"llm_client": None}}

    command = await _run_critique_impl("draft", "completeness", state, config, "call_1")

    assert command.update["tool_errors"][0]["code"] == "tool_not_available"
    assert command.update["messages"][0].status == "error"


@pytest.mark.asyncio
async def test_run_critique_bypass_after_max_rounds_returns_tool_error():
    state = _draft_state()
    state["critique_rounds"] = CRITIQUE_ROUNDS_MAX
    config = {"configurable": {"llm_client": None}}

    command = await _run_critique_impl("draft", "completeness", state, config, "call_1")

    assert command.update["tool_errors"][0]["code"] == "tool_not_available"
    assert "critique round limit" in command.update["messages"][0].content


def test_write_draft_always_available_regardless_of_critique_cap():
    state = _draft_state()
    state["critique_rounds"] = CRITIQUE_ROUNDS_MAX
    assert "write_draft" in _tool_names(state)


def test_ask_user_always_available_regardless_of_caps():
    state = _draft_state()
    state["critique_rounds"] = CRITIQUE_ROUNDS_MAX
    state["messages"] = [
        type("M", (), {"tool_calls": [{"name": "critique_note"}]})() for _ in range(5)
    ]
    assert "ask_user" in _tool_names(state)


@pytest.mark.asyncio
async def test_run_critique_increments_critique_rounds():
    state = _draft_state()
    state["critique_rounds"] = 1
    config = {"configurable": {"llm_client": None}}

    command = await _run_critique_impl("draft", "completeness", state, config, "call_1")

    assert command.update["critique_rounds"] == 2


@pytest.mark.asyncio
async def test_run_critique_updates_quality_report():
    state = _draft_state()
    config = {"configurable": {"llm_client": None}}

    command = await _run_critique_impl("draft", "completeness", state, config, "call_1")

    report = command.update["quality_report"]
    assert report is not None
    assert report["mode"] == "completeness"
    # All five derived fields are present (the reflection feedback contract).
    for key in ("blocking_issues", "non_blocking_warnings", "revision_plan",
                "quality_gate_result", "recommended_next_action"):
        assert key in report


def _scripted_client(score: float, findings: list[str], suggestions: list[str]):
    from unittest.mock import AsyncMock

    client = AsyncMock()
    client.generate = AsyncMock(return_value=(
        {"score": score, "findings": findings, "suggestions": suggestions}, None
    ))
    return client


@pytest.mark.asyncio
async def test_run_critique_pass_gate_classification():
    state = _draft_state()
    config = {"configurable": {"llm_client": _scripted_client(0.9, ["nit nho"], ["mo rong"])}}

    report = (await _run_critique_impl("draft", "completeness", state, config, "c1")).update["quality_report"]

    assert report["quality_gate_result"] == "pass"
    assert report["blocking_issues"] == []
    assert report["non_blocking_warnings"] == ["nit nho"]
    assert report["recommended_next_action"] == "finalize"


@pytest.mark.asyncio
async def test_run_critique_fail_gate_classification():
    state = _draft_state()
    config = {"configurable": {"llm_client": _scripted_client(0.5, ["missing metric"], ["them KPI"])}}

    report = (await _run_critique_impl("draft", "completeness", state, config, "c1")).update["quality_report"]

    assert report["quality_gate_result"] == "fail"
    assert report["blocking_issues"] == ["missing metric"]
    assert report["revision_plan"] == ["them KPI"]
    assert report["recommended_next_action"] == "revise"


@pytest.mark.asyncio
async def test_run_critique_escalates_at_rounds_cap_when_failing():
    state = _draft_state()
    state["critique_rounds"] = CRITIQUE_ROUNDS_MAX - 1  # this critique reaches the cap
    config = {"configurable": {"llm_client": _scripted_client(0.5, ["missing metric"], ["them KPI"])}}

    report = (await _run_critique_impl("draft", "completeness", state, config, "c1")).update["quality_report"]

    assert report["quality_gate_result"] == "fail"
    assert report["recommended_next_action"] == "escalate"


@pytest.mark.asyncio
async def test_run_critique_degraded_path_fails_gate():
    state = _draft_state()
    config = {"configurable": {"llm_client": None}}

    report = (await _run_critique_impl("draft", "completeness", state, config, "c1")).update["quality_report"]

    # No-LLM: score 0.0 → gate fails by design, but findings empty so no blocking_issues listed.
    assert report["quality_gate_result"] == "fail"
    assert report["blocking_issues"] == []


@pytest.mark.asyncio
async def test_run_critique_writes_draft_hash():
    import hashlib

    state = _draft_state()
    config = {"configurable": {"llm_client": None}}

    command = await _run_critique_impl("draft", "completeness", state, config, "c1")

    body = _draft_body(state)
    assert command.update["last_critiqued_draft_hash"] == hashlib.md5(body.encode()).hexdigest()[:8]


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
        {"score": 0.8, "findings": ["missing metric"], "suggestions": ["them KPI"]}, None
    ))

    result = await _invoke_judge("body", "completeness", llm_client=client)

    assert result["score"] == 0.8
    assert result["findings"] == ["missing metric"]


@pytest.mark.asyncio
async def test_run_critique_degrades_when_output_unparseable():
    """A parse failure (e.g. truncated JSON) must degrade, not crash the tool-loop."""
    from unittest.mock import AsyncMock

    client = AsyncMock()
    client.generate = AsyncMock(side_effect=ValueError("Could not parse JSON from LLM response"))

    result = await _invoke_judge("body", "completeness", llm_client=client)

    assert set(result) == {"mode", "score", "findings", "suggestions"}
    assert result["score"] == 0.0
    assert result["suggestions"] == ["judge_unparseable"]
