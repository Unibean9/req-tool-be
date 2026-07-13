"""Every illegal session-phase transition that orchestrator_node auto-adopts emits a dedicated
structured log/metric carrying the previous and adopted phase, without changing the
adopt-derived-phase behavior itself."""

from unittest.mock import patch

import pytest

from app.graphs.nodes import orchestrator_node


def _illegal_transition_state():
    return {
        "session_phase": "intent",
        "user_confirmed": "2026-07-12T00:00:00Z",
        "draft_body": "a fully drafted section body",
        "critique_rounds": 1,
        "quality_report": {"quality_gate_result": "pass"},
    }


def _illegal_metric_calls(mock_log):
    return [call for call in mock_log.call_args_list if call.args[:1] == ("session_phase_illegal_transition",)]


@pytest.mark.asyncio
async def test_illegal_transition_emits_structured_metric_with_old_and_new_phase():
    with patch("app.graphs.gate_logging.log_gate_decision") as mock_log:
        update = await orchestrator_node(_illegal_transition_state(), {})

    calls = _illegal_metric_calls(mock_log)
    assert len(calls) == 1
    kwargs = calls[0].kwargs
    assert kwargs["extra"]["previous_phase"] == "intent"
    assert kwargs["extra"]["adopted_phase"] == update["session_phase"]


@pytest.mark.asyncio
async def test_illegal_transition_still_adopts_derived_phase():
    update = await orchestrator_node(_illegal_transition_state(), {})

    assert update["session_phase"] == "review"


@pytest.mark.asyncio
async def test_legal_transition_does_not_emit_illegal_metric():
    state = {
        "session_phase": None,
        "user_confirmed": None,
    }

    with patch("app.graphs.gate_logging.log_gate_decision") as mock_log:
        await orchestrator_node(state, {})

    assert _illegal_metric_calls(mock_log) == []
