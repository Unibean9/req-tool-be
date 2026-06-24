"""D6 — confirm_intent tool, schema registration, and the intent phase gate.

The intent phase (user_confirmed is None) restricts the menu to exploration + confirmation tools;
confirm_intent flips user_confirmed=True and unlocks the artifact menu (one-shot, no reset path).
"""

import pytest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage

from app.graphs.agent_tools import _confirm_intent_impl, get_available_tools, NOTE_STEP_LIMIT
from app.graphs.nodes import (
    TOOL_SELECTION_SCHEMA,
    _TOOL_ARG_KEYS,
    _TOOL_REQUIRED_ARGS,
    _INTERRUPT_BEARING_TOOLS,
    _gate_selected_tools,
)


def _names(state):
    return {t.name for t in get_available_tools(state)}


def _note_turn(call_id: str):
    return AIMessage(
        content="", tool_calls=[{"id": call_id, "name": "explore_note", "args": {"content": "x"}}]
    )


# ---------------------------------------------------------------------------
# Schema registration
# ---------------------------------------------------------------------------

def test_confirm_intent_in_tool_selection_schema():
    enum = TOOL_SELECTION_SCHEMA["properties"]["tools"]["items"]["properties"]["name"]["enum"]
    assert "confirm_intent" in enum


def test_confirm_intent_arg_and_required_keys():
    assert _TOOL_ARG_KEYS["confirm_intent"] == ["summary"]
    assert _TOOL_REQUIRED_ARGS["confirm_intent"] == ["summary"]


def test_confirm_intent_is_interrupt_bearing():
    assert "confirm_intent" in _INTERRUPT_BEARING_TOOLS


# ---------------------------------------------------------------------------
# Intent phase gate (user_confirmed is None)
# ---------------------------------------------------------------------------

def test_intent_phase_hides_artifact_tools():
    names = _names({"messages": [], "user_confirmed": None})
    assert "write_draft" not in names
    assert "finalize" not in names
    assert "run_critique" not in names


def test_intent_phase_offers_confirm_intent():
    assert "confirm_intent" in _names({"messages": [], "user_confirmed": None})


def test_artifact_phase_hides_confirm_intent():
    # One-shot: confirm_intent disappears once user_confirmed=True.
    names = _names({"messages": [], "user_confirmed": True})
    assert "confirm_intent" not in names
    assert "write_draft" in names


def test_note_step_limit_applies_in_intent_phase():
    messages = [_note_turn(f"c{i}") for i in range(NOTE_STEP_LIMIT)]
    names = _names({"messages": messages, "user_confirmed": None})
    assert "explore_note" not in names
    assert "critique_note" not in names
    assert "confirm_intent" in names  # confirmation survives the note step-limit


# ---------------------------------------------------------------------------
# confirm_intent behavior
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_confirm_intent_sets_user_confirmed():
    state = {"messages": [], "user_confirmed": None}
    config = {"configurable": {"thread_id": "00000000-0000-0000-0000-000000000001"}}

    with patch(
        "app.graphs.agent_tools.nodes._save_and_interrupt_ask", new_callable=AsyncMock
    ) as mock_save:
        mock_save.return_value = "ok"
        command = await _confirm_intent_impl("Building Y for audience A", state, config, "tc-001")

    assert command.update["user_confirmed"] is True
    # stream_response keeps the session ACTIVE (D4); kind=assessment surfaces a summary, not a question.
    assert mock_save.call_args.kwargs["interrupt_kind"] == "stream_response"
    assert mock_save.call_args.kwargs["kind"] == "assessment"


def test_empty_summary_coerced_to_ask_user():
    # Empty required arg degrades to a re-ask rather than dispatching a blank confirmation.
    state = {"messages": [], "user_confirmed": None}
    gated = _gate_selected_tools(state, [{"name": "confirm_intent", "args": {"summary": ""}}])
    assert gated[0]["name"] == "ask_user"


def test_confirm_intent_solo_enforced_against_note():
    # Interrupt-bearing: paired with a note, only confirm_intent survives the gate.
    state = {"messages": [], "user_confirmed": None}
    gated = _gate_selected_tools(
        state,
        [
            {"name": "explore_note", "args": {"content": "x"}},
            {"name": "confirm_intent", "args": {"summary": "Build Y for A"}},
        ],
    )
    assert [g["name"] for g in gated] == ["confirm_intent"]
