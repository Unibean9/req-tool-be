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


# ---------------------------------------------------------------------------
# Interaction tool audit tracking (Bug fix: tool call table was write-only)
# ---------------------------------------------------------------------------

def _make_session_factory(mock_db):
    """Build an async-context-manager session factory backed by mock_db."""
    from unittest.mock import AsyncMock, MagicMock

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_db)
    ctx.__aexit__ = AsyncMock(return_value=None)
    factory = MagicMock(return_value=ctx)
    return factory


def _make_mock_db(already_exists: bool):
    """AsyncMock db where execute().scalar() returns a bool (sync, not a coroutine)."""
    from unittest.mock import AsyncMock, MagicMock

    execute_result = MagicMock()
    execute_result.scalar.return_value = already_exists
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=execute_result)
    mock_db.add = MagicMock()  # synchronous
    return mock_db


@pytest.mark.asyncio
async def test_confirm_intent_creates_audit_row():
    from app.graphs.agent_tools import _audit_interaction_tool_call
    from app.models.agent import AgentToolCallStatus

    mock_db = _make_mock_db(already_exists=False)
    state = {"messages": [], "last_agent_run_id": "00000000-0000-0000-0000-000000000002"}
    config = {"configurable": {"session_factory": _make_session_factory(mock_db)}}

    await _audit_interaction_tool_call(state, config, tool_name="confirm_intent:tc-001", message="Intent summary")

    mock_db.add.assert_called_once()
    added = mock_db.add.call_args[0][0]
    assert added.tool_name == "confirm_intent:tc-001"
    assert added.status == AgentToolCallStatus.EXECUTED
    assert added.input_snapshot == {"message": "Intent summary"}
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_audit_skips_without_run_id():
    from app.graphs.agent_tools import _audit_interaction_tool_call

    state = {"messages": []}  # no last_agent_run_id
    config = {"configurable": {}}

    await _audit_interaction_tool_call(state, config, tool_name="ask_user:tc-x", message="hello")


@pytest.mark.asyncio
async def test_audit_idempotent_when_row_exists():
    from app.graphs.agent_tools import _audit_interaction_tool_call

    mock_db = _make_mock_db(already_exists=True)
    state = {"messages": [], "last_agent_run_id": "00000000-0000-0000-0000-000000000003"}
    config = {"configurable": {"session_factory": _make_session_factory(mock_db)}}

    await _audit_interaction_tool_call(state, config, tool_name="respond:tc-002", message="assessment")

    mock_db.add.assert_not_called()
    mock_db.commit.assert_not_awaited()


def test_write_draft_in_intent_phase_coerces_to_confirm_intent():
    # write_draft not available → should redirect to confirm_intent (not ask_user) so the agent
    # surfaces its prepared summary and triggers the proper intent-confirmation flow.
    state = {"messages": [], "user_confirmed": None}
    gated = _gate_selected_tools(state, [{"name": "write_draft", "args": {"title": "Vision doc", "body": "## Vision\nBuild X for Y."}}])
    assert len(gated) == 1
    assert gated[0]["name"] == "confirm_intent"
    assert "Vision doc" in gated[0]["args"]["summary"]


def test_write_draft_in_artifact_phase_not_coerced():
    # Once user_confirmed=True, write_draft is available and should pass through unchanged.
    state = {"messages": [], "user_confirmed": True}
    gated = _gate_selected_tools(state, [{"name": "write_draft", "args": {"title": "T", "body": "B"}}])
    assert gated[0]["name"] == "write_draft"


def test_write_draft_without_body_falls_back_to_ask_user():
    # If write_draft has no body/title, confirm_intent summary is empty → fall back to ask_user.
    state = {"messages": [], "user_confirmed": None}
    gated = _gate_selected_tools(state, [{"name": "write_draft", "args": {}}])
    assert gated[0]["name"] == "ask_user"


@pytest.mark.asyncio
async def test_confirm_intent_impl_calls_audit():
    state = {"messages": [], "user_confirmed": None, "last_agent_run_id": "00000000-0000-0000-0000-000000000004"}
    config = {"configurable": {"thread_id": "00000000-0000-0000-0000-000000000001"}}

    with (
        patch("app.graphs.agent_tools.nodes._save_and_interrupt_ask", new_callable=AsyncMock) as mock_save,
        patch("app.graphs.agent_tools._audit_interaction_tool_call", new_callable=AsyncMock) as mock_audit,
    ):
        mock_save.return_value = "ok"
        await _confirm_intent_impl("Build Y for A", state, config, "tc-007")

    mock_audit.assert_awaited_once()
    call_kwargs = mock_audit.call_args
    assert call_kwargs.kwargs["tool_name"] == "confirm_intent:tc-007"
    assert call_kwargs.kwargs["message"] == "Build Y for A"
