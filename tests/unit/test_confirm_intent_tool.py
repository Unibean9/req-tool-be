"""D6 — confirm_intent tool, schema registration, and self-validation.

The current design supersedes the quick-wins "broad menu, tools self-reject" design here: the
per-phase menu now hides out-of-phase tools (INTENT hides
write_draft; ELICIT hides confirm_intent), and `_gate_selected_tools` DROPS an out-of-phase
selection rather than dispatching it for a tool-level error. The self-correction channel is
preserved via the feedback block (the phase is named back to the model); tools still self-reject
for CONDITION gates (missing arg, unmet critique) when they are in phase.
"""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.graphs.agent_tools import (
    _confirm_intent_impl,
    _write_draft_impl,
    get_available_tools,
)
from app.graphs.nodes import (
    _INTERRUPT_BEARING_TOOLS,
    _build_tool_schemas,
    _gate_selected_tools,
    required_args,
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

def test_confirm_intent_arg_and_required_keys():
    state = {"messages": [], "user_confirmed": None}
    tool = next(t for t in get_available_tools(state) if t.name == "confirm_intent")
    parameters = _build_tool_schemas([tool])[0]["parameters"]
    assert list(parameters["properties"].keys()) == ["summary"]
    assert required_args(tool) == ["summary"]


def test_confirm_intent_is_interrupt_bearing():
    assert "confirm_intent" in _INTERRUPT_BEARING_TOOLS


# ---------------------------------------------------------------------------
# Per-phase menu: INTENT hides drafting/quality tools, ELICIT hides confirm_intent
# ---------------------------------------------------------------------------

def test_intent_phase_hides_draft_and_quality_tools():
    # INTENT phase excludes write_draft/run_critique/finalize so the agent cannot draft pre-intent.
    names = _names({"messages": [], "user_confirmed": None})
    assert "write_draft" not in names
    assert "finalize" not in names
    assert "run_critique" not in names


def test_intent_phase_offers_confirm_intent():
    names = _names({"messages": [], "user_confirmed": None})
    assert "confirm_intent" in names


def test_elicit_phase_hides_confirm_intent_but_offers_write_draft():
    # user_confirmed with no evidence yet -> ELICIT phase: confirm_intent is behind us, drafting opens.
    names = _names({"messages": [], "user_confirmed": True})
    assert "confirm_intent" not in names
    assert "write_draft" in names


def test_write_draft_available_without_elicit():
    names = _names(
        {
            "messages": [],
            "user_confirmed": True,
            "session_elicit_count": 0,
        }
    )
    assert "write_draft" in names


def test_note_tools_remain_available_in_intent_phase_after_many_notes():
    messages = [_note_turn(f"c{i}") for i in range(5)]
    names = _names({"messages": messages, "user_confirmed": None})
    assert "note" in names
    assert "confirm_intent" in names


# ---------------------------------------------------------------------------
# confirm_intent behavior
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_confirm_intent_sets_user_confirmed():
    state = {"messages": [], "user_confirmed": None}
    config = {"configurable": {"thread_id": "00000000-0000-0000-0000-000000000001"}}

    with patch(
        "app.graphs.interrupts._save_and_interrupt_ask", new_callable=AsyncMock
    ) as mock_save:
        mock_save.return_value = "ok"
        command = await _confirm_intent_impl("Building Y for audience A", state, config, "tc-001")

    assert command.update["user_confirmed"] is True
    # stream_response keeps the session ACTIVE (D4); kind=assessment surfaces a summary, not a question.
    assert mock_save.call_args.kwargs["interrupt_kind"] == "stream_response"
    assert mock_save.call_args.kwargs["kind"] == "assessment"


def test_empty_summary_passes_gate_for_tool_feedback():
    # Gate does not change tools; confirm_intent self-rejects with a ToolMessage error.
    state = {"messages": [], "user_confirmed": None}
    gated = _gate_selected_tools(state, [{"name": "confirm_intent", "args": {"summary": ""}}])
    assert gated[0]["name"] == "confirm_intent"


@pytest.mark.asyncio
async def test_confirm_intent_empty_summary_returns_tool_error():
    state = {"messages": [], "user_confirmed": None}
    config = {"configurable": {"thread_id": "00000000-0000-0000-0000-000000000001"}}

    command = await _confirm_intent_impl("", state, config, "tc-empty")

    assert command.update["tool_errors"][0]["code"] == "missing_required_arg"
    msg = command.update["messages"][0]
    assert msg.status == "error"
    assert "summary" in msg.content


def test_confirm_intent_keeps_note_alongside():
    # Interrupt-bearing, but a side-effect-free note rides along so its facts persist this turn;
    # solo enforcement only drops OTHER interrupt-bearing / non-note tools.
    state = {"messages": [], "user_confirmed": None}
    gated = _gate_selected_tools(
        state,
        [
            {"name": "explore_note", "args": {"content": "x"}},
            {"name": "confirm_intent", "args": {"summary": "Build Y for A"}},
        ],
    )
    assert [g["name"] for g in gated] == ["explore_note", "confirm_intent"]


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


def test_write_draft_in_intent_phase_is_dropped_by_gate():
    # write_draft is out of phase in INTENT, so the gate drops it (model is told via feedback).
    state = {"messages": [], "user_confirmed": None}
    gated = _gate_selected_tools(state, [{"name": "write_draft", "args": {"title": "Vision doc", "body": "## Vision\nBuild X for Y."}}])
    assert gated == []


def test_write_draft_in_artifact_phase_not_coerced():
    # Solo gate does not change tools; availability/depth gate is handled by write_draft returning a ToolMessage error.
    state = {"messages": [], "user_confirmed": True}
    gated = _gate_selected_tools(state, [{"name": "write_draft", "args": {"title": "T", "body": "B"}}])
    assert gated[0]["name"] == "write_draft"


def test_write_draft_unavailable_before_confirm_intent():
    # Drafting is not offered until intent is confirmed.
    assert "write_draft" not in _names({"messages": [], "user_confirmed": None})


@pytest.mark.asyncio
async def test_write_draft_without_body_returns_tool_error():
    # In phase (confirmed), empty body is a tool-level error and is no longer coerced to ask_user.
    state = {"messages": [], "user_confirmed": True}
    gated = _gate_selected_tools(state, [{"name": "write_draft", "args": {}}])
    assert gated[0]["name"] == "write_draft"

    command = await _write_draft_impl("", "", state, {"configurable": {}}, "tc-empty-body")
    assert command.update["tool_errors"][0]["code"] == "missing_required_arg"
    assert "body" in command.update["messages"][0].content


@pytest.mark.asyncio
async def test_confirm_intent_impl_calls_audit():
    state = {"messages": [], "user_confirmed": None, "last_agent_run_id": "00000000-0000-0000-0000-000000000004"}
    config = {"configurable": {"thread_id": "00000000-0000-0000-0000-000000000001"}}

    with (
        patch("app.graphs.interrupts._save_and_interrupt_ask", new_callable=AsyncMock) as mock_save,
        patch("app.graphs.agent_tools._audit_interaction_tool_call", new_callable=AsyncMock) as mock_audit,
    ):
        mock_save.return_value = "ok"
        await _confirm_intent_impl("Build Y for A", state, config, "tc-007")

    mock_audit.assert_awaited_once()
    call_kwargs = mock_audit.call_args
    assert call_kwargs.kwargs["tool_name"] == "confirm_intent:tc-007"
    assert call_kwargs.kwargs["message"] == "Build Y for A"
