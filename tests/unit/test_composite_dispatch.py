"""D1 — Composite Dispatch tests (Phase 4 evaluation step).

Covers: gate precedence, multi-tool_calls, backward-compat negative test.
"""

import pytest
from langchain_core.messages import AIMessage

from app.graphs.nodes import _gate_selected_tools, _INTERRUPT_BEARING_TOOLS
from tests.integration.test_graph_nodes import _config, _make_agent_session, _session_factory, _state
from tests.unit.test_tool_parity import _project


# ---------------------------------------------------------------------------
# _gate_selected_tools unit tests
# ---------------------------------------------------------------------------

def test_gate_passes_non_interrupt_tools_through():
    state = _state()
    requested = [
        {"name": "critique_note", "args": {"content": "note"}},
        {"name": "explore_note", "args": {"content": "note2"}},
    ]
    result = _gate_selected_tools(state, requested)
    assert [r["name"] for r in result] == ["critique_note", "explore_note"]


def test_gate_coerces_unavailable_tool_to_ask_user():
    state = _state()  # finalize not available (no working_draft, no passed critique)
    requested = [{"name": "finalize", "args": {"summary": "done"}}]
    result = _gate_selected_tools(state, requested)
    assert len(result) == 1
    assert result[0]["name"] == "ask_user"


def test_gate_interrupt_bearing_keeps_only_first_interrupt_tool():
    """ask_user paired with explore_note → keep only ask_user (interrupt-bearing drops the rest)."""
    state = _state()
    requested = [
        {"name": "ask_user", "args": {"message": "?"}},
        {"name": "explore_note", "args": {"content": "note"}},
    ]
    result = _gate_selected_tools(state, requested)
    assert len(result) == 1
    assert result[0]["name"] == "ask_user"


def test_gate_coerce_then_interrupt_enforcement():
    """Unavailable finalize → coerced to ask_user → ask_user is interrupt-bearing → drop explore_note."""
    state = _state()  # finalize not available
    requested = [
        {"name": "finalize", "args": {"summary": "done"}},
        {"name": "explore_note", "args": {"content": "note"}},
    ]
    result = _gate_selected_tools(state, requested)
    assert len(result) == 1
    assert result[0]["name"] == "ask_user"


def test_gate_interrupt_tools_set_is_complete():
    for tool in ("ask_user", "respond", "write_draft", "finalize"):
        assert tool in _INTERRUPT_BEARING_TOOLS


# ---------------------------------------------------------------------------
# analyze_node composite dispatch integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_composite_non_interrupt_tools_emit_two_tool_calls(client, db_session):
    """Two non-interrupt tools → AIMessage with 2 tool_calls."""
    from app.graphs.nodes import analyze_node

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    from unittest.mock import AsyncMock
    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=({
        "tools": [
            {"name": "critique_note", "args": {"content": "Note A"}},
            {"name": "explore_note", "args": {"content": "Note B"}},
        ],
        "active_mode": "critique",
    }, None))

    state = _state()
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    out = await analyze_node(state, config)
    tool_calls = out["messages"][-1].tool_calls
    assert len(tool_calls) == 2
    assert tool_calls[0]["name"] == "critique_note"
    assert tool_calls[1]["name"] == "explore_note"
    # IDs are unique per call.
    assert tool_calls[0]["id"] != tool_calls[1]["id"]
    run_id = out["last_agent_run_id"]
    assert tool_calls[0]["id"] == f"{run_id}:0"
    assert tool_calls[1]["id"] == f"{run_id}:1"


@pytest.mark.asyncio
async def test_composite_gate_keeps_interrupt_bearing_drops_non_interrupt(client, db_session):
    """ask_user + explore_note → gate keeps only ask_user."""
    from app.graphs.nodes import analyze_node
    from unittest.mock import AsyncMock

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=({
        "tools": [
            {"name": "ask_user", "args": {"message": "Câu hỏi?"}},
            {"name": "explore_note", "args": {"content": "note"}},
        ],
    }, None))

    state = _state()
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    out = await analyze_node(state, config)
    tool_calls = out["messages"][-1].tool_calls
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "ask_user"
    assert out["analysis_result"]["gated_tool"] == "explore_note"


@pytest.mark.asyncio
async def test_backward_compat_old_format_degrades_gracefully(client, db_session):
    """Negative test: LLM returns old format {"tool": "ask_user"} — dispatcher handles gracefully."""
    from app.graphs.nodes import analyze_node
    from unittest.mock import AsyncMock

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value=({
        "tool": "ask_user",
        "message": "Old format message",
    }, None))

    state = _state()
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    out = await analyze_node(state, config)
    # Must not crash; dispatches ask_user (backward-compat reconstructs the tools list)
    tool_calls = out["messages"][-1].tool_calls
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "ask_user"
