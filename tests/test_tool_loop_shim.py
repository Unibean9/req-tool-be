"""Phase 5 — tool-loop shim adapter (Slice A).

The production LLM clients only emit JSON dicts (response_format), never native tool_calls. The
shim lets the tool-loop run on that same JSON discipline: analyze_node asks for a TOOL_SELECTION_SCHEMA
dict and converts it into an AIMessage(tool_calls=[...]) that the ToolNode dispatches.

These tests pin the flag-gated routing (T1) and the dict→AIMessage conversion (T1b). The flag
defaults false, so the enum path stays the live default until Slice C flips it.
"""

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END

from app.config import settings
from app.graphs.nodes import route_node
from tests.test_graph_nodes import _config, _make_agent_session, _session_factory, _state
from tests.test_tool_parity import _project


# ---------------------------------------------------------------------------
# T1 — feature flag routing behavior
# ---------------------------------------------------------------------------

def test_tool_loop_only_false_uses_enum_path(monkeypatch):
    monkeypatch.setattr(settings, "tool_loop_only", False)
    state = {"turn_count": 0, "analysis_result": {"next_action": "ask", "message": "x"}}
    assert route_node(state) == "ask_human"


def test_tool_loop_only_true_routes_tool_calls(monkeypatch):
    monkeypatch.setattr(settings, "tool_loop_only", True)
    ai = AIMessage(content="", tool_calls=[{"id": "r1", "name": "ask_user", "args": {"message": "x"}}])
    state = {"turn_count": 0, "messages": [ai], "analysis_result": None}
    assert route_node(state) == "tools"


def test_tool_loop_only_true_no_tool_calls_routes_end(monkeypatch):
    monkeypatch.setattr(settings, "tool_loop_only", True)
    state = {"turn_count": 0, "messages": [AIMessage(content="hi")], "analysis_result": None}
    assert route_node(state) == END


# ---------------------------------------------------------------------------
# T1b — shim adapter: tool-selection dict → AIMessage(tool_calls)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_selection_converts_to_ai_message_tool_calls(monkeypatch, client, db_session):
    monkeypatch.setattr(settings, "tool_loop_only", True)
    from app.graphs.nodes import analyze_node
    from tests.scenarios.scripted_llm import ScriptedLLM, tool_select

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    llm = ScriptedLLM(tool_brain=[tool_select("ask_user", message="Bạn muốn xây gì?", active_mode="qa")])
    state = _state(artifact_type="goal")
    config = _config(str(agent_session.id), str(project_id), llm_client=llm)
    config["configurable"]["session_factory"] = _session_factory()

    out = await analyze_node(state, config)

    ai = out["messages"][-1]
    assert isinstance(ai, AIMessage)
    assert ai.tool_calls[0]["name"] == "ask_user"
    assert ai.tool_calls[0]["args"]["message"] == "Bạn muốn xây gì?"
    # tool_call.id == AgentRun.id so the tool idempotency keys line up on resume.
    assert ai.tool_calls[0]["id"] == out["last_agent_run_id"]
    # analytic fields still persist so eval (active_mode) and coverage do not regress.
    assert out["analysis_result"]["active_mode"] == "qa"


@pytest.mark.asyncio
async def test_tool_selection_unavailable_tool_coerced_to_ask(monkeypatch, client, db_session):
    """finalize hard-gate: with an empty working_draft finalize is not offered, so a model that
    picks it anyway is coerced to ask_user rather than dispatching an ungated finalize (S4)."""
    monkeypatch.setattr(settings, "tool_loop_only", True)
    from app.graphs.nodes import analyze_node
    from tests.scenarios.scripted_llm import ScriptedLLM, tool_select

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    llm = ScriptedLLM(tool_brain=[tool_select("finalize", summary="done")])
    state = _state(artifact_type="goal")  # working_draft is None -> finalize gated out
    config = _config(str(agent_session.id), str(project_id), llm_client=llm)
    config["configurable"]["session_factory"] = _session_factory()

    out = await analyze_node(state, config)
    call = out["messages"][-1].tool_calls[0]
    assert call["name"] == "ask_user"
    # Coercion must not produce a blank question (HIGH): a gated finalize carries no message.
    assert call["args"]["message"].strip()


@pytest.mark.asyncio
async def test_write_note_selection_dispatches_without_crash(monkeypatch, client, db_session):
    """write_note is offered by get_available_tools, so the compiled ToolNode must carry it or its
    dispatch raises. Exercises a real graph turn: analyze picks write_note -> ToolNode -> loops back."""
    monkeypatch.setattr(settings, "tool_loop_only", True)
    from app.graphs.graph import build_graph
    from langgraph.checkpoint.memory import MemorySaver
    from tests.scenarios.scripted_llm import ScriptedLLM, tool_select

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    # First a note, then an ask so the loop pauses for the human instead of running unbounded.
    llm = ScriptedLLM(tool_brain=[
        tool_select("write_note", content="Giả định: user là sinh viên.", active_mode="critique"),
        tool_select("ask_user", message="Bạn muốn xây gì?"),
    ])
    graph = build_graph(checkpointer=MemorySaver())
    state = _state(artifact_type="goal")
    state["intent"] = "task"
    config = _config(str(agent_session.id), str(project_id), llm_client=llm)
    config["configurable"]["session_factory"] = _session_factory()

    out = await graph.ainvoke(state, config)
    assert "__interrupt__" in out  # note ran, then ask_user paused — no ToolNode dispatch crash
