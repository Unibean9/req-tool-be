"""Phase 5 — tool-loop shim adapter (Slice A).

The production LLM clients only emit JSON dicts (response_format), never native tool_calls. The
shim lets the tool-loop run on that same JSON discipline: analyze_node asks for a TOOL_SELECTION_SCHEMA
dict and converts it into an AIMessage(tool_calls=[...]) that the ToolNode dispatches.

These tests pin the tool-loop routing (T1) and the dict→AIMessage conversion (T1b).
"""

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END

from app.graphs.nodes import route_node
from tests.test_graph_nodes import _config, _make_agent_session, _session_factory, _state
from tests.test_tool_parity import _project

# ---------------------------------------------------------------------------
# T1 — tool-loop routing behavior
# ---------------------------------------------------------------------------

def test_tool_calls_route_to_tools():
    ai = AIMessage(content="", tool_calls=[{"id": "r1", "name": "ask_user", "args": {"message": "x"}}])
    state = {"turn_count": 0, "messages": [ai], "analysis_result": None}
    assert route_node(state) == "tools"


def test_no_tool_calls_routes_end():
    state = {"turn_count": 0, "messages": [AIMessage(content="hi")], "analysis_result": None}
    assert route_node(state) == END


# ---------------------------------------------------------------------------
# T1b — shim adapter: tool-selection dict → AIMessage(tool_calls)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_selection_converts_to_ai_message_tool_calls(client, db_session):
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
    # analytic fields still persist so eval (active_mode) and coverage do not regress; the legacy
    # 'qa' is normalized to the spec §7.1 'discovery' baseline (phase-06).
    assert out["analysis_result"]["active_mode"] == "discovery"


@pytest.mark.asyncio
async def test_respond_selection_derives_mode_and_dispatches_arg(client, db_session):
    """respond is the user-facing critique/explore surface: the shim clamps active_mode to a
    proactive mode and passes it as the tool's `mode` arg, so a critique cannot silently fall back
    to Q&A (the note-default-'qa' regression)."""
    from app.graphs.nodes import analyze_node
    from tests.scenarios.scripted_llm import ScriptedLLM, tool_select

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    llm = ScriptedLLM(tool_brain=[
        tool_select("respond", message="Giả định 'mỗi tuần' có vẻ rủi ro nhất.", active_mode="critique")
    ])
    state = _state(artifact_type="goal")
    config = _config(str(agent_session.id), str(project_id), llm_client=llm)
    config["configurable"]["session_factory"] = _session_factory()

    out = await analyze_node(state, config)
    call = out["messages"][-1].tool_calls[0]
    assert call["name"] == "respond"
    assert call["args"]["mode"] == "critique"
    assert out["analysis_result"]["active_mode"] == "critique"


@pytest.mark.asyncio
async def test_tool_selection_unavailable_tool_coerced_to_ask(client, db_session):
    """finalize hard-gate: with an empty working_draft finalize is not offered, so a model that
    picks it anyway is coerced to ask_user rather than dispatching an ungated finalize (S4)."""
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
async def test_empty_tool_selection_ends_turn(client, db_session):
    """An empty selection (no tool) is the loop terminal: a plain AIMessage with no tool_calls so
    route_node ends the turn instead of dispatching."""
    from app.graphs.nodes import analyze_node
    from tests.scenarios.scripted_llm import ScriptedLLM

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    llm = ScriptedLLM(tool_brain=[])  # exhausted immediately -> {} -> done
    state = _state(artifact_type="goal")
    config = _config(str(agent_session.id), str(project_id), llm_client=llm)
    config["configurable"]["session_factory"] = _session_factory()

    out = await analyze_node(state, config)
    ai = out["messages"][-1]
    assert isinstance(ai, AIMessage)
    assert not ai.tool_calls
    assert route_node({**state, **out}) == END


@pytest.mark.asyncio
async def test_write_note_selection_dispatches_without_crash(client, db_session):
    """critique_note is offered by get_available_tools, so the compiled ToolNode must carry it or its
    dispatch raises. Exercises a real graph turn: analyze picks the note -> ToolNode -> loops back."""
    from langgraph.checkpoint.memory import MemorySaver

    from app.graphs.graph import build_graph
    from tests.scenarios.scripted_llm import ScriptedLLM, tool_select

    project_id = await _project(client)
    agent_session = await _make_agent_session(client, db_session, project_id)

    # First a note, then an ask so the loop pauses for the human instead of running unbounded.
    llm = ScriptedLLM(tool_brain=[
        tool_select("critique_note", content="Giả định: user là sinh viên.", active_mode="critique"),
        tool_select("ask_user", message="Bạn muốn xây gì?"),
    ])
    graph = build_graph(checkpointer=MemorySaver())
    state = _state(artifact_type="goal")
    state["intent"] = "task"
    config = _config(str(agent_session.id), str(project_id), llm_client=llm)
    config["configurable"]["session_factory"] = _session_factory()

    out = await graph.ainvoke(state, config)
    assert "__interrupt__" in out  # note ran, then ask_user paused — no ToolNode dispatch crash


@pytest.mark.asyncio
async def test_critique_note_populates_assumptions_in_state():
    """A note carrying an ASSUMPTION: tag appends a structured object to state via the Command update."""
    from app.graphs.agent_tools import _write_note_impl

    state = _state(artifact_type="goal")
    command = await _write_note_impl(
        "ASSUMPTION: users have smartphones | confidence: high | status: unconfirmed",
        state,
        tool_call_id="call_1",
    )

    assert command.update["assumptions"]
    assert command.update["assumptions"][0]["statement"] == "users have smartphones"


@pytest.mark.asyncio
async def test_note_appends_to_existing_assumptions():
    """The note merges with prior state assumptions rather than replacing them."""
    from app.graphs.agent_tools import _write_note_impl

    state = _state(artifact_type="goal")
    state["assumptions"] = [{"statement": "prior", "source": "", "confidence": "",
                             "impact": "", "owner": "", "status": ""}]
    command = await _write_note_impl(
        "ASSUMPTION: new one | confidence: low", state, tool_call_id="call_2"
    )

    statements = [a["statement"] for a in command.update["assumptions"]]
    assert statements == ["prior", "new one"]
