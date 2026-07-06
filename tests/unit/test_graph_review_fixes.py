from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.graphs.analysis.context_loader import TurnContext
from app.graphs.analysis.turn_audit import _tool_call_fingerprint
from app.graphs.decision_graph import impact
from app.graphs.nodes import _analysis_turn_result, route_node
from app.graphs.state import WorkflowState, build_initial_workflow_state


def _base_state(**overrides):
    state = build_initial_workflow_state(artifact_type="brd", workflow_area="analysis", step_key=None)
    state.update(overrides)
    return state


def _ctx(state):
    return TurnContext(
        effective_state=state,
        focus_reset_update={},
        artifacts=[],
        lifecycle_reports=[],
        artifact_history=[],
        coverage={"coverage_complete": False, "section_coverage": {}},
        draft_body=None,
        previous_draft_body=None,
    )


def test_parallel_list_channels_merge_same_superstep():
    def left(_state):
        return {
            "risks": [{"statement": "risk a"}],
            "key_facts": [{"statement": "fact a"}],
            "tool_errors": [{"code": "a"}],
        }

    def right(_state):
        return {
            "risks": [{"statement": "risk b"}],
            "key_facts": [{"statement": "fact b"}],
            "tool_errors": [{"code": "b"}],
        }

    builder = StateGraph(WorkflowState)
    builder.add_node("left", left)
    builder.add_node("right", right)
    builder.add_edge(START, "left")
    builder.add_edge(START, "right")
    builder.add_edge("left", END)
    builder.add_edge("right", END)
    graph = builder.compile()

    result = graph.invoke(_base_state())

    assert {item["statement"] for item in result["risks"]} == {"risk a", "risk b"}
    assert {item["statement"] for item in result["key_facts"]} == {"fact a", "fact b"}
    assert {item["code"] for item in result["tool_errors"]} == {"a", "b"}


def test_note_tool_emits_delta_only_so_reducer_does_not_duplicate():
    """Additive reducers concatenate existing + returned: the note tool must emit only the NEW
    entries, or every pre-existing risk/key_fact would be duplicated on each note write."""
    import asyncio
    import operator

    from app.graphs.agent_tools import _write_note_impl

    prior_risk = {"statement": "old risk", "likelihood": "", "impact": "", "mitigation": "", "owner": "", "status": ""}
    state = _base_state(risks=[prior_risk])

    command = asyncio.run(_write_note_impl("RISK: new risk here", state, "tc-1", "critique_note"))

    emitted = command.update["risks"]
    assert [item["statement"] for item in emitted] == ["new risk here"]
    merged = operator.add(state["risks"], emitted)
    assert [item["statement"] for item in merged] == ["old risk", "new risk here"]


def test_analysis_result_closes_tool_calls_when_turn_cap_would_end():
    state = _base_state(turn_count=settings.max_agent_turns - 1)
    tool_call = {"id": "call-1", "name": "ask_user", "args": {"message": "Need scope?"}}

    result = _analysis_turn_result(
        ctx=_ctx(state),
        analysis_result={"tools": [tool_call]},
        run_id="run-1",
        locale="en",
        next_feedback={},
        dispatched_tools=[{"name": "ask_user", "args": {"message": "Need scope?"}}],
        dispatched_tool_calls=[tool_call],
    )

    assert isinstance(result["messages"][0], AIMessage)
    assert isinstance(result["messages"][1], ToolMessage)
    assert result["messages"][1].tool_call_id == "call-1"
    assert result["messages"][1].status == "error"
    assert route_node({**state, **result}) == END


def test_analysis_result_closes_repeated_tool_calls_before_route_end():
    args = {"content": "same note"}
    fingerprint = _tool_call_fingerprint("explore_note", args)
    state = _base_state(recent_tool_calls=[fingerprint, fingerprint])
    tool_call = {"id": "call-2", "name": "explore_note", "args": args}

    result = _analysis_turn_result(
        ctx=_ctx(state),
        analysis_result={"tools": [tool_call]},
        run_id="run-1",
        locale="en",
        next_feedback={},
        dispatched_tools=[{"name": "explore_note", "args": args}],
        dispatched_tool_calls=[tool_call],
    )

    assert isinstance(result["messages"][0], AIMessage)
    assert isinstance(result["messages"][1], ToolMessage)
    assert "repeated_tool_calls" in result["messages"][1].content
    assert route_node({**state, **result}) == END


def test_analysis_result_does_not_append_empty_ai_message_without_dispatch():
    state = _base_state()

    result = _analysis_turn_result(
        ctx=_ctx(state),
        analysis_result={"tools": []},
        run_id="run-1",
        locale="en",
        next_feedback={},
        dispatched_tools=[],
        dispatched_tool_calls=[],
    )

    assert "messages" not in result


def test_default_impact_selector_uses_text_overlap_not_demo_domain_tokens(decision_graph_factory):
    nodes = decision_graph_factory(
        {
            "id": "S1",
            "kind": "scope",
            "statement": "Cashier enters customer phone number at payment",
            "status": "confirmed",
        },
    )

    result = impact("add delivery channel", nodes, [], llm=None)

    assert result["affected_node_ids"] == []
    assert result["decision_nodes"]["S1"]["status"] == "confirmed"
