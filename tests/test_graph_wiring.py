"""Graph wiring (edge-loop). Tests written before editing graph.py."""

from unittest.mock import AsyncMock

import pytest
from langgraph.graph import END, StateGraph

from app.graphs.critic import CRITIC_SCHEMA, quality_gate_node, route_after_gate
from app.graphs.graph import build_graph
from app.graphs.nodes import route_after_confirm, route_node
from app.graphs.state import WorkflowState


def _edge_pairs():
    g = build_graph(checkpointer=None).get_graph()
    return {(e.source, e.target) for e in g.edges}


def _node_names():
    return set(build_graph(checkpointer=None).get_graph().nodes)


# --- Group 1: Graph structure ---

def test_graph_has_quality_gate_node():
    assert "quality_gate" in _node_names()


def test_graph_has_summarize_node():
    assert "summarize" in _node_names()


def test_quality_gate_has_loopback_edge():
    pairs = _edge_pairs()
    assert ("quality_gate", "quality_gate") in pairs
    assert ("quality_gate", "propose_artifacts") in pairs


def test_ask_human_can_route_through_summarize_before_analyze():
    pairs = _edge_pairs()
    assert ("ask_human", "summarize") in pairs
    assert ("summarize", "analyze") in pairs


# --- Phase 2: ToolNode parallel scaffolding ---

def test_tools_node_in_graph():
    assert "tools" in _node_names()


def test_route_node_mapping_has_tools_key():
    # Structural check only: the "tools" branch is registered on analyze's conditional
    # edges. It stays unreachable until Phase 5 removes the next_action enum, so this
    # asserts the mapping wiring, not a live call-through.
    pairs = _edge_pairs()
    assert ("analyze", "tools") in pairs


def test_tools_loops_back_to_analyze():
    assert ("tools", "analyze") in _edge_pairs()


def test_route_node_never_returns_tools_for_enum_actions():
    # Safety net for the scaffolding: an empty ToolNode raises on dispatch, so the unreachable
    # "tools" branch must stay unreachable for every schema-conforming next_action. If a future
    # change lets route_node return "tools" before the node carries real tools, this fails here
    # rather than as an opaque ValueError at runtime dispatch.
    for action in ("ask", "propose", "done"):
        state = {"turn_count": 0, "analysis_result": {"next_action": action}}
        assert route_node(state) != "tools"


# --- Group 2: Routing through the compiled graph (finding #2) ---

def test_confirm_propose_routes_to_quality_gate():
    # route_after_confirm still returns the string "propose_artifacts" (function unchanged)
    assert route_after_confirm({"user_confirmed": True}) == "propose_artifacts"
    # but that label must dispatch to the quality_gate node, NOT propose_artifacts
    pairs = _edge_pairs()
    assert ("confirm", "quality_gate") in pairs
    assert ("confirm", "propose_artifacts") not in pairs


@pytest.mark.asyncio
async def test_gate_loops_then_forwards():
    """Critic always below threshold -> quality_gate runs exactly max_critique_rounds times then advances."""
    llm = AsyncMock()
    llm.generate = AsyncMock(
        return_value=({"scores": {}, "overall": 0.3, "rationale": "x", "suggestions": ["sửa"]}, None)
    )

    # Minimal graph using the real node + route, with a terminal stub for propose_artifacts
    async def _terminal(state, config):
        return {}

    builder = StateGraph(WorkflowState)
    builder.add_node("quality_gate", quality_gate_node)
    builder.add_node("propose_artifacts", _terminal)
    builder.set_entry_point("quality_gate")
    builder.add_conditional_edges(
        "quality_gate",
        route_after_gate,
        {"quality_gate": "quality_gate", "propose_artifacts": "propose_artifacts"},
    )
    builder.add_edge("propose_artifacts", END)
    graph = builder.compile()

    state = {
        "artifact_type": "story",
        "messages": [],
        "analysis_result": {
            "next_action": "propose",
            "proposals": [{"artifact_type": "story", "title": "T", "body": "Given A, when B, then C"}],
        },
        "critique_rounds": 0,
    }
    await graph.ainvoke(state, {"configurable": {"llm_client": llm}})

    critic_calls = [c for c in llm.generate.call_args_list if c.kwargs.get("response_format") is CRITIC_SCHEMA]
    assert len(critic_calls) == 2  # exactly max_critique_rounds, no infinite loop


# --- Group 3: Confirm routing flow ---

def test_confirm_yes_flows_through_gate():
    assert route_after_confirm({"user_confirmed": True}) == "propose_artifacts"
    assert ("confirm", "quality_gate") in _edge_pairs()


def test_confirm_no_bypasses_gate():
    assert route_after_confirm({"user_confirmed": False}) == "analyze"
    assert ("confirm", "analyze") in _edge_pairs()
