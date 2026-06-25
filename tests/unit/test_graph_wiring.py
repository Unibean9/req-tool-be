"""Graph wiring tests for the pure tool-loop topology (Slice C)."""

from langchain_core.messages import AIMessage

from app.graphs.graph import build_graph
from app.graphs.nodes import route_node


def _edge_pairs():
    g = build_graph(checkpointer=None).get_graph()
    return {(e.source, e.target) for e in g.edges}


def _node_names():
    return set(build_graph(checkpointer=None).get_graph().nodes)


# --- Group 1: Node presence ---

def test_graph_compiles():
    assert build_graph(checkpointer=None) is not None


def test_graph_has_tool_loop_nodes():
    names = _node_names()
    assert {"triage", "converse", "analyze", "summarize", "tools"} <= names


def test_removed_nodes_absent():
    names = _node_names()
    # The legacy intent pre-router / greeting node are gone; triage + converse replace them.
    assert {"ask_human", "confirm", "quality_gate", "propose_artifacts",
            "intent_router", "greeting"}.isdisjoint(names)


# --- Group 2: Topology / edges ---

def test_entry_point_is_triage():
    g = build_graph(checkpointer=None).get_graph()
    assert ("__start__", "triage") in {(e.source, e.target) for e in g.edges}


def test_triage_branches_to_converse_and_analyze():
    pairs = _edge_pairs()
    assert ("triage", "converse") in pairs
    assert ("triage", "analyze") in pairs


def test_converse_flows_to_analyze():
    assert ("converse", "analyze") in _edge_pairs()


def test_analyze_branches_to_tools_and_end():
    pairs = _edge_pairs()
    assert ("analyze", "tools") in pairs
    assert ("analyze", "__end__") in pairs


def test_tools_loops_back_through_summarize_and_analyze():
    pairs = _edge_pairs()
    assert ("tools", "summarize") in pairs
    assert ("tools", "analyze") in pairs
    assert ("summarize", "analyze") in pairs


# --- Group 3: route_node dispatch ---

def test_route_node_dispatches_to_tools_on_tool_calls():
    state = {
        "turn_count": 0,
        "messages": [AIMessage(content="", tool_calls=[{"id": "1", "name": "ask_user", "args": {}}])],
    }
    assert route_node(state) == "tools"


def test_route_node_ends_without_tool_calls():
    state = {"turn_count": 0, "messages": [AIMessage(content="done")]}
    assert route_node(state) == "__end__"


def test_route_node_ends_at_turn_cap():
    # turn_count >= max_agent_turns ends the loop regardless of pending tool_calls.
    from app.config import settings

    state = {
        "turn_count": settings.max_agent_turns,
        "messages": [AIMessage(content="", tool_calls=[{"id": "1", "name": "ask_user", "args": {}}])],
    }
    assert route_node(state) == "__end__"
