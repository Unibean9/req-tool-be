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
    assert {"intent_router", "greeting", "analyze", "summarize", "tools"} <= names


def test_removed_nodes_absent():
    names = _node_names()
    assert {"ask_human", "confirm", "quality_gate", "propose_artifacts"}.isdisjoint(names)


# --- Group 2: Topology / edges ---

def test_entry_point_is_intent_router():
    g = build_graph(checkpointer=None).get_graph()
    assert ("__start__", "intent_router") in {(e.source, e.target) for e in g.edges}


def test_intent_router_branches_to_greeting_and_analyze():
    pairs = _edge_pairs()
    assert ("intent_router", "greeting") in pairs
    assert ("intent_router", "analyze") in pairs


def test_greeting_flows_to_analyze():
    assert ("greeting", "analyze") in _edge_pairs()


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
    state = {
        "turn_count": 10,
        "messages": [AIMessage(content="", tool_calls=[{"id": "1", "name": "ask_user", "args": {}}])],
    }
    assert route_node(state) == "__end__"
