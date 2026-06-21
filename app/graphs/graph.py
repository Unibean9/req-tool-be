from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.graphs.critic import quality_gate_node, route_after_gate
from app.graphs.nodes import (
    analyze_node,
    ask_human_node,
    confirm_node,
    greeting_node,
    intent_router_node,
    propose_artifacts_node,
    route_after_confirm,
    route_after_intent,
    route_before_analyze,
    route_node,
    summarize_node,
)
from app.graphs.state import WorkflowState


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    builder = StateGraph(WorkflowState)

    builder.add_node("intent_router", intent_router_node)
    builder.add_node("greeting", greeting_node)
    builder.add_node("analyze", analyze_node)
    builder.add_node("ask_human", ask_human_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("confirm", confirm_node)
    builder.add_node("quality_gate", quality_gate_node)
    builder.add_node("propose_artifacts", propose_artifacts_node)
    # Phase 2 scaffolding: a ToolNode wired in parallel to the enum branches. It carries no tools
    # yet and route_node cannot reach it until the next_action enum is removed (Phase 5); registering
    # it now lets the harness exercise tool-call dispatch without a topology change later. Dispatching
    # an empty ToolNode raises ValueError, so route_node must never return "tools" while it is empty —
    # test_route_node_never_returns_tools_for_enum_actions guards that.
    builder.add_node("tools", ToolNode([]))

    # New entry point: classify intent first. greeting/smalltalk → greeting (chitchat), else → analyze.
    # On resume LangGraph re-enters the interrupted node directly, so intent_router does not re-run.
    builder.set_entry_point("intent_router")
    builder.add_conditional_edges("intent_router", route_after_intent, {
        "greeting": "greeting",
        "analyze": "analyze",
    })
    builder.add_edge("greeting", "analyze")
    builder.add_conditional_edges("analyze", route_node, {
        "ask_human": "ask_human",
        "confirm": "confirm",
        "tools": "tools",
        END: END,
    })
    builder.add_edge("tools", "analyze")
    builder.add_conditional_edges("ask_human", route_before_analyze, {
        "summarize": "summarize",
        "analyze": "analyze",
    })
    builder.add_edge("summarize", "analyze")
    # route_after_confirm's "propose_artifacts" label dispatches into quality_gate
    # (inserts the gate between confirm and propose_artifacts).
    builder.add_conditional_edges("confirm", route_after_confirm, {
        "propose_artifacts": "quality_gate",
        "analyze": "analyze",
    })
    # Edge-loop: the gate loops back to itself or advances to propose_artifacts (cap in route_after_gate)
    builder.add_conditional_edges("quality_gate", route_after_gate, {
        "quality_gate": "quality_gate",
        "propose_artifacts": "propose_artifacts",
    })
    builder.add_edge("propose_artifacts", END)

    return builder.compile(checkpointer=checkpointer)
