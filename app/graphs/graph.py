from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.graphs.agent_tools import ask_user, finalize, write_draft, write_note
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
    # Phase 3 enum-parity tools (ask_user/write_draft/finalize) plus the Phase 4 explore tool
    # (write_note). The tool-loop (tool_loop_only=True) can dispatch any tool get_available_tools
    # offers, so write_note must be registered here or its dispatch would raise.
    builder.add_node("tools", ToolNode([ask_user, write_draft, finalize, write_note]))

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
