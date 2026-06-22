from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.graphs.agent_tools import (
    ask_user,
    critique_note,
    explore_note,
    finalize,
    respond,
    run_critique,
    write_draft,
)
from app.graphs.nodes import (
    analyze_node,
    greeting_node,
    intent_router_node,
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
    builder.add_node("summarize", summarize_node)
    # Tool-loop: analyze emits an AIMessage(tool_calls) via the shim, route_node dispatches to this
    # ToolNode, the tool runs (and may interrupt for HITL), then we loop back to analyze. The periodic
    # conversation summary is checked on the way back (route_before_analyze).
    builder.add_node(
        "tools",
        ToolNode([ask_user, write_draft, finalize, critique_note, explore_note, respond, run_critique]),
    )

    # Entry: classify intent first. greeting/smalltalk → greeting (chitchat), else → analyze.
    # On resume LangGraph re-enters the interrupted node directly, so intent_router does not re-run.
    builder.set_entry_point("intent_router")
    builder.add_conditional_edges("intent_router", route_after_intent, {
        "greeting": "greeting",
        "analyze": "analyze",
    })
    builder.add_edge("greeting", "analyze")
    builder.add_conditional_edges("analyze", route_node, {
        "tools": "tools",
        END: END,
    })
    builder.add_conditional_edges("tools", route_before_analyze, {
        "summarize": "summarize",
        "analyze": "analyze",
    })
    builder.add_edge("summarize", "analyze")

    return builder.compile(checkpointer=checkpointer)
