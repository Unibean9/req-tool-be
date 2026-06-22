from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.graphs.agent_tools import (
    ask_user,
    critique_note,
    explore_note,
    finalize,
    recommend_next_workflow,
    respond,
    run_critique,
    run_readiness_check,
    write_draft,
)
from app.graphs.nodes import (
    analyze_node,
    route_before_analyze,
    route_node,
    summarize_node,
)
from app.graphs.state import WorkflowState


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    builder = StateGraph(WorkflowState)

    builder.add_node("analyze", analyze_node)
    builder.add_node("summarize", summarize_node)
    # Tool-loop: analyze emits an AIMessage(tool_calls) via the shim, route_node dispatches to this
    # ToolNode, the tool runs (and may interrupt for HITL), then we loop back to analyze. The periodic
    # conversation summary is checked on the way back (route_before_analyze).
    builder.add_node(
        "tools",
        ToolNode([
            ask_user, write_draft, finalize, critique_note, explore_note,
            respond, run_critique, recommend_next_workflow, run_readiness_check,
        ]),
    )

    # Entry: the analyst directly. It reads the conversation and current state, detects locale, and
    # handles greetings/smalltalk in-loop via respond/ask_user — no separate intent pre-router.
    builder.set_entry_point("analyze")
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
