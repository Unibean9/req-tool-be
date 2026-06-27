from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.graphs.agent_tools import (
    get_all_analyzer_tools,
)
from app.graphs.nodes import (
    analyze_node,
    converse_node,
    orchestrator_node,
    route_after_triage,
    route_before_analyze,
    route_node,
    summarize_node,
    triage_node,
)
from app.graphs.state import WorkflowState


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    builder = StateGraph(WorkflowState)

    builder.add_node("triage", triage_node)
    builder.add_node("converse", converse_node)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("analyze", analyze_node)
    builder.add_node("summarize", summarize_node)
    # Tool-loop: analyze emits an AIMessage(tool_calls) via the shim, route_node dispatches to this
    # ToolNode, the tool runs (and may interrupt for HITL), then we loop back to analyze. The periodic
    # conversation summary is checked on the way back (route_before_analyze).
    builder.add_node(
        "tools",
        ToolNode(get_all_analyzer_tools()),
    )

    # Entry: a cheap triage classifies each fresh turn. A conversational turn (greeting/smalltalk)
    # peels off to converse (reply + pause) without paying the full analyst pass; everything else
    # goes straight to analyze. converse flows into analyze on resume so the human's real reply is
    # then analyzed. On resume LangGraph re-enters the interrupted node, so triage does not re-run.
    builder.set_entry_point("triage")
    builder.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "converse": "converse",
            "analyze": "orchestrator",
        },
    )
    builder.add_edge("converse", "orchestrator")
    builder.add_edge("orchestrator", "analyze")
    builder.add_conditional_edges(
        "analyze",
        route_node,
        {
            "tools": "tools",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "tools",
        route_before_analyze,
        {
            "summarize": "summarize",
            "analyze": "orchestrator",
        },
    )
    builder.add_edge("summarize", "orchestrator")

    return builder.compile(checkpointer=checkpointer)
