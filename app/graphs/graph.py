from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from app.graphs.nodes import (
    analyze_node,
    ask_human_node,
    confirm_node,
    propose_artifacts_node,
    route_after_confirm,
    route_node,
)
from app.graphs.state import WorkflowState


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    builder = StateGraph(WorkflowState)

    builder.add_node("analyze", analyze_node)
    builder.add_node("ask_human", ask_human_node)
    builder.add_node("confirm", confirm_node)
    builder.add_node("propose_artifacts", propose_artifacts_node)

    builder.set_entry_point("analyze")
    builder.add_conditional_edges("analyze", route_node, {
        "ask_human": "ask_human",
        "confirm": "confirm",
        END: END,
    })
    builder.add_edge("ask_human", "analyze")
    builder.add_conditional_edges("confirm", route_after_confirm, {
        "propose_artifacts": "propose_artifacts",
        "analyze": "analyze",
    })
    builder.add_edge("propose_artifacts", END)

    return builder.compile(checkpointer=checkpointer)
