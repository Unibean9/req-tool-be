from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from app.graphs.critic import quality_gate_node, route_after_gate
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
    builder.add_node("quality_gate", quality_gate_node)
    builder.add_node("propose_artifacts", propose_artifacts_node)

    builder.set_entry_point("analyze")
    builder.add_conditional_edges("analyze", route_node, {
        "ask_human": "ask_human",
        "confirm": "confirm",
        END: END,
    })
    builder.add_edge("ask_human", "analyze")
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
