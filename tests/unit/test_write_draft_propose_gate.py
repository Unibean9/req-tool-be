"""write_draft is the propose/approval gate, not a graph mutator.

When decision nodes exist the proposed body is the rendered view (the model's body arg is ignored);
mutation lives in the create/update/supersede tools. Without a graph, the model's body still drives.
"""

from app.graphs.agent_tools import _resolve_proposed_body
from app.graphs.decision_graph import render_view


def test_proposed_body_is_rendered_view_when_nodes_present(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N3", "kind": "objective", "statement": "Mục tiêu từ graph", "status": "confirmed"},
    )
    state = {"decision_nodes": nodes, "artifact_type": "brd"}

    resolved = _resolve_proposed_body(state, "BODY MODEL TỰ BỊA")

    assert resolved == render_view(nodes, "brd")
    assert "TỰ BỊA" not in resolved


def test_proposed_body_falls_back_to_model_body_without_graph():
    state = {"decision_nodes": {}, "artifact_type": "brd"}

    resolved = _resolve_proposed_body(state, "## Vision\nBody của model.")

    assert resolved == "## Vision\nBody của model."
