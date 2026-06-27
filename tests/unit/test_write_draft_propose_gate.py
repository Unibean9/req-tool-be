"""write_draft is the propose/approval gate, not a graph mutator.

When decision nodes exist the proposed body is the rendered view (the model's body arg is ignored);
mutation lives in the create/update/supersede tools. Without a graph, the model's body still drives.
"""

from app.graphs.agent_tools import _resolve_proposed_body
from app.graphs.decision_graph import create_node, render_view


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


def test_proposed_body_renders_document_item_contract_when_nodes_present():
    nodes = {
        "V1": create_node(
            kind="objective",
            statement="Help students schedule study groups faster.",
            origin={"source": "test"},
            status="confirmed",
            node_id="V1",
            section="## Vision",
        ),
        "O1": create_node(
            kind="objective",
            statement="Reduce schedule agreement time below 10 minutes.",
            origin={"source": "test"},
            status="confirmed",
            node_id="O1",
            section="## Objectives",
        ),
        "M1": create_node(
            kind="objective",
            statement="Measure successful group scheduling rate.",
            origin={"source": "test"},
            status="confirmed",
            node_id="M1",
            section="## Success Metrics",
            fields={
                "goal": "Schedule study groups",
                "user/business value": "Students reduce coordination loops",
                "metric": "Successful scheduling rate",
                "target": "80%",
                "timeframe": "First semester",
            },
        ),
    }
    state = {"decision_nodes": nodes, "artifact_type": "vision_objectives"}

    resolved = _resolve_proposed_body(state, "MODEL BODY WITH A POLISHED TABLE")

    assert "BODY MODEL" not in resolved
    assert "## Vision" in resolved
    assert "## Objectives" in resolved
    assert "## Success Metrics" in resolved
    assert "## Vision & Objectives" not in resolved
    assert "| goal | user/business value | metric | target | timeframe |" in resolved
