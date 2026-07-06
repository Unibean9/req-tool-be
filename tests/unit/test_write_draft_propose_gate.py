"""write_draft is the propose/approval gate, not a graph mutator.

Partial graph renders must not hide a complete model body or persisted draft.
"""

from app.graphs.agent_tools import _resolve_proposed_body
from app.graphs.decision_graph import create_node, render_view


def test_proposed_body_is_rendered_view_when_nodes_present(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N3", "kind": "objective", "statement": "Goal tu graph", "status": "confirmed"},
    )
    state = {"decision_nodes": nodes, "artifact_type": "brd"}

    resolved = _resolve_proposed_body(state, "FABRICATED MODEL BODY")

    assert resolved == render_view(nodes, "brd")
    assert "FABRICATED" not in resolved


def test_proposed_body_falls_back_to_model_body_without_graph():
    state = {"decision_nodes": {}, "artifact_type": "brd"}

    resolved = _resolve_proposed_body(state, "## Vision\nModel body.")

    assert resolved == "## Vision\nModel body."


def test_partial_document_item_graph_does_not_override_complete_body():
    nodes = {
        "V1": create_node(
            kind="objective",
            statement="Help students schedule study groups faster.",
            origin={"source": "test"},
            status="confirmed",
            node_id="V1",
            section="## Vision",
        )
    }
    body = "\n\n".join(
        [
            "## Vision\nModel-authored vision.",
            "## Objectives\n- Reduce schedule agreement time.",
            "## Success Metrics\n- Successful scheduling rate reaches 80%.",
        ]
    )
    state = {"decision_nodes": nodes, "artifact_type": "vision_objectives"}

    resolved = _resolve_proposed_body(state, body)

    assert resolved == body
    assert "## Success Metrics" in resolved


def test_partial_document_item_graph_keeps_complete_current_draft_when_body_is_incomplete():
    nodes = {
        "V1": create_node(
            kind="objective",
            statement="Help students schedule study groups faster.",
            origin={"source": "test"},
            status="confirmed",
            node_id="V1",
            section="## Vision",
        )
    }
    draft_body = "\n\n".join(
        [
            "## Vision\nVersion 2 vision.",
            "## Objectives\nVersion 2 objectives.",
            "## Success Metrics\nVersion 2 metrics.",
        ]
    )
    state = {"decision_nodes": nodes, "artifact_type": "vision_objectives", "draft_body": draft_body}

    resolved = _resolve_proposed_body(state, "## Vision\nIncomplete model body.")

    assert resolved == draft_body


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
