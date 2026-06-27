import pytest

from app.graphs.nodes import orchestrator_node


@pytest.mark.asyncio
async def test_orchestrator_resurfaces_parked_question(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N7", "kind": "fact", "status": "confirmed"},
        {"id": "Q4", "kind": "open_question", "status": "parked", "blocks": ["N7"]},
    )

    update = await orchestrator_node({"decision_nodes": nodes, "artifact_type": "brd"}, {})

    assert update["feedback_summary"]["resurfaced_questions"][0]["id"] == "Q4"


@pytest.mark.asyncio
async def test_completeness_sweep_only_on_trigger(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N7", "kind": "objective", "status": "confirmed"},
        {"id": "N8", "kind": "assumption", "status": "confirmed"},
    )

    update = await orchestrator_node({"decision_nodes": nodes, "artifact_type": "brd"}, {})

    assert "decision_nodes" not in update
    assert "sweep_gaps" not in update["feedback_summary"]


@pytest.mark.asyncio
async def test_orchestrator_depth_transition_creates_parked_questions(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N7", "kind": "objective", "status": "confirmed"},
        {"id": "N8", "kind": "assumption", "status": "confirmed"},
    )

    update = await orchestrator_node({"decision_nodes": nodes, "artifact_type": "prd"}, {})

    created = update["feedback_summary"]["created_parked_questions"]
    assert created
    assert all(update["decision_nodes"][item["id"]]["status"] == "parked" for item in created)
