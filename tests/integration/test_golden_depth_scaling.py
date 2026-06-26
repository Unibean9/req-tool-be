"""Golden depth-scaling and parked resurface contract.

A parked question resurfaces the same turn its blocker resolves; PRD depth adds business
rules + edge-cases; completeness sweep creates parked questions for gaps.
"""

import pytest

from app.graphs.decision_graph import (
    add_parked_questions_for_gaps,
    completeness_sweep,
    render_view,
    scan_parked_questions,
    update_node,
)
from app.graphs.nodes import orchestrator_node
from tests.integration.golden_fixtures import part4_parked_graph

pytestmark = pytest.mark.integration


def test_parked_question_resurfaces_when_blocker_resolved():
    nodes = part4_parked_graph()
    assert nodes["Q4"]["status"] == "parked"
    assert nodes["Q4"]["blocks"] == ["N7"]

    resolved = update_node(nodes, "N7", status="confirmed", statement="Quán khoảng 150 khách/ngày")
    resurfaced = scan_parked_questions(resolved)

    assert [node["id"] for node in resurfaced] == ["Q4"]


def test_depth_scaling_prd_includes_rules_and_edge_cases():
    nodes = {
        "R1": {
            "id": "R1", "kind": "fact", "statement": "Business rule: 1 ghé / khách / ngày",
            "status": "inferred", "origin": {}, "depends_on": [], "supersedes": None,
            "superseded_by": None, "blocks": [], "answer": None,
        },
        "R2": {
            "id": "R2", "kind": "fact", "statement": "Business rule: ghé tính theo đơn",
            "status": "needs_confirmation", "origin": {}, "depends_on": [], "supersedes": None,
            "superseded_by": None, "blocks": [], "answer": None,
        },
    }
    gaps = completeness_sweep(nodes, artifact_type="prd")
    nodes, _created = add_parked_questions_for_gaps(nodes, gaps, {"turn": 12, "by": "agent"})

    out = render_view(nodes, "prd")

    assert "Business Rules" in out
    assert out.count("Business rule") >= 2
    assert out.count("Edge-case") >= 3


@pytest.mark.asyncio
async def test_completeness_sweep_creates_parked_questions_for_gaps(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N7", "kind": "objective", "status": "confirmed"},
        {"id": "N8", "kind": "assumption", "status": "confirmed"},
    )

    update = await orchestrator_node({"decision_nodes": nodes, "artifact_type": "prd"}, {})

    created = update["feedback_summary"]["created_parked_questions"]
    assert len(created) >= 3
    for item in created:
        node = update["decision_nodes"][item["id"]]
        assert node["kind"] == "open_question"
        assert node["status"] == "parked"
        assert node["blocks"] == []
