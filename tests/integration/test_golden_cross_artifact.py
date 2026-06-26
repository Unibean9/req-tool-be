"""Golden cross-artifact sync contract.

A cross-artifact change marks exactly the dependent nodes stale (needs_confirmation),
never silently rewrites them, and parks sync-debt with blocks when the user defers.
"""

import pytest

from app.graphs.decision_graph import impact, park_sync_debt

pytestmark = pytest.mark.integration


def _p5_nodes(decision_graph_factory):
    return decision_graph_factory(
        {"id": "R1", "kind": "fact", "statement": "1 ghé / khách / ngày", "status": "confirmed"},
        {
            "id": "S1",
            "kind": "scope",
            "statement": "Bước 1: thu ngân nhập SĐT khách lúc thanh toán",
            "status": "confirmed",
            "depends_on": ["R1"],
        },
        {"id": "O1", "kind": "objective", "statement": "Tăng tỉ lệ khách quay lại", "status": "confirmed"},
    )


def _selector(_change, _nodes, _stale_artifacts):
    return ["R1", "S1"]


def test_cross_artifact_change_marks_dependent_nodes_stale(decision_graph_factory):
    nodes = _p5_nodes(decision_graph_factory)
    assert nodes["R1"]["status"] == "confirmed"

    result = impact(
        "cho tích điểm cả khi đặt qua app giao hàng",
        nodes,
        [{"source_id": "brd", "target_id": "prd"}],
        _selector,
        "brd",
    )
    updated = result["decision_nodes"]

    assert result["affected_node_ids"] == ["R1", "S1"]
    assert updated["R1"]["status"] == "needs_confirmation"
    assert updated["S1"]["status"] == "needs_confirmation"
    assert updated["O1"]["status"] == "confirmed"


def test_stale_nodes_not_silently_fixed(decision_graph_factory):
    nodes = _p5_nodes(decision_graph_factory)

    result = impact("thêm kênh giao hàng", nodes, [{"source_id": "brd", "target_id": "prd"}], _selector, "brd")

    assert result["stale_artifact_ids"] == ["prd"]
    assert result["decision_nodes"]["R1"]["statement"] == nodes["R1"]["statement"]
    assert result["decision_nodes"]["S1"]["statement"] == nodes["S1"]["statement"]


def test_sync_debt_parked_with_blocks(decision_graph_factory):
    nodes = _p5_nodes(decision_graph_factory)
    result = impact("thêm kênh giao hàng", nodes, [{"source_id": "brd", "target_id": "prd"}], _selector, "brd")

    updated, debt = park_sync_debt(
        result["decision_nodes"],
        "Định nghĩa tích điểm đa kênh",
        result["affected_node_ids"],
        {"turn": 13, "by": "agent"},
    )

    assert debt["kind"] == "open_question"
    assert debt["status"] == "parked"
    assert debt["blocks"] == ["R1", "S1"]
    assert updated["R1"]["status"] == "needs_confirmation"
    assert updated["S1"]["status"] == "needs_confirmation"
