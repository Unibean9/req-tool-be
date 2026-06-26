from app.graphs.decision_graph import impact, park_sync_debt


def _selector(_change, _nodes, _stale_artifacts):
    return ["R1", "S1"]


def test_impact_finds_affected_nodes_via_artifact_link(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "R1", "kind": "fact", "statement": "1 ghé / khách / ngày", "status": "confirmed"},
        {"id": "S1", "kind": "scope", "statement": "Thu ngân nhập SĐT", "status": "confirmed"},
        {"id": "U1", "kind": "risk", "statement": "Không liên quan", "status": "confirmed"},
    )

    result = impact("thêm kênh giao hàng", nodes, [{"source_id": "brd", "target_id": "prd"}], _selector, "brd")

    assert result["affected_node_ids"] == ["R1", "S1"]


def test_impact_marks_exactly_affected_nodes_needs_confirmation(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "R1", "kind": "fact", "status": "confirmed"},
        {"id": "S1", "kind": "scope", "status": "confirmed"},
        {"id": "U1", "kind": "risk", "status": "confirmed"},
    )

    result = impact("thêm kênh giao hàng", nodes, [{"source_id": "brd", "target_id": "prd"}], _selector, "brd")
    updated = result["decision_nodes"]

    assert updated["R1"]["status"] == "needs_confirmation"
    assert updated["S1"]["status"] == "needs_confirmation"
    assert updated["U1"]["status"] == "confirmed"


def test_impact_does_not_modify_statement(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "R1", "kind": "fact", "statement": "1 ghé / khách / ngày", "status": "confirmed"},
    )

    result = impact("thêm kênh giao hàng", nodes, [{"source_id": "brd", "target_id": "prd"}], lambda *_: ["R1"], "brd")

    assert result["decision_nodes"]["R1"]["statement"] == nodes["R1"]["statement"]


def test_impact_flags_artifact_stale(decision_graph_factory):
    nodes = decision_graph_factory({"id": "R1", "kind": "fact", "status": "confirmed"})

    result = impact("change", nodes, [{"source_id": "brd", "target_id": "prd"}], lambda *_: ["R1"], "brd")

    assert result["stale_artifact_ids"] == ["prd"]


def test_impact_empty_when_no_relevant_nodes(decision_graph_factory):
    nodes = decision_graph_factory({"id": "R1", "kind": "fact", "status": "confirmed"})

    result = impact("đổi màu logo", nodes, [{"source_id": "brd", "target_id": "prd"}], lambda *_: [], "brd")

    assert result["affected_node_ids"] == []
    assert result["decision_nodes"]["R1"]["status"] == "confirmed"


def test_impact_traversal_with_visited_set_guard(decision_graph_factory):
    nodes = decision_graph_factory({"id": "R1", "kind": "fact", "status": "confirmed"})
    links = [{"source_id": "A", "target_id": "B"}, {"source_id": "B", "target_id": "A"}]

    result = impact("change", nodes, links, lambda *_: [], "A")

    assert set(result["visited_artifact_ids"]) == {"A", "B"}


def test_sync_debt_parked_with_blocks(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "R1", "kind": "fact", "status": "needs_confirmation"},
        {"id": "S1", "kind": "scope", "status": "needs_confirmation"},
    )

    updated, debt = park_sync_debt(
        nodes,
        "Định nghĩa tích điểm đa kênh",
        ["R1", "S1"],
        {"turn": 1, "by": "agent"},
    )

    assert debt["status"] == "parked"
    assert debt["blocks"] == ["R1", "S1"]
    assert updated[debt["id"]] == debt
