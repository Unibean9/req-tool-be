"""Golden Phần 2 — co-creation (made green in Phase 06).

Confirming an objective updates the existing node in place (not a supersede); MoSCoW
elicitation pushes out-of-scope items to parked.
"""

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.xfail(reason="golden TDD stub — agent turn drive lands in Phase 06", strict=False),
]


def test_objective_confirmation_updates_node_not_creates_new(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N3", "kind": "objective", "statement": "Tăng doanh thu Y%", "status": "proposed"}
    )
    assert nodes["N3"]["status"] == "proposed"
    # User: "Chắc tầm 10%" -> N3.status==confirmed, statement contains "10%",
    # no node with supersedes=N3, total node count unchanged.
    raise NotImplementedError("drive confirmation turn — Phase 06")


def test_moscow_pushes_out_of_scope_to_parked():
    # elicit(technique="moscow", ...) -> "Out v1" items become open_question parked;
    # "Must" items become proposed/confirmed.
    raise NotImplementedError("drive moscow elicitation turn — Phase 06")
