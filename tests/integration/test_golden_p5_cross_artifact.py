"""Golden Phần 5 — cross-artifact sync (made green in Phase 08).

A cross-artifact change marks exactly the dependent nodes stale (needs_confirmation),
never silently rewrites them, and parks sync-debt with blocks when the user defers.
"""

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.xfail(reason="golden TDD stub — impact() drive lands in Phase 08", strict=False),
]


def test_cross_artifact_change_marks_dependent_nodes_stale(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "R1", "kind": "fact", "statement": "Quy tắc tích điểm", "status": "confirmed"},
        {"id": "S1", "kind": "scope", "statement": "Bước 1 luồng tích điểm", "status": "confirmed", "depends_on": ["R1"]},
    )
    assert nodes["R1"]["status"] == "confirmed"
    # User: "cho tích điểm cả khi đặt qua app giao hàng" -> R1 + S1 status==needs_confirmation,
    # EXACTLY 2 nodes marked stale (no speculative rewrite).
    raise NotImplementedError("drive cross-artifact impact turn — Phase 08")


def test_stale_nodes_not_silently_fixed():
    # Agent does NOT edit R1/step-1 content; flags PRD artifact stale; asks user whether to
    # open an analysis branch or just note it.
    raise NotImplementedError("assert no silent fix — Phase 08")


def test_sync_debt_parked_with_blocks():
    # User chooses "làm sau" -> Q8 open_question parked with blocks containing R1 + step-1 ids;
    # the 2 stale nodes remain visible in the view.
    raise NotImplementedError("assert sync-debt parking — Phase 08")
