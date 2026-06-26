"""Golden Phần 4 — depth scaling & parked resurface (made green in Phase 07).

A parked question resurfaces the same turn its blocker resolves; PRD depth adds business
rules + edge-cases; completeness sweep creates parked questions for gaps.
"""

import pytest

from tests.integration.golden_fixtures import part4_parked_graph

pytestmark = [
    pytest.mark.integration,
    pytest.mark.xfail(reason="golden TDD stub — orchestrator drive lands in Phase 07", strict=False),
]


def test_parked_question_resurfaces_when_blocker_resolved():
    nodes = part4_parked_graph()
    assert nodes["Q4"]["status"] == "parked"
    assert nodes["Q4"]["blocks"] == ["N7"]
    # User: "Quán tôi tầm 150 khách/ngày" resolves N7 -> agent mentions Q4 the SAME turn,
    # resurfaced without the user asking about it.
    raise NotImplementedError("drive orchestrator scan turn — Phase 07")


def test_depth_scaling_prd_includes_rules_and_edge_cases():
    # PRD draft contains >=2 Business Rules and >=3 Edge-cases; response mentions depth=PRD.
    raise NotImplementedError("drive PRD depth turn — Phase 07")


def test_completeness_sweep_creates_parked_questions_for_gaps():
    # After PRD draft, agent creates Q5/Q6/Q7 parked with blocks=[] for unresolved edge-cases.
    raise NotImplementedError("drive completeness sweep — Phase 07")
