"""Golden TDD — parked-question resurfacing (made green in Phase 07).

A parked open_question resurfaces once every node it blocks reaches a resolved status
(confirmed OR inferred — not only confirmed).
"""

import pytest

try:
    from app.graphs.decision_graph import scan_parked_questions

    _PENDING = None
except ImportError as exc:  # pragma: no cover - resolves once Phase 07 lands
    scan_parked_questions = None
    _PENDING = str(exc)

pytestmark = pytest.mark.xfail(
    _PENDING is not None,
    reason=f"scan_parked_questions pending (Phase 07): {_PENDING}",
    strict=False,
)


def _parked_with_blocker(decision_graph_factory, blocker_status: str):
    return decision_graph_factory(
        {"id": "N7", "kind": "fact", "status": blocker_status},
        {"id": "Q4", "kind": "open_question", "status": "parked", "blocks": ["N7"]},
    )


def test_parked_resurface_detects_confirmed_blocker(decision_graph_factory):
    nodes = _parked_with_blocker(decision_graph_factory, "confirmed")

    resurfaced = scan_parked_questions(nodes)

    assert [n["id"] for n in resurfaced] == ["Q4"]


def test_parked_resurface_detects_inferred_blocker(decision_graph_factory):
    nodes = _parked_with_blocker(decision_graph_factory, "inferred")

    resurfaced = scan_parked_questions(nodes)

    assert [n["id"] for n in resurfaced] == ["Q4"]


def test_parked_no_resurface_when_blocker_needs_confirmation(decision_graph_factory):
    nodes = _parked_with_blocker(decision_graph_factory, "needs_confirmation")

    resurfaced = scan_parked_questions(nodes)

    assert resurfaced == []
