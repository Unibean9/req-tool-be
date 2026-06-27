"""Parked-question resurfacing contract.

A parked open_question resurfaces once every node it blocks reaches a resolved status
(confirmed OR inferred — not only confirmed).
"""

import pytest

try:
    from app.graphs.decision_graph import (
        MAX_SWEEP_QUESTIONS,
        add_parked_questions_for_gaps,
        completeness_sweep,
        is_brd_stable,
        scan_parked_questions,
    )

    _PENDING = None
except ImportError as exc:  # pragma: no cover - resolves once decision_graph resurfacing ships
    scan_parked_questions = completeness_sweep = add_parked_questions_for_gaps = is_brd_stable = None
    _PENDING = str(exc)

pytestmark = pytest.mark.xfail(
    _PENDING is not None,
    reason=f"scan_parked_questions not yet available: {_PENDING}",
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


def test_parked_empty_blocks_not_resurfaced(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "Q5", "kind": "open_question", "status": "parked", "blocks": []},
    )

    assert scan_parked_questions(nodes) == []


def test_brd_stable_ignores_parked_and_superseded(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N7", "kind": "objective", "status": "confirmed"},
        {"id": "N8", "kind": "assumption", "status": "inferred"},
        {"id": "Q4", "kind": "open_question", "status": "parked"},
        {"id": "N1", "kind": "decision", "status": "superseded"},
    )

    assert is_brd_stable(nodes) is True


def test_brd_not_stable_with_needs_confirmation(decision_graph_factory):
    nodes = decision_graph_factory({"id": "N7", "kind": "objective", "status": "needs_confirmation"})

    assert is_brd_stable(nodes) is False


def test_completeness_sweep_identifies_prd_edge_case_gaps(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "R1", "kind": "fact", "statement": "Business rule: 1 visit / customer / day", "status": "confirmed"},
        {"id": "F1", "kind": "scope", "statement": "Main flow: enter phone number and add points", "status": "confirmed"},
    )

    gaps = completeness_sweep(nodes, artifact_type="prd")

    assert any("Edge-case" in gap for gap in gaps)


def test_completeness_sweep_creates_parked_questions(decision_graph_factory):
    nodes = decision_graph_factory({"id": "R1", "kind": "fact", "statement": "Business rule", "status": "confirmed"})
    gaps = completeness_sweep(nodes, artifact_type="prd", max_questions=2)

    updated, created = add_parked_questions_for_gaps(nodes, gaps, {"turn": 1, "by": "agent"})

    assert len(created) == len(gaps)
    assert all(node["status"] == "parked" for node in created)
    assert all(node["blocks"] == [] for node in created)
    assert set(updated) > set(nodes)


def test_completeness_sweep_deduplicates_existing_gaps(decision_graph_factory):
    existing = "Edge-case: customer forgot phone number at purchase -> can points be added later?"
    nodes = decision_graph_factory(
        {"id": "Q5", "kind": "open_question", "statement": existing, "status": "parked"},
    )

    gaps = completeness_sweep(nodes, artifact_type="prd")

    assert existing not in gaps


def test_completeness_sweep_caps_questions_per_run(decision_graph_factory):
    nodes = decision_graph_factory()

    gaps = completeness_sweep(nodes, artifact_type="prd")

    assert len(gaps) <= MAX_SWEEP_QUESTIONS
