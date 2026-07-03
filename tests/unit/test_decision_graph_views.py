"""Phase 5 — assumptions/open-questions as derived views over the decision graph.

One test per divergence class in evidence/phase-05-divergence-audit.md, plus the migration helper.
"""

from app.graphs.decision_graph import (
    create_node,
    derive_assumptions,
    derive_open_questions,
    migrate_legacy_notes,
    synthesis_assumption_signals,
)


def _node(kind, status, statement):
    return {"kind": kind, "status": status, "statement": statement}


# --- derive_assumptions -----------------------------------------------------


def test_derive_assumptions_returns_active_assumption_nodes_with_status():
    nodes = {
        "A1": _node("assumption", "confirmed", "Loss baseline ~10%"),
        "A2": _node("assumption", "needs_confirmation", "MVP targets small garages"),
        "O1": _node("open_question", "proposed", "One store or many?"),  # not an assumption
    }
    derived = derive_assumptions(nodes)
    assert {a["statement"] for a in derived} == {"Loss baseline ~10%", "MVP targets small garages"}
    assert {a["status"] for a in derived} == {"confirmed", "needs_confirmation"}


def test_derive_assumptions_excludes_superseded_and_parked():
    nodes = {
        "A1": _node("assumption", "superseded", "old"),
        "A2": _node("assumption", "parked", "deferred"),
        "A3": _node("assumption", "confirmed", "kept"),
    }
    assert [a["statement"] for a in derive_assumptions(nodes)] == ["kept"]


# --- derive_open_questions --------------------------------------------------


def test_derive_open_questions_uses_question_key_and_skips_parked():
    nodes = {
        "Q1": _node("open_question", "proposed", "Which gateway?"),
        "Q2": _node("open_question", "parked", "Is POS included?"),
        "A1": _node("assumption", "confirmed", "not a question"),
    }
    derived = derive_open_questions(nodes)
    assert derived == [{"question": "Which gateway?", "status": "proposed"}]


def test_derive_views_on_empty_graph():
    assert derive_assumptions({}) == []
    assert derive_open_questions({}) == []


# --- synthesis split (D5: any needs_confirmation kind counts pending) --------


def test_synthesis_counts_cascade_stale_non_assumption_node_as_pending():
    # A supersede cascade marks a scope node needs_confirmation — it must still block readiness (D5).
    nodes = {"S1": _node("scope", "needs_confirmation", "v1 scope must be reconfirmed")}
    confirmed, pending = synthesis_assumption_signals(nodes)
    assert confirmed == []
    assert pending == ["v1 scope must be reconfirmed"]


# --- migrate_legacy_notes (D1/D2/D3) ----------------------------------------


def test_migrate_creates_needs_confirmation_assumption_and_proposed_open_question():
    nodes, migrated = migrate_legacy_notes(
        {},
        [{"statement": "users have smartphones"}],
        [{"question": "which region first?"}],
        {"by": "migration"},
    )
    assert migrated == 2
    assumptions = derive_assumptions(nodes)
    open_questions = derive_open_questions(nodes)
    assert assumptions == [{"statement": "users have smartphones", "status": "needs_confirmation"}]
    assert open_questions == [{"question": "which region first?", "status": "proposed"}]


def test_migrate_is_idempotent_against_existing_matching_nodes():
    existing = {"A1": create_node(kind="assumption", statement="users have smartphones", origin={})}
    nodes, migrated = migrate_legacy_notes(
        existing, [{"statement": "Users have smartphones"}], None, {"by": "migration"}
    )
    assert migrated == 0
    assert len(nodes) == 1


def test_migrate_skips_blank_entries():
    nodes, migrated = migrate_legacy_notes({}, [{"statement": "  "}], [{"question": ""}], {"by": "migration"})
    assert migrated == 0
    assert nodes == {}
