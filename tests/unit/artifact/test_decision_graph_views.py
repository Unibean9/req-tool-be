"""Decision graph views: synthesis signal split, legacy-note migration, render_view."""

from app.graphs.decision_graph import (
    create_node,
    migrate_legacy_notes,
    render_view,
    synthesis_assumption_signals,
)


def _node(kind, status, statement):
    return {"kind": kind, "status": status, "statement": statement}


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
    assumption = next(n for n in nodes.values() if n["kind"] == "assumption")
    open_question = next(n for n in nodes.values() if n["kind"] == "open_question")
    assert assumption["statement"] == "users have smartphones"
    assert assumption["status"] == "needs_confirmation"
    assert open_question["statement"] == "which region first?"
    assert open_question["status"] == "proposed"


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


# --- render_view (superseded hidden, parked folded, active shown) ------------


def test_render_view_excludes_superseded_nodes(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N1", "kind": "decision", "statement": "Huong cu da bo", "status": "superseded"},
        {"id": "N3", "kind": "objective", "statement": "Goal being confirmed", "status": "confirmed"},
        {"id": "N4", "kind": "scope", "statement": "Pham vi treo lai", "status": "parked"},
    )

    out = render_view(nodes, "brd")

    assert "Huong cu da bo" not in out
    assert "Goal being confirmed" in out
    assert "Pham vi treo lai" in out
