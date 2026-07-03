"""DecisionNode CRUD and cascade inference contract.

Uses xfail-until-module-exists so CI stays green before decision_graph ships;
once the import resolves, every assertion runs real.
"""

import pytest

try:
    from app.graphs.decision_graph import (
        create_node,
        get_dependents,
        infer_cascade_mode,
        supersede_node,
        update_node,
    )

    _PENDING = None
except ImportError as exc:  # pragma: no cover - resolves once decision_graph lands
    create_node = update_node = supersede_node = get_dependents = infer_cascade_mode = None
    _PENDING = str(exc)

pytestmark = pytest.mark.xfail(
    _PENDING is not None,
    reason=f"decision_graph not yet available: {_PENDING}",
    strict=False,
)

_DECISION_NODE_KEYS = {
    "id", "kind", "statement", "status", "origin",
    "depends_on", "supersedes", "superseded_by", "blocks", "answer",
    "section", "fields",
}


def _origin(turn=1):
    return {"turn": turn, "by": "agent", "technique": None, "source": None}


def test_create_node_valid_schema():
    node = create_node(kind="objective", statement="Reduce loss", origin=_origin(), depends_on=[])

    assert set(node) == _DECISION_NODE_KEYS
    assert node["status"] == "proposed"
    assert node["supersedes"] is None
    assert node["superseded_by"] is None
    assert node["blocks"] == []
    assert node["answer"] is None
    assert node["section"] is None
    assert node["fields"] is None

    other = create_node(kind="objective", statement="Khac", origin=_origin(), depends_on=[])
    assert node["id"] and node["id"] != other["id"]


def test_update_node_status_and_statement(decision_graph_factory):
    nodes = decision_graph_factory({"id": "N3", "kind": "objective", "status": "proposed"})

    result = update_node(nodes, "N3", status="confirmed", statement="new")

    assert result["N3"]["status"] == "confirmed"
    assert result["N3"]["statement"] == "new"
    assert result["N3"]["id"] == "N3"


def test_update_node_rejects_invalid_status(decision_graph_factory):
    nodes = decision_graph_factory({"id": "N3", "kind": "objective", "status": "proposed"})

    with pytest.raises(ValueError):
        update_node(nodes, "N3", status="done")


def test_update_node_rejects_superseded_history_edit(decision_graph_factory):
    nodes = decision_graph_factory({"id": "N1", "kind": "decision", "status": "superseded"})

    with pytest.raises(ValueError):
        update_node(nodes, "N1", status="confirmed")


def test_update_node_does_not_supersede(decision_graph_factory):
    nodes = decision_graph_factory({"id": "N3", "kind": "objective", "status": "proposed"})

    result = update_node(nodes, "N3", status="confirmed")

    assert len(result) == len(nodes)
    assert not any(n.get("supersedes") == "N3" for n in result.values())


def test_supersede_node_reconfirm_full_lifecycle(decision_graph_factory):
    nodes = decision_graph_factory({"id": "N1", "kind": "objective", "status": "confirmed"})

    result = supersede_node(nodes, "N1", "Huong new", _origin(), cascade_mode="reconfirm")

    new_id = result["N1"]["superseded_by"]
    assert result["N1"]["status"] == "superseded"
    assert result[new_id]["status"] == "proposed"
    assert result[new_id]["supersedes"] == "N1"
    assert len(result) == len(nodes) + 1


def test_supersede_node_abandon_full_lifecycle(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N1", "kind": "decision", "status": "confirmed"},
        {"id": "N3", "depends_on": ["N1"], "status": "confirmed"},
        {"id": "N4", "depends_on": ["N1"], "status": "confirmed"},
    )

    result = supersede_node(nodes, "N1", "Dao huong", _origin(), cascade_mode="abandon")

    assert result["N1"]["status"] == "superseded"
    assert result["N3"]["status"] == "parked"
    assert result["N4"]["status"] == "parked"
    assert {"N3", "N4"} <= set(result)


def test_cascade_mode_inferred_when_not_specified(decision_graph_factory):
    # Case 1 — root direction node with several dependents → inferred abandon.
    direction = decision_graph_factory(
        {"id": "N1", "kind": "decision", "depends_on": [], "status": "confirmed"},
        {"id": "N3", "depends_on": ["N1"], "status": "confirmed"},
        {"id": "N4", "depends_on": ["N1"], "status": "confirmed"},
        {"id": "N5", "depends_on": ["N1"], "status": "confirmed"},
    )
    assert infer_cascade_mode(direction, "N1") == "abandon"
    out = supersede_node(direction, "N1", "Dao", _origin())
    assert out["N3"]["status"] == "parked"
    assert out["N4"]["status"] == "parked"
    assert out["N5"]["status"] == "parked"

    # Case 2 — local node with a parent → inferred reconfirm.
    local = decision_graph_factory(
        {"id": "N1", "kind": "decision", "status": "confirmed"},
        {"id": "N3", "kind": "objective", "depends_on": ["N1"], "status": "confirmed"},
        {"id": "N7", "depends_on": ["N3"], "status": "confirmed"},
    )
    assert infer_cascade_mode(local, "N3") == "reconfirm"
    out = supersede_node(local, "N3", "Sua gia tri", _origin())
    assert out["N7"]["status"] == "needs_confirmation"


def test_get_dependents_with_cycle_guard(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "A", "depends_on": ["B"]},
        {"id": "B", "depends_on": ["A"]},
    )

    deps = get_dependents(nodes, "A")

    assert isinstance(deps, list)
    assert len(deps) <= len(nodes)
    assert "A" not in deps


def test_supersede_node_cycle_does_not_overwrite_old_status(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "A", "kind": "decision", "depends_on": ["B"], "status": "confirmed"},
        {"id": "B", "depends_on": ["A"], "status": "confirmed"},
    )

    result = supersede_node(nodes, "A", "Dao huong", _origin(), cascade_mode="abandon")

    new_id = result["A"]["superseded_by"]
    assert result["A"]["status"] == "superseded"
    assert result["B"]["status"] == "parked"
    assert result[new_id]["status"] == "proposed"


def test_supersede_reconfirm_marks_dependents_needs_confirmation(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N1", "kind": "objective", "status": "confirmed"},
        {"id": "N3", "depends_on": ["N1"], "status": "confirmed"},
        {"id": "N4", "depends_on": ["N1"], "status": "confirmed"},
    )

    result = supersede_node(nodes, "N1", "Chinh cuc bo", _origin(), cascade_mode="reconfirm")

    assert result["N1"]["status"] == "superseded"
    assert result["N3"]["status"] == "needs_confirmation"
    assert result["N4"]["status"] == "needs_confirmation"


def test_supersede_ripple_does_not_affect_unrelated_nodes(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N1", "kind": "decision", "status": "confirmed"},
        {"id": "N3", "depends_on": ["N1"], "status": "confirmed"},
        {"id": "N2", "kind": "decision", "status": "confirmed"},
        {"id": "N5", "depends_on": ["N2"], "status": "confirmed"},
    )

    result = supersede_node(nodes, "N1", "Doi N1", _origin(), cascade_mode="abandon")

    assert result["N5"]["status"] == "confirmed"


def test_supersede_does_not_revive_already_superseded_dependent(decision_graph_factory):
    # N5 was already superseded by N5b in an earlier edit; both still depend_on the root N1.
    # Superseding N1 must ripple N5b but leave N5 frozen — reviving it would rewrite history.
    nodes = decision_graph_factory(
        {"id": "N1", "kind": "decision", "status": "confirmed"},
        {"id": "N5", "depends_on": ["N1"], "status": "superseded", "superseded_by": "N5b"},
        {"id": "N5b", "depends_on": ["N1"], "status": "confirmed", "supersedes": "N5"},
    )

    result = supersede_node(nodes, "N1", "Dao huong", _origin(), cascade_mode="abandon")

    assert result["N5"]["status"] == "superseded"
    assert result["N5"]["superseded_by"] == "N5b"
    assert result["N5b"]["status"] == "parked"
