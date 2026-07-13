"""DecisionNode CRUD contract.

Uses xfail-until-module-exists so CI stays green before decision_graph ships;
once the import resolves, every assertion runs real.
"""

import pytest

try:
    from app.graphs.decision_graph import create_node, update_node

    _PENDING = None
except ImportError as exc:  # pragma: no cover - resolves once decision_graph lands
    create_node = update_node = None
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


