"""Golden TDD — supersession & ripple invariants (made green in Phase 04).

Until app.graphs.decision_graph ships, the module-level xfail condition is active and
these run as expected-failures (CI stays green). Once the module imports, the condition
flips False and the assertions run for real.
"""

import pytest

try:
    from app.graphs.decision_graph import get_dependents, supersede_node

    _PENDING = None
except ImportError as exc:  # pragma: no cover - resolves once Phase 04 lands
    get_dependents = supersede_node = None
    _PENDING = str(exc)

pytestmark = pytest.mark.xfail(
    _PENDING is not None,
    reason=f"decision_graph pending (Phase 04): {_PENDING}",
    strict=False,
)


def test_no_destructive_mutation_creates_new_node(decision_graph_factory):
    nodes = decision_graph_factory({"id": "N1", "kind": "decision", "status": "confirmed"})
    origin = {"turn": 3, "by": "user", "technique": None, "source": None}

    result = supersede_node(nodes, "N1", "Hướng mới", origin, cascade_mode="reconfirm")

    old = result["N1"]
    new_id = old["superseded_by"]
    assert old["status"] == "superseded"
    assert new_id is not None
    new = result[new_id]
    assert new["supersedes"] == "N1"
    assert new["status"] != "superseded"
    assert len(result) == len(nodes) + 1


def test_supersede_reconfirm_marks_dependents_needs_confirmation(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N1", "kind": "objective", "status": "confirmed"},
        {"id": "N3", "depends_on": ["N1"], "status": "confirmed"},
        {"id": "N4", "depends_on": ["N1"], "status": "confirmed"},
    )
    origin = {"turn": 4, "by": "user", "technique": None, "source": None}

    result = supersede_node(nodes, "N1", "Chỉnh cục bộ", origin, cascade_mode="reconfirm")

    assert result["N1"]["status"] == "superseded"
    assert result["N3"]["status"] == "needs_confirmation"
    assert result["N4"]["status"] == "needs_confirmation"


def test_supersede_abandon_marks_dependents_parked(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N1", "kind": "decision", "status": "confirmed"},
        {"id": "N3", "depends_on": ["N1"], "status": "confirmed"},
        {"id": "N4", "depends_on": ["N1"], "status": "confirmed"},
        {"id": "N5", "depends_on": ["N1"], "status": "confirmed"},
    )
    origin = {"turn": 5, "by": "user", "technique": None, "source": None}

    result = supersede_node(nodes, "N1", "Đảo hướng gốc", origin, cascade_mode="abandon")

    assert result["N1"]["status"] == "superseded"
    assert result["N3"]["status"] == "parked"
    assert result["N4"]["status"] == "parked"
    assert result["N5"]["status"] == "parked"


def test_supersede_ripple_does_not_affect_unrelated_nodes(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N1", "kind": "decision", "status": "confirmed"},
        {"id": "N3", "depends_on": ["N1"], "status": "confirmed"},
        {"id": "N2", "kind": "decision", "status": "confirmed"},
        {"id": "N5", "depends_on": ["N2"], "status": "confirmed"},
    )
    origin = {"turn": 6, "by": "user", "technique": None, "source": None}

    result = supersede_node(nodes, "N1", "Đổi N1", origin, cascade_mode="abandon")

    assert result["N5"]["status"] == "confirmed"


def test_supersede_cycle_guard_raises_error(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N1", "depends_on": ["N2"]},
        {"id": "N2", "depends_on": ["N1"]},
    )

    # Must terminate (visited-set guard) rather than recurse forever; either a finite
    # result or an explicit ValueError describing the cycle is acceptable.
    try:
        deps = get_dependents(nodes, "N1")
    except ValueError:
        return
    assert isinstance(deps, list)
    assert len(deps) <= len(nodes)
