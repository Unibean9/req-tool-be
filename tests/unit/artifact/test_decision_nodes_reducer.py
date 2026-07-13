"""merge_decision_nodes must keep concurrent same-turn node writes from clobbering each other."""

from app.graphs.state import merge_decision_nodes


def test_concurrent_adds_from_same_snapshot_both_survive():
    snapshot = {"N0": {"id": "N0"}}
    right_a = {**snapshot, "N1": {"id": "N1"}}  # tool A: snapshot + N1
    right_b = {**snapshot, "N2": {"id": "N2"}}  # tool B: snapshot + N2 (same pre-turn snapshot)

    after_a = merge_decision_nodes(snapshot, right_a)
    after_b = merge_decision_nodes(after_a, right_b)

    assert set(after_b) == {"N0", "N1", "N2"}


def test_full_dict_update_replaces_per_key():
    left = {"N1": {"id": "N1", "status": "proposed"}}
    right = {"N1": {"id": "N1", "status": "superseded"}}
    assert merge_decision_nodes(left, right)["N1"]["status"] == "superseded"


def test_empty_sides():
    assert merge_decision_nodes(None, {"N1": {}}) == {"N1": {}}
    assert merge_decision_nodes({"N1": {}}, None) == {"N1": {}}
    assert merge_decision_nodes(None, None) == {}
