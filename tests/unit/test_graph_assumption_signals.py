"""Readiness gate must see assumptions recorded as decision-graph nodes, not only legacy notes.

Guards against the dual-source-of-truth divergence: a graph-only session (no critique_note/explore_note
calls) used to leave open_questions=[] so write_draft reported SUFFICIENT while unresolved nodes remained.
"""

from app.graphs.agent_tools import _graph_assumption_signals


def _node(kind, status, statement):
    return {"kind": kind, "status": status, "statement": statement}


def test_needs_confirmation_node_is_pending():
    confirmed, pending = _graph_assumption_signals(
        {"N1": _node("assumption", "needs_confirmation", "Dùng thẻ ghé-tặng")}
    )
    assert pending == ["Dùng thẻ ghé-tặng"]
    assert confirmed == []


def test_open_question_node_is_pending_but_parked_is_not():
    nodes = {
        "Q1": _node("open_question", "proposed", "Một quán hay nhiều chi nhánh?"),
        "Q2": _node("open_question", "parked", "Có tích hợp POS không?"),
    }
    _confirmed, pending = _graph_assumption_signals(nodes)
    assert pending == ["Một quán hay nhiều chi nhánh?"]


def test_confirmed_assumption_node_is_confirmed_not_pending():
    confirmed, pending = _graph_assumption_signals(
        {"N1": _node("assumption", "confirmed", "Baseline thất thoát ~10%")}
    )
    assert confirmed == ["Baseline thất thoát ~10%"]
    assert pending == []


def test_resolved_open_question_does_not_resurface_as_pending():
    # An open_question answered into confirmed/inferred is settled — it must not count as pending.
    _confirmed, pending = _graph_assumption_signals(
        {"Q4": _node("open_question", "confirmed", "Quy mô khách/ngày")}
    )
    assert pending == []


def test_empty_graph_yields_no_signals():
    assert _graph_assumption_signals({}) == ([], [])
