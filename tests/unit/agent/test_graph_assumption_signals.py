"""Synthesis assumption signals derived from the decision graph.

Guards against the dual-source-of-truth divergence: a graph-only session (no critique_note/explore_note
calls) used to leave open_questions=[] so write_draft reported SUFFICIENT while unresolved nodes remained.
Now assumptions/open-questions are derived from the graph, so synthesis always sees them.
"""

from app.graphs.decision_graph import synthesis_assumption_signals


def _node(kind, status, statement):
    return {"kind": kind, "status": status, "statement": statement}


def test_needs_confirmation_node_is_pending():
    confirmed, pending = synthesis_assumption_signals(
        {"N1": _node("assumption", "needs_confirmation", "Use visit-gift card")}
    )
    assert pending == ["Use visit-gift card"]
    assert confirmed == []


def test_open_question_node_is_pending_but_parked_is_not():
    nodes = {
        "Q1": _node("open_question", "proposed", "One store or multiple branches?"),
        "Q2": _node("open_question", "parked", "Is POS integration included?"),
    }
    _confirmed, pending = synthesis_assumption_signals(nodes)
    assert pending == ["One store or multiple branches?"]


def test_confirmed_assumption_node_is_confirmed_not_pending():
    confirmed, pending = synthesis_assumption_signals(
        {"N1": _node("assumption", "confirmed", "Loss baseline ~10%")}
    )
    assert confirmed == ["Loss baseline ~10%"]
    assert pending == []


def test_resolved_open_question_does_not_resurface_as_pending():
    # An open_question answered into confirmed/inferred is settled — it must not count as pending.
    _confirmed, pending = synthesis_assumption_signals(
        {"Q4": _node("open_question", "confirmed", "Customer scale/day")}
    )
    assert pending == []


def test_empty_graph_yields_no_signals():
    assert synthesis_assumption_signals({}) == ([], [])
