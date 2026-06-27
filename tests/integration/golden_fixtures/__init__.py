"""Snapshot decision-graph states at the start of each golden Part.

Each builder returns a `decision_nodes` dict (id -> DecisionNode) representing the
state BEFORE the agent acts in that Part, so integration tests can drive one turn and
assert the transition.
"""

from tests.golden_decision_helpers import graph_from, make_decision_node


def part1_empty_state() -> dict[str, dict]:
    """Part 1 cold-start: empty project, no nodes yet."""
    return {}


def part3_pre_reversal_graph() -> dict[str, dict]:
    """Part 3 pre-reversal: N1 root decision with three dependents, all confirmed.

    N1 is a root (depends_on=[]) direction-setting decision; N3/N4/N5 build on it.
    After supersede_node(..., cascade_mode="abandon") the three dependents become
    `parked` (NOT `needs_confirmation`) — this is the abandon branch of the ripple
    invariant, distinct from the reconfirm branch used for local edits.
    """
    n1 = make_decision_node(
        "N1", kind="decision", statement="Prioritize operations-first", status="confirmed", depends_on=[]
    )
    n3 = make_decision_node(
        "N3", kind="objective", statement="Reduce preparation time", status="confirmed", depends_on=["N1"]
    )
    n4 = make_decision_node(
        "N4", kind="scope", statement="Realtime kitchen screen", status="confirmed", depends_on=["N1"]
    )
    n5 = make_decision_node(
        "N5", kind="assumption", statement="Staff use tablets at the counter", status="confirmed", depends_on=["N1"]
    )
    return graph_from([n1, n3, n4, n5])


def part4_parked_graph() -> dict[str, dict]:
    """Part 4 depth-scaling: Q4 parked, blocked by N7 which is not yet resolved.

    When N7 transitions to confirmed/inferred, scan_parked_questions must resurface Q4.
    """
    n7 = make_decision_node(
        "N7", kind="fact", statement="Store scale (customers/day)", status="needs_confirmation"
    )
    q4 = make_decision_node(
        "Q4",
        kind="open_question",
        statement="Do we need to design for peak load?",
        status="parked",
        blocks=["N7"],
    )
    return graph_from([n7, q4])
