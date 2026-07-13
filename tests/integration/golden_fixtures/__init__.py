"""Snapshot decision-graph states at the start of each golden Part.

Each builder returns a `decision_nodes` dict (id -> DecisionNode) representing the
state BEFORE the agent acts in that Part, so integration tests can drive one turn and
assert the transition.
"""

from tests.golden_decision_helpers import graph_from, make_decision_node


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
