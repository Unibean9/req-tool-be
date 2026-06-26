"""Golden Phần 1 — cold-start exploration (made green in Phase 06).

Empty project + shallow input => agent explores (reads graph, runs elicitation) before
drafting, and every node it creates carries technique provenance.

Stubs: fixture state is built so the harness validates it; the agent-turn drive is the
NotImplementedError placeholder each implementing phase replaces.
"""

import pytest

from tests.integration.golden_fixtures import part1_empty_state

pytestmark = [
    pytest.mark.integration,
    pytest.mark.xfail(reason="golden TDD stub — agent turn drive lands in Phase 06", strict=False),
]


def test_cold_start_explores_before_drafting():
    state = part1_empty_state()
    assert state == {}
    # Expect after one turn: no write_draft on turn 1; read_artifact_graph called (0 nodes);
    # elicit() called >=1 with technique in {comparable_products, 5_whys, reverse};
    # created nodes have origin.technique != None.
    raise NotImplementedError("drive agent cold-start turn — Phase 06")


def test_nodes_created_with_technique_provenance():
    state = part1_empty_state()
    assert state == {}
    # Expect each agent-created node: origin.by == "agent" and origin.technique set;
    # at least one technique in {"5_whys", "comparable_products"}.
    raise NotImplementedError("assert provenance after cold-start turn — Phase 06")
