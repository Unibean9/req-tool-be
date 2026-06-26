"""Golden Phần 3 — decision reversal (made green in Phase 06).

Reversing a confirmed root decision supersedes (keeps history), ripples to dependents in
abandon mode (parked), and the agent reads dependents + pushes back before writing.
"""

import pytest

from tests.integration.golden_fixtures import part3_pre_reversal_graph

pytestmark = [
    pytest.mark.integration,
    pytest.mark.xfail(reason="golden TDD stub — agent turn drive lands in Phase 06", strict=False),
]


def test_decision_reversal_supersedes_not_deletes():
    nodes = part3_pre_reversal_graph()
    assert nodes["N1"]["status"] == "confirmed"
    # User: "Đổi qua giữ chân khách" -> N1 still present, status=superseded;
    # new N6 with supersedes=N1, statement contains "loyalty"; N1.superseded_by==N6.id.
    raise NotImplementedError("drive reversal turn — Phase 06")


def test_decision_reversal_uses_abandon_mode():
    nodes = part3_pre_reversal_graph()
    assert [nodes[n]["status"] for n in ("N3", "N4", "N5")] == ["confirmed"] * 3
    # After reversal: N3/N4/N5 status == parked (abandon, NOT needs_confirmation).
    raise NotImplementedError("assert abandon ripple — Phase 06")


def test_agent_reads_dependents_before_mutating():
    nodes = part3_pre_reversal_graph()
    assert sorted(n for n in nodes if "N1" in nodes[n]["depends_on"]) == ["N3", "N4", "N5"]
    # Agent calls read_artifact_graph(dependents_of=N1) BEFORE creating N6;
    # affected nodes surfaced in response text.
    raise NotImplementedError("assert read-before-write ordering — Phase 06")


def test_agent_pushback_before_accepting_direction_change():
    # Agent response contains a challenge/consequence warning and does not write
    # immediately — waits for user confirmation.
    raise NotImplementedError("assert pushback behavior — Phase 06")
