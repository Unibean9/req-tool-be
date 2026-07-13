"""Acceptance criterion #31 (plan.md): `_finalize_gate_open` — the sole source of
`log_gate_decision("finalize", ...)` — must be invoked by the menu's finalize rule if
and only if `has_draft and critique_rounds > 0`, and exactly once per
`get_available_tools` call when it is invoked at all (proving the rule wraps it, not
duplicates its evaluation).
"""

from unittest.mock import patch

from app.graphs.agent_tools import get_available_tools


def _finalize_calls(mock_log):
    return [call for call in mock_log.call_args_list if call.args and call.args[0] == "finalize"]


def test_finalize_gate_not_consulted_without_draft():
    state = {"draft_body": "", "critique_rounds": 1}
    with patch("app.graphs.agent_tools.log_gate_decision") as mock_log:
        get_available_tools(state)
    assert _finalize_calls(mock_log) == []


def test_finalize_gate_not_consulted_when_critique_rounds_zero():
    state = {"draft_body": "A draft", "critique_rounds": 0}
    with patch("app.graphs.agent_tools.log_gate_decision") as mock_log:
        get_available_tools(state)
    assert _finalize_calls(mock_log) == []


def test_finalize_gate_consulted_at_most_twice_when_precondition_holds():
    """`_finalize_gate_open` is called from two independent sites when the precondition holds —
    the menu's finalize rule AND `current_session_phase` -> `_phase_signals` (phase derivation
    also consults the gate) — matching pre-existing (pre-Phase-3) behavior exactly: this is a
    verified discrepancy from the plan's "called at least once, exactly once" premise, not a
    regression (see implementation-notes.md). The invariant this test actually protects is that
    the rule itself doesn't invoke the gate MORE than the original inline code did (2, not 3+)."""
    state = {"draft_body": "A draft", "critique_rounds": 1}
    with patch("app.graphs.agent_tools.log_gate_decision") as mock_log:
        get_available_tools(state)
    assert len(_finalize_calls(mock_log)) == 2
