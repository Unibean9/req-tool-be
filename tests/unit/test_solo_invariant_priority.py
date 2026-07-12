"""`SoloInvariantBatchRule` must keep the interrupt-bearing tool according to the explicit priority
table (`_SOLO_INVARIANT_PRIORITY`), not "whichever tool the model listed first"."""

from unittest.mock import patch

from app.graphs.gating.dispatch_rules import SoloInvariantBatchRule


def test_solo_invariant_keeps_higher_priority_tool_regardless_of_input_order():
    rule = SoloInvariantBatchRule()
    calls = [{"name": "respond", "args": {}}, {"name": "ask_user", "args": {}}]

    with patch("app.graphs.analysis.tool_gating._log_tool_error") as mock_log:
        result = rule.evaluate(calls, {})

    assert [c["name"] for c in result] == ["ask_user"]
    mock_log.assert_called_once_with(
        "dropped_interrupt_tool",
        "respond",
        "dropped: an interrupt-bearing tool was already selected this turn",
    )


def test_solo_invariant_keeps_higher_priority_tool_second_pair():
    rule = SoloInvariantBatchRule()
    calls = [{"name": "finalize", "args": {}}, {"name": "write_draft", "args": {}}]

    with patch("app.graphs.analysis.tool_gating._log_tool_error"):
        result = rule.evaluate(calls, {})

    assert [c["name"] for c in result] == ["write_draft"]


def test_solo_invariant_single_interrupt_tool_keeps_original_order():
    # Only one interrupt-bearing tool in the batch: no competition, so the priority sort does not
    # apply and the note keeps riding along in its original relative position.
    rule = SoloInvariantBatchRule()
    calls = [{"name": "note", "args": {}}, {"name": "ask_user", "args": {}}]

    result = rule.evaluate(calls, {})

    assert [c["name"] for c in result] == ["note", "ask_user"]


def test_solo_invariant_note_rides_along_with_priority_winner():
    rule = SoloInvariantBatchRule()
    calls = [
        {"name": "note", "args": {}},
        {"name": "respond", "args": {}},
        {"name": "ask_user", "args": {}},
    ]

    with patch("app.graphs.analysis.tool_gating._log_tool_error"):
        result = rule.evaluate(calls, {})

    assert [c["name"] for c in result] == ["ask_user", "note"]
