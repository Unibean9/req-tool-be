"""Dropped tools must be fed back to the model so it can self-correct (verbose-failure principle).

A tool the gate silently removed leaves the model blind; it then re-pairs the same tools next turn.
The drop is surfaced once via feedback_summary['dropped_tools'] in the per-turn prompt.
"""

from app.graphs.nodes import _build_feedback_control_block, _dropped_tool_names


def test_dropped_names_are_the_set_difference():
    requested = [{"name": "write_draft"}, {"name": "run_critique"}, {"name": "explore_note"}]
    kept = [{"name": "write_draft"}, {"name": "explore_note"}]
    assert _dropped_tool_names(requested, kept) == ["run_critique"]


def test_no_drop_yields_empty():
    requested = [{"name": "explore_note"}, {"name": "critique_note"}]
    assert _dropped_tool_names(requested, requested) == []


def test_feedback_block_renders_dropped_tools():
    block = _build_feedback_control_block({"feedback_summary": {"dropped_tools": ["run_critique"]}})
    assert "run_critique" in block
    assert "bi bo qua" in block


def test_feedback_block_empty_without_dropped():
    assert _build_feedback_control_block({"feedback_summary": {}}) == ""
