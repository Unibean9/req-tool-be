"""Error recovery (plan 260703 Phase 2, sub-parts 1-2).

1. RecoverableToolError carries an optional `recovery` hint threaded into the tool_errors state
   entry and appended to the error ToolMessage.
2. Repeated same-code errors (>= threshold) surface an escalation line in the feedback block;
   a single occurrence does not.
"""

from app.graphs.agent_tools import (
    RecoverableToolError,
    _missing_required_arg_update,
    _recoverable_tool_update,
)
from app.graphs.analysis.prompt_assembly import _build_feedback_control_block


def test_recovery_threaded_into_entry_and_tool_message():
    cmd = _recoverable_tool_update(
        RecoverableToolError(code="demo", message="Cannot demo: bad.", recovery="Do X and retry."),
        tool_call_id="tc1",
    )
    entry = cmd.update["tool_errors"][0]
    assert entry["code"] == "demo"
    assert entry["recovery"] == "Do X and retry."
    assert cmd.update["messages"][0].content == "Cannot demo: bad. Do X and retry."


def test_absent_recovery_keeps_legacy_shape():
    cmd = _recoverable_tool_update(
        RecoverableToolError(code="demo", message="Cannot demo: bad."), tool_call_id="tc1"
    )
    entry = cmd.update["tool_errors"][0]
    assert "recovery" not in entry
    assert cmd.update["messages"][0].content == "Cannot demo: bad."


def test_standard_constructor_supplies_recovery():
    cmd = _missing_required_arg_update("write_draft", "body", "tc1")
    entry = cmd.update["tool_errors"][0]
    assert entry["code"] == "missing_required_arg"
    assert "body" in entry["recovery"]


def test_repeated_error_escalates_at_threshold():
    state = {
        "tool_errors": [
            {"code": "missing_required_arg", "recovery": "Provide 'body' and call write_draft again."},
            {"code": "missing_required_arg", "recovery": "Provide 'body' and call write_draft again."},
        ]
    }
    block = _build_feedback_control_block(state)
    assert "repeated tool_errors" in block
    assert "missing_required_arg" in block
    assert "failed 2 times" in block
    assert "Provide 'body'" in block


def test_single_error_does_not_escalate():
    state = {"tool_errors": [{"code": "missing_required_arg"}]}
    assert _build_feedback_control_block(state) == ""


def test_stale_base_version_renders_rebase_steer():
    state = {
        "feedback_summary": {
            "stale_base_version": {"base_version_id": "v1", "current_version_id": "v2", "artifact_id": "a1"}
        }
    }
    block = _build_feedback_control_block(state)
    assert "stale_base_version" in block
    assert "re-read the artifact and rebase" in block
    assert "v1" in block and "v2" in block


def test_resurfaced_question_wording_unchanged_below_ignored_threshold():
    state = {
        "feedback_summary": {
            "resurfaced_questions": [{"id": "Q4", "statement": "Confirm rollout timeline"}],
            "ignored_counts": {"resurfaced_questions": 1},
        }
    }
    block = _build_feedback_control_block(state)
    assert "resurfaced_questions: Q4" in block
    assert "URGENT" not in block


def test_resurfaced_question_wording_escalates_at_ignored_threshold():
    state = {
        "feedback_summary": {
            "resurfaced_questions": [{"id": "Q4", "statement": "Confirm rollout timeline"}],
            "ignored_counts": {"resurfaced_questions": 2},
        }
    }
    block = _build_feedback_control_block(state)
    assert "URGENT (ignored 2 turns)" in block
    assert "Q4" in block
