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
from app.graphs.state import TOOL_ERRORS_PER_CODE_LIMIT, merge_tool_errors


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


def test_tool_errors_cap_keeps_latest_entries_per_code():
    errors = [
        {"code": "a", "message": f"a-{index}"}
        for index in range(TOOL_ERRORS_PER_CODE_LIMIT + 2)
    ] + [{"code": "b", "message": "b-0"}]

    merged = merge_tool_errors([], errors)

    assert [item["message"] for item in merged if item["code"] == "a"] == ["a-2", "a-3", "a-4"]
    assert [item["message"] for item in merged if item["code"] == "b"] == ["b-0"]


def test_tool_errors_cap_preserves_repeated_error_escalation():
    state = {
        "tool_errors": merge_tool_errors(
            [],
            [
                {"code": "missing_required_arg", "recovery": "Provide 'body' and call write_draft again."},
                {"code": "missing_required_arg", "recovery": "Provide 'body' and call write_draft again."},
                {"code": "missing_required_arg", "recovery": "Provide 'body' and call write_draft again."},
                {"code": "other_error"},
            ],
        )
    }

    block = _build_feedback_control_block(state)

    assert "missing_required_arg" in block
    assert "failed 3 times" in block


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


def test_lifecycle_persist_rejection_renders_rebase_steer():
    state = {
        "feedback_summary": {
            "lifecycle_persist_rejection": {
                "stale_predecessors": [{"artifact_id": "a1", "reason": "predecessor_version_changed"}]
            }
        }
    }

    block = _build_feedback_control_block(state)

    assert "lifecycle_persist_rejection" in block
    assert "re-read upstream artifacts and rebase" in block
    assert "predecessor_version_changed" in block


def test_candidate_readiness_rejection_renders_revision_steer():
    state = {
        "feedback_summary": {
            "candidate_readiness_rejection": {
                "state": "poorly_structured",
                "blocking_reasons": ["Missing target needing confirmation"],
            }
        }
    }

    block = _build_feedback_control_block(state)

    assert "candidate_readiness_rejection" in block
    assert "revise before proposing again" in block
    assert "Missing target" in block


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
