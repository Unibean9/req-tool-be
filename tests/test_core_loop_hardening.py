"""Tests for the core-loop hardening deltas.

Phase 1 — artifact lifecycle helpers (current_draft_body, artifact_stage).
Phase 2 — per-tool required-arg validation before emit (_missing_required_arg).
Phase 3 — fail-loud degrade reasons replacing silent coerce (_degrade_reason).
"""

from app.graphs.agent_tools import artifact_stage, current_draft_body
from app.graphs.nodes import _degrade_reason, _missing_required_arg
from tests.test_graph_nodes import _state


# --------------------------------------------------------------------------- Phase 1

def test_current_draft_body_prefers_draft_body_over_working_draft():
    state = _state()
    state["draft_body"] = "DB draft"
    state["working_draft"] = "session draft"
    assert current_draft_body(state) == "DB draft"


def test_current_draft_body_falls_back_to_working_draft():
    state = _state()
    state["working_draft"] = "session draft"
    assert current_draft_body(state) == "session draft"


def test_current_draft_body_empty_when_neither_present():
    assert current_draft_body(_state()) == ""


def test_artifact_stage_empty_without_draft():
    assert artifact_stage(_state()) == "empty"


def test_artifact_stage_drafting_when_draft_uncritiqued():
    state = _state()
    state["working_draft"] = "draft"
    assert artifact_stage(state) == "drafting"


def test_artifact_stage_critiqued_when_rounds_but_gate_not_passed():
    state = _state()
    state["working_draft"] = "draft"
    state["critique_rounds"] = 1
    state["quality_report"] = {"quality_gate_result": "fail"}
    assert artifact_stage(state) == "critiqued"


def test_artifact_stage_gate_passed_when_report_passes():
    state = _state()
    state["working_draft"] = "draft"
    state["critique_rounds"] = 1
    state["quality_report"] = {"quality_gate_result": "pass"}
    assert artifact_stage(state) == "gate_passed"


# --------------------------------------------------------------------------- Phase 2

def test_missing_required_arg_flags_empty_body_for_write_draft():
    assert _missing_required_arg("write_draft", {"title": "T", "body": ""}) == "body"


def test_missing_required_arg_flags_empty_mode_for_run_critique():
    assert _missing_required_arg("run_critique", {"target": "draft", "mode": ""}) == "mode"


def test_missing_required_arg_ignores_cosmetic_target_for_run_critique():
    # target is ARG001 (cosmetic) — only mode is required, so a present mode passes.
    assert _missing_required_arg("run_critique", {"target": "", "mode": "completeness"}) is None


def test_missing_required_arg_none_when_satisfied():
    assert _missing_required_arg("finalize", {"summary": "done"}) is None


def test_missing_required_arg_none_for_tool_without_required_args():
    assert _missing_required_arg("ask_user", {"message": ""}) is None


def test_degrade_reason_for_missing_required_arg_names_field():
    degrade = _degrade_reason(_state(), "write_draft", "write_draft", {"body": ""})
    assert degrade is not None
    assert degrade["gated_tool"] == "write_draft"
    assert "body" in degrade["gated_reason"]
    assert degrade["message"]  # a non-empty re-ask is staged


# --------------------------------------------------------------------------- Phase 3

def test_degrade_reason_for_out_of_menu_pick_is_observable():
    # finalize is not in the menu for an empty session — _gate_selected_tool clamps it to ask_user.
    degrade = _degrade_reason(_state(), "finalize", "ask_user", {})
    assert degrade is not None
    assert "gated: finalize" in degrade["gated_reason"]
    assert degrade["gated_tool"] == "finalize"


def test_degrade_reason_none_for_well_formed_available_pick():
    state = _state()
    state["working_draft"] = "draft"
    assert _degrade_reason(state, "write_draft", "write_draft", {"body": "content"}) is None


def test_degrade_reason_none_when_model_picks_ask_user():
    assert _degrade_reason(_state(), "ask_user", "ask_user", {"message": "hi"}) is None
