"""Phase 4 Test E — isolated tests for `app/graphs/gating/dispatch_rules.py`'s
three rules, independent of the real dispatch call sites (mirrors Phase 3's
`test_menu_gating_rules.py` isolated-rule pattern)."""

import logging
from unittest.mock import patch

from app.graphs.gating.dispatch_rules import (
    ColdStartDraftBlockRule,
    LifecycleWriteDraftBlockRule,
    SoloInvariantBatchRule,
)
from app.graphs.gating.verdict import VerdictKind
from app.graphs.session_phase import DRAFT
from app.graphs.state import build_initial_workflow_state


def _lifecycle_state(lifecycle_state: str):
    state = build_initial_workflow_state(artifact_type="vision_objectives", workflow_area="analysis", step_key=None)
    state.update(
        {
            "user_confirmed": True,
            "session_phase": DRAFT,
            "focused_artifact_id": "artifact-vision",
            "lifecycle_reports": [
                {
                    "artifact_type": "vision_objectives",
                    "artifact_id": "artifact-vision",
                    "state": lifecycle_state,
                    "reason": f"{lifecycle_state} reason",
                    "allowed_actions": [],
                    "blockers": [],
                }
            ],
        }
    )
    return state


# --- SoloInvariantBatchRule -------------------------------------------------


def test_solo_invariant_passthrough_when_no_interrupt_bearing_tool():
    rule = SoloInvariantBatchRule()
    calls = [{"name": "critique_note", "args": {}}, {"name": "explore_note", "args": {}}]
    assert rule.evaluate(calls, {}) == calls


def test_solo_invariant_keeps_first_interrupt_and_note_drops_rest():
    rule = SoloInvariantBatchRule()
    calls = [
        {"name": "ask_user", "args": {}},
        {"name": "respond", "args": {}},
        {"name": "explore_note", "args": {}},
    ]
    with patch("app.graphs.analysis.tool_gating._log_tool_error") as mock_log:
        result = rule.evaluate(calls, {})

    assert [c["name"] for c in result] == ["ask_user", "explore_note"]
    mock_log.assert_called_once_with(
        "dropped_interrupt_tool",
        "respond",
        "dropped: an interrupt-bearing tool was already selected this turn",
    )


def test_solo_invariant_drops_non_note_paired_with_interrupt():
    rule = SoloInvariantBatchRule()
    calls = [{"name": "write_draft", "args": {}}, {"name": "read_artifact", "args": {}}]
    with patch("app.graphs.analysis.tool_gating._log_tool_error") as mock_log:
        result = rule.evaluate(calls, {})

    assert [c["name"] for c in result] == ["write_draft"]
    mock_log.assert_called_once_with(
        "dropped_with_interrupt_tool",
        "read_artifact",
        "dropped: paired with an interrupt-bearing tool",
    )


# --- LifecycleWriteDraftBlockRule -------------------------------------------


def test_lifecycle_write_draft_block_rule_allows_when_no_block(caplog):
    rule = LifecycleWriteDraftBlockRule()
    with caplog.at_level(logging.INFO, logger="app.graphs.gate_logging"):
        verdict = rule.evaluate({"name": "write_draft", "args": {}}, {})

    assert verdict.is_allow
    assert caplog.records == []


def test_lifecycle_write_draft_block_rule_denies_and_logs():
    rule = LifecycleWriteDraftBlockRule()
    state = _lifecycle_state("orphan")

    with patch("app.graphs.gating.dispatch_rules.log_gate_decision") as mock_log:
        verdict = rule.evaluate({"name": "write_draft", "args": {}}, state)

    assert verdict.kind == VerdictKind.DENY
    assert verdict.reason == "orphan_artifact_relink_or_retire_required"
    mock_log.assert_called_once_with(
        "lifecycle_tool_impl", "blocked", reason="orphan_artifact_relink_or_retire_required"
    )


def test_lifecycle_write_draft_block_rule_respects_stale_curation_exception():
    rule = LifecycleWriteDraftBlockRule()
    state = _lifecycle_state("stale")

    verdict = rule.evaluate(
        {"name": "write_draft", "args": {"curation_action": "UPDATE", "curation_justification": "why"}}, state
    )

    assert verdict.is_allow


# --- ColdStartDraftBlockRule -------------------------------------------------


def test_cold_start_draft_block_rule_denies_thin_cold_start():
    rule = ColdStartDraftBlockRule()
    state = {"decision_nodes": {}, "session_elicit_count": 0, "user_confirmed": None}

    verdict = rule.evaluate({"name": "write_draft"}, state)

    assert verdict.kind == VerdictKind.DENY
    assert verdict.reason == "cold_start_requires_elicitation"


def test_cold_start_draft_block_rule_allows_once_decision_nodes_exist():
    rule = ColdStartDraftBlockRule()
    state = {"decision_nodes": {"n1": {}}, "session_elicit_count": 0, "user_confirmed": None}

    verdict = rule.evaluate({"name": "write_draft"}, state)

    assert verdict.is_allow
