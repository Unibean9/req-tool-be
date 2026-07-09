"""Isolated tests for `app/graphs/gating/menu_rules.py`'s per-call rules — each rule's
`evaluate` exercised directly with synthetic state dicts, independent of `get_available_tools`.
Covers the pass-through-for-irrelevant-tools contract and, since nothing else exercises it this
phase, `PhaseLifecycleMenuRule`'s dispatch-mode branch.
"""

from unittest.mock import patch

from app.graphs.gating.menu_rules import (
    DecisionGraphMenuRule,
    FinalizeMenuRule,
    PhaseLifecycleMenuRule,
    RecommendNextWorkflowMenuRule,
    RunCritiqueMenuRule,
    RunReadinessCheckMenuRule,
)
from app.graphs.gating.rules import Mode
from app.graphs.gating.verdict import VerdictKind


# --- pass-through for irrelevant tools --------------------------------------


def test_finalize_rule_passes_through_other_tools():
    rule = FinalizeMenuRule()
    verdict = rule.evaluate({"name": "ask_user"}, {})
    assert verdict.kind is VerdictKind.ALLOW


def test_run_critique_rule_passes_through_other_tools():
    rule = RunCritiqueMenuRule()
    verdict = rule.evaluate({"name": "ask_user"}, {})
    assert verdict.kind is VerdictKind.ALLOW


def test_recommend_next_workflow_rule_passes_through_other_tools():
    rule = RecommendNextWorkflowMenuRule()
    verdict = rule.evaluate({"name": "ask_user"}, {})
    assert verdict.kind is VerdictKind.ALLOW


def test_run_readiness_check_rule_passes_through_other_tools():
    rule = RunReadinessCheckMenuRule()
    verdict = rule.evaluate({"name": "ask_user"}, {})
    assert verdict.kind is VerdictKind.ALLOW


def test_decision_graph_rule_passes_through_other_tools():
    rule = DecisionGraphMenuRule()
    verdict = rule.evaluate({"name": "ask_user"}, {})
    assert verdict.kind is VerdictKind.ALLOW


# --- FinalizeMenuRule --------------------------------------------------------


def test_finalize_rule_denies_without_draft():
    rule = FinalizeMenuRule()
    verdict = rule.evaluate({"name": "finalize"}, {"draft_body": "", "critique_rounds": 1})
    assert verdict.kind is VerdictKind.DENY


def test_finalize_rule_denies_when_critique_rounds_zero():
    rule = FinalizeMenuRule()
    verdict = rule.evaluate({"name": "finalize"}, {"draft_body": "x", "critique_rounds": 0})
    assert verdict.kind is VerdictKind.DENY


def test_finalize_rule_allows_when_gate_open():
    rule = FinalizeMenuRule()
    with patch("app.graphs.gating.menu_rules.agent_tools._finalize_gate_open", return_value=True):
        verdict = rule.evaluate({"name": "finalize"}, {"draft_body": "x", "critique_rounds": 1})
    assert verdict.kind is VerdictKind.ALLOW


def test_finalize_rule_does_not_consult_gate_when_precondition_fails():
    """`_finalize_gate_open` (the sole source of the "finalize" gate-decision log) must not even
    be called when has_draft/critique_rounds already fail — matching original inline logic."""
    rule = FinalizeMenuRule()
    with patch("app.graphs.gating.menu_rules.agent_tools._finalize_gate_open") as mock_gate:
        rule.evaluate({"name": "finalize"}, {"draft_body": "", "critique_rounds": 1})
        rule.evaluate({"name": "finalize"}, {"draft_body": "x", "critique_rounds": 0})
    mock_gate.assert_not_called()


# --- RunCritiqueMenuRule / RunReadinessCheckMenuRule / RecommendNextWorkflowMenuRule --------


def test_run_critique_rule_denies_at_rounds_cap(monkeypatch):
    monkeypatch.setattr("app.graphs.gating.menu_rules.settings.max_critique_rounds", 2)
    rule = RunCritiqueMenuRule()
    verdict = rule.evaluate({"name": "run_critique"}, {"draft_body": "x", "critique_rounds": 2})
    assert verdict.kind is VerdictKind.DENY


def test_run_critique_rule_allows_below_cap(monkeypatch):
    monkeypatch.setattr("app.graphs.gating.menu_rules.settings.max_critique_rounds", 2)
    rule = RunCritiqueMenuRule()
    verdict = rule.evaluate({"name": "run_critique"}, {"draft_body": "x", "critique_rounds": 1})
    assert verdict.kind is VerdictKind.ALLOW


def test_run_readiness_check_rule_requires_draft_and_round():
    rule = RunReadinessCheckMenuRule()
    assert rule.evaluate({"name": "run_readiness_check"}, {"draft_body": "x", "critique_rounds": 0}).kind is VerdictKind.DENY
    assert rule.evaluate({"name": "run_readiness_check"}, {"draft_body": "", "critique_rounds": 1}).kind is VerdictKind.DENY
    assert rule.evaluate({"name": "run_readiness_check"}, {"draft_body": "x", "critique_rounds": 1}).kind is VerdictKind.ALLOW


def test_recommend_next_workflow_rule_allows_on_coverage_signal():
    rule = RecommendNextWorkflowMenuRule()
    state = {"draft_body": "", "section_coverage": {"a": "filled", "b": "filled"}}
    assert rule.evaluate({"name": "recommend_next_workflow"}, state).kind is VerdictKind.ALLOW


def test_recommend_next_workflow_rule_denies_without_draft_or_signal():
    rule = RecommendNextWorkflowMenuRule()
    state = {"draft_body": "", "section_coverage": {"a": "empty"}}
    assert rule.evaluate({"name": "recommend_next_workflow"}, state).kind is VerdictKind.DENY


# --- DecisionGraphMenuRule ----------------------------------------------------


def test_decision_graph_rule_denies_when_flag_off():
    rule = DecisionGraphMenuRule()
    with patch("app.graphs.gating.menu_rules.agent_tools._decision_graph_menu", return_value=[]):
        verdict = rule.evaluate({"name": "create_decision_node"}, {})
    assert verdict.kind is VerdictKind.DENY


def test_decision_graph_rule_allows_when_menu_offers_tool():
    class _T:
        name = "create_decision_node"

    rule = DecisionGraphMenuRule()
    with patch("app.graphs.gating.menu_rules.agent_tools._decision_graph_menu", return_value=[_T()]):
        verdict = rule.evaluate({"name": "create_decision_node"}, {})
    assert verdict.kind is VerdictKind.ALLOW


# --- PhaseLifecycleMenuRule: menu mode ---------------------------------------


def test_phase_lifecycle_menu_denies_phase_excluded_tool_silently():
    rule = PhaseLifecycleMenuRule(mode=Mode.MENU)
    verdict = rule.evaluate({"name": "write_draft", "phase": "intent"}, {})
    assert verdict.kind is VerdictKind.DENY


def test_phase_lifecycle_menu_allows_when_no_lifecycle_block():
    rule = PhaseLifecycleMenuRule(mode=Mode.MENU)
    verdict = rule.evaluate({"name": "write_draft", "phase": None}, {})
    assert verdict.kind is VerdictKind.ALLOW


def test_phase_lifecycle_menu_allows_and_logs_nothing_for_stale_curation_exception():
    rule = PhaseLifecycleMenuRule(mode=Mode.MENU)
    state = {
        "focused_artifact_id": "a1",
        "lifecycle_reports": [{"artifact_type": "brd", "artifact_id": "a1", "state": "stale"}],
    }
    with patch("app.graphs.gating.menu_rules.log_gate_decision") as mock_log:
        verdict = rule.evaluate({"name": "write_draft", "phase": None}, state)
    assert verdict.kind is VerdictKind.ALLOW
    mock_log.assert_not_called()


def test_phase_lifecycle_menu_denies_and_logs_for_other_truthy_reasons():
    rule = PhaseLifecycleMenuRule(mode=Mode.MENU)
    state = {
        "focused_artifact_id": "a1",
        "lifecycle_reports": [{"artifact_type": "brd", "artifact_id": "a1", "state": "current"}],
    }
    with patch("app.graphs.gating.menu_rules.log_gate_decision") as mock_log:
        verdict = rule.evaluate({"name": "write_draft", "phase": None}, state)
    assert verdict.kind is VerdictKind.DENY
    mock_log.assert_called_once()
    args, kwargs = mock_log.call_args
    assert args[0] == "lifecycle_tool_menu"
    assert args[1] == "blocked"
    assert kwargs["reason"] == "current_artifact_reproposal_blocked"


# --- PhaseLifecycleMenuRule: dispatch mode (wired in a later phase; correct + testable now) ---


def test_phase_lifecycle_dispatch_denies_phase_excluded_tool_silently():
    rule = PhaseLifecycleMenuRule(mode=Mode.DISPATCH)
    with patch("app.graphs.gating.menu_rules.log_gate_decision") as mock_log:
        verdict = rule.evaluate({"name": "write_draft", "phase": "intent"}, {})
    assert verdict.kind is VerdictKind.DENY
    mock_log.assert_not_called()


def test_phase_lifecycle_dispatch_denies_stale_without_curation_args():
    """Unlike menu-mode, dispatch has no exception for stale_artifact_requires_curation_action —
    every truthy reason blocks."""
    rule = PhaseLifecycleMenuRule(mode=Mode.DISPATCH)
    state = {
        "focused_artifact_id": "a1",
        "lifecycle_reports": [{"artifact_type": "brd", "artifact_id": "a1", "state": "stale"}],
    }
    with patch("app.graphs.gating.menu_rules.log_gate_decision") as mock_log:
        verdict = rule.evaluate({"name": "write_draft", "phase": None, "args": {}}, state)
    assert verdict.kind is VerdictKind.DENY
    mock_log.assert_called_once()
    args, kwargs = mock_log.call_args
    assert args[0] == "lifecycle_tool_gate"
    assert kwargs["reason"] == "stale_artifact_requires_curation_action"
    assert kwargs["extra"] == {"tool": "write_draft", "lifecycle_state": "stale"}


def test_phase_lifecycle_dispatch_allows_stale_with_curation_args():
    rule = PhaseLifecycleMenuRule(mode=Mode.DISPATCH)
    state = {
        "focused_artifact_id": "a1",
        "lifecycle_reports": [{"artifact_type": "brd", "artifact_id": "a1", "state": "stale"}],
    }
    args = {"curation_action": "UPDATE", "curation_justification": "Reconciles changed body."}
    verdict = rule.evaluate({"name": "write_draft", "phase": None, "args": args}, state)
    assert verdict.kind is VerdictKind.ALLOW


def test_phase_lifecycle_dispatch_uses_real_args_not_menu_empty_dict():
    """Dispatch passes the tool call's real args through to lifecycle_tool_block_reason (menu
    always passes {}); a stale-curation dispatch call without those real args stays blocked."""
    rule = PhaseLifecycleMenuRule(mode=Mode.DISPATCH)
    state = {
        "focused_artifact_id": "a1",
        "lifecycle_reports": [{"artifact_type": "brd", "artifact_id": "a1", "state": "stale"}],
    }
    verdict = rule.evaluate({"name": "write_draft", "phase": None}, state)
    assert verdict.kind is VerdictKind.DENY
