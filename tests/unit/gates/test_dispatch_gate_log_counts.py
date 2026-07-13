"""Phase 4 Test B — dispatch-time phase/lifecycle log-call-count equivalence.

Guards the double-fire hazard described in `phase-04-brief.md`: `_gate_selected_tools`
must log `dropped_out_of_phase_tool` (its own call site, via `_log_tool_error`) exactly
once per out-of-phase tool and must NOT also trigger `log_gate_decision` for that same
drop; a lifecycle-blocked tool must trigger `log_gate_decision("lifecycle_tool_gate",
"blocked", ...)` exactly once (from `PhaseLifecycleMenuRule`), not twice.
"""

import logging
from unittest.mock import patch

from app.graphs.analysis.tool_gating import _gate_selected_tools
from app.graphs.session_phase import DRAFT, REVIEW
from app.graphs.state import build_initial_workflow_state


def _state(**overrides):
    state = build_initial_workflow_state(artifact_type="vision_objectives", workflow_area="analysis", step_key=None)
    state.update(overrides)
    return state


def test_out_of_phase_drop_logs_once_via_log_tool_error_only():
    state = _state(session_phase=REVIEW, user_confirmed=True)

    with patch("app.graphs.gating.menu_rules.log_gate_decision") as mock_gate_log, patch(
        "app.graphs.analysis.tool_gating._log_tool_error"
    ) as mock_tool_error:
        kept = _gate_selected_tools(state, [{"name": "elicit", "args": {}}])

    assert kept == []
    mock_tool_error.assert_called_once_with(
        "dropped_out_of_phase_tool",
        "elicit",
        "dropped: not available in session phase 'review'",
    )
    mock_gate_log.assert_not_called()


def test_lifecycle_blocked_drop_logs_exactly_once_via_log_gate_decision(caplog):
    state = _state(
        user_confirmed=True,
        session_phase=DRAFT,
        focused_artifact_id="artifact-vision",
        lifecycle_reports=[
            {
                "artifact_type": "vision_objectives",
                "artifact_id": "artifact-vision",
                "state": "stale",
                "reason": "stale reason",
                "allowed_actions": [],
                "blockers": [],
            }
        ],
    )

    with caplog.at_level(logging.INFO, logger="app.graphs.gate_logging"):
        kept = _gate_selected_tools(state, [{"name": "write_draft", "args": {}}])

    assert kept == []
    matches = [r for r in caplog.records if "gate=lifecycle_tool_gate verdict=blocked" in r.getMessage()]
    assert len(matches) == 1
