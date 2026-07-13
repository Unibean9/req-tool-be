"""Phase 4 Test C — write_draft's lifecycle-block / cold-start-block hard gates,
now routed through `LifecycleWriteDraftBlockRule`/`ColdStartDraftBlockRule` called
directly by `_write_draft_impl`, are byte-identical to the pre-Phase-4 inline checks:
same `RecoverableToolError` code/message, same `log_gate_decision` call (lifecycle) /
no call at all (cold-start). See `phase-04-brief.md` section C for the exact pre-
Phase-4 call sites this test pins.
"""

import logging

import pytest

from app.graphs.agent_tools import _write_draft_impl
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


@pytest.mark.asyncio
async def test_lifecycle_block_error_and_log_are_byte_identical(caplog):
    state = _lifecycle_state("current")

    with caplog.at_level(logging.INFO, logger="app.graphs.gate_logging"):
        command = await _write_draft_impl(
            "Vision", "## Vision\nDraft body.", state, {"configurable": {}}, "tc1"
        )

    errors = command.update["tool_errors"]
    assert len(errors) == 1
    assert errors[0]["code"] == "current_artifact_reproposal_blocked"
    assert errors[0]["message"] == (
        "Cannot write_draft: lifecycle state blocks this proposal (current_artifact_reproposal_blocked)."
    )
    assert any(
        "gate=lifecycle_tool_impl verdict=blocked" in r.getMessage()
        and "current_artifact_reproposal_blocked" in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_cold_start_block_error_and_no_gate_log(caplog):
    state = {"messages": [], "user_confirmed": None, "decision_nodes": {}, "session_elicit_count": 0}

    with caplog.at_level(logging.INFO, logger="app.graphs.gate_logging"):
        command = await _write_draft_impl(
            "Draft", "Content drafted too soon", state, {"configurable": {}}, "tc1"
        )

    errors = command.update["tool_errors"]
    assert len(errors) == 1
    assert errors[0]["code"] == "cold_start_requires_elicitation"
    assert "elicit" in command.update["messages"][0].content
    assert caplog.records == []
