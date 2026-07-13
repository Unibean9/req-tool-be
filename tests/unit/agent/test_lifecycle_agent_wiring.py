import inspect
import logging

from langchain_core.messages import AIMessage

from app.graphs import agent_tools
from app.graphs.agent_tools import get_available_tools
from app.graphs.analysis.tool_gating import gate_model_selection
from app.graphs.nodes import _build_tool_selection_prompt
from app.graphs.session_phase import DRAFT
from app.graphs.state import build_initial_workflow_state


def _state(lifecycle_state: str):
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
                    "allowed_actions": ["reconcile"] if lifecycle_state == "stale" else [],
                    "blockers": [],
                }
            ],
        }
    )
    return state


def _tool_names(state):
    return {tool.name for tool in get_available_tools(state)}


def test_current_lifecycle_hides_write_draft_but_keeps_recovery_tools():
    names = _tool_names(_state("current"))

    assert "write_draft" not in names
    assert {"ask_user", "respond", "read_artifact"} <= names


def test_stale_lifecycle_keeps_write_draft_menu_for_declared_reconcile():
    names = _tool_names(_state("stale"))

    assert "write_draft" in names


def test_gate_model_selection_blocks_plain_stale_write_and_logs(caplog):
    message = AIMessage(
        content="",
        tool_calls=[{"id": "c1", "name": "write_draft", "args": {"title": "Vision", "body": "## Vision\nx"}}],
    )

    with caplog.at_level(logging.INFO, logger="app.graphs.gate_logging"):
        _model, gated, dropped, feedback, _out_of_phase = gate_model_selection(_state("stale"), message)

    assert gated == []
    assert dropped == ["write_draft"]
    assert feedback["lifecycle_blocked_tools"] == [
        {
            "name": "write_draft",
            "reason": "stale_artifact_requires_curation_action",
            "state": "stale",
        }
    ]
    assert "dropped_tools" not in feedback
    assert any("gate=lifecycle_tool_gate verdict=blocked" in record.getMessage() for record in caplog.records)


def test_gate_model_selection_allows_stale_write_with_curation_fields():
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "c1",
                "name": "write_draft",
                "args": {
                    "title": "Vision",
                    "body": "## Vision\nx",
                    "curation_action": "UPDATE",
                    "curation_justification": "Reconciles changed problem statement.",
                },
            }
        ],
    )

    _model, gated, dropped, feedback, _out_of_phase = gate_model_selection(_state("stale"), message)

    assert [item["name"] for item in gated] == ["write_draft"]
    assert dropped == []
    assert "lifecycle_blocked_tools" not in feedback


def test_prompt_includes_situation_report_and_change_history():
    state = _state("stale")
    state["artifact_history"] = [
        {
            "artifact_type": "vision_objectives",
            "artifact_id": "artifact-vision",
            "version_id": "version-2",
            "version_number": 2,
            "change_source": "manual",
        }
    ]

    prompt = _build_tool_selection_prompt(state, [])

    assert "SITUATION REPORT" in prompt
    assert "state=STALE" in prompt
    assert "actions=reconcile" in prompt
    assert "RECENT ARTIFACT CHANGES" in prompt
    assert "source=manual" in prompt


def test_finalize_predecessor_gate_uses_resolver_predecessor_set():
    source = inspect.getsource(agent_tools._finalize_impl)

    assert "ancestor_types(artifact_type)" in source
    assert "ARTIFACT_PREDECESSORS.get(artifact_type" not in source
