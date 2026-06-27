import hashlib

import pytest
from langchain_core.messages import AIMessage

from app.graphs.agent_tools import _cached_draft_body, current_draft_body, get_available_tools
from app.graphs.decision_graph import render_view
from app.graphs.nodes import _INTERRUPT_BEARING_TOOLS, _gate_selected_tools
from app.graphs.state import WorkflowState, build_initial_workflow_state
from app.schemas.artifact_synthesis import ArtifactReadinessState


def _names(state):
    return {tool.name for tool in get_available_tools(state)}


def _note_turn(call_id: str):
    return AIMessage(
        content="",
        tool_calls=[{"id": call_id, "name": "critique_note", "args": {"content": "x"}}],
    )


def _quality_pass(body: str) -> dict:
    return {
        "quality_report": {"quality_gate_result": "pass", "blocking_issues": []},
        "last_critiqued_draft_hash": hashlib.md5(body.encode()).hexdigest()[:8],
        "candidate_readiness": {
            "state": ArtifactReadinessState.SUFFICIENT,
            "score": 1.0,
            "gaps": [],
        },
    }


def test_only_safety_gates_remain_without_intent_or_note_filter(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N1", "kind": "objective", "statement": "Reduce processing time", "status": "confirmed"},
    )
    state = {
        "messages": [_note_turn(f"n{i}") for i in range(5)],
        "user_confirmed": None,
        "decision_nodes": nodes,
        "artifact_type": "brd",
        "session_elicit_count": 0,
    }

    names = _names(state)

    assert {"write_draft", "critique_note", "explore_note", "run_critique"} <= names
    assert "finalize" not in names


def test_write_draft_available_without_elicit():
    names = _names(
        {
            "messages": [],
            "user_confirmed": True,
            "decision_nodes": {},
            "session_elicit_count": 0,
        }
    )

    assert "write_draft" in names


def test_finalize_quality_gate_still_active(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N1", "kind": "objective", "statement": "Reduce processing time", "status": "confirmed"},
    )
    body = render_view(nodes, "brd")
    blocked = {
        "messages": [],
        "user_confirmed": True,
        "decision_nodes": nodes,
        "artifact_type": "brd",
        "critique_rounds": 1,
    }
    allowed = {**blocked, **_quality_pass(body)}

    assert "finalize" not in _names(blocked)
    assert "finalize" in _names(allowed)


def test_solo_interrupt_enforcement_still_active():
    first, second = list(_INTERRUPT_BEARING_TOOLS)[:2]
    raw = [{"name": first, "args": {}}, {"name": second, "args": {}}]

    gated = _gate_selected_tools({"messages": [], "user_confirmed": True}, raw)

    assert [tool["name"] for tool in gated] == [first]


def test_working_draft_field_removed_from_state():
    state = build_initial_workflow_state(artifact_type="brd", workflow_area="analysis", step_key=None)

    assert "working_draft" not in WorkflowState.__annotations__
    assert "working_draft" not in state


@pytest.mark.asyncio
async def test_render_view_is_sole_source_for_draft_body(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N1", "kind": "objective", "statement": "Source tu graph", "status": "confirmed"},
    )
    state = {
        "decision_nodes": nodes,
        "artifact_type": "brd",
        "draft_body": "DB cu",
        "working_draft": "checkpoint cu",
    }

    assert await current_draft_body(state) == render_view(nodes, "brd")
    assert _cached_draft_body(state) == render_view(nodes, "brd")


@pytest.mark.asyncio
async def test_legacy_checkpoint_working_draft_is_ignored():
    state = {"decision_nodes": {}, "artifact_type": "brd", "working_draft": "checkpoint cu"}

    assert await current_draft_body(state) == ""
    assert _cached_draft_body(state) == ""
