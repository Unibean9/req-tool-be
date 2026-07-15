"""Contract hồi quy cho channel feedback_summary plain theo từng writer."""

import uuid
from types import SimpleNamespace
from typing import Annotated, get_origin, get_type_hints
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.graphs.analysis.context_loader import load_turn_context
from app.graphs.analysis.tool_gating import gate_model_selection
from app.graphs.nodes import orchestrator_node, summarize_node
from app.graphs.state import WorkflowState
from app.services.agent_service import AgentService
from tests.factories import _state

CUMULATIVE = {
    "stale_base_version": {"version_id": "v2"},
    "lifecycle_persist_rejection": {"reason": "stale"},
    "candidate_readiness_rejection": {"reason": "missing evidence"},
}
SWEEP = {
    "depth_signal": {"level": "deep"},
    "sweep_gaps": [{"section": "scope"}],
    "created_parked_questions": [{"id": "Q1"}],
}


def test_gate_preserves_cumulative_and_sweep_feedback_keys():
    state = _state()
    state["feedback_summary"] = {**CUMULATIVE, **SWEEP}
    message = AIMessage(content="", tool_calls=[{"id": "c1", "name": "respond", "args": {"message": "ok"}}])

    _model, _gated, _dropped, feedback, _out_of_phase = gate_model_selection(state, message)

    for key, value in {**CUMULATIVE, **SWEEP}.items():
        assert feedback[key] == value


@pytest.mark.asyncio
async def test_orchestrator_preserves_unowned_cumulative_and_sweep_keys(decision_graph_factory):
    nodes = decision_graph_factory({"id": "N1", "kind": "fact", "status": "confirmed"})

    update = await orchestrator_node(
        {"decision_nodes": nodes, "artifact_type": "brd", "feedback_summary": {**CUMULATIVE, **SWEEP}}, {}
    )

    for key, value in {**CUMULATIVE, **SWEEP}.items():
        assert update["feedback_summary"][key] == value


@pytest.mark.asyncio
async def test_summarize_omits_feedback_summary_so_graph_state_preserves_it():
    state = _state()
    state["feedback_summary"] = {**CUMULATIVE, **SWEEP}
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=({"summary": "condensed"}, {}))

    with patch("app.graphs.nodes.route_before_analyze", return_value="summarize"):
        update = await summarize_node(state, {"configurable": {"llm_client": llm}})

    assert update == {"conversation_summary": "condensed"}
    assert "feedback_summary" not in update


@pytest.mark.asyncio
async def test_real_writer_chain_preserves_cumulative_keys_across_turns(decision_graph_factory):
    state = _state()
    state["feedback_summary"] = dict(CUMULATIVE)
    state["decision_nodes"] = decision_graph_factory({"id": "N1", "kind": "fact", "status": "confirmed"})
    state["artifact_type"] = "brd"
    message = AIMessage(content="", tool_calls=[{"id": "c1", "name": "respond", "args": {"message": "ok"}}])

    for _turn in range(2):
        _model, _gated, _dropped, gated_feedback, _out_of_phase = gate_model_selection(state, message)
        state["feedback_summary"] = gated_feedback
        state.update(await orchestrator_node(state, {}))
        for key, value in CUMULATIVE.items():
            assert state["feedback_summary"][key] == value

    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=({"summary": "condensed"}, {}))
    with patch("app.graphs.nodes.route_before_analyze", return_value="summarize"):
        summary_update = await summarize_node(state, {"configurable": {"llm_client": llm}})
    state.update(summary_update)

    for key, value in CUMULATIVE.items():
        assert state["feedback_summary"][key] == value


@pytest.mark.asyncio
async def test_sweep_feedback_persists_until_an_explicit_channel_reset(decision_graph_factory):
    nodes = decision_graph_factory(
        {"id": "N1", "kind": "objective", "status": "confirmed"},
        {"id": "N2", "kind": "assumption", "status": "confirmed"},
    )
    state = {"decision_nodes": nodes, "artifact_type": "prd", "completeness_sweep_requested": True}

    first = await orchestrator_node(state, {})
    assert {"depth_signal", "sweep_gaps", "created_parked_questions"} <= first["feedback_summary"].keys()

    # Runtime does not pop these keys at the end of the cycle. The done flag blocks the next
    # automatic sweep, and clone-then-write keeps the first cycle's result until focus/recovery
    # resets the channel.
    next_state = {**state, **first, "completeness_sweep_requested": False}
    second = await orchestrator_node(next_state, {})
    for key in ("depth_signal", "sweep_gaps", "created_parked_questions"):
        assert second["feedback_summary"][key] == first["feedback_summary"][key]


@pytest.mark.asyncio
async def test_focus_change_nukes_feedback_summary():
    class Result:
        def scalar_one_or_none(self):
            return uuid.uuid4()

    class DB:
        async def execute(self, _query):
            return Result()

    class SessionContext:
        async def __aenter__(self):
            return DB()

        async def __aexit__(self, *_args):
            return None

    # Session factory của SQLAlchemy trả trực tiếp một async context manager.
    def session_factory():
        return SessionContext()

    state = _state()
    state.update(feedback_summary={**CUMULATIVE, **SWEEP}, focused_artifact_id=None)
    config = {
        "configurable": {
            "session_factory": session_factory,
            "thread_id": str(uuid.uuid4()),
            "project_id": str(uuid.uuid4()),
        }
    }
    coverage = {"coverage_complete": False, "section_coverage": {}}
    with (
        patch("app.graphs.analysis.context_loader.read_artifacts", AsyncMock(return_value=[])),
        patch("app.graphs.analysis.context_loader._load_lifecycle_reports", AsyncMock(return_value=[])),
        patch("app.graphs.analysis.context_loader._load_artifact_history", AsyncMock(return_value=[])),
        patch("app.graphs.analysis.context_loader.read_current_body", AsyncMock(return_value=None)),
        patch("app.graphs.analysis.context_loader._document_coverage", AsyncMock(return_value=coverage)),
    ):
        context = await load_turn_context(state, config)

    assert context.focus_reset_update["feedback_summary"] is None
    assert context.effective_state["feedback_summary"] is None


@pytest.mark.asyncio
async def test_recovery_seed_replaces_channel_and_resets_old_ignored_counts():
    db = SimpleNamespace(commit=AsyncMock(), add=lambda _row: None)
    service = AgentService(db=db, graph=None, session_factory=None)
    service._check_and_resume = AsyncMock()
    tool_call = SimpleNamespace(id=uuid.uuid4(), tool_name="write_draft", status=None, resolved_at=None)
    replacement = {"stale_base_version": {"version_id": "v3"}}

    await service._supersede_tool_call_for_in_loop_recovery(
        project_id=uuid.uuid4(), session_id=uuid.uuid4(), tool_call=tool_call, feedback_summary=replacement
    )

    state_update = service._check_and_resume.await_args.kwargs["state_update"]
    assert state_update == {"feedback_summary": replacement}
    assert "ignored_counts" not in state_update["feedback_summary"]

    checkpoint = {"channel_values": {"feedback_summary": {"ignored_counts": {"resurfaced_questions": 9}}}}
    command = await service._resume_command(
        SimpleNamespace(graph_checkpoint=checkpoint), {"all_resolved": True}, state_update
    )
    assert command.update["feedback_summary"] == replacement
    assert "ignored_counts" not in command.update["feedback_summary"]


def test_feedback_summary_state_channel_has_no_additive_reducer():
    annotation = get_type_hints(WorkflowState, include_extras=True)["feedback_summary"]

    assert get_origin(annotation) is not Annotated
