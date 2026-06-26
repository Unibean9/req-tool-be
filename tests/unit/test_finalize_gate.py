"""Finalize gate: graph view non-empty AND critique_rounds > 0 AND quality gate passed AND candidate_readiness sufficient."""

import hashlib
from unittest.mock import patch

import pytest

from app.graphs.agent_tools import _finalize_impl, current_draft_body, get_available_tools
from app.graphs.decision_graph import create_node, render_view
from app.schemas.artifact_synthesis import ArtifactReadinessState


def _names(state):
    return {t.name for t in get_available_tools(state)}


def _hash(body: str) -> str:
    return hashlib.md5(body.encode()).hexdigest()[:8]


def _state_with_graph(statement: str = "Một bản nháp") -> dict:
    return {
        "messages": [],
        "user_confirmed": True,
        "artifact_type": "brd",
        "decision_nodes": {
            "N1": create_node(
                kind="objective",
                statement=statement,
                origin={"source": "test"},
                status="confirmed",
            )
        },
    }


def _current_body(state: dict) -> str:
    return render_view(state["decision_nodes"], state["artifact_type"])


def _passing_state(draft: str = "Một bản nháp", critique_rounds: int = 1) -> dict:
    """A state where every finalize-gate condition is satisfied (pass gate + fresh hash + sufficient readiness)."""
    state = {
        **_state_with_graph(draft),
        "critique_rounds": critique_rounds,
        "quality_report": {"quality_gate_result": "pass", "blocking_issues": []},
        "candidate_readiness": {"state": ArtifactReadinessState.SUFFICIENT, "score": 1.0, "gaps": []},
    }
    state["last_critiqued_draft_hash"] = _hash(_current_body(state))
    return state


def test_finalize_not_available_when_critique_rounds_zero():
    state = _passing_state(critique_rounds=0)
    assert "finalize" not in _names(state)


def test_finalize_available_when_critique_rounds_positive():
    assert "finalize" in _names(_passing_state())


def test_finalize_not_available_without_decision_graph_view():
    state = _passing_state()
    state["decision_nodes"] = {}
    assert "finalize" not in _names(state)


def test_finalize_ignores_db_draft_without_graph_view():
    draft = "Bản nháp tải từ DB"
    state = {
        "messages": [],
        "user_confirmed": True,
        "artifact_type": "brd",
        "decision_nodes": {},
        "draft_body": draft,
        "critique_rounds": 1,
        "quality_report": {"quality_gate_result": "pass", "blocking_issues": []},
        "last_critiqued_draft_hash": _hash(draft),
        "candidate_readiness": {"state": ArtifactReadinessState.SUFFICIENT, "score": 1.0, "gaps": []},
    }
    names = _names(state)
    assert "finalize" not in names
    assert "run_readiness_check" not in names


def test_finalize_not_available_when_gate_fails():
    state = _passing_state()
    state["quality_report"] = {"quality_gate_result": "fail", "blocking_issues": ["thiếu metric"]}
    assert "finalize" not in _names(state)


def test_finalize_not_available_when_draft_edited_after_critique():
    state = _passing_state()
    state["decision_nodes"] = {
        "N1": create_node(
            kind="objective",
            statement="Bản nháp đã sửa sau critique",
            origin={"source": "test"},
            status="confirmed",
        )
    }
    assert "finalize" not in _names(state)


def test_finalize_available_on_escape_hatch_at_rounds_cap():
    from app.graphs.agent_tools import CRITIQUE_ROUNDS_MAX

    state = _passing_state(critique_rounds=CRITIQUE_ROUNDS_MAX)
    # Stale hash, but the rounds cap is reached → escape hatch keeps finalize available.
    state["last_critiqued_draft_hash"] = "deadbeef"
    assert "finalize" in _names(state)


@pytest.mark.asyncio
async def test_finalize_interrupt_triggers_when_available():
    """_finalize_impl interrupts for human confirmation (the approval step) rather than erroring."""

    class _Session:
        status = None
        interrupt_type = None

    session_row = _Session()

    class _Result:
        def scalar_one(self_inner):
            return session_row

    class _DB:
        async def execute(self_inner, *a, **k):
            return _Result()

        async def commit(self_inner):
            return None

    def _factory():
        class _Ctx:
            async def __aenter__(self_inner):
                return _DB()

            async def __aexit__(self_inner, *a):
                return False

        return _Ctx()

    config = {"configurable": {"session_factory": _factory, "thread_id": "00000000-0000-0000-0000-000000000001"}}
    state = {
        **_passing_state("draft"),
        "critique_rounds": 1,
    }

    with patch("app.graphs.agent_tools.interrupt") as mock_interrupt:
        await _finalize_impl("Hoàn tất phiên.", state, config, "call_1")

    mock_interrupt.assert_called_once()
    assert session_row.status is not None


@pytest.mark.asyncio
async def test_finalize_hard_blocks_when_gate_fails():
    """Even if reached directly, _finalize_impl refuses a failing gate — ToolMessage, no interrupt."""
    config = {"configurable": {"session_factory": None, "thread_id": "00000000-0000-0000-0000-000000000001"}}
    state = {
        **_state_with_graph("draft"),
        "critique_rounds": 1,
        "quality_report": {"quality_gate_result": "fail", "blocking_issues": ["thiếu tiêu chí đo lường"]},
        "candidate_readiness": {"state": ArtifactReadinessState.SUFFICIENT, "score": 1.0, "gaps": []},
    }

    with patch("app.graphs.agent_tools.interrupt") as mock_interrupt:
        command = await _finalize_impl("Hoàn tất phiên.", state, config, "call_1")

    mock_interrupt.assert_not_called()
    msg = command.update["messages"][0]
    assert command.update["tool_errors"][0]["code"] == "finalize_gate_blocked"
    assert msg.status == "error"
    assert "Không thể finalize" in msg.content
    assert "thiếu tiêu chí đo lường" in msg.content


@pytest.mark.asyncio
async def test_finalize_hard_blocks_without_current_draft_body():
    config = {"configurable": {"session_factory": None, "thread_id": "00000000-0000-0000-0000-000000000001"}}
    state = {
        "critique_rounds": 2,
        "quality_report": {"quality_gate_result": "pass", "blocking_issues": []},
        "last_critiqued_draft_hash": _hash("draft"),
    }

    with patch("app.graphs.agent_tools.interrupt") as mock_interrupt:
        command = await _finalize_impl("Hoàn tất phiên.", state, config, "call_1")

    mock_interrupt.assert_not_called()
    assert command.update["tool_errors"][0]["code"] == "finalize_gate_blocked"
    assert "Không thể finalize" in command.update["messages"][0].content


@pytest.mark.asyncio
async def test_current_draft_body_ignores_focused_artifact_without_graph():
    state = {
        "focused_artifact_id": "00000000-0000-0000-0000-000000000001",
        "draft_body": "stale",
    }
    config = {"configurable": {"session_factory": None}}
    assert await current_draft_body(state, config) == ""


@pytest.mark.asyncio
async def test_finalize_hard_blocks_when_focused_artifact_was_not_critiqued():
    config = {"configurable": {"session_factory": None, "thread_id": "00000000-0000-0000-0000-000000000001"}}
    state = {
        **_state_with_graph("Draft item B"),
        "critique_rounds": 1,
        "quality_report": {"quality_gate_result": "pass", "blocking_issues": []},
        "last_critiqued_draft_hash": _hash("Draft section A"),
        "candidate_readiness": {"state": ArtifactReadinessState.SUFFICIENT, "score": 1.0, "gaps": []},
    }

    with patch("app.graphs.agent_tools.interrupt") as mock_interrupt:
        command = await _finalize_impl("Hoàn tất phiên.", state, config, "call_1")

    mock_interrupt.assert_not_called()
    msg = command.update["messages"][0]
    assert "Không thể finalize" in msg.content
    assert "bản nháp hiện tại" in msg.content
